# Spinning Up a *Second* Hermes Agent (and Not Botching It)

> Companion to the [main tutorial](../README.md). The main guide gets **one** agent running on a
> DGX Spark. This guide covers standing up a **second, fully independent agent** on the same box —
> its own Linux user, its own `$HERMES_HOME`, its own memory, backups, and Discord bot.
>
> **Why this document exists:** we actually did this — provisioned a second agent named *Deirdre*
> alongside the first (*Corwin*) — and it went wrong in five distinct, instructive ways. Every
> section below is a real failure we hit, with the symptom, the root cause, the fix, and a
> **verification step that proves the fix** rather than assuming it. If you only read one thing,
> read [§7 The Five-Point Pre-Flight Checklist](#7-the-five-point-pre-flight-checklist).

---

## Table of Contents

1. [Model & Provider Wiring — the silent-downgrade trap](#1-model--provider-wiring--the-silent-downgrade-trap)
2. [Fallback Chain — redundancy in hardware is not redundancy in config](#2-fallback-chain--redundancy-in-hardware-is-not-redundancy-in-config)
3. [Memory (Mnemosyne) — the three-way break](#3-memory-mnemosyne--the-three-way-break)
4. [Backups — an untested backup is not a backup](#4-backups--an-untested-backup-is-not-a-backup)
5. [Discord — roles, permissions, and the READY that never comes](#5-discord--roles-permissions-and-the-ready-that-never-comes)
6. [Account & Identity Isolation](#6-account--identity-isolation)
7. [The Five-Point Pre-Flight Checklist](#7-the-five-point-pre-flight-checklist)

---

## 0. Ground rules before you start

A second agent is a **separate Linux user**, not a second profile under the first user. This keeps
memory, secrets, sessions, and the gateway service cleanly isolated.

```bash
# As an admin (not as the first agent):
sudo adduser --disabled-password --gecos "" deirdre-ai
sudo usermod -aG sudo,docker deirdre-ai      # match whatever the first agent has, deliberately
sudo loginctl enable-linger deirdre-ai       # so the user's gateway service survives logout
```

Everything after this runs **as that new user**, in that user's `~/.hermes`. When you operate the
new user's services from another account, remember the per-user systemd bus is keyed on UID:

```bash
# Run a --user systemctl command for uid 1002 from a different login:
sudo -u deirdre-ai XDG_RUNTIME_DIR=/run/user/1002 \
  DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1002/bus \
  systemctl --user restart hermes-gateway.service
```

> **Pitfall (real):** `sudo cmd <<'HEREDOC'` fails auth because the heredoc consumes sudo's password
> channel. Write the script to a temp file and run `sudo python3 /tmp/x.py` instead. With
> passwordless sudo configured this is moot, but don't rely on that on a fresh box.

---

## 1. Model & Provider Wiring — the silent-downgrade trap

**Symptom.** The agent boots and answers, but is noticeably dim — shallow reasoning, misreads its
own state. ("Barely conscious," in our notes.) Nothing errors.

**Root cause.** The config had a *bare* custom provider:

```yaml
# BROKEN — what we actually shipped first
model:
  default: nemotron
  provider: custom
  base_url: http://localhost:8000/v1
  api_key: EMPTY
# ...and NO custom_providers: block anywhere in the file.
```

With no `custom_providers:` entry, `provider: custom` resolves straight to whatever `base_url`
says — here, the **local Nemotron** on `localhost:8000`. The agent silently ran on the weak local
model when we thought we'd pointed it at a frontier model. It never complained because, from the
config's point of view, this is a perfectly valid setup.

**Fix.** Define the provider explicitly and point `model.default` at the real model:

```yaml
model:
  default: Claude Opus 4.8
  provider: custom
  base_url: https://<your-gateway>/v1
  api_key: <your-key>

custom_providers:
  - name: Argo
    base_url: https://<your-gateway>/v1
    api_key: <your-key>
    model: Claude Opus 4.8
    models:
      - Claude Opus 4.8
      - Claude Sonnet 5
      # ...whatever the gateway serves
  - name: Local Nemotron
    base_url: http://localhost:8000/v1
    model: nemotron
    api_mode: chat_completions
    models:
      nemotron:
        context_length: 262144
```

**Verify — don't assume.** Make the *running* agent tell you what it's on, and cross-check against
the raw endpoint:

```bash
# 1. Ask the live agent:
hermes chat -q "State your exact model and provider base_url in one line."

# 2. Cross-check the raw endpoint actually answers (some servers 200 with empty content):
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nemotron","messages":[{"role":"user","content":"say ALIVE"}]}' \
  | python3 -m json.tool
```

> A `200` with `content: null` is **not** "working" — a real failure we hit was a healthy `/v1/models`
> masking a completion path that returned nothing. Read the payload, not just the status code.

---

## 2. Fallback Chain — redundancy in hardware is not redundancy in config

**Symptom.** Single point of failure: if the primary gateway drops, the agent dies. The local GPU is
*right there* serving a model, but the agent never uses it.

**Root cause.** No `fallback_providers` configured. Everything routed through one provider.

**Fix.** Register the local model as a fallback. `hermes fallback add` is an **interactive picker**
(no scripting flags), and it has a **"Custom endpoint (enter URL manually)"** option near the bottom —
use it to point at `http://localhost:8000/v1`. Or declare it directly:

```yaml
fallback_providers:
  - provider: custom
    model: nemotron
    base_url: http://localhost:8000/v1
    api_mode: chat_completions
```

> **Design note:** a fallback to *another model on the same gateway* is worthless — when the gateway
> is down, both go down in the same breath. The only fallback that survives a gateway outage is one
> on a **different substrate**, i.e. the local GPU.

**Verify — force a real failover.** A fallback in the list is a hypothesis until you watch it catch:

```bash
# In an ISOLATED copy of the config (never your live one):
export HERMES_HOME=/tmp/hermes-failover-test
cp -r ~/.hermes "$HERMES_HOME"
# Break ONLY the top-level model.base_url (the primary Hermes actually uses):
#   point it at a dead port, e.g. http://127.0.0.1:59999/v1
# Then fire one query and watch for the switch line:
hermes chat -q "Reply exactly: FAILOVER-WORKS"
# Expect: "🔄 Switched to fallback model: ... → nemotron via custom"  then the answer.
```

If you `sed` the base_url, confirm you hit the **top-level** `model.base_url` and not the first
`base_url` in `custom_providers` — they're different lines and only the top-level one is the primary
route.

---

## 3. Memory (Mnemosyne) — the three-way break

Long-term memory was wired wrong in **three independent ways** at once. Each is silent on its own.

### 3a. The Hermes↔Mnemosyne bridge plugin isn't installed

**Symptom.** `hermes memory status` shows `Provider: mnemosyne` but `Plugin: NOT installed ✗`.
Logs show `ModuleNotFoundError: No module named 'mnemosyne_hermes'`.

**Root cause.** The core `mnemosyne` package can be present in the venv while the **bridge**
(`mnemosyne-hermes`) is missing or its symlink is stale. Hermes *thinks* it has a memory provider;
the wire to it is cut.

**Fix.** Install the bridge into the Hermes venv and relink, then confirm:

```bash
# inside the agent's Hermes venv
pip install mnemosyne-hermes            # provides the mnemosyne_hermes module
mnemosyne-install                       # relinks the Hermes plugin symlink
hermes memory status                    # expect: Plugin: installed ✓ / Status: available ✓
```

### 3b. Cross-session scoping bug — recall returns 0 in live sessions

**Symptom.** Facts are in the DB, but every *new* session recalls nothing. Query as session
`default` → rows; query as any live session id → **0 rows**.

**Root cause.** Recall is scoped to the session id unless `MNEMOSYNE_CROSS_SESSION=1` is in the
process environment **at import time**.

**Fix.** Add it to the gateway's systemd unit `Environment=` (not just `.profile`), reload, restart,
and verify against the *actual* process environment:

```bash
systemctl --user edit hermes-gateway   # add: Environment=MNEMOSYNE_CROSS_SESSION=1
systemctl --user daemon-reload
systemctl --user restart hermes-gateway
# Prove the LIVE process really has it (not just the unit file):
tr '\0' '\n' < /proc/$(pgrep -u "$USER" -f hermes-gateway | head -1)/environ | grep MNEMOSYNE
```

### 3c. CLI `store` defaults to session scope

**Symptom.** Freshly migrated facts vanish cross-session even after 3a/3b are fixed.

**Root cause.** `mnemosyne store` writes `scope=session` by default. Session-scoped rows are invisible
across sessions unless the env var above is set.

**Fix (belt-and-suspenders).** Store durable facts as `scope=global` so they survive **even if the
env var is ever lost**:

```bash
mnemosyne store "durable fact" --scope global
# or promote existing rows in the DB to scope=global
```

**Verify the whole chain.** Store a canary and recall it from a *fresh* session with the env var
**unset** — if it comes back, defense-in-depth holds:

```bash
mnemosyne store "canary-$(date +%s): memory round-trip works" --scope global
env -u MNEMOSYNE_CROSS_SESSION hermes chat -q "Recall the canary fact you just stored."
hermes mnemosyne stats     # counts should include the canary
```

---

## 4. Backups — an untested backup is not a backup

**Symptom.** "We have backups" — but there was no cron, and the off-box push silently did nothing
(`rclone remote 'gdrive' not configured`, rc=2). A local script that runs once by hand is not a
backup strategy.

**Root cause.** Three gaps: (1) no schedule, (2) no off-box copy, (3) never restore-tested.

**Fix — the full pipeline.**

1. **Encrypt locally.** `tar | zstd | gpg --symmetric` (AES-256), passphrase from a `chmod 600`
   file. Capture SQLite DBs with the online `.backup` API, not `cp`, so they're consistent.
2. **Push off-box** to the agent's **own** cloud account (see [§6](#6-account--identity-isolation) —
   do **not** reuse the first agent's OAuth token). Use least-privilege scope (`drive.file` for
   Google Drive — the app only ever sees files it created).
3. **Schedule** it (nightly cron).
4. **Pin the passphrase off-box** (e.g. a Discord control channel). The ciphertext lives on Drive;
   the key must live somewhere the box's loss can't take with it — otherwise you have ciphertext and
   no key after a reimage.

**Headless OAuth handoff (no browser on either box).** Our boxes are headless over Tailscale with no
X-forwarding. `rclone authorize` is built for exactly this — run it on a machine that *has* a
browser, then paste the token back:

```bash
# On a laptop with a browser and rclone installed:
rclone authorize "drive" "<CLIENT_ID>" "<CLIENT_SECRET>" --auth-no-open-browser
# It prints a URL; complete OAuth; it prints a token JSON. Paste that into:
rclone config create gdrive drive scope=drive.file token='<TOKEN_JSON>' \
  client_id=<CLIENT_ID> client_secret=<CLIENT_SECRET>
```

> **rclone v1.74 gotchas (confirmed):** the old JSON-blob form of `authorize` errors with
> `illegal base64 data`, and `--scope` is rejected as an unknown flag. Use the positional
> `authorize "drive" "<id>" "<secret>"` form above. The loopback callback listens on
> `127.0.0.1:53682` on the box; if you're driving it from a remote laptop, SSH-tunnel it:
> `ssh -N -L 53682:localhost:53682 user@box`.

**Verify — restore from the OFF-BOX copy, not local disk.** This is the only test that proves
disaster recovery:

```bash
# Pull the newest archive FROM the cloud (simulating a wiped box):
rclone copy gdrive:deirdre-backups/<newest>.tar.zst.gpg /tmp/restore/
# Decrypt + decompress + extract with the pinned passphrase:
gpg --batch --pinentry-mode loopback \
  --passphrase "$(cat ~/.hermes/.backup-passphrase)" \
  -d /tmp/restore/<newest>.tar.zst.gpg | zstd -dq | tar -xf - -C /tmp/restore/
# Prove the critical DB survived:
sqlite3 /tmp/restore/hermes-home/mnemosyne/data/mnemosyne.db 'PRAGMA integrity_check;'  # expect: ok
```

Also keep a **negative test**: the wrapper must **fail loud** when the off-box remote is missing
(rc≠0, visible error) rather than silently "succeeding." A backup that lies about success is worse
than none.

---

## 5. Discord — roles, permissions, and the READY that never comes

**Symptom 1.** The second bot doesn't render as a server member and can't pin
(`403 Missing Permissions` / `MANAGE_MESSAGES`).

**Root cause.** The bot had **no role** (`roles: []`). The first bot had a `managed: true` role
auto-created when it was invited with the right OAuth scope; the second was invited without it.

**Fix.** Re-invite the bot via an OAuth2 URL that grants the needed scope/permissions (or an admin
role). Then **verify with the exact call that failed** — pin a throwaway message and unpin it:

```bash
# via the discord admin tooling: pin_message → expect success (not 403), then unpin to clean up.
```

**Symptom 2 (still open — flag it, don't hand-wave it).** Even after the role/permission fix, the
second gateway's Discord session **never reaches READY**: the gateway boots, loads tools, and runs,
but `~/.hermes/logs/gateway.log` contains **no** `logged in / READY / connected / guild` line the way
the first agent's does. Member-list invisibility traces to this, not to the Presence intent (ticking
the same intents as the first bot did **not** change it).

> **Lesson for the tutorial:** "the bot answers messages" and "the bot's gateway fully logged in"
> are different states. Grep the gateway log for a READY/guild line as an explicit health check when
> provisioning a new bot. This one is **unresolved** as of writing — documented here honestly so the
> next person doesn't assume it's fixed.

---

## 6. Account & Identity Isolation

Each agent gets its **own** external identities. Do not share tokens between agents — a shared OAuth
refresh token means one compromised agent exposes the other's cloud storage.

| Resource            | First agent (Corwin)          | Second agent (Deirdre)         |
|---------------------|-------------------------------|--------------------------------|
| Linux user / UID    | `videau-ai`                   | `deirdre-ai` (1002)            |
| `$HERMES_HOME`      | `/home/videau-ai/.hermes`     | `/home/deirdre-ai/.hermes`     |
| Gmail               | `brice.ai.videau@gmail.com`   | `brice.ai2.videau@gmail.com`   |
| GitHub              | `bricevideau-ai`              | `bricevideau-ai2`              |
| Mnemosyne DB        | own `mnemosyne.db`            | own `mnemosyne.db`             |
| Discord bot         | own token + managed role      | own token + managed role       |
| Cloud backup remote | own rclone token/scope        | own rclone token/scope         |

Naming convention that scaled cleanly for us: the second agent's online accounts mirror the first
with a `2` suffix.

**Verify isolation.** Confirm the two agents don't share a backup key or a Drive:

```bash
sha256sum ~/.hermes/.backup-passphrase   # compare across agents — hashes MUST differ
rclone about gdrive:                      # a fresh 2nd account shows ~0 B used, not the 1st's data
```

---

## 7. The Five-Point Pre-Flight Checklist

Before you call a second agent "done," each of these must be **proven**, not assumed:

- [ ] **Model.** Live agent reports the intended frontier model — not the local fallback.
      `custom_providers:` block present; `api_key` real. (`hermes chat -q "state your model"`)
- [ ] **Fallback.** Real failover observed against a dead primary in an isolated home
      (saw the "🔄 Switched to fallback model" line).
- [ ] **Memory.** `hermes memory status` → `Plugin: installed ✓`; `MNEMOSYNE_CROSS_SESSION=1` in the
      live process env; canary fact recalled from a fresh session; durable facts `scope=global`.
- [ ] **Backup.** Encrypted archive pushed off-box on a cron; **restored from the cloud copy** and
      `PRAGMA integrity_check` = `ok`; passphrase pinned off-box; negative test fails loud.
- [ ] **Discord.** Bot has a managed role; the exact privileged action that failed now succeeds
      (pin/unpin); gateway log shows a READY/guild line — or the gap is explicitly documented.

> The through-line: **prove it, don't report it.** Every failure above was invisible to a casual
> "does it answer?" check and only surfaced under a test that tried to make it fail. Configure with
> that adversarial mindset and a second agent comes up clean the first time.

---

## 7a. Automated: `verify-agent-health.sh`

The manual checklist above is the spec; [`scripts/verify-agent-health.sh`](../scripts/verify-agent-health.sh)
is the machine-checkable implementation. It probes the **runtime**, not just the config file —
because "config says on" is not "actually working" (every one of our four failures passed a config
read and still didn't work). Point it at any agent's `$HERMES_HOME` and it returns a non-zero exit
if any axis is broken, so it drops straight into CI or a multi-box rollup.

```bash
# Check the current user's agent
./scripts/verify-agent-health.sh

# Check another agent on the box (e.g. from an admin account)
sudo HERMES_HOME=/home/second-agent/.hermes ./scripts/verify-agent-health.sh

# Machine-readable, for rolling up results across a fleet of Spark boxes
./scripts/verify-agent-health.sh --json
```

What it verifies, mapped to the four failure axes from this doc:

| Check | Axis | What "PASS" actually proves (runtime, not config) |
|---|---|---|
| `model.default` / `model.provider` | §1 | Primary is a real provider, **not** silently `localhost` + `EMPTY` key |
| `mem.config` / `mem.binary` / `mem.db` | §3 | `provider: mnemosyne` **and** the `mnemosyne` binary resolves **and** the DB is non-trivial |
| `backup.local` / `backup.offbox` | §4 | A recent `*.tar.zst.gpg` exists **and** the last off-box push returned `rc=0` |
| `gw.run` / `gw.perms` | §5 | A gateway process is actually **running** for the account **and** no Discord `403 / 50013` in the logs |

Real output from this repo's reference box (`piment`), run against both live agents:

```
=== Hermes agent health check :: /home/<agent>/.hermes ===
--- Axis 1: model/provider ---
  [PASS] model.default = Claude Opus 4.8
  [PASS] primary base_url set (https://.../argoapi/v1)
--- Axis 2: long-term memory (Mnemosyne) ---
  [PASS] memory provider = mnemosyne in config
  [PASS] mnemosyne CLI resolves (backend installed)
  [PASS] mnemosyne.db present (1499136 bytes)
--- Axis 3: backups (local + off-box) ---
  [PASS] recent encrypted archive (4 h old): <agent>_20260724_195649.tar.zst.gpg
  [PASS] last off-box push succeeded (rc=0)
--- Axis 4: Discord gateway ---
  [PASS] gateway process is running (pid 688234, user <agent>)
  [FAIL] Discord 403 Missing Permissions (50013) seen — bot role lacks perms
=== Summary: 8 pass, 0 warn, 1 fail ===
```

Note the honest `[FAIL]` on the last line: on our box the bot genuinely lacks `MANAGE_MESSAGES`
(the pin/unpin permission). That is a **Discord OAuth-invite-scope** fix, not a config-file fix — and
the script is doing its job by refusing to report green when a real permission gap exists. A checker
that only ever prints PASS is worthless; this one exits non-zero until the gap is closed or explicitly
waived.

> **Two bugs we hit writing this script — worth knowing if you adapt it.** (1) `find "$HOME"` breaks
> under `sudo` because `$HOME` becomes `/root`; derive the account root from `$HERMES_HOME`'s parent
> instead. (2) The gateway-exit-diag log **only ever records failures** — it never logs a healthy
> running state — so counting "zero clean exits" falsely flags a perfectly healthy gateway as
> crash-looping. The authoritative signal for "is the gateway up?" is a live `pgrep`, not a log count.

