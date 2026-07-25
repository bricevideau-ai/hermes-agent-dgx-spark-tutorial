# Spinning Up a Hermes Agent on an NVIDIA DGX Spark

A hands-on, reproducible guide to standing up [Hermes Agent](https://github.com/NousResearch/hermes-agent) — an open-source, provider-agnostic AI agent by Nous Research — on an **NVIDIA DGX Spark** (GB10, ARM64). This walks from a bare box to a working agent you can talk to in the terminal, on Discord, and drive against both hosted APIs and a **local LLM served on the Spark itself**.

> Written and validated on a real DGX Spark (`piment`): Ubuntu 24.04.4 LTS, `aarch64`, NVIDIA GB10, driver 580.173.02, CUDA 13.0, Python 3.11.15, Hermes Agent v0.19.0.

---

## Table of Contents

1. [Why Hermes on a DGX Spark](#1-why-hermes-on-a-dgx-spark)
2. [Prerequisites & Hardware Baseline](#2-prerequisites--hardware-baseline)
3. [Install Hermes](#3-install-hermes)
4. [First Run & Model Setup](#4-first-run--model-setup)
5. [Verifying the Install (`hermes doctor`)](#5-verifying-the-install-hermes-doctor)
6. [Wiring Up a Local LLM on the Spark](#6-wiring-up-a-local-llm-on-the-spark)
7. [Adding the Discord Gateway](#7-adding-the-discord-gateway)
8. [Running Hermes as a Persistent Service](#8-running-hermes-as-a-persistent-service)
9. [Long-Term Memory (Mnemosyne)](#9-long-term-memory-mnemosyne)
10. [Google Integration (Gmail, Calendar, Drive, Docs)](#10-google-integration-gmail-calendar-drive-docs)
11. [Skills & Cron](#11-skills--cron)
12. [Reproducibility Checklist](#12-reproducibility-checklist)
13. [Troubleshooting on ARM64](#13-troubleshooting-on-arm64)
14. [Running a Second Agent on the Same Box](#14-running-a-second-agent-on-the-same-box)

---

## 1. Why Hermes on a DGX Spark

The DGX Spark is a compact, ARM64 (Grace-Blackwell GB10) developer box with a large unified-memory pool — well suited to running mid-sized LLMs locally. Pairing it with Hermes gives you:

- **A durable, self-improving agent** with persistent memory and skills, not just a chat window.
- **Provider independence** — swap between hosted APIs (Anthropic, OpenAI, OpenRouter, a private gateway) and a **local model served on the Spark** with a single config change.
- **A path off metered tokens** — offload as much agentic reasoning as the local GPU can handle, falling back to hosted models only when needed.
- **Multi-surface access** — the same agent on the CLI, Discord, Slack, email, and more.

The strategic goal in our deployment: measure how much agentic reasoning can be offloaded to a local model on the Spark to reduce dependence on paid token budgets, while keeping hosted models as a fallback.

---

## 2. Prerequisites & Hardware Baseline

Confirm your box before installing. On our reference machine:

```bash
uname -m                 # aarch64
cat /etc/os-release      # Ubuntu 24.04.4 LTS (Noble Numbat)
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader   # NVIDIA GB10, 580.173.02
nvcc --version           # CUDA 13.0
python3 --version        # 3.11.x
```

You need:

- **Ubuntu 24.04 LTS (ARM64)** or similar, with the NVIDIA driver already installed (the DGX Spark ships with it).
- **Python 3.10+** (3.11 recommended; the installer provisions its own via `uv` if needed).
- **`curl`, `git`, `build-essential`** for any packages that compile from source on ARM64.
- Outbound network access for the installer and (optionally) hosted model APIs.

```bash
sudo apt-get update
sudo apt-get install -y curl git build-essential
```

> **ARM64 note:** Almost everything in the Python ecosystem now ships `aarch64` wheels, but a few packages still build from source. Having `build-essential` (and occasionally `cmake`/`ninja`) present up front saves headaches.

---

## 3. Install Hermes

The official installer sets up `uv`, a pinned Python, a virtualenv, and the `hermes` launcher:

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

Then reload your shell (or `source ~/.bashrc`) so `hermes` is on `PATH`:

```bash
hermes --version
# Hermes Agent v0.19.0 (...)
```

Config and secrets live under `~/.hermes/`:

| Path | Purpose |
|------|---------|
| `~/.hermes/config.yaml` | Main configuration |
| `~/.hermes/.env` | API keys and secrets |
| `~/.hermes/skills/` | Installed skills |
| `~/.hermes/state.db` | Session store (SQLite + FTS5) |
| `~/.hermes/logs/` | Gateway and error logs |

---

## 4. First Run & Model Setup

Pick a model/provider interactively:

```bash
hermes setup      # full wizard
# or just the model picker:
hermes model
```

Hermes is provider-agnostic. Common choices:

- **Hosted API** — Anthropic, OpenAI, OpenRouter, Google, DeepSeek, xAI, etc. Set the relevant key in `~/.hermes/.env` (e.g. `OPENROUTER_API_KEY=...`).
- **Private/OpenAI-compatible gateway** — set `model.base_url` + `model.api_key` in `config.yaml` (see §6, which uses the same mechanism for a local model).

Quick smoke test:

```bash
hermes chat -q "In one sentence, what are you running on?"
```

Then drop into an interactive session:

```bash
hermes
```

---

## 5. Verifying the Install (`hermes doctor`)

```bash
hermes doctor          # checks dependencies + config
hermes status --all    # component status
hermes config check    # missing/outdated config keys
```

Fix anything flagged before moving on. `hermes doctor --fix` auto-resolves common issues.

---

## 6. Wiring Up a Local LLM on the Spark

This is the payoff of running on a DGX Spark: serve a model **on the box** and point Hermes at it. Any OpenAI-compatible server works — [vLLM](https://github.com/vllm-project/vllm), [llama.cpp](https://github.com/ggerganov/llama.cpp), SGLang, or Ollama. Below uses vLLM as the pattern.

### 6.1 Serve a model (vLLM example)

On ARM64 + CUDA, install vLLM into its own venv (keep it isolated from Hermes' venv):

```bash
python3 -m venv ~/vllm-venv
source ~/vllm-venv/bin/activate
pip install --upgrade pip
pip install vllm      # ARM64 CUDA wheels; build from source if no wheel for your CUDA
```

Launch an OpenAI-compatible server (swap in your model of choice):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <org/model-name> \
  --host 127.0.0.1 --port 8000 \
  --served-model-name local-model
```

Verify it answers:

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

> **Tip:** For the DGX Spark's unified-memory architecture, size the model + KV cache to the available pool and tune `--max-model-len` / `--gpu-memory-utilization`. Track each run (model, quant, flags, throughput, latency) in a results log so experiments are comparable across boxes.

### 6.2 Point Hermes at the local server

Add a custom provider in `~/.hermes/config.yaml`:

```yaml
model:
  default: local-model
  provider: custom
  base_url: http://127.0.0.1:8000/v1
  api_key: not-needed        # vLLM ignores it unless you set --api-key
```

Or via CLI:

```bash
hermes config set model.provider custom
hermes config set model.base_url http://127.0.0.1:8000/v1
hermes config set model.default local-model
```

Restart your session and test:

```bash
hermes chat -q "Say hello from the local model on the Spark."
```

Now you can flip between local and hosted models by editing `model.provider` / `model.base_url` — the rest of your agent (skills, memory, gateway) is unaffected.

---

## 7. Adding the Discord Gateway

Hermes' gateway runs the *same agent* — same tools, memory, and skills — on messaging platforms.
Discord is the most common target, and it's also **the step most likely to go subtly wrong**: the bot
can happily answer messages while still being misconfigured in ways that only bite later (can't pin,
doesn't show as a member, ignores everyone / ignores no one). Do it in this exact order.

### 7.1 Create the application and bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications) → **New
   Application**. Name it after the agent (e.g. `Corwin`).
2. Left sidebar → **Bot**. The bot user is created automatically.
3. On the **Bot** page, click **Reset Token**, then **Copy** — this is the value that goes in
   `~/.hermes/.env` (below). You only see it once; if you lose it, reset again.

### 7.2 Enable the privileged intent (the #1 silent failure)

Still on the **Bot** page, under **Privileged Gateway Intents**, turn on **Message Content Intent**
and click **Save Changes**.

> Without this the bot connects, shows green/online, and **silently ignores every message** — no error
> anywhere. If your bot is online but deaf, this is almost always why. (`Presence` and `Server
> Members` intents are optional and off by default; leave them off unless a skill needs them.)

### 7.3 Invite the bot **with a role** (the trap that cost us a day on the second agent)

Generate the invite under **OAuth2 → URL Generator**:

- **Scopes:** check **both** `bot` **and** `applications.commands`.
- **Bot Permissions:** at minimum `View Channels`, `Send Messages`, `Read Message History`,
  `Embed Links`, `Attach Files`, `Add Reactions`, and — critically — **`Manage Messages`** if you
  want the bot to pin/unpin. If in doubt, `Administrator` is simplest and you can scope down later.

Copy the generated URL at the bottom, open it, pick your server, and **Authorize**.

> **Why the permission checkboxes matter — a real failure.** When you invite a bot *with* a
> permissions bitmask, Discord auto-creates a **managed role** for it carrying those permissions. We
> invited our second agent with a bare `bot`-scope link and **no permissions selected**, so it landed
> on `@everyone` only with `roles: []`. Result: `403 Missing Permissions (50013)` on every pin, and it
> didn't render in the member sidebar. **A bot cannot grant itself a role** — the only fix is to
> re-invite with the correct OAuth2 URL. Get the permissions right in the invite the first time.

### 7.4 Wire the token into Hermes and run

```bash
# In ~/.hermes/.env :
#   DISCORD_BOT_TOKEN=<the token from step 7.1>
#   DISCORD_ALLOWED_USERS=<your_numeric_user_id>   # recommended: only you can command it

hermes gateway setup       # interactive platform config; select Discord, confirm the token
hermes gateway run         # FOREGROUND first — watch the logs before daemonizing
```

To get your numeric user ID: Discord → **User Settings → Advanced → Developer Mode ON**, then
right-click your name → **Copy User ID**.

### 7.5 Verify it's actually READY (not just "online")

"Shows as online" and "logged in and usable" are different states. Prove READY from the logs:

```bash
grep -E "Connected as" ~/.hermes/logs/gateway.log | tail
# Expect a line like:  [Discord] Connected as Corwin#8416   then   ✓ discord connected
```

That `Connected as …` line **only** prints on discord.py's `on_ready` event — that is the
authoritative "the session logged in" signal. A generic `response ready` line is unrelated
request-handling noise and does **not** prove login. Then send the bot a message from an allowed
account and confirm it replies. If you enabled `Manage Messages`, pin a throwaway message and unpin
it to confirm the permission actually landed (a non-empty `roles` array + a successful pin, not a 403).

> Provisioning a **second** bot on the same server hits extra role/permission edges — see the
> companion doc's [§5 Discord — roles and permissions](docs/spinning-up-a-second-agent.md#5-discord--roles-and-permissions).

---

## 8. Running Hermes as a Persistent Service

Once the gateway works interactively, install it as a background service so it survives logout/reboot:

```bash
hermes gateway install
hermes gateway start
hermes gateway status
```

**Critical on a headless/SSH box:** enable linger so the user service keeps running after you disconnect:

```bash
sudo loginctl enable-linger "$USER"
```

If the service ever crash-loops:

```bash
systemctl --user reset-failed hermes-gateway
```

Logs:

```bash
grep -iE "failed to send|error" ~/.hermes/logs/gateway.log | tail -20
```

---

## 9. Long-Term Memory (Mnemosyne)

This is the single feature that turns Hermes from a chat window into an agent that *remembers you*
across sessions — and it's the one most people wire up wrong, because it involves **two packages plus
a bridge**, and every failure mode is silent. Install it deliberately and verify each layer.

### 9.1 What the pieces are

| Piece | Package | Role |
|---|---|---|
| Memory engine | `mnemosyne-memory[all]` | The store itself (SQLite), **plus the vector-search + embeddings extras** (`sqlite-vec`, `fastembed`) and the `mnemosyne` CLI |
| Hermes bridge | `mnemosyne-hermes` | The plugin that lets Hermes *call* the engine (`mnemosyne_hermes` module) |

Two independent silent-failure modes to defend against:
1. Hermes can show `Provider: mnemosyne` while the **bridge is missing** — it thinks it has memory and every recall returns nothing.
2. **The engine can install without its vector backend.** A bare `pip install mnemosyne-memory` pulls in *only* `PyYAML` — `sqlite-vec` and `fastembed` are gated behind the **`[all]`** extra. Install bare and the store still "works" and a canary can even *store*, but semantic recall silently falls back to weak/keyword matching. **Always install `mnemosyne-memory[all]`.** (This one bit us: a canary passed on a box that happened to have `[all]` already, while the doc told readers to install bare — verify install steps in a *fresh* venv, not your provisioned box.)

You must confirm both the bridge **and** the vector backend (see §9.4).

### 9.2 Install into the Hermes venv (not system Python)

Everything must land in the **agent's own** Hermes virtualenv — on this box that is
`~/.hermes/hermes-agent/venv`. Activate it explicitly (the `~/.local/bin/hermes` launcher is a
wrapper script, **not** a symlink, so don't try to derive the venv from it — activate the known path):

```bash
source ~/.hermes/hermes-agent/venv/bin/activate
python -c "import sys; print(sys.executable)"   # sanity: should print a path under ~/.hermes/.../venv

pip install "mnemosyne-memory[all]"   # engine + `mnemosyne` CLI + vector backend (sqlite-vec) + embeddings (fastembed)
pip install mnemosyne-hermes          # the Hermes bridge (provides the mnemosyne_hermes module)
mnemosyne-install                     # relinks the Hermes plugin symlink so Hermes sees the bridge
```

> **Don't drop the `[all]`.** Bare `pip install mnemosyne-memory` requires only `PyYAML`; the
> semantic-recall dependencies (`sqlite-vec`, `fastembed`) come **only** with the `[all]` extra.
> Confirm they landed: `python -c "import sqlite_vec, fastembed; print('vector deps OK')"`.

> **Version note (this box, validated):** `mnemosyne-memory 3.14.0` (+ `sqlite-vec 0.1.9`,
> `fastembed 0.8.0`) + `mnemosyne-hermes 0.5.0`.
> `mnemosyne-install` lives inside the venv; the `mnemosyne` CLI is exposed on `~/.local/bin`.

> **Restart after `mnemosyne-install` — or the change is invisible.** The relinked plugin symlink is
> **not** picked up by an already-running process. If your gateway is up, restart it (and start a
> fresh CLI session) *before* checking status, or `hermes memory status` will report stale state and
> you'll think the install failed:
> ```bash
> systemctl --user restart hermes-gateway    # if running as a service; else just start a new session
> ```

### 9.3 Point Hermes at Mnemosyne and make recall survive across sessions

```bash
hermes config set memory.provider mnemosyne
```

The subtle part is **cross-session recall**: by default a fact stored in one session is invisible to
*other* sessions. Getting this right means understanding that **scope is set on write, and there are
three separate code paths that each read the "default scope" setting from a *different* place** — a
map we only fully untangled by reading the source and reproducing each path.

First, the recall behavior by stored scope (proven with a fresh DB and real `BeamMemory` store→recall
across two session IDs):

| Stored scope | no env var | `MNEMOSYNE_CROSS_SESSION=1` |
|---|---|---|
| `scope=global`  | ✅ recalled cross-session | ✅ recalled |
| `scope=session` | ❌ not recalled            | ✅ recalled |

So the goal is: **make your durable memories store at `scope=global`.** How you set that depends on
which path writes them.

**1. The agent (bridge) path — the one that matters. Set it in Hermes config.**
When *the agent* stores a memory (`mnemosyne_remember`), the bridge resolves the default scope from
**`memory.mnemosyne.default_scope` in Hermes' own `~/.hermes/config.yaml`**:

```bash
hermes config set memory.mnemosyne.default_scope global
hermes config get memory.mnemosyne.default_scope        # -> global
```

This is the **primary, sanctioned mechanism** and the one to set. Verified against the source: the
bridge (`mnemosyne_hermes`) reads this exact key via `read_hermes_config_key()` and applies it to every
`remember()` that doesn't pass an explicit scope.

> ⚠️ **Do *not* use `mnemosyne config set default_scope global` for this.** That writes Mnemosyne's
> *own* `~/.hermes/mnemosyne/config.yaml`, which the agent bridge never reads — and the `mnemosyne`
> **CLI** `store` doesn't read it either (see path 2). It looks like it works (`mnemosyne config get`
> reflects it) but changes nothing about what scope gets stored. Use the `hermes config` key above.

**2. The CLI path (`mnemosyne store`) — env var only (known bug).**
If you seed memories from the shell with `mnemosyne store`, its default scope is read **only** from the
`MNEMOSYNE_DEFAULT_SCOPE` environment variable — it ignores *both* config files and falls back to
`session`. To store globally from the CLI:

```bash
MNEMOSYNE_DEFAULT_SCOPE=global mnemosyne store "a durable fact"   # else it lands scope=session
```

> This is an upstream bug: `mnemosyne config set default_scope global` writes the config file,
> `mnemosyne config get` reads it back (looks applied), but `store` bypasses the config resolver
> (`cli.py`: `_resolve_default_scope()` reads only the env var). Reported upstream.

**3. The recall override `MNEMOSYNE_CROSS_SESSION=1` — for legacy session-scoped rows.**
This drops session filtering entirely (the filter becomes `(1=1)`), so it *additionally* exposes any
`session`-scoped rows you stored **before** switching to global. Set it only if you have such legacy
rows, or want the absolute guarantee. Set it via a **systemd drop-in** so it survives
`hermes gateway install` regenerating the unit (editing the main unit directly gets clobbered):


```bash
mkdir -p ~/.config/systemd/user/hermes-gateway.service.d
cat > ~/.config/systemd/user/hermes-gateway.service.d/10-mnemosyne-cross-session.conf <<'EOF'
[Service]
Environment="MNEMOSYNE_CROSS_SESSION=1"
EOF
systemctl --user daemon-reload && systemctl --user restart hermes-gateway   # same restart rule as 9.2

# Prove it landed in the LIVE process, not just the unit file:
tr '\0' '\n' < /proc/$(pgrep -u "$USER" -f hermes-gateway | head -1)/environ | grep MNEMOSYNE
# expect: MNEMOSYNE_CROSS_SESSION=1
```

> **⚠️ Known bug: the `cross_session` *config key* is a no-op for recall — it's env-var-only.**
> `cross_session` appears in Mnemosyne's config map with a documented `config.yaml > env vars`
> precedence, so it *looks* like `cross_session: true` in `config.yaml` should work. It does **not**:
> the recall path reads the toggle straight from the process environment at import time
> (`beam.py`: `_CROSS_SESSION = os.environ.get("MNEMOSYNE_CROSS_SESSION","0")=="1"`) and never consults
> the config resolver. Verified on this box: with `cross_session: true` in config,
> `config.get("cross_session")` returns `True` while the recall gate `_cross_session_enabled()` still
> returns `False`. **So the *override* only works via the env var.** (Reported upstream.) The
> `default_scope: global` mechanism above is unaffected by this bug — it goes through the normal SQL
> filter, not the toggle.

See the companion doc's
[§3b](docs/spinning-up-a-second-agent.md#3b-cross-session-scoping-bug--recall-returns-0-in-live-sessions)
and [§3c](docs/spinning-up-a-second-agent.md#3c-cli-store-defaults-to-session-scope).

### 9.4 Verify the whole chain — prove recall, don't trust the status line

```bash
# 1. Both the provider AND the plugin must report healthy:
hermes memory status
#    Expect:  Provider: mnemosyne   AND   Plugin: installed ✓ / Status: available ✓
#    A "Plugin: NOT installed ✗" here means 9.2 didn't take — re-run pip install + mnemosyne-install,
#    then RESTART (9.2) before re-checking.

# 2. Round-trip a canary and PROVE it comes back. `mnemosyne store` is positional
#    (store <content> [source] [importance]) — there is NO --scope flag; don't add one.
CANARY="canary-$(date +%s)"
mnemosyne store "$CANARY: memory round-trip works"    # -> "Stored: <id>"
mnemosyne recall "$CANARY"                            # MUST print the row back (id + content + score)
hermes mnemosyne stats                               # count should have incremented

# 3. Confirm the VECTOR backend is actually live (not a bare install that degraded to keyword match):
python -c "import sqlite_vec, fastembed; print('vector deps OK')"   # both must import
mnemosyne diagnose 2>/dev/null | grep -iE "sqlite-vec|vec_working|coverage"
#    Expect something like: sqlite-vec coverage complete (vec_working rows=N).
#    If these are MISSING, you installed bare instead of mnemosyne-memory[all] (§9.2) — reinstall.
```

The `mnemosyne recall` returning your canary row is the actual proof — a `store` that merely
"doesn't error" is not a verified round-trip. To also confirm the *agent* (not just the CLI) can
recall it, ask it in a session: `hermes chat -q "Recall the $CANARY fact."` — but note a bare CLI
`hermes chat` runs in a different process env than the gateway, so if the CLI recalls it and the
gateway doesn't, that's the cross-session/env-var boundary from §9.3, not a broken store.

If `hermes memory status` is green but recall returns nothing, the usual culprits are: the bridge
went into the wrong venv (§9.2), no restart after install (§9.2), or `MNEMOSYNE_CROSS_SESSION` isn't
in the *live gateway process* env (§9.3). The companion doc walks each independent break with a proof
step: [§3 Memory (Mnemosyne) — the three-way break](docs/spinning-up-a-second-agent.md#3-memory-mnemosyne--the-three-way-break).

---

## 10. Google Integration (Gmail, Calendar, Drive, Docs)

Hermes talks to Google Workspace through the bundled **`google-workspace`** skill, which manages
OAuth for you. There are **two paths** — pick by what you actually need, because they have very
different setup costs.

### 10.1 Decide: App Password (email only) vs. OAuth (full Workspace)

- **Just email?** Skip Google Cloud entirely. Use the **`himalaya`** skill with a Gmail **App
  Password** (Google Account → **Security → 2-Step Verification → App passwords**). Two minutes, no
  cloud project. Load the skill and follow its setup.
- **Calendar / Drive / Sheets / Docs (or email + those)?** Use the `google-workspace` skill with
  OAuth, below.

> **On this box:** email is handled via `himalaya` + an App Password; the OAuth `google-workspace`
> path is what you'd add for Drive/Calendar. Don't reuse one agent's Google token for another agent —
> each agent gets its own credentials (see the companion doc's
> [§6 Account & Identity Isolation](docs/spinning-up-a-second-agent.md#6-account--identity-isolation)).

### 10.2 One-time: create an OAuth client in Google Cloud (~5 min)

1. Create/select a project: <https://console.cloud.google.com/projectselector2/home/dashboard>
2. Enable the APIs you need from the **API Library**
   (<https://console.cloud.google.com/apis/library>): Gmail, Calendar, Drive, Sheets, Docs, People.
3. **Credentials → Create Credentials → OAuth 2.0 Client ID → Application type: _Desktop app_ →
   Create.** (Desktop-app type is what the skill's PKCE flow expects.)
4. If the OAuth app is still in **Testing**, add your Google account as a **test user** at
   <https://console.cloud.google.com/auth/audience> — otherwise you'll get `Error 403: access_denied`.
5. **Download the client-secret JSON** and note its path.

### 10.3 Authorize (works fully headless — no browser on the box)

The skill drives OAuth step-by-step so it works over SSH/Discord/Telegram with no local browser:

```bash
GSETUP="python ~/.hermes/skills/productivity/google-workspace/scripts/setup.py"

$GSETUP --check                                     # AUTHENTICATED? then you're done
$GSETUP --client-secret /path/to/client_secret.json # register the client
$GSETUP --auth-url                                  # prints an auth_url — open it in ANY browser
# After approving, the browser lands on a failed http://localhost:1/... page — THAT IS EXPECTED.
# Copy the ENTIRE redirected URL from the address bar and exchange it:
$GSETUP --auth-code "PASTE_THE_FULL_REDIRECT_URL_OR_CODE"
$GSETUP --check                                     # expect: AUTHENTICATED
```

`setup.py`'s real flags are exactly: `--check`, `--check-live`, `--client-secret`, `--auth-url`,
`--auth-code`, `--revoke`, `--install-deps`. There is **no `--services` or `--format` flag** — the
requested scopes are fixed in the skill (in `google_api.py`'s `SCOPES` list: Gmail read/send/modify,
Calendar, Drive). If you want true least-privilege, edit that `SCOPES` list before authorizing rather
than passing a flag. The token lands at `~/.hermes/google_token.json` and **auto-refreshes** from
then on. To revoke: `$GSETUP --revoke`.

### 10.4 Use it

```bash
GAPI="python ~/.hermes/skills/productivity/google-workspace/scripts/google_api.py"
$GAPI gmail search "is:unread" --max 10
$GAPI calendar list
$GAPI drive upload /path/to/report.pdf
```

> **Off-box backups to Drive** use a *different* mechanism — `rclone` with least-privilege
> `drive.file` scope and its own headless-authorize flow — covered in the companion doc's
> [§4 Backups](docs/spinning-up-a-second-agent.md#4-backups--an-untested-backup-is-not-a-backup).
> Don't conflate the `google-workspace` OAuth token (LLM/skill actions) with the rclone remote (bulk
> file sync); they are separate credentials with separate scopes on purpose.

---

## 11. Skills & Cron

Beyond memory, two more things make Hermes durable:

- **Skills** — reusable procedures the agent loads on demand and can author itself.
  ```bash
  hermes skills list
  hermes skills browse
  hermes skills install <id>
  ```
- **Cron** — durable scheduled agent runs.
  ```bash
  hermes cron create "0 9 * * *"    # e.g. a daily briefing
  hermes cron list
  ```

For a research program (e.g. benchmarking many local models over time), pair cron jobs with a skill
that runs a standard benchmark suite and appends results to a persistent registry — so runs are
reproducible and comparable across DGX Spark boxes.

---

## 12. Reproducibility Checklist

For team deployments across multiple Spark boxes, capture:

- [ ] OS + arch (`uname -a`, `/etc/os-release`)
- [ ] GPU + driver + CUDA (`nvidia-smi`, `nvcc --version`)
- [ ] Hermes version (`hermes --version`)
- [ ] `~/.hermes/config.yaml` (model, provider, base_url — **redact secrets**)
- [ ] Local-serving stack + exact launch flags (vLLM version, `--model`, `--max-model-len`, `--gpu-memory-utilization`)
- [ ] Per-run benchmark results (model, quant, throughput, latency, context length)

Commit these (minus secrets) so another box can be brought up identically.

---

## 13. Troubleshooting on ARM64

| Symptom | Fix |
|---|---|
| `hermes` not found after install | `source ~/.bashrc`; confirm the installer added it to `PATH` |
| Model/provider errors | `hermes doctor`; check the API key in `~/.hermes/.env`; `hermes auth` for OAuth providers |
| Discord bot **online but ignores messages** | Enable **Message Content Intent** (Developer Portal → Bot → Privileged Gateway Intents) — see §7.2 |
| Discord bot **can't pin / `403 Missing Permissions (50013)`** | Bot was invited without a role; re-invite via an OAuth2 URL granting `Manage Messages` — see §7.3 |
| Discord bot not in member sidebar | Same root cause as above — no managed role from the invite |
| `hermes memory status` shows `Plugin: NOT installed ✗` | Install `mnemosyne-hermes` into the **Hermes venv** and run `mnemosyne-install` — see §9.2 |
| Memory status green but recall returns nothing | Scope is `session` not `global` (§9.3), or bridge went into the wrong venv, or `MNEMOSYNE_CROSS_SESSION` missing from the live gateway env |
| Google OAuth `Error 403: access_denied` | Add your account as a **test user** at the OAuth audience page — see §10.2 |
| Gateway dies on SSH logout | `sudo loginctl enable-linger $USER` |
| Gateway crash loop | `systemctl --user reset-failed hermes-gateway` |
| pip package builds from source (ARM64) | Ensure `build-essential` (+ `cmake`/`ninja`) are installed |
| vLLM OOM on GB10 | Lower `--max-model-len` and `--gpu-memory-utilization`; pick a smaller/quantized model |
| Auxiliary tasks (vision/compression) fail silently | Set `OPENROUTER_API_KEY` or `GOOGLE_API_KEY`, or configure `auxiliary.*.provider` |

---

## 14. Running a Second Agent on the Same Box

Standing up a **second, fully independent agent** (its own Linux user, `$HERMES_HOME`, memory,
backups, and Discord bot) has its own set of sharp edges — model wiring that silently downgrades to
the local model, memory that recalls nothing cross-session, backups that never actually push
off-box, and a Discord bot with no role. We hit all of them provisioning our second agent, and wrote
them up with symptom → root cause → fix → **verification** for each:

**→ [Spinning Up a Second Hermes Agent (and Not Botching It)](docs/spinning-up-a-second-agent.md)**

Start with its [Five-Point Pre-Flight Checklist](docs/spinning-up-a-second-agent.md#7-the-five-point-pre-flight-checklist).

---

## License

MIT. Contributions and corrections welcome — open an issue or PR.

## References

- Hermes Agent: <https://github.com/NousResearch/hermes-agent>
- Docs: <https://hermes-agent.nousresearch.com/docs/>
- vLLM: <https://github.com/vllm-project/vllm>
- NVIDIA DGX Spark: <https://www.nvidia.com/en-us/products/workstations/dgx-spark/>
