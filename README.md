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
13. [Troubleshooting](#13-troubleshooting)
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

### 7.2 Enable the privileged intent

Still on the **Bot** page, under **Privileged Gateway Intents**, turn on **Message Content Intent**
and click **Save Changes**. (`Presence` and `Server Members` intents are optional and off by default;
leave them off unless a skill needs them.)

> Skipping this is the single most common Discord failure — the bot goes online but silently ignores
> every message. If that happens, see [§13.2 Discord troubleshooting](#132-discord-gateway).

### 7.3 Invite the bot with a role

Generate the invite under **OAuth2 → URL Generator**:

- **Scopes:** check **both** `bot` **and** `applications.commands`.
- **Bot Permissions:** at minimum `View Channels`, `Send Messages`, `Read Message History`,
  `Embed Links`, `Attach Files`, `Add Reactions`, and — critically — **`Manage Messages`** if you
  want the bot to pin/unpin. If in doubt, `Administrator` is simplest and you can scope down later.

Copy the generated URL at the bottom, open it, pick your server, and **Authorize**.

> **Get the permissions right in the invite the first time** — a bot cannot grant itself a role
> afterward. If it lands with no role (can't pin, missing from the member sidebar), see
> [§13.2 Discord troubleshooting](#132-discord-gateway).

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
systemctl --user restart hermes-gateway   # REQUIRED: the relinked plugin is invisible until restart (else start a fresh session)
```

> **Keep the `[all]` extra** — bare `mnemosyne-memory` ships without semantic recall, and you **must
> restart** after `mnemosyne-install` or the plugin looks uninstalled. Both are silent traps; the
> *why* and how to confirm are in [§13.3 Memory (Mnemosyne) troubleshooting](#133-memory-mnemosyne).

> **Version note (this box, validated):** `mnemosyne-memory 3.14.0` (+ `sqlite-vec 0.1.9`,
> `fastembed 0.8.0`) + `mnemosyne-hermes 0.5.0`.

### 9.3 Point Hermes at Mnemosyne and make recall survive across sessions

On a **fresh install**, two `hermes config set` commands are all you need for durable, cross-session
memory. Set the provider, then set the default write scope to `global`:

```bash
hermes config set memory.provider mnemosyne
hermes config set memory.mnemosyne.default_scope global
hermes config get memory.mnemosyne.default_scope        # -> global
```

The reason cross-session recall hinges on that second key: the recall filter is
`(session_id = ? OR scope = 'global')`, so a fact is only visible to *other* sessions if it was stored
at **`scope=global`**. Setting `memory.mnemosyne.default_scope global` from the start makes every fact
the agent stores land at global scope, so it survives across sessions with no further tuning.

`memory.mnemosyne.default_scope` is the **primary, sanctioned mechanism** — the one to set. Verified
against the source: the Hermes bridge (`mnemosyne_hermes`) reads *this exact key* from Hermes' own
`~/.hermes/config.yaml` via `read_hermes_config_key()` and applies it to every `remember()` that
doesn't pass an explicit scope.

> **⚠️ Fresh install: you do NOT need `MNEMOSYNE_CROSS_SESSION`.** With `default_scope global` set
> from the start, every fact is already global and recalls cross-session on its own. The
> `MNEMOSYNE_CROSS_SESSION=1` override is only for *migrating* a pre-existing DB that already holds
> legacy `scope=session` rows — see [§9.5 Memory troubleshooting](#95-memory-troubleshooting). Don't
> set it reflexively.

#### Which tool stores durable facts — Mnemosyne (⚙️), not the legacy `memory` tool (🧠)

Configuring the provider is only half the job: the agent has to write durable facts with the **right
tool**. A correctly-deployed agent stores anything that must survive across sessions with the
**Mnemosyne** tools — `mnemosyne_remember`, `mnemosyne_recall`, and friends. In the Hermes UI these
render with a **⚙️ cog** icon.

The older built-in **`memory`** tool (the one backed by `MEMORY.md` / `USER.md`, rendered with a
**🧠 brain** icon) is **deprecated for durable storage**. It is for *ephemeral session state* only —
scratch notes within the current conversation — and does **not** participate in Mnemosyne's
cross-session recall. If your agent "remembers" via the 🧠 brain tool, those facts will not come back
in a fresh session no matter how `default_scope` is set.

> **Rule of thumb:** durable fact → **Mnemosyne (⚙️ cog)**. Throwaway session note → legacy `memory`
> (🧠 brain). When in doubt, use Mnemosyne.

> **⚠️ One trap worth flagging up front:** if you seed facts from the *shell* with `mnemosyne store`,
> it silently defaults to `scope=session` (it reads scope from an env var, not from the config key you
> just set) — so shell-seeded facts won't recall cross-session unless you override it. This bites fresh
> readers; the fix is in [§9.5 Memory troubleshooting](#95-memory-troubleshooting).

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

# 4. PROVE the agent path stores at scope=global (this is the proof your §9.3 config took).
#    After the agent (via the gateway) stores something with mnemosyne_remember, read the row back:
sqlite3 ~/.hermes/mnemosyne/data/mnemosyne.db \
  "SELECT scope, substr(content,1,40) FROM working_memory ORDER BY rowid DESC LIMIT 3;"
#    Expect the newest agent-written rows to show scope=global. If they show scope=session, the
#    default_scope key from §9.3 didn't take — re-check `hermes config get memory.mnemosyne.default_scope`.
```

The `mnemosyne recall` returning your canary row is the actual proof — a `store` that merely
"doesn't error" is not a verified round-trip. To also confirm the *agent* (not just the CLI) can
recall it, ask it in a session: `hermes chat -q "Recall the $CANARY fact."` — but note a bare CLI
`hermes chat` runs in a different process env than the gateway, so if the CLI recalls it and the
gateway doesn't, that's the cross-session/env-var boundary from §9.3, not a broken store.

If `hermes memory status` is green but recall returns nothing, the usual culprits are: the bridge
went into the wrong venv (§9.2), no restart after install (§9.2), or `MNEMOSYNE_CROSS_SESSION` isn't
in the *live gateway process* env (§9.5). The companion doc walks each independent break with a proof
step: [§3 Memory (Mnemosyne) — the three-way break](docs/spinning-up-a-second-agent.md#3-memory-mnemosyne--the-three-way-break).

### 9.5 Memory troubleshooting

Everything in §9.3–§9.4 is the clean, correct deploy path for a **fresh install**. The items below are
**workarounds and known upstream traps** — you should not need them on a fresh, correctly-configured
box. They're collected here so they don't clutter the happy path. Reach for them only when a specific
symptom below matches. The companion doc has the full break-by-break analysis:
[§3 Memory (Mnemosyne) — the three-way break](docs/spinning-up-a-second-agent.md#3-memory-mnemosyne--the-three-way-break).

**Recall behavior by stored scope** (proven with a fresh DB and real `BeamMemory` store→recall across
two session IDs) — the reason `scope=global` is what you want:

| Stored scope | no env var | `MNEMOSYNE_CROSS_SESSION=1` |
|---|---|---|
| `scope=global`  | ✅ recalled cross-session | ✅ recalled |
| `scope=session` | ❌ not recalled            | ✅ recalled |

#### 9.5.1 Migrating a pre-existing DB with legacy `scope=session` rows — `MNEMOSYNE_CROSS_SESSION=1`

**When you need this:** *only* if you switched to `default_scope=global` **after** already storing
facts at the default `scope=session` (i.e. you're migrating an existing DB). On a **fresh install**
configured with `default_scope=global` from the start, **you do not need this** — every new fact is
already global. Do not set it reflexively.

`MNEMOSYNE_CROSS_SESSION=1` drops session filtering entirely (the recall filter becomes `(1=1)`), so it
*additionally* exposes any `session`-scoped rows written before the switch. Set it via a **systemd
drop-in** so it survives `hermes gateway install` regenerating the unit (editing the main unit directly
gets clobbered):

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
> (Re-verified on Mnemosyne **v3.14.0**.) `cross_session` appears in Mnemosyne's config map with a
> documented `config.yaml > env vars` precedence, so it *looks* like `cross_session: true` in
> `config.yaml` should work. It does **not**: the recall path reads the toggle straight from the
> process environment at import time (`beam.py`: `_cross_session_enabled()` reads only
> `os.environ["MNEMOSYNE_CROSS_SESSION"]`) and never consults the config resolver. Verified on this
> box: with `cross_session: true` in config, `config.get("cross_session")` returns `True` while the
> recall gate `_cross_session_enabled()` still returns `False`. **So the *override* only works via the
> env var.** (Reported upstream.) The `default_scope: global` mechanism in §9.3 is unaffected by this
> bug — it goes through the normal SQL filter, not the toggle.

#### 9.5.2 Seeding facts from the shell — `mnemosyne store` defaults to `scope=session`

If you seed memories from the shell with `mnemosyne store`, its default scope is read **only** from the
`MNEMOSYNE_DEFAULT_SCOPE` environment variable — it ignores *both* config files (including the
`hermes config` key from §9.3) and falls back to `session`. To store globally from the CLI:

```bash
MNEMOSYNE_DEFAULT_SCOPE=global mnemosyne store "a durable fact"   # else it lands scope=session
```

> **⚠️ Upstream bug (re-verified on Mnemosyne v3.14.0):** `mnemosyne config set default_scope global`
> writes Mnemosyne's own `~/.hermes/mnemosyne/config.yaml`, and `mnemosyne config get` reads it back
> (so it *looks* applied), but `store` bypasses the config resolver entirely — `cli.py`'s `cmd_store`
> → `_resolve_default_scope()` reads only the `MNEMOSYNE_DEFAULT_SCOPE` env var. So
> `mnemosyne config set default_scope global` is effectively a **no-op** for what scope actually gets
> stored. Note the agent bridge does *not* use this file either; it reads `memory.mnemosyne.default_scope`
> from *Hermes* config (§9.3). Reported upstream.

#### 9.5.3 False negative: a hand-built provider in a REPL always reports `scope=session`

**Symptom:** you spin up a `MnemosyneMemoryProvider()` in a Python REPL to "check" the scope, and it
reports `session` even though `memory.mnemosyne.default_scope` is correctly `global` — making a
correctly-configured system *look broken*.

**Root cause (a harness gap, not a bug):** the provider learns where Hermes' config lives *only* from a
`hermes_home` kwarg passed at init (`self._hermes_home = kwargs.get("hermes_home", "")` — there is
**no** fallback to `HERMES_HOME` or `get_hermes_home()`). A hand-rolled `MnemosyneMemoryProvider()`
gets `_hermes_home=""`, `read_hermes_config_key("", …)` returns `None`, and the default scope silently
stays `session`. We reproduced exactly this false negative.

**Fix — validate through the live agent, not a hand-built provider.** Use the §9.4 step #4 proof: have
the agent store a fact via the gateway, then read the row's scope back from the DB and expect
`scope=global`.

See also the companion doc's
[§3b](docs/spinning-up-a-second-agent.md#3b-cross-session-scoping--recall-returns-0-in-live-sessions)
and [§3c](docs/spinning-up-a-second-agent.md#3c-cli-store-scope-and-how-to-actually-prove-a-round-trip).

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

## 13. Troubleshooting

The main walkthrough (§1–§12) is the **clean deploy path** for a fresh box. This section collects the
**known traps, silent failures, and "this bit us" war-stories** — each as *symptom → cause → proof/fix* —
so they don't clutter the happy path. Reach for a subsection only when its symptom matches.

### 13.1 Quick reference

| Symptom | Fix |
|---|---|
| `hermes` not found after install | `source ~/.bashrc`; confirm the installer added it to `PATH` |
| Model/provider errors | `hermes doctor`; check the API key in `~/.hermes/.env`; `hermes auth` for OAuth providers |
| Discord bot **online but ignores messages** | Enable **Message Content Intent** — see [§13.2](#132-discord-gateway) |
| Discord bot **can't pin / `403 Missing Permissions (50013)`** | Re-invite with a role granting `Manage Messages` — see [§13.2](#132-discord-gateway) |
| Discord bot not in member sidebar | Same root cause — no managed role from the invite — see [§13.2](#132-discord-gateway) |
| `hermes memory status` shows `Plugin: NOT installed ✗` | Install into the **Hermes venv** + `mnemosyne-install` + **restart** — see [§13.3](#133-memory-mnemosyne) |
| Memory status green but recall returns nothing | Scope is `session` not `global` (§9.3), wrong venv, or missing restart — see [§13.3](#133-memory-mnemosyne) |
| Google OAuth `Error 403: access_denied` | Add your account as a **test user** at the OAuth audience page — see §10.2 |
| Gateway dies on SSH logout | `sudo loginctl enable-linger $USER` |
| Gateway crash loop | `systemctl --user reset-failed hermes-gateway` |

### 13.2 Discord gateway

**Bot is online (green) but silently ignores every message — no error anywhere.**
Cause: **Message Content Intent** was not enabled. When you invite a bot *with* a permissions
bitmask, Discord relies on this privileged intent to deliver message text; without it the session
connects and shows online but receives empty message bodies. Fix: Developer Portal → **Bot** →
**Privileged Gateway Intents** → enable **Message Content Intent** → **Save Changes** (§7.2), then
restart the gateway.
*Proof it's fixed:* send the bot an allowed message and confirm it replies.

**Bot can't pin (`403 Missing Permissions (50013)`) and/or doesn't render in the member sidebar.**
Cause — *the trap that cost us a day on the second agent:* when you invite a bot with a permissions
bitmask, Discord auto-creates a **managed role** carrying those permissions. We invited our second
agent with a bare `bot`-scope link and **no permissions selected**, so it landed on `@everyone` only
with `roles: []` — hence 403 on every pin and no member-sidebar entry. **A bot cannot grant itself a
role.** Fix: re-invite via an **OAuth2 URL** with the correct **Bot Permissions** (at minimum
`Manage Messages` for pinning) — see §7.3.
*Proof it's fixed:* the bot now shows a non-empty `roles` array and a pin/unpin round-trips without a 403.

> Provisioning a **second** bot on the same server hits extra role/permission edges — see the companion
> doc's [§5 Discord — roles and permissions](docs/spinning-up-a-second-agent.md#5-discord--roles-and-permissions).

### 13.3 Memory (Mnemosyne)

**`hermes memory status` shows `Plugin: NOT installed ✗` right after a correct install.**
Two silent causes:
- **Missing `[all]` extra.** Bare `pip install mnemosyne-memory` pulls only `PyYAML`; the
  semantic-recall deps (`sqlite-vec`, `fastembed`) come **only** with `mnemosyne-memory[all]` (§9.2).
  *Proof:* `python -c "import sqlite_vec, fastembed; print('vector deps OK')"` must succeed.
- **No restart after `mnemosyne-install`.** The relinked plugin symlink is **not** picked up by an
  already-running process, so status reports stale "not installed" state. Fix: restart the gateway
  (and start a fresh CLI session) *before* re-checking:
  ```bash
  systemctl --user restart hermes-gateway    # if running as a service; else just start a new session
  ```
  *(`mnemosyne-install` lives inside the venv; the `mnemosyne` CLI is exposed on `~/.local/bin`.)*

**Memory status is green but recall returns nothing.** Usual culprits: scope is `session` not
`global` (§9.3), the bridge went into the wrong venv (§9.2), or no restart after install (above). The
full break-by-break analysis with a proof step for each is in the companion doc's
[§3 Memory (Mnemosyne) — the three-way break](docs/spinning-up-a-second-agent.md#3-memory-mnemosyne--the-three-way-break),
and the scope/migration overrides live in [§9.5 Memory troubleshooting](#95-memory-troubleshooting).

### 13.4 Local serving & ARM64 build issues

| Symptom | Fix |
|---|---|
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
