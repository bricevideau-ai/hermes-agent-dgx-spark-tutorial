# Spinning Up a Hermes Agent on an NVIDIA DGX Spark

A hands-on, reproducible guide to standing up [Hermes Agent](https://github.com/NousResearch/hermes-agent) — an open-source, provider-agnostic AI agent by Nous Research — on an **NVIDIA DGX Spark** (GB10, ARM64). This walks from a bare box to a working agent you can talk to in the terminal, on Discord, and drive against both hosted APIs and a **local LLM served on the Spark itself**.

> Written and validated on a real DGX Spark (`piment`): Ubuntu 24.04.4 LTS, `aarch64`, NVIDIA GB10, driver 580.173.02, CUDA 13.0, Python 3.11.15, Hermes Agent v0.19.0.

**One tutorial, one agent — and more if you want them.** This is a single linear guide: follow it top to bottom to bring up an agent. Everything here applies to *every* agent you provision — there is no separate "first agent" vs "second agent" procedure, because there isn't one: the prerequisites, the install, the model wiring, the memory setup, and every trap are identical whether it's your only agent or your fourth. The handful of steps that *only* matter once you run **more than one agent on the same box** (isolating identities, sharing a slice of memory between them) are called out inline with a clear marker:

> 🔀 **Multiple agents only** — skip this if you're running a single agent on the box.

So: read straight through for one agent; watch for the 🔀 markers if you're adding another.

---

## Table of Contents

1. [Why Hermes on a DGX Spark](#1-why-hermes-on-a-dgx-spark)
2. [Prerequisites & Hardware Baseline](#2-prerequisites--hardware-baseline)
3. [Install Hermes](#3-install-hermes)
4. [Wiring Up a Local LLM on the Spark](#4-wiring-up-a-local-llm-on-the-spark)
5. [Model & Provider Wiring](#5-model--provider-wiring)
6. [Verifying the Install (`hermes doctor`)](#6-verifying-the-install-hermes-doctor)
7. [Adding the Discord Gateway](#7-adding-the-discord-gateway)
8. [Running Hermes as a Persistent Service](#8-running-hermes-as-a-persistent-service)
9. [Long-Term Memory (Mnemosyne)](#9-long-term-memory-mnemosyne)
10. [Web Search & Extraction](#10-web-search--extraction)
11. [Google Integration (Gmail, Calendar, Drive, Docs)](#11-google-integration-gmail-calendar-drive-docs)
12. [Backups — an untested backup is not a backup](#12-backups--an-untested-backup-is-not-a-backup)
13. [Skills & Cron](#13-skills--cron)
14. [Running More Than One Agent on the Same Box](#14-running-more-than-one-agent-on-the-same-box) 🔀
15. [Verifying a Healthy Agent](#15-verifying-a-healthy-agent)
16. [Reproducibility Checklist](#16-reproducibility-checklist)
17. [Troubleshooting](#17-troubleshooting)

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
- **`sudo`** on the box — you need it for the base packages below, for `enable-linger` (§8), and, if you run more than one agent, for creating each agent's Linux user (§14). The sudo requirement is the same whether you run one agent or several.
- Outbound network access for the installer and (optionally) hosted model APIs.

```bash
sudo apt-get update
sudo apt-get install -y curl git build-essential
```

> **ARM64 note:** Almost everything in the Python ecosystem now ships `aarch64` wheels, but a few packages still build from source. Having `build-essential` (and occasionally `cmake`/`ninja`) present up front saves headaches.

### 2.1 Each agent is a dedicated Linux user

Run **each** agent as its own Linux user, in that user's home. Even for your very first agent this keeps memory, secrets, sessions, and the gateway service cleanly isolated under one `$HERMES_HOME` (`~/.hermes`), and it means adding a second agent later is just "repeat as a new user" rather than a migration. A second agent is **never** a second profile under the first user — it's a separate user.

```bash
# As an admin, create the agent's user (repeat per agent, e.g. corwin-ai, deirdre-ai):
sudo adduser --disabled-password --gecos "" <agent-user>
sudo usermod -aG sudo,docker <agent-user>     # grant the same groups the box's agents use, deliberately
sudo loginctl enable-linger <agent-user>      # so the user's gateway service survives logout (see §8)
```

Everything after this runs **as that user**, in that user's `~/.hermes`.

> **Operating another user's `--user` services.** The per-user systemd bus is keyed on UID, so from a *different* login you must point at the target UID's runtime dir:
>
> ```bash
> # Run a --user systemctl command for uid 1002 from a different login:
> sudo -u <agent-user> XDG_RUNTIME_DIR=/run/user/1002 \
>   DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1002/bus \
>   systemctl --user restart hermes-gateway.service
> ```

> **Pitfall (real):** `sudo cmd <<'HEREDOC'` fails auth because the heredoc consumes sudo's password channel. Write the script to a temp file and run `sudo python3 /tmp/x.py` instead. With passwordless sudo configured this is moot, but don't rely on that on a fresh box.

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

## 4. Wiring Up a Local LLM on the Spark

This is the payoff of running on a DGX Spark: serve a model **on the box** and point Hermes at it. Any OpenAI-compatible server works — [vLLM](https://github.com/vllm-project/vllm), [llama.cpp](https://github.com/ggerganov/llama.cpp), SGLang, or Ollama. Below uses vLLM as the pattern.

### 4.1 Serve a model (vLLM example)

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

### 4.2 Point Hermes at the local server

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

Now you can flip between local and hosted models by editing `model.provider` / `model.base_url` — the rest of your agent (skills, memory, gateway) is unaffected. This local endpoint is also what §5.2 uses as a failover target and what §9 uses as the memory-consolidation summarizer.

> **Which model is "main"?** Setting `model.default: local-model` here gives you a fully self-contained agent that runs entirely on the box — the fastest way to confirm the whole stack works end-to-end. In [§5](#5-model--provider-wiring) you'll decide your *production* primary: many Spark setups run a frontier hosted model (e.g. Claude Opus) as `model.default` and keep this local endpoint as the **fallback** (§5.2) and auxiliary-task model (§5.3). Either choice is valid — §5 simply overwrites `model.default` with whichever you pick.

---

## 5. Model & Provider Wiring

Pick a model/provider interactively:

```bash
hermes setup      # full wizard
# or just the model picker:
hermes model
```

Hermes is provider-agnostic. Common choices:

- **Hosted API** — Anthropic, OpenAI, OpenRouter, Google, DeepSeek, xAI, etc. Set the relevant key in `~/.hermes/.env` (e.g. `OPENROUTER_API_KEY=...`).
- **Private/OpenAI-compatible gateway** — set `model.base_url` + `model.api_key` in `config.yaml` (the same mechanism §4 uses for a local model).

Quick smoke test, then drop into an interactive session:

```bash
hermes chat -q "In one sentence, what are you running on?"
hermes            # interactive
```

### 5.1 Define `custom_providers:` explicitly

A `provider: custom` entry with **no** matching `custom_providers:` block resolves straight to whatever `base_url` points at, with no model metadata and no validation. If that URL happens to be a weak local endpoint, the agent runs on it silently — it boots, answers, and never errors, because the config is technically valid. The result is an agent that is quietly running on the wrong model.

Avoid this by always declaring the provider explicitly and pointing `model.default` at the real model:

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
    api_mode: anthropic_messages   # Anthropic models — see the api_mode note below
    models:
      - Claude Opus 4.8
      - Claude Sonnet 5
      # ...whatever the gateway serves
  - name: Local vLLM
    base_url: http://localhost:8000/v1
    model: <local-model>
    api_mode: chat_completions     # OpenAI-compatible server (vLLM) — chat_completions is correct
    models:
      <local-model>:
        context_length: 262144
```

> **`api_mode` must match the model family, not the gateway.** Anthropic
> models (Claude Opus, Sonnet, Haiku) speak the Anthropic Messages wire and
> must use `api_mode: anthropic_messages` — even when reached through an
> OpenAI-style gateway URL. Routing a Claude model over `chat_completions`
> can appear to work on some relays but fails hard on others: on an Argo-style
> gateway it returns **`403 PERMISSION_DENIED`** (the chat-completions path is
> a different, non-entitled backend endpoint), leaving the model dead with no
> obvious config error. Use `chat_completions` only for genuinely
> OpenAI-compatible servers — your local vLLM, OpenAI, most aggregators. The
> quick test: a Claude model that 403s on completions but returns 200 on
> `/v1/messages` is telling you to switch modes.

**Verify — don't assume.** Make the *running* agent report what it's on, and cross-check the raw endpoint (some servers return `200` with empty content):

```bash
# 1. Ask the live agent:
hermes chat -q "State your exact model and provider base_url in one line."

# 2. Cross-check the raw endpoint actually completes:
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<local-model>","messages":[{"role":"user","content":"say ALIVE"}]}' \
  | python3 -m json.tool
```

A `200` with `content: null` is **not** working. Read the payload, not just the status code.

### 5.2 Fallback chain — redundancy in hardware is not redundancy in config

Route everything through a single provider and you have a single point of failure: if the primary gateway drops, the agent dies — even with a local GPU sitting right there serving a model. Configure a `fallback_providers` chain so the agent survives an outage on a **different substrate**.

`hermes fallback add` is an interactive picker (no scripting flags); its **"Custom endpoint (enter URL manually)"** option lets you point at `http://localhost:8000/v1`. Or declare it directly:

```yaml
fallback_providers:
  - provider: custom
    model: <local-model>
    base_url: http://localhost:8000/v1
    api_mode: chat_completions
```

> **Design note:** a fallback to *another model on the same gateway* is worthless — when the gateway is down, both go down together. The only fallback that survives a gateway outage is one on a **different substrate**, i.e. the local GPU. Standing up the local LLM in §4 pays off twice: cheap inference *and* a real failover target.

**Verify — force a real failover.** A fallback in the list is a hypothesis until you watch it catch. In an **isolated** config copy (never your live one), break only the top-level `model.base_url` and fire one query:

```bash
export HERMES_HOME=/tmp/hermes-failover-test
cp -r ~/.hermes "$HERMES_HOME"
# Edit $HERMES_HOME/config.yaml: set the TOP-LEVEL model.base_url to a dead
# port, e.g. http://127.0.0.1:59999/v1 (not the first base_url under
# custom_providers — that's a different line and not the primary route).
hermes chat -q "Reply exactly: FAILOVER-WORKS"
# Expect: "🔄 Switched to fallback model: ... → <local-model> via custom" then the answer.
```

### 5.3 Fallback for auxiliary tasks (compression, title generation)

The main model isn't the only thing that calls out. Housekeeping tasks —
conversation **compression** and **title generation** — run on their own model,
configured under `auxiliary:`. On a Spark you'll want these on the cheap local
model, but they must still work when the local server is down, so give each one
its own `fallback_chain` to a hosted model.

Point the primary at the local endpoint and the fallback at a **cheaper hosted
model than your main** — there's no reason to spend a frontier model's rate on
housekeeping. Here the main agent is Opus, so the auxiliary fallback is Sonnet:

```yaml
auxiliary:
  compression:
    provider: custom
    model: <local-model>
    base_url: http://localhost:8000/v1
    api_key: local
    timeout: 180
    fallback_chain:
      - provider: custom
        model: Claude Sonnet 5
        api_mode: anthropic_messages   # Claude fallback — Anthropic wire, NOT chat_completions
        base_url: https://<your-gateway>/v1
        api_key: <your-key>
        timeout: 300
  title_generation:
    provider: custom
    model: <local-model>
    base_url: http://localhost:8000/v1
    api_key: local
    timeout: 60
    fallback_chain:
      - provider: custom
        model: Claude Sonnet 5
        api_mode: anthropic_messages   # Claude fallback — Anthropic wire, NOT chat_completions
        base_url: https://<your-gateway>/v1
        api_key: <your-key>
        timeout: 300
```

Notes that bite if you skip them:

- Both `provider: custom` **and** `base_url` are required on the primary and on
  each fallback entry — a bare `base_url` without `provider: custom` won't
  engage the fallback ladder.
- **Set `api_mode: anthropic_messages` on every Claude fallback entry.** A
  fallback entry inherits nothing from the main model's `api_mode`; omit it and
  the entry defaults to `chat_completions`, which 403s on a Claude model (§5.1).
  A fallback that fails the same way as the thing it's backing up is not a
  fallback — prove it with the `/v1/messages` curl below.
- `fallback_chain` is a YAML **block sequence** (a real list). `hermes config
  set auxiliary.compression.fallback_chain '[...]'` stores a literal string, not
  a list — author list-valued keys with `hermes config edit`, not `config set`.
- Changes to `config.yaml` are read at process start: they take effect on the
  **next gateway restart**, not live.

**Verify.** Confirm the fallback model actually answers on your gateway before
trusting it, then confirm the config parses as a list:

```bash
# 1. Prove the fallback model completes on the gateway.
#    Claude models use the Anthropic Messages endpoint (/v1/messages), NOT
#    /v1/chat/completions — the same api_mode: anthropic_messages rule from §5.1.
curl -s https://<your-gateway>/v1/messages \
  -H "Authorization: Bearer ***" -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"Claude Sonnet 5","max_tokens":10,"messages":[{"role":"user","content":"say OK"}]}' \
  | python3 -m json.tool

# 2. Prove the config read back as a list, not a string:
hermes config get auxiliary.compression.fallback_chain.0.model   # -> Claude Sonnet 5
```

---

## 6. Verifying the Install (`hermes doctor`)

```bash
hermes doctor          # checks dependencies + config
hermes status --all    # component status
hermes config check    # missing/outdated config keys
```

Fix anything flagged before moving on. `hermes doctor --fix` auto-resolves common issues.

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
> every message. If that happens, see [§17.2 Discord troubleshooting](#172-discord-gateway).

### 7.3 Invite the bot with a role

Generate the invite under **OAuth2 → URL Generator**:

- **Scopes:** check **both** `bot` **and** `applications.commands`.
- **Bot Permissions:** at minimum `View Channels`, `Send Messages`, `Read Message History`,
  `Embed Links`, `Attach Files`, `Add Reactions`, and — critically — **`Manage Messages`** if you
  want the bot to pin/unpin. If in doubt, `Administrator` is simplest and you can scope down later.

Copy the generated URL at the bottom, open it, pick your server, and **Authorize**.

> **Get the permissions right in the invite the first time** — a bot cannot grant itself a role
> afterward. When you invite a bot *with* a permissions bitmask, Discord auto-creates a **managed
> role** carrying those permissions. Invite with a bare `bot`-scope link and **no permissions
> selected** and it lands on `@everyone` only with `roles: []` — which means `403` on every pin and
> no member-sidebar entry. **A bot cannot grant itself a role afterward.** (This is the trap that cost
> us a day when we brought up a second bot — see §7.6.) If it lands with no role, re-invite via an
> OAuth2 URL with the correct permissions; see also [§17.2](#172-discord-gateway).

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

### 7.6 🔀 Multiple agents only — roles and permissions for a second bot

> 🔀 **Multiple agents only** — skip this if you're running a single agent on the box.

Provisioning a **second** bot on the same server hits extra role/permission edges.

**Symptom.** The second bot doesn't render as a server member and can't pin
(`403 Missing Permissions` / `MANAGE_MESSAGES`).

**Root cause.** The bot was invited **without a role**, so it landed in the server on `@everyone`
only (`roles: []`). The first bot had a `managed: true` role auto-created when it was invited with the
right OAuth scope; the second invite lacked it. No role → no `MANAGE_MESSAGES` → pin returns `403`,
and (combined with intents) it doesn't render in the member sidebar.

**Fix.** Re-invite the bot via an OAuth2 URL that grants a managed role with the needed permissions
(admin is simplest; scope it down if you prefer). This is a human step in the Discord UI — the bot
cannot grant itself a role.

**Verify — two independent checks, both against ground truth:**

```bash
# 1. The bot's gateway actually reaches READY (discord.py's on_ready).
#    The "Connected as <bot>#<discriminator>" line ONLY prints on on_ready — that IS READY.
grep -E "Connected as|discord connected" ~/.hermes/logs/gateway.log | tail
# Expect a line like:  [Discord] Connected as Deirdre#0968   then   ✓ discord connected
# (One per restart. If this line is absent, the session is NOT logged in — a real red flag.)

# 2. The role landed and the exact privileged action that 403'd now succeeds.
#    Via the discord admin tooling: member_info → expect a non-empty "roles" array;
#    then pin_message a throwaway message → expect success (not 403) → unpin to clean up.
```

> **Lesson (the durable part):** "the bot answers messages" and "the bot's gateway is fully logged
> in (READY) with the right role" are **different states**. Don't infer one from the other. Make the
> READY line and a non-empty `roles` array explicit health checks when provisioning a new bot — and
> read the *right* log line: `Connected as …` is READY; a generic `response ready` line is unrelated
> request-handling noise and does **not** prove login. (We initially misdiagnosed this by grepping
> too loosely and matching `response ready`; the fix was to grep for `Connected as` specifically.)

---

## 8. Running Hermes as a Persistent Service

Once the gateway works interactively, install it as a background service so it survives logout/reboot:

```bash
hermes gateway install
hermes gateway start
hermes gateway status
```

**Critical on a headless/SSH box:** enable linger so the user service keeps running after you disconnect (you already did this in §2.1 when you created the user; re-run it if unsure):

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
> *why* and how to confirm are in [§17.3 Memory (Mnemosyne) troubleshooting](#173-memory-mnemosyne).

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
> legacy `scope=session` rows — see [§17.3.1 Migrating a pre-existing DB](#1731-migrating-a-pre-existing-db-with-legacy-scopesession-rows). Don't
> set it reflexively.

#### The consolidation summarizer — route it to your local vLLM, and size it for a reasoning model

Mnemosyne's sleep/consolidation cycle compresses old working-memory rows into episodic summaries by
calling an LLM. Which LLM is controlled by these env vars in `~/.hermes/.env`:

```bash
# In ~/.hermes/.env — the model that writes episodic summaries during consolidation:
MNEMOSYNE_LLM_ENABLED=true
MNEMOSYNE_LLM_BASE_URL=http://localhost:8000/v1   # your local vLLM (§4) — no API key needed
MNEMOSYNE_LLM_MODEL=<whatever that endpoint serves>
```

**Why point it at the local vLLM.** Mnemosyne's **default** consolidation LLM is a CPU llama-cpp GGUF.
It's slow on a box whose GPU is already serving a good model, and a reasoning GGUF emits `<think>`
sludge the cleaner doesn't fully strip. The local vLLM you stood up in §4 is sitting right there.
The remote-API path fires when `MNEMOSYNE_LLM_BASE_URL` is set **and** `MNEMOSYNE_LLM_ENABLED` is not
false; `LLM_ENABLED` **defaults to `true`**, so in practice *just setting the base URL* switches
consolidation onto vLLM. (If you want to read the source: the call lives in
`mnemosyne/core/local_llm.py::summarize_memories()`, which is what dispatches to the vLLM vs. CPU-GGUF
path.) Measured A/B on our boxes: vLLM produced a faithful ~450-char summary in
**~16 s**; the CPU GGUF took **~21 s+** and emitted `<think>` sludge with no usable output.

> **⚠️ Footgun: `MNEMOSYNE_FORCE_LOCAL`.** Setting `MNEMOSYNE_FORCE_LOCAL=1` (or `true`/`yes`) forces
> consolidation **back** to the CPU GGUF *even when the base URL is set*. If you set it "just to test
> the local path," clear it afterward — a stray `MNEMOSYNE_FORCE_LOCAL` is a silent way to end up back
> on the slow, `<think>`-leaking path.

If the summarizer is a **reasoning model** — one that emits a `<think>…</think>` block before its
answer — the two token/timeout defaults will silently corrupt the episodic tier, so set them explicitly:

```bash
# In ~/.hermes/.env :
MNEMOSYNE_LLM_MAX_TOKENS=16384   # default is 2048 — far too small for a reasoning model
MNEMOSYNE_LLM_TIMEOUT=300        # default is 60s — a long <think> can exceed it
```

> **⚠️ `16384` is only reachable if your `MNEMOSYNE_LLM_BASE_URL` endpoint allows it.** The effective
> cap is bounded by the served model's context window and, for the local vLLM path, by vLLM's
> `--max-model-len`: it must be **≥ this value plus your prompt tokens**, or vLLM will clamp or error
> on the request. If your endpoint's context is smaller, pick the largest value it supports (and
> shrink the consolidation batch if needed) rather than a number the server can't honor.

**Why the default (`2048`) corrupts episodic memory.** After the summarizer replies, Mnemosyne runs the
output through `_clean_output()`, which strips the reasoning block with the regex
`re.sub(r"<think>.*?</think>", "", …)`. That regex needs the **closing** `</think>` tag to match. A
reasoning model routinely spends more than 2048 tokens inside `<think>`, so a `MAX_TOKENS=2048` cap
truncates the response *mid-`<think>`* — the closing tag never arrives, the strip matches nothing, and
Mnemosyne stores the **raw, unfinished reasoning fragment** as the episodic row. The tier fills with
garbage that looks like the model talking to itself instead of a clean fact summary. Raising the cap to
`16384` lets the model finish its reasoning and emit a real summary; `TIMEOUT=300` gives that longer
generation room to complete instead of erroring out at 60s.

There's a subtler second failure that points the same way: even when `<think>` *does* close, if the cap
then truncates the **actual summary after it**, `_clean_output()` strips the reasoning block cleanly and
stores a **clean-looking but silently truncated** fact — no garbage marker to catch it. So the cap must
be sized for the whole *reasoning-plus-summary* length, not just enough to close the think block.

> **⚠️ All `MNEMOSYNE_*` vars are read at import time**, not per-call (in `mnemosyne/core/local_llm.py`,
> `LLM_MAX_TOKENS` / `LLM_TIMEOUT` / `LLM_BASE_URL` / `LLM_ENABLED` are module-level constants read once
> at import). Editing `~/.hermes/.env` does **not** update a *running* gateway — restart it so the new
> values load, and verify with the clean-env probe in
> [§17.3.4 Where the `MNEMOSYNE_*` env vars go](#1734-where-the-mnemosyne_-env-vars-go-and-how-to-verify-them):
>
> ```bash
> systemctl --user restart hermes-gateway   # same restart rule as §9.2
> ```

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
> readers; the fix is in [§17.3.2 Seeding facts from the shell](#1732-seeding-facts-from-the-shell--mnemosyne-store-defaults-to-scopesession).

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
went into the wrong venv (§9.2), no restart after install (§9.2), or a scope/migration issue — the
full break-by-break analysis with a proof step for each is in
[§17.3 Memory (Mnemosyne) troubleshooting](#173-memory-mnemosyne).

### 9.5 Auto-sleep fires only every 10th turn — the gate that looks like a broken install

A freshly-restarted, *correctly-wired* gateway shows `episodic: 0` even though `working > 50`. It looks
like consolidation is broken. It isn't — consolidation is built in (no cron needed), but the auto-sleep
trigger is gated on a **per-session turn counter** that lives in the Mnemosyne↔Hermes bridge:

```python
# mnemosyne_hermes/__init__.py — the gate (symbol-anchored; line numbers drift between releases):
self._turn_count = 0                                  # in __init__ — resets every restart
...
self._turn_count += 1                                 # every turn
if self._auto_sleep_enabled and self._turn_count % 10 == 0:
    self._maybe_auto_sleep()                          # threshold only CHECKED on turns 10, 20, 30…
```

So:

- Auto-sleep **only even checks** the threshold on every **10th turn** of a live session.
- `turn_count` is **per-session** and **resets to 0 on every gateway restart**.
- Therefore a freshly-restarted gateway with `< 10` turns will show `episodic: 0` even with
  `working > 50`. **That is the turn gate, not a wiring bug.**

There's a second, quieter no-op: even on a turn that *is* a multiple of 10 with `working > threshold`,
`_maybe_auto_sleep()` returns without logging if the eligibility check finds **zero** rows old enough
(cutoff = `now − TTL/2`) — i.e. everything old is already consolidated.

**Confirm it's the gate, not a config break.** Grep the gateway journal for the auto-sleep log line —
its presence proves the trigger fired; its absence means `< 10` turns since restart or nothing
eligible, not a broken config:

```bash
journalctl --user -u hermes-gateway | grep "Mnemosyne auto-sleep:"
# A hit looks like:  Mnemosyne auto-sleep: working=63, eligible=41 > threshold=50
```

> **Defaults & knobs (verified — grep, don't trust a line number):** the threshold is `working > 50`
> (config key `sleep_threshold`); auto-sleep is on by default (config key `auto_sleep` / legacy
> `MNEMOSYNE_AUTO_SLEEP_ENABLED`). When it does fire it runs `sleep_all_sessions` (cross-session) in a
> daemon thread.

**To consolidate a backlog right now** (bypass the turn gate), run a fresh subprocess with the env vars
set — a fresh process gets a fresh reflect budget and doesn't need the live turn counter:

```bash
# From the agent's Hermes venv, with the MNEMOSYNE_LLM_* vars exported:
python - <<'PY'
from mnemosyne.core import memory as M
M.sleep_all_sessions(dry_run=False, force=True)   # force=True ignores the age cutoff
PY
```

> `force=True` consolidates all unconsolidated working rows regardless of age; `sleep_all_sessions`
> (vs `sleep`) walks **every** session, not just the current one. Over a dozen sessions with a
> 30B-class reasoning model this can exceed a few minutes; run it in the background rather than a
> short foreground timeout.

**Verify the remote path actually ran** — don't infer it from timing:

```bash
# During a consolidation run, confirm an ESTABLISHED socket to the vLLM port:
ss -tnp | grep :8000
# And spot-check a fresh episodic summary is clean prose, NOT <think>...:
sqlite3 ~/.hermes/mnemosyne/data/mnemosyne.db \
  "SELECT substr(content,1,120) FROM episodic_memory ORDER BY rowid DESC LIMIT 1;"
```

> An `hf_hub_download` / "unauthenticated HF Hub" warning during the run is **noise**, not proof the
> local GGUF was used — Mnemosyne lazily probes the local tokenizer to size chunks even when the
> remote path handles the actual summary. Confirm the remote path by the live socket to vLLM and clean
> summary text, not by the absence of that warning.

---

## 10. Web Search & Extraction

Give the agent eyes on the internet **without a paid search API key** (no Tavily/Exa/Brave/Parallel).
Hermes splits web capability into two independent axes, each with its own backend selector, and picks a
backend by *availability*, not by a key you bought:

- `web.backend` — the shared default for both capabilities.
- `web.search_backend` — override for `web_search` only. **If empty, it falls back to `web.backend`.**
- `web.extract_backend` — override for `web_extract` only. Same fallback rule.

The fallback is literal (`tools/web_tools.py`, `_get_capability_backend()`):

```python
specific = (cfg.get(f"{capability}_backend") or "").lower().strip()
if specific and _is_backend_available(specific):
    return specific
return _get_backend()          # falls back to web.backend
```

**The sensible, zero-paid-key split** (this is what both our agents run):

```bash
hermes config set web.backend ddgs                 # search via DuckDuckGo
hermes config set web.extract_backend firecrawl    # extract via self-hosted Firecrawl
hermes config set web.search_backend ""            # empty → inherits web.backend (ddgs)
```

```yaml
# ~/.hermes/config.yaml
web:
  backend: ddgs
  search_backend: ''        # empty on purpose — inherits web.backend
  extract_backend: firecrawl
```

### 10.1 `ddgs` is a PYTHON PACKAGE, not an API

There is **no key and no endpoint** for DuckDuckGo search. `ddgs` is a pip package that scrapes DDG;
Hermes treats the backend as available **iff `import ddgs` succeeds** — nothing else. (It's the *only*
backend whose availability is a package-import probe rather than an env-var/key check.) The entire
"setup" is:

```bash
# inside the agent's Hermes venv
pip install ddgs
python -c "import ddgs; print('ddgs', ddgs.__version__)"   # prove the import works
```

> Do **not** mistake DuckDuckGo for a "free API endpoint" — there isn't one. If `import ddgs` fails,
> Hermes silently treats the backend as unavailable and search falls through to whatever else is
> configured (often nothing).

### 10.2 Firecrawl extract points at a self-hosted instance, no key

`web_extract` uses Firecrawl pointed at a **local** Firecrawl you host yourself — free, private, no
rate-limited SaaS key. Firecrawl is available when **either** `FIRECRAWL_API_KEY` **or**
`FIRECRAWL_API_URL` is set, so the URL alone is enough:

```bash
# Base ORIGIN only — no /v1 suffix, no key.
echo 'FIRECRAWL_API_URL=http://localhost:3002' >> ~/.hermes/.env
```

> **Two footguns:** (1) use the bare origin `http://localhost:3002` — **not** `.../v1`; Hermes appends
> the path itself, and a `/v1` suffix double-paths the request. (2) `FIRECRAWL_API_URL` is a genuine
> credential-adjacent endpoint, so it lives in `~/.hermes/.env`, not `config.yaml`.

### 10.3 Hosting the local Firecrawl container stack

> 🔀 **On a multi-agent box, host it once.** Only **one** agent runs the Firecrawl stack; every other
> agent on the box just points `FIRECRAWL_API_URL` at the existing one — **no clone, no second stack.**
> On our box the stack runs once under the first agent's user
> (`/home/videau-ai/services/firecrawl/`, port `:3002`) and the second agent consumes it purely as a
> **client**. Running the recipe below on a second agent would spin up a redundant 5-container stack or
> collide on `:3002`. The steps below are for the **one** agent that hosts it (or a single-agent box).

Self-hosted Firecrawl is a small **multi-container** stack (not a single image): the API plus Redis,
RabbitMQ, a Postgres (`nuq-postgres`), and a Playwright browser service. On ARM64 (DGX Spark / GB10)
**don't build from source** — use the published `arm64` images via a compose override.

```bash
# 1. Clone the repo somewhere stable (we keep services under ~/services)
mkdir -p ~/services && cd ~/services
git clone https://github.com/firecrawl/firecrawl.git
cd firecrawl

# 2. Env: copy the example and keep it minimal for self-host
cp apps/api/.env.example .env    # self-host defaults are fine; USE_DB_AUTHENTICATION=false
```

Add an override so Docker pulls prebuilt arm64 images instead of compiling (this is exactly the file
we run on `piment`):

```yaml
# ~/services/firecrawl/docker-compose.override.yaml
# Override: use published arm64 images instead of building from source.
name: firecrawl
services:
  api:
    build: !reset null
    image: ghcr.io/firecrawl/firecrawl:latest
  playwright-service:
    build: !reset null
    image: ghcr.io/firecrawl/playwright-service:latest
  nuq-postgres:
    build: !reset null
    image: ghcr.io/firecrawl/nuq-postgres:latest
```

```bash
# 3. Bring the stack up detached; it publishes the API on :3002
docker compose up -d

# 4. Confirm all five containers are Up
docker compose ps
# Expect: firecrawl-api-1, firecrawl-redis-1, firecrawl-rabbitmq-1,
#         firecrawl-nuq-postgres-1, firecrawl-playwright-service-1  — all "Up"
```

> The API container binds `0.0.0.0:3002->3002` via `docker-proxy`, which is why
> `FIRECRAWL_API_URL=http://localhost:3002` works from the host even though the agent runs outside the
> compose network. If you firewall the box, `:3002` should stay host-local — it's an unauthenticated
> endpoint by design in the self-host config.

**Verify — the container answers before you blame Hermes:**

```bash
curl -s http://localhost:3002/ ; echo
# Expect: {"message":"Firecrawl API","documentation_url":"https://docs.firecrawl.dev"}
```

> **Scope caveat (honesty note).** The *running* stack, the override file, and the `:3002` health check
> above are verified against ground truth on `piment`. The from-scratch cold-start — `git clone` +
> `cp apps/api/.env.example .env` + first `docker compose up -d` on a genuinely **fresh** box — is
> **not** independently clean-room tested here; both our agents verified against an already-running
> stack. Treat the clone/env steps as the documented-but-unproven path and expect to read Firecrawl's
> own self-host docs if a fresh bring-up hiccups.

### 10.4 Prove BOTH capabilities end-to-end

A configured backend is a hypothesis until a live query returns real content:

```bash
# Search (ddgs): must return live, real URLs — not an "no backend available" error.
hermes chat -q "Use web_search for 'NVIDIA DGX Spark GB10 specifications' and list 3 result URLs."

# Extract (self-hosted firecrawl): must return page text with no error.
hermes chat -q "Use web_extract on https://example.com and quote the first sentence."
```

Both must come back with genuine content. `ddgs` (package, no key) + self-hosted Firecrawl
(`FIRECRAWL_API_URL`, no key) gives the agent full web eyes for **$0** and keeps every fetch private to
the box.

---

## 11. Google Integration (Gmail, Calendar, Drive, Docs)

Hermes talks to Google Workspace through the bundled **`google-workspace`** skill, which manages
OAuth for you. There are **two paths** — pick by what you actually need, because they have very
different setup costs.

### 11.1 Decide: App Password (email only) vs. OAuth (full Workspace)

- **Just email?** Skip Google Cloud entirely. Use the **`himalaya`** skill with a Gmail **App
  Password** (Google Account → **Security → 2-Step Verification → App passwords**). Two minutes, no
  cloud project. Load the skill and follow its setup.
- **Calendar / Drive / Sheets / Docs (or email + those)?** Use the `google-workspace` skill with
  OAuth, below.

> **Per-agent credentials.** Each agent gets its **own** Google account and token — never reuse one
> agent's Google token for another (see [§14.1 Account & Identity Isolation](#141-account--identity-isolation)).

### 11.2 One-time: create an OAuth client in Google Cloud (~5 min)

1. Create/select a project: <https://console.cloud.google.com/projectselector2/home/dashboard>
2. Enable the APIs you need from the **API Library**
   (<https://console.cloud.google.com/apis/library>): Gmail, Calendar, Drive, Sheets, Docs, People.
3. **Credentials → Create Credentials → OAuth 2.0 Client ID → Application type: _Desktop app_ →
   Create.** (Desktop-app type is what the skill's PKCE flow expects.)
4. If the OAuth app is still in **Testing**, add your Google account as a **test user** at
   <https://console.cloud.google.com/auth/audience> — otherwise you'll get `Error 403: access_denied`.
5. **Download the client-secret JSON** and note its path.

### 11.3 Authorize (works fully headless — no browser on the box)

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

### 11.4 Use it

```bash
GAPI="python ~/.hermes/skills/productivity/google-workspace/scripts/google_api.py"
$GAPI gmail search "is:unread" --max 10
$GAPI calendar list
$GAPI drive upload /path/to/report.pdf
```

> **Off-box backups to Drive** use a *different* mechanism — `rclone` with least-privilege
> `drive.file` scope and its own headless-authorize flow — covered in [§12 Backups](#12-backups--an-untested-backup-is-not-a-backup).
> Don't conflate the `google-workspace` OAuth token (LLM/skill actions) with the rclone remote (bulk
> file sync); they are separate credentials with separate scopes on purpose.

---

## 12. Backups — an untested backup is not a backup

**Symptom.** "We have backups" — but there was no cron, and the off-box push silently did nothing
(`rclone remote 'gdrive' not configured`, rc=2). A local script that runs once by hand is not a
backup strategy.

**Root cause.** Three gaps: (1) no schedule, (2) no off-box copy, (3) never restore-tested.

**Fix — the full pipeline.**

1. **Encrypt locally.** `tar | zstd | gpg --symmetric` (AES-256), passphrase from a `chmod 600`
   file. Capture SQLite DBs with the online `.backup` API, not `cp`, so they're consistent.
2. **Push off-box** to the agent's **own** cloud account (see [§14.1](#141-account--identity-isolation) —
   on a multi-agent box, do **not** reuse another agent's OAuth token). Use least-privilege scope
   (`drive.file` for Google Drive — the app only ever sees files it created).
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
rclone copy gdrive:<agent>-backups/<newest>.tar.zst.gpg /tmp/restore/
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

## 13. Skills & Cron

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

## 14. Running More Than One Agent on the Same Box

> 🔀 **Multiple agents only** — the entire section is for the case where you run **two or more** agents
> on one box. A single agent needs none of it. Everything before this point already produced a complete,
> working agent; you provisioned it as its own Linux user (§2.1), so adding another is mostly "repeat
> §2–§13 as a new user," plus the cross-agent concerns below: keeping their identities isolated,
> and (optionally) letting them share a slice of memory and a folder of skills.

Provisioning a second agent is the **same tutorial again** under a new Linux user. Create the user
exactly as in [§2.1](#21-each-agent-is-a-dedicated-linux-user), then walk §3–§13 as that user. The
only genuinely new material is isolation (§14.1), shared memory (§14.2), and shared skills (§14.3).

### 14.1 Account & Identity Isolation

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

### 14.2 Shared Memory — a cross-agent surface DB two uids can both write

Two agents on one box eventually want a **shared** slice of memory: a place where agent A writes a
fact and agent B can recall it. Mnemosyne has a built-in channel for exactly this — the **surface
bank** (`mnemosyne_shared_remember` / `mnemosyne_shared_recall`) — and you can point both agents'
surface banks at **one SQLite file** owned by a shared group. It works, but four separate traps sit
between "seems configured" and "actually works." We hit all four; this is the verified recipe.

> **Scope, first.** The surface bank is for *compact cross-agent metadata* — stable facts,
> preferences, provisioning summaries. It is **not** full memory sharing: each agent's private bank
> (`mnemosyne_remember`) stays per-agent. Don't set this up expecting agent B to see agent A's whole
> memory; that's not what the surface bank is.

#### 14.2.1 The group + directory (setgid is mandatory)

```bash
# As admin. One group both agent users belong to:
sudo groupadd agent-shared                       # gid lands at e.g. 1003
sudo usermod -aG agent-shared videau-ai          # agent A (uid 1001)
sudo usermod -aG agent-shared deirdre-ai         # agent B (uid 1002)

# A directory the group owns, with the SETGID bit so new files inherit the group:
sudo install -d -m 2775 -g agent-shared /var/lib/agent-shared
# Verify: the 's' in drwxrwsr-x is the setgid bit — without it, files land in the
# creator's PRIMARY group and the other agent is locked out.
stat -c '%A %U:%G' /var/lib/agent-shared     # => drwxrwsr-x root:agent-shared
```

#### 14.2.2 The trap that wastes an hour: `usermod` needs a *user-manager* cycle, not a service restart

After `usermod -aG`, the new group is **not** live in the running agent. The obvious fix — restart the
gateway service — **does not work**:

```bash
# WRONG — this does NOT pick up the new group:
systemctl --user restart hermes-gateway.service
# Check the PROCESS, not `id` (id reads the group DB and lies about the running process):
grep '^Groups:' /proc/<gateway-pid>/status     # still MISSING 1003
```

Why: the **`systemd --user` manager** caches its supplementary groups at spawn and passes *its* set to
every child it launches. It predates your `usermod`, so it hands the stale set to the freshly-restarted
gateway. Even `daemon-reexec` preserves the cached set. The only fix is to cycle the **user manager
itself**:

```bash
# As admin — kills the user's whole session; linger respawns the manager from PID 1
# with freshly-computed credentials, and the gateway auto-starts (if the unit is enabled):
sudo loginctl terminate-user deirdre-ai
# (or: sudo systemctl restart user@1002.service)

# Caveat we hit: terminate can briefly clear the lingering state, so the manager may not
# auto-respawn in the first ~10-15s. If it doesn't come back, nudge it:
sudo loginctl enable-linger deirdre-ai
sudo systemctl start user@1002.service

# VERIFY at the process level (not `id`), and functionally:
grep '^Groups:' /proc/<new-gateway-pid>/status         # => now includes 1003
sudo -u deirdre-ai touch /var/lib/agent-shared/.probe   # => succeeds; file is group agent-shared
```

> **An agent cannot do this to itself** — terminating its own session kills the very process issuing
> the command, and the in-process safety guard blocks it (even a `sudo -u` targeting a different user
> trips the guard, which pattern-matches the restart string). This is where two agents earn their
> keep: **agent A (a separate uid/process) runs the terminate for agent B**, and vice-versa. If you
> have no second agent yet and only an admin shell, run it there. Treat the guard as a real safety
> boundary, not an obstacle to game.

#### 14.2.3 The trap that looks like a hard blocker but isn't: SQLite's creation-mode 0644

Point both agents' config at the shared file:

```bash
# NEVER hand-edit config.yaml — `hermes config set` authors the nested path correctly:
hermes config set memory.mnemosyne.shared_surface_path /var/lib/agent-shared/mnemosyne.db
hermes config set memory.mnemosyne.shared_surface_read true
```

> `hermes config set` prints a **"not a recognized config key"** warning for these two — it's a
> **false alarm**. The Mnemosyne plugin reads `memory.mnemosyne.<key>` directly; the warning is just
> the Hermes core schema not knowing the plugin's keys. The key is live at provider access.

Now the trap. If you let the two agents create the DB by racing, **whichever uid opens it first owns
the file at mode `0644`** — because **SQLite forces new database files to `0644`, ignoring your umask
entirely.** The other uid then gets `attempt to write a readonly database`. It is **not** a
fundamental "two uids can't share one SQLite DB" limit — it's purely a creation-order + file-mode
artifact:

```bash
# umask 0002 does NOT help — SQLite ignores it:
umask 0002; python3 -c "import sqlite3,os,stat; sqlite3.connect('/tmp/t.db').execute('create table x(a)'); print(oct(stat.S_IMODE(os.stat('/tmp/t.db').st_mode)))"
# => 0o644   (not 0o664)
# POSIX default ACLs also DON'T save you — SQLite sets an explicit group mask of r--.
```

**The fix is a one-time pre-create at 0660** — durable, because SQLite only forces the mode at *initial
creation* and never resets an existing file's permissions on reopen:

```bash
# Pre-create the shared DB (as either agent), then set it group-writable ONCE:
sudo -u deirdre-ai /path/to/hermes/venv/bin/python -c "import sqlite3; c=sqlite3.connect('/var/lib/agent-shared/mnemosyne.db'); c.execute('PRAGMA journal_mode=WAL'); c.close()"
sudo chgrp agent-shared /var/lib/agent-shared/mnemosyne.db
sudo chmod 0660         /var/lib/agent-shared/mnemosyne.db
stat -c '%A %U:%G' /var/lib/agent-shared/mnemosyne.db   # => -rw-rw---- <uid>:agent-shared
# Reopen-and-write does NOT reset it back to 0644 — verified: perms survive reopen.
```

#### 14.2.4 The WAL sidecars — why both gateways must run umask 0002

WAL mode creates `-wal` and `-shm` sidecar files next to the DB. Two facts matter: they're created on
open and checkpoint-deleted on the last connection close (so you can't just chmod them once — they're a
moving target), and whichever process creates a sidecar owns its mode. If that lands `0644`, the other
uid is momentarily locked out even though the main DB is `0660`.

The clean, no-daemon fix is to ensure **both gateways run `umask 0002`** (they do by default under the
systemd user session). Under 0002, the sidecars each process creates land `0660` group-writable, so
cross-uid concurrent writing works.

```bash
# Confirm each gateway's umask at the process level:
grep '^Umask:' /proc/<gateway-pid>/status      # => 0002
```

A concurrent two-writer smoke test (WAL + `busy_timeout`) should report **zero readonly errors and zero
lock waits** — `busy_timeout` handles same-uid lock *contention*; the 0660 + umask 0002 combination
handles the cross-uid *permission* wall. They are different problems; you need both.

#### 14.2.5 Prove the channel end-to-end — through the live gateway, not just the file

The real acceptance test is a round-trip through each agent's **live gateway provider**, not a
standalone script hitting the file:

```
# Agent A writes a marker via its tool:
#   mnemosyne_shared_remember("PROVISIONING-PROOF-<A> ...")
# Agent B recalls it via ITS tool and confirms bank:surface:
#   mnemosyne_shared_recall("provisioning proof")  ->  result tagged  bank: surface
```

To be sure the recall genuinely ran *through the gateway* (not a direct DB read), confirm the tool
process is a child of the gateway: `echo ppid=$PPID` should equal the gateway MainPID. Do it **both
directions** — A→B and B→A — before declaring the channel proven. In our bring-up the file layer passed
well before the live-provider layer did (a stale provider had the pre-config path cached), so the file
test alone would have been a false green.

| Check | Proves |
|---|---|
| `/proc/<gw-pid>/status` Groups includes the shared gid | The running gateway (not just the shell) holds the group |
| `sudo -u <other> touch` in the dir succeeds | Setgid + membership are functionally live, not just in the group DB |
| Shared DB is `-rw-rw---- :agent-shared` and survives reopen | The 0644 creation-mode trap is neutralized durably |
| Live `-wal`/`-shm` are `0660 agent-shared` under concurrent write | umask 0002 keeps sidecars group-writable; no cross-uid lockout |
| `mnemosyne_shared_recall` returns the other agent's row `bank:surface`, both directions, tool `ppid == gateway pid` | The channel works through both **live providers**, not just the file |

---

### 14.3 Shared Skills — one git-backed folder both agents load from

The surface DB (§14.2) shares *facts*. The next thing two agents want to share is *procedures* —
**skills**. Hermes loads skills from `~/.hermes/skills/` **plus** any directory listed in
`skills.external_dirs`, so you can point both agents at **one shared, git-backed skills folder**
and have a skill authored by agent A load in agent B. The plumbing reuses the exact `agent-shared`
group + setgid directory from [§14.2.1](#1421-the-group--directory-setgid-is-mandatory), so if you did
§14.2 the hard part is already done. What's new here is a different failure surface: **provenance,
load-path precedence, and file modes** — and every one of them will happily let you *think* it worked
while it's quietly broken. This is the verified recipe.

> **Move, not copy.** The rule that keeps this sane: when a skill goes into the shared folder, it is
> **removed** from the author's local `~/.hermes/skills/` in the same breath. A local copy left behind
> **shadows** the shared one (local dir wins name resolution), so the two agents silently diverge — you
> edit the shared copy and the author keeps loading their stale local. Share = relocate, not duplicate.

#### 14.3.1 The directory + config wiring

Reuse the §14.2.1 group and setgid dir; a git repo inside it gives you history and rollback:

```bash
# As admin (or either agent, if the parent dir is already group-writable from §14.2.1):
sudo install -d -m 2775 -g agent-shared /var/lib/agent-shared/skills
cd /var/lib/agent-shared/skills && git init -q          # history + rollback for shared edits

# Cross-uid git safety: the repo is owned by one uid but read/written by both. Without this,
# git refuses to operate for the non-owner ("detected dubious ownership"). Run as EACH agent:
git config --global --add safe.directory /var/lib/agent-shared/skills
```

Then point each agent's config at it. `skills.external_dirs` is a **YAML list**, and getting a list
value in correctly is **the config trap that eats the most time:**

```bash
# Read the current value first — it may already exist and you want to APPEND, not overwrite:
hermes config get skills.external_dirs
```

The value must land as a genuine YAML **block-sequence**. The reliable, verified way to add a list
item is a **round-trip YAML editor** (`ruamel.yaml`, which preserves formatting and comments) driven
through `hermes config edit` — *not* a bare `hermes config set`. Two mechanism facts matter, and both
bite if you skip them:

- `hermes config edit` takes **no flags** — there is no `--editor`. It opens the config in whatever
  `$EDITOR` points to, invoking it with the **config file path as `$1`**.
- Hermes `exec`s `$EDITOR` as a **single `argv[0]`** — it does *not* shell-split it. So a multi-word
  value like `EDITOR="python3 add-dir.py"` fails with `FileNotFoundError: 'python3 add-dir.py'`.
  `$EDITOR` must be **one executable**. The clean way is a tiny wrapper script:

```python
# scripts/add-external-skills-dir.py  — the round-trip edit (reads the config path as argv[1])
import sys
from ruamel.yaml import YAML
yaml = YAML()                      # round-trip mode: preserves comments + formatting
path = sys.argv[1]
NEW = "/var/lib/agent-shared/skills"
with open(path) as f:
    cfg = yaml.load(f)
skills = cfg.setdefault("skills", {})
dirs = skills.get("external_dirs")
if not isinstance(dirs, list):     # normalize a missing/scalar value into a real list
    dirs = [] if dirs is None else [dirs]
    skills["external_dirs"] = dirs
if NEW not in dirs:
    dirs.append(NEW)
with open(path, "w") as f:
    yaml.dump(cfg, f)
```

```bash
# scripts/edit-config-wrapper.sh  — a single executable for $EDITOR (no shell-split needed)
cat > scripts/edit-config-wrapper.sh <<'SH'
#!/usr/bin/env bash
exec python3 "$(dirname "$0")/add-external-skills-dir.py" "$1"
SH
chmod +x scripts/edit-config-wrapper.sh

# Drive the round-trip edit through hermes config edit (NOT hermes config set):
EDITOR="$PWD/scripts/edit-config-wrapper.sh" hermes config edit

# Verify it's a LIST with a bare-string item (this read-back IS the acceptance test):
hermes config get skills.external_dirs        # => - /var/lib/agent-shared/skills
```

> **Why not just `hermes config set`?** Every `set` form fails to author a real list here (all
> sandbox-verified):
> - `hermes config set skills.external_dirs.0 <path>` — the `.0` index creates a **dict** `{'0': <path>}`, which the loader silently ignores.
> - `hermes config set skills.external_dirs '["/var/lib/agent-shared/skills"]'` — stores the literal **string** `["/var/lib/agent-shared/skills"]`, not a list.
> - `hermes config set skills.external_dirs <path>` — stores a bare **scalar string**, not a one-element list.
>
> There is no `--append` flag and no `set` form that produces a YAML block-sequence — the round-trip
> editor above is the only path. **Never trust that an edit "took" without reading it back** with
> `hermes config get` and confirming the `- ` list marker.

#### 14.3.2 GATE 1 — provenance: never share a Hermes/plugin-bundled skill

Before promoting a skill, prove it is **agent-authored**, not something Hermes or a plugin ships. If
you share a bundled skill and then (per the move-not-copy rule) delete the local copy, on a box where
the bundle *isn't* present you've just deleted the only copy — and on a box where it *is*, you now have
a shared fork that drifts from the upstream bundle. We hit the live version of this: `mnemosyne-memory-override`
looked like a normal local skill but is **plugin-shipped** (installed by the `mnemosyne_hermes` package),
and deleting its local copy would have removed a guardrail with **no fallback to load**. Check five
sources, and include a **positive control** so the check can't silently pass everything:

```bash
VENV=~/.hermes/hermes-agent/venv           # adjust to your tree
NAME=<skill-to-vet>

# 1. Bundled manifest — the authoritative sha-pinned list of core-shipped skills.
#    Format is  name:sha  (colon-delimited, NOT space) — match on the colon:
grep -q "^$NAME:" ~/.hermes/skills/.bundled_manifest && echo "BUNDLED (core) — do NOT share" || echo "not in core manifest"
# 2. Plugin/venv package data — skills shipped inside an installed pip package:
find "$VENV"/lib/python*/site-packages -path '*/skills/*/SKILL.md' 2>/dev/null | grep -i "$NAME" && echo "PLUGIN-shipped — do NOT share" || echo "not plugin-shipped"
# 3. Hermes core repo tree + 4. optional-skills/ tree (if you have a checkout):
#    grep the skill name under hermes-agent/skills/ and hermes-agent/optional-skills/
# 5. Git first-author (in the shared repo, once promoted) should be an AGENT identity, not "Hermes Agent".

# POSITIVE CONTROL: the check must actually FIRE on something you KNOW is bundled — otherwise a
# check that returns "clean" for everything (e.g. wrong manifest path OR wrong delimiter) would pass
# silently. Pick a name that IS in your manifest and confirm the grep hits it. If it does NOT print
# "control OK", your check is broken — fix it before trusting any "clean" verdict:
CONTROL=$(head -1 ~/.hermes/skills/.bundled_manifest | cut -d: -f1)   # a known-bundled name
grep -q "^$CONTROL:" ~/.hermes/skills/.bundled_manifest && echo "control OK — check fires on bundled skills" || echo "CONTROL FAILED — the check itself is broken, do not trust its output"
```

Only a skill that is **absent from all bundled sources and first-authored by an agent** is eligible
to share. Plugin-owned guardrails like `mnemosyne-memory-override` **stay local on every box.**

#### 14.3.3 GATE 2 — load-path probe: prove the shared copy loads *before* deleting the local

"Identical content" **never** proves "safe to delete the local." Local precedence means your loader
could be serving the local copy while the shared one is broken (wrong path, bad perms, a name
collision — see 14.3.5), and you'd only find out *after* deleting the local, when there's nothing to
fall back to. The only proof is a **reversible load-path probe**:

```bash
# 1. Inject a unique marker into the SHARED copy so a shared-load is unmistakable. Put it on the
#    COMMITTED shared file (so step 4's `git checkout --` can cleanly revert it):
echo "<!-- PROV-<agent>-$(date +%s) -->" >> /var/lib/agent-shared/skills/$NAME/SKILL.md

# 2. Move (don't delete) the local aside so precedence can't mask the shared copy.
#    `**` needs globstar (OFF by default in a non-interactive shell) — enable it or give the exact path:
shopt -s globstar
mv ~/.hermes/skills/**/$NAME ~/.hermes/_probe_stash/$NAME    # or the exact ~/.hermes/skills/<cat>/$NAME path

# 3. Load via the agent tool and confirm BOTH: it resolves to the shared dir AND returns the marker:
#      skill_view($NAME)  ->  skill_dir == /var/lib/agent-shared/skills/$NAME   AND   marker present
# 4. Restore the local, revert the marker on the shared copy (git -C /var/lib/agent-shared/skills
#    checkout -- $NAME/SKILL.md), THEN — and only then — delete the local for real. Re-verify with
#    NOTHING stashed: this is the real end-state.
```

Do the probe on **both** agents' boxes, each against its own loader — a shared copy readable by one
uid can still be unreadable by the other (see 14.3.4).

#### 14.3.4 GATE 3 — file modes: tool-authored files land `0600`, and no passive trick fixes it

The nastiest surprise. When an agent authors a file through its **tool layer** (the gateway process),
the file lands `-rw-------` (`0600`) — because the gateway runs with a restrictive umask (`0077`),
**not** your interactive shell's `0002`. So a skill agent A writes into the shared folder via its
tools is **unreadable by agent B**, and B's loader silently can't see it. We chased the "elegant"
durable fixes and **none of them work on this setup:**

```bash
# A shell `umask 0002` does NOT help — the GATEWAY writes the file, not your shell.
# A POSIX default ACL does NOT help either — the file's 0600 mode clamps the ACL mask to ---:
#   setfacl -d -m group:agent-shared:rw /var/lib/agent-shared/skills   # looks set...
#   getfacl <file>  ->  group:agent-shared:rw-   #effective:---         # ...but neutered by the mask
# (And a repo-wide ACL is structurally impossible anyway: ownership spans two uids, so each
#  agent can only setfacl files it owns.)
```

The reliable fix is an **explicit `chmod` at hand-off**, verified from the *other* agent's uid — not a
mount trick that silently doesn't fire:

```bash
# After authoring/promoting, the AUTHOR fixes the mode on exactly the files it owns:
chmod 0664 /var/lib/agent-shared/skills/$NAME/SKILL.md
find /var/lib/agent-shared/skills/$NAME -type f -exec chmod 0664 {} +
# PROVE it from the OTHER uid (this is the real test — the author's own read tells you nothing):
sudo -u <other-agent-uid> test -r /var/lib/agent-shared/skills/$NAME/SKILL.md && echo "readable by other agent" || echo "STILL BLOCKED"
# Sweep for any group-unreadable file left in the repo, so it's closed, not just patched for one file:
find /var/lib/agent-shared/skills -type f ! -perm -g=r
```

#### 14.3.5 The staging-teardown trap — leftover merge dirs get scanned as *live skills*

When two agents both authored a same-named skill, you'll stage copies (e.g. under
`_merge-staging/<agent>/`) to diff and merge them into one shared version. **Remove that staging tree
before you delete any local copy.** Because the staging dirs live *inside* an `external_dirs` folder,
the loader scans them as **live skills** — so after a merge you have three copies of the name (the
promoted root copy plus the two staging copies), and the loader throws an **ambiguous-name collision**
that makes the skill *unloadable*. It stays invisible while a local copy still exists (local precedence
masks it) and only detonates the moment you delete the local — exactly when you have the least fallback.
We hit this live; the 14.3.3 mv-aside probe is what surfaced it before any real delete.

```bash
# Correct order:
#   1. promote the merged skill to the repo ROOT
#   2. git rm -r _merge-staging        # tear down staging FIRST (content is preserved in git history)
#   3. mv-aside + GATE 2 probe the ROOT copy on BOTH boxes
#   4. only now delete the local copies
```

#### 14.3.6 Dual-writer hygiene — `git status` before you touch a shared tree

Both agents can commit to the same repo. Before any tree-wide operation (a `chmod -R`, a bulk edit, a
`git add -A`), run `git status` first — the *other* agent may have staged or in-flight work you'd
clobber or accidentally commit. We had a near-miss where one agent's `chmod` pass touched files the
other was actively probing. No harm done, but the discipline is cheap: **look before you write on
shared state.**

#### 14.3.7 What "done" actually means here

| Check | Proves |
|---|---|
| `hermes config get skills.external_dirs` shows a `- ` list item, not a dict/string | The `external_dirs` wiring took (not the `.0`-index or JSON-string trap) |
| Candidate skill is absent from `.bundled_manifest`, venv package data, core + optional trees; git first-author is an agent | GATE 1 — it's genuinely agent-authored, safe to share (not a plugin guardrail like `memory-override`) |
| `skill_view` resolves to `skill_dir=/var/lib/agent-shared/...` with the local moved aside, returning the injected marker | GATE 2 — the shared copy actually loads; safe to delete the local |
| `sudo -u <other-uid> test -r` succeeds on every shared file; `find ... ! -perm -g=r` is empty | GATE 3 — the tool-authored `0600` mode is fixed for the other agent, repo-wide |
| No `_merge-staging` (or other nested skill dirs) left under the shared folder | The ambiguous-name collision trap is neutralized before any local delete |
| Final no-stash `skill_view` on both boxes resolves to the shared dir with no local shadow | The real steady state: one shared source of truth, both agents loading it |

---

## 15. Verifying a Healthy Agent

Before you call an agent "done," each axis below must be **proven**, not assumed. Every failure in this
tutorial was invisible to a casual "does it answer?" check and only surfaced under a test that tried to
make it fail. Configure with that adversarial mindset and an agent comes up clean the first time.

### 15.1 The checklist

- [ ] **Model.** Live agent reports the intended frontier model — not the local fallback.
      `custom_providers:` block present; `api_key` real. (`hermes chat -q "state your model"` — §5.1)
- [ ] **Fallback.** Real failover observed against a dead primary in an isolated home (saw the
      "🔄 Switched to fallback model" line — §5.2).
- [ ] **Memory.** `hermes memory status` → `Plugin: installed ✓`; canary fact recalled; durable facts
      `scope=global` (§9.3–§9.4).
- [ ] **Consolidation.** `MNEMOSYNE_LLM_BASE_URL` points at the local vLLM in `~/.hermes/.env`; a fresh
      episodic summary is clean prose (no `<think>` leak); the auto-sleep journal line appears after ≥10
      turns, or a forced `sleep_all_sessions(force=True)` backfill produced episodic rows (§9.5).
- [ ] **Web.** `import ddgs` succeeds in the venv; `FIRECRAWL_API_URL` (bare origin, no `/v1`) set and
      the local Firecrawl stack answers on `:3002`; live `web_search` returns real URLs and
      `web_extract` returns page text — both with **no paid key** (§10).
- [ ] **Backup.** Encrypted archive pushed off-box on a cron; **restored from the cloud copy** and
      `PRAGMA integrity_check` = `ok`; passphrase pinned off-box; negative test fails loud (§12).
- [ ] **Discord.** Bot has a managed role (`member_info` shows a non-empty `roles`); the exact
      privileged action that failed now succeeds (pin/unpin); gateway log shows a `Connected as …`
      READY line, not a generic `response ready` line (§7.5–§7.6).
- [ ] 🔀 **(Multi-agent)** Identities isolated — distinct backup passphrases and cloud remotes (§14.1);
      if sharing memory, the surface-bank round-trip passes both directions through the live gateways
      (§14.2.5); if sharing skills, `hermes config get skills.external_dirs` shows a `- ` list item and
      a moved-aside `skill_view` load-path probe resolves to the shared dir on **both** boxes (§14.3).

> The through-line: **prove it, don't report it.**

### 15.2 Automated: `verify-agent-health.sh`

The manual checklist is the spec; [`scripts/verify-agent-health.sh`](scripts/verify-agent-health.sh) is
the machine-checkable implementation. It probes the **runtime**, not just the config file — because
"config says on" is not "actually working" (every one of our failures passed a config read and still
didn't work). Point it at any agent's `$HERMES_HOME` and it returns non-zero if any axis is broken, so
it drops straight into CI or a multi-box rollup.

```bash
# Check the current user's agent
./scripts/verify-agent-health.sh

# Check another agent on the box (e.g. from an admin account)
sudo HERMES_HOME=/home/second-agent/.hermes ./scripts/verify-agent-health.sh

# Machine-readable, for rolling up results across a fleet of Spark boxes
./scripts/verify-agent-health.sh --json
```

What it verifies, mapped to the failure axes in this tutorial:

| Check | Axis | What "PASS" actually proves (runtime, not config) |
|---|---|---|
| `model.default` / `model.provider` | §5 | Primary is a real provider, **not** silently `localhost` + `EMPTY` key |
| `mem.config` / `mem.binary` / `mem.db` | §9 | `provider: mnemosyne` **and** the `mnemosyne` binary resolves **and** the DB is non-trivial |
| `backup.local` / `backup.offbox` | §12 | A recent `*.tar.zst.gpg` exists **and** the last off-box push returned `rc=0` |
| `gw.run` / `gw.perms` | §7 | A gateway process is actually **running** for the account **and** no Discord `403 / 50013` in the logs |

Real output from this repo's reference box (`piment`), run against a live agent:

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

Note the honest `[FAIL]`: on our box the bot genuinely lacked `MANAGE_MESSAGES` (the pin/unpin
permission). That's a **Discord OAuth-invite-scope** fix (§7.6), not a config-file fix — and the script
is doing its job by refusing to report green when a real permission gap exists. A checker that only ever
prints PASS is worthless; this one exits non-zero until the gap is closed or explicitly waived.

> **Two bugs we hit writing this script — worth knowing if you adapt it.** (1) `find "$HOME"` breaks
> under `sudo` because `$HOME` becomes `/root`; derive the account root from `$HERMES_HOME`'s parent
> instead. (2) The gateway-exit-diag log **only ever records failures** — it never logs a healthy
> running state — so counting "zero clean exits" falsely flags a perfectly healthy gateway as
> crash-looping. The authoritative signal for "is the gateway up?" is a live `pgrep`, not a log count.

---

## 16. Reproducibility Checklist

For team deployments across multiple Spark boxes, capture:

- [ ] OS + arch (`uname -a`, `/etc/os-release`)
- [ ] GPU + driver + CUDA (`nvidia-smi`, `nvcc --version`)
- [ ] Hermes version (`hermes --version`)
- [ ] `~/.hermes/config.yaml` (model, provider, base_url — **redact secrets**)
- [ ] Local-serving stack + exact launch flags (vLLM version, `--model`, `--max-model-len`, `--gpu-memory-utilization`)
- [ ] Per-run benchmark results (model, quant, throughput, latency, context length)

Commit these (minus secrets) so another box can be brought up identically.

---

## 17. Troubleshooting

The main walkthrough (§1–§16) is the **clean deploy path** for a fresh box. This section collects the
**known traps, silent failures, and "this bit us" war-stories** — each as *symptom → cause → proof/fix* —
so they don't clutter the happy path. Reach for a subsection only when its symptom matches.

### 17.1 Quick reference

| Symptom | Fix |
|---|---|
| `hermes` not found after install | `source ~/.bashrc`; confirm the installer added it to `PATH` |
| Agent boots but is "dim" / weak reasoning | Bare `provider: custom` with no `custom_providers:` block silently ran the local model — see [§5.1](#51-define-custom_providers-explicitly) |
| Model/provider errors | `hermes doctor`; check the API key in `~/.hermes/.env`; `hermes auth` for OAuth providers |
| Discord bot **online but ignores messages** | Enable **Message Content Intent** — see [§17.2](#172-discord-gateway) |
| Discord bot **can't pin / `403 Missing Permissions (50013)`** | Re-invite with a role granting `Manage Messages` — see [§17.2](#172-discord-gateway) |
| Discord bot not in member sidebar | Same root cause — no managed role from the invite — see [§17.2](#172-discord-gateway) |
| `hermes memory status` shows `Plugin: NOT installed ✗` | Install into the **Hermes venv** + `mnemosyne-install` + **restart** — see [§17.3](#173-memory-mnemosyne) |
| Memory status green but recall returns nothing | Scope is `session` not `global` (§9.3), wrong venv, or missing restart — see [§17.3](#173-memory-mnemosyne) |
| Episodic summaries are `<think>` garbage | Reasoning-model summarizer truncated at 2048 tokens — raise `MNEMOSYNE_LLM_MAX_TOKENS` (§9.3) |
| `episodic: 0` on a fresh gateway | The 10th-turn auto-sleep gate, not a break — see [§9.5](#95-auto-sleep-fires-only-every-10th-turn--the-gate-that-looks-like-a-broken-install) |
| Google OAuth `Error 403: access_denied` | Add your account as a **test user** at the OAuth audience page — see §11.2 |
| Gateway dies on SSH logout | `sudo loginctl enable-linger $USER` |
| Gateway crash loop | `systemctl --user reset-failed hermes-gateway` |
| Web search returns "no backend available" | `pip install ddgs` in the Hermes venv — see [§10.1](#101-ddgs-is-a-python-package-not-an-api) |
| 🔀 2nd agent: `attempt to write a readonly database` (shared DB) | SQLite created it `0644` — pre-create at `0660` — see [§14.2.3](#1423-the-trap-that-looks-like-a-hard-blocker-but-isnt-sqlites-creation-mode-0644) |
| 🔀 2nd agent: new group not live after `usermod` | Cycle the **user manager**, not the service — see [§14.2.2](#1422-the-trap-that-wastes-an-hour-usermod-needs-a-user-manager-cycle-not-a-service-restart) |
| 🔀 Shared skill authored by agent A unreadable by agent B | Tool-authored files land `0600`; `chmod 0664` at hand-off + verify from the other uid — see [§14.3.4](#1434-gate-3--file-modes-tool-authored-files-land-0600-and-no-passive-trick-fixes-it) |
| 🔀 `skills.external_dirs` set but shared skills don't load | It stored as a dict/string, not a YAML list — use the round-trip editor — see [§14.3.1](#1431-the-directory--config-wiring) |
| 🔀 Shared skill becomes unloadable ("ambiguous name") after deleting local | Leftover `_merge-staging` dirs scanned as live skills — tear them down first — see [§14.3.5](#1435-the-staging-teardown-trap--leftover-merge-dirs-get-scanned-as-live-skills) |

### 17.2 Discord gateway

**Bot is online (green) but silently ignores every message — no error anywhere.**
Cause: **Message Content Intent** was not enabled. When you invite a bot *with* a permissions
bitmask, Discord relies on this privileged intent to deliver message text; without it the session
connects and shows online but receives empty message bodies. Fix: Developer Portal → **Bot** →
**Privileged Gateway Intents** → enable **Message Content Intent** → **Save Changes** (§7.2), then
restart the gateway.
*Proof it's fixed:* send the bot an allowed message and confirm it replies.

**Bot can't pin (`403 Missing Permissions (50013)`) and/or doesn't render in the member sidebar.**
Cause: when you invite a bot with a permissions bitmask, Discord auto-creates a **managed role**
carrying those permissions. Invite with a bare `bot`-scope link and **no permissions selected** and it
lands on `@everyone` only with `roles: []` — hence 403 on every pin and no member-sidebar entry. **A
bot cannot grant itself a role.** Fix: re-invite via an **OAuth2 URL** with the correct **Bot
Permissions** (at minimum `Manage Messages` for pinning) — see §7.3. On a multi-agent box this is the
day-one trap for the *second* bot specifically — see [§7.6](#76--multiple-agents-only--roles-and-permissions-for-a-second-bot).
*Proof it's fixed:* the bot now shows a non-empty `roles` array and a pin/unpin round-trips without a 403.

### 17.3 Memory (Mnemosyne)

The clean deploy path is §9.1–§9.5. The items below are **workarounds and known upstream traps** — you
should not need them on a fresh, correctly-configured box. Reach for one only when its symptom matches.

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

**Memory status is green but recall returns nothing.** Usual culprits: scope is `session` not `global`
(§9.3), the bridge went into the wrong venv (§9.2), or no restart after install (above). The scope
mechanics, by stored scope, proven with a fresh DB and real store→recall across two session IDs:

| Stored scope | no env var | `MNEMOSYNE_CROSS_SESSION=1` |
|---|---|---|
| `scope=global`  | ✅ recalled cross-session | ✅ recalled |
| `scope=session` | ❌ not recalled            | ✅ recalled |

This is why `default_scope=global` (§9.3) is the fix on a fresh install: global rows recall
cross-session with no env var at all.

#### 17.3.1 Migrating a pre-existing DB with legacy `scope=session` rows

**When you need this:** *only* if you switched to `default_scope=global` **after** already storing
facts at the default `scope=session` (i.e. you're migrating an existing DB). On a **fresh install**
configured with `default_scope=global` from the start, **you do not need this** — every new fact is
already global.

`MNEMOSYNE_CROSS_SESSION=1` drops session filtering entirely (the recall filter becomes `(1=1)`), so it
*additionally* exposes any `session`-scoped rows written before the switch. Set it via a **systemd
drop-in** so it survives `hermes gateway install` regenerating the unit:

```bash
mkdir -p ~/.config/systemd/user/hermes-gateway.service.d
cat > ~/.config/systemd/user/hermes-gateway.service.d/10-mnemosyne-cross-session.conf <<'EOF'
[Service]
Environment="MNEMOSYNE_CROSS_SESSION=1"
EOF
systemctl --user daemon-reload && systemctl --user restart hermes-gateway   # same restart rule as §9.2

# Prove it landed in the LIVE process, not just the unit file:
tr '\0' '\n' < /proc/$(pgrep -u "$USER" -f hermes-gateway | head -1)/environ | grep MNEMOSYNE
# expect: MNEMOSYNE_CROSS_SESSION=1
# (Only reliable for a systemd drop-in var — NOT for a var living in .env; see §17.3.4.)
```

> **⚠️ Known bug: the `cross_session` *config key* is a no-op for recall — it's env-var-only.**
> (Re-verified on Mnemosyne v3.14.0.) `beam.py`'s `_cross_session_enabled()` reads only
> `os.environ["MNEMOSYNE_CROSS_SESSION"]` at import time and never consults the config resolver.
> Verified: with `cross_session: true` in config, `config.get("cross_session")` returns `True` while
> `_cross_session_enabled()` still returns `False`. **The override only works via the env var.** The
> `default_scope: global` mechanism in §9.3 is unaffected — it goes through the normal SQL filter, not
> the toggle.

#### 17.3.2 Seeding facts from the shell — `mnemosyne store` defaults to `scope=session`

If you seed memories from the shell with `mnemosyne store`, its default scope is read **only** from the
`MNEMOSYNE_DEFAULT_SCOPE` environment variable — it ignores *both* config files (including the
`hermes config` key from §9.3) and falls back to `session`. To store globally from the CLI:

```bash
MNEMOSYNE_DEFAULT_SCOPE=global mnemosyne store "a durable fact"   # else it lands scope=session
```

> **⚠️ Do not pass `--scope` to `mnemosyne store`.** It is **positional** —
> `store <content> [source] [importance]`, no flags. `mnemosyne store "fact" --scope global` fails with
> `Error: importance must be a number: global` and **stores nothing**.

> **⚠️ Upstream bug (re-verified on Mnemosyne v3.14.0):** `mnemosyne config set default_scope global`
> writes Mnemosyne's own `~/.hermes/mnemosyne/config.yaml`, and `mnemosyne config get` reads it back
> (so it *looks* applied), but `store` bypasses the config resolver entirely — `cli.py`'s `cmd_store`
> → `_resolve_default_scope()` reads only the `MNEMOSYNE_DEFAULT_SCOPE` env var. So
> `mnemosyne config set default_scope global` is effectively a **no-op** for what scope actually gets
> stored. The agent bridge does *not* use this file either; it reads `memory.mnemosyne.default_scope`
> from *Hermes* config (§9.3). Reported upstream.

#### 17.3.3 False negative: a hand-built provider in a REPL always reports `scope=session`

**Symptom:** you spin up a `MnemosyneMemoryProvider()` in a Python REPL to "check" the scope, and it
reports `session` even though `memory.mnemosyne.default_scope` is correctly `global` — making a
correctly-configured system *look broken*.

**Root cause (a harness gap, not a bug):** the provider learns where Hermes' config lives *only* from a
`hermes_home` kwarg passed at init (`self._hermes_home = kwargs.get("hermes_home", "")` — there is
**no** fallback to `HERMES_HOME` or `get_hermes_home()`). A hand-rolled `MnemosyneMemoryProvider()`
gets `_hermes_home=""`, `read_hermes_config_key("", …)` returns `None`, and the default scope silently
stays `session`.

**Fix — validate through the live agent, not a hand-built provider.** Use the §9.4 step #4 proof: have
the agent store a fact via the gateway, then read the row's scope back from the DB and expect
`scope=global`.

#### 17.3.4 Where the `MNEMOSYNE_*` env vars go, and how to verify them

**These are env vars → they live in `~/.hermes/.env`, *not* a systemd drop-in.** Hermes' invariant:
behavioral *settings* go in `config.yaml` (via `hermes config set`); *env vars / secrets* go in
`~/.hermes/.env`. `gateway/run.py` calls `load_hermes_dotenv(...)` at **module-import time** and applies
the **user** `~/.hermes/.env` with **`override=True`**, *before* the Mnemosyne bridge imports
`local_llm.py` whose `LLM_*` constants are read once at import — so `.env` populates `os.environ` in
time. A systemd `Environment=` drop-in also works but only under that exact unit; `.env` is portable
across CLI, gateway, and one-off subprocess, so it's the canonical home.

```bash
# ~/.hermes/.env  (direct edit is sanctioned — .env is not write-guarded; `hermes config env-path`
#  only PRINTS the path.)
MNEMOSYNE_LLM_ENABLED=true
MNEMOSYNE_LLM_BASE_URL=http://localhost:8000/v1
MNEMOSYNE_LLM_MODEL=<served-model-name>
```

> **⚠️ Use the *user* `~/.hermes/.env`, not a project `.env`.** Only the user env is loaded with
> `override=True`. A **project** `.env` is loaded with `override=(not loaded)`, so if a user `.env`
> already exists, `MNEMOSYNE_*` in a project `.env` gets `override=False` and a stale shell export can
> win. Put the vars in `~/.hermes/.env`.

> **⚠️ Verification gotcha — do NOT check `.env`-loaded vars via `/proc/<pid>/environ`.** That file is
> an **exec-time snapshot**; `load_hermes_dotenv()` mutates `os.environ` **at runtime**, invisible to
> `/proc/<pid>/environ`. So a var loaded from `.env` reads as "missing" there even though the process
> has it. (The `/proc/.../environ` check is only reliable for a var set via a systemd `Environment=`
> drop-in — e.g. `MNEMOSYNE_CROSS_SESSION` in §17.3.1.) The correct probe replays the loader in a clean
> environment:

```bash
# From the agent's Hermes venv dir. Proves what .env actually loads — independent of /proc.
env -i HOME=/home/<user> HERMES_HOME=/home/<user>/.hermes PATH=/usr/bin:/bin \
  ./venv/bin/python -c "import os; from pathlib import Path; \
from hermes_cli.env_loader import load_hermes_dotenv; \
load_hermes_dotenv(hermes_home=Path(os.environ['HERMES_HOME']), project_env=None); \
print({k: os.environ.get(k) for k in \
  ('MNEMOSYNE_CROSS_SESSION','MNEMOSYNE_LLM_ENABLED','MNEMOSYNE_LLM_BASE_URL','MNEMOSYNE_LLM_MODEL')})"
```

#### 17.3.5 The `auto_sleep`-defaults-to-false bug — and why the env var alone may not fix it

There was a real bug where the Hermes Mnemosyne plugin's config schema set `auto_sleep`'s default to
**`False`**, overriding Mnemosyne core's default of `True`. On an affected fresh install, memory
*appears* to work but **never consolidates**. Tracked as
[NousResearch/hermes-agent#59836](https://github.com/NousResearch/hermes-agent/issues/59836); first
patch [mnemosyne-oss/mnemosyne#420](https://github.com/mnemosyne-oss/mnemosyne/pull/420) (partial),
**superseded by the merged full fix [#429](https://github.com/mnemosyne-oss/mnemosyne/pull/429)**.

**Check the effective default on *your* installed version — don't assume:**

```bash
python -c "import mnemosyne_hermes, os; print(os.path.dirname(mnemosyne_hermes.__file__))"
grep -n '"key": "auto_sleep"' "$(python -c 'import mnemosyne_hermes,os;print(os.path.dirname(mnemosyne_hermes.__file__))')/__init__.py"
# Fixed version reads:  ... "default": True    Buggy version reads:  ... "default": False
```

**The load-bearing detail: resolution order. `config.yaml` beats the env var.**

```
kwargs  >  config.yaml key  >  MNEMOSYNE_AUTO_SLEEP_ENABLED env var  >  hardcoded schema default
```

So if a broken install already **wrote `auto_sleep: false` into `config.yaml`**, then setting
`MNEMOSYNE_AUTO_SLEEP_ENABLED=true` in `.env` is **silently ignored** — the config key wins. (The
inverse historical trap: per [mnemosyne-oss/mnemosyne#48](https://github.com/mnemosyne-oss/mnemosyne/issues/48),
fixed May 2026, *pre-#48 builds* ignored the config key entirely and **only** the env var worked. So:
current build → set the config key; pre-#48 build → env var is the only lever.)

**Fix it in the right layer (priority order):**

1. **Preferred — set the config key** (it wins the precedence race on current builds):
   ```bash
   grep -n 'auto_sleep' ~/.hermes/config.yaml || echo "no auto_sleep key — env var/default governs"
   hermes config set memory.mnemosyne.auto_sleep true
   ```
2. **Belt-and-suspenders — the env var**, for versions where the config path isn't wired or the key is
   absent (add to `~/.hermes/.env` per §17.3.4):
   ```bash
   MNEMOSYNE_AUTO_SLEEP_ENABLED=true
   ```
   Harmless on fixed versions; the actual workaround on pre-#429 builds — **but only when no
   `config.yaml auto_sleep` key shadows it.**

> **Verify the whole chain.** After changing either, grep `~/.hermes/config.yaml` for the key, confirm
> the env var resolves (§17.3.4 clean-env probe), **restart the gateway**, and treat the auto-sleep
> journal line (§9.5) as the proof it worked — not the fact that you set a flag.

### 17.4 Local serving & ARM64 build issues

| Symptom | Fix |
|---|---|
| pip package builds from source (ARM64) | Ensure `build-essential` (+ `cmake`/`ninja`) are installed |
| vLLM OOM on GB10 | Lower `--max-model-len` and `--gpu-memory-utilization`; pick a smaller/quantized model |
| Auxiliary tasks (vision/compression) fail silently | Set `OPENROUTER_API_KEY` or `GOOGLE_API_KEY`, or configure `auxiliary.*.provider` |

---

## License

MIT. Contributions and corrections welcome — open an issue or PR.

## References

- Hermes Agent: <https://github.com/NousResearch/hermes-agent>
- Docs: <https://hermes-agent.nousresearch.com/docs/>
- vLLM: <https://github.com/vllm-project/vllm>
- NVIDIA DGX Spark: <https://www.nvidia.com/en-us/products/workstations/dgx-spark/>
