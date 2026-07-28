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
   - [3d. Consolidation → local vLLM (not CPU llama-cpp)](#3d-consolidation-summaries--route-them-to-your-local-vllm-not-cpu-llama-cpp)
   - [3e. Auto-sleep fires only every 10th turn — the gate](#3e-auto-sleep-actually-fires-only-on-every-10th-turn--the-non-obvious-gate)
   - [3f. Where the `MNEMOSYNE_*` env vars go (`.env`, not a drop-in)](#3f-where-the-mnemosyne_-env-vars-actually-go--and-how-to-verify-them)
   - [3g. The `auto_sleep`-defaults-to-false bug — and why the env var alone may not fix it](#3g-the-auto_sleep-defaults-to-false-bug--and-why-the-env-var-alone-may-not-fix-it)
4. [Backups — an untested backup is not a backup](#4-backups--an-untested-backup-is-not-a-backup)
5. [Web Search & Extraction — free DDG search + self-hosted Firecrawl](#45-web-search--extraction--free-ddg-search--self-hosted-firecrawl)
6. [Discord — roles and permissions](#5-discord--roles-and-permissions)
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

Long-term memory was wired wrong in **several independent ways** at once — a missing `[all]` extra
(no vector backend), a missing bridge plugin, a cross-session scoping env var, and CLI store scope.
Each is silent on its own.

### 3a. The engine's vector extras or the Hermes↔Mnemosyne bridge aren't installed

**Symptom.** Either `hermes memory status` shows `Provider: mnemosyne` but `Plugin: NOT installed ✗`
(logs: `ModuleNotFoundError: No module named 'mnemosyne_hermes'`), **or** everything looks green,
`store` works, but semantic `recall` returns weak/irrelevant matches.

**Root cause (two distinct install gaps).**
1. **Bridge missing.** The core `mnemosyne` package is in the venv while the **bridge**
   (`mnemosyne-hermes`) is missing or its symlink is stale. Hermes *thinks* it has a memory provider;
   the wire to it is cut.
2. **Vector backend missing.** `pip install mnemosyne-memory` (bare) pulls in **only `PyYAML`** — the
   vector-search + embeddings deps (`sqlite-vec`, `fastembed`) are gated behind the **`[all]`** extra.
   Install bare and the store still accepts writes, but semantic recall silently degrades to keyword
   fallback. This is easy to miss because a canary on an *already-provisioned* box (which had `[all]`)
   passes — you must verify install steps in a **fresh venv**.

**Fix.** Install the engine **with `[all]`** and the bridge into the Hermes venv, relink, then confirm
all three: plugin, vector deps, and live coverage.

```bash
# inside the agent's Hermes venv
pip install "mnemosyne-memory[all]"     # engine + sqlite-vec + fastembed (NOT bare — bare omits both)
pip install mnemosyne-hermes            # provides the mnemosyne_hermes module
mnemosyne-install                       # relinks the Hermes plugin symlink
# restart the running process so it sees the relinked plugin (a live process won't):
systemctl --user restart hermes-gateway 2>/dev/null || true

hermes memory status                    # expect: Plugin: installed ✓ / Status: available ✓
python -c "import sqlite_vec, fastembed; print('vector deps OK')"          # both must import
mnemosyne diagnose 2>/dev/null | grep -iE "sqlite-vec|coverage|vec_working"
#   expect: Working-memory sqlite-vec coverage complete: vec_working rows=N ...
```

### 3b. Cross-session scoping — recall returns 0 in live sessions

**Symptom.** Facts are in the DB, but every *new* session recalls nothing. Query as session
`default` → rows; query as any live session id → **0 rows**.

**Root cause.** The recall filter is `(session_id = ? OR scope = 'global')`. A row is only visible in a
*different* session if it was stored at **`scope=global`**, or if the process-wide override
`MNEMOSYNE_CROSS_SESSION=1` is set (which drops session filtering to `(1=1)`). Facts stored at the
default `scope=session` fail both conditions, so they vanish cross-session. (Verified with a fresh-DB
truth table: `global` recalls cross-session with no env var; `session` does not until the override is
set.)

**Fix — set the scope on the path that actually writes your memories.** Scope resolution has three
separate code paths, each reading the default from a *different* source (all verified against source):

```bash
# 1. AGENT path (mnemosyne_remember — what the agent uses). Read from HERMES config:
hermes config set memory.mnemosyne.default_scope global
hermes config get memory.mnemosyne.default_scope     # -> global   (bridge reads THIS key)

# 2. CLI path (mnemosyne store). Reads ONLY the env var, ignores both config files:
MNEMOSYNE_DEFAULT_SCOPE=global mnemosyne store "a durable fact"   # else lands scope=session

# 3. OVERRIDE for LEGACY session-scoped rows already in the DB (recall-side):
#    ONLY needed when MIGRATING a DB that has legacy scope=session rows written before you switched
#    to default_scope=global. A FRESH install configured with default_scope=global does NOT need this.
systemctl --user edit hermes-gateway   # add: Environment=MNEMOSYNE_CROSS_SESSION=1
systemctl --user daemon-reload && systemctl --user restart hermes-gateway
tr '\0' '\n' < /proc/$(pgrep -u "$USER" -f hermes-gateway | head -1)/environ | grep MNEMOSYNE
# NOTE: /proc/<pid>/environ only shows vars set via the systemd drop-in / exec-time env.
# If you instead put MNEMOSYNE_CROSS_SESSION in ~/.hermes/.env, it will NOT appear here even
# though the process has it — see §3f for why, and for the correct .env verification probe.
```

> **⚠️ Two distinct upstream bugs make the naive config command a no-op — do not trust it:**
> - **`mnemosyne config set default_scope global` does nothing to what gets stored.** The CLI `store`
>   reads scope only from `MNEMOSYNE_DEFAULT_SCOPE` (`cli.py`: `_resolve_default_scope()`), and the
>   agent bridge reads `memory.mnemosyne.default_scope` from *Hermes* config — neither reads Mnemosyne's
>   own `config.yaml` that this command writes. `mnemosyne config get` still reflects it, which is the
>   trap.
> - **`cross_session: true` in config is a no-op for recall.** `beam.py` reads `MNEMOSYNE_CROSS_SESSION`
>   from `os.environ` at import time and never consults the config resolver. Verified:
>   `config.get("cross_session")==True` while `_cross_session_enabled()==False`. The override works only
>   via the env var.

> **⚠️ Verify through the live agent, or you'll get a false negative.** The provider learns Hermes'
> config location *only* from a `hermes_home` kwarg at init (`self._hermes_home = kwargs.get("hermes_home","")`
> — no fallback to `HERMES_HOME`/`get_hermes_home()`). A hand-built `MnemosyneMemoryProvider()` in a REPL
> therefore reads no Hermes config and reports `scope=session` even when `memory.mnemosyne.default_scope`
> is correctly `global` — a harness gap, not a bug. Prove the agent path by storing via the agent and
> reading the row back: `sqlite3 ~/.hermes/mnemosyne/data/mnemosyne.db "SELECT scope,substr(content,1,40)
> FROM working_memory ORDER BY rowid DESC LIMIT 3;"` → newest agent-written rows should be `global`.

### 3c. CLI `store` scope, and how to actually prove a round-trip

**Symptom.** Freshly migrated facts vanish cross-session even after 3a/3b are fixed.

**Root cause.** Rows stored at the default `scope=session` are invisible across sessions. Store durable
facts at **`scope=global`** — but *how* you set the default depends on which path writes them (§3b): the
agent uses `memory.mnemosyne.default_scope` in Hermes config; the `mnemosyne store` CLI uses the
`MNEMOSYNE_DEFAULT_SCOPE` env var. `MNEMOSYNE_CROSS_SESSION=1` is only needed to surface *legacy* rows
written at session scope before you switched.

> **Do not pass `--scope` to `mnemosyne store`.** `mnemosyne store` is **positional** —
> `store <content> [source] [importance]`, no flags. `mnemosyne store "fact" --scope global` fails
> with `Error: importance must be a number: global` (it parses `--scope`→source, `global`→importance)
> and **stores nothing**. From the CLI, set global scope with the `MNEMOSYNE_DEFAULT_SCOPE=global` env
> var (a store-time flag does not exist).

**Verify the whole chain — prove recall, don't just store.** A `store` that "doesn't error" proves
nothing; make the row come *back*:

```bash
CANARY="canary-$(date +%s)"
mnemosyne store "$CANARY: memory round-trip works"   # positional; -> "Stored: <id>"
mnemosyne recall "$CANARY"                            # MUST return the row (id + content + score)
hermes mnemosyne stats                               # count should include the canary
```

To prove it survives the cross-session boundary specifically, recall it from a *fresh* agent session
and confirm the live gateway (which carries the env var) sees it too — if the CLI recalls it but the
gateway doesn't, that's a §3b env-var gap, not a store problem.

### 3d. Consolidation summaries — route them to your local vLLM, not CPU llama-cpp

**Symptom.** Memory *consolidation* (the "sleep" pass that compresses working memories into episodic
summaries) is slow — minutes per batch — and the resulting "summary" is garbage: raw `<think>`
reasoning tokens leaked verbatim instead of a clean summary.

**Root cause.** Mnemosyne's **default** consolidation LLM is a CPU llama-cpp GGUF. It's slow on a
box whose GPU is already serving a good model, and a reasoning GGUF emits `<think>` sludge that the
cleaner doesn't fully strip. Meanwhile the local vLLM you already stood up for the agent (§1) is
sitting right there, idle for this purpose.

**Fix — point consolidation at the local vLLM (no API key needed).** The selection lives in
`mnemosyne/core/local_llm.py::summarize_memories()`. The remote-API path fires when
`MNEMOSYNE_LLM_BASE_URL` is set **and** `MNEMOSYNE_LLM_ENABLED` is not false — verified against source
(find it in your own tree, since line numbers drift between releases):

```bash
# Locate the remote-path condition by symbol, not by line number:
python -c "import mnemosyne.core.local_llm as m; print(m.__file__)"   # -> path to local_llm.py
grep -n "def summarize_memories\|MNEMOSYNE_FORCE_LOCAL\|LLM_ENABLED and LLM_BASE_URL" <that path>
```

```python
# mnemosyne/core/local_llm.py, inside summarize_memories()
if LLM_ENABLED and LLM_BASE_URL and not os.environ.get("MNEMOSYNE_FORCE_LOCAL", ...):
    raw = _call_remote_llm(prompt)      # ← OpenAI-compatible call to your vLLM
```

`LLM_ENABLED` **defaults to `true`** (`grep -n '^LLM_ENABLED' local_llm.py` → `os.environ.get("MNEMOSYNE_LLM_ENABLED", "true")`),
so in practice *just setting the base URL* is enough to switch consolidation onto vLLM. Set these
three (see [§3f](#3f-where-the-mnemosyne_-env-vars-actually-go--and-how-to-verify-them) for **where** they go):

```bash
MNEMOSYNE_LLM_ENABLED=true
MNEMOSYNE_LLM_BASE_URL=http://localhost:8000/v1     # your vLLM OpenAI endpoint
MNEMOSYNE_LLM_MODEL=<served-model-name>             # e.g. qwen — the name vLLM serves
```

vLLM needs **no API key** for a local, unauthenticated endpoint. Measured A/B on our boxes: vLLM
(`qwen`) produced a faithful ~450-char summary in **~16 s**; the CPU GGUF took **~21 s+** and emitted
`<think>` sludge with no usable output.

> **⚠️ Footgun: `MNEMOSYNE_FORCE_LOCAL`.** Setting `MNEMOSYNE_FORCE_LOCAL=1` (or `true`/`yes`) forces
> consolidation **back** to the CPU GGUF *even when the base URL is set* — it's the `not os.environ.get("MNEMOSYNE_FORCE_LOCAL"...)`
> clause in the `summarize_memories()` condition above that short-circuits the remote path. If you set
> it "just to test the local path," clear it afterward — a stray `MNEMOSYNE_FORCE_LOCAL` is a silent way
> to end up back on the slow, `<think>`-leaking path.

**One-time backfill (bypass the turn gate in [§3e](#3e-auto-sleep-actually-fires-only-on-every-10th-turn--the-non-obvious-gate)).** To consolidate a
backlog *right now* without waiting for the auto-sleep trigger, run a fresh subprocess with the env
vars set (a fresh process gets a fresh reflect budget and doesn't need the live turn counter):

```bash
# From the agent's Hermes venv, with the MNEMOSYNE_LLM_* vars exported:
python - <<'PY'
from mnemosyne.core import memory as M
M.sleep_all_sessions(dry_run=False, force=True)   # force=True ignores the age cutoff
PY
```

> `force=True` sets the age cutoff to "everything," so it consolidates all unconsolidated working
> rows regardless of age; `sleep_all_sessions` (vs `sleep`) walks **every** session, not just the
> current one — the right choice when your working rows are spread across many Discord-thread/CLI
> sessions. Over a dozen sessions with a 30B-class reasoning model this can exceed a few minutes;
> run it in the background rather than a short foreground timeout.

**Verify — confirm the remote path actually ran.** Don't infer it from timing alone:

```bash
# During a consolidation run, confirm an ESTABLISHED socket to the vLLM port:
ss -tnp | grep :8000
# And spot-check a fresh episodic summary is clean prose, NOT <think>...:
sqlite3 ~/.hermes/mnemosyne/data/mnemosyne.db \
  "SELECT substr(content,1,120) FROM episodic_memory ORDER BY rowid DESC LIMIT 1;"
```

> An `hf_hub_download` / "unauthenticated HF Hub" warning during the run is **noise**, not proof the
> local GGUF was used — Mnemosyne lazily probes the local tokenizer to size chunks even when the
> remote path handles the actual summary. Confirm the remote path by the live socket to vLLM and by
> clean summary text, not by the absence of that warning.

### 3e. Auto-sleep actually fires only on every 10th turn — the non-obvious gate

**Symptom.** A freshly-restarted, *correctly-wired* gateway shows `episodic: 0` even though
`working > 50`. It looks like consolidation is broken. It isn't.

**Root cause — the turn-count gate.** Consolidation is built in (no cron needed), but the auto-sleep
trigger is gated on a **per-session turn counter**. It lives in the **Mnemosyne↔Hermes bridge**, not
in `gateway/run.py` or `run_agent.py` — find it by symbol (line numbers drift between releases, and
this file churns):

```bash
# Locate the bridge and the gate, in your own tree:
python -c "import mnemosyne_hermes, os; print(os.path.dirname(mnemosyne_hermes.__file__))"
grep -n "_turn_count\|% 10\|_maybe_auto_sleep\|_auto_sleep_threshold\|auto-sleep: working" \
  <that dir>/__init__.py
```

```python
# mnemosyne_hermes/__init__.py — the gate (symbol-anchored, not line-anchored):
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

**There is a second, quieter no-op** worth knowing so you don't misdiagnose it as the gate: even on a
turn that *is* a multiple of 10 with `working > threshold`, `_maybe_auto_sleep()` returns **without
logging anything** if the eligibility check (`_count_unconsolidated_before(cutoff)`, where `cutoff`
is `now − TTL/2`) finds **zero** rows — i.e. everything old is already consolidated. So `episodic: 0`
+ `working > 50` + no journal line has **two** innocent causes: (1) `< 10` turns since restart, or
(2) nothing is actually eligible yet.

**Confirm it's the gate, not a config break.** Grep the gateway journal for the auto-sleep log line.
Its **presence** proves the trigger fired; its **absence** means either `< 10` turns since restart or
nothing eligible — not a broken config:

```bash
journalctl --user -u hermes-gateway | grep "Mnemosyne auto-sleep:"
# A hit looks like:  Mnemosyne auto-sleep: working=63, eligible=41 > threshold=50
# Absent  ⇒  <10 turns since restart (the gate) OR nothing eligible — NOT a wiring fault.
```

> **Defaults & knobs (verified — grep, don't trust a line number):** the threshold is `working > 50`
> (`grep -n '_auto_sleep_threshold = ' __init__.py` → `= 50`; config key `sleep_threshold`); auto-sleep
> is on by default (config key `auto_sleep` / legacy `MNEMOSYNE_AUTO_SLEEP_ENABLED`). When it does fire
> it runs `sleep_all_sessions` (cross-session) in a daemon thread. To consolidate immediately without
> waiting out ten turns, use the one-time backfill in [§3d](#3d-consolidation-summaries--route-them-to-your-local-vllm-not-cpu-llama-cpp).

### 3f. Where the `MNEMOSYNE_*` env vars actually go — and how to verify them

**These are env vars → they live in `~/.hermes/.env`, *not* a systemd drop-in.** Hermes' hard
invariant: behavioral *settings* go in `config.yaml` (via `hermes config set`, which is write-guarded);
*env vars / secrets* go in `~/.hermes/.env`. The `MNEMOSYNE_*` knobs are env vars, so they belong in
`.env`.

**Why `.env` and not a drop-in — verified against source.** `gateway/run.py` calls
`load_hermes_dotenv(...)` at **module-import time** (top-level, not inside a function), and the loader
applies the **user** `~/.hermes/.env` with **`override=True`**. That runs **before** the Mnemosyne
bridge imports `mnemosyne/core/local_llm.py`, whose `LLM_BASE_URL` / `LLM_ENABLED` / `LLM_MODEL` are
**module-level constants read once at import**. So `.env` populates `os.environ` in time for those
constants to pick up your values. A systemd `Environment=` drop-in also works, but `.env` is
**portable across launch methods** (CLI, gateway, one-off subprocess) while a drop-in only applies
under that exact unit — so `.env` is the canonical, single-source home. Confirm the mechanism in your
own tree by symbol (line numbers move release-to-release — on our two 0.19.0 checkouts alone
`load_hermes_dotenv` sat ~80 lines apart):

```bash
grep -n "load_hermes_dotenv" gateway/run.py            # top-level call, before the bridge imports
grep -n "override=True" hermes_cli/env_loader.py       # the user-.env load line
grep -nE "^LLM_(BASE_URL|ENABLED|REMOTE_MODEL) " \
  "$(python -c 'import mnemosyne.core.local_llm as m; print(m.__file__)')"   # import-time constants
```

> **⚠️ Use the *user* `~/.hermes/.env`, not the project `.env`.** Only the user env is loaded with
> `override=True`. The **project** `.env` is loaded with `override=(not loaded)` and the ops env with
> `override=False` (grep `env_loader.py` for the three `_load_dotenv_with_fallback(...)` calls). So if
> a user `.env` already exists and you drop `MNEMOSYNE_*` into a *project* `.env` instead, its
> `override` flips to `False` and a stale shell export can win. Put the vars in `~/.hermes/.env`.

```bash
# ~/.hermes/.env  (direct edit is sanctioned — .env is not write-guarded; no CLI writes it.
#  `hermes config env-path` only PRINTS the path.)
MNEMOSYNE_LLM_ENABLED=true
MNEMOSYNE_LLM_BASE_URL=http://localhost:8000/v1
MNEMOSYNE_LLM_MODEL=qwen
```

> **Hygiene:** if the same var is *also* exported from a systemd drop-in **and** `~/.profile` **and**
> `~/.bashrc`, you get four sources of truth and inevitable drift. Consolidate to `~/.hermes/.env`
> and comment out the rest.

> **⚠️ Verification gotcha — do NOT check `.env`-loaded vars via `/proc/<pid>/environ`.** That file is
> an **exec-time snapshot**: it reflects only what was in the environment when the process was `exec`'d.
> `load_hermes_dotenv()` mutates `os.environ` **at runtime**, and that mutation is **invisible** to
> `/proc/<pid>/environ` — so a var loaded from `.env` reads as "missing" there even though the process
> genuinely has it. (This means the `/proc/.../environ` check suggested in §3b is only reliable for a
> var set via a systemd `Environment=` drop-in — **not** for one living in `.env`.) The correct probe
> replays the loader in a clean environment and prints what it resolves:

```bash
# From the agent's Hermes venv dir. Proves what .env actually loads — independent of /proc.
env -i HOME=/home/<user> HERMES_HOME=/home/<user>/.hermes PATH=/usr/bin:/bin \
  ./venv/bin/python -c "import os; from pathlib import Path; \
from hermes_cli.env_loader import load_hermes_dotenv; \
load_hermes_dotenv(hermes_home=Path(os.environ['HERMES_HOME']), project_env=None); \
print({k: os.environ.get(k) for k in \
  ('MNEMOSYNE_CROSS_SESSION','MNEMOSYNE_LLM_ENABLED','MNEMOSYNE_LLM_BASE_URL','MNEMOSYNE_LLM_MODEL')})"
```

### 3g. The `auto_sleep`-defaults-to-false bug — and why the env var alone may not fix it

There was a real bug where the Hermes Mnemosyne plugin's config schema set `auto_sleep`'s
default to **`False`**, overriding Mnemosyne core's default of `True`. On an affected fresh
install, memory *appears* to work but **never consolidates** — the agent quietly keeps forgetting
cross-session context. Tracked as [NousResearch/hermes-agent#59836](https://github.com/NousResearch/hermes-agent/issues/59836);
first patch attempt [mnemosyne-oss/mnemosyne#420](https://github.com/mnemosyne-oss/mnemosyne/pull/420)
(flipped the schema line but left a runtime fallback gap), **superseded by the merged full fix
[#429](https://github.com/mnemosyne-oss/mnemosyne/pull/429)** which covers both provider surfaces
plus regression tests.

**Are you affected? Check the effective default on *your* installed version — don't assume.**

```bash
# Grep the schema default in your actually-installed plugin (path varies; find it first):
python -c "import mnemosyne_hermes, os; print(os.path.dirname(mnemosyne_hermes.__file__))"
grep -n '"key": "auto_sleep"' "$(python -c 'import mnemosyne_hermes,os;print(os.path.dirname(mnemosyne_hermes.__file__))')/__init__.py"
# Fixed version reads:  ... "default": True  (description: "Set false to disable")
# Buggy version reads:  ... "default": False (description: "Set true to enable")
```

**The load-bearing detail: resolution order. `config.yaml` beats the env var.** In the plugin,
`auto_sleep` is resolved (verified — `grep -n 'auto_sleep = kwargs.get' __init__.py`, read the block):

```
kwargs  >  config.yaml key  >  MNEMOSYNE_AUTO_SLEEP_ENABLED env var  >  hardcoded schema default
```

So if a broken install already **wrote `auto_sleep: false` into `config.yaml`**, then setting
`MNEMOSYNE_AUTO_SLEEP_ENABLED=true` in `.env` is **silently ignored** — the config key wins. The
env var only takes effect when there is *no* competing `config.yaml` key.

**The inverse historical trap — very old builds ignore the config key entirely.** Per
[mnemosyne-oss/mnemosyne#48](https://github.com/mnemosyne-oss/mnemosyne/issues/48) (fixed May 2026,
commit `2b0a478`), on **pre-fix builds** the `memory.mnemosyne.auto_sleep` key was *advertised in the
schema but never applied at runtime* — `initialize()` didn't read config and `save_config()` was a
no-op, so back then **only the env var worked**. The fix added an `_apply_provider_config()` read
path (kwargs → `config.yaml` → env fallback). So the version-aware truth is:

- **Fixed build (current, post-#48):** the config key is honored and *wins* the precedence race — use it.
- **Pre-#48 build:** the config key is silently ignored → `MNEMOSYNE_AUTO_SLEEP_ENABLED=true` is the
  *only* lever that works.

Which is exactly why you **confirm your build honors the key** (set it, restart, look for the
[§3e](#3e-auto-sleep-actually-fires-only-on-every-10th-turn--the-non-obvious-gate) journal line)
rather than assume — and why the env var below is worth setting as a fallback regardless.

**Fix it in the right layer (in priority order):**

1. **Preferred — set the config key** (this is a behavioral setting, and it wins the precedence
   race). Check what you have, then set it true:
   ```bash
   grep -n 'auto_sleep' ~/.hermes/config.yaml || echo "no auto_sleep key — env var/default governs"
   # If it's present and false (or you want to be explicit), set it true:
   hermes config set memory.mnemosyne.auto_sleep true
   ```
2. **Belt-and-suspenders — the env var**, for installs/versions where the config path isn't wired
   or the key is absent. Add to `~/.hermes/.env` (see [§3f](#3f-where-the-mnemosyne_-env-vars-actually-go--and-how-to-verify-them)
   for why `.env` and how to verify):
   ```bash
   MNEMOSYNE_AUTO_SLEEP_ENABLED=true
   ```
   It's harmless on already-fixed versions (the fixed default is `True` anyway), and it's the actual
   workaround on pre-#429 builds — **but only when no `config.yaml auto_sleep` key shadows it.**

> **Verify the whole chain, don't trust one layer.** After changing either, confirm the *effective*
> value by grepping `~/.hermes/config.yaml` for the key **and** confirming the env var resolves from
> `.env` (the [§3f](#3f-where-the-mnemosyne_-env-vars-actually-go--and-how-to-verify-them) clean-env
> probe), then **restart the gateway** — like every `MNEMOSYNE_*` change, this only takes effect on
> the next restart (the running gateway read its env at startup). Proof it worked is the auto-sleep
> journal line from [§3e](#3e-auto-sleep-actually-fires-only-on-every-10th-turn--the-non-obvious-gate),
> not the fact that you set a flag.

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

## 4.5. Web Search & Extraction — free DDG search + self-hosted Firecrawl

**Symptom.** The agent can't search the web, or you assume you need a paid search API key
(Tavily/Exa/Brave/Parallel) to give it eyes on the internet. You don't.

**Root cause / the mental model.** Hermes splits web capability into two independent axes, each with
its own backend selector, and Hermes picks a backend by *availability*, not by a key you bought:

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

### Gotcha 1 — `ddgs` is a PYTHON PACKAGE, not an API

There is **no key and no endpoint** for DuckDuckGo search. `ddgs` is a pip package that scrapes DDG;
Hermes treats the backend as available **iff `import ddgs` succeeds** — nothing else. (In
`web_tools.py`, `ddgs` is the *only* backend whose availability is a package-import probe rather than
an env-var/key check.) So the entire "setup" is:

```bash
# inside the agent's Hermes venv
pip install ddgs
python -c "import ddgs; print('ddgs', ddgs.__version__)"   # prove the import works
```

> Do **not** mistake DuckDuckGo for a "free API endpoint" — there isn't one. The package does the
> work locally. If `import ddgs` fails, Hermes silently treats the backend as unavailable and your
> search falls through to whatever else is configured (often nothing).

### Gotcha 2 — Firecrawl extract points at a **self-hosted** instance, no key

`web_extract` uses Firecrawl, but pointed at a **local** Firecrawl you host yourself — free, private,
no rate-limited SaaS key. Firecrawl is considered available when **either** `FIRECRAWL_API_KEY`
**or** `FIRECRAWL_API_URL` is set (`grep -n 'FIRECRAWL_API_URL' tools/web_tools.py` → the `_has_env(...)` check in the capability table), so the URL alone is enough:

```bash
# Base ORIGIN only — no /v1 suffix, no key.
echo 'FIRECRAWL_API_URL=http://localhost:3002' >> ~/.hermes/.env
```

> **Two footguns:** (1) use the bare origin `http://localhost:3002` — **not** `.../v1`; Hermes appends
> the path itself, and a `/v1` suffix double-paths the request. (2) `FIRECRAWL_API_URL` is a genuine
> credential-adjacent endpoint, so it lives in `~/.hermes/.env`, not `config.yaml`.

### Hosting the local Firecrawl container stack

> **First, decide whether you even need to host it.** On a **shared host**, only **one** agent runs the
> Firecrawl stack; every other agent on the box just points `FIRECRAWL_API_URL` at the existing one —
> **no clone, no second stack.** On our box the stack is hosted once under the first agent's user
> (`/home/videau-ai/services/firecrawl/`, port `:3002`), and the second agent (`deirdre-ai`) consumes it
> purely as a **client** — there is no `~/services/firecrawl` under the second user at all. Following the
> recipe below on a second agent would spin up a *redundant* 5-container stack or collide on `:3002`.
> So: **host once per box, point every other agent at it.** The steps below are for the **one** agent
> that hosts it (or a single-agent box).

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
> **not** independently clean-room tested here; both our agents verified against an
> already-running stack. Treat the clone/env steps as the documented-but-unproven path and expect to
> read Firecrawl's own self-host docs if a fresh bring-up hiccups.

### Verify — prove BOTH capabilities end-to-end (don't assume)

A configured backend is a hypothesis until a live query returns real content. Ask the running agent
to actually use each path:

```bash
# Search (ddgs): must return live, real URLs — not an "no backend available" error.
hermes chat -q "Use web_search for 'NVIDIA DGX Spark GB10 specifications' and list 3 result URLs."

# Extract (self-hosted firecrawl): must return page text with no error.
hermes chat -q "Use web_extract on https://example.com and quote the first sentence."
```

Both must come back with genuine content. On our boxes: DDG search returns live results end-to-end,
and extract pulls page text straight from the local Firecrawl with no key and no SaaS round-trip.

> **The through-line for this axis:** "a search key costs money" is a false assumption. `ddgs`
> (package, no key) + self-hosted Firecrawl (`FIRECRAWL_API_URL`, no key) gives a second agent full
> web eyes for **$0** and keeps every fetch private to the box.

---

## 5. Discord — roles and permissions

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
- [ ] **Consolidation.** `MNEMOSYNE_LLM_BASE_URL` points at the local vLLM in `~/.hermes/.env` (not a
      drop-in); a fresh episodic summary is clean prose (no `<think>` leak); the auto-sleep journal
      line appears after ≥10 turns, or a forced `sleep_all_sessions(force=True)` backfill produced
      episodic rows.
- [ ] **Backup.** Encrypted archive pushed off-box on a cron; **restored from the cloud copy** and
      `PRAGMA integrity_check` = `ok`; passphrase pinned off-box; negative test fails loud.
- [ ] **Web.** `import ddgs` succeeds in the venv; `FIRECRAWL_API_URL` (bare origin, no `/v1`) set and
      the local Firecrawl stack answers on `:3002`; live `web_search` returns real URLs and
      `web_extract` returns page text — both with **no paid key**.
- [ ] **Discord.** Bot has a managed role (`member_info` shows a non-empty `roles`); the exact
      privileged action that failed now succeeds (pin/unpin); gateway log shows a `Connected as …`
      READY line (not a generic `response ready` line).

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

