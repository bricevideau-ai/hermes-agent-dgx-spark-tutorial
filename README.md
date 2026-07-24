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
9. [Skills, Memory & Cron](#9-skills-memory--cron)
10. [Reproducibility Checklist](#10-reproducibility-checklist)
11. [Troubleshooting on ARM64](#11-troubleshooting-on-arm64)

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

Hermes' gateway runs the *same agent* on messaging platforms with full tool access.

```bash
hermes gateway setup       # interactive platform config
```

For Discord:

1. Create a bot at the [Discord Developer Portal](https://discord.com/developers/applications).
2. **Enable Message Content Intent** under Bot → Privileged Gateway Intents (the bot is silent without it).
3. Put the token in `~/.hermes/.env` (`DISCORD_BOT_TOKEN=...`).
4. (Recommended) Restrict who can command it: `DISCORD_ALLOWED_USERS=<your_user_id>`.
5. Invite the bot to your server with the OAuth2 URL from the portal.

Run it in the foreground first to watch the logs:

```bash
hermes gateway run
```

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

## 9. Skills, Memory & Cron

What makes Hermes more than a chat wrapper:

- **Skills** — reusable procedures the agent loads on demand and can author itself.
  ```bash
  hermes skills list
  hermes skills browse
  hermes skills install <id>
  ```
- **Memory** — persistent facts/preferences across sessions.
  ```bash
  hermes memory status
  ```
- **Cron** — durable scheduled agent runs.
  ```bash
  hermes cron create "0 9 * * *"    # e.g. a daily briefing
  hermes cron list
  ```

For a research program (e.g. benchmarking many local models over time), pair cron jobs with a skill that runs a standard benchmark suite and appends results to a persistent registry — so runs are reproducible and comparable across DGX Spark boxes.

---

## 10. Reproducibility Checklist

For team deployments across multiple Spark boxes, capture:

- [ ] OS + arch (`uname -a`, `/etc/os-release`)
- [ ] GPU + driver + CUDA (`nvidia-smi`, `nvcc --version`)
- [ ] Hermes version (`hermes --version`)
- [ ] `~/.hermes/config.yaml` (model, provider, base_url — **redact secrets**)
- [ ] Local-serving stack + exact launch flags (vLLM version, `--model`, `--max-model-len`, `--gpu-memory-utilization`)
- [ ] Per-run benchmark results (model, quant, throughput, latency, context length)

Commit these (minus secrets) so another box can be brought up identically.

---

## 11. Troubleshooting on ARM64

| Symptom | Fix |
|---|---|
| `hermes` not found after install | `source ~/.bashrc`; confirm the installer added it to `PATH` |
| Model/provider errors | `hermes doctor`; check the API key in `~/.hermes/.env`; `hermes auth` for OAuth providers |
| Discord bot silent | Enable **Message Content Intent** in the Developer Portal |
| Gateway dies on SSH logout | `sudo loginctl enable-linger $USER` |
| Gateway crash loop | `systemctl --user reset-failed hermes-gateway` |
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
