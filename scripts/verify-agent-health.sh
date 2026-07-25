#!/usr/bin/env bash
# verify-agent-health.sh — Post-onboarding health check for a Hermes agent.
#
# Verifies the FOUR independent axes that were each independently broken during
# a real botched onboarding (see the "Post-Mortem" section of the README):
#   1. Model/provider block           — correct provider, base_url, api_key; not silently on a weak fallback
#   2. Long-term memory (Mnemosyne)    — configured AND the backend binary is actually installed
#   3. Backups                         — local snapshots produce encrypted archives AND off-box push works
#   4. Discord gateway integration     — gateway runs clean (not crash-looping) AND holds required permissions
#
# Config "on" is not the same as runtime "working". Each check probes the RUNTIME,
# not just the config file. Exit code is non-zero if any axis fails.
#
# Usage:
#   ./verify-agent-health.sh                 # checks the current user's ~/.hermes
#   HERMES_HOME=/home/deirdre-ai/.hermes ./verify-agent-health.sh
#   ./verify-agent-health.sh --json          # machine-readable summary (for multi-box rollups)
#   ./verify-agent-health.sh --probe-discord --channel=<id>
#          # OPT-IN: live pin/unpin test that PROVES MANAGE_MESSAGES now (write action,
#          # has side effects — posts+pins+deletes a throwaway message). Default run is
#          # read-only and idempotent; the probe is only for when you want hard proof.

set -uo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
CONFIG="$HERMES_HOME/config.yaml"
# Account root = parent of .hermes (works under sudo where $HOME would be root's).
ACCT_HOME="$(dirname "$HERMES_HOME")"
acct_user="$(basename "$ACCT_HOME")"
JSON=0
PROBE_DISCORD=0
PROBE_CHANNEL="${PROBE_CHANNEL:-}"
for arg in "$@"; do
  case "$arg" in
    --json)          JSON=1 ;;
    --probe-discord) PROBE_DISCORD=1 ;;
    --channel=*)     PROBE_CHANNEL="${arg#--channel=}" ;;
  esac
done

PASS=0; FAIL=0; WARN=0
declare -A RESULT

green(){ printf '\033[32m%s\033[0m' "$1"; }
red(){ printf '\033[31m%s\033[0m' "$1"; }
yellow(){ printf '\033[33m%s\033[0m' "$1"; }

ok(){   RESULT["$1"]=PASS; PASS=$((PASS+1)); [[ $JSON -eq 0 ]] && echo "  [$(green PASS)] $2"; }
bad(){  RESULT["$1"]=FAIL; FAIL=$((FAIL+1)); [[ $JSON -eq 0 ]] && echo "  [$(red  FAIL)] $2"; }
warn(){ RESULT["$1"]=WARN; WARN=$((WARN+1)); [[ $JSON -eq 0 ]] && echo "  [$(yellow WARN)] $2"; }

[[ $JSON -eq 0 ]] && echo "=== Hermes agent health check :: $HERMES_HOME ==="

if [[ ! -f "$CONFIG" ]]; then
  echo "FATAL: no config.yaml at $CONFIG" >&2; exit 2
fi

# ---------------------------------------------------------------------------
# AXIS 1 — Model / provider block
# ---------------------------------------------------------------------------
[[ $JSON -eq 0 ]] && echo "--- Axis 1: model/provider ---"
# Anchor to the top-level `model:` block. Blind `grep default: | head -1` is fragile —
# reorder the config and it silently grabs auxiliary.default or a memory sub-key.
# Pull the lines under `model:` up to the next top-level key, then read from those.
model_block="$(awk '/^model:/{f=1;next} /^[A-Za-z_]/{f=0} f' "$CONFIG")"
default_model="$(grep -E '^\s*default:' <<<"$model_block" | head -1 | sed 's/.*default:\s*//')"
base_url="$(grep -E '^\s*base_url:' <<<"$model_block" | head -1 | sed 's/.*base_url:\s*//')"
api_key="$(grep -E '^\s*api_key:' <<<"$model_block" | head -1 | sed 's/.*api_key:\s*//')"
# Fallback: if the model block yielded nothing (unusual layout), fall back to first match.
[[ -z "$default_model" ]] && default_model="$(grep -E '^\s*default:' "$CONFIG" | head -1 | sed 's/.*default:\s*//')"

if [[ -z "$default_model" ]]; then
  bad model.default "model.default is empty"
else
  ok model.default "model.default = $default_model"
fi
# The classic trap: silently pointed at a bare localhost with an EMPTY key and no
# real provider definition => runs on a weak local fallback while looking 'configured'.
if [[ "$base_url" == *"localhost"* || "$base_url" == *"127.0.0.1"* ]] && \
   { [[ -z "$api_key" || "$api_key" == "EMPTY" ]]; }; then
  bad model.provider "primary points at localhost with EMPTY api_key — likely silently on weak local model"
elif [[ -z "$base_url" ]]; then
  warn model.provider "no base_url on primary (may be a built-in provider — verify intended)"
else
  ok model.provider "primary base_url set ($base_url)"
fi

# ---------------------------------------------------------------------------
# AXIS 2 — Long-term memory (Mnemosyne): configured AND installed
# ---------------------------------------------------------------------------
[[ $JSON -eq 0 ]] && echo "--- Axis 2: long-term memory (Mnemosyne) ---"
mem_provider="$(grep -E '^\s*provider:\s*mnemosyne' "$CONFIG" | head -1)"
if [[ -n "$mem_provider" ]]; then
  ok mem.config "memory provider = mnemosyne in config"
  # PRESENCE IS NOT FUNCTION. A wrapper stub existing on disk proves nothing —
  # that exact 'config says mnemosyne + a file exists' check would have reported
  # PASS during the original silently-broken state. So probe FUNCTION: ask Hermes
  # itself whether the memory backend is installed AND available.
  # Run as the owning account when we're inspecting a different agent's home.
  if [[ "$acct_user" != "$(id -un)" ]]; then
    mem_status="$(sudo -u "$acct_user" -H bash -lc 'timeout 60 hermes memory status 2>/dev/null')"
  else
    mem_status="$(timeout 60 hermes memory status 2>/dev/null)"
  fi
  if grep -qE 'Plugin:\s+installed' <<<"$mem_status" && grep -qE 'Status:\s+available' <<<"$mem_status"; then
    ok mem.binary "memory backend functional (Plugin installed ✓, Status available ✓)"
  elif [[ -z "$mem_status" ]]; then
    warn mem.binary "'hermes memory status' returned nothing — could not confirm backend function (run as the agent user)"
  else
    bad mem.binary "provider=mnemosyne but backend NOT functional (Plugin/Status not confirmed) — durable memory is a no-op"
  fi
  # DB should exist and be non-trivial
  db="$HERMES_HOME/mnemosyne/data/mnemosyne.db"
  if [[ -s "$db" ]]; then
    sz=$(stat -c%s "$db" 2>/dev/null || echo 0)
    if [[ "$sz" -gt 65536 ]]; then ok mem.db "mnemosyne.db present (${sz} bytes)"; else warn mem.db "mnemosyne.db tiny (${sz} bytes) — may be empty"; fi
  else
    warn mem.db "no mnemosyne.db yet at $db"
  fi
else
  warn mem.config "memory provider is not mnemosyne (skip if intentional)"
fi

# ---------------------------------------------------------------------------
# AXIS 3 — Backups: local encrypted archive AND working off-box push
# ---------------------------------------------------------------------------
[[ $JSON -eq 0 ]] && echo "--- Axis 3: backups (local + off-box) ---"
backup_log="$(ls -1t "$HERMES_HOME"/logs/backup-*.log 2>/dev/null | head -1)"
# newest encrypted archive anywhere under the account
newest_arch="$(find "$ACCT_HOME" -maxdepth 3 -name '*.tar.zst.gpg' 2>/dev/null | xargs -r ls -1t 2>/dev/null | head -1)"
if [[ -n "$newest_arch" ]]; then
  age_h=$(( ( $(date +%s) - $(stat -c%Y "$newest_arch") ) / 3600 ))
  if [[ $age_h -le 26 ]]; then ok backup.local "recent encrypted archive ($age_h h old): $(basename "$newest_arch")"
  else warn backup.local "newest archive is ${age_h}h old — backups may have stopped: $(basename "$newest_arch")"; fi
else
  bad backup.local "no *.tar.zst.gpg archive found — local backups inoperative"
fi
# Off-box push: look for the last offbox rc in the backup log
if [[ -n "$backup_log" ]]; then
  last_offbox="$(grep -oE 'offbox rc=[0-9]+' "$backup_log" | tail -1)"
  if [[ "$last_offbox" == "offbox rc=0" ]]; then ok backup.offbox "last off-box push succeeded (rc=0)"
  elif [[ -n "$last_offbox" ]]; then bad backup.offbox "last off-box push FAILED ($last_offbox) — untested off-box = no off-box backup"
  else warn backup.offbox "no off-box push recorded in $backup_log"; fi
else
  warn backup.offbox "no backup log found under $HERMES_HOME/logs/"
fi

# ---------------------------------------------------------------------------
# AXIS 4 — Discord gateway: running clean AND correct permissions
# ---------------------------------------------------------------------------
[[ $JSON -eq 0 ]] && echo "--- Axis 4: Discord gateway ---"
# Authoritative signal: is a gateway process actually alive for this account?
gw_live="$(pgrep -u "$acct_user" -f 'gateway run' 2>/dev/null | head -1)"
gw_diag="$(ls -1t "$HERMES_HOME"/logs/gateway-exit-diag.log 2>/dev/null | head -1)"
if [[ -n "$gw_live" ]]; then
  ok gw.run "gateway process is running (pid $gw_live, user $acct_user)"
elif [[ -n "$gw_diag" ]]; then
  # No live process — the exit-diag log only records failures, so many non-zero
  # exits with nothing running is a genuine crash-loop / down signal.
  recent_exit="$(grep -c 'gateway.exit_nonzero' "$gw_diag" 2>/dev/null | head -1)"; recent_exit="${recent_exit:-0}"
  if [[ "$recent_exit" -gt 3 ]]; then
    bad gw.run "no gateway process running and $recent_exit non-zero exits logged — gateway is down/crash-looping"
  else
    bad gw.run "no gateway process running for $acct_user"
  fi
else
  warn gw.run "no gateway process and no gateway-exit-diag.log — is the gateway configured?"
fi
# Discord session READY — process alive != Discord session logged in.
# The "Connected as <bot>#<discriminator>" line prints ONLY on discord.py's on_ready;
# that IS the READY signal. A generic "response ready" line is request-handling noise
# and does NOT prove login (this exact loose-grep confusion caused a real misdiagnosis).
gw_log="$HERMES_HOME/logs/gateway.log"
if [[ -f "$gw_log" ]]; then
  if grep -qE 'Connected as .+#|discord connected' "$gw_log"; then
    ok gw.ready "Discord session reached READY ('Connected as ...' present in gateway.log)"
  else
    bad gw.ready "no 'Connected as ...' READY line in gateway.log — Discord session not logged in (a live process is NOT proof of READY)"
  fi
else
  warn gw.ready "no gateway.log at $gw_log — cannot confirm Discord READY"
fi
# --- Discord perms: LIVE PROBE is the arbiter; log-grep is the read-only fallback. ---
# Order matters: run the opt-in live probe FIRST so its ground-truth result can override
# a stale historical 403 in the log below (Deirdre's ruling: when the probe proves the
# capability is present NOW, a historical-only 403 must not stand as a hard FAIL).
#
# gw.perms.probe — OPT-IN positive live probe. Pins + unpins a throwaway message via the
# Discord REST API, so it PROVES MANAGE_MESSAGES right now. WRITE action with side effects
# (posts/pins/deletes a message) => off by default:
#   ./verify-agent-health.sh --probe-discord --channel=<channel_id>
probe_result="skip"   # skip | pass | fail
if [[ $PROBE_DISCORD -eq 1 ]]; then
  tok="$(grep -E '^DISCORD_BOT_TOKEN=' "$HERMES_HOME/.env" 2>/dev/null | head -1 | cut -d= -f2-)"
  if [[ -z "$tok" ]]; then
    warn gw.perms.probe "--probe-discord set but no DISCORD_BOT_TOKEN in $HERMES_HOME/.env"
  elif [[ -z "$PROBE_CHANNEL" ]]; then
    warn gw.perms.probe "--probe-discord set but no --channel=<id> given (need a channel to test a pin)"
  elif ! command -v curl >/dev/null 2>&1 || ! command -v python3 >/dev/null 2>&1; then
    warn gw.perms.probe "--probe-discord needs curl + python3 to call the Discord API"
  else
    api="https://discord.com/api/v10"; auth="Authorization: Bot $tok"
    # 1) post a throwaway message
    msg_id="$(curl -s -H "$auth" -H 'Content-Type: application/json' \
      -d '{"content":"health-check pin probe (auto-deleted)"}' \
      "$api/channels/$PROBE_CHANNEL/messages" 2>/dev/null \
      | python3 -c 'import sys,json;print(json.load(sys.stdin).get("id",""))' 2>/dev/null)"
    if [[ -z "$msg_id" ]]; then
      bad gw.perms.probe "could not post probe message to channel $PROBE_CHANNEL (token/channel/perms?)"
      probe_result="fail"
    else
      # 2) pin it — the MANAGE_MESSAGES-gated action that 403'd before the role fix
      pin_code="$(curl -s -o /dev/null -w '%{http_code}' -X PUT -H "$auth" \
        "$api/channels/$PROBE_CHANNEL/pins/$msg_id" 2>/dev/null)"
      if [[ "$pin_code" == "204" ]]; then
        ok gw.perms.probe "live pin succeeded (HTTP 204) — MANAGE_MESSAGES confirmed present NOW"
        probe_result="pass"
        curl -s -o /dev/null -X DELETE -H "$auth" "$api/channels/$PROBE_CHANNEL/pins/$msg_id" 2>/dev/null
      elif [[ "$pin_code" == "403" ]]; then
        bad gw.perms.probe "live pin returned 403 — MANAGE_MESSAGES is genuinely missing right now"
        probe_result="fail"
      else
        warn gw.perms.probe "live pin returned HTTP $pin_code (inconclusive)"
      fi
      # 3) clean up the throwaway message regardless
      curl -s -o /dev/null -X DELETE -H "$auth" "$api/channels/$PROBE_CHANNEL/messages/$msg_id" 2>/dev/null
    fi
  fi
fi

# Discord permission wall (MANAGE_MESSAGES / 50013), read-only log fallback. NOTE: grepping
# errors.log for absence-of-error is NOT proof the permission exists — the log rotates/clears,
# and a fresh/empty log would then falsely report PASS. So: if a 50013 is present -> FAIL
# (hard evidence of the gap); if the log is missing/empty -> WARN (cannot confirm), never PASS.
# ARBITER OVERRIDE: if the live probe PASSed this run, a historical 50013 is stale evidence and
# is downgraded to WARN — the probe is ground truth. Authoritative check is the live probe or
# the bot's role bitfield via the Discord API.
err_log="$HERMES_HOME/logs/errors.log"
if [[ ! -s "$err_log" ]]; then
  warn gw.perms "errors.log missing/empty — cannot confirm Discord perms from logs (query the bot role bitfield via API to be sure)"
elif grep -qiE 'Discord API .*403|code.*50013|Missing Permissions' "$err_log" 2>/dev/null; then
  if [[ "$probe_result" == "pass" ]]; then
    warn gw.perms "historical 50013 in errors.log, but live probe PASSed this run — treating the log 403 as stale (probe is the arbiter)"
  else
    bad gw.perms "Discord 403 Missing Permissions (50013) seen — bot role lacks perms (fix via OAuth invite scope, not config). Run --probe-discord to confirm current state."
  fi
else
  warn gw.perms "no 50013 in errors.log — but absence-of-error != permission present; confirm via bot role bitfield (Discord API) or --probe-discord for a true PASS"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
if [[ $JSON -eq 1 ]]; then
  printf '{"host":"%s","hermes_home":"%s","pass":%d,"warn":%d,"fail":%d,"checks":{' \
    "$(hostname)" "$HERMES_HOME" "$PASS" "$WARN" "$FAIL"
  first=1
  for k in "${!RESULT[@]}"; do
    [[ $first -eq 0 ]] && printf ','; first=0
    printf '"%s":"%s"' "$k" "${RESULT[$k]}"
  done
  printf '}}\n'
else
  echo "=== Summary: $(green "$PASS pass"), $(yellow "$WARN warn"), $(red "$FAIL fail") ==="
fi

if [[ $FAIL -gt 0 ]]; then
  exit 1
else
  exit 0
fi
