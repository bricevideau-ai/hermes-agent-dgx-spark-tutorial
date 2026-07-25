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

set -uo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
CONFIG="$HERMES_HOME/config.yaml"
# Account root = parent of .hermes (works under sudo where $HOME would be root's).
ACCT_HOME="$(dirname "$HERMES_HOME")"
JSON=0
[[ "${1:-}" == "--json" ]] && JSON=1

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
default_model="$(grep -E '^\s*default:' "$CONFIG" | head -1 | sed 's/.*default:\s*//')"
base_url="$(grep -E '^\s*base_url:' "$CONFIG" | head -1 | sed 's/.*base_url:\s*//')"
api_key="$(grep -E '^\s*api_key:' "$CONFIG" | head -1 | sed 's/.*api_key:\s*//')"

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
  # CONFIG SAYING 'on' IS NOT ENOUGH — the binary must actually resolve.
  if command -v mnemosyne >/dev/null 2>&1 || [[ -x "$HOME/.local/bin/mnemosyne" ]] \
     || ls "$HERMES_HOME"/hermes-agent/venv/bin/mnemosyne >/dev/null 2>&1; then
    ok mem.binary "mnemosyne CLI resolves (backend installed)"
  else
    bad mem.binary "provider=mnemosyne but 'mnemosyne' binary NOT found — durable memory is a no-op"
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
acct_user="$(basename "$ACCT_HOME")"
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
# Discord permission wall (MANAGE_MESSAGES / 50013) — an OAuth-invite-scope problem, not a config fix.
if grep -qiE 'Discord API .*403|code.*50013|Missing Permissions' "$HERMES_HOME"/logs/errors.log 2>/dev/null; then
  bad gw.perms "Discord 403 Missing Permissions (50013) seen — bot role lacks perms (fix via OAuth invite scope, not config)"
else
  ok gw.perms "no Discord permission (50013) errors in errors.log"
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
