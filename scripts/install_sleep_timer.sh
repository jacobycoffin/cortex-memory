#!/usr/bin/env bash
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/lib/hermes-agent/venv/bin/python}"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

fail() {
  printf 'Cortex Sleep timer install failed: %s\n' "$1" >&2
  exit 1
}

resolve_executable() {
  local candidate="$1"
  local candidate_dir
  local resolved
  if [[ "$candidate" == */* ]]; then
    [[ -x "$candidate" ]] || return 1
    candidate_dir="$(cd "$(dirname "$candidate")" && pwd -P)" || return 1
    printf '%s/%s\n' "$candidate_dir" "$(basename "$candidate")"
    return 0
  fi
  resolved="$(command -v "$candidate")" || return 1
  [[ -x "$resolved" ]] || return 1
  printf '%s\n' "$resolved"
}

escape_sed_replacement() {
  printf '%s' "$1" | sed 's/[&|\\]/\\&/g'
}

validate_unit_path() {
  case "$1" in
    *$'\n'*|*$'\r'*|*$'\t'*) fail "$2 cannot contain control characters" ;;
    *'%'*|*'"'*|*'\'*) fail "$2 contains a character that is unsafe in a systemd unit" ;;
  esac
}

validate_unit_path "$HERMES_HOME" "HERMES_HOME"
validate_unit_path "$PYTHON_BIN" "PYTHON_BIN"
case "$UNIT_DIR" in
  *$'\n'*|*$'\r'*) fail "systemd unit directory cannot contain newline characters" ;;
esac

PYTHON_BIN="$(resolve_executable "$PYTHON_BIN")" || fail "Python runtime not found or not executable: $PYTHON_BIN"
mkdir -p "$HERMES_HOME/cortex" "$UNIT_DIR"
HERMES_HOME="$(cd "$HERMES_HOME" && pwd -P)"
UNIT_DIR="$(cd "$UNIT_DIR" && pwd -P)"
validate_unit_path "$HERMES_HOME" "resolved HERMES_HOME"
validate_unit_path "$PYTHON_BIN" "resolved PYTHON_BIN"

TEMPLATE_DIR="$HERMES_HOME/plugins/cortex/scripts"
SERVICE_TEMPLATE="$TEMPLATE_DIR/cortex-sleep.service.in"
TIMER_TEMPLATE="$TEMPLATE_DIR/cortex-sleep.timer"

[[ -f "$HERMES_HOME/plugins/cortex/__main__.py" ]] || fail "Cortex is not installed in $HERMES_HOME/plugins/cortex"
[[ -f "$SERVICE_TEMPLATE" ]] || fail "service template not found: $SERVICE_TEMPLATE"
[[ -f "$TIMER_TEMPLATE" ]] || fail "timer template not found: $TIMER_TEMPLATE"

SLEEP_ENV="$HERMES_HOME/cortex/sleep.env"
if [[ ! -f "$SLEEP_ENV" ]]; then
  printf 'CORTEX_SLEEP_TOKEN_BUDGET=0\n' >"$SLEEP_ENV"
fi
chmod 0600 "$SLEEP_ENV"

if ! PYTHONPATH="$HERMES_HOME/plugins" "$PYTHON_BIN" -m cortex sleep --help >/dev/null 2>&1; then
  fail "the installed Cortex version does not provide the sleep command"
fi

service_tmp="$(mktemp)"
timer_tmp="$(mktemp)"
cleanup() {
  rm -f "$service_tmp" "$timer_tmp"
}
trap cleanup EXIT INT TERM

hermes_replacement="$(escape_sed_replacement "$HERMES_HOME")"
python_replacement="$(escape_sed_replacement "$PYTHON_BIN")"
sed \
  -e "s|@@HERMES_HOME@@|$hermes_replacement|g" \
  -e "s|@@PYTHON_BIN@@|$python_replacement|g" \
  "$SERVICE_TEMPLATE" >"$service_tmp"
cp "$TIMER_TEMPLATE" "$timer_tmp"
chmod 0644 "$service_tmp" "$timer_tmp"
cp "$service_tmp" "$UNIT_DIR/cortex-sleep.service"
cp "$timer_tmp" "$UNIT_DIR/cortex-sleep.timer"

systemctl --user daemon-reload
systemctl --user enable --now cortex-sleep.timer

printf 'Enabled nightly Cortex Sleep in shadow mode.\n'
printf 'Default reflection token budget: 0 (review the existing env file if already configured).\n'
printf 'Optional reflection settings: %s\n' "$SLEEP_ENV"
printf 'Timer: 03:00 local time with up to 2 hours randomized delay; missed runs are persistent.\n'
printf 'Inspect the schedule with: systemctl --user list-timers cortex-sleep.timer\n'
