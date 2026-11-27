#!/usr/bin/env bash
# Install the cortex-rs user unit. Does not stop, disable, or rewrite any
# other unit. The env file holds paths only; it never holds a password.
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-${HOME:-}/.hermes}"
PORT="${PORT:-8090}"
UPSTREAM="http://127.0.0.1:8100"
UNIT_DIR="${UNIT_DIR:-${XDG_CONFIG_HOME:-${HOME:-}/.config}/systemd/user}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ASSET_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TEMPLATE="$SCRIPT_DIR/cortex-serve.service.in"

fail() {
  printf 'cortex-rs service install failed: %s\n' "$1" >&2
  exit 1
}

escape_sed_replacement() {
  printf '%s' "$1" | sed 's/[&|\\]/\\&/g'
}

validate_unit_path() {
  case "$1" in
    ''|*[![:print:]]*) fail "$2 cannot be empty or contain control characters" ;;
    *' '*) fail "$2 cannot contain spaces" ;;
    *'%'*) fail "$2 contains a character that is unsafe in a systemd unit" ;;
    *'"'*) fail "$2 contains a character that is unsafe in a systemd unit" ;;
    *'\'*) fail "$2 contains a character that is unsafe in a systemd unit" ;;
  esac
}

resolve_executable() {
  local candidate="$1"
  local resolved
  if [[ "$candidate" == */* ]]; then
    [[ -x "$candidate" ]] || return 1
    printf '%s/%s\n' "$(cd -- "$(dirname -- "$candidate")" && pwd -P)" "$(basename -- "$candidate")"
    return 0
  fi
  resolved="$(command -v "$candidate")" || return 1
  [[ -x "$resolved" ]] || return 1
  if [[ "$resolved" == */* ]]; then
    printf '%s/%s\n' "$(cd -- "$(dirname -- "$resolved")" && pwd -P)" "$(basename -- "$resolved")"
  else
    printf '%s\n' "$resolved"
  fi
}

port_is_open() {
  python3 - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.5)
try:
    sock.connect(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
sys.exit(0)
PY
}

[[ -n "$HERMES_HOME" && "$HERMES_HOME" != "/.hermes" ]] || fail "set HERMES_HOME (default is \$HOME/.hermes)"
[[ -n "$UNIT_DIR" && "$UNIT_DIR" != "/.config/systemd/user" ]] || fail "set UNIT_DIR or HOME"
[[ "$PORT" =~ ^[0-9]+$ ]] || fail "PORT must be a TCP port from 1 to 65535"
if (( PORT < 1 || PORT > 65535 )); then
  fail "PORT must be a TCP port from 1 to 65535"
fi
[[ -f "$TEMPLATE" ]] || fail "service template not found: $TEMPLATE"
command -v python3 >/dev/null 2>&1 || fail "python3 is required to check whether the port is in use"

if [[ -z "${BIN:-}" ]]; then
  if [[ -x "$ASSET_DIR/rust/target/release/cortex-rs" ]]; then
    BIN="$ASSET_DIR/rust/target/release/cortex-rs"
  elif [[ -x "$ASSET_DIR/rust/target/debug/cortex-rs" ]]; then
    BIN="$ASSET_DIR/rust/target/debug/cortex-rs"
  else
    BIN="$(command -v cortex-rs || true)"
  fi
fi
[[ -n "${BIN:-}" ]] || fail "cortex-rs was not found; build it or set BIN"
candidate="$BIN"
BIN="$(resolve_executable "$candidate")" || fail "cortex-rs is not executable: $candidate"

validate_unit_path "$HERMES_HOME" "HERMES_HOME"
validate_unit_path "$UNIT_DIR" "UNIT_DIR"
validate_unit_path "$BIN" "BIN"
validate_unit_path "$ASSET_DIR" "asset directory"
validate_unit_path "$PORT" "PORT"

if port_is_open "$PORT"; then
  printf '127.0.0.1:%s is already in use; leaving the occupant running\n' "$PORT" >&2
  exit 1
fi

mkdir -p "$HERMES_HOME/cortex" "$UNIT_DIR"
HERMES_HOME="$(cd -- "$HERMES_HOME" && pwd -P)"
UNIT_DIR="$(cd -- "$UNIT_DIR" && pwd -P)"
ASSET_DIR="$(cd -- "$ASSET_DIR" && pwd -P)"
validate_unit_path "$HERMES_HOME" "resolved HERMES_HOME"
validate_unit_path "$UNIT_DIR" "resolved UNIT_DIR"
validate_unit_path "$ASSET_DIR" "resolved asset directory"

ENV_FILE="$HERMES_HOME/cortex/cortex-serve.env"
if [[ ! -f "$ENV_FILE" ]]; then
  {
    printf 'CORTEX_DB=%s\n' "$HERMES_HOME/cortex/cortex.db"
    printf 'CORTEX_ASSET_DIR=%s\n' "$ASSET_DIR"
    printf 'CORTEX_PYTHON_UPSTREAM=%s\n' "$UPSTREAM"
    printf 'CORTEX_DASHBOARD_AUTH_FILE=%s\n' "$HERMES_HOME/cortex/dashboard-auth.json"
  } >"$ENV_FILE"
fi
chmod 600 "$ENV_FILE"

service_tmp="$(mktemp)"
trap 'rm -f "$service_tmp"' EXIT
sed \
  -e "s|@@HERMES_HOME@@|$(escape_sed_replacement "$HERMES_HOME")|g" \
  -e "s|@@BIN@@|$(escape_sed_replacement "$BIN")|g" \
  -e "s|@@PORT@@|$(escape_sed_replacement "$PORT")|g" \
  "$TEMPLATE" >"$service_tmp"
chmod 644 "$service_tmp"
cp "$service_tmp" "$UNIT_DIR/cortex-serve.service"

if [[ "${CORTEX_SERVE_INSTALL_NO_SYSTEMCTL:-}" != "1" ]]; then
  systemctl --user daemon-reload
  systemctl --user enable cortex-serve.service
fi

printf 'Installed cortex-rs on 127.0.0.1:%s\n' "$PORT"
printf 'Python upstream: %s\n' "$UPSTREAM"
