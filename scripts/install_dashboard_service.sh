#!/usr/bin/env bash
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/lib/hermes-agent/venv/bin/python}"
PORT="${PORT:-8100}"
USERNAME="${USERNAME:-cortex}"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
ENV_FILE="$HERMES_HOME/cortex/dashboard.env"

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Python runtime not found or not executable: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$UNIT_DIR" "$HERMES_HOME/cortex"
if [[ ! -f "$ENV_FILE" ]]; then
  umask 077
  password="$(openssl rand -hex 24)"
  printf 'CORTEX_DASHBOARD_USER=%s\nCORTEX_DASHBOARD_PASSWORD=%s\n' "$USERNAME" "$password" >"$ENV_FILE"
fi
chmod 600 "$ENV_FILE"

service_tmp="$(mktemp)"
trap 'rm -f "$service_tmp"' EXIT
sed \
  -e "s|@@HERMES_HOME@@|$HERMES_HOME|g" \
  -e "s|@@PYTHON_BIN@@|$PYTHON_BIN|g" \
  -e "s|@@PORT@@|$PORT|g" \
  "$HERMES_HOME/plugins/cortex/scripts/cortex-dashboard.service.in" >"$service_tmp"
cp "$service_tmp" "$UNIT_DIR/cortex-dashboard.service"

systemctl --user daemon-reload
systemctl --user enable cortex-dashboard.service
printf 'Installed authenticated Cortex dashboard on 127.0.0.1:%s\n' "$PORT"
printf 'Credentials are stored in %s (mode 600).\n' "$ENV_FILE"
