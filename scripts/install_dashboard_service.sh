#!/usr/bin/env bash
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/lib/hermes-agent/venv/bin/python}"
PORT="${PORT:-8100}"
USERNAME="${USERNAME:-cortex}"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
ENV_FILE="$HERMES_HOME/cortex/dashboard.env"
AUTH_FILE="$HERMES_HOME/cortex/dashboard-auth.json"

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Python runtime not found or not executable: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$UNIT_DIR" "$HERMES_HOME/cortex"
if [[ ! -f "$ENV_FILE" ]]; then
  umask 077
  printf 'CORTEX_DASHBOARD_USER=%s\n' "$USERNAME" >"$ENV_FILE"
else
  existing_username="$(sed -n 's/^CORTEX_DASHBOARD_USER=//p' "$ENV_FILE" | tail -n 1)"
  if [[ -n "$existing_username" ]]; then
    USERNAME="$existing_username"
  fi
fi
chmod 600 "$ENV_FILE"

if [[ ! -f "$AUTH_FILE" ]]; then
  printf 'Created the first dashboard login. Save this temporary password now:\n'
  PYTHONPATH="$HERMES_HOME/plugins" "$PYTHON_BIN" -m cortex \
    --db "$HERMES_HOME/cortex/cortex.db" \
    dashboard-password --username "$USERNAME" --auth-file "$AUTH_FILE"
fi

if grep -q '^CORTEX_DASHBOARD_PASSWORD=' "$ENV_FILE"; then
  env_tmp="$(mktemp)"
  sed '/^CORTEX_DASHBOARD_PASSWORD=/d' "$ENV_FILE" >"$env_tmp"
  chmod 600 "$env_tmp"
  mv "$env_tmp" "$ENV_FILE"
fi

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
printf 'The password hash is stored in %s (mode 600).\n' "$AUTH_FILE"
printf 'Run `python -m cortex dashboard-password` and restart the service if access is lost.\n'
