#!/usr/bin/env bash
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
VAULT_PATH="${VAULT_PATH:-$HERMES_HOME/obsidian-vault}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/lib/hermes-agent/venv/bin/python}"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Python runtime not found or not executable: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d "$VAULT_PATH" ]]; then
  printf 'Vault directory not found: %s\n' "$VAULT_PATH" >&2
  exit 1
fi

mkdir -p "$UNIT_DIR" "$HERMES_HOME/cortex"

service_tmp="$(mktemp)"
timer_tmp="$(mktemp)"
trap 'rm -f "$service_tmp" "$timer_tmp"' EXIT

sed \
  -e "s|@@HERMES_HOME@@|$HERMES_HOME|g" \
  -e "s|@@VAULT_PATH@@|$VAULT_PATH|g" \
  -e "s|@@PYTHON_BIN@@|$PYTHON_BIN|g" \
  "$HERMES_HOME/plugins/cortex/scripts/cortex-vault-index.service.in" >"$service_tmp"
cp "$service_tmp" "$UNIT_DIR/cortex-vault-index.service"
cp "$HERMES_HOME/plugins/cortex/scripts/cortex-vault-index.timer" "$timer_tmp"
cp "$timer_tmp" "$UNIT_DIR/cortex-vault-index.timer"

systemctl --user daemon-reload
systemctl --user enable --now cortex-vault-index.timer
printf 'Enabled incremental Cortex vault indexing for %s\n' "$VAULT_PATH"
