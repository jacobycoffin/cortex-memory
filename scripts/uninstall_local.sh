#!/usr/bin/env bash
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
TARGET="$HERMES_HOME/plugins/cortex"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

if command -v systemctl >/dev/null 2>&1; then
  systemctl --user disable --now cortex-dashboard.service cortex-vault-index.timer 2>/dev/null || true
fi
rm -f \
  "$UNIT_DIR/cortex-dashboard.service" \
  "$UNIT_DIR/cortex-vault-index.service" \
  "$UNIT_DIR/cortex-vault-index.timer"
if command -v systemctl >/dev/null 2>&1; then
  systemctl --user daemon-reload 2>/dev/null || true
fi

rm -rf "$TARGET"
printf 'Removed Cortex plugin code from %s\n' "$TARGET"
printf 'Preserved memory database and dashboard credentials under %s/cortex\n' "$HERMES_HOME"
printf 'Select another provider with: hermes memory setup\n'
