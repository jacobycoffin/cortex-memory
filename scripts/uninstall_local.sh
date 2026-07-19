#!/usr/bin/env bash
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
TARGET="$HERMES_HOME/plugins/cortex"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

if command -v systemctl >/dev/null 2>&1; then
  if ! systemctl --user disable --now cortex-auto-judge.timer 2>/dev/null; then
    if systemctl --user is-active --quiet cortex-auto-judge.timer 2>/dev/null; then
      printf 'Refusing to remove Cortex while cortex-auto-judge.timer is still active.\n' >&2
      exit 1
    fi
  fi
  if systemctl --user is-active --quiet cortex-auto-judge.service 2>/dev/null; then
    if ! systemctl --user stop cortex-auto-judge.service; then
      printf 'Refusing to remove Cortex while cortex-auto-judge.service is still running.\n' >&2
      exit 1
    fi
  fi
  systemctl --user disable --now \
    cortex-dashboard.service \
    cortex-sleep.timer \
    cortex-vault-index.timer 2>/dev/null || true
fi
rm -f \
  "$UNIT_DIR/cortex-dashboard.service" \
  "$UNIT_DIR/cortex-auto-judge.service" \
  "$UNIT_DIR/cortex-auto-judge.timer" \
  "$UNIT_DIR/cortex-sleep.service" \
  "$UNIT_DIR/cortex-sleep.timer" \
  "$UNIT_DIR/cortex-vault-index.service" \
  "$UNIT_DIR/cortex-vault-index.timer"
if command -v systemctl >/dev/null 2>&1; then
  systemctl --user daemon-reload 2>/dev/null || true
fi

rm -rf "$TARGET"
printf 'Removed Cortex plugin code from %s\n' "$TARGET"
printf 'Preserved memory database and dashboard credentials under %s/cortex\n' "$HERMES_HOME"
printf 'Select another provider with: hermes memory setup\n'
