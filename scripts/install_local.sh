#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
TARGET="$HERMES_HOME/plugins/cortex"
BACKUP_ROOT="$HERMES_HOME/backups"

if [[ -d "$TARGET" ]]; then
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$BACKUP_ROOT"
  cp -a "$TARGET" "$BACKUP_ROOT/cortex-plugin-$timestamp"
  printf 'Backed up the previous Cortex plugin to %s\n' "$BACKUP_ROOT/cortex-plugin-$timestamp"
fi

mkdir -p "$TARGET"
for file in __init__.py __main__.py attribution.py client.py cli.py cognition.py dashboard.py dashboard_auth.py dashboard.html favicon.svg favicon.ico apple-touch-icon.png extraction.py retrieval.py security.py semantics.py sleep.py store.py tooling.py vault.py plugin.yaml README.md LICENSE CHANGELOG.md; do
  cp "$SOURCE_DIR/$file" "$TARGET/$file"
done
mkdir -p "$TARGET/scripts"
cp "$SOURCE_DIR/scripts/cortex-vault-index.service.in" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/cortex-vault-index.timer" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/install_vault_timer.sh" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/cortex-dashboard.service.in" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/install_dashboard_service.sh" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/cortex-sleep.service.in" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/cortex-sleep.timer" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/install_sleep_timer.sh" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/uninstall_local.sh" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/benchmark.py" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/benchmark_compare.py" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/benchmark_e2e.py" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/benchmark_aggregate.py" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/benchmark_social_card.py" "$TARGET/scripts/"
cp "$SOURCE_DIR/scripts/benchmark_adaptive.py" "$TARGET/scripts/"
mkdir -p "$TARGET/benchmarks"
cp "$SOURCE_DIR/benchmarks/__init__.py" "$TARGET/benchmarks/"
cp "$SOURCE_DIR/benchmarks/core.py" "$TARGET/benchmarks/"
mkdir -p "$TARGET/docs"
cp "$SOURCE_DIR/docs/"*.md "$TARGET/docs/"
chmod +x "$TARGET/scripts/install_vault_timer.sh"
chmod +x "$TARGET/scripts/install_dashboard_service.sh"
chmod +x "$TARGET/scripts/install_sleep_timer.sh"
chmod +x "$TARGET/scripts/uninstall_local.sh"
chmod +x "$TARGET/scripts/benchmark.py"
chmod +x "$TARGET/scripts/benchmark_compare.py"
chmod +x "$TARGET/scripts/benchmark_e2e.py"
chmod +x "$TARGET/scripts/benchmark_aggregate.py"
chmod +x "$TARGET/scripts/benchmark_social_card.py"
chmod +x "$TARGET/scripts/benchmark_adaptive.py"

printf 'Installed Cortex to %s\n' "$TARGET"
printf 'Cortex is installed but not activated. Run: hermes memory setup\n'
