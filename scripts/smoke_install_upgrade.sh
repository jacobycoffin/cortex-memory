#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/cortex-install-smoke.XXXXXX")"

cleanup() {
  rm -rf "$TEMP_ROOT"
}
trap cleanup EXIT INT TERM

export HERMES_HOME="$TEMP_ROOT/hermes-home"
export CORTEX_INSTALL_AUTO_JUDGE_TIMER=0
PLUGIN_DIR="$HERMES_HOME/plugins/cortex"
DB_PATH="$HERMES_HOME/cortex/cortex.db"
LEGACY_DB_PATH="$HERMES_HOME/cortex/legacy.db"
MEMORY_ID_PATH="$HERMES_HOME/cortex/smoke-memory-id"
MEMORY_TEXT="Cortex install smoke sentinel survives an in-place upgrade."
UPGRADE_MARKER="previous Cortex installation"

fail() {
  printf 'install smoke failed: %s\n' "$1" >&2
  exit 1
}

printf 'Testing a clean install in %s\n' "$HERMES_HOME"
"$SOURCE_DIR/scripts/install_local.sh"

for relative_path in \
  __init__.py \
  __main__.py \
  adaptive_reconsolidation.py \
  adaptive_weights.py \
  autojudge.py \
  benchmarking.py \
  brain_mechanics.py \
  cli.py \
  client.py \
  cortex_schema.py \
  dashboard.py \
  dashboard_auth.py \
  dashboard.html \
  favicon.svg \
  favicon.ico \
  harness.py \
  apple-touch-icon.png \
  metacognition.py \
  plugin.yaml \
  refinery.py \
  relevance_pruning.py \
  review_copilot.py \
  schema_formation.py \
  semantic_consolidation.py \
  serializers.py \
  sleep.py \
  store.py \
  benchmarks/core.py \
  scripts/cortex-sleep.service.in \
  scripts/cortex-sleep.timer \
  scripts/cortex-auto-judge.service.in \
  scripts/cortex-auto-judge.timer \
  scripts/install_dashboard_service.sh \
  scripts/install_sleep_timer.sh \
  scripts/install_auto_judge_timer.sh \
  scripts/install_vault_timer.sh \
  docs/QUICKSTART.md
do
  [[ -f "$PLUGIN_DIR/$relative_path" ]] || fail "missing installed file: $relative_path"
done

# Completeness: every top-level runtime module in the source tree must ship
# in the installed plugin. A new extraction that forgets the installer breaks
# `import cortex.store` only in the installed copy — source-checkout tests
# never see it (serializers.py regression, 2026-09-09).
for module_file in "$SOURCE_DIR"/*.py; do
  module_name="$(basename "$module_file")"
  [[ -f "$PLUGIN_DIR/$module_name" ]] || fail "installed plugin missing runtime module: $module_name"
done

PYTHONPATH="$HERMES_HOME/plugins" "$PYTHON_BIN" -m cortex harness-contract --tool-name cortex_memory >/dev/null

for relative_path in \
  scripts/install_dashboard_service.sh \
  scripts/install_sleep_timer.sh \
  scripts/install_auto_judge_timer.sh \
  scripts/install_vault_timer.sh \
  scripts/uninstall_local.sh
do
  [[ -x "$PLUGIN_DIR/$relative_path" ]] || fail "installed script is not executable: $relative_path"
done

PYTHONPATH="$HERMES_HOME/plugins" "$PYTHON_BIN" - \
  "$DB_PATH" "$LEGACY_DB_PATH" "$MEMORY_ID_PATH" "$MEMORY_TEXT" <<'PY'
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import cortex
from cortex.store import SCHEMA_VERSION, CortexStore


db_path = Path(sys.argv[1])
legacy_db_path = Path(sys.argv[2])
memory_id_path = Path(sys.argv[3])
memory_text = sys.argv[4]
store = CortexStore(db_path)
try:
    memory_id, created = store.add_memory(
        memory_text,
        kind="operational",
        source_type="release_smoke",
    )
    assert created, "clean database unexpectedly contained the sentinel"
    assert store.get_memory(memory_id)["content"] == memory_text
    assert store.stats()["schema_version"] == SCHEMA_VERSION
    assert store.audit()["ok"] is True
    memory_id_path.write_text(memory_id, encoding="utf-8")
finally:
    store.close()

assert cortex.__file__ is not None
assert Path(cortex.__file__).resolve().parent.name == "cortex"

# Exercise the real prototype-to-current migration through the installed code.
conn = sqlite3.connect(legacy_db_path)
conn.executescript(
    """
    CREATE TABLE memories (
        id TEXT PRIMARY KEY, kind TEXT NOT NULL, content TEXT NOT NULL, content_hash TEXT NOT NULL,
        source_type TEXT NOT NULL, source_ref TEXT, session_id TEXT, created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL, valid_from TEXT, valid_to TEXT, confidence REAL NOT NULL,
        importance REAL NOT NULL, volatility REAL NOT NULL, trust REAL NOT NULL, state TEXT NOT NULL,
        pinned INTEGER NOT NULL, quarantine_reason TEXT, retrieved_count INTEGER NOT NULL,
        injected_count INTEGER NOT NULL, used_count INTEGER NOT NULL, success_count INTEGER NOT NULL,
        confirmed_count INTEGER NOT NULL, correction_count INTEGER NOT NULL,
        false_positive_count INTEGER NOT NULL, duplicate_count INTEGER NOT NULL,
        last_retrieved_at TEXT, last_injected_at TEXT, last_used_at TEXT
    );
    """
)
now = datetime.now(timezone.utc).isoformat()
conn.execute(
    "INSERT INTO memories VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
    (
        "legacy-smoke-id", "semantic", "Legacy install-smoke memory survives migration.",
        "legacy-smoke-hash", "conversation", None, "legacy-session", now, now,
        None, None, 0.7, 0.6, 0.4, 0.7, "active", 0, None, 0, 0, 0, 0, 0, 0,
        0, 0, None, None, None,
    ),
)
conn.commit()
conn.close()

migrated = CortexStore(legacy_db_path)
try:
    assert migrated.get_memory("legacy-smoke-id")["content"] == "Legacy install-smoke memory survives migration."
    assert migrated.stats()["schema_version"] == SCHEMA_VERSION
finally:
    migrated.close()
PY

PYTHONPATH="$HERMES_HOME/plugins" "$PYTHON_BIN" -m cortex --db "$DB_PATH" stats >/dev/null

printf '%s\n' "$UPGRADE_MARKER" >"$PLUGIN_DIR/CHANGELOG.md"
printf 'Testing an in-place reinstall and backup\n'
"$SOURCE_DIR/scripts/install_local.sh"

backup_count=0
marker_backup=""
for backup_dir in "$HERMES_HOME"/backups/cortex-plugin-*; do
  [[ -d "$backup_dir" ]] || continue
  backup_count=$((backup_count + 1))
  if [[ -f "$backup_dir/CHANGELOG.md" ]] && grep -qF "$UPGRADE_MARKER" "$backup_dir/CHANGELOG.md"; then
    marker_backup="$backup_dir"
  fi
done

[[ "$backup_count" -eq 1 ]] || fail "expected one plugin backup, found $backup_count"
[[ -n "$marker_backup" ]] || fail "the plugin backup did not preserve the previous installation"
cmp -s "$SOURCE_DIR/CHANGELOG.md" "$PLUGIN_DIR/CHANGELOG.md" || fail "reinstall did not restore current source files"

PYTHONPATH="$HERMES_HOME/plugins" "$PYTHON_BIN" - \
  "$DB_PATH" "$LEGACY_DB_PATH" "$MEMORY_ID_PATH" "$MEMORY_TEXT" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

from cortex.store import SCHEMA_VERSION, CortexStore


db_path = Path(sys.argv[1])
legacy_db_path = Path(sys.argv[2])
memory_id_path = Path(sys.argv[3])
memory_text = sys.argv[4]
memory_id = memory_id_path.read_text(encoding="utf-8").strip()
store = CortexStore(db_path)
try:
    assert store.get_memory(memory_id)["content"] == memory_text
    assert store.stats()["schema_version"] == SCHEMA_VERSION
    assert store.audit()["ok"] is True
finally:
    store.close()

migrated = CortexStore(legacy_db_path)
try:
    assert migrated.get_memory("legacy-smoke-id")["content"] == "Legacy install-smoke memory survives migration."
    assert migrated.stats()["schema_version"] == SCHEMA_VERSION
finally:
    migrated.close()
PY

printf 'Install and upgrade smoke test passed\n'
