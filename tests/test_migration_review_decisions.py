"""Legacy operator_review_decisions shapes must migrate without losing rows.

Regression (2026-09-14): the schema-26 table rebuild copied rows with a bare
``SELECT *``. A pre-schema-17 database gets ``decision_scope`` APPENDED at
column 16 by ``ALTER TABLE``, while the rebuilt table declares it at 13 — so the
positional copy shifted actor/created_at/reversed_at and failed
``NOT NULL constraint failed: ...created_at`` for any database that had at least
one review row. Retrying could not fix it, and a failed attempt left a stray
``operator_review_decisions_v26`` table behind.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.store import CortexStore

# Pre-schema-17 shape: FK on proposal_id, decision_scope not yet present.
LEGACY_16_DDL = """
CREATE TABLE operator_review_decisions (
    review_id TEXT PRIMARY KEY,
    item_type TEXT NOT NULL,
    item_key TEXT NOT NULL,
    proposal_id TEXT REFERENCES memory_creation_proposals(proposal_id),
    src_id TEXT,
    dst_id TEXT,
    action TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    reason_text TEXT,
    prior_json TEXT NOT NULL DEFAULT '{}',
    effect_json TEXT NOT NULL DEFAULT '{}',
    learning_signal_json TEXT NOT NULL DEFAULT '{}',
    actor TEXT NOT NULL,
    created_at TEXT NOT NULL,
    reversed_at TEXT
)
"""

# Schema 17-25 shape: FK still present, decision_scope declared inline at 13.
LEGACY_17_25_DDL = """
CREATE TABLE operator_review_decisions (
    review_id TEXT PRIMARY KEY,
    item_type TEXT NOT NULL,
    item_key TEXT NOT NULL,
    proposal_id TEXT REFERENCES memory_creation_proposals(proposal_id),
    src_id TEXT,
    dst_id TEXT,
    action TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    reason_text TEXT,
    prior_json TEXT NOT NULL DEFAULT '{}',
    effect_json TEXT NOT NULL DEFAULT '{}',
    learning_signal_json TEXT NOT NULL DEFAULT '{}',
    decision_scope TEXT NOT NULL DEFAULT 'item_only'
      CHECK(decision_scope IN ('item_only','exact_duplicates','policy_evidence')),
    actor TEXT NOT NULL,
    created_at TEXT NOT NULL,
    reversed_at TEXT
)
"""

ROWS_16 = [
    ("r-1", "memory", "item-a", "remember", "test", "first", "{}", "{}", "{}", "cortex-auto-judge-v1", "2026-09-01T10:00:00+00:00", None),
    ("r-2", "memory", "item-b", "reject", "test", "second", "{}", "{}", "{}", "cortex-auto-judge-v1", "2026-09-02T11:00:00+00:00", "2026-09-03T09:00:00+00:00"),
]


class ReviewDecisionMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "cortex.db"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _craft_legacy(self, ddl: str, rows: list[tuple], *, with_scope: bool) -> None:
        store = CortexStore(self.db)
        store.close()
        conn = sqlite3.connect(self.db)
        try:
            conn.execute("DROP TABLE operator_review_decisions")
            conn.execute(ddl)
            if with_scope:
                conn.execute(
                    """INSERT INTO operator_review_decisions(
                           review_id, item_type, item_key, action, reason_code, reason_text,
                           decision_scope, actor, created_at)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    ("r-1", "memory", "item-a", "remember", "test", "first", "policy_evidence", "operator", "2026-09-01T10:00:00+00:00"),
                )
            else:
                conn.executemany(
                    """INSERT INTO operator_review_decisions(
                           review_id, item_type, item_key, action, reason_code, reason_text,
                           prior_json, effect_json, learning_signal_json, actor, created_at, reversed_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    rows,
                )
            conn.commit()
        finally:
            conn.close()

    def _read_rows(self) -> list[dict]:
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT review_id, actor, created_at, reversed_at, decision_scope "
                "FROM operator_review_decisions ORDER BY review_id"
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def test_pre_schema_17_rows_migrate_intact(self) -> None:
        self._craft_legacy(LEGACY_16_DDL, ROWS_16, with_scope=False)

        store = CortexStore(self.db)
        store.close()

        rows = self._read_rows()
        self.assertEqual([row["review_id"] for row in rows], ["r-1", "r-2"])
        self.assertEqual(rows[0]["actor"], "cortex-auto-judge-v1")
        self.assertEqual(rows[0]["created_at"], "2026-09-01T10:00:00+00:00")
        self.assertIsNone(rows[0]["reversed_at"])
        self.assertEqual(rows[1]["reversed_at"], "2026-09-03T09:00:00+00:00")
        # Pre-schema-17 decisions were training evidence before explicit reach
        # controls existed; migration preserves that meaning.
        self.assertEqual(rows[0]["decision_scope"], "policy_evidence")

        # A second open is a no-op and keeps the rows.
        store = CortexStore(self.db)
        store.close()
        self.assertEqual(len(self._read_rows()), 2)

    def test_schema_17_to_25_rows_migrate_intact(self) -> None:
        self._craft_legacy(LEGACY_17_25_DDL, [], with_scope=True)

        store = CortexStore(self.db)
        store.close()

        rows = self._read_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["actor"], "operator")
        self.assertEqual(rows[0]["created_at"], "2026-09-01T10:00:00+00:00")
        self.assertEqual(rows[0]["decision_scope"], "policy_evidence")

    def test_foreign_keys_are_restored_after_migration(self) -> None:
        self._craft_legacy(LEGACY_16_DDL, ROWS_16, with_scope=False)
        store = CortexStore(self.db)
        try:
            fk_state = store._conn.execute("PRAGMA foreign_keys").fetchone()[0]
            self.assertEqual(fk_state, 1, "the migration must restore foreign_keys=ON")
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
