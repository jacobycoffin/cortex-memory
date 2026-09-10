"""v0 reconsolidation trace machinery tests (Governing Principle §0).

Covers: additive schema (strength/lability_until/access_history), strength
delta model, shadow-vs-apply trace updates, lability windows, decay, and the
access-history dashboard read.
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT

from cortex.store import CortexStore, utc_now, utc_now_dt


def make_store() -> tuple[CortexStore, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory()
    return CortexStore(Path(tmp.name) / "cortex.db"), tmp


class TraceV0Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.tmp = make_store()
        self.memory_id, _ = self.store.add_memory(
            "Kaya's verification standard: every claim is backed by real tool output.",
            kind="preference",
            confidence=0.9,
            importance=0.9,
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _columns(self) -> set[str]:
        rows = self.store._conn.execute("PRAGMA table_info(memories)").fetchall()
        return {r[1] for r in rows}

    def _tables(self) -> set[str]:
        rows = self.store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        return {r[0] for r in rows}

    def test_migration_adds_trace_columns_and_table(self) -> None:
        cols = self._columns()
        self.assertIn("strength", cols)
        self.assertIn("lability_until", cols)
        self.assertIn("access_history", self._tables())
        # Default strength is 1.0
        row = self.store._conn.execute(
            "SELECT strength FROM memories WHERE id=?", (self.memory_id,)
        ).fetchone()
        self.assertEqual(row["strength"], 1.0)

    def test_migration_is_idempotent_on_reopen(self) -> None:
        # Reopen the same DB — schema block + _migrate_columns must not fail.
        db_path = Path(self.tmp.name) / "cortex.db"
        self.store.close()
        reopened = CortexStore(db_path)
        try:
            cols = {r[1] for r in reopened._conn.execute("PRAGMA table_info(memories)").fetchall()}
            self.assertIn("strength", cols)
            tables = {r[0] for r in reopened._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            self.assertIn("access_history", tables)
        finally:
            reopened.close()

    def test_strength_deltas(self) -> None:
        self.assertEqual(self.store.strength_delta("retrieved"), 0.15)
        self.assertEqual(self.store.strength_delta("helpful"), 0.30)
        self.assertEqual(self.store.strength_delta("wrong"), -0.50)
        self.assertEqual(self.store.strength_delta("irrelevant"), -0.50)
        self.assertEqual(self.store.strength_delta("nonsense"), 0.0)

    def test_shadow_trace_logs_without_mutation(self) -> None:
        result = self.store.record_access_trace(
            self.memory_id, event="helpful", task_type="process_review",
            context_summary="reviewed the verified-outcomes process",
        )
        self.assertEqual(result["shadow"], 1)
        self.assertEqual(result["strength_after"], 1.3)
        # memory strength unchanged in shadow mode
        row = self.store._conn.execute(
            "SELECT strength FROM memories WHERE id=?", (self.memory_id,)
        ).fetchone()
        self.assertEqual(row["strength"], 1.0)
        # history row written with context framing
        hist = self.store.access_history(self.memory_id)
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["shadow"], 1)
        self.assertEqual(hist[0]["task_type"], "process_review")
        self.assertEqual(hist[0]["context_summary"], "reviewed the verified-outcomes process")
        self.assertEqual(hist[0]["strength_before"], 1.0)
        self.assertEqual(hist[0]["strength_after"], 1.3)

    def test_apply_trace_mutates_strength(self) -> None:
        result = self.store.record_access_trace(self.memory_id, event="helpful", apply=True)
        self.assertEqual(result["shadow"], 0)
        row = self.store._conn.execute(
            "SELECT strength FROM memories WHERE id=?", (self.memory_id,)
        ).fetchone()
        self.assertAlmostEqual(row["strength"], 1.3, places=4)
        hist = self.store.access_history(self.memory_id)
        self.assertEqual(hist[0]["shadow"], 0)

    def test_lability_window_and_candidates(self) -> None:
        result = self.store.record_access_trace(
            self.memory_id, event="wrong", apply=True, lability_minutes=30
        )
        self.assertIsNotNone(result["lability_until"])
        cands = self.store.pending_reconsolidation_candidates()
        self.assertTrue(any(c["id"] == self.memory_id for c in cands))
        # Memory inside window exposes strength
        self.assertIn("strength", cands[0])

    def test_lability_not_opened_on_positive_delta(self) -> None:
        result = self.store.record_access_trace(
            self.memory_id, event="helpful", apply=True, lability_minutes=30
        )
        self.assertIsNone(result["lability_until"])

    def test_decay_over_time(self) -> None:
        past = (utc_now_dt() - timedelta(days=10)).isoformat(timespec="milliseconds")
        self.store._conn.execute(
            "UPDATE memories SET created_at=? WHERE id=?", (past, self.memory_id)
        )
        self.store._conn.commit()
        future = (utc_now_dt() + timedelta(days=10)).isoformat(timespec="milliseconds")
        computed = self.store.compute_strength(self.memory_id, at=future)
        self.assertIsNotNone(computed)
        # 1.0 * 0.995^20 (10 days before + 10 days after) ≈ 0.905
        expected = 1.0 * (0.995 ** 20)
        self.assertAlmostEqual(computed["strength"], expected, places=3)

    def test_unknown_memory_returns_error(self) -> None:
        result = self.store.record_access_trace("missing-memory-id")
        self.assertIn("error", result)

    def test_access_history_limit_and_filter(self) -> None:
        other_id, _ = self.store.add_memory("A second memory for history filtering.")
        for _ in range(3):
            self.store.record_access_trace(self.memory_id, event="retrieved")
        self.store.record_access_trace(other_id, event="retrieved")
        all_rows = self.store.access_history(limit=10)
        self.assertEqual(len(all_rows), 4)
        only_first = self.store.access_history(self.memory_id, limit=10)
        self.assertEqual(len(only_first), 3)
        limited = self.store.access_history(self.memory_id, limit=2)
        self.assertEqual(len(limited), 2)


if __name__ == "__main__":
    unittest.main()
