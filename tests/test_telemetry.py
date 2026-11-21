from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.store import CortexStore
from cortex.telemetry import OP_BUDGETS_MS, slow_path_report, trace_operation


class TelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "cortex.db"
        self.store = CortexStore(self.db)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _spans(self, op: str | None = None) -> list[sqlite3.Row]:
        sql = "SELECT op,duration_ms,ok,task_id,detail_json FROM operation_spans"
        params: tuple = ()
        if op:
            sql += " WHERE op=?"
            params = (op,)
        return self.store._conn.execute(sql + " ORDER BY rowid", params).fetchall()

    def test_write_is_traced(self) -> None:
        self.store.add_memory("The edge server runs the Hermes backend under ~/.hermes.", kind="operational")
        spans = self._spans("write")
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["ok"], 1)
        self.assertGreaterEqual(spans[0]["duration_ms"], 0.0)

    def test_recall_is_traced_with_mode_and_counts(self) -> None:
        self.store.record_recall_run(
            session_id="s1",
            query="edge server",
            mode="balanced",
            reason="test",
            requested_limit=6,
            token_budget=1200,
            candidate_count=11,
            selected_count=4,
            estimated_tokens=430,
            prepare_ms=42.5,
            abstained=False,
            task_id="task-1",
            stage_ms={"fts": 3.0, "scoring": 12.0},
        )
        spans = self._spans("recall")
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["task_id"], "task-1")
        self.assertAlmostEqual(spans[0]["duration_ms"], 42.5, places=3)
        self.assertIn('"mode":"balanced"', spans[0]["detail_json"])

    def test_trace_operation_records_success_run_id(self) -> None:
        @trace_operation("prune")
        def job(store: CortexStore) -> dict:
            return {"run_id": "run-abc", "mode": "shadow"}

        job(self.store)
        spans = self._spans("prune")
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["ok"], 1)
        self.assertIn("run-abc", spans[0]["detail_json"])

    def test_trace_operation_records_failure_and_reraises(self) -> None:
        @trace_operation("consolidate")
        def job(store: CortexStore) -> dict:
            raise RuntimeError("provider exploded")

        with self.assertRaises(RuntimeError):
            job(self.store)
        spans = self._spans("consolidate")
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["ok"], 0)

    def test_report_breaches_on_p95_not_on_one_outlier(self) -> None:
        # One 900ms outlier among 19 normal writes must not be a p95 breach: that
        # is the whole reason the alert reads p95 and not max.
        durations = [10.0] * 19 + [900.0]
        with self.store.transaction() as conn:
            for index, duration in enumerate(durations):
                conn.execute(
                    """INSERT INTO operation_spans(span_id,op,duration_ms,ok,detail_json,created_at)
                       VALUES(?,?,?,1,'{}',datetime('now'))""",
                    (f"span-{index}", "write", duration),
                )
        clean = slow_path_report(self.store._conn, window_hours=1)
        self.assertEqual(clean["breaches"], [])
        self.assertEqual(clean["ops"]["write"]["max_ms"], 900.0)
        tight = slow_path_report(
            self.store._conn, window_hours=1, budgets={"write": 5.0}
        )
        self.assertEqual(tight["breaches"], ["write"])
        self.assertEqual(tight["ops"]["write"]["count"], 20)

    def test_small_samples_need_a_gross_outlier_to_breach(self) -> None:
        # Three spans: p95 is just the slowest one, so a merely-over-budget call
        # must not page. A 3x outlier on the same budget must.
        with self.store.transaction() as conn:
            for index, duration in enumerate((200.0, 210.0, 300.0)):
                conn.execute(
                    """INSERT INTO operation_spans(span_id,op,duration_ms,ok,detail_json,created_at)
                       VALUES(?,?,?,1,'{}',datetime('now'))""",
                    (f"small-{index}", "write", duration),
                )
        near = slow_path_report(self.store._conn, window_hours=1)
        self.assertEqual(near["breaches"], [])
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO operation_spans(span_id,op,duration_ms,ok,detail_json,created_at)
                   VALUES('gross','write',9000.0,1,'{}',datetime('now'))"""
            )
        gross = slow_path_report(self.store._conn, window_hours=1)
        self.assertEqual(gross["breaches"], ["write"])
        self.assertEqual(gross["ops"]["write"]["breach"], "outlier")

    def test_report_flags_failed_spans(self) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO operation_spans(span_id,op,duration_ms,ok,detail_json,created_at)
                   VALUES('bad','prune',5.0,0,'{}',datetime('now'))"""
            )
        report = slow_path_report(self.store._conn, window_hours=1)
        self.assertEqual(report["ops"]["prune"]["errors"], 1)

    def test_report_survives_absent_table(self) -> None:
        legacy = sqlite3.connect(":memory:")
        report = slow_path_report(legacy, window_hours=1)
        self.assertIn("error", report)
        self.assertEqual(report["breaches"], [])

    def test_budgets_cover_every_traced_operation(self) -> None:
        self.assertEqual(set(OP_BUDGETS_MS), {"recall", "write", "prune", "consolidate"})


if __name__ == "__main__":
    unittest.main()
