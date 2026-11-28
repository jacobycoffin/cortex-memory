"""Procedures must learn from the tasks that used them.

`procedure_outcomes` is the only evidence the confidence model reads, and on the live store it did
not exist: 318 tasks had used a procedure and carried a helpful/harmful label, while the table those
labels were supposed to feed had never been created. Two things made that possible:

1. **A second grading path existed and skipped the bridge.** `apply_task_outcome` graded the
   procedures a task used; `label_task_outcome` (the dashboard's path) wrote the same label to the
   same rows and never called it.
2. **Nothing reconciled.** A label written by a path that forgot is invisible forever - the numbers
   stay at `not measurable yet` and read as "nothing has happened yet" rather than "something is
   broken".

These tests pin both: the label path grades, and the reconciler derives runs from outcomes already
on disk without ever double-counting them.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import procedure_confidence
from cortex.cli import main as cli_main
from cortex.store import CortexStore


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "cortex.db"
        self.store = CortexStore(self.db)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def cli(self, *argv: str) -> tuple[int, str]:
        output = io.StringIO()
        with patch.object(sys, "argv", ["cortex-memory", "--db", str(self.db), *argv]):
            with redirect_stdout(output):
                try:
                    code = cli_main()
                except SystemExit as exc:
                    code = int(exc.code or 0)
        return code, output.getvalue()

    def procedure(self, text: str) -> str:
        memory_id, _ = self.store.add_memory(text, kind="procedure", source_category="TOOL_VERIFIED")
        return memory_id

    def record_use(self, task_id: str, memory_id: str, *, outcome: str,
                   at: str = "2026-11-26T12:00:00+00:00") -> None:
        with self.store._lock:
            self.store._conn.execute(
                """INSERT INTO usage_records(usage_id,task_id,memory_id,created_at,used,outcome,resolved_at)
                   VALUES(?,?,?,?,1,?,?)""",
                (f"u-{task_id}-{memory_id}", task_id, memory_id, at, outcome, at),
            )
            self.store._conn.commit()

    def outcomes(self) -> list[tuple[str, str]]:
        with self.store._lock:
            try:
                rows = self.store._conn.execute(
                    "SELECT procedure_id, outcome FROM procedure_outcomes ORDER BY procedure_id"
                ).fetchall()
            except Exception:
                return []
        return [(row[0], row[1]) for row in rows]

    def table_exists(self) -> bool:
        with self.store._lock:
            return bool(
                self.store._conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='procedure_outcomes'"
                ).fetchone()
            )


class ReconcilerTests(_Base):
    def test_helpful_task_grades_its_procedure_as_success(self) -> None:
        procedure = self.procedure("Restart the tunnel before blaming DNS.")
        self.record_use("task-1", procedure, outcome="helpful")
        result = procedure_confidence.backfill_from_usage(self.store, apply=True)
        self.assertTrue(result["ok"])
        self.assertEqual(result["newly_recorded"], 1)
        self.assertEqual(self.outcomes(), [(procedure, procedure_confidence.SUCCESS)])

    def test_harmful_task_grades_its_procedure_as_failure(self) -> None:
        procedure = self.procedure("Delete the WAL file to fix a lock.")
        self.record_use("task-2", procedure, outcome="harmful")
        procedure_confidence.backfill_from_usage(self.store, apply=True)
        self.assertEqual(self.outcomes(), [(procedure, procedure_confidence.FAILURE)])

    def test_a_second_pass_writes_nothing(self) -> None:
        """Idempotent on the usage row's own timestamp, not on the clock."""
        procedure = self.procedure("Snapshot before applying a nightly cycle.")
        self.record_use("task-3", procedure, outcome="validated")
        first = procedure_confidence.backfill_from_usage(self.store, apply=True)
        second = procedure_confidence.backfill_from_usage(self.store, apply=True)
        self.assertEqual(first["newly_recorded"], 1)
        self.assertEqual(second["newly_recorded"], 0)
        self.assertEqual(second["already_recorded"], 1)
        self.assertEqual(len(self.outcomes()), 1)

    def test_dry_run_writes_nothing(self) -> None:
        procedure = self.procedure("Check the port before installing the service.")
        self.record_use("task-4", procedure, outcome="helpful")
        result = procedure_confidence.backfill_from_usage(self.store, apply=False)
        self.assertEqual(result["newly_recorded"], 1)  # would record
        self.assertFalse(result["applied"])
        self.assertEqual(self.outcomes(), [])

    def test_ungraded_and_unused_rows_are_ignored(self) -> None:
        procedure = self.procedure("Never batch diverging targets.")
        self.record_use("task-5", procedure, outcome="used")  # not a grade
        result = procedure_confidence.backfill_from_usage(self.store, apply=True)
        self.assertEqual(result["candidates"], 0)
        self.assertEqual(self.outcomes(), [])

    def test_non_procedure_memories_are_ignored(self) -> None:
        memory_id, _ = self.store.add_memory("A plain fact.", kind="semantic",
                                             source_category="TOOL_VERIFIED")
        self.record_use("task-6", memory_id, outcome="helpful")
        result = procedure_confidence.backfill_from_usage(self.store, apply=True)
        self.assertEqual(result["candidates"], 0)

    def test_an_unreadable_timestamp_is_skipped_and_counted_not_guessed(self) -> None:
        """`created_at` is NOT NULL, so the reachable bad case is a value that will not parse.
        A run must never be anchored to an invented moment - it is skipped and reported."""
        procedure = self.procedure("Anchor the run to the usage row.")
        with self.store._lock:
            self.store._conn.execute(
                """INSERT INTO usage_records(usage_id,task_id,memory_id,created_at,used,outcome,resolved_at)
                   VALUES('u-bad','task-7',?,'not-a-timestamp',1,'helpful',NULL)""",
                (procedure,),
            )
            self.store._conn.commit()
        result = procedure_confidence.backfill_from_usage(self.store, apply=True)
        self.assertEqual(result["skipped_total"], 1)
        self.assertIn("unreadable timestamp", result["skipped"][0])
        self.assertEqual(self.outcomes(), [])


class LabelPathTests(_Base):
    def test_the_dashboard_label_path_grades_the_procedure_it_used(self) -> None:
        """`apply_task_outcome` always bridged to procedure grading; this path did not, so a
        procedure used in a dashboard-labelled task stayed untested forever."""
        procedure = self.procedure("Cut the branch from origin/main.")
        self.record_use("task-8", procedure, outcome="used")
        self.store.label_task_outcome("task-8", "helpful", actor="test-operator")
        self.assertEqual(self.outcomes(), [(procedure, procedure_confidence.SUCCESS)])

    def test_label_path_failure_to_grade_does_not_lose_the_label(self) -> None:
        procedure = self.procedure("A procedure that will not grade.")
        self.record_use("task-9", procedure, outcome="used")
        with patch.object(procedure_confidence, "record", side_effect=RuntimeError("boom")):
            result = self.store.label_task_outcome("task-9", "corrected", actor="test-operator")
        self.assertTrue(result["changed"])
        self.assertEqual(result["outcome"], "corrected")


class CliTests(_Base):
    def test_cli_dry_run_then_apply(self) -> None:
        procedure = self.procedure("Read the queue before it expires.")
        self.record_use("task-10", procedure, outcome="helpful")
        code, out = self.cli("procedures-sync")
        self.assertEqual(code, 0, out)
        payload = json.loads(out)
        self.assertFalse(payload["applied"])
        self.assertEqual(payload["newly_recorded"], 1)
        self.assertEqual(self.outcomes(), [])
        code, out = self.cli("procedures-sync", "--apply")
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads(out)["newly_recorded"], 1)
        self.assertEqual(len(self.outcomes()), 1)

    def test_status_moves_off_not_measurable_once_graded(self) -> None:
        from cortex import plain_summary

        procedure = self.procedure("Prove the numbers before reporting them.")
        self.record_use("task-11", procedure, outcome="helpful")
        before = next(
            line for line in plain_summary.status_lines(self.store._conn)
            if "procedures graded" in line
        )
        self.assertIn("not measurable yet", before)
        procedure_confidence.backfill_from_usage(self.store, apply=True)
        after = next(
            line for line in plain_summary.status_lines(self.store._conn)
            if "procedures graded" in line
        )
        self.assertNotIn("not measurable yet", after)
        self.assertIn("1", after)


if __name__ == "__main__":
    unittest.main()
