"""A procedure that was used is also graded (#142 + #115 meeting #6).

``procedure_confidence`` computes a verdict from runs, but nothing wrote a run: a procedure could be recalled into a
hundred turns and still read ``untested``, because the outcome was recorded against the memory's general track record
while the procedure table stayed empty. These tests pin the bridge — one run per used procedure, mapped from the task
outcome, and written only for procedures that were genuinely used.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import procedure_confidence as pc
from cortex.store import CortexStore


class BridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, kind: str = "procedure", content: str = "Run the suite on the Mac with -n 4.") -> str:
        return self.store.add_memory(
            content=content, kind=kind, source_type="user_turn", source_category="USER_EXPLICIT",
        )[0]

    def graded_task(self, memory_id: str, outcome: str) -> str:
        """A real recall cycle: staged batch, then attribution, then the outcome label."""
        task_id = self.store.create_usage_batch(
            [(memory_id, 0.9)],
            query="how do i run the test suite",
            session_id="bridge-test",
            task_type="code",
            recall_mode="focused",
            requested_budget=600,
            estimated_tokens=540,
        )
        self.store.resolve_usage(task_id, {memory_id: 0.9})
        self.store.apply_task_outcome(task_id, outcome)
        return task_id

    def runs(self, procedure_id: str) -> int:
        graded = pc.confidence(self.store, procedure_id)
        return graded.successes + graded.failures

    # -- the loop closes ---------------------------------------------------------------

    def test_a_validated_task_records_a_successful_run(self) -> None:
        procedure_id = self.memory()
        self.assertEqual(self.runs(procedure_id), 0)

        self.graded_task(procedure_id, "validated")

        graded = pc.confidence(self.store, procedure_id)
        self.assertEqual(graded.successes, 1)
        self.assertEqual(graded.failures, 0)
        self.assertNotEqual(graded.verdict, "untested")

    def test_a_corrected_task_records_a_failed_run(self) -> None:
        procedure_id = self.memory()
        self.graded_task(procedure_id, "corrected")
        graded = pc.confidence(self.store, procedure_id)
        self.assertEqual(graded.successes, 0)
        self.assertEqual(graded.failures, 1)
        self.assertEqual(graded.verdict, "suspect")

    def test_helpful_and_harmful_follow_the_same_mapping(self) -> None:
        helpful = self.memory(content="Helpful path.")
        harmful = self.memory(content="Harmful path.")
        self.graded_task(helpful, "helpful")
        self.graded_task(harmful, "harmful")
        self.assertEqual(pc.confidence(self.store, helpful).successes, 1)
        self.assertEqual(pc.confidence(self.store, harmful).failures, 1)

    def test_four_successes_make_a_procedure_trusted(self) -> None:
        procedure_id = self.memory()
        base = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)
        original = pc._now
        ticks = {"count": 0}

        def ticking() -> datetime:
            ticks["count"] += 1
            return base + timedelta(minutes=ticks["count"])

        pc._now = ticking
        try:
            for _ in range(4):
                self.graded_task(procedure_id, "validated")
        finally:
            pc._now = original
        self.assertEqual(pc.confidence(self.store, procedure_id).verdict, "trusted")

    def test_two_runs_in_the_same_moment_are_one_run(self) -> None:
        """Record idsempotently on (procedure, outcome, moment); the bridge inherits that rather than inflating evidence."""
        procedure_id = self.memory()
        self.graded_task(procedure_id, "validated")
        self.graded_task(procedure_id, "validated")
        self.assertEqual(self.runs(procedure_id), 1)

    # -- and nothing else is graded ----------------------------------------------------

    def test_a_non_procedure_memory_is_not_graded_as_a_procedure(self) -> None:
        memory_id = self.memory(kind="semantic", content="The dashboard listens on port 8100.")
        self.graded_task(memory_id, "validated")
        self.assertEqual(self.runs(memory_id), 0)

    def test_a_procedure_that_was_not_used_is_not_graded(self) -> None:
        used = self.memory(content="Used procedure.")
        unused = self.memory(content="Unused procedure.")
        self.graded_task(used, "validated")
        self.assertEqual(self.runs(used), 1)
        self.assertEqual(self.runs(unused), 0)

    def test_an_unknown_procedure_id_cannot_manufacture_a_run(self) -> None:
        self.assertEqual(self.store._record_procedure_outcomes(["no-such-memory"], "validated", task_id="t"), [])

    def test_an_unmapped_outcome_records_nothing_and_does_not_raise(self) -> None:
        procedure_id = self.memory()
        self.assertEqual(self.store._record_procedure_outcomes([procedure_id], "probably fine", task_id="t"), [])
        self.assertEqual(self.runs(procedure_id), 0)

    def test_no_memories_no_runs(self) -> None:
        self.assertEqual(self.store._record_procedure_outcomes([], "validated", task_id="t"), [])

    # -- the label itself is never at risk ---------------------------------------------

    def test_the_outcome_is_still_applied_when_grading_fails(self) -> None:
        procedure_id = self.memory()
        task_id = self.store.create_usage_batch(
            [(procedure_id, 0.9)], query="how do i run the suite", session_id="bridge-test",
            task_type="code", recall_mode="focused", requested_budget=600, estimated_tokens=540,
        )
        self.store.resolve_usage(task_id, {procedure_id: 0.9})
        original = pc.record

        def explode(*_args, **_kwargs):
            raise RuntimeError("grading is unavailable")

        pc.record = explode
        try:
            self.store.apply_task_outcome(task_id, "validated")
        finally:
            pc.record = original

        row = self.store._conn.execute(
            "SELECT outcome FROM usage_records WHERE task_id=? AND used=1", (task_id,)
        ).fetchone()
        self.assertEqual(str(row["outcome"]), "validated")
        self.assertEqual(self.runs(procedure_id), 0)


if __name__ == "__main__":
    unittest.main()
