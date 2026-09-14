"""A purged memory must not poison usage resolution for the rest of the task.

Regression (2026-09-14): ``memory_context_outcomes`` carries a real foreign key
on ``memory_id``, while a recorded trace can still list an id whose memory was
hard-deleted (the operator's documented purge path). The outcome insert for the
missing id then aborted the whole resolution transaction with
``FOREIGN KEY constraint failed`` — permanently, because the trace never
changes — so the surviving memory never received its attributed outcome and
every later ``resolve_usage`` for that task raised.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.store import CortexStore


class PurgeResilienceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "cortex.db"
        self.store = CortexStore(self.db)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _seed_task(self) -> tuple[str, str, str]:
        first, _ = self.store.add_memory(
            "Purge resilience memory one backed the staged operational task."
        )
        second, _ = self.store.add_memory(
            "Purge resilience memory two backed the staged operational task."
        )
        task_id = self.store.create_usage_batch(
            [(first, 0.8), (second, 0.6)],
            query="Which purge resilience memory applies?",
            session_id="purge-session",
            task_type="ops",
            recall_mode="focused",
        )
        self.store.record_memory_trace_decision(
            task_id=task_id,
            session_id="purge-session",
            goal="Follow the purge resilience checklist",
            context_summary="task_type=ops",
            task_type="ops",
            recall_mode="focused",
            retrieval_used=True,
            retrieval_reason="both memories influenced the task",
            queries=["Which purge resilience memory applies?"],
            candidate_memories=[
                {
                    "memory_id": first,
                    "kind": "semantic",
                    "selected": True,
                    "score": 0.8,
                    "components": {},
                    "reason": "selected",
                },
                {
                    "memory_id": second,
                    "kind": "semantic",
                    "selected": True,
                    "score": 0.7,
                    "components": {},
                    "reason": "selected",
                },
            ],
        )
        return task_id, first, second

    def test_resolution_survives_a_purged_candidate(self) -> None:
        task_id, purged_id, survivor_id = self._seed_task()

        # The operator's documented purge: hard-delete one memory on a raw
        # connection (no store API exists for hard deletes).
        raw = sqlite3.connect(self.db)
        try:
            raw.execute("DELETE FROM memories WHERE id=?", (purged_id,))
            raw.commit()
        finally:
            raw.close()

        # Pre-fix this raised sqlite3.IntegrityError (FOREIGN KEY constraint
        # failed), leaving the surviving memory unlabeled forever.
        self.store.resolve_usage(task_id, {survivor_id: 1.0})

        with self.store._lock:
            outcomes = {
                str(row["memory_id"]): str(row["outcome"])
                for row in self.store._conn.execute(
                    "SELECT memory_id, outcome FROM memory_context_outcomes WHERE task_id=?",
                    (task_id,),
                ).fetchall()
            }
        self.assertNotIn(purged_id, outcomes, "no outcome row may be written for a purged memory")
        self.assertEqual(outcomes.get(survivor_id), "used")

        with self.store._lock:
            usage = {
                str(row["memory_id"]): int(row["used"])
                for row in self.store._conn.execute(
                    "SELECT memory_id, used FROM usage_records WHERE task_id=?", (task_id,)
                ).fetchall()
            }
        self.assertEqual(usage.get(survivor_id), 1)

    def test_second_resolution_is_not_poisoned(self) -> None:
        task_id, purged_id, survivor_id = self._seed_task()
        raw = sqlite3.connect(self.db)
        try:
            raw.execute("DELETE FROM memories WHERE id=?", (purged_id,))
            raw.commit()
        finally:
            raw.close()

        self.store.resolve_usage(task_id, {survivor_id: 1.0})
        # Re-resolving the same task must not raise: the purged id is skipped
        # every time, not just on the first attempt.
        self.store.resolve_usage(task_id, {survivor_id: 1.0})


if __name__ == "__main__":
    unittest.main()
