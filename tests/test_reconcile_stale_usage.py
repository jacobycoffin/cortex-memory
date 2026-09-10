"""Regression tests for the back-ported ``reconcile_stale_usage`` hotfix.

Two coupled fixes are covered here:

* ``CortexStore.reconcile_stale_usage`` (store.py) — resolves orphaned
  ``usage_records`` rows left ``pending`` by interrupted processes, and
  abandons matching stale ``agent_task_observations`` rows.
* The ``CortexMemoryProvider.on_session_end`` wiring (``__init__.py``) — the
  session-end hook invokes ``reconcile_stale_usage`` by default (config flag
  ``reconcile_stale_usage`` absent or truthy) and skips it when set false.

Both files were previously patched only in the installed plugin; these tests
pin the behaviour into the repository tree.
"""

from __future__ import annotations

import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (bootstraps the ``cortex`` package)

from cortex import CortexMemoryProvider
from cortex.research import complete_agent_tasks, record_agent_task_start
from cortex.store import CortexStore


SESSION_ID = "session-reconcile-fixture"
QUERY = "regression fixture query for reconcile_stale_usage"


def _ago(hours: float) -> str:
    """An ISO-8601 timestamp ``hours`` in the past (UTC)."""

    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="milliseconds")


class _RecordingStore:
    """Minimal stand-in for ``CortexStore`` that records session-end calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def maintenance(self, **kwargs) -> None:  # noqa: ANN003 - signature mirror
        self.calls.append("maintenance")

    def reconcile_stale_usage(self, *args, **kwargs) -> dict:
        self.calls.append("reconcile_stale_usage")
        return {"orphaned_tasks": 0, "resolved_usage_rows": 0}

    def consolidate(self, *args, **kwargs) -> None:
        self.calls.append("consolidate")


class ReconcileStaleUsageStoreTests(unittest.TestCase):
    """Behavioural tests for ``CortexStore.reconcile_stale_usage``."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.memory_id, _ = self.store.add_memory("Cortex reconcile fixture memory one.")
        self.memory_id_2, _ = self.store.add_memory("Cortex reconcile fixture memory two.")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    # -- fixture helpers -------------------------------------------------

    def _seed_usage(self, task_id: str, *, age_hours: float, memory_ids: list[str] | None = None) -> str:
        """Create realistic pending ``usage_records`` rows for ``task_id``."""

        items = [(mid, 0.5) for mid in (memory_ids or [self.memory_id])]
        self.store.create_usage_batch(
            items,
            query=QUERY,
            session_id=SESSION_ID,
            task_type="general",
            recall_mode="adaptive",
            task_id=task_id,
        )
        self._age_usage(task_id, age_hours)
        return task_id

    def _age_usage(self, task_id: str, age_hours: float) -> None:
        with self.store._lock:
            self.store._conn.execute(
                "UPDATE usage_records SET created_at=? WHERE task_id=?",
                (_ago(age_hours), task_id),
            )
            self.store._conn.commit()

    def _seed_observation(
        self,
        task_id: str,
        *,
        age_hours: float | None = None,
        complete: bool = False,
    ) -> None:
        """Create a pending ``agent_task_observations`` row for ``task_id``."""

        record_agent_task_start(
            self.store,
            task_id=task_id,
            session_id=SESSION_ID,
            task_type="general",
            query=QUERY,
            recall_condition="adaptive",
            recall_mode="adaptive",
            memory_count=1,
            context_tokens=10,
            prepare_ms=1.0,
        )
        if age_hours is not None:
            with self.store._lock:
                self.store._conn.execute(
                    "UPDATE agent_task_observations SET started_at=? WHERE task_id=?",
                    (_ago(age_hours), task_id),
                )
                self.store._conn.commit()
        if complete:
            complete_agent_tasks(
                self.store, [task_id], response_ms=1.0, tool_calls=0, tool_successes=0
            )

    def _usage_outcomes(self, task_id: str) -> list[str]:
        with self.store._lock:
            rows = self.store._conn.execute(
                "SELECT outcome FROM usage_records WHERE task_id=? ORDER BY usage_id",
                (task_id,),
            ).fetchall()
        return [str(row["outcome"]) for row in rows]

    def _observation(self, task_id: str):
        with self.store._lock:
            return self.store._conn.execute(
                "SELECT outcome, completed_at FROM agent_task_observations WHERE task_id=?",
                (task_id,),
            ).fetchone()

    def _pending_usage_task_ids(self) -> set[str]:
        with self.store._lock:
            rows = self.store._conn.execute(
                "SELECT DISTINCT task_id FROM usage_records WHERE outcome='pending'"
            ).fetchall()
        return {str(row["task_id"]) for row in rows}

    # -- tests -----------------------------------------------------------

    def test_genuine_orphan_usage_row_is_resolved(self) -> None:
        """An old pending usage row with no completed observation is orphaned."""

        task_id = str(uuid.uuid4())
        self._seed_usage(task_id, age_hours=48)
        self._seed_observation(task_id, age_hours=48)
        self.assertEqual(self._usage_outcomes(task_id), ["pending"])

        result = self.store.reconcile_stale_usage()

        self.assertEqual(result["orphaned_tasks"], 1)
        self.assertEqual(result["resolved_usage_rows"], 1)
        outcomes = self._usage_outcomes(task_id)
        self.assertEqual(outcomes, ["ignored"])
        self.assertNotIn("pending", outcomes)

    def test_pending_usage_with_completed_observation_is_left_pending(self) -> None:
        """The key negative case: a completed task is not an orphan."""

        task_id = str(uuid.uuid4())
        self._seed_usage(task_id, age_hours=48)
        self._seed_observation(task_id, age_hours=48, complete=True)
        observation = self._observation(task_id)
        self.assertIsNotNone(observation["completed_at"])

        result = self.store.reconcile_stale_usage()

        self.assertEqual(result["orphaned_tasks"], 0)
        self.assertEqual(result["resolved_usage_rows"], 0)
        self.assertEqual(self._usage_outcomes(task_id), ["pending"])

    def test_recent_pending_usage_is_left_pending(self) -> None:
        """A pending usage row younger than ``stale_hours`` is not touched."""

        task_id = str(uuid.uuid4())
        self._seed_usage(task_id, age_hours=1)

        result = self.store.reconcile_stale_usage()

        self.assertEqual(result["orphaned_tasks"], 0)
        self.assertEqual(result["resolved_usage_rows"], 0)
        self.assertEqual(self._usage_outcomes(task_id), ["pending"])

    def test_stale_pending_observation_is_abandoned(self) -> None:
        """Old pending observations are marked ignored with a completed_at."""

        task_id = str(uuid.uuid4())
        self._seed_observation(task_id, age_hours=48)
        before = self._observation(task_id)
        self.assertEqual(str(before["outcome"]), "pending")
        self.assertIsNone(before["completed_at"])

        self.store.reconcile_stale_usage()

        after = self._observation(task_id)
        self.assertEqual(str(after["outcome"]), "ignored")
        self.assertIsNotNone(after["completed_at"])

    def test_recent_pending_observation_is_untouched(self) -> None:
        """Observations younger than ``stale_hours`` stay pending."""

        task_id = str(uuid.uuid4())
        self._seed_observation(task_id, age_hours=1)

        self.store.reconcile_stale_usage()

        after = self._observation(task_id)
        self.assertEqual(str(after["outcome"]), "pending")
        self.assertIsNone(after["completed_at"])

    def test_mixed_fixture_returns_expected_shape_and_counts(self) -> None:
        """Returned dict shape and counts over a realistic mixed population."""

        orphan_multi = str(uuid.uuid4())  # 2 usage rows, no completed observation
        self._seed_usage(orphan_multi, age_hours=48, memory_ids=[self.memory_id, self.memory_id_2])
        orphan_single = str(uuid.uuid4())  # 1 usage row, no completed observation
        self._seed_usage(orphan_single, age_hours=72)
        resolved_task = str(uuid.uuid4())  # completed task: must be kept
        self._seed_usage(resolved_task, age_hours=48)
        self._seed_observation(resolved_task, age_hours=48, complete=True)
        recent_task = str(uuid.uuid4())  # too young: must be kept
        self._seed_usage(recent_task, age_hours=1)
        abandoned_observation = str(uuid.uuid4())  # stale observation, no usage rows
        self._seed_observation(abandoned_observation, age_hours=96)

        result = self.store.reconcile_stale_usage()

        self.assertEqual(set(result.keys()), {"orphaned_tasks", "resolved_usage_rows"})
        self.assertIsInstance(result["orphaned_tasks"], int)
        self.assertIsInstance(result["resolved_usage_rows"], int)
        self.assertEqual(result["orphaned_tasks"], 2)
        self.assertEqual(result["resolved_usage_rows"], 3)
        self.assertEqual(self._usage_outcomes(orphan_multi), ["ignored", "ignored"])
        self.assertEqual(self._usage_outcomes(orphan_single), ["ignored"])
        self.assertEqual(self._usage_outcomes(resolved_task), ["pending"])
        self.assertEqual(self._usage_outcomes(recent_task), ["pending"])
        observation = self._observation(abandoned_observation)
        self.assertEqual(str(observation["outcome"]), "ignored")
        self.assertIsNotNone(observation["completed_at"])

    def test_limit_caps_selected_orphan_tasks(self) -> None:
        """``limit`` bounds how many orphaned task groups are reconciled."""

        task_ids = [str(uuid.uuid4()) for _ in range(3)]
        for task_id in task_ids:
            self._seed_usage(task_id, age_hours=48)

        first = self.store.reconcile_stale_usage(limit=2)
        self.assertEqual(first["orphaned_tasks"], 2)
        self.assertEqual(len(self._pending_usage_task_ids()), 1)

        second = self.store.reconcile_stale_usage(limit=2)
        self.assertEqual(second["orphaned_tasks"], 1)
        self.assertEqual(self._pending_usage_task_ids(), set())

    def test_second_call_resolves_nothing(self) -> None:
        """Idempotence: a repeat pass finds nothing left to reconcile."""

        task_id = str(uuid.uuid4())
        self._seed_usage(task_id, age_hours=48)
        self._seed_observation(task_id, age_hours=48)

        first = self.store.reconcile_stale_usage()
        self.assertEqual(first["orphaned_tasks"], 1)
        self.assertEqual(first["resolved_usage_rows"], 1)

        second = self.store.reconcile_stale_usage()
        self.assertEqual(second, {"orphaned_tasks": 0, "resolved_usage_rows": 0})
        self.assertEqual(self._usage_outcomes(task_id), ["ignored"])


class SessionEndReconcileWiringTests(unittest.TestCase):
    """Behavioural tests for the session-end hook that drives the pass."""

    def _run_session_end(self, overrides: dict | None) -> list[str]:
        provider = CortexMemoryProvider()
        provider._config = dict(provider._config)
        if overrides is not None:
            provider._config.update(overrides)
        recorder = _RecordingStore()
        provider._store = recorder  # type: ignore[assignment]
        provider.on_session_end([])
        return recorder.calls

    def test_session_end_calls_reconcile_by_default(self) -> None:
        calls = self._run_session_end(None)
        self.assertIn("reconcile_stale_usage", calls)
        # The surrounding session-end work still runs.
        self.assertIn("maintenance", calls)
        self.assertIn("consolidate", calls)

    def test_session_end_calls_reconcile_when_flag_true(self) -> None:
        calls = self._run_session_end({"reconcile_stale_usage": True})
        self.assertIn("reconcile_stale_usage", calls)

    def test_session_end_skips_reconcile_when_flag_false(self) -> None:
        calls = self._run_session_end({"reconcile_stale_usage": False})
        self.assertNotIn("reconcile_stale_usage", calls)
        self.assertIn("maintenance", calls)
        self.assertIn("consolidate", calls)


if __name__ == "__main__":
    unittest.main()
