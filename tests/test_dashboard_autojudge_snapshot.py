"""Regression tests for the auto-judge snapshot payload (2026-09-14).

The dashboard's Auto-Judge view chart reads ``daily_decisions`` positionally
(``d[0]`` = day, ``d[1]`` = action, ``d[2]`` = count). The payload used to embed
raw ``sqlite3.Row`` objects, which are not JSON-serializable; the API's
``json.dumps(..., default=str)`` degraded each row to an opaque repr string, so
the chart read single characters ("<", "s", ...) instead of the actual values.

The snapshot now emits plain ``[day, action, count]`` triples, and the whole
payload round-trips through ``json.dumps`` without the ``default=str`` crutch.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.dashboard import _auto_judge_snapshot
from cortex.store import CortexStore


class AutoJudgeSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _insert_decision(self, action: str, created_at: str, *, actor: str = "cortex-auto-judge-test") -> None:
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO operator_review_decisions(
                       review_id, item_type, item_key, action, reason_code, actor, created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (str(uuid.uuid4()), "memory", "item", action, "test", actor, created_at),
            )

    def test_daily_decisions_are_positional_arrays(self) -> None:
        self._insert_decision("remember", "2026-09-13T10:00:00+00:00")
        self._insert_decision("remember", "2026-09-13T11:00:00+00:00")
        self._insert_decision("reject", "2026-09-13T12:00:00+00:00")
        self._insert_decision("remember", "2026-09-14T09:00:00+00:00")

        snapshot = _auto_judge_snapshot(self.store)
        daily = snapshot["daily_decisions"]
        self.assertIsInstance(daily, list)
        for item in daily:
            self.assertIsInstance(item, list, f"daily item is not an array: {item!r}")
            self.assertEqual(len(item), 3)
        counts = {(item[0], item[1]): item[2] for item in daily}
        self.assertEqual(counts[("2026-09-13", "remember")], 2)
        self.assertEqual(counts[("2026-09-13", "reject")], 1)
        self.assertEqual(counts[("2026-09-14", "remember")], 1)

    def test_old_decisions_fall_outside_the_window(self) -> None:
        self._insert_decision("remember", "2026-01-01T10:00:00+00:00")
        snapshot = _auto_judge_snapshot(self.store)
        self.assertEqual(snapshot["daily_decisions"], [])

    def test_snapshot_is_json_serializable_without_default(self) -> None:
        """The endpoint must not depend on default=str to serialize the payload."""
        self._insert_decision("remember", "2026-09-14T09:00:00+00:00")
        snapshot = _auto_judge_snapshot(self.store)
        try:
            encoded = json.dumps(snapshot)
        except TypeError as error:  # pragma: no cover - failure path
            self.fail(f"snapshot is not JSON-serializable without default=str: {error}")
        decoded = json.loads(encoded)
        self.assertIsInstance(decoded["daily_decisions"], list)
        for item in decoded["daily_decisions"]:
            self.assertIsInstance(item, list)

    def test_summary_counts_only_auto_judge_actors(self) -> None:
        self._insert_decision("remember", "2026-09-14T09:00:00+00:00")
        self._insert_decision("remember", "2026-09-14T09:05:00+00:00", actor="operator")
        snapshot = _auto_judge_snapshot(self.store)
        self.assertEqual(snapshot["summary"]["total"], 1)


if __name__ == "__main__":
    unittest.main()
