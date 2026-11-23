"""A lesson is only learned if you can show the question it changes (#211)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import lesson_demo as ld
from cortex import recall_replay
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.dashboard = self.memory("The dashboard runs on port 8100.")
        self.staging = self.memory("The dashboard runs on port 8100 in staging.")
        self.unrelated = self.memory("The sourdough starter is fed at eight in the morning.")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def snapshot_to(self, path: Path) -> None:
        """Copy the live store safely while it is open (WAL files make a plain copy unusable)."""
        import sqlite3

        source = sqlite3.connect(str(Path(self.tmp.name) / "cortex.db"))
        destination = sqlite3.connect(str(path))
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()

    def turn(self, query: str, ids: list[str], task: str = "task-1") -> None:
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO recall_runs
                   (recall_id, task_id, session_id, query, mode, reason, requested_limit,
                    token_budget, created_at)
                   VALUES (?, ?, 's1', ?, 'adaptive', '', 6, 700, ?)""",
                (f"run-{task}", task, query, NOW.isoformat()),
            )
        recall_replay.record(
            self.store, task_id=task, block="a block", memory_ids=ids, session_id="s1"
        )
        # record() writes inside the caller's transaction by contract, so commit it here or a
        # snapshot taken straight afterwards will not see the row.
        self.store._conn.commit()

    def test_no_recorded_turns_means_nothing_to_show(self) -> None:
        self.assertEqual(ld.recorded(self.store), [])
        self.assertEqual(ld.demonstrations(self.store), [])

    def test_a_recorded_turn_is_read_back_with_its_query(self) -> None:
        self.turn("which port does the dashboard use", [self.dashboard])
        turns = ld.recorded(self.store)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["query"], "which port does the dashboard use")

    def test_a_turn_without_a_query_is_skipped(self) -> None:
        recall_replay.record(
            self.store, task_id="orphan", block="b", memory_ids=[], session_id="s1"
        )
        self.assertEqual(ld.recorded(self.store), [])

    def test_an_unchanged_answer_is_reported_as_unchanged(self) -> None:
        self.turn("the sourdough starter feeding time", [self.unrelated])
        demo = ld.compare(self.store, ld.recorded(self.store)[0])
        self.assertFalse(demo.changed)
        self.assertIn("unchanged", demo.describe(self.store))

    def test_a_changed_answer_names_what_arrived_and_left(self) -> None:
        # The stored turn claims the unrelated memory was injected for a dashboard question.
        self.turn("dashboard port 8100", [self.unrelated])
        demo = ld.compare(self.store, ld.recorded(self.store)[0])
        self.assertTrue(demo.changed)
        self.assertIn(self.unrelated, demo.left)
        self.assertTrue(demo.arrived)

    def test_the_sentence_shows_both_sides_with_content(self) -> None:
        self.turn("dashboard port 8100", [self.unrelated])
        sentence = ld.compare(self.store, ld.recorded(self.store)[0]).describe(self.store)
        self.assertIn('Q (drift, not a demonstration): "dashboard port 8100"', sentence)
        self.assertIn("left", sentence)
        self.assertIn("now", sentence)
        self.assertIn("sourdough", sentence)

    def test_only_changed_turns_are_listed_by_default(self) -> None:
        self.turn("the sourdough starter feeding time", [self.unrelated], task="t1")
        self.turn("dashboard port 8100", [self.unrelated], task="t2")
        demos = ld.demonstrations(self.store)
        self.assertTrue(all(demo.changed for demo in demos))

    def test_all_turns_can_be_listed_on_request(self) -> None:
        self.turn("the sourdough starter feeding time", [self.unrelated], task="t1")
        demos = ld.demonstrations(self.store, only_changed=False)
        self.assertEqual(len(demos), 1)

    def test_the_report_counts_compared_and_changed(self) -> None:
        self.turn("the sourdough starter feeding time", [self.unrelated], task="t1")
        self.turn("dashboard port 8100", [self.unrelated], task="t2")
        described = ld.report(self.store)
        self.assertEqual(described["turns"], 2)
        self.assertEqual(described["compared"], 2)
        self.assertLessEqual(described["changed"], 2)

    def test_the_dict_round_trips(self) -> None:
        self.turn("dashboard port 8100", [self.unrelated])
        payload = ld.compare(self.store, ld.recorded(self.store)[0]).as_dict(self.store)
        self.assertIn("sentence", payload)
        self.assertIn("arrived", payload)
        self.assertIsInstance(payload["changed"], bool)

    def test_a_baseline_comparison_with_no_change_reports_no_change(self) -> None:
        from shutil import copyfile

        self.turn("dashboard port 8100", [self.dashboard])
        before_path = Path(self.tmp.name) / "before.db"
        after_path = Path(self.tmp.name) / "after.db"
        self.snapshot_to(before_path)
        self.snapshot_to(after_path)
        before = CortexStore(before_path)
        after = CortexStore(after_path)
        try:
            turn = ld.recorded(after, limit=1)[0]
            demo = ld.compare(after, turn, baseline=before)
            self.assertEqual(demo.basis, "baseline")
            self.assertFalse(demo.changed)
        finally:
            before.close()
            after.close()

    def test_a_baseline_comparison_shows_a_recorded_near_miss(self) -> None:
        """The demonstration this item exists for: the same query, one change, one visible difference."""
        from shutil import copyfile

        from cortex import near_miss

        self.turn("dashboard port 8100", [self.dashboard])
        before_path = Path(self.tmp.name) / "before2.db"
        after_path = Path(self.tmp.name) / "after2.db"
        self.snapshot_to(before_path)
        self.snapshot_to(after_path)
        before = CortexStore(before_path)
        after = CortexStore(after_path)
        try:
            turn = ld.recorded(after, limit=1)[0]
            selected = ld._now_selection(before, turn["query"], limit=6,
                                         session_id=turn["session_id"], token_budget=700)
            self.assertGreaterEqual(len(selected), 2)
            near_miss.record(after, selected[0], selected[1], reason="test", at=NOW)
            demo = ld.compare(after, turn, baseline=before)
            self.assertEqual(demo.basis, "baseline")
            self.assertTrue(demo.changed)
            self.assertIn(selected[1], demo.left)
        finally:
            before.close()
            after.close()

    def test_the_drift_basis_is_labelled_in_the_sentence(self) -> None:
        self.turn("dashboard port 8100", [self.unrelated])
        sentence = ld.compare(self.store, ld.recorded(self.store)[0]).describe(self.store)
        self.assertIn("drift, not a demonstration", sentence)

    def test_turns_carry_the_limits_they_ran_with(self) -> None:
        self.turn("dashboard port 8100", [self.dashboard])
        turn = ld.recorded(self.store)[0]
        self.assertEqual(turn["limit"], 6)
        self.assertEqual(turn["token_budget"], 700)

    def test_an_empty_store_is_quiet(self) -> None:
        other = CortexStore(Path(self.tmp.name) / "empty.db")
        try:
            self.assertEqual(ld.report(other)["turns"], 0)
            self.assertEqual(ld.demonstrations(other), [])
        finally:
            other.close()


if __name__ == "__main__":
    unittest.main()
