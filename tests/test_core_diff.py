"""Core memory diff view: what changed in the always-on block, and when (#124)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.core_diff import diff, hash_block, history, latest, record
from cortex.promotion import demote, promote
from cortex.store import CortexStore


class SnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _memory(self, content: str, **kwargs) -> str:
        created = self.store.add_memory(
            content=content,
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            **kwargs,
        )
        return created[0] if isinstance(created, tuple) else created

    def test_record_writes_one_row(self) -> None:
        self.assertIsNone(latest(self.store))
        written = record(self.store)
        self.assertIsNotNone(written)
        self.assertEqual(len(history(self.store)), 1)

    def test_an_unchanged_block_writes_nothing(self) -> None:
        """A quiet store must not accumulate a row per turn."""
        record(self.store)
        self.assertIsNone(record(self.store))
        self.assertIsNone(record(self.store))
        self.assertEqual(len(history(self.store)), 1)

    def test_the_hash_identifies_the_content(self) -> None:
        snapshot = record(self.store)
        self.assertEqual(snapshot.block_hash, hash_block(snapshot.text))

    def test_history_is_newest_first_and_bounded(self) -> None:
        self._memory("First pinned fact.", pinned=True)
        record(self.store)
        self._memory("Second pinned fact.", pinned=True)
        record(self.store)
        rows = history(self.store)
        self.assertEqual(len(rows), 2)
        self.assertGreaterEqual(rows[0].chars, rows[1].chars)

    def test_promotion_records_a_snapshot(self) -> None:
        """Membership changes are exactly what a reader needs to see, so they snapshot."""
        memory_id = self._memory("Promotable fact.")
        record(self.store)
        before = len(history(self.store))
        promote(self.store, memory_id, by="pin")
        self.assertGreater(len(history(self.store)), before)

    def test_demotion_records_a_snapshot(self) -> None:
        memory_id = self._memory("Fact that leaves core.")
        promote(self.store, memory_id, by="pin")
        count = len(history(self.store))
        demote(self.store, memory_id)
        self.assertGreater(len(history(self.store)), count)


class DiffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_the_first_diff_reports_the_first_snapshot(self) -> None:
        record(self.store)
        self.assertIn("first snapshot", diff(self.store).summary())

    def test_an_added_memory_shows_as_an_added_line(self) -> None:
        record(self.store)
        self.store.add_memory(
            content="A newly pinned rule that must never be broken.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            pinned=True,
        )
        record(self.store)
        change = diff(self.store)
        self.assertTrue(change.changed)
        self.assertTrue(any("must never be broken" in line for line in change.added_lines))
        self.assertTrue(change.added_ids)

    def test_a_removed_memory_shows_as_a_removed_line(self) -> None:
        """Importance is set low so the filler does not silently keep the line in view.

        Unpinning alone changes the line (the `[pinned]` suffix goes) but leaves the memory in
        the block through the importance fill — which is correct behaviour and would have made
        this test pass for the wrong reason.
        """
        memory_id = self.store.add_memory(
            content="A rule that will leave core.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            pinned=True,
            importance=0.01,
        )[0]
        # Enough filler to fill the 2,000-character block, so the unimportant memory cannot
        # sneak back in through the importance fill — an unpinned fact with low importance
        # still gets rendered while there is room, which is correct but hides the removal.
        for index in range(40):
            self.store.add_memory(
                content=f"Filler fact {index} that outranks it and fills the block.",
                kind="semantic",
                source_type="user_turn",
                source_category="USER_EXPLICIT",
                importance=0.9,
            )
        record(self.store)
        demote(self.store, memory_id)
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET pinned = 0 WHERE id = ?", (memory_id,))
        record(self.store)
        change = diff(self.store)
        self.assertTrue(any("will leave core" in line for line in change.removed_lines))
        self.assertIn(memory_id, change.removed_ids)

    def test_an_edited_memory_shows_as_a_changed_line(self) -> None:
        """An id-level comparison would call this unchanged; the block did not."""
        memory_id = self.store.add_memory(
            content="The maintenance window is Sunday.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            pinned=True,
        )[0]
        record(self.store)
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE memories SET content = ? WHERE id = ?",
                ("The maintenance window is Tuesday.", memory_id),
            )
        record(self.store)
        change = diff(self.store)
        self.assertTrue(any("Tuesday" in line for line in change.added_lines))
        self.assertTrue(any("Sunday" in line for line in change.removed_lines))
        self.assertEqual(change.added_ids, ())
        self.assertEqual(change.removed_ids, ())

    def test_the_summary_states_the_size_delta(self) -> None:
        record(self.store)
        self.store.add_memory(
            content="Another pinned fact for the block.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            pinned=True,
        )
        record(self.store)
        self.assertIn("chars", diff(self.store).summary())

    def test_the_older_index_reaches_further_back(self) -> None:
        record(self.store)
        for index in range(3):
            self.store.add_memory(
                content=f"Pinned fact number {index}.",
                kind="semantic",
                source_type="user_turn",
                source_category="USER_EXPLICIT",
                pinned=True,
            )
            record(self.store)
        self.assertIsNotNone(diff(self.store, older_index=1).before)
        self.assertIsNotNone(diff(self.store, older_index=3).before)


if __name__ == "__main__":
    unittest.main()
