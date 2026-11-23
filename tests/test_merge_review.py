"""A merge is shown side by side, needs consent, and keeps the old record (#29)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import merge_review
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
LEFT = "The dashboard runs on port 8100 behind nginx."
RIGHT = "Nginx proxies the dashboard on port 8100."


class DiffTests(unittest.TestCase):
    def test_identical_text_has_an_empty_diff(self) -> None:
        self.assertEqual(merge_review.diff("a b c", "a b c"), [])

    def test_an_added_word_is_marked(self) -> None:
        self.assertIn("+ staging", merge_review.diff("port 8100", "port 8100 staging"))

    def test_a_removed_word_is_marked(self) -> None:
        self.assertIn("- staging", merge_review.diff("port 8100 staging", "port 8100"))

    def test_a_replacement_shows_both_sides(self) -> None:
        changes = merge_review.diff("port 8100", "port 8200")
        self.assertIn("- 8100", changes)
        self.assertIn("+ 8200", changes)

    def test_the_diff_is_bounded(self) -> None:
        left = " ".join(f"w{index}" for index in range(200))
        right = " ".join(f"v{index}" for index in range(200))
        self.assertLessEqual(len(merge_review.diff(left, right, limit=5)), 6)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.left = self.memory(LEFT)
        self.right = self.memory(RIGHT)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def state(self, memory_id: str) -> str:
        return str(self.store._conn.execute(
            "SELECT state FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()[0])

    def content(self, memory_id: str) -> str:
        return str(self.store._conn.execute(
            "SELECT content FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()[0])

    def proposal(self):
        merged = "The dashboard runs on port 8100 behind nginx, which proxies it."
        return merge_review.propose(self.store, self.left, self.right, merged_text=merged)

    def test_a_proposal_shows_both_sides(self) -> None:
        proposal = self.proposal()
        self.assertEqual(proposal.target_text, LEFT)
        self.assertEqual(proposal.source_text, RIGHT)
        self.assertIn("left", proposal.describe())
        self.assertIn("right", proposal.describe())

    def test_a_proposal_carries_the_diff(self) -> None:
        self.assertTrue(self.proposal().diff)

    def test_a_proposal_for_a_missing_memory_is_none(self) -> None:
        self.assertIsNone(merge_review.propose(self.store, self.left, "no-such-id"))

    def test_merging_without_consent_changes_nothing(self) -> None:
        result = merge_review.apply(self.store, self.proposal())
        self.assertFalse(result.ok)
        self.assertIn("consent required", result.reason)
        self.assertEqual(self.content(self.left), LEFT)
        self.assertEqual(self.state(self.right), "active")

    def test_consent_must_name_who_gave_it(self) -> None:
        result = merge_review.apply(self.store, self.proposal(), consent=True)
        self.assertFalse(result.ok)
        self.assertIn("name who", result.reason)

    def test_a_merge_writes_the_target_and_supersedes_the_source(self) -> None:
        result = merge_review.apply(self.store, self.proposal(), consent=True, consented_by="jacoby")
        self.assertTrue(result.ok)
        self.assertIn("proxies it", self.content(self.left))
        self.assertEqual(self.state(self.right), "superseded")

    def test_the_old_record_is_retained(self) -> None:
        merge_review.apply(self.store, self.proposal(), consent=True, consented_by="jacoby")
        # the source keeps its own text; it is superseded, not erased
        self.assertEqual(self.content(self.right), RIGHT)

    def test_the_history_keeps_both_prior_texts(self) -> None:
        merge_review.apply(self.store, self.proposal(), consent=True, consented_by="jacoby")
        entry = merge_review.history(self.store)[0]
        self.assertEqual(entry["target_before"], LEFT)
        self.assertEqual(entry["source_before"], RIGHT)
        self.assertEqual(entry["consented_by"], "jacoby")

    def test_undo_restores_both_sides_exactly(self) -> None:
        merge_review.apply(self.store, self.proposal(), consent=True, consented_by="jacoby")
        result = merge_review.undo(self.store)
        self.assertTrue(result["undone"])
        self.assertEqual(self.content(self.left), LEFT)
        self.assertEqual(self.content(self.right), RIGHT)
        self.assertEqual(self.state(self.right), "active")

    def test_a_merge_is_not_undone_twice(self) -> None:
        merge_review.apply(self.store, self.proposal(), consent=True, consented_by="jacoby")
        merge_review.undo(self.store)
        self.assertFalse(merge_review.undo(self.store)["undone"])

    def test_undoing_an_unknown_merge_is_reported(self) -> None:
        result = merge_review.undo(self.store, "no-such-merge")
        self.assertFalse(result["undone"])
        self.assertIn("no merge", result["reason"])

    def test_a_merge_of_a_vanished_memory_is_refused(self) -> None:
        proposal = self.proposal()
        with self.store.transaction() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (self.right,))
        result = merge_review.apply(self.store, proposal, consent=True, consented_by="jacoby")
        self.assertFalse(result.ok)
        self.assertIn("no longer exists", result.reason)

    def test_the_report_counts_undoable_merges(self) -> None:
        merge_review.apply(self.store, self.proposal(), consent=True, consented_by="jacoby")
        described = merge_review.report(self.store)
        self.assertEqual(described["merges"], 1)
        self.assertEqual(described["undoable"], 1)
        merge_review.undo(self.store)
        self.assertEqual(merge_review.report(self.store)["undoable"], 0)

    def test_the_proposal_round_trips(self) -> None:
        payload = self.proposal().as_dict()
        for key in ("left", "right", "merged", "diff", "preview"):
            self.assertIn(key, payload)

    def test_an_empty_store_is_quiet(self) -> None:
        other = CortexStore(Path(self.tmp.name) / "empty.db")
        try:
            self.assertEqual(merge_review.history(other), [])
            self.assertEqual(merge_review.report(other)["merges"], 0)
            self.assertFalse(merge_review.undo(other)["undone"])
        finally:
            other.close()


if __name__ == "__main__":
    unittest.main()
