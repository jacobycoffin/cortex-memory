"""A summary is a new node, never a replacement (#205)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import summaries
from cortex.store import CortexStore

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str, *, confidence: float = 0.8, kind: str = "semantic") -> str:
        return self.store.add_memory(
            content=text, kind=kind, source_type="user_turn", source_category="USER_STATED",
            confidence=confidence,
        )[0]

    def two_members(self) -> tuple[str, str]:
        return (self.memory("The dashboard listens on port 8100 behind nginx.", confidence=0.9),
                self.memory("Deploys go through the wrapper on CT116.", confidence=0.7))

    def test_a_summary_is_a_new_node(self) -> None:
        first, second = self.two_members()
        made = summaries.create(self.store, [first, second], "The dashboard deploys via CT116.",
                                at=NOW)
        self.assertTrue(made["created"])
        self.assertNotIn(made["memory_id"], {first, second})

    def test_the_members_are_untouched(self) -> None:
        first, second = self.two_members()
        before = [self.store._conn.execute(
            "SELECT content, state FROM memories WHERE id = ?", (item,)).fetchone()
            for item in (first, second)]
        summaries.create(self.store, [first, second], "A summary.", at=NOW)
        after = [self.store._conn.execute(
            "SELECT content, state FROM memories WHERE id = ?", (item,)).fetchone()
            for item in (first, second)]
        self.assertEqual([tuple(row) for row in before], [tuple(row) for row in after])

    def test_the_confidence_is_capped_by_the_weakest_member(self) -> None:
        first, second = self.two_members()
        made = summaries.create(self.store, [first, second], "A summary.", at=NOW)
        self.assertAlmostEqual(made["confidence"], 0.7, places=6)

    def test_the_summary_carries_its_role(self) -> None:
        first, second = self.two_members()
        made = summaries.create(self.store, [first, second], "A summary.", at=NOW)
        role = self.store._conn.execute(
            "SELECT record_role FROM memories WHERE id = ?", (made["memory_id"],)).fetchone()[0]
        self.assertEqual(role, "summary")

    def test_the_members_can_be_read_back(self) -> None:
        first, second = self.two_members()
        made = summaries.create(self.store, [first, second], "A summary.", at=NOW)
        self.assertEqual(sorted(summaries.members_of(self.store, made["memory_id"])),
                         sorted([first, second]))

    def test_a_member_can_find_its_summary(self) -> None:
        first, second = self.two_members()
        made = summaries.create(self.store, [first, second], "A summary.", at=NOW)
        self.assertIn(made["memory_id"], summaries.covered_by(self.store, first))

    def test_an_unknown_member_is_refused(self) -> None:
        made = summaries.create(self.store, ["nope"], "A summary.", at=NOW)
        self.assertFalse(made["created"])

    def test_a_summary_with_no_members_is_refused(self) -> None:
        made = summaries.create(self.store, [], "A summary.", at=NOW)
        self.assertFalse(made["created"])
        self.assertIn("at least one member", made["reason"])

    def test_a_fresh_summary_is_not_stale(self) -> None:
        first, second = self.two_members()
        made = summaries.create(self.store, [first, second], "A summary.", at=NOW)
        self.assertFalse(summaries.stale(self.store, made["memory_id"]).dirty)

    def test_a_changed_member_makes_it_stale(self) -> None:
        first, second = self.two_members()
        made = summaries.create(self.store, [first, second], "A summary.", at=NOW)
        self.store.add_memory(
            content="The dashboard listens on port 8200 behind nginx.", kind="semantic",
            source_type="user_turn", source_category="USER_CORRECTION",
        )
        # Replace the member's text in place, the way a correction would.
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET content = ? WHERE id = ?",
                         ("The dashboard listens on port 8200 now.", first))
            conn.execute("UPDATE memories SET content_hash = ? WHERE id = ?",
                         ("changed-hash", first))
        self.assertTrue(summaries.stale(self.store, made["memory_id"]).dirty)

    def test_a_timestamp_only_change_is_not_staleness(self) -> None:
        """This store stamps `updated_at` in bulk; a moved timestamp is not a changed fact."""
        first, second = self.two_members()
        made = summaries.create(self.store, [first, second], "A summary.", at=NOW)
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET updated_at = ? WHERE id = ?",
                         ("2026-12-01T00:00:00+00:00", first))
        self.assertFalse(summaries.stale(self.store, made["memory_id"]).dirty)

    def test_refresh_supersedes_rather_than_edits(self) -> None:
        first, second = self.two_members()
        made = summaries.create(self.store, [first, second], "A summary.", at=NOW)
        fresh = summaries.refresh(self.store, made["memory_id"], "A better summary.", at=NOW)
        self.assertTrue(fresh["created"])
        old = self.store._conn.execute(
            "SELECT state, content FROM memories WHERE id = ?", (made["memory_id"],)).fetchone()
        self.assertEqual(old["state"], "superseded")
        self.assertEqual(old["content"], "A summary.")

    def test_refresh_records_what_it_replaced(self) -> None:
        first, second = self.two_members()
        made = summaries.create(self.store, [first, second], "A summary.", at=NOW)
        fresh = summaries.refresh(self.store, made["memory_id"], "A better summary.", at=NOW)
        winner = self.store._conn.execute(
            "SELECT supersedes_id FROM memories WHERE id = ?", (fresh["memory_id"],)).fetchone()
        self.assertEqual(winner["supersedes_id"], made["memory_id"])

    def test_refreshing_a_non_summary_is_refused(self) -> None:
        plain = self.memory("Not a summary of anything.")
        self.assertFalse(summaries.refresh(self.store, plain, "still not")["created"])

    def test_summaries_lists_only_summaries(self) -> None:
        first, second = self.two_members()
        made = summaries.create(self.store, [first, second], "A summary.", at=NOW)
        listed = [row["id"] for row in summaries.summaries(self.store)]
        self.assertEqual(listed, [made["memory_id"]])

    def test_the_report_counts_stale_summaries(self) -> None:
        first, second = self.two_members()
        summaries.create(self.store, [first, second], "A summary.", at=NOW)
        described = summaries.report(self.store)
        self.assertEqual(described["summaries"], 1)
        self.assertEqual(described["stale"], 0)

    def test_an_empty_store_is_quiet(self) -> None:
        self.assertEqual(summaries.report(self.store)["summaries"], 0)
        self.assertEqual(summaries.members_of(self.store, "nope"), [])
        self.assertFalse(summaries.stale(self.store, "nope").dirty)


if __name__ == "__main__":
    unittest.main()
