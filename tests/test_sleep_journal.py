"""The dream journal (#111): records restraint, never becomes a queue."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import sleep_journal
from cortex.store import CortexStore

NIGHT = datetime(2026, 11, 24, 23, 50, tzinfo=timezone.utc)
TOMORROW = datetime(2026, 11, 25, 23, 50, tzinfo=timezone.utc)


class JournalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str = "the shelf needs a second coat") -> str:
        return self.store.add_memory(content=text, kind="semantic", source_type="user_turn",
                                     source_category="USER_STATED")[0]

    # -- what a notice is --------------------------------------------------------------

    def test_a_notice_is_recorded_with_its_verdict(self) -> None:
        memory_id = self.memory()
        result = sleep_journal.notice(
            self.store, kind="near_duplicate", note="two vault notes look alike",
            why="below_evidence_bar", memory_ids=[memory_id], night=sleep_journal.night_of(NIGHT),
        )
        self.assertTrue(result["ok"])
        entries = sleep_journal.journal(self.store, night=sleep_journal.night_of(NIGHT))
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].why, "below_evidence_bar")
        self.assertIn("below_evidence_bar", entries[0].render())

    def test_an_entry_without_a_verdict_is_refused(self) -> None:
        result = sleep_journal.notice(self.store, kind="stale", note="old", why="because")
        self.assertFalse(result["ok"])
        self.assertIn("unknown why", result["reason"])

    def test_an_unknown_kind_is_refused(self) -> None:
        result = sleep_journal.notice(self.store, kind="vibes", note="something felt off",
                                      why="needs_decision")
        self.assertFalse(result["ok"])
        self.assertIn("unknown kind", result["reason"])

    def test_an_entry_needs_a_note(self) -> None:
        result = sleep_journal.notice(self.store, kind="stale", note="   ", why="budget")
        self.assertFalse(result["ok"])

    # -- it must not become a queue ----------------------------------------------------

    def test_writing_a_notice_changes_no_memory(self) -> None:
        memory_id = self.memory("a memory that must stay exactly as it is")
        before = dict(self.store._conn.execute(
            "SELECT state, confidence, used_count, importance FROM memories WHERE id = ?",
            (memory_id,)).fetchone())
        sleep_journal.notice(self.store, kind="low_confidence", note="confidence looks high",
                             why="below_evidence_bar", memory_ids=[memory_id])
        after = dict(self.store._conn.execute(
            "SELECT state, confidence, used_count, importance FROM memories WHERE id = ?",
            (memory_id,)).fetchone())
        self.assertEqual(before, after, "a journal entry is a record of restraint, not an action")

    def test_noticing_the_same_thing_twice_increments_instead_of_duplicating(self) -> None:
        memory_id = self.memory()
        night = sleep_journal.night_of(NIGHT)
        sleep_journal.notice(self.store, kind="stale", note="not reached in 40 days",
                             why="not_scheduled", memory_ids=[memory_id], night=night)
        second = sleep_journal.notice(self.store, kind="stale", note="not reached in 40 days",
                                      why="not_scheduled", memory_ids=[memory_id], night=night)
        self.assertTrue(second["deduplicated"])
        self.assertEqual(second["times"], 2)
        entries = sleep_journal.journal(self.store, night=night)
        self.assertEqual(len(entries), 1)
        self.assertIn("x2", entries[0].render())

    def test_the_same_ids_in_a_different_order_are_the_same_notice(self) -> None:
        first, second = self.memory("one"), self.memory("two")
        night = sleep_journal.night_of(NIGHT)
        sleep_journal.notice(self.store, kind="contradiction", note="these disagree",
                             why="contradicted", memory_ids=[first, second], night=night)
        result = sleep_journal.notice(self.store, kind="contradiction", note="these disagree",
                                      why="contradicted", memory_ids=[second, first], night=night)
        self.assertTrue(result["deduplicated"])

    # -- bounded, per night ------------------------------------------------------------

    def test_the_journal_is_bounded_and_refuses_rather_than_growing(self) -> None:
        night = sleep_journal.night_of(NIGHT)
        for index in range(3):
            self.assertTrue(sleep_journal.notice(
                self.store, kind="stale", note=f"stale memory {index}", why="not_scheduled",
                night=night, limit=3)["ok"])
        refused = sleep_journal.notice(self.store, kind="stale", note="one too many",
                                      why="not_scheduled", night=night, limit=3)
        self.assertFalse(refused["ok"])
        self.assertTrue(refused.get("full"))
        self.assertIn("aggregate", refused["reason"])

    def test_nights_are_separate(self) -> None:
        sleep_journal.notice(self.store, kind="stale", note="a", why="budget",
                             night=sleep_journal.night_of(NIGHT))
        sleep_journal.notice(self.store, kind="stale", note="b", why="budget",
                             night=sleep_journal.night_of(TOMORROW))
        self.assertEqual(len(sleep_journal.journal(self.store, night=sleep_journal.night_of(NIGHT))), 1)
        self.assertEqual(len(sleep_journal.journal(self.store, night=sleep_journal.night_of(TOMORROW))), 1)

    def test_a_cycle_across_midnight_is_one_night(self) -> None:
        self.assertEqual(
            sleep_journal.night_of(datetime(2026, 11, 24, 23, 50, tzinfo=timezone.utc)),
            sleep_journal.night_of(datetime(2026, 11, 25, 0, 20, tzinfo=timezone.utc)))

    # -- reading ----------------------------------------------------------------------

    def test_a_quiet_night_reads_as_a_sentence(self) -> None:
        lines = sleep_journal.render(sleep_journal.journal(self.store, night=sleep_journal.night_of(NIGHT)))
        self.assertEqual(len(lines), 1)
        self.assertIn("noticed nothing", lines[0])

    def test_the_headline_counts_the_verdicts(self) -> None:
        night = sleep_journal.night_of(NIGHT)
        sleep_journal.notice(self.store, kind="stale", note="a", why="budget", night=night)
        sleep_journal.notice(self.store, kind="stale", note="b", why="budget", night=night)
        sleep_journal.notice(self.store, kind="stale", note="c", why="not_scheduled", night=night)
        head = sleep_journal.render(sleep_journal.journal(self.store, night=night))[0]
        self.assertIn("2 budget", head)
        self.assertIn("1 not_scheduled", head)

    def test_rendering_is_bounded_and_says_how_much_it_cut(self) -> None:
        night = sleep_journal.night_of(NIGHT)
        for index in range(6):
            sleep_journal.notice(self.store, kind="stale", note=f"stale {index}", why="budget",
                                 night=night, limit=20)
        lines = sleep_journal.render(sleep_journal.journal(self.store, night=night), limit=2)
        self.assertTrue(any("4 more" in line for line in lines))

    def test_the_journal_keeps_ids_and_not_content(self) -> None:
        memory_id = self.memory("a private sentence that must not be copied into the journal")
        sleep_journal.notice(self.store, kind="low_confidence", note="confidence looks high",
                             why="below_evidence_bar", memory_ids=[memory_id])
        raw = self.store._conn.execute(
            f"SELECT note, memory_ids FROM {sleep_journal.JOURNAL_TABLE}").fetchone()
        self.assertNotIn("private sentence", raw["note"])
        self.assertIn(memory_id, raw["memory_ids"])

    # -- retention ---------------------------------------------------------------------

    def test_pruning_drops_old_nights_and_keeps_recent_ones(self) -> None:
        old = sleep_journal.night_of(datetime(2026, 9, 1, 12, tzinfo=timezone.utc))
        recent = sleep_journal.night_of(NIGHT)
        sleep_journal.notice(self.store, kind="stale", note="ancient", why="budget", night=old)
        sleep_journal.notice(self.store, kind="stale", note="recent", why="budget", night=recent)
        dropped = sleep_journal.prune(self.store, keep_nights=30, now=NIGHT + timedelta(days=1))
        self.assertEqual(dropped, 1)
        self.assertEqual(len(sleep_journal.journal(self.store, night=recent)), 1)
        self.assertEqual(sleep_journal.journal(self.store, night=old), [])


if __name__ == "__main__":
    unittest.main()
