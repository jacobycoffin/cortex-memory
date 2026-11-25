"""The overnight diff (#106): changes only, quiet is an answer, the cut is stated."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import sleep_diff
from cortex.store import CortexStore

NOW = datetime(2026, 11, 24, 7, 0, tzinfo=timezone.utc)


class DiffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str = "the proxmox box runs 8.4", *, hours_ago: float = 1) -> str:
        memory_id = self.store.add_memory(
            content=text, kind="semantic", source_type="user_turn", source_category="USER_STATED",
        )[0]
        stamp = (NOW - timedelta(hours=hours_ago)).isoformat()
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET created_at = ? WHERE id = ?", (stamp, memory_id))
        return memory_id

    def lifecycle(self, memory_id: str, from_state: str, to_state: str, *,
                  hours_ago: float = 1, reason: str = "the sweep decided so") -> None:
        with self.store.transaction() as conn:
            # `lifecycle_events.event_id` is an INTEGER rowid: the id belongs to the database, not the test.
            conn.execute(
                "INSERT INTO lifecycle_events (memory_id, from_state, to_state, reason, "
                "retention_score, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (memory_id, from_state, to_state, reason, 0.2,
                 (NOW - timedelta(hours=hours_ago)).isoformat()),
            )

    def snapshot(self) -> tuple:
        tables = tuple(sorted(r["name"] for r in self.store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")))
        counts = tuple(int(self.store._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
                       for t in tables)
        return tables, counts

    # -- the window ------------------------------------------------------------------

    def test_a_quiet_night_is_reported_as_quiet_not_as_an_empty_page(self) -> None:
        report = sleep_diff.report(self.store, hours=12, until=NOW)
        self.assertTrue(report.quiet)
        self.assertEqual(sleep_diff.render(report), ["Nothing changed in the last 12h."])

    def test_only_changes_inside_the_window_are_reported(self) -> None:
        self.memory("added last night", hours_ago=2)
        self.memory("added three days ago", hours_ago=72)
        report = sleep_diff.report(self.store, hours=12, until=NOW)
        self.assertEqual(len(report.changes), 1)
        self.assertIn("added last night", report.changes[0].sentence)

    def test_the_window_is_stated_in_the_output(self) -> None:
        report = sleep_diff.report(self.store, hours=6, until=NOW)
        self.assertEqual(report.window_hours, 6)
        self.assertTrue(report.since and report.until)

    # -- what a change says ----------------------------------------------------------

    def test_a_new_memory_reads_as_learned(self) -> None:
        self.memory("the walnut shelf needs a second coat", hours_ago=1)
        line = sleep_diff.render(sleep_diff.report(self.store, until=NOW))[1]
        self.assertIn("learned:", line)

    def test_a_retirement_reads_in_plain_english_with_its_reason(self) -> None:
        memory_id = self.memory(hours_ago=3)
        self.lifecycle(memory_id, "active", "archived", hours_ago=1, reason="duplicate of an import")
        report = sleep_diff.report(self.store, until=NOW)
        rendered = "\n".join(sleep_diff.render(report))
        self.assertIn("retired", rendered)
        self.assertIn("duplicate of an import", rendered)

    def test_an_unknown_transition_is_shown_with_both_states(self) -> None:
        memory_id = self.memory(hours_ago=3)
        self.lifecycle(memory_id, "cold", "quarantine", hours_ago=1)
        rendered = "\n".join(sleep_diff.render(sleep_diff.report(self.store, until=NOW)))
        self.assertIn("cold -> quarantine", rendered)

    def test_every_line_names_its_table(self) -> None:
        memory_id = self.memory(hours_ago=2)
        self.lifecycle(memory_id, "active", "archived", hours_ago=1)
        report = sleep_diff.report(self.store, until=NOW)
        for change in report.changes:
            self.assertTrue(change.table, change.sentence)
        rendered = "\n".join(sleep_diff.render(report))
        self.assertIn("[memories]", rendered)
        self.assertIn("[lifecycle_events]", rendered)

    def test_a_change_this_module_cannot_classify_is_still_reported(self) -> None:
        memory_id = self.memory(hours_ago=3)
        self.lifecycle(memory_id, "cold", "quarantine", hours_ago=1)
        kinds = {change.kind for change in sleep_diff.report(self.store, until=NOW).changes}
        self.assertTrue(kinds, "an unrecognised transition must not vanish")
        self.assertIn("changed", kinds)

    # -- size and honesty ------------------------------------------------------------

    def test_the_report_is_bounded_and_says_how_much_it_cut(self) -> None:
        for index in range(20):
            self.memory(f"memory number {index} added overnight", hours_ago=1)
        report = sleep_diff.report(self.store, until=NOW, limit=5)
        self.assertEqual(len(report.changes), 5)
        self.assertEqual(report.more, 15)
        self.assertTrue(any("15 more" in line for line in sleep_diff.render(report)))

    def test_the_counts_match_the_changes(self) -> None:
        self.memory("one", hours_ago=1)
        memory_id = self.memory("two", hours_ago=2)
        self.lifecycle(memory_id, "active", "archived", hours_ago=1)
        report = sleep_diff.report(self.store, until=NOW)
        self.assertEqual(sum(report.counts.values()), len(report.changes))
        self.assertEqual(report.counts.get("added"), 2)
        self.assertEqual(report.counts.get("retired"), 1)

    def test_counting_does_not_change_when_lines_are_cut(self) -> None:
        for index in range(8):
            self.memory(f"memory {index}", hours_ago=1)
        small = sleep_diff.report(self.store, until=NOW, limit=2)
        large = sleep_diff.report(self.store, until=NOW, limit=50)
        self.assertEqual(small.counts, large.counts)

    def test_a_change_without_a_preview_says_so_instead_of_inventing_one(self) -> None:
        self.memory("a private note that must not be echoed", hours_ago=1)
        report = sleep_diff.report(self.store, until=NOW, previews=False)
        self.assertNotIn("private note", report.changes[0].sentence)
        self.assertIn("learned", report.changes[0].sentence)

    def test_reading_the_report_writes_nothing(self) -> None:
        self.memory(hours_ago=1)
        before = self.snapshot()
        sleep_diff.report(self.store, until=NOW)
        self.assertEqual(before, self.snapshot())

    def test_a_broken_store_says_it_could_not_read_rather_than_nothing_changed(self) -> None:
        class Broken:
            _conn = None

        report = sleep_diff.report(Broken(), until=NOW)
        self.assertFalse(report.readable)
        self.assertFalse(report.quiet, "'cannot read' must not look like 'nothing changed'")
        self.assertIn("could not be read", sleep_diff.render(report)[0])

    def test_a_quiet_report_still_renders_one_line(self) -> None:
        self.assertEqual(len(sleep_diff.render(sleep_diff.report(self.store, until=NOW))), 1)


if __name__ == "__main__":
    unittest.main()
