"""Answer "what did I believe on this date" from the version log (#55)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import as_of
from cortex.store import CortexStore

EARLY = datetime(2026, 1, 10, 9, 0, tzinfo=timezone.utc)
MIDDLE = datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)
LATE = datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    """The version log has a foreign key to `memories`, so every test starts from a real row.

    `add_memory` writes the first version itself; `fresh()` removes it so a test controls exactly
    which versions exist — otherwise "two versions" is really three.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def fresh(self, content: str = "The dashboard runs on port 8100.", *,
              kind: str = "semantic") -> str:
        memory_id = self.store.add_memory(
            content=content, kind=kind, source_type="user_turn", source_category="USER_EXPLICIT",
        )[0]
        with self.store.transaction() as conn:
            conn.execute("DELETE FROM memory_versions WHERE memory_id = ?", (memory_id,))
        return memory_id

    def version(self, memory_id: str, content: str, start: datetime, *,
                end: datetime | None = None, confidence: float = 0.8) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO memory_versions
                   (memory_id, content, confidence, state, system_from, system_to)
                   VALUES (?, ?, ?, 'active', ?, ?)""",
                (memory_id, content, confidence, start.isoformat(),
                 end.isoformat() if end else None),
            )

    def test_a_version_open_at_the_instant_is_believed(self) -> None:
        memory_id = self.fresh()
        self.version(memory_id, "port 8100", EARLY)
        self.assertEqual([b.memory_id for b in as_of.believed(self.store, MIDDLE)], [memory_id])

    def test_a_fact_recorded_later_was_not_believed_earlier(self) -> None:
        memory_id = self.fresh()
        self.version(memory_id, "port 8200", LATE)
        self.assertEqual(as_of.believed(self.store, MIDDLE), [])

    def test_a_fact_retired_before_the_instant_was_not_believed(self) -> None:
        memory_id = self.fresh()
        self.version(memory_id, "port 8100", EARLY, end=MIDDLE)
        self.assertEqual(as_of.believed(self.store, LATE), [])

    def test_a_fact_retired_after_the_instant_was_believed(self) -> None:
        memory_id = self.fresh()
        self.version(memory_id, "port 8100", EARLY, end=LATE)
        self.assertEqual(len(as_of.believed(self.store, MIDDLE)), 1)

    def test_the_latest_version_before_the_instant_wins(self) -> None:
        memory_id = self.fresh()
        self.version(memory_id, "port 8100", EARLY, end=MIDDLE)
        self.version(memory_id, "port 8200", MIDDLE)
        self.assertEqual(as_of.believed(self.store, LATE)[0].content, "port 8200")

    def test_one_version_per_memory_is_returned(self) -> None:
        memory_id = self.fresh()
        self.version(memory_id, "port 8100", EARLY)
        self.version(memory_id, "port 8200", EARLY)
        self.assertEqual(len(as_of.believed(self.store, LATE)), 1)

    def test_a_bare_date_is_the_start_of_that_day(self) -> None:
        """A bare date asks about 00:00 UTC, not the end of the day."""
        memory_id = self.fresh()
        self.version(memory_id, "port 8100", datetime(2026, 2, 1, 9, 0, tzinfo=timezone.utc))
        self.assertEqual(len(as_of.believed(self.store, date(2026, 3, 1))), 1)
        self.assertEqual(as_of.believed(self.store, date(2026, 1, 15)), [])

    def test_a_fact_recorded_later_that_day_was_not_believed_at_its_start(self) -> None:
        memory_id = self.fresh()
        self.version(memory_id, "port 8200", MIDDLE)
        self.assertEqual(as_of.believed(self.store, date(2026, 3, 1)), [])
        self.assertEqual(len(as_of.believed(self.store, date(2026, 3, 2))), 1)

    def test_the_kind_comes_from_the_memory_row(self) -> None:
        memory_id = self.fresh(kind="procedure")
        self.version(memory_id, "port 8100", EARLY)
        self.assertEqual(as_of.believed(self.store, MIDDLE)[0].kind, "procedure")

    def test_the_answer_says_when_the_log_starts(self) -> None:
        memory_id = self.fresh()
        self.version(memory_id, "port 8100", MIDDLE)
        answered = as_of.answer(self.store, datetime(2025, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(answered["count"], 0)
        self.assertIn("version log starts", answered["note"])

    def test_the_answer_can_be_narrowed_to_a_subject(self) -> None:
        first = self.fresh("The dashboard runs on port 8100.")
        second = self.fresh("Backups run at two in the morning.")
        self.version(first, "The dashboard runs on port 8100.", EARLY)
        self.version(second, "Backups run at two in the morning.", EARLY)
        answered = as_of.answer(self.store, MIDDLE, query="dashboard")
        self.assertEqual(answered["count"], 1)
        self.assertIn("dashboard", answered["facts"][0]["content"])

    def test_changes_report_what_was_learned_and_retired(self) -> None:
        memory_id = self.fresh()
        self.version(memory_id, "port 8100", EARLY, end=MIDDLE)
        self.version(memory_id, "port 8200", MIDDLE)
        window = as_of.changes(self.store, EARLY, LATE)
        self.assertGreaterEqual(len(window["learned"]), 1)
        self.assertGreaterEqual(len(window["retired"]), 1)
        self.assertIn("learned", window["sentence"])

    def test_history_lists_every_version(self) -> None:
        memory_id = self.fresh()
        self.version(memory_id, "port 8100", EARLY, end=MIDDLE)
        self.version(memory_id, "port 8200", MIDDLE)
        self.assertEqual(len(as_of.history(self.store, memory_id)), 2)

    def test_the_timeline_names_the_current_interval(self) -> None:
        memory_id = self.fresh()
        self.version(memory_id, "port 8100", EARLY, end=MIDDLE)
        self.version(memory_id, "port 8200", MIDDLE)
        timeline = as_of.timeline(self.store, memory_id)
        self.assertEqual(len(timeline.intervals), 2)
        self.assertEqual(timeline.current.content, "port 8200")
        self.assertFalse(timeline.as_list()[0]["current"])

    def test_coverage_counts_closed_versions(self) -> None:
        memory_id = self.fresh()
        self.version(memory_id, "port 8100", EARLY, end=MIDDLE)
        self.version(memory_id, "port 8200", MIDDLE)
        described = as_of.coverage(self.store)
        self.assertEqual(described["versions"], 2)
        self.assertEqual(described["closed"], 1)
        self.assertAlmostEqual(described["closed_share"], 0.5, places=4)

    def test_a_belief_describes_itself(self) -> None:
        memory_id = self.fresh()
        self.version(memory_id, "port 8100", EARLY)
        self.assertIn("now", as_of.believed(self.store, MIDDLE)[0].describe())

    def test_an_instant_accepts_a_string_and_a_date(self) -> None:
        self.assertTrue(as_of.instant("2026-03-01").startswith("2026-03-01T00:00:00"))
        self.assertTrue(as_of.instant(date(2026, 3, 1)).startswith("2026-03-01"))
        self.assertTrue(as_of.instant(EARLY).startswith("2026-01-10"))

    def test_an_empty_store_is_quiet(self) -> None:
        self.assertEqual(as_of.believed(self.store, MIDDLE), [])
        self.assertEqual(as_of.coverage(self.store)["versions"], 0)
        self.assertEqual(as_of.changes(self.store, EARLY, LATE)["learned"], [])
        self.assertEqual(as_of.timeline(self.store, "no-such-id").intervals, [])


if __name__ == "__main__":
    unittest.main()
