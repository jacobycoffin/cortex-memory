"""The shape of the store over time (#170)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import timeline_view
from cortex.store import CortexStore

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str, *, days_ago: int = 0, kind: str = "semantic") -> str:
        memory_id = self.store.add_memory(
            content=text, kind=kind, source_type="user_turn", source_category="USER_STATED",
        )[0]
        stamp = (NOW - timedelta(days=days_ago)).isoformat()
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET created_at = ? WHERE id = ?", (stamp, memory_id))
        return memory_id

    def test_the_series_covers_every_day_in_the_window(self) -> None:
        """A quiet day is data: omitting it hides exactly the period worth noticing."""
        self.memory("today")
        found = timeline_view.series(self.store, days=7, now=NOW)
        self.assertEqual(len(found.points), 7)
        self.assertEqual(sum(1 for point in found.points if point["count"] == 0), 6)

    def test_the_span_is_part_of_the_answer(self) -> None:
        found = timeline_view.series(self.store, days=3, now=NOW)
        self.assertEqual(found.end, "2026-11-24")
        self.assertEqual(found.start, "2026-11-22")

    def test_a_day_counts_only_its_own_memories(self) -> None:
        self.memory("today", days_ago=0)
        self.memory("yesterday one", days_ago=1)
        self.memory("yesterday two", days_ago=1)
        points = {point["bucket_start"]: point["count"]
                  for point in timeline_view.series(self.store, days=3, now=NOW).points}
        self.assertEqual(points["2026-11-24"], 1)
        self.assertEqual(points["2026-11-23"], 2)

    def test_archived_memories_are_not_in_the_timeline(self) -> None:
        keep = self.memory("kept")
        gone = self.memory("gone")
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET state = 'archived' WHERE id = ?", (gone,))
        total = timeline_view.summary(self.store, days=3, now=NOW)["total"]
        self.assertEqual(total, 1)
        self.assertTrue(keep)

    def test_kinds_are_counted_per_day(self) -> None:
        self.memory("a fact", kind="semantic")
        self.memory("a preference", kind="preference")
        point = timeline_view.series(self.store, days=2, now=NOW).points[-1]
        self.assertEqual(point["kinds"], {"preference": 1, "semantic": 1})

    def test_the_weekly_bucket_starts_on_monday(self) -> None:
        self.memory("a fact")
        found = timeline_view.series(self.store, days=14, bucket="week", now=NOW)
        for point in found.points:
            self.assertEqual(datetime.fromisoformat(point["bucket_start"]).weekday(), 0)

    def test_the_monthly_bucket_starts_on_the_first(self) -> None:
        self.memory("a fact")
        found = timeline_view.series(self.store, days=60, bucket="month", now=NOW)
        for point in found.points:
            self.assertTrue(point["bucket_start"].endswith("-01"))

    def test_an_unknown_bucket_falls_back_to_daily(self) -> None:
        self.memory("a fact")
        self.assertEqual(timeline_view.series(self.store, bucket="hour").bucket, "day")

    def test_an_empty_window_says_so(self) -> None:
        found = timeline_view.series(self.store, days=5, now=NOW)
        self.assertIn("nothing was written", found.note)

    def test_the_summary_reports_the_busiest_day(self) -> None:
        for index in range(3):
            self.memory(f"busy {index}", days_ago=2)
        self.memory("quiet", days_ago=4)
        described = timeline_view.summary(self.store, days=7, now=NOW)
        self.assertEqual(described["busiest"]["day"], "2026-11-22")
        self.assertEqual(described["busiest"]["count"], 3)

    def test_the_summary_reports_the_longest_silence(self) -> None:
        self.memory("old", days_ago=9)
        self.memory("new", days_ago=0)
        described = timeline_view.summary(self.store, days=10, now=NOW)
        self.assertGreaterEqual(described["longest_quiet_streak"], 8)

    def test_the_summary_counts_all_time_separately_from_the_window(self) -> None:
        self.memory("ancient", days_ago=400)
        self.memory("recent", days_ago=1)
        described = timeline_view.summary(self.store, days=7, now=NOW)
        self.assertEqual(described["total"], 1)
        self.assertEqual(described["all_time"], 2)

    def test_the_first_and_last_memory_are_named(self) -> None:
        self.memory("first", days_ago=30)
        self.memory("last", days_ago=1)
        described = timeline_view.summary(self.store, days=40, now=NOW)
        self.assertEqual(described["first_memory"], "2026-10-25")
        self.assertEqual(described["last_memory"], "2026-11-23")

    def test_months_run_oldest_to_newest(self) -> None:
        self.memory("a", days_ago=40)
        self.memory("b", days_ago=1)
        rows = timeline_view.months(self.store)
        self.assertEqual([row["month"] for row in rows], ["2026-10", "2026-11"])

    def test_the_sparkline_keeps_a_spike_when_downsampling(self) -> None:
        points = [{"count": 1}] * 100
        points[50] = {"count": 100}
        line = timeline_view.sparkline(points, width=10)
        self.assertEqual(len(line), 10)
        self.assertIn(timeline_view.SPARK[-1], line)

    def test_the_sparkline_is_empty_for_no_points(self) -> None:
        self.assertEqual(timeline_view.sparkline([]), "")

    def test_the_render_states_the_window_and_the_total(self) -> None:
        self.memory("a fact")
        rendered = "\n".join(timeline_view.render(self.store, days=7, now=NOW))
        self.assertIn("1 memories", rendered)
        self.assertIn("quiet day", rendered)

    def test_the_render_shows_the_months(self) -> None:
        self.memory("a fact")
        self.assertIn("by month", "\n".join(timeline_view.render(self.store, days=7, now=NOW)))

    def test_kinds_over_aggregates_the_window(self) -> None:
        self.memory("a", kind="semantic")
        self.memory("b", kind="semantic")
        self.memory("c", kind="preference")
        self.assertEqual(timeline_view.kinds_over(self.store, days=7, now=NOW),
                         {"semantic": 2, "preference": 1})

    def test_a_broken_store_yields_an_empty_window(self) -> None:
        class Broken:
            _conn = None

        found = timeline_view.series(Broken(), days=5, now=NOW)
        self.assertEqual(sum(point["count"] for point in found.points), 0)


if __name__ == "__main__":
    unittest.main()
