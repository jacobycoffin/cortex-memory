"""Consolidate when the box is quiet (#212)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import idle_window
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def touch(self, *, seconds_ago: float) -> None:
        moment = (NOW - timedelta(seconds=seconds_ago)).isoformat()
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO episodes (session_id, user_content, assistant_content, created_at,
                                         content_hash)
                   VALUES ('s', 'a turn happened', 'ok', ?, ?)""",
                (moment, f"h-{seconds_ago}"),
            )

    def test_a_fresh_store_counts_as_idle(self) -> None:
        """Nothing has happened, so the first consolidation must be allowed to run."""
        verdict = idle_window.should_run(self.store, now=NOW)
        self.assertTrue(verdict.run)
        self.assertEqual(verdict.activity.last_seen, "")

    def test_recent_activity_is_busy(self) -> None:
        self.touch(seconds_ago=30)
        verdict = idle_window.should_run(self.store, min_idle_seconds=600, now=NOW)
        self.assertFalse(verdict.run)
        self.assertIn("busy", verdict.reason)

    def test_the_reason_names_the_gap(self) -> None:
        self.touch(seconds_ago=30)
        verdict = idle_window.should_run(self.store, min_idle_seconds=600, now=NOW)
        self.assertIn("need 600s", verdict.reason)

    def test_a_quiet_store_runs(self) -> None:
        self.touch(seconds_ago=3600)
        self.assertTrue(idle_window.should_run(self.store, min_idle_seconds=600, now=NOW).run)

    def test_activity_is_read_from_the_whole_store(self) -> None:
        """A turn is not the only activity: a recall ten seconds ago makes the store busy."""
        memory_id = self.store.add_memory(
            content="Cortex prefers conservative defaults.", kind="semantic",
            source_type="user_turn", source_category="USER_STATED",
        )[0]
        self.touch(seconds_ago=900)
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO access_log (memory_id, event, query, session_id, created_at)
                   VALUES (?, 'used', 'q', 's', ?)""",
                (memory_id, (NOW - timedelta(seconds=10)).isoformat()),
            )
        found = idle_window.activity(self.store, now=NOW)
        self.assertEqual(found.idle_seconds, 10.0)
        self.assertIn("access_log", found.sources)

    def test_a_work_that_runs_is_recorded(self) -> None:
        self.touch(seconds_ago=3600)
        result = idle_window.run_once(self.store, lambda: "did the thing", now=NOW, note="consolidate")
        self.assertTrue(result["ran"])
        self.assertEqual(result["status"], "done")
        self.assertEqual(idle_window.history(self.store)[0]["note"], "consolidate")

    def test_a_busy_store_does_not_run_the_work(self) -> None:
        self.touch(seconds_ago=5)
        calls: list[int] = []
        result = idle_window.run_once(self.store, lambda: calls.append(1), now=NOW)
        self.assertFalse(result["ran"])
        self.assertEqual(calls, [])

    def test_a_failure_is_recorded_and_the_lock_released(self) -> None:
        self.touch(seconds_ago=3600)

        def boom() -> None:
            raise RuntimeError("consolidation exploded")

        result = idle_window.run_once(self.store, boom, now=NOW)
        self.assertTrue(result["ran"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("exploded", idle_window.history(self.store)[0]["note"])

    def test_a_running_lock_blocks_a_second_run(self) -> None:
        self.touch(seconds_ago=3600)
        idle_window._ensure(self.store)
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO idle_runs (run_id, started_at, status) VALUES ('busy', ?, 'running')",
                ((NOW - timedelta(seconds=30)).isoformat(),),
            )
        verdict = idle_window.should_run(self.store, now=NOW)
        self.assertFalse(verdict.run)
        self.assertEqual(verdict.lock_holder, "busy")

    def test_a_stale_lock_expires(self) -> None:
        """A crashed run must not wedge the pass forever."""
        self.touch(seconds_ago=10_000)
        idle_window._ensure(self.store)
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO idle_runs (run_id, started_at, status) VALUES ('stale', ?, 'running')",
                ((NOW - timedelta(hours=5)).isoformat(),),
            )
        self.assertTrue(idle_window.should_run(self.store, now=NOW).run)

    def test_two_runs_in_a_row_both_finish(self) -> None:
        self.touch(seconds_ago=3600)
        first = idle_window.run_once(self.store, lambda: 1, now=NOW, note="a")
        second = idle_window.run_once(self.store, lambda: 2, now=NOW + timedelta(seconds=5), note="b")
        self.assertTrue(first["ran"] and second["ran"])
        self.assertEqual(len(idle_window.history(self.store)), 2)

    def test_the_report_counts_statuses(self) -> None:
        self.touch(seconds_ago=3600)
        idle_window.run_once(self.store, lambda: None, now=NOW, note="a")
        idle_window.run_once(self.store, lambda: None, now=NOW + timedelta(seconds=1), note="b")
        described = idle_window.report(self.store, now=NOW + timedelta(seconds=2))
        self.assertEqual(described["runs"], 2)
        self.assertEqual(described["by_status"]["done"], 2)

    def test_the_report_says_why_it_would_not_run(self) -> None:
        self.touch(seconds_ago=1)
        described = idle_window.report(self.store, now=NOW)
        self.assertFalse(described["run"])
        self.assertIn("busy", described["reason"])

    def test_the_activity_round_trips(self) -> None:
        self.touch(seconds_ago=120)
        payload = idle_window.activity(self.store, now=NOW).as_dict()
        self.assertIn("idle_seconds", payload)
        self.assertIsInstance(payload["sources"], dict)

    def test_an_unparseable_timestamp_is_ignored(self) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO episodes (session_id, user_content, assistant_content, created_at,
                                         content_hash)
                   VALUES ('s', 'x', 'y', 'not a date', 'h-bad')""",
            )
        self.assertEqual(idle_window.activity(self.store, now=NOW).last_seen, "")


if __name__ == "__main__":
    unittest.main()
