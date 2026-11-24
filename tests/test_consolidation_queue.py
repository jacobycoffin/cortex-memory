"""Two speeds: drain a little while working, sweep when quiet (#209)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import consolidation_queue as queue
from cortex.store import CortexStore

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str) -> str:
        return self.store.add_memory(
            content=text, kind="semantic", source_type="user_turn", source_category="USER_STATED",
        )[0]

    def touch(self, *, seconds_ago: float) -> None:
        moment = (NOW - timedelta(seconds=seconds_ago)).isoformat()
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO episodes (session_id, user_content, assistant_content, created_at,
                                         content_hash)
                   VALUES ('s', 'a turn', 'ok', ?, ?)""",
                (moment, f"h-{seconds_ago}"),
            )

    def test_enqueuing_keeps_a_reference_only(self) -> None:
        memory_id = self.memory("Something worth consolidating.")
        self.assertEqual(queue.enqueue(self.store, [memory_id], reason="written", at=NOW), 1)
        rows = queue.queued(self.store)
        self.assertEqual([row["memory_id"] for row in rows], [memory_id])
        self.assertEqual(rows[0]["reason"], "written")

    def test_enqueuing_the_same_memory_twice_adds_one_row(self) -> None:
        memory_id = self.memory("Something worth consolidating.")
        queue.enqueue(self.store, [memory_id], at=NOW)
        self.assertEqual(queue.enqueue(self.store, [memory_id], at=NOW), 0)
        self.assertEqual(queue.depth(self.store)["queued"], 1)

    def test_the_fast_drain_is_bounded_by_item_count(self) -> None:
        ids = [self.memory(f"fact {index}") for index in range(8)]
        queue.enqueue(self.store, ids, at=NOW)
        seen: list[str] = []
        found = queue.drain(self.store, seen.append, max_items=3, budget_ms=10 ** 6, now=NOW)
        self.assertEqual(found.done, 3)
        self.assertEqual(len(seen), 3)
        self.assertEqual(queue.depth(self.store)["queued"], 5)

    def test_the_fast_drain_stops_on_the_time_budget(self) -> None:
        ids = [self.memory(f"fact {index}") for index in range(4)]
        queue.enqueue(self.store, ids, at=NOW)

        def slow(memory_id: str) -> None:
            import time
            time.sleep(0.02)

        found = queue.drain(self.store, slow, max_items=4, budget_ms=10, now=NOW)
        self.assertTrue(found.stopped_on_budget)
        self.assertLess(found.items, 4)

    def test_a_failure_is_retried_not_dropped(self) -> None:
        memory_id = self.memory("A fact.")
        queue.enqueue(self.store, [memory_id], at=NOW)

        def boom(_memory_id: str) -> None:
            raise RuntimeError("consolidation exploded")

        found = queue.drain(self.store, boom, now=NOW)
        self.assertEqual(found.failed, 1)
        self.assertEqual(found.parked, 0)
        self.assertEqual(queue.depth(self.store)["queued"], 1)

    def test_after_the_attempt_ceiling_the_item_is_parked_with_its_error(self) -> None:
        memory_id = self.memory("A fact.")
        queue.enqueue(self.store, [memory_id], at=NOW)

        def boom(_memory_id: str) -> None:
            raise RuntimeError("still exploding")

        for _ in range(queue.MAX_ATTEMPTS):
            queue.drain(self.store, boom, now=NOW)
        counts = queue.depth(self.store)
        self.assertEqual(counts.get("parked"), 1)
        self.assertEqual(counts.get("queued", 0), 0)
        row = self.store._conn.execute(
            "SELECT last_error FROM consolidation_queue WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        self.assertIn("exploding", row["last_error"])

    def test_a_parked_item_is_requeued_when_it_is_written_again(self) -> None:
        """New evidence deserves another attempt; leaving it parked would silently drop it."""
        memory_id = self.memory("A fact.")
        queue.enqueue(self.store, [memory_id], at=NOW)
        with self.store.transaction() as conn:
            conn.execute("UPDATE consolidation_queue SET state = 'parked' WHERE memory_id = ?",
                         (memory_id,))
        queue.enqueue(self.store, [memory_id], reason="written again", at=NOW)
        self.assertEqual(queue.depth(self.store).get("queued"), 1)

    def test_the_sweep_refuses_while_the_store_is_busy(self) -> None:
        self.touch(seconds_ago=5)
        result = queue.sweep(self.store, lambda _memory_id: None, now=NOW)
        self.assertFalse(result["ran"])
        self.assertIn("busy", result["reason"])

    def test_the_sweep_runs_when_quiet_and_drains_everything(self) -> None:
        self.touch(seconds_ago=3600)
        ids = [self.memory(f"fact {index}") for index in range(6)]
        queue.enqueue(self.store, ids, at=NOW)
        seen: list[str] = []
        result = queue.sweep(self.store, seen.append, min_idle_seconds=600, now=NOW)
        self.assertTrue(result["ran"])
        self.assertEqual(len(seen), 6)
        self.assertEqual(queue.depth(self.store).get("done"), 6)

    def test_the_sweep_holds_the_idle_lock(self) -> None:
        self.touch(seconds_ago=3600)
        queue.sweep(self.store, lambda _memory_id: None, now=NOW)
        self.assertEqual(len(__import__("cortex.idle_window", fromlist=["history"]).history(
            self.store)), 1)

    def test_the_report_states_its_limits(self) -> None:
        described = queue.report(self.store)
        self.assertEqual(described["limits"]["max_attempts"], queue.MAX_ATTEMPTS)
        self.assertIn("references", described["note"])

    def test_an_empty_store_is_quiet(self) -> None:
        self.assertEqual(queue.enqueue(self.store, []), 0)
        self.assertEqual(queue.drain(self.store, lambda _memory_id: None, now=NOW).describe(),
                         "nothing queued")
        self.assertEqual(queue.report(self.store)["queued"], 0)


if __name__ == "__main__":
    unittest.main()
