"""Replay scheduling (#107): bounded, regret first, frequency counts against replay, deterministic order."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import replay_schedule
from cortex.store import CortexStore

NOW = datetime(2026, 11, 24, 3, 0, tzinfo=timezone.utc)


class ScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str = "the proxmox box runs 8.4", **kwargs) -> str:
        return self.store.add_memory(
            content=text, kind=kwargs.pop("kind", "semantic"),
            source_type="user_turn", source_category="USER_STATED", **kwargs,
        )[0]

    def touch(self, memory_id: str, *, used: int = 0, retrieved: int = 0, days_ago: float = 1) -> None:
        stamp = (NOW - timedelta(days=days_ago)).isoformat()
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE memories SET used_count = ?, retrieved_count = ?, last_used_at = ?, "
                "last_retrieved_at = ? WHERE id = ?",
                (used, retrieved, stamp if used else None, stamp, memory_id),
            )

    def regret(self, memory_id: str, score: float = 1.0) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO pruning_regret (memory_id, query, score, restored, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (memory_id, "a query", score, 1, NOW.isoformat()),
            )

    def state(self, memory_id: str, state: str) -> None:
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET state = ? WHERE id = ?", (state, memory_id))

    # -- the budget ------------------------------------------------------------------

    def test_the_budget_is_respected_and_the_overflow_is_stated(self) -> None:
        for index in range(10):
            self.memory(f"memory number {index} about the shelf")
        plan = replay_schedule.schedule(self.store, per_night=3, now=NOW)
        self.assertEqual(len(plan.entries), 3)
        self.assertGreater(plan.deferred, 0)
        self.assertTrue(any("deferred" in line for line in plan.render()))

    def test_a_zero_budget_schedules_nothing_and_says_so(self) -> None:
        self.memory()
        plan = replay_schedule.schedule(self.store, per_night=0, now=NOW)
        self.assertEqual(plan.entries, [])
        self.assertEqual(plan.deferred, plan.candidates)

    # -- frequency is not value ------------------------------------------------------

    def test_a_never_used_memory_outranks_a_frequently_used_one(self) -> None:
        idle = self.memory("a memory nothing ever asks about")
        busy = self.memory("a memory the assistant reaches for constantly")
        self.touch(busy, used=40, retrieved=131, days_ago=1)
        self.touch(idle, used=0, retrieved=0, days_ago=40)
        plan = replay_schedule.schedule(self.store, per_night=2, now=NOW)
        order = [entry.memory_id for entry in plan.entries]
        self.assertLess(order.index(idle), order.index(busy))

    def test_recent_use_lowers_priority(self) -> None:
        fresh = self.memory("used yesterday")
        old = self.memory("used two months ago")
        self.touch(fresh, used=2, retrieved=5, days_ago=1)
        self.touch(old, used=2, retrieved=5, days_ago=60)
        plan = replay_schedule.schedule(self.store, per_night=2, now=NOW)
        order = [entry.memory_id for entry in plan.entries]
        self.assertLess(order.index(old), order.index(fresh))

    # -- regret outranks -------------------------------------------------------------

    def test_a_regretted_memory_is_scheduled_first_with_its_signal(self) -> None:
        regretted = self.memory("a memory that was pruned and later needed")
        self.touch(regretted, used=10, retrieved=20, days_ago=1)  # heavily used, so it would otherwise rank low
        for index in range(5):
            self.memory(f"ordinary memory {index}")
        self.regret(regretted, score=1.0)
        plan = replay_schedule.schedule(self.store, per_night=3, now=NOW)
        self.assertEqual(plan.entries[0].memory_id, regretted)
        self.assertGreater(plan.entries[0].signals["regret"], 0)
        self.assertIn("regret", plan.entries[0].reason)

    def test_regret_accumulates_rather_than_overwriting(self) -> None:
        memory_id = self.memory("pruned twice")
        self.regret(memory_id, score=1.0)
        self.regret(memory_id, score=0.5)
        plan = replay_schedule.schedule(self.store, per_night=1, now=NOW)
        self.assertAlmostEqual(plan.entries[0].signals["regret"], 1.5, places=3)

    # -- what is never replayed ------------------------------------------------------

    def test_archived_quarantined_and_tombstoned_are_not_scheduled(self) -> None:
        for index, state in enumerate(("archived", "quarantine", "tombstoned")):
            memory_id = self.memory(f"memory in state {state}")
            self.state(memory_id, state)
        plan = replay_schedule.schedule(self.store, per_night=10, now=NOW)
        self.assertEqual(plan.entries, [], "a state is a decision, not an oversight")
        self.assertIn("archived", plan.excluded)

    def test_cold_memories_are_replayable(self) -> None:
        """Cold is 'set aside', not 'decided against': a replay is exactly how one earns its way back."""
        memory_id = self.memory("a cold memory worth another look")
        self.state(memory_id, "cold")
        plan = replay_schedule.schedule(self.store, per_night=5, now=NOW)
        self.assertIn(memory_id, [entry.memory_id for entry in plan.entries])

    # -- determinism and honesty -----------------------------------------------------

    def test_the_order_is_deterministic_for_equal_scores(self) -> None:
        ids = [self.memory(f"identical memory {index}") for index in range(6)]
        first = [entry.memory_id for entry in replay_schedule.schedule(self.store, per_night=6, now=NOW).entries]
        second = [entry.memory_id for entry in replay_schedule.schedule(self.store, per_night=6, now=NOW).entries]
        self.assertEqual(first, second, "a schedule that reshuffles cannot be checked against what ran")
        self.assertEqual(sorted(first), sorted(ids))

    def test_every_entry_states_the_signal_that_put_it_there(self) -> None:
        self.memory("a plain memory")
        for entry in replay_schedule.schedule(self.store, per_night=5, now=NOW).entries:
            self.assertTrue(entry.reason)
            self.assertTrue(entry.signals)

    def test_an_empty_store_produces_an_honest_empty_schedule(self) -> None:
        plan = replay_schedule.schedule(self.store, now=NOW)
        self.assertTrue(plan.readable)
        self.assertEqual(plan.entries, [])
        self.assertIn("nothing to replay", plan.render()[0])

    def test_scheduling_writes_nothing(self) -> None:
        self.memory()
        tables = tuple(sorted(r["name"] for r in self.store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")))
        counts = tuple(int(self.store._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
                       for t in tables)
        replay_schedule.schedule(self.store, now=NOW)
        self.assertEqual((tables, tuple(int(self.store._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
                                        for t in tables)), (tables, counts))

    def test_a_broken_store_says_it_could_not_read(self) -> None:
        class Broken:
            _conn = None

        plan = replay_schedule.schedule(Broken(), now=NOW)
        self.assertFalse(plan.readable)
        self.assertIn("could not be read", plan.render()[0])

    def test_the_report_counts_what_the_schedule_contains(self) -> None:
        regretted = self.memory("pruned before")
        self.regret(regretted)
        for index in range(4):
            self.memory(f"ordinary {index}")
        summary = replay_schedule.report(self.store, per_night=3, now=NOW)
        self.assertEqual(summary["scheduled"], 3)
        self.assertEqual(summary["with_regret"], 1)
        self.assertGreaterEqual(summary["never_used_scheduled"], 1)


if __name__ == "__main__":
    unittest.main()
