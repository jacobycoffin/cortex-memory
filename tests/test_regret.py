"""A memory that came back widens the window it was forgotten through (#90)."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import forget_ladder as ladder
from cortex import regret
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class RegretTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str = "A fact.", *, kind: str = "semantic",
               quiet_days: float = 200, state: str = "active") -> str:
        created = self.store.add_memory(
            content=content,
            kind=kind,
            source_type="user_turn",
            source_category="USER_EXPLICIT",
        )
        memory_id = created[0] if isinstance(created, tuple) else created
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE memories SET created_at = ?, last_used_at = ? WHERE id = ?",
                ((NOW - timedelta(days=900)).isoformat(),
                 (NOW - timedelta(days=quiet_days)).isoformat(), memory_id),
            )
        if state != "active":
            self.store.set_state(memory_id, state, reason="fixture")
        return memory_id

    def revival(self, memory_id: str, *, from_state: str = "cold",
                days_ago: float = 1) -> None:
        """Simulate the ladder having moved it down, then a human restoring it."""
        stamp = (NOW - timedelta(days=days_ago)).isoformat()
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO lifecycle_events(memory_id, from_state, to_state, reason,
                                                retention_score, created_at)
                   VALUES (?, ?, 'active', 'restored', NULL, ?)""",
                (memory_id, from_state, stamp),
            )


class FactorTests(RegretTestCase):
    def test_no_regrets_means_no_change(self) -> None:
        self.assertEqual(regret.factor(self.store, "semantic"), 1.0)

    def test_each_revival_adds_a_step(self) -> None:
        first = self.memory("one", kind="semantic")
        second = self.memory("two", kind="semantic")
        self.revival(first)
        self.assertEqual(regret.factor(self.store, "semantic"), 1.25)
        self.revival(second)
        self.assertEqual(regret.factor(self.store, "semantic"), 1.5)

    def test_the_factor_is_capped(self) -> None:
        for index in range(20):
            self.revival(self.memory(f"fact {index}", kind="semantic"))
        self.assertEqual(regret.factor(self.store, "semantic"), regret.CAP)

    def test_only_the_kind_that_was_forgotten_is_widened(self) -> None:
        self.revival(self.memory("episode", kind="episode"))
        self.assertEqual(regret.factor(self.store, "episode"), 1.25)
        self.assertEqual(regret.factor(self.store, "semantic"), 1.0)

    def test_an_old_revival_stops_counting(self) -> None:
        self.revival(self.memory("old", kind="semantic"), days_ago=400)
        self.assertEqual(regret.revivals(self.store), {})
        self.assertEqual(regret.factor(self.store, "semantic"), 1.0)

    def test_a_promotion_is_not_a_revival(self) -> None:
        """Coming *up* from active is not regret; only coming back from a demotion is."""
        memory_id = self.memory("promoted", kind="semantic")
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO lifecycle_events(memory_id, from_state, to_state, reason,
                                                retention_score, created_at)
                   VALUES (?, 'active', 'active', 'noop', NULL, ?)""",
                (memory_id, NOW.isoformat()),
            )
        self.assertEqual(regret.revivals(self.store), {})

    def test_the_guard_can_be_switched_off(self) -> None:
        self.revival(self.memory("one", kind="semantic"))
        with mock.patch.dict(os.environ, {"CORTEX_FORGET_REGRET": "0"}):
            self.assertEqual(regret.factor(self.store, "semantic"), 1.0)


class WindowTests(RegretTestCase):
    def test_windows_scale_up(self) -> None:
        self.assertEqual(regret.windows((90, 180, 365), 1.5), (135, 270, 548))

    def test_windows_never_shrink(self) -> None:
        self.assertEqual(regret.windows((90, 180, 365), 0.5), (90, 180, 365))


class LadderIntegrationTests(RegretTestCase):
    def test_a_widened_window_keeps_a_memory_out_of_the_sweep(self) -> None:
        memory_id = self.memory("borderline", kind="semantic", quiet_days=100)
        warm = ladder.plan(self.store, at=NOW)
        self.assertEqual([step.memory_id for step in warm.steps], [memory_id])
        self.revival(memory_id)
        held = ladder.plan(self.store, at=NOW)
        self.assertEqual(held.steps, [])

    def test_the_widened_window_is_visible_in_explain(self) -> None:
        memory_id = self.memory("borderline", kind="semantic", quiet_days=100)
        self.revival(memory_id)
        explained = ladder.explain(self.store, memory_id, at=NOW)
        # 90/180/365 × 1.25, rounded — the exact numbers matter less than that they grew
        self.assertEqual(explained["dwell_days"], [112, 225, 456])
        self.assertIn("warm", explained["verdict"])

    def test_the_guard_is_reported_in_the_policy(self) -> None:
        described = ladder.policy()
        self.assertIn("regret_guard", described)
        self.assertEqual(described["regret_guard"]["cap"], regret.CAP)

    def test_a_memory_still_moves_when_it_is_genuinely_ancient(self) -> None:
        """The guard delays forgetting; it must not prevent it forever."""
        memory_id = self.memory("ancient", kind="semantic", quiet_days=900)
        self.revival(memory_id)
        sweep = ladder.plan(self.store, at=NOW)
        self.assertEqual([step.memory_id for step in sweep.steps], [memory_id])


if __name__ == "__main__":
    unittest.main()
