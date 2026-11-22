"""Dormancy: offered many times, never used, so demote sooner (#94)."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import dormancy
from cortex import forget_ladder as ladder
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class AssessTests(unittest.TestCase):
    def test_one_retrieval_without_use_is_not_a_pattern(self) -> None:
        result = dormancy.assess({"retrieved_count": 1, "used_count": 0})
        self.assertFalse(result.dormant)

    def test_a_high_retrieval_count_without_use_is(self) -> None:
        result = dormancy.assess({"retrieved_count": 14, "used_count": 0})
        self.assertTrue(result.dormant)
        self.assertEqual(result.ratio, 1.0)

    def test_use_brings_the_ratio_down(self) -> None:
        used_often = dormancy.assess({"retrieved_count": 14, "used_count": 13})
        never = dormancy.assess({"retrieved_count": 14, "used_count": 0})
        self.assertFalse(used_often.dormant)
        self.assertTrue(never.dormant)

    def test_injections_count_as_offers(self) -> None:
        result = dormancy.assess({"retrieved_count": 2, "injected_count": 12, "used_count": 0})
        self.assertEqual(result.retrieved, 14)
        self.assertTrue(result.dormant)

    def test_garbage_counters_do_not_crash(self) -> None:
        result = dormancy.assess({"retrieved_count": "many", "used_count": None})
        self.assertFalse(result.dormant)

    def test_the_minimum_sample_is_configurable(self) -> None:
        row = {"retrieved_count": 2, "used_count": 0}
        self.assertFalse(dormancy.assess(row).dormant)
        with mock.patch.dict(os.environ, {"CORTEX_DORMANT_MIN_RETRIEVED": "2"}):
            self.assertTrue(dormancy.assess(row).dormant)

    def test_dormancy_can_be_switched_off(self) -> None:
        row = {"retrieved_count": 20, "used_count": 0}
        self.assertEqual(dormancy.scale(row), dormancy.DORMANT_SCALE)
        with mock.patch.dict(os.environ, {"CORTEX_DORMANCY": "0"}):
            self.assertEqual(dormancy.scale(row), 1.0)


class WindowTests(unittest.TestCase):
    def test_a_dormant_memory_gets_a_shorter_window(self) -> None:
        row = {"retrieved_count": 14, "used_count": 0}
        self.assertEqual(
            dormancy.apply_to((90, 180, 365), row),
            (45, 90, 182),
        )

    def test_a_useful_memory_keeps_its_window(self) -> None:
        row = {"retrieved_count": 14, "used_count": 13}
        self.assertEqual(dormancy.apply_to((90, 180, 365), row), (90, 180, 365))

    def test_the_shortened_window_has_a_floor(self) -> None:
        row = {"retrieved_count": 14, "used_count": 0}
        self.assertEqual(dormancy.apply_to((4, 8, 12), row), (7, 7, 7))


class LadderIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, *, quiet_days: float, retrieved: int, used: int,
               kind: str = "semantic") -> str:
        memory_id = self.store.add_memory(
            content=f"fact retrieved {retrieved} used {used}",
            kind=kind,
            source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]
        with self.store.transaction() as conn:
            conn.execute(
                """UPDATE memories SET created_at = ?, last_used_at = ?,
                          retrieved_count = ?, injected_count = 0, used_count = ?
                   WHERE id = ?""",
                ((NOW - timedelta(days=900)).isoformat(),
                 (NOW - timedelta(days=quiet_days)).isoformat(), retrieved, used, memory_id),
            )
        return memory_id

    def test_noise_is_demoted_before_ordinary_memories(self) -> None:
        noisy = self.memory(quiet_days=50, retrieved=14, used=0)
        ordinary = self.memory(quiet_days=80, retrieved=0, used=0)
        steps = {step.memory_id: step for step in ladder.plan(self.store, at=NOW).steps}
        self.assertIn(noisy, steps)
        self.assertNotIn(ordinary, steps)

    def test_a_dormant_memory_is_not_skipped_entirely(self) -> None:
        noisy = self.memory(quiet_days=400, retrieved=14, used=0)
        steps = [step.memory_id for step in ladder.plan(self.store, at=NOW).steps]
        self.assertIn(noisy, steps)

    def test_protection_still_wins_over_dormancy(self) -> None:
        noisy = self.memory(quiet_days=400, retrieved=14, used=0)
        self.store.set_pinned(noisy, True)
        self.assertEqual(ladder.plan(self.store, at=NOW).steps, [])

    def test_a_never_walk_kind_is_not_demoted_by_noise(self) -> None:
        noisy = self.memory(quiet_days=400, retrieved=14, used=0, kind="procedural")
        self.assertEqual(ladder.plan(self.store, at=NOW).steps, [])

    def test_the_report_names_the_noisiest(self) -> None:
        self.memory(quiet_days=30, retrieved=40, used=0)
        self.memory(quiet_days=30, retrieved=20, used=19)
        described = dormancy.report(self.store)
        self.assertEqual(described["count"], 1)
        self.assertEqual(described["worst"][0]["retrieved"], 40)


if __name__ == "__main__":
    unittest.main()
