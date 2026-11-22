"""Episodic decay is off by default, applies only to episodes, and rewards rehearsal (#93)."""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import decay
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


def episode(days_old: float, *, used: int = 0, kind: str = "episode") -> dict:
    return {
        "id": "m1",
        "kind": kind,
        "content": "A note about Tuesday.",
        "created_at": (NOW - timedelta(days=days_old)).isoformat(),
        "last_used_at": (NOW - timedelta(days=days_old)).isoformat(),
        "used_count": used,
    }


class EnabledByDefaultOffTests(unittest.TestCase):
    def test_decay_is_off_unless_asked_for(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CORTEX_DECAY", None)
            self.assertFalse(decay.enabled())
            self.assertEqual(decay.strength(episode(60), at=NOW), 1.0)
            self.assertEqual(decay.penalty(episode(60), at=NOW), 0.0)
            self.assertEqual(decay.score_weight(episode(60), at=NOW), 1.0)

    def test_it_can_be_switched_on(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_DECAY": "1"}):
            self.assertTrue(decay.enabled())
            self.assertLess(decay.strength(episode(60), at=NOW), 1.0)


class KindScopeTests(unittest.TestCase):
    def test_only_episodic_kinds_decay(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_DECAY": "1"}):
            for kind in ("semantic", "procedural", "preference", "identity", "commitment"):
                row = episode(300, kind=kind)
                self.assertEqual(decay.strength(row, at=NOW), 1.0, kind)
                self.assertEqual(decay.penalty(row, at=NOW), 0.0, kind)
            for kind in ("episode", "episodic"):
                self.assertLess(decay.strength(episode(300, kind=kind), at=NOW), 1.0)

    def test_a_preference_does_not_become_less_true_next_month(self) -> None:
        """The reason the scope is narrow: the model is only defensible for working notes."""
        with mock.patch.dict(os.environ, {"CORTEX_DECAY": "1"}):
            self.assertEqual(decay.score_weight(episode(3650, kind="preference"), at=NOW), 1.0)


class CurveTests(unittest.TestCase):
    def test_retention_falls_with_time(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_DECAY": "1"}):
            fresh = decay.strength(episode(0), at=NOW)
            week = decay.strength(episode(7), at=NOW)
            month = decay.strength(episode(30), at=NOW)
            self.assertAlmostEqual(fresh, 1.0, places=6)
            self.assertGreater(fresh, week)
            self.assertGreater(week, month)

    def test_one_half_life_leaves_about_half(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_DECAY": "1"}):
            value = decay.strength(episode(decay.HALF_LIFE_DAYS), at=NOW)
            self.assertAlmostEqual(value, 0.5, delta=0.03)

    def test_the_curve_never_reaches_zero(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_DECAY": "1"}):
            self.assertEqual(decay.strength(episode(5000), at=NOW), decay.FLOOR)

    def test_rehearsal_flattens_the_curve(self) -> None:
        """The difference between a forgetting curve and a countdown timer."""
        with mock.patch.dict(os.environ, {"CORTEX_DECAY": "1"}):
            never = decay.strength(episode(30, used=0), at=NOW)
            rehearsed = decay.strength(episode(30, used=5), at=NOW)
            self.assertGreater(rehearsed, never)

    def test_stability_grows_logarithmically(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_DECAY": "1"}):
            base = decay.stability_days({"used_count": 0})
            more = decay.stability_days({"used_count": len(str(int(math.e**2))) + 6})
            self.assertGreater(more, base)

    def test_a_memory_with_no_timestamp_does_not_decay(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_DECAY": "1"}):
            self.assertEqual(decay.strength({"id": "x", "kind": "episode"}, at=NOW), 1.0)

    def test_the_half_life_is_configurable(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_DECAY": "1",
                                          "CORTEX_DECAY_HALF_LIFE_DAYS": "1"}):
            one_day = decay.strength(episode(1), at=NOW)
            self.assertAlmostEqual(one_day, 0.5, delta=0.03)


class ScoringTests(unittest.TestCase):
    def test_the_penalty_is_small_and_bounded(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_DECAY": "1"}):
            worst = decay.score_weight(episode(9999), at=NOW)
            self.assertGreater(worst, 0.85)
            self.assertLess(worst, 1.0)


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_the_report_counts_only_episodes(self) -> None:
        for index, kind in enumerate(("episode", "episode", "semantic")):
            created = self.store.add_memory(
                content=f"fact {index}",
                kind=kind,
                source_type="user_turn",
                source_category="USER_EXPLICIT",
            )[0]
            with self.store.transaction() as conn:
                conn.execute(
                    "UPDATE memories SET created_at = ?, last_used_at = ? WHERE id = ?",
                    ((NOW - timedelta(days=90)).isoformat(),
                     (NOW - timedelta(days=90)).isoformat(), created),
                )
        with mock.patch.dict(os.environ, {"CORTEX_DECAY": "1"}):
            described = decay.describe(self.store, at=NOW)
            self.assertEqual(described["episodes"], 2)
            self.assertEqual(described["faded_below_half"], 2)
            self.assertTrue(described["enabled"])

    def test_the_report_is_empty_when_decay_is_off(self) -> None:
        self.store.add_memory(
            content="an episode",
            kind="episode",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
        )
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CORTEX_DECAY", None)
            described = decay.describe(self.store, at=NOW)
            self.assertEqual(described["episodes"], 0)
            self.assertFalse(described["enabled"])


if __name__ == "__main__":
    unittest.main()
