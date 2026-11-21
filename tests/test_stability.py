"""Stability must beat freshness on ties, without ever beating relevance (item #9)."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.stability import (
    HALF_LIFE_DAYS,
    MAX_BOOST,
    apply_stability_blend,
    corroboration,
    stability,
)

NOW = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)


def memory(**overrides: Any) -> dict[str, Any]:
    base = {
        "id": "m",
        "confirmed_count": 0,
        "validated_count": 0,
        "helpful_count": 0,
        "last_helpful_at": None,
        "last_used_at": None,
        "observed_at": NOW.isoformat(),
        "updated_at": NOW.isoformat(),
        "created_at": NOW.isoformat(),
    }
    base.update(overrides)
    return base


@dataclass
class Fake:
    memory: dict[str, Any]
    score: float
    components: dict[str, float] = field(default_factory=dict)


class CorroborationTests(unittest.TestCase):
    def test_supports_are_summed(self) -> None:
        self.assertEqual(corroboration(memory(confirmed_count=2, helpful_count=1)), 3)

    def test_missing_and_bad_values_are_ignored(self) -> None:
        self.assertEqual(corroboration({"confirmed_count": "x", "helpful_count": -4}), 0)

    def test_retrieval_counters_do_not_count(self) -> None:
        self.assertEqual(corroboration({"retrieved_count": 900, "used_count": 300}), 0)


class StabilityTests(unittest.TestCase):
    def test_uncorroborated_memory_has_no_stability(self) -> None:
        self.assertEqual(stability(memory(), now=NOW), 0.0)

    def test_one_support_is_half_weight(self) -> None:
        single = stability(memory(confirmed_count=1), now=NOW)
        five = stability(memory(confirmed_count=5), now=NOW)
        self.assertAlmostEqual(five, 1.0, places=3)
        self.assertLess(single, five / 2 + 0.05)

    def test_more_supports_saturate(self) -> None:
        five = stability(memory(confirmed_count=5), now=NOW)
        fifty = stability(memory(confirmed_count=50), now=NOW)
        self.assertLessEqual(fifty, 1.0)
        self.assertGreater(fifty, five - 1e-9)

    def test_old_support_decays(self) -> None:
        fresh = stability(memory(confirmed_count=5), now=NOW)
        old_stamp = (NOW - timedelta(days=HALF_LIFE_DAYS)).isoformat()
        aged = stability(
            memory(
                confirmed_count=5,
                observed_at=old_stamp,
                updated_at=old_stamp,
                created_at=old_stamp,
            ),
            now=NOW,
        )
        self.assertAlmostEqual(aged, fresh / 2, places=2)

    def test_repeatedly_confirmed_beats_one_fresh_statement(self) -> None:
        """The whole point: one recent sentence does not bury a year of confirmation."""
        stale = (NOW - timedelta(days=200)).isoformat()
        established = memory(
            confirmed_count=6,
            last_helpful_at=stale,
            observed_at=stale,
            updated_at=stale,
            created_at=stale,
        )
        one_off = memory(confirmed_count=1)
        self.assertGreater(stability(established, now=NOW), stability(one_off, now=NOW))

    def test_stability_is_bounded(self) -> None:
        self.assertLessEqual(stability(memory(confirmed_count=10_000), now=NOW), 1.0)


class BlendTests(unittest.TestCase):
    def test_boost_is_small_and_labelled(self) -> None:
        result = Fake(memory(confirmed_count=5), 0.50)
        out, count = apply_stability_blend([result], now=NOW)
        self.assertEqual(count, 1)
        self.assertAlmostEqual(out[0].score, 0.50 + MAX_BOOST, places=6)
        self.assertIn("stability", out[0].components)

    def test_boost_cannot_beat_a_real_relevance_gap(self) -> None:
        stable = Fake(memory(confirmed_count=50), 0.50)
        relevant = Fake(memory(), 0.60)
        out, _ = apply_stability_blend([stable, relevant], now=NOW)
        by_score = sorted(out, key=lambda r: -r.score)
        self.assertEqual(by_score[0].score, 0.60)
        self.assertLess(by_score[1].score, 0.60)

    def test_input_is_not_mutated(self) -> None:
        result = Fake(memory(confirmed_count=3), 0.4)
        apply_stability_blend([result], now=NOW)
        self.assertEqual(result.score, 0.4)
        self.assertEqual(result.components, {})

    def test_unstable_results_pass_through(self) -> None:
        result = Fake(memory(), 0.4)
        out, count = apply_stability_blend([result], now=NOW)
        self.assertEqual((count, out[0].score), (0, 0.4))


if __name__ == "__main__":
    unittest.main()
