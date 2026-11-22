"""Decay follows how fast a fact goes stale (#54)."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import decay, volatility
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


def row(*, kind: str = "semantic", volatility_class: str | None = None,
        age_days: float = 30.0, used: int = 0) -> dict:
    return {
        "kind": kind,
        "volatility": volatility_class,
        "last_used_at": (NOW - timedelta(days=age_days)).isoformat(),
        "created_at": (NOW - timedelta(days=age_days)).isoformat(),
        "used_count": used,
    }


class HalfLifeTests(unittest.TestCase):
    def test_each_class_has_its_own_half_life(self) -> None:
        self.assertLess(decay._half_life("ephemeral"), decay._half_life("seasonal"))
        self.assertLess(decay._half_life("seasonal"), decay._half_life("durable"))

    def test_an_unknown_class_falls_back_to_the_default(self) -> None:
        self.assertEqual(decay._half_life("mystery"), decay.HALF_LIFE_DAYS)
        self.assertEqual(decay._half_life(None), decay.HALF_LIFE_DAYS)

    def test_an_explicit_override_wins(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_DECAY_HALF_LIFE_DAYS": "3"}):
            self.assertEqual(decay._half_life("durable"), 3.0)

    def test_a_malformed_override_is_ignored(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_DECAY_HALF_LIFE_DAYS": "soon"}):
            self.assertEqual(decay._half_life("ephemeral"), 7.0)


class AppliesTests(unittest.TestCase):
    def test_decay_is_off_unless_enabled(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_DECAY": ""}):
            self.assertFalse(decay.applies(row(kind="episode")))

    def test_an_episode_decays_when_enabled(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_DECAY": "1"}):
            self.assertTrue(decay.applies(row(kind="episode")))

    def test_a_volatile_semantic_fact_decays_when_enabled(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_DECAY": "1"}):
            self.assertTrue(decay.applies(row(kind="semantic", volatility_class="ephemeral")))

    def test_an_unclassified_semantic_fact_does_not_decay(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_DECAY": "1"}):
            self.assertFalse(decay.applies(row(kind="semantic")))

    def test_a_preference_still_never_decays(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_DECAY": "1"}):
            self.assertFalse(decay.applies(row(kind="preference")))


class ShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["CORTEX_DECAY"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("CORTEX_DECAY", None)

    def test_an_ephemeral_fact_fades_fastest(self) -> None:
        ephemeral = decay.strength(row(volatility_class="ephemeral", age_days=30), at=NOW)
        seasonal = decay.strength(row(volatility_class="seasonal", age_days=30), at=NOW)
        durable = decay.strength(row(volatility_class="durable", age_days=30), at=NOW)
        self.assertLess(ephemeral, seasonal)
        self.assertLess(seasonal, durable)

    def test_a_durable_fact_plateaus(self) -> None:
        """A year later a stable fact is still mostly there — that is what plateau means."""
        strength = decay.strength(row(volatility_class="durable", age_days=365), at=NOW)
        self.assertGreater(strength, 0.6)

    def test_an_ephemeral_fact_is_mostly_gone_in_a_month(self) -> None:
        strength = decay.strength(row(volatility_class="ephemeral", age_days=30), at=NOW)
        self.assertLess(strength, 0.2)

    def test_reinforcement_slows_the_decay(self) -> None:
        fresh = decay.strength(row(volatility_class="ephemeral", age_days=30, used=0), at=NOW)
        used = decay.strength(row(volatility_class="ephemeral", age_days=30, used=5), at=NOW)
        self.assertGreater(used, fresh)

    def test_the_strength_never_leaves_the_floor(self) -> None:
        self.assertGreaterEqual(
            decay.strength(row(volatility_class="ephemeral", age_days=10_000), at=NOW),
            decay.FLOOR,
        )

    def test_the_score_weight_is_bounded(self) -> None:
        worst = decay.score_weight(row(volatility_class="ephemeral", age_days=10_000), at=NOW)
        self.assertGreaterEqual(worst, 1.0 - decay.WEIGHT)

    def test_the_same_row_without_a_class_does_not_decay(self) -> None:
        self.assertEqual(decay.strength(row(kind="semantic", age_days=365), at=NOW), 1.0)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_the_class_comes_from_the_volatility_table(self) -> None:
        memory_id = self.store.add_memory(
            content="I am on call this week.", kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]
        assigned = volatility.assign(self.store, memory_id, "I am on call this week.",
                                     kind="semantic")
        self.assertEqual(assigned.volatility, volatility.EPHEMERAL)
        self.assertEqual(volatility.lookup(self.store, memory_id), volatility.EPHEMERAL)

    def test_describe_still_works_without_volatility(self) -> None:
        self.store.add_memory(
            content="A plain fact.", kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )
        described = decay.describe(self.store)
        self.assertIn("episodes", described)


if __name__ == "__main__":
    unittest.main()
