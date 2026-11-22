"""Importance comes from a rubric, not a flat 0.5 (#32)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import importance_rubric
from cortex.store import CortexStore


class BaseTests(unittest.TestCase):
    def test_identity_outranks_everything_else(self) -> None:
        value = importance_rubric.base("identity")
        for kind in ("preference", "semantic", "episode", "tool"):
            self.assertGreater(value, importance_rubric.base(kind), kind)

    def test_the_order_matches_the_rubric(self) -> None:
        order = ["identity", "decision", "preference", "procedure", "semantic", "episode"]
        values = [importance_rubric.base(kind) for kind in order]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_an_unknown_kind_gets_the_old_default(self) -> None:
        self.assertEqual(importance_rubric.base("hiccup"), importance_rubric.DEFAULT_BASE)

    def test_plural_and_singular_kinds_agree(self) -> None:
        self.assertEqual(importance_rubric.base("episode"),
                         importance_rubric.base("episodic"))
        self.assertEqual(importance_rubric.base("procedure"),
                         importance_rubric.base("procedural"))


class ScoreTests(unittest.TestCase):
    def test_a_flat_unknown_kind_scores_the_flat_default(self) -> None:
        scored = importance_rubric.score("Something happened.", "hiccup")
        self.assertAlmostEqual(scored.importance, 0.45, places=3)  # 0.5 base, -0.05 fragment

    def test_a_preference_scores_above_a_semantic_fact(self) -> None:
        preference = importance_rubric.score("Prefers short answers.", "preference")
        semantic = importance_rubric.score("The r630 hosts three containers.", "semantic")
        self.assertGreater(preference.importance, semantic.importance)

    def test_an_episode_scores_low(self) -> None:
        self.assertLess(
            importance_rubric.score("Fixed the tunnel this morning.", "episode").importance, 0.5
        )

    def test_a_durable_detail_raises_importance(self) -> None:
        plain = importance_rubric.score("The dashboard is reachable.", "semantic")
        with_port = importance_rubric.score("The dashboard runs on port 8100.", "semantic")
        self.assertGreater(with_port.importance, plain.importance)
        self.assertIn("durable_detail", with_port.modifiers)

    def test_an_ephemeral_marker_lowers_importance(self) -> None:
        scored = importance_rubric.score("The build is failing right now.", "semantic")
        self.assertIn("ephemeral", scored.modifiers)

    def test_a_hedged_statement_lowers_importance(self) -> None:
        scored = importance_rubric.score("I think the window is probably Sunday.", "semantic")
        self.assertIn("hedged", scored.modifiers)

    def test_user_stated_raises_importance(self) -> None:
        plain = importance_rubric.score("The port is 8100.", "semantic")
        stated = importance_rubric.score("The port is 8100.", "semantic",
                                         source_category="USER_EXPLICIT")
        self.assertGreater(stated.importance, plain.importance)

    def test_corroboration_raises_importance_but_is_capped(self) -> None:
        one = importance_rubric.score("The port is 8100.", "semantic", corroboration=1)
        many = importance_rubric.score("The port is 8100.", "semantic", corroboration=20)
        self.assertGreater(many.importance, one.importance)
        self.assertLessEqual(many.modifiers["corroborated"], 0.04)

    def test_modifiers_stay_inside_the_band(self) -> None:
        scored = importance_rubric.score(
            "I think the password is port 8100 right now, maybe.", "semantic",
            source_category="USER_EXPLICIT", corroboration=5,
        )
        self.assertLessEqual(abs(sum(scored.modifiers.values())),
                             importance_rubric.MODIFIER_BAND + 1e-6)

    def test_importance_is_clamped(self) -> None:
        for kind in ("identity", "episode"):
            value = importance_rubric.score(
                "port 8100 password api key", kind, source_category="USER_EXPLICIT",
                corroboration=9,
            ).importance
            self.assertGreaterEqual(value, importance_rubric.MIN_IMPORTANCE)
            self.assertLessEqual(value, importance_rubric.MAX_IMPORTANCE)

    def test_scoring_is_deterministic(self) -> None:
        first = importance_rubric.score("The port is 8100.", "semantic")
        second = importance_rubric.score("The port is 8100.", "semantic")
        self.assertEqual(first.importance, second.importance)
        self.assertEqual(first.modifiers, second.modifiers)

    def test_the_explanation_names_the_base_and_modifiers(self) -> None:
        described = importance_rubric.score("The port is 8100.", "semantic").describe()
        self.assertIn("semantic base 0.66", described)
        self.assertIn("durable_detail", described)


class FlatDetectionTests(unittest.TestCase):
    def test_the_placeholder_is_recognised(self) -> None:
        self.assertTrue(importance_rubric.is_flat(0.5))
        self.assertTrue(importance_rubric.is_flat("0.5"))

    def test_a_chosen_number_is_not_flat(self) -> None:
        self.assertFalse(importance_rubric.is_flat(0.78))
        self.assertFalse(importance_rubric.is_flat(0.5001))

    def test_a_missing_value_counts_as_flat(self) -> None:
        self.assertTrue(importance_rubric.is_flat(None))
        self.assertTrue(importance_rubric.is_flat("nonsense"))


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _add(self, content: str, kind: str) -> str:
        return self.store.add_memory(
            content=content, kind=kind, source_type="user_turn",
            source_category="USER_EXPLICIT", importance=0.5,
        )[0]

    def test_fill_rescores_only_the_flat_ones(self) -> None:
        flat = self._add("The dashboard runs on port 8100.", "semantic")
        chosen = self.store.add_memory(
            content="Chosen importance.", kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT", importance=0.91,
        )[0]
        result = importance_rubric.fill(self.store)
        self.assertEqual(result["candidates"], 1)
        self.assertNotEqual(self.store.get_memory(flat)["importance"], 0.5)
        self.assertEqual(self.store.get_memory(chosen)["importance"], 0.91)

    def test_a_rescored_memory_that_still_lands_on_0_5_is_not_counted(self) -> None:
        self._add("Something happened.", "unknown")
        result = importance_rubric.fill(self.store)
        self.assertEqual(result["updated"], 0)

    def test_the_distribution_reports_the_flat_share(self) -> None:
        self._add("The dashboard runs on port 8100.", "semantic")
        self._add("Prefers short answers.", "preference")
        described = importance_rubric.distribution(self.store)
        self.assertEqual(described["active"], 2)
        self.assertEqual(described["flat"], 2)
        self.assertIn("preference", described["by_kind"])


if __name__ == "__main__":
    unittest.main()
