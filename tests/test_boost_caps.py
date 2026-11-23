"""Popularity may break a tie; it may not become truth (#203)."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import boost_caps


@dataclass
class Result:
    score: float = 1.0
    components: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=lambda: {"id": "m1"})


class ClampTests(unittest.TestCase):
    def test_each_family_has_its_own_ceiling(self) -> None:
        self.assertEqual(boost_caps.clamp("usage", 5.0), boost_caps.CAPS["usage"])
        self.assertEqual(boost_caps.clamp("entity", 5.0), boost_caps.CAPS["entity"])

    def test_a_small_boost_is_untouched(self) -> None:
        self.assertEqual(boost_caps.clamp("usage", 0.04), 0.04)

    def test_a_negative_boost_is_zero(self) -> None:
        self.assertEqual(boost_caps.clamp("usage", -1.0), 0.0)

    def test_an_unknown_family_is_only_floored(self) -> None:
        self.assertEqual(boost_caps.clamp("mystery", 9.0), 9.0)

    def test_the_total_cap_is_above_one_family_and_below_the_sum(self) -> None:
        self.assertGreater(boost_caps.TOTAL_CAP, max(boost_caps.CAPS.values()))
        self.assertLess(boost_caps.TOTAL_CAP, sum(boost_caps.CAPS.values()))


class TotalTests(unittest.TestCase):
    def test_no_components_is_no_boost(self) -> None:
        self.assertEqual(boost_caps.total_of({}), 0.0)
        self.assertEqual(boost_caps.total_of(None), 0.0)

    def test_the_usage_boost_is_counted(self) -> None:
        self.assertAlmostEqual(boost_caps.total_of({"use_boost": 0.05}), 0.05)

    def test_families_add_up(self) -> None:
        components = {"use_boost": 0.09, "entity_boost": 0.15, "session_boost": 0.05}
        self.assertAlmostEqual(boost_caps.total_of(components), 0.29)

    def test_a_family_is_clamped_before_it_is_summed(self) -> None:
        self.assertAlmostEqual(boost_caps.total_of({"use_boost": 5.0}), boost_caps.CAPS["usage"])

    def test_a_malformed_component_is_ignored(self) -> None:
        self.assertAlmostEqual(boost_caps.total_of({"use_boost": "nonsense"}), 0.0)

    def test_unrelated_components_are_ignored(self) -> None:
        """`entity_anchored` is a marker, not a magnitude, and must not be double counted."""
        self.assertEqual(boost_caps.total_of({"entity_anchored": 1.0}), 0.0)


class EnforceTests(unittest.TestCase):
    def test_a_normal_result_is_untouched(self) -> None:
        results, capped = boost_caps.enforce([Result(score=1.0, components={"use_boost": 0.06})])
        self.assertEqual(capped, 0)
        self.assertEqual(results[0].score, 1.0)

    def test_an_over_cap_result_loses_the_excess(self) -> None:
        components = {"use_boost": 0.09, "entity_boost": 0.15, "session_boost": 0.10,
                      "association_boost": 0.10}
        results, capped = boost_caps.enforce([Result(score=1.0, components=components)])
        self.assertEqual(capped, 1)
        self.assertAlmostEqual(results[0].score, 1.0 - (0.44 - boost_caps.TOTAL_CAP), places=6)

    def test_the_capped_result_says_so(self) -> None:
        components = {"use_boost": 0.09, "entity_boost": 0.15, "session_boost": 0.10}
        results, _ = boost_caps.enforce([Result(score=1.0, components=components)])
        self.assertIn("boost_capped", results[0].components)
        self.assertGreater(results[0].components["boost_total"], boost_caps.TOTAL_CAP)

    def test_the_components_are_preserved_for_explanation(self) -> None:
        components = {"use_boost": 0.09, "entity_boost": 0.15, "session_boost": 0.10}
        results, _ = boost_caps.enforce([Result(score=1.0, components=components)])
        self.assertEqual(results[0].components["use_boost"], 0.09)
        self.assertEqual(results[0].components["entity_boost"], 0.15)

    def test_the_score_is_never_raised(self) -> None:
        results, _ = boost_caps.enforce([Result(score=0.5, components={"use_boost": 0.02})])
        self.assertLessEqual(results[0].score, 0.5)

    def test_a_result_without_components_passes_through(self) -> None:
        results, capped = boost_caps.enforce([Result(score=1.0)])
        self.assertEqual(capped, 0)
        self.assertEqual(results[0].score, 1.0)

    def test_enforce_is_idempotent(self) -> None:
        components = {"use_boost": 0.09, "entity_boost": 0.15, "session_boost": 0.10}
        once, _ = boost_caps.enforce([Result(score=1.0, components=components)])
        twice, capped = boost_caps.enforce(once)
        self.assertEqual(capped, 1)
        self.assertLess(twice[0].score, 1.0)

    def test_the_report_describes_the_caps(self) -> None:
        described = boost_caps.report()
        self.assertEqual(described["total_cap"], boost_caps.TOTAL_CAP)
        self.assertGreater(described["sum_of_families"], described["total_cap"])

    def test_the_report_can_summarise_observed_results(self) -> None:
        results = [
            Result(components={"use_boost": 0.09}),
            Result(components={"use_boost": 0.09, "entity_boost": 0.15, "session_boost": 0.10}),
        ]
        described = boost_caps.report(results)
        self.assertEqual(described["observed"], 2)
        self.assertEqual(described["over_cap"], 1)
        self.assertAlmostEqual(described["largest"], 0.34, places=4)

    def test_an_empty_result_set_is_quiet(self) -> None:
        results, capped = boost_caps.enforce([])
        self.assertEqual((results, capped), ([], 0))


if __name__ == "__main__":
    unittest.main()
