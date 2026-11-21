"""Abstention must be safe when enabled, and off until it is (#14)."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.abstention import MIN_USEFUL_THRESHOLD, should_abstain


@dataclass
class Fake:
    memory: dict[str, Any]
    score: float
    components: dict[str, float] = field(default_factory=dict)


def result(score: float, **memory: Any) -> Fake:
    return Fake({"id": "m", "content": "something", **memory}, score)


class DefaultTests(unittest.TestCase):
    def test_policy_is_disabled_by_default(self) -> None:
        abstain, reason = should_abstain([result(0.01)])
        self.assertFalse(abstain)
        self.assertIn("disabled", reason)

    def test_a_nonsense_threshold_is_refused(self) -> None:
        abstain, reason = should_abstain([result(0.01)], threshold=0.001)
        self.assertFalse(abstain)
        self.assertIn("too low", reason)

    def test_nothing_retrieved_is_not_an_abstention(self) -> None:
        abstain, reason = should_abstain([], threshold=0.6)
        self.assertFalse(abstain)
        self.assertIn("nothing to abstain", reason)


class GuardTests(unittest.TestCase):
    def test_low_score_abstains_when_enabled(self) -> None:
        abstain, _ = should_abstain([result(0.30)], threshold=0.55)
        self.assertTrue(abstain)

    def test_high_score_does_not_abstain(self) -> None:
        abstain, _ = should_abstain([result(0.70)], threshold=0.55)
        self.assertFalse(abstain)

    def test_pinned_result_is_never_silenced(self) -> None:
        abstain, reason = should_abstain([result(0.10, pinned=1)], threshold=0.55)
        self.assertFalse(abstain)
        self.assertIn("pinned", reason)

    def test_a_previously_used_result_is_never_silenced(self) -> None:
        abstain, reason = should_abstain(
            [Fake({"id": "m", "content": "x"}, 0.10, {"use_boost": 0.03})], threshold=0.55
        )
        self.assertFalse(abstain)
        self.assertIn("used before", reason)

    def test_a_corroborated_result_is_never_silenced(self) -> None:
        abstain, reason = should_abstain(
            [Fake({"id": "m", "content": "x"}, 0.10, {"stability": 0.4})], threshold=0.55
        )
        self.assertFalse(abstain)
        self.assertIn("corroboration", reason)

    def test_negative_cue_demotion_is_not_treated_as_weakness(self) -> None:
        """A memory demoted by an explicit exclusion is not evidence of a bad query."""
        abstain, reason = should_abstain(
            [Fake({"id": "m", "content": "x"}, 0.10, {"negative_cue": 1.0})], threshold=0.55
        )
        self.assertFalse(abstain)
        self.assertIn("negative cue", reason)

    def test_threshold_must_exceed_the_useless_floor(self) -> None:
        self.assertGreater(MIN_USEFUL_THRESHOLD, 0.0)


if __name__ == "__main__":
    unittest.main()
