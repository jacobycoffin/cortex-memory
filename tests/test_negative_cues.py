"""Negative cues: do not return the thing the query ruled out (item #7)."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.negative_cues import MAX_TERMS, PENALTY, excluded_terms, suppress


@dataclass
class Fake:
    memory: dict[str, Any]
    score: float
    components: dict[str, float] = field(default_factory=dict)


class ExtractionTests(unittest.TestCase):
    def test_plain_negation(self) -> None:
        self.assertEqual(excluded_terms("not the maintenance window"), ["maintenance window"])

    def test_other_than(self) -> None:
        self.assertEqual(excluded_terms("what about other than last weeks plan"), ["last weeks plan"])

    def test_except_and_besides(self) -> None:
        self.assertEqual(excluded_terms("except the truck"), ["truck"])
        self.assertEqual(excluded_terms("besides the gym"), ["gym"])

    def test_instead_of_and_anything_but(self) -> None:
        self.assertEqual(excluded_terms("use instead of the old router"), ["old router"])
        self.assertEqual(excluded_terms("anything but the blue one"), ["blue"])

    def test_dont_include(self) -> None:
        self.assertEqual(excluded_terms("don't include the vr games note"), ["vr games note"])

    def test_no_exclusion_returns_empty(self) -> None:
        self.assertEqual(excluded_terms("what is the maintenance window"), [])
        self.assertEqual(excluded_terms(""), [])
        self.assertEqual(excluded_terms(None), [])

    def test_not_inside_a_word_is_ignored(self) -> None:
        self.assertEqual(excluded_terms("take a note about notation"), [])

    def test_trailing_filler_is_trimmed(self) -> None:
        self.assertEqual(excluded_terms("not the truck one"), ["truck"])

    def test_terms_are_capped(self) -> None:
        query = "not alpha, not beta, not gamma, not delta, not epsilon, not zeta"
        self.assertLessEqual(len(excluded_terms(query)), MAX_TERMS)


class SuppressionTests(unittest.TestCase):
    def test_excluded_memory_is_demoted_not_deleted(self) -> None:
        keep = Fake({"id": "a", "content": "The maintenance window is Sunday 02:00."}, 0.5)
        drop = Fake({"id": "b", "content": "The maintenance window moved to Monday."}, 0.9)
        out, count = suppress([keep, drop], ["monday"])
        self.assertEqual(count, 1)
        by_id = {r.memory["id"]: r for r in out}
        self.assertEqual(len(out), 2)  # nothing is destroyed
        self.assertEqual(by_id["a"].score, 0.5)
        self.assertAlmostEqual(by_id["b"].score, 0.9 * PENALTY, places=6)
        self.assertEqual(by_id["b"].components["negative_cue"], 1.0)

    def test_whole_word_matching(self) -> None:
        result = Fake({"id": "a", "content": "Trucking logs for the fleet."}, 0.5)
        out, count = suppress([result], ["truck"])
        self.assertEqual(count, 0)
        self.assertEqual(out[0].score, 0.5)

    def test_multi_word_term_needs_all_words(self) -> None:
        partial = Fake({"id": "a", "content": "The old router is in the closet."}, 0.5)
        out, count = suppress([partial], ["old switch"])
        self.assertEqual(count, 0)

    def test_no_terms_is_a_no_op(self) -> None:
        result = Fake({"id": "a", "content": "anything"}, 0.5)
        out, count = suppress([result], [])
        self.assertEqual((count, out[0].score), (0, 0.5))

    def test_input_objects_are_not_mutated(self) -> None:
        result = Fake({"id": "a", "content": "Monday plan"}, 0.7)
        suppress([result], ["monday"])
        self.assertEqual(result.score, 0.7)
        self.assertEqual(result.components, {})


if __name__ == "__main__":
    unittest.main()
