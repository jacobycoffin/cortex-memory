"""Density beats rank when the budget, not the count, is the real constraint (item #12)."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.density import apply, by_confidence, density, fill_budget, tokens_of


@dataclass
class Fake:
    memory: dict[str, Any]
    score: float
    estimated_tokens: int = 0
    components: dict[str, float] = field(default_factory=dict)


def item(name: str, score: float, tokens: int) -> Fake:
    return Fake(memory={"id": name, "content": "x" * (tokens * 4)}, score=score,
                estimated_tokens=tokens)


class TokenTests(unittest.TestCase):
    def test_declared_tokens_are_used(self) -> None:
        self.assertEqual(tokens_of(item("a", 0.5, 40)), 40)

    def test_missing_tokens_fall_back_to_content_length(self) -> None:
        result = Fake(memory={"id": "a", "content": "x" * 400}, score=0.5, estimated_tokens=0)
        self.assertEqual(tokens_of(result), 100)

    def test_tokens_are_never_zero(self) -> None:
        result = Fake(memory={"id": "a", "content": ""}, score=0.5, estimated_tokens=0)
        self.assertEqual(tokens_of(result), 1)

    def test_density_is_score_per_token(self) -> None:
        self.assertAlmostEqual(density(item("a", 0.60, 30)), 0.02, places=6)


class FillTests(unittest.TestCase):
    def test_dense_items_win_over_long_ones(self) -> None:
        short_sharp = item("short", 0.55, 30)
        long_dull = item("long", 0.60, 900)
        chosen = [c.result.memory["id"] for c in fill_budget([long_dull, short_sharp], 120)]
        self.assertIn("short", chosen)
        self.assertNotIn("long", chosen)  # does not fit, and is far less dense

    def test_five_long_notes_do_not_eat_the_whole_budget(self) -> None:
        long_notes = [item(f"long{i}", 0.50 + i / 100, 900) for i in range(5)]
        shorts = [item(f"short{i}", 0.45, 40) for i in range(12)]
        chosen = fill_budget([*long_notes, *shorts], 4000)
        self.assertGreater(len(chosen), 5)  # more evidence for the same tokens

    def test_the_budget_is_never_exceeded(self) -> None:
        pool = [item(f"i{i}", 0.5, 100) for i in range(20)]
        chosen = fill_budget(pool, 350)
        self.assertLessEqual(sum(c.tokens for c in chosen), 350)

    def test_low_density_noise_is_skipped_even_when_it_fits(self) -> None:
        good = item("good", 0.80, 40)
        noise = item("noise", 0.02, 4)
        chosen = [c.result.memory["id"] for c in fill_budget([good, noise], 4000)]
        self.assertEqual(chosen, ["good"])

    def test_a_small_item_is_chosen_when_a_big_one_does_not_fit(self) -> None:
        big = item("big", 0.90, 500)
        small = item("small", 0.60, 50)
        chosen = [c.result.memory["id"] for c in fill_budget([big, small], 100)]
        self.assertEqual(chosen, ["small"])

    def test_max_items_caps_the_count(self) -> None:
        pool = [item(f"i{i}", 0.7, 10) for i in range(10)]
        self.assertEqual(len(fill_budget(pool, 4000, max_items=3)), 3)

    def test_ordering_is_deterministic(self) -> None:
        pool = [item("b", 0.5, 20), item("a", 0.5, 20)]
        first = [c.result.memory["id"] for c in fill_budget(pool, 400)]
        second = [c.result.memory["id"] for c in fill_budget(list(reversed(pool)), 400)]
        self.assertEqual(first, second)

    def test_empty_pool_and_zero_budget(self) -> None:
        self.assertEqual(fill_budget([], 4000), [])
        self.assertEqual(fill_budget([item("a", 0.9, 10)], 0), [])

    def test_apply_returns_results_in_density_order(self) -> None:
        pool = [item("low", 0.4, 100), item("high", 0.8, 20)]
        # With the default noise floor only the dense item qualifies (0.004 is 10x below
        # 0.04); with the floor removed both are kept, ordered by density.
        self.assertEqual([r.memory["id"] for r in apply(pool, 1000)], ["high"])
        self.assertEqual(
            [r.memory["id"] for r in apply(pool, 1000, min_ratio=0.0)], ["high", "low"]
        )


class ConfidenceTests(unittest.TestCase):
    def test_the_weak_tail_is_not_bought(self) -> None:
        strong = [item(f"s{i}", 0.60 - i / 100, 100) for i in range(3)]
        weak = [item(f"w{i}", 0.20, 100) for i in range(10)]
        kept = by_confidence([*strong, *weak], 4000)
        self.assertTrue(all(r.score >= 0.55 for r in kept))
        self.assertLess(len(kept), 13)

    def test_score_order_is_preserved(self) -> None:
        # All three are within 85% of the best, so all are bought - in score order.
        pool = [item("b", 0.86, 50), item("a", 0.90, 50), item("c", 0.88, 50)]
        kept = by_confidence(pool, 4000)
        self.assertEqual([r.memory["id"] for r in kept], ["a", "c", "b"])

    def test_the_budget_still_caps_the_spend(self) -> None:
        pool = [item(f"i{i}", 0.9, 100) for i in range(20)]
        kept = by_confidence(pool, 350)
        self.assertLessEqual(sum(tokens_of(r) for r in kept), 350)

    def test_max_items_is_respected(self) -> None:
        pool = [item(f"i{i}", 0.9, 10) for i in range(10)]
        self.assertEqual(len(by_confidence(pool, 4000, max_items=4)), 4)

    def test_empty_and_zero_budget(self) -> None:
        self.assertEqual(by_confidence([], 4000), [])
        self.assertEqual(by_confidence([item("a", 0.9, 10)], 0), [])

    def test_ordering_is_deterministic(self) -> None:
        pool = [item("b", 0.7, 20), item("a", 0.7, 20)]
        self.assertEqual(
            [r.memory["id"] for r in by_confidence(pool, 400)],
            [r.memory["id"] for r in by_confidence(list(reversed(pool)), 400)],
        )


if __name__ == "__main__":
    unittest.main()
