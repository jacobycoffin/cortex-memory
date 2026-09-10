"""Tests for the pure rank-fusion module (``cortex.fusion``)."""

from __future__ import annotations

import math
import unittest
import warnings
from fractions import Fraction

from tests._bootstrap import ROOT  # noqa: F401

from cortex.fusion import _fused_scores, cosine_ranking, fuse_semantic, reciprocal_rank_fusion


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_hand_computed_rrf_scores_and_order(self) -> None:
        rankings = [["a", "b", "c", "d"], ["b", "c", "d", "a"]]
        k = 60

        # Exact scores, computed by hand as fractions of the RRF formula:
        # score = 1/(k + rank + 1) summed over every ranking the id appears in.
        expected = {
            "a": Fraction(1, 61) + Fraction(1, 64),  # 0.032018...
            "b": Fraction(1, 62) + Fraction(1, 61),  # 0.032522...
            "c": Fraction(1, 63) + Fraction(1, 62),  # 0.032002...
            "d": Fraction(1, 64) + Fraction(1, 63),  # 0.031498...
        }

        observed = _fused_scores(rankings, k, None)
        self.assertEqual(set(observed), set(expected))
        for doc_id, exact in expected.items():
            self.assertAlmostEqual(observed[doc_id], float(exact), places=15)

        # The four exact scores are all distinct, so the expected order follows
        # directly from the hand-computed numbers: b > a > c > d.
        ordered = sorted(expected, key=lambda doc_id: -float(expected[doc_id]))
        self.assertEqual(ordered, ["b", "a", "c", "d"])
        self.assertEqual(reciprocal_rank_fusion(rankings, k=k), ["b", "a", "c", "d"])
        self.assertEqual(reciprocal_rank_fusion(rankings, k=1), ["b", "a", "c", "d"])

    def test_denominator_is_k_plus_rank_plus_one(self) -> None:
        # Guards the "+1": with k=0, rank 0 must score exactly 1/1, rank 1 1/2,
        # rank 2 1/3 and rank 3 1/4. Dropping the "+1" would make these 1/0, 1/1,
        # 1/2, 1/3 instead.
        observed = _fused_scores([["a", "b", "c", "d"]], 0, None)
        self.assertAlmostEqual(observed["a"], 1.0, places=15)
        self.assertAlmostEqual(observed["b"], 0.5, places=15)
        self.assertAlmostEqual(observed["c"], float(Fraction(1, 3)), places=15)
        self.assertAlmostEqual(observed["d"], 0.25, places=15)
        # And the default k=60 is honoured, not silently ignored.
        default_k = _fused_scores([["a"]], 60, None)
        self.assertAlmostEqual(default_k["a"], float(Fraction(1, 61)), places=15)

    def test_high_in_both_beats_top_of_one(self) -> None:
        cortex = ["top_c", "m", "p"]
        semantic = ["top_s", "m", "q"]
        self.assertEqual(
            reciprocal_rank_fusion([cortex, semantic], k=60),
            ["m", "top_c", "top_s", "p", "q"],
        )

        # Explicitly: the doc ranked second by both outranks the doc ranked
        # first by only one of the two lists.
        scores = _fused_scores([cortex, semantic], 60, None)
        self.assertGreater(scores["m"], scores["top_c"])
        self.assertGreater(scores["m"], scores["top_s"])

    def test_duplicate_ids_in_one_ranking_count_once(self) -> None:
        # "a" appears at index 0 and index 1; only its first occurrence is
        # scored, so it earns 1/61 and NOT 2/61. "b" keeps its own 0-based index
        # in the ranking (index 2), i.e. the duplicate does not re-pack ranks.
        observed = _fused_scores([["a", "a", "b"]], 60, None)
        self.assertAlmostEqual(observed["a"], float(Fraction(1, 61)), places=15)
        self.assertNotAlmostEqual(observed["a"], float(Fraction(2, 61)), places=15)
        self.assertAlmostEqual(observed["b"], float(Fraction(1, 63)), places=15)
        self.assertEqual(reciprocal_rank_fusion([["a", "a", "b"]], k=60), ["a", "b"])
        # Same duplicate pair repeated: still one score per id, order unchanged.
        self.assertEqual(reciprocal_rank_fusion([["a", "a", "a"]], k=60), ["a"])
        self.assertAlmostEqual(_fused_scores([["a", "a", "a"]], 60, None)["a"], float(Fraction(1, 61)), places=15)
        self.assertEqual(reciprocal_rank_fusion([["a", "a", "b", "b", "c"]], k=60), ["a", "b", "c"])

    def test_id_in_both_rankings_counts_twice(self) -> None:
        observed = _fused_scores([["a"], ["a"]], 60, None)
        self.assertAlmostEqual(observed["a"], float(Fraction(2, 61)), places=15)
        self.assertEqual(reciprocal_rank_fusion([["a"], ["a"]], k=60), ["a"])

    def test_ids_present_in_only_some_rankings_are_kept(self) -> None:
        # a (rank 0 of list 1) and c (rank 0 of list 2) tie at 1/61; the tie is
        # broken by first appearance, so a comes before c. b scores 1/62.
        self.assertEqual(reciprocal_rank_fusion([["a", "b"], ["c"]], k=60), ["a", "c", "b"])
        scores = _fused_scores([["a", "b"], ["c"]], 60, None)
        self.assertEqual(sorted(scores), ["a", "b", "c"])

    def test_empty_inputs_return_empty_list(self) -> None:
        self.assertEqual(reciprocal_rank_fusion([]), [])
        self.assertEqual(reciprocal_rank_fusion([[], []]), [])
        self.assertEqual(reciprocal_rank_fusion([[], []], weights=[1.0, 0.5]), [])
        self.assertEqual(fuse_semantic([], []), [])
        self.assertEqual(cosine_ranking([1.0, 0.0], {}), [])
        self.assertEqual(cosine_ranking([], {"a": [1.0]}), [])

    def test_weights_validation(self) -> None:
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0, -0.5])
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([["a"]], weights=[-0.1])
        with self.assertRaises(ValueError):
            fuse_semantic(["a"], ["b"], semantic_weight=-1.0)
        # Valid calls must not raise.
        self.assertEqual(reciprocal_rank_fusion([["a"], ["b"]], weights=(1.0, 0.5)), ["a", "b"])
        self.assertEqual(reciprocal_rank_fusion([], weights=[]), [])

    def test_weights_change_relative_influence(self) -> None:
        rankings = [["a", "b"], ["b", "a"]]
        self.assertEqual(reciprocal_rank_fusion(rankings, weights=[1.0, 0.1]), ["a", "b"])
        self.assertEqual(reciprocal_rank_fusion(rankings, weights=[0.1, 1.0]), ["b", "a"])
        scores = _fused_scores(rankings, 60, [1.0, 0.1])
        self.assertAlmostEqual(scores["a"], 1.0 / 61 + 0.1 / 62, places=15)
        self.assertAlmostEqual(scores["b"], 1.0 / 62 + 0.1 / 61, places=15)

    def test_tie_break_is_first_appearance(self) -> None:
        # "b" and "a" swap positions between the two lists, so their fused scores
        # are exactly equal; first appearance order (b first) decides.
        scores = _fused_scores([["b", "a"], ["a", "b"]], 60, None)
        self.assertAlmostEqual(scores["b"], scores["a"], places=15)
        self.assertEqual(reciprocal_rank_fusion([["b", "a"], ["a", "b"]], k=60), ["b", "a"])

        # Same shape of tie, opposite first appearance.
        self.assertEqual(reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60), ["a", "b"])

    def test_fuse_semantic_is_rrf_over_two_rankings(self) -> None:
        cortex = ["top_c", "m", "p"]
        semantic = ["top_s", "m", "q"]
        self.assertEqual(
            fuse_semantic(cortex, semantic, k=60, semantic_weight=1.0),
            reciprocal_rank_fusion([cortex, semantic], k=60, weights=[1.0, 1.0]),
        )
        self.assertEqual(fuse_semantic(cortex, semantic, k=60), ["m", "top_c", "top_s", "p", "q"])
        # A heavier semantic weight promotes the semantic list's own top pick.
        self.assertEqual(fuse_semantic(["a", "b"], ["b", "a"], semantic_weight=0.01), ["a", "b"])
        self.assertEqual(fuse_semantic(["a", "b"], ["b", "a"], semantic_weight=100.0), ["b", "a"])


class CosineRankingTests(unittest.TestCase):
    def test_orders_by_true_cosine_similarity(self) -> None:
        query = [1.0, 0.0]
        candidates = {
            "identical": [2.0, 0.0],  # cos 1.0
            "oblique": [1.0, 1.0],  # cos sqrt(2)/2 = 0.7071...
            "orthogonal": [0.0, 5.0],  # cos 0.0
            "opposite": [-1.0, 0.0],  # cos -1.0
        }
        self.assertEqual(
            cosine_ranking(query, candidates),
            ["identical", "oblique", "orthogonal", "opposite"],
        )
        # The expected middle value really is the cosine of a 45-degree vector.
        self.assertAlmostEqual(1.0 / math.sqrt(2.0), math.sqrt(2) / 2, places=15)

    def test_cosine_ignores_vector_magnitude(self) -> None:
        # A tiny vector that points the same way must beat a huge orthogonal one:
        # a dot-product or L2 ranking would get this backwards.
        query = [1.0, 0.0]
        candidates = {"tiny_aligned": [0.001, 0.0], "huge_orthogonal": [0.0, 1000.0]}
        self.assertEqual(cosine_ranking(query, candidates), ["tiny_aligned", "huge_orthogonal"])

    def test_negative_similarity_sorts_last(self) -> None:
        query = [1.0, 0.0]
        candidates = {"opposite": [-3.0, 0.0], "faintly_positive": [0.01, 1.0]}
        self.assertEqual(cosine_ranking(query, candidates), ["faintly_positive", "opposite"])

    def test_wrong_length_vector_is_skipped(self) -> None:
        query = [1.0, 0.0]
        candidates = {
            "good_1": [2.0, 0.0],
            "wrong_len": [1.0, 0.0, 0.0],
            "zero_norm": [0.0, 0.0],
            "empty": [],
            "good_2": [5.0, 0.0],
        }
        # No exception, wrong/zero vectors absent, ties resolved by mapping order.
        self.assertEqual(cosine_ranking(query, candidates), ["good_1", "good_2"])

    def test_zero_norm_query_skips_everything(self) -> None:
        self.assertEqual(cosine_ranking([0.0, 0.0], {"a": [1.0, 0.0]}), [])
        self.assertEqual(cosine_ranking([1.0, 0.0], {"a": [0.0, 0.0], "b": [1.0, 1.0]}), ["b"])

    def test_degenerate_vectors_are_skipped_without_runtime_warnings(self) -> None:
        # A zero-norm candidate must be skipped *before* the division, so no
        # 0/0 float error is ever raised by numpy. Turning warnings into errors
        # makes a silent divide-by-zero fail the test.
        query = [1.0, 0.0]
        candidates = {"zero_norm": [0.0, 0.0], "ok": [1.0, 0.0]}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(cosine_ranking(query, candidates), ["ok"])
            self.assertEqual(cosine_ranking([0.0, 0.0], {"zero": [0.0, 0.0]}), [])
            # A zero-norm *query* must short-circuit before any division either.
            self.assertEqual(cosine_ranking([0.0, 0.0], {"a": [1.0, 0.0]}), [])

    def test_non_finite_or_malformed_vectors_are_skipped(self) -> None:
        query = [1.0, 0.0]
        candidates = {
            "nan_vec": [float("nan"), 0.0],
            "inf_vec": [float("inf"), 0.0],
            "not_numbers": ["a", "b"],
            "ok": [1.0, 0.0],
        }
        self.assertEqual(cosine_ranking(query, candidates), ["ok"])

    def test_tie_break_by_mapping_order(self) -> None:
        query = [1.0, 0.0]
        candidates = {"zzz": [1.0, 0.0], "aaa": [4.0, 0.0]}
        # Both have cos 1.0; mapping order wins over lexicographic order.
        self.assertEqual(cosine_ranking(query, candidates), ["zzz", "aaa"])


class DeterminismTests(unittest.TestCase):
    def test_repeated_calls_are_identical(self) -> None:
        rankings = [["m3", "m1", "m2", "m1"], ["m2", "m3", "m4"], ["m4", "m1"]]
        weights = [1.0, 0.5, 2.0]

        first = reciprocal_rank_fusion(rankings, k=17, weights=weights)
        self.assertEqual(sorted(first), ["m1", "m2", "m3", "m4"])
        self.assertTrue(all(isinstance(doc_id, str) for doc_id in first))
        for _ in range(25):
            again = reciprocal_rank_fusion(rankings, k=17, weights=weights)
            self.assertEqual(again, first)
            self.assertEqual(repr(again), repr(first))

        query = [1.0, 0.0, 0.5]
        candidates = {
            "x": [1.0, 0.0, 0.5],
            "y": [-1.0, 0.0, 0.0],
            "z": [0.25, 0.25, 0.25],
            "w": [0.0, 1.0, 0.0],
        }
        first_cos = cosine_ranking(query, candidates)
        self.assertEqual(len(first_cos), 4)
        for _ in range(25):
            self.assertEqual(cosine_ranking(query, candidates), first_cos)

        first_fuse = fuse_semantic(rankings[0], rankings[1], k=17, semantic_weight=0.5)
        for _ in range(25):
            self.assertEqual(
                fuse_semantic(rankings[0], rankings[1], k=17, semantic_weight=0.5),
                first_fuse,
            )


if __name__ == "__main__":
    unittest.main()
