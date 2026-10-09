"""Regression tests for incremental diversified selection (2026-09-14).

The selection loop used to re-compute ``_memory_similarity`` for every
remaining candidate against every selected memory on every pick, so comparison
work grew with the square of the candidate count (the probe corpus below cost
~1,900 set intersections for 36 candidates / 3 selections). Selection now keeps
a running maximum-similarity value per remaining candidate and refreshes it only
when a memory is actually picked.

Two properties are pinned here and both must hold for any future change:

1. the selection output is unchanged — the same picks with the same
   per-candidate reasons as the re-computing implementation produced;
2. similarity comparisons are bounded by ``candidates x selected`` work instead
   of ``candidates x selected x loop-iterations``.
"""

from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import retrieval
from cortex.retrieval import MemoryRetriever
from cortex.store import CortexStore

QUERY = "amber pipeline deployment checklist verified backup release gate"

# Selection output for this exact corpus, captured from the pre-change (fully
# re-computing) implementation and verified byte-identical after the change.
EXPECTED_SELECTED = [
    "Amber pipeline deployment checklist verified backup release gate step nine zero one.",
    "Amber pipeline deployment checklist verified backup release gate: "
    + " ".join(f"longdetail{i}" for i in range(60)),
    "Amber pipeline deployment checklist verified backup release gate filler 9.",
]
EXPECTED_REASONS = {
    "selected by score, context budget, and diversity constraints": 3,
    "near-duplicate of a stronger selected memory": 30,
    "insufficient direct relevance for focused recall": 3,
}


class SelectIncrementalSimilarityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        # near-duplicate pair (jaccard ~0.92): one is picked, the other must be
        # rejected as a near-duplicate of it.
        self.store.add_memory(
            "Amber pipeline deployment checklist verified backup release gate step nine zero zero.",
            kind="semantic",
            confidence=0.9,
            importance=0.95,
        )
        self.store.add_memory(
            "Amber pipeline deployment checklist verified backup release gate step nine zero one.",
            kind="semantic",
            confidence=0.9,
            importance=0.95,
        )
        # claim family trio (same subject/predicate) — diversity-cap territory.
        self.store.add_memory(
            "Two distinct approvers must sign off on every Amber release.",
            kind="semantic",
            subject="amber-release",
            predicate="approval-count",
            confidence=0.85,
            importance=0.85,
        )
        self.store.add_memory(
            "Amber releases need sign-off from two separate approvers.",
            kind="semantic",
            subject="amber-release",
            predicate="approval-count",
            confidence=0.85,
            importance=0.85,
        )
        self.store.add_memory(
            "For Amber, release approval requires two different reviewers.",
            kind="semantic",
            subject="amber-release",
            predicate="approval-count",
            confidence=0.85,
            importance=0.85,
        )
        # long body that can straddle the token budget.
        self.store.add_memory(
            "Amber pipeline deployment checklist verified backup release gate: "
            + " ".join(f"longdetail{i}" for i in range(60)),
            kind="semantic",
            confidence=0.9,
            importance=0.9,
        )
        # filler candidates with heavy lexical overlap.
        for i in range(30):
            self.store.add_memory(
                f"Amber pipeline deployment checklist verified backup release gate filler {i}.",
                kind="operational",
                confidence=0.7,
                importance=0.6,
            )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _search(self, **kwargs: object):
        calls = {"count": 0}
        original = retrieval._memory_similarity

        def counting(left, right):
            calls["count"] += 1
            return original(left, right)

        retrieval._memory_similarity = counting  # type: ignore[assignment]
        try:
            selected, diagnostics = MemoryRetriever(self.store).search_detailed(
                QUERY, graph_depth=0, **kwargs
            )
        finally:
            retrieval._memory_similarity = original  # type: ignore[assignment]
        return selected, diagnostics, calls["count"]

    def test_selection_output_is_unchanged(self) -> None:
        selected, diagnostics, _calls = self._search(limit=6, token_budget=700, threshold=0.45)
        self.assertEqual([result.memory["content"] for result in selected], EXPECTED_SELECTED)
        reasons = Counter(decision["reason"] for decision in diagnostics.candidate_decisions)
        self.assertEqual(dict(reasons), EXPECTED_REASONS)

    def test_budget_rejections_survive_incremental_tracking(self) -> None:
        selected, diagnostics, _calls = self._search(limit=6, token_budget=46, threshold=0.45)
        self.assertEqual(len(selected), 1)
        reasons = Counter(decision["reason"] for decision in diagnostics.candidate_decisions)
        self.assertEqual(
            reasons["would exceed the task context budget"],
            diagnostics.candidate_count - 5,
            f"expected every non-special candidate to hit the budget gate, got {dict(reasons)}",
        )
        self.assertEqual(reasons["near-duplicate of a stronger selected memory"], 1)

    def test_similarity_comparisons_stay_bounded(self) -> None:
        selected, diagnostics, call_count = self._search(limit=6, token_budget=700, threshold=0.45)
        # Each pick compares every remaining candidate once against the picked
        # memory, so total comparisons cannot exceed candidates x picks (+1 slack
        # per pick for the near-duplicate re-check path).
        bound = diagnostics.candidate_count * (len(selected) + 1)
        self.assertLessEqual(
            call_count,
            bound,
            "selection is re-computing candidate-vs-selected pairs per loop "
            f"iteration again: {call_count} comparisons for "
            f"{diagnostics.candidate_count} candidates / {len(selected)} picks "
            f"(bound {bound})",
        )


class SelectPoolWalkTests(unittest.TestCase):
    """The pool walk must stay single-pass when nothing is being selected.

    Picking the diversified maximum is a full rescan of `remaining`. While no
    memory has been selected yet every candidate's diversified value is exactly
    its score and the pool is already in descending score order, so the scan used
    to re-derive the same head n times — 51,360 comparisons for a 320-candidate
    pool whose results all failed the gates. The head is taken directly now.

    Both properties below are behavioural: the walk must still visit every
    candidate exactly once with a recorded reason, and the head must still be the
    highest-scoring candidate.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.store.add_memory(
            "Amber pipeline deployment checklist verified backup release gate step nine zero one.",
            kind="semantic",
            confidence=0.9,
            importance=0.95,
        )
        for i in range(30):
            self.store.add_memory(
                f"Amber pipeline deployment checklist verified backup release gate filler {i}.",
                kind="operational",
                confidence=0.7,
                importance=0.6,
            )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_rejected_pool_is_walked_once_with_a_reason_for_every_candidate(self) -> None:
        selected, diagnostics = MemoryRetriever(self.store, threshold=0.16).search_detailed(
            QUERY, limit=64, token_budget=4000, threshold=0.99, graph_depth=0
        )
        self.assertEqual(selected, [])
        self.assertTrue(diagnostics.abstained)
        decisions = diagnostics.candidate_decisions
        self.assertEqual(len(decisions), diagnostics.candidate_count)
        self.assertEqual(len({row["memory_id"] for row in decisions}), len(decisions))
        self.assertEqual(
            Counter(row["reason"] for row in decisions),
            Counter({"score below the active retrieval threshold": diagnostics.candidate_count}),
        )

    def test_first_pick_is_the_highest_scoring_candidate(self) -> None:
        with_limit_one, diagnostics = MemoryRetriever(self.store, threshold=0.0).search_detailed(
            QUERY, limit=1, token_budget=4000, threshold=0.0, graph_depth=0
        )
        self.assertEqual(len(with_limit_one), 1)
        # Decision rows round their score to 6 places, so compare like for like.
        self.assertAlmostEqual(
            float(with_limit_one[0].score),
            max(float(row["score"]) for row in diagnostics.candidate_decisions),
            places=6,
            msg="the first pick must be the head of the score-ordered pool",
        )


if __name__ == "__main__":
    unittest.main()
