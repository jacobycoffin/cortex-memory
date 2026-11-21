"""The second stage must help, and must never invent relevance (item #6)."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.rerank import bm25, rerank, terms


@dataclass
class Fake:
    memory: dict[str, Any]
    score: float
    components: dict[str, float] = field(default_factory=dict)


def doc(content: str, score: float) -> Fake:
    return Fake({"id": content[:8], "content": content}, score)


class Bm25Tests(unittest.TestCase):
    def test_an_exact_term_match_outscores_a_partial_one(self) -> None:
        documents = [terms("the rtsp path of the front camera"), terms("camera settings overview")]
        scores = bm25("rtsp path front camera", documents)
        self.assertGreater(scores[0], scores[1])

    def test_no_query_terms_scores_zero(self) -> None:
        self.assertEqual(bm25("", [terms("anything")]), [0.0])

    def test_no_documents_scores_nothing(self) -> None:
        self.assertEqual(bm25("anything", []), [])

    def test_a_rare_term_counts_for_more_than_a_common_one(self) -> None:
        documents = [
            terms("cortex backup schedule"),        # rare: backup
            terms("cortex notes about cortex"),     # common: cortex
            terms("cortex weekly review"),
        ]
        scores = bm25("backup cortex", documents)
        self.assertGreater(scores[0], scores[1])

    def test_scores_are_deterministic(self) -> None:
        documents = [terms("alpha beta gamma"), terms("beta gamma delta")]
        self.assertEqual(bm25("beta", documents), bm25("beta", documents))


class RerankTests(unittest.TestCase):
    def test_a_lower_scored_exact_match_is_lifted(self) -> None:
        weak_but_exact = doc("The rtsp path for the front camera is /stream1.", 0.30)
        strong_but_vague = doc("Camera notes about settings and preferences in general.", 0.60)
        out, changed = rerank([strong_but_vague, weak_but_exact], "rtsp path front camera")
        self.assertGreater(changed, 0)
        by_id = {r.memory["content"][:8]: r.score for r in out}
        self.assertGreater(by_id["The rtsp"], 0.30)

    def test_an_irrelevant_memory_cannot_be_promoted_above_everything(self) -> None:
        """The rerank contributes at most half the score."""
        relevant = doc("The rtsp path for the front camera is /stream1.", 0.90)
        irrelevant = doc("rtsp path front camera", 0.00)
        out, _ = rerank([relevant, irrelevant], "rtsp path front camera")
        scores = {r.memory["content"][:8]: r.score for r in out}
        self.assertGreater(scores["The rtsp"], scores["rtsp pat"])

    def test_results_beyond_top_n_are_untouched(self) -> None:
        head = [doc("alpha beta", 0.5)]
        tail = [doc("gamma delta", 0.1)]
        out, _ = rerank(head + tail, "alpha", top_n=1)
        self.assertEqual(out[-1].score, 0.1)
        self.assertNotIn("bm25", out[-1].components)

    def test_an_empty_query_is_a_no_op(self) -> None:
        result = doc("anything at all", 0.4)
        out, changed = rerank([result], "")
        self.assertEqual((changed, out[0].score), (0, 0.4))

    def test_empty_results_are_safe(self) -> None:
        self.assertEqual(rerank([], "anything"), ([], 0))

    def test_inputs_are_not_mutated(self) -> None:
        result = doc("rtsp path", 0.4)
        rerank([result], "rtsp")
        self.assertEqual((result.score, result.components), (0.4, {}))

    def test_components_record_the_rerank(self) -> None:
        out, _ = rerank([doc("rtsp path front", 0.4)], "rtsp")
        self.assertIn("bm25", out[0].components)
        self.assertIn("rerank_delta", out[0].components)


if __name__ == "__main__":
    unittest.main()
