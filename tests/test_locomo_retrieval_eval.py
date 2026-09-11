"""
Tests for the shipped LoCoMo retrieval harness (scripts/locomo_retrieval_eval.py).

The harness is semantic fusion's **evidence path**. Two things are pinned here:

1. ``score_ranking`` — the single scorer both arms share. If the fusion arm and
   the paired weight-0 arm ever diverged in how they score, every delta this
   harness publishes would become meaningless, so the scorer is tested directly.
2. ``embed_store`` failing LOUD when the embedding model is unavailable. A fusion
   run that silently fell back to the feature-only path would publish a "fusion"
   number that never used fusion. That is the failure mode this repo keeps
   rediscovering ("the mechanism ships, the evidence path doesn't"), so it is
   asserted rather than trusted.

The CLI default is pinned too: fusion OFF (weight 0.0) must stay the shipped
default so the published baseline remains reproducible.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "locomo_retrieval_eval.py"


def _load_harness():
    """Load the harness by path — it is a script, not an importable module."""
    spec = importlib.util.spec_from_file_location("locomo_retrieval_eval", SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("could not load the LoCoMo retrieval harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = _load_harness()


def _ensure_cortex_importable() -> bool:
    """Mirror tests/_bootstrap.py so this module can import ``cortex.*``."""
    try:
        import cortex  # noqa: F401

        return True
    except ModuleNotFoundError:
        pass
    spec = importlib.util.spec_from_file_location(
        "cortex", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        return False
    module = importlib.util.module_from_spec(spec)
    sys.modules["cortex"] = module
    spec.loader.exec_module(module)
    return True


class ScoreRankingTests(unittest.TestCase):
    """The scorer shared by the fusion arm and the paired baseline arm."""

    def test_perfect_ranking_scores_one(self):
        hit, recall, rr, ndcg, session_hit, abstained = harness.score_ranking(
            ["a", "b", "c"], {"a"}, {1}, {"a": 1, "b": 2, "c": 2}, 10, False
        )
        self.assertEqual(hit, 1)
        self.assertEqual(recall, 1.0)
        self.assertEqual(rr, 1.0)
        self.assertAlmostEqual(ndcg, 1.0)
        self.assertEqual(session_hit, 1)
        self.assertFalse(abstained)

    def test_evidence_below_the_cut_off_does_not_count(self):
        hit, *_ = harness.score_ranking(
            ["x", "y"], {"a"}, {1}, {"x": 1, "y": 1}, 1, False
        )
        self.assertEqual(hit, 0)

    def test_reciprocal_rank_reflects_position(self):
        _, _, rr, *_ = harness.score_ranking(
            ["x", "y", "a"], {"a"}, {2}, {"x": 9, "y": 9, "a": 2}, 10, False
        )
        self.assertAlmostEqual(rr, 1.0 / 3.0)

    def test_partial_recall_when_only_some_evidence_is_found(self):
        _, recall, *_ = harness.score_ranking(
            ["a"], {"a", "b"}, {1}, {"a": 1}, 10, False
        )
        self.assertAlmostEqual(recall, 0.5)

    def test_clean_miss_scores_zero_and_is_not_nan(self):
        hit, recall, rr, ndcg, session_hit, _ = harness.score_ranking(
            ["x"], {"a"}, {7}, {"x": 1}, 10, False
        )
        self.assertEqual((hit, recall, rr, ndcg, session_hit), (0, 0.0, 0.0, 0.0, 0))

    def test_session_hit_is_deliberately_coarser_than_hit(self):
        """A wrong turn from the RIGHT session still counts as a session hit.

        session-hit@k answers "did we reach the right conversation slice", not
        "did we find the evidence turn". It is a weaker, separate signal and must
        not be read as a turn-level success — this test exists to stop anyone
        (including a future me) from conflating the two.
        """
        hit, _, _, _, session_hit, _ = harness.score_ranking(
            ["x"], {"a"}, {1}, {"x": 1}, 10, False
        )
        self.assertEqual(hit, 0)
        self.assertEqual(session_hit, 1)

    def test_session_hit_ignores_turns_outside_the_evidence_session(self):
        _, _, _, _, session_hit, _ = harness.score_ranking(
            ["x"], {"a"}, {7}, {"x": 3}, 10, False
        )
        self.assertEqual(session_hit, 0)

    def test_abstain_is_passed_through_untouched(self):
        *_, abstained = harness.score_ranking([], {"a"}, {1}, {}, 10, True)
        self.assertTrue(abstained)

    def test_paired_metric_names_are_unique(self):
        names = list(harness._PAIRED_METRICS)
        self.assertEqual(len(names), len(set(names)))


class FusionDefaultTests(unittest.TestCase):
    """Fusion must stay opt-in: the shipped baseline has to keep reproducing."""

    def test_cli_defaults_leave_fusion_off(self):
        args = harness.build_arg_parser().parse_args([])
        self.assertEqual(args.semantic_weight, 0.0)
        self.assertEqual(args.semantic_pool, 20)

    def test_semantic_weight_is_accepted(self):
        args = harness.build_arg_parser().parse_args(["--semantic-weight", "10"])
        self.assertEqual(args.semantic_weight, 10.0)


class EmbedStoreFailsLoudTests(unittest.TestCase):
    """A fusion number without embeddings would be a fabricated result."""

    def test_refuses_to_run_without_an_embedding_model(self):
        if not _ensure_cortex_importable():  # pragma: no cover - environment dependent
            self.skipTest("cortex package not importable here")

        import cortex.embeddings as embeddings_module

        class _Unavailable:
            available = False
            model_dir = "/nonexistent-model-dir"
            load_error = "model files missing"

        original = embeddings_module.get_embedder
        embeddings_module.get_embedder = lambda model_dir=None: _Unavailable()
        try:
            with self.assertRaises(RuntimeError) as ctx:
                harness.embed_store(object(), {"mem-1": "some text"})
        finally:
            embeddings_module.get_embedder = original

        self.assertIn("refusing to report a fusion number", str(ctx.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
