"""
Integration guards for retrieval-time semantic fusion.

These use a STUB embedder, so they run in milliseconds and do not depend on the
221 MB model being installed. What they protect:

 1. fusion is OFF by default — enabling it must be an explicit choice, and the
    default path must not change at all
 2. a semantic hit that the lexical pass never surfaced still reaches the
    candidate pool (the whole point of the feature — re-ranking alone cannot
    rescue a memory that was never retrieved)
 3. missing model / missing vectors degrade to previous behaviour, never raise
 4. an embedding for a DIFFERENT model_id is never used

Everything runs against a temp database. The live cortex.db is never touched.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cortex import embeddings as embeddings_module
from cortex.retrieval import MemoryRetriever
from cortex.store import CortexStore


class _StubEmbedder:
    """Deterministic stand-in for the ONNX embedder."""

    model_id = "stub-model"
    dim = 4

    def __init__(self, available: bool = True, vector: list[float] | None = None):
        self._available = available
        self._vector = vector if vector is not None else [1.0, 0.0, 0.0, 0.0]

    @property
    def available(self) -> bool:
        return self._available

    def embed_one(self, text: str) -> list[float]:      # noqa: ARG002
        return list(self._vector) if self._available else []

    def embed(self, texts):
        return [list(self._vector) for _ in texts] if self._available else []


class SemanticFusionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(str(Path(self._tmp.name) / "cortex.db"))
        self._orig_get_embedder = embeddings_module.get_embedder

        self.ids = {}
        for key, text in (
            ("alpha", "alpha memory about gardening and soil"),
            ("beta", "beta memory about car repair"),
            ("gamma", "gamma memory about music practice"),
        ):
            result = self.store.add_memory(text, state="active")
            self.ids[key] = result[0] if isinstance(result, tuple) else result

    def tearDown(self):
        embeddings_module.get_embedder = self._orig_get_embedder
        self._tmp.cleanup()

    def _patch_embedder(self, embedder):
        embeddings_module.get_embedder = lambda model_dir=None: embedder

    def _retriever(self, **kwargs) -> MemoryRetriever:
        return MemoryRetriever(self.store, **kwargs)

    # ------------------------------------------------------------------ guards

    def test_fusion_is_off_by_default(self):
        retriever = self._retriever()
        self.assertEqual(retriever.semantic_weight, 0.0)
        # With no embedder patched in at all, a search must still work.
        results, _ = retriever.search_detailed("gardening soil", limit=5)
        self.assertIsInstance(results, list)

    def test_default_path_ignores_semantic_candidates(self):
        """With weight 0, stored embeddings must not influence the result."""
        self.store.set_memory_embedding(self.ids["gamma"], "stub-model", [1.0, 0.0, 0.0, 0.0])
        self._patch_embedder(_StubEmbedder())

        off = [r.memory["id"] for r in self._retriever().search_detailed("gardening soil", limit=3)[0]]

        # Now delete the embedding and re-run: with fusion off the answer must
        # be identical, proving vectors are not consulted on the default path.
        self.store.delete_memory_embeddings("stub-model")
        off_again = [r.memory["id"] for r in self._retriever().search_detailed("gardening soil", limit=3)[0]]
        self.assertEqual(off, off_again)

    def test_semantic_hit_reaches_the_pool_when_enabled(self):
        """A memory the lexical pass scores near-zero must still surface."""
        # Give 'gamma' a vector identical to the query, and give the query text
        # no lexical overlap with it at all.
        self.store.set_memory_embedding(self.ids["gamma"], "stub-model", [1.0, 0.0, 0.0, 0.0])
        self._patch_embedder(_StubEmbedder(vector=[1.0, 0.0, 0.0, 0.0]))

        results, _ = self._retriever(semantic_weight=30.0).search_detailed(
            "zzzz qqqq unrelated token soup", limit=3)
        ids = [r.memory["id"] for r in results]
        self.assertIn(self.ids["gamma"], ids,
                      "semantic candidate did not reach the pool — fusion is not augmenting")

    def test_missing_model_degrades_to_default_behaviour(self):
        self.store.set_memory_embedding(self.ids["gamma"], "stub-model", [1.0, 0.0, 0.0, 0.0])
        self._patch_embedder(_StubEmbedder(available=False))

        on = [r.memory["id"] for r in self._retriever(semantic_weight=30.0)
              .search_detailed("gardening soil", limit=3)[0]]
        off = [r.memory["id"] for r in self._retriever()
               .search_detailed("gardening soil", limit=3)[0]]
        self.assertEqual(on, off, "unavailable model must not change retrieval")

    def test_no_stored_vectors_degrades_to_default_behaviour(self):
        self._patch_embedder(_StubEmbedder())          # available, but store has no vectors
        on = [r.memory["id"] for r in self._retriever(semantic_weight=30.0)
              .search_detailed("gardening soil", limit=3)[0]]
        off = [r.memory["id"] for r in self._retriever()
               .search_detailed("gardening soil", limit=3)[0]]
        self.assertEqual(on, off)

    def test_other_models_embeddings_are_ignored(self):
        """Vectors for a different model must never be used."""
        self.store.set_memory_embedding(self.ids["gamma"], "some-other-model", [1.0, 0.0, 0.0, 0.0])
        self._patch_embedder(_StubEmbedder())          # model_id == 'stub-model'
        on = [r.memory["id"] for r in self._retriever(semantic_weight=30.0)
              .search_detailed("zzzz qqqq unrelated", limit=3)[0]]
        off = [r.memory["id"] for r in self._retriever()
               .search_detailed("zzzz qqqq unrelated", limit=3)[0]]
        self.assertEqual(on, off, "a foreign model_id's vectors were used")


if __name__ == "__main__":
    unittest.main()
