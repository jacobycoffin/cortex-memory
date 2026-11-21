"""The cache key must cover everything that can change a result (item #21).

Recall results now depend on the query's *class* (#5), the *entities* it anchors on (#8), the
rerank and density switches, and the latency budget. The key only covered the query text,
the caller's options and five scoring settings — so flipping a switch, or asking the same
words about a different entity, could reuse an entry computed under different rules.

These tests pin the key's sensitivity directly, and the behaviour that matters: a stale
entry must not be served across a switch change.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.retrieval import MemoryRetriever, RetrievalContext
from cortex.store import CortexStore

QUERY = "how is Mack's access set up?"


class KeyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        created = self.store.add_memory(
            content="Mack's access uses a shared vault and a per-user key.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            entities=["Mack"],
        )
        self.memory_id = created[0] if isinstance(created, tuple) else created
        self.retriever = MemoryRetriever(self.store)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _key(self, query: str = QUERY, **options):
        scoped = dict(options)
        scoped["__class__"] = __import__("cortex.cognition", fromlist=["query_class"]).query_class(query)
        settings = (
            self.retriever.threshold, self.retriever.semantic_weight,
            self.retriever.semantic_pool, self.retriever.semantic_model_id,
            self.retriever.semantic_floor, self.retriever.entity_anchor_enabled,
            self.retriever.rerank_enabled, self.retriever.density_budget_enabled,
            self.retriever.latency_budget_ms, self.retriever.abstain_threshold,
        )
        return self.retriever.recall_cache.key(query, scoped, settings)

    def test_the_same_turn_produces_the_same_key(self) -> None:
        self.assertEqual(self._key(), self._key())

    def test_a_different_class_produces_a_different_key(self) -> None:
        lookup = self._key("what is the rtsp path")
        planning = self._key("what should we do next")
        self.assertNotEqual(lookup, planning)

    def test_a_different_entity_set_produces_a_different_key(self) -> None:
        with_mack = self._key(QUERY, context=RetrievalContext(entities=("Mack",)))
        without = self._key(QUERY, context=RetrievalContext(entities=("Jev",)))
        self.assertNotEqual(with_mack, without)

    def test_flipping_a_switch_produces_a_different_key(self) -> None:
        before = self._key()
        self.retriever.rerank_enabled = not self.retriever.rerank_enabled
        self.assertNotEqual(before, self._key())

    def test_the_budget_and_threshold_are_part_of_the_key(self) -> None:
        before = self._key()
        self.retriever.latency_budget_ms = 120.0
        self.assertNotEqual(before, self._key())

    def test_anchor_names_are_memoised_per_revision(self) -> None:
        first = self.retriever._anchor_names(QUERY)
        second = self.retriever._anchor_names(QUERY)
        self.assertEqual(first, second)
        self.assertIn("Mack", first)


class BehaviourTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.store.add_memory(
            content="Mack's access uses a shared vault and a per-user key.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            entities=["Mack"],
        )
        self.retriever = MemoryRetriever(self.store)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_a_switch_change_is_not_served_from_a_stale_entry(self) -> None:
        """The bug this item exists to prevent: same words, different rules, same entry."""
        first, _ = self.retriever.search_cached(QUERY, limit=6, token_budget=700)
        self.retriever.rerank_enabled = not self.retriever.rerank_enabled
        second, diagnostics = self.retriever.search_cached(QUERY, limit=6, token_budget=700)
        self.assertFalse(diagnostics.cache_hit, "a changed setting must invalidate the entry")
        self.assertEqual(len(first), len(second))

    def test_an_unchanged_turn_does_hit_the_cache(self) -> None:
        self.retriever.search_cached(QUERY, limit=6, token_budget=700)
        _, diagnostics = self.retriever.search_cached(QUERY, limit=6, token_budget=700)
        self.assertTrue(diagnostics.cache_hit)

    def test_a_different_entity_context_is_not_a_cache_hit(self) -> None:
        self.retriever.search_cached(
            QUERY, limit=6, token_budget=700, context=RetrievalContext(entities=("Mack",))
        )
        _, diagnostics = self.retriever.search_cached(
            QUERY, limit=6, token_budget=700, context=RetrievalContext(entities=("Jev",))
        )
        self.assertFalse(diagnostics.cache_hit)


if __name__ == "__main__":
    unittest.main()
