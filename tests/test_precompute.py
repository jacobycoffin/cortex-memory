"""Precompute must be correct, bounded and cheap (item #23)."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.precompute import Precompute, revision_of
from cortex.retrieval import MemoryRetriever
from cortex.store import CortexStore


class MemoTests(unittest.TestCase):
    def test_a_value_is_computed_once(self) -> None:
        memo = Precompute()
        calls = []

        def compute() -> str:
            calls.append(1)
            return "value"

        self.assertEqual(memo.get("k", compute, revision=1), "value")
        self.assertEqual(memo.get("k", compute, revision=1), "value")
        self.assertEqual(len(calls), 1)
        self.assertEqual(memo.hits, 1)

    def test_a_new_revision_recomputes(self) -> None:
        memo = Precompute()
        memo.get("k", lambda: "old", revision=1)
        self.assertEqual(memo.get("k", lambda: "new", revision=2), "new")

    def test_the_ttl_expires_the_value(self) -> None:
        memo = Precompute(ttl_seconds=0.01)
        memo.get("k", lambda: "old", revision=1)
        time.sleep(0.03)
        self.assertEqual(memo.get("k", lambda: "new", revision=1), "new")

    def test_priming_avoids_the_compute(self) -> None:
        memo = Precompute()
        memo.prime("k", "warm", revision=1)
        self.assertEqual(memo.get("k", lambda: "cold", revision=1), "warm")

    def test_stats_and_clear(self) -> None:
        memo = Precompute()
        memo.get("a", lambda: 1)
        memo.get("a", lambda: 1)
        stats = memo.stats()
        self.assertEqual((stats["hits"], stats["misses"]), (1, 1))
        memo.clear()
        self.assertEqual(memo.stats()["keys"], 0)

    def test_revision_of_a_broken_store_is_none(self) -> None:
        class Broken:
            def retrieval_revision(self) -> tuple[int, int]:
                raise RuntimeError("boom")

        self.assertIsNone(revision_of(Broken()))


class RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.store.add_memory(
            content="The maintenance window is Sunday 02:00 on CT116.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            importance=0.8,
        )
        self.retriever = MemoryRetriever(self.store)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_prospective_and_weights_are_memoised(self) -> None:
        first, _ = self.retriever.search_detailed("maintenance window", limit=6, token_budget=700)
        second, _ = self.retriever.search_detailed("maintenance window", limit=6, token_budget=700)
        self.assertEqual([r.memory["id"] for r in first], [r.memory["id"] for r in second])
        stats = self.retriever.precompute.stats()
        self.assertGreater(stats["hits"], 0)

    def test_a_write_invalidates_the_memoised_inputs(self) -> None:
        self.retriever.search_detailed("maintenance window", limit=6, token_budget=700)
        before = revision_of(self.store)
        self.store.add_memory(
            content="The maintenance window moved to Tuesday 03:00.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            importance=0.9,
        )
        after = revision_of(self.store)
        self.assertNotEqual(before, after)

    def test_edge_sets_match_the_store_queries(self) -> None:
        ids = [
            str(row[0])
            for row in self.store._conn.execute("SELECT id FROM memories").fetchall()
        ]
        srcs, dsts = self.retriever._edge_sets("supersedes")
        self.assertEqual(dsts, self.store.superseded_ids(ids))


if __name__ == "__main__":
    unittest.main()
