"""Synthetic concurrency, invalidation, and audit checks for recall warming."""
from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from tests._bootstrap import ROOT  # noqa: F401
from cortex.client import CortexMemory
from cortex.hermes_provider import CortexMemoryProvider
from cortex.harness import CortexHarnessAdapter
from cortex.preload import BackgroundPreloader, RecallCache
from cortex.retrieval import RetrievalContext, SCORE_SIGNAL_WEIGHTS


class PreloaderQueueTests(unittest.TestCase):
    def test_latest_pending_hint_wins_and_close_rejects_work(self):
        started, release = threading.Event(), threading.Event()
        seen = []
        def warm(query, options):
            seen.append(query)
            if query == "first":
                started.set()
                self.assertTrue(release.wait(5))
        worker = BackgroundPreloader(warm)
        try:
            self.assertTrue(worker.submit("first", {}))
            self.assertTrue(started.wait(5))
            worker.submit("old pending", {})
            worker.submit("new pending", {})
            release.set()
            self.assertTrue(worker.wait_idle(5))
            self.assertEqual(seen, ["first", "new pending"])
            self.assertEqual(worker.stats()["replaced"], 1)
        finally:
            release.set()
            worker.close()
        self.assertFalse(worker.submit("after close", {}))

    def test_failure_is_contained_and_worker_can_recover(self):
        def warm(query, options):
            if query == "fail":
                raise RuntimeError("synthetic failure")
        worker = BackgroundPreloader(warm)
        try:
            worker.submit("fail", {})
            self.assertTrue(worker.wait_idle())
            worker.submit("recover", {})
            self.assertTrue(worker.wait_idle())
            self.assertEqual(worker.stats()["failed"], 1)
            self.assertEqual(worker.stats()["completed"], 1)
            self.assertFalse(worker.submit("x" * 4097, {}))
            self.assertFalse(worker.submit("oversized candidate pool", {"limit": 21}))
            self.assertFalse(worker.submit("oversized token budget", {"token_budget": 4001}))
            self.assertFalse(worker.submit("oversized context", {"context": RetrievalContext(scope={str(i): "value" for i in range(65)})}))
        finally:
            worker.close()

    def test_cache_does_not_publish_a_mixed_revision_result(self):
        cache = RecallCache()
        revision = [1]
        def compute():
            revision[0] += 1
            return ["synthetic"]
        cache.search("key", lambda: (revision[0], 1), compute, speculative=True)
        self.assertEqual(cache.stats()["entries"], 0)
        self.assertEqual(cache.stats()["discarded"], 1)

    def test_cache_is_bounded_expires_and_does_not_alias_mutable_results(self):
        cache = RecallCache(ttl_seconds=1, max_entries=2)
        value = {"nested": [1]}
        with patch("cortex.preload.time.monotonic", return_value=0):
            cache.search("first", lambda: (1, 1), lambda: value)
            result = cache.search("first", lambda: (1, 1), lambda: None)
            result["nested"].append(2)
            self.assertEqual(cache.search("first", lambda: (1, 1), lambda: None), value)
            cache.search("second", lambda: (1, 1), lambda: value)
            cache.search("third", lambda: (1, 1), lambda: value)
            self.assertEqual(cache.stats()["entries"], 2)
        with patch("cortex.preload.time.monotonic", return_value=2):
            self.assertEqual(cache.search("third", lambda: (1, 1), lambda: "fresh"), "fresh")


class RecallWarmingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "memory.db"
        self.memory = CortexMemory(self.path)
        self.query = "What is the synthetic deployment gateway address?"
        self.mid, _ = self.memory.remember(
            "The synthetic deployment gateway address is 192.0.2.20.", kind="decision",
            source_category="USER_EXPLICIT", confidence=0.95, importance=0.8,
        )

    def tearDown(self):
        self.memory.close()
        self.tmp.cleanup()

    def test_warm_is_read_only_and_real_recall_has_its_own_usage_records(self):
        store = self.memory.store
        before = store._conn.total_changes
        with patch.object(self.memory.retriever, "search_detailed", wraps=self.memory.retriever.search_detailed) as search:
            self.assertTrue(self.memory.preload(self.query))
            self.assertTrue(self.memory._preloader.wait_idle())
            self.assertEqual(store._conn.total_changes, before)
            batch = self.memory.recall(self.query)
            self.assertEqual(search.call_count, 1)
            self.assertIn(self.mid, [row["id"] for row in batch.memories])
            second = self.memory.recall(self.query)
            self.assertNotEqual(batch.task_id, second.task_id)
            self.assertEqual(self.memory.preload_stats()["cache"]["preload_hits"], 2)
        self.assertEqual(store._conn.execute("SELECT COUNT(*) FROM recall_runs").fetchone()[0], 2)
        self.assertEqual(store.get_memory(self.mid)["used_count"], 0)
        self.assertTrue(store.audit()["ok"])

    def test_warm_invalidates_on_correction_and_preserves_old_version(self):
        self.memory.preload(self.query)
        self.assertTrue(self.memory._preloader.wait_idle())
        self.assertTrue(self.memory.store.correct_memory(self.mid, "The synthetic deployment gateway address is 192.0.2.21."))
        batch = self.memory.recall(self.query)
        self.assertEqual(batch.memories[0]["id"], self.mid)
        self.assertIn("192.0.2.21", batch.memories[0]["content"])
        self.assertTrue(any("192.0.2.20" in version["content"] for version in self.memory.store.versions(self.mid)))
        self.assertIsNotNone(self.memory.store.get_memory(self.mid))
        self.assertEqual(self.memory.preload_stats()["cache"]["preload_hits"], 0)

    def test_external_write_invalidates_warm_result(self):
        self.memory.preload(self.query)
        self.assertTrue(self.memory._preloader.wait_idle())
        with sqlite3.connect(self.path) as conn:
            conn.execute("UPDATE memories SET state='archived' WHERE id=?", (self.mid,))
        batch = self.memory.recall(self.query)
        self.assertEqual(batch.memories, [])

    def test_context_entities_and_versions_partition_the_cache(self):
        retriever = self.memory.retriever
        with patch.object(retriever, "search_detailed", wraps=retriever.search_detailed) as search:
            for context in (
                RetrievalContext(entities=("one",), applicable_versions=("v1",)),
                RetrievalContext(entities=("two",), applicable_versions=("v1",)),
                RetrievalContext(entities=("two",), applicable_versions=("v2",)),
            ):
                retriever.search_cached(self.query, context=context)
            self.assertEqual(search.call_count, 3)

    def test_provenance_change_invalidates_cached_evidence(self):
        self.memory.preload(self.query)
        self.assertTrue(self.memory._preloader.wait_idle())
        with self.memory.store.transaction() as conn:
            conn.execute("UPDATE memories SET source_ref='synthetic-reference' WHERE id=?", (self.mid,))
        batch = self.memory.recall(self.query)
        self.assertEqual(batch.memories[0]["source_ref"], "synthetic-reference")
        self.assertEqual(self.memory.preload_stats()["cache"]["preload_hits"], 0)

    def test_feedback_invalidates_warming(self):
        batch = self.memory.recall(self.query)
        self.memory.preload(self.query)
        self.assertTrue(self.memory._preloader.wait_idle())
        batch.finish(used_memory_ids=[self.mid], outcome="harmful")
        with patch.object(self.memory.retriever, "search_detailed", wraps=self.memory.retriever.search_detailed) as search:
            self.memory.recall(self.query)
            self.assertEqual(search.call_count, 1)

    def test_approved_scoring_profile_invalidates_warm_ranking(self):
        self.memory.preload(self.query)
        self.assertTrue(self.memory._preloader.wait_idle())
        weights = {**SCORE_SIGNAL_WEIGHTS, "lexical": .25, "utility": .06}
        with self.memory.store.transaction() as conn:
            conn.execute("""INSERT INTO scoring_weight_history
                (history_id,task_type,policy_version,weights_json,status,activated_at,actor)
                VALUES('synthetic-profile','__all__','synthetic-v1',?,'active','2026-01-01','test')""",
                (json.dumps(weights),))
        batch = self.memory.recall(self.query)
        self.assertEqual(batch.memories[0]["components"]["live_weight_lexical"], .25)
        self.assertEqual(self.memory.preload_stats()["cache"]["preload_hits"], 0)

    def test_cached_stage_timing_reports_lookup_instead_of_old_search_work(self):
        options = {"context": RetrievalContext(goal=self.query)}
        self.memory.retriever.search_cached(self.query, speculative=True, **options)
        _, diagnostics = self.memory.retriever.search_cached(self.query, **options)
        self.assertTrue(diagnostics.cache_hit)
        self.assertTrue(diagnostics.preloaded)
        self.assertEqual(diagnostics.stage_ms["feature"], 0)
        self.assertGreater(diagnostics.stage_ms["cache_lookup"], 0)

    def test_core_ledger_is_atomic_and_uses_one_commit(self):
        statements = []
        self.memory.store._conn.set_trace_callback(statements.append)
        batch = self.memory.recall(self.query)
        self.memory.store._conn.set_trace_callback(None)
        self.assertEqual(statements.count("COMMIT"), 1)
        self.assertIsNotNone(self.memory.store.memory_traces(task_id=batch.task_id))
        before = self.memory.store._conn.execute("SELECT COUNT(*) FROM usage_records").fetchone()[0]
        with patch.object(self.memory.store, "record_memory_trace_decision", side_effect=RuntimeError("synthetic")):
            with self.assertRaises(RuntimeError):
                self.memory.recall(self.query)
        self.assertEqual(self.memory.store._conn.execute("SELECT COUNT(*) FROM usage_records").fetchone()[0], before)
        self.assertEqual(self.memory.store._conn.execute("SELECT COUNT(*) FROM recall_runs").fetchone()[0], 1)

    def test_hermes_hint_warms_without_usage_and_shutdown_drains_worker(self):
        provider = CortexMemoryProvider(dict(db_path=str(self.path), background_preload=True, background_preload_continuity=True,
                                            auto_capture=False, adaptive_budget_learning=False))
        provider.initialize("test", hermes_home=Path(self.tmp.name))
        try:
            before = provider._store._conn.total_changes
            provider.queue_prefetch(self.query, session_id="test")
            self.assertTrue(provider._preloader.wait_idle())
            self.assertEqual(provider._store._conn.total_changes, before)
            statements = []
            provider._store._conn.set_trace_callback(statements.append)
            context = provider.prefetch(self.query, session_id="test")
            provider._store._conn.set_trace_callback(None)
            self.assertIn("192.0.2.20", context)
            self.assertEqual(statements.count("COMMIT"), 1)
            self.assertEqual(provider.preload_stats()["cache"]["preload_hits"], 1)
            provider.sync_turn(self.query, "Synthetic response.", session_id="test")
            self.assertTrue(provider._preloader.wait_idle())
            self.assertGreaterEqual(provider.preload_stats()["worker"]["completed"], 2)
        finally:
            provider.shutdown()
        self.assertIsNone(provider._preloader)

    def test_hermes_ledger_failure_rolls_back_access_and_pending_attribution(self):
        provider = CortexMemoryProvider(dict(db_path=str(self.path), auto_capture=False))
        provider.initialize("test", hermes_home=Path(self.tmp.name))
        try:
            with patch.object(provider._store, "record_memory_trace_decision", side_effect=RuntimeError("synthetic")):
                with self.assertRaises(RuntimeError):
                    provider.prefetch(self.query, session_id="test")
            self.assertEqual(provider._store._conn.execute("SELECT COUNT(*) FROM recall_runs").fetchone()[0], 0)
            self.assertEqual(provider._store._conn.execute("SELECT COUNT(*) FROM usage_records").fetchone()[0], 0)
            self.assertEqual(provider._store.get_memory(self.mid)["injected_count"], 0)
            self.assertEqual(provider._peek_current_prefetches("test"), ([], []))
        finally:
            provider.shutdown()

    def test_harness_hint_uses_the_same_plan_and_greetings_do_not_warm(self):
        adapter = CortexHarnessAdapter(self.path, token_budget=500)
        try:
            self.assertFalse(adapter.preload("Hello"))
            self.assertTrue(adapter.preload(self.query, active_project="Synthetic", session_id="test"))
            self.assertTrue(adapter.memory._preloader.wait_idle())
            with patch.object(adapter.memory.retriever, "search_detailed", wraps=adapter.memory.retriever.search_detailed) as search:
                turn = adapter.before_turn(self.query, active_project="Synthetic", session_id="test")
                self.assertIn("192.0.2.20", turn.context)
                self.assertEqual(search.call_count, 0)
                self.assertEqual(adapter.memory.preload_stats()["cache"]["preload_hits"], 1)
        finally:
            adapter.close()

    def test_hermes_background_work_is_disabled_by_default(self):
        provider = CortexMemoryProvider(dict(db_path=str(self.path)))
        provider.initialize("test", hermes_home=Path(self.tmp.name))
        try:
            provider.queue_prefetch(self.query)
            self.assertFalse(provider.preload_stats()["enabled"])
        finally:
            provider.shutdown()


if __name__ == "__main__":
    unittest.main()
