from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tests._bootstrap import ROOT

from cortex import CortexMemoryProvider
from cortex.cognition import plan_recall
from cortex.store import CortexStore


class DeterministicAttentionGateTests(unittest.TestCase):
    def test_more_social_only_turns_abstain(self) -> None:
        self.assertFalse(plan_recall("Good afternoon!").needs_memory)
        self.assertFalse(plan_recall("Got it.").needs_memory)

    def test_plain_arithmetic_abstains(self) -> None:
        plan = plan_recall("What is 17 * 23?")
        self.assertFalse(plan.needs_memory)
        self.assertEqual(plan.mode, "none")

    def test_memory_cue_prevents_transform_abstention(self) -> None:
        plan = plan_recall("Rewrite this using my usual style: release ready")
        self.assertTrue(plan.needs_memory)
        self.assertEqual(plan.mode, "focused")


class OutcomeDrivenBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.memory_id, _ = self.store.add_memory("The release theme is amber.", kind="decision")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _observation(self, task_type: str, *, used: bool, helpful: bool = False) -> None:
        task_id = self.store.create_usage_batch(
            [(self.memory_id, 0.8)],
            query="What is the release theme?",
            session_id="budget-test",
            task_type=task_type,
            recall_mode="focused",
            requested_budget=600,
            estimated_tokens=540,
        )
        self.store.resolve_usage(task_id, {self.memory_id: 0.9} if used else {})
        if helpful:
            self.store.apply_task_outcome(task_id, "helpful")

    def test_pending_outcomes_do_not_change_budget(self) -> None:
        for _index in range(12):
            self.store.create_usage_batch(
                [(self.memory_id, 0.8)],
                query="pending",
                session_id="budget-test",
                task_type="pending_task",
                recall_mode="focused",
                requested_budget=600,
                estimated_tokens=590,
            )
        diagnostics = self.store.recommend_token_budget("pending_task", "focused", 600)
        self.assertEqual(diagnostics["budget"], 600)
        self.assertEqual(diagnostics["sample_count"], 0)

    def test_repeated_ignored_context_shrinks_budget_conservatively(self) -> None:
        for _index in range(8):
            self._observation("ignored_task", used=False)
        diagnostics = self.store.recommend_token_budget("ignored_task", "focused", 600)
        self.assertEqual(diagnostics["budget"], 510)
        self.assertEqual(diagnostics["adjustment"], "shrink")
        self.assertEqual(diagnostics["scope"], "task_mode")
        self.assertEqual(diagnostics["ignored_ratio"], 1.0)
        aggregate = next(
            row
            for row in self.store.dashboard_snapshot()["recall_budgets"]
            if row["task_type"] == "ignored_task" and row["mode"] == "focused"
        )
        self.assertEqual(aggregate["sample_count"], 8)
        self.assertEqual(aggregate["ignored_count"], 8)

    def test_helpful_saturated_context_can_expand_within_cap(self) -> None:
        for index in range(8):
            self._observation("helpful_task", used=True, helpful=index < 3)
        diagnostics = self.store.recommend_token_budget(
            "helpful_task", "focused", 600, max_budget=640
        )
        self.assertEqual(diagnostics["budget"], 640)
        self.assertEqual(diagnostics["adjustment"], "expand")
        self.assertEqual(diagnostics["positive_count"], 3)
        self.assertGreaterEqual(diagnostics["average_fill"], 0.8)
        self.assertEqual(
            self.store.recommend_token_budget(
                "helpful_task", "focused", 100, max_budget=100
            )["budget"],
            100,
        )


class OutcomeDrivenAttentionLearningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.memory_id, _ = self.store.add_memory(
            "Plex deploys to the Proxmox server.", kind="procedure"
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _observation(
        self,
        *,
        mode: str,
        used: bool,
        helpful: bool = False,
    ) -> str:
        task_id = self.store.create_usage_batch(
            [(self.memory_id, 0.8)],
            query="Deploy Plex to Proxmox",
            session_id="attention-test",
            task_type="tool_execution",
            recall_mode=mode,
            requested_budget=600,
            estimated_tokens=520,
        )
        decision = self.store.record_attention_observation(
            task_id,
            task_type="tool_execution",
            topics=("plex", "proxmox"),
            live_mode=mode,
            live_budget=600,
            max_budget=700,
            selected_count=1,
        )
        self.assertFalse(decision["applied"])
        self.store.resolve_usage(task_id, {self.memory_id: 0.9} if used else {})
        if helpful:
            self.store.apply_task_outcome(task_id, "helpful")
        return task_id

    def test_helpful_topic_can_recommend_a_richer_prior_mode_in_shadow(self) -> None:
        for index in range(4):
            self._observation(mode="procedural", used=True, helpful=index < 2)

        recommendation = self.store.attention_recommendation(
            "tool_execution",
            ("plex",),
            "lean",
            600,
            max_budget=700,
        )
        self.assertEqual(recommendation["shadow_mode"], "procedural")
        self.assertEqual(recommendation["shadow_budget"], 660)
        self.assertEqual(recommendation["budget_delta"], 0.1)
        self.assertFalse(recommendation["applied"])
        row = next(
            item
            for item in self.store.attention_learning_summary()["weights"]
            if item["topic_key"] == "plex"
        )
        self.assertEqual(row["used_count"], 4)
        self.assertEqual(row["helpful_count"], 2)
        self.assertEqual(row["weight_delta"], 0.1)

    def test_repeated_ignored_deep_context_recommends_one_level_less(self) -> None:
        for _index in range(4):
            self._observation(mode="deep", used=False)

        recommendation = self.store.attention_recommendation(
            "tool_execution",
            ("proxmox",),
            "deep",
            600,
            max_budget=700,
        )
        self.assertEqual(recommendation["shadow_mode"], "focused")
        self.assertEqual(recommendation["shadow_budget"], 540)
        self.assertEqual(recommendation["budget_delta"], -0.1)

    def test_inactive_topic_weight_loses_half_its_strength_after_thirty_days(self) -> None:
        for _index in range(4):
            self._observation(mode="procedural", used=True)
        stale = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(timespec="milliseconds")
        with self.store.transaction() as conn:
            conn.execute(
                """UPDATE attention_weights SET last_observed_at=?
                   WHERE task_type='tool_execution' AND topic_key='plex'""",
                (stale,),
            )

        recommendation = self.store.attention_recommendation(
            "tool_execution",
            ("plex",),
            "procedural",
            600,
            max_budget=700,
        )
        evidence = next(
            item for item in recommendation["evidence"] if item["topic_key"] == "plex"
        )
        self.assertAlmostEqual(evidence["decay_factor"], 0.5, places=3)
        self.assertAlmostEqual(evidence["effective_delta"], 0.05, places=3)

    def test_pending_and_no_context_observations_never_train_topic_weights(self) -> None:
        pending = self.store.create_usage_batch(
            [(self.memory_id, 0.8)],
            query="Plex",
            session_id="attention-test",
            task_type="tool_execution",
            recall_mode="focused",
            requested_budget=600,
            estimated_tokens=400,
        )
        self.store.record_attention_observation(
            pending,
            task_type="tool_execution",
            topics=("plex",),
            live_mode="focused",
            live_budget=600,
            max_budget=700,
            selected_count=1,
        )
        self.store.record_attention_observation(
            "no-context",
            task_type="tool_execution",
            topics=("weather",),
            live_mode="none",
            live_budget=0,
            max_budget=700,
            selected_count=0,
        )

        report = self.store.attention_learning_summary()
        self.assertEqual(report["summary"]["resolved_count"], 0)
        self.assertEqual(report["summary"]["no_context_count"], 1)
        self.assertFalse(report["weights"])

    def test_dashboard_outcome_label_and_undo_rebuild_attention_counts(self) -> None:
        task_id = self._observation(mode="focused", used=True)
        labeled = self.store.label_task_outcome(task_id, "helpful", actor="attention-test")
        self.assertTrue(labeled["changed"])
        learned = next(
            item
            for item in self.store.attention_learning_summary()["weights"]
            if item["topic_key"] == "plex"
        )
        self.assertEqual(learned["helpful_count"], 1)

        self.assertTrue(self.store.undo_task_outcome_label(task_id, actor="attention-test"))
        restored = next(
            item
            for item in self.store.attention_learning_summary()["weights"]
            if item["topic_key"] == "plex"
        )
        self.assertEqual(restored["helpful_count"], 0)


class SafeRetrievalCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.provider = CortexMemoryProvider(
            {
                "db_path": "$HERMES_HOME/cortex/test.db",
                "auto_capture": False,
                "retrieval_threshold": 0.01,
                "query_cache_ttl_seconds": 45,
            }
        )
        self.provider.initialize("cache-session", hermes_home=self.tmp.name, agent_context="primary")
        self.memory_id, _ = self.provider._store.add_memory(
            "The launch color decision is amber.", kind="decision", confidence=0.95
        )

    def tearDown(self) -> None:
        self.provider.shutdown()
        self.tmp.cleanup()

    def test_cache_hit_still_tracks_each_injection_and_usage_batch(self) -> None:
        query = "What did I decide for the launch color?"
        with patch.object(
            self.provider._retriever,
            "search_detailed",
            wraps=self.provider._retriever.search_detailed,
        ) as search:
            first = self.provider.prefetch(query, session_id="cache-session")
            second = self.provider.prefetch(query, session_id="cache-session")

        self.assertEqual(first, second)
        self.assertEqual(search.call_count, 1)
        memory = self.provider._store.get_memory(self.memory_id)
        self.assertEqual(memory["retrieved_count"], 2)
        self.assertEqual(memory["selected_count"], 2)
        self.assertEqual(memory["injected_count"], 2)

    def test_context_feedback_invalidates_cached_ranking(self) -> None:
        query = "What did I decide for the launch color?"
        with patch.object(
            self.provider._retriever,
            "search_detailed",
            wraps=self.provider._retriever.search_detailed,
        ) as search:
            self.provider.prefetch(query, session_id="cache-session")
            self.provider.sync_turn(
                query,
                "The launch color decision is amber.",
                session_id="cache-session",
            )
            self.provider.prefetch(query, session_id="cache-session")

        self.assertEqual(search.call_count, 2)
        usage_count = self.provider._store._conn.execute(
            "SELECT COUNT(*) count FROM usage_records WHERE memory_id=?", (self.memory_id,)
        ).fetchone()["count"]
        self.assertEqual(usage_count, 2)
        prediction_count = self.provider._store._conn.execute(
            "SELECT COUNT(*) count FROM metacognitive_predictions WHERE memory_id=?",
            (self.memory_id,),
        ).fetchone()["count"]
        self.assertEqual(prediction_count, 2)

    def test_material_memory_change_invalidates_cached_result(self) -> None:
        query = "What did I decide for the launch color?"
        with patch.object(
            self.provider._retriever,
            "search_detailed",
            wraps=self.provider._retriever.search_detailed,
        ) as search:
            first = self.provider.prefetch(query, session_id="cache-session")
            self.assertIn("amber", first)
            external_store = CortexStore(self.provider._store.path)
            try:
                external_store.correct_memory(
                    self.memory_id, "The launch color decision is violet.", reason="test correction"
                )
            finally:
                external_store.close()
            second = self.provider.prefetch(query, session_id="cache-session")

        self.assertEqual(search.call_count, 2)
        self.assertIn("violet", second)
        self.assertNotIn("amber", second)

    def test_raw_sqlite_write_remains_compatible_and_invalidates_cache(self) -> None:
        query = "What did I decide for the launch color?"
        with patch.object(
            self.provider._retriever,
            "search_detailed",
            wraps=self.provider._retriever.search_detailed,
        ) as search:
            self.provider.prefetch(query, session_id="cache-session")
            external = sqlite3.connect(self.provider._store.path)
            try:
                external.execute(
                    "UPDATE memories SET importance=? WHERE id=?",
                    (0.99, self.memory_id),
                )
                external.commit()
            finally:
                external.close()
            self.provider.prefetch(query, session_id="cache-session")

        self.assertEqual(search.call_count, 2)


class MetacognitionEnforcementTests(unittest.TestCase):
    def test_enforcement_request_stays_shadow_until_promotion_gate_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = CortexMemoryProvider(
                {
                    "db_path": "$HERMES_HOME/cortex/test.db",
                    "auto_capture": False,
                    "retrieval_threshold": 0.0,
                    "metacognition_mode": "enforce",
                }
            )
            provider.initialize("monitor-session", hermes_home=tmp, agent_context="primary")
            try:
                memory_id, _ = provider._store.add_memory(
                    "The experimental service endpoint is the orange gateway.",
                    source_category="AGENT_INFERENCE",
                    confidence=0.25,
                    currentness_confidence=0.2,
                    trust=0.2,
                    volatility=1.0,
                )
                with provider._store.transaction() as conn:
                    conn.execute(
                        """UPDATE memories SET dirty=1,dirty_reason='test',harmful_count=5,
                                  false_positive_count=5,injected_count=5
                           WHERE id=?""",
                        (memory_id,),
                    )

                context = provider.prefetch(
                    "What is the experimental service endpoint?",
                    session_id="monitor-session",
                )

                self.assertIn("experimental service endpoint", context)
                prediction = provider._store._conn.execute(
                    """SELECT decision,applied,outcome FROM metacognitive_predictions
                       WHERE memory_id=?""",
                    (memory_id,),
                ).fetchone()
                self.assertEqual(dict(prediction), {"decision": "abstain", "applied": 0, "outcome": "pending"})
                self.assertEqual(provider._store.get_memory(memory_id)["injected_count"], 6)
            finally:
                provider.shutdown()

    def test_enforcement_can_withhold_after_promotion_gate_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            CortexStore,
            "metacognition_enforcement_gate",
            return_value={"ready": True},
        ):
            provider = CortexMemoryProvider(
                {
                    "db_path": "$HERMES_HOME/cortex/test.db",
                    "auto_capture": False,
                    "retrieval_threshold": 0.0,
                    "metacognition_mode": "enforce",
                }
            )
            provider.initialize("monitor-session", hermes_home=tmp, agent_context="primary")
            try:
                memory_id, _ = provider._store.add_memory(
                    "The experimental service endpoint is the orange gateway.",
                    source_category="AGENT_INFERENCE",
                    confidence=0.25,
                    currentness_confidence=0.2,
                    trust=0.2,
                    volatility=1.0,
                )
                with provider._store.transaction() as conn:
                    conn.execute(
                        """UPDATE memories SET dirty=1,dirty_reason='test',harmful_count=5,
                                  false_positive_count=5,injected_count=5
                           WHERE id=?""",
                        (memory_id,),
                    )

                context = provider.prefetch(
                    "What is the experimental service endpoint?",
                    session_id="monitor-session",
                )

                self.assertEqual(context, "")
                prediction = provider._store._conn.execute(
                    """SELECT decision,applied,outcome FROM metacognitive_predictions
                       WHERE memory_id=?""",
                    (memory_id,),
                ).fetchone()
                self.assertEqual(dict(prediction), {"decision": "abstain", "applied": 1, "outcome": "withheld"})
            finally:
                provider.shutdown()


if __name__ == "__main__":
    unittest.main()
