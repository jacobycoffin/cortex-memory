from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


from tests._bootstrap import ROOT

from cortex import CortexMemoryProvider
from cortex.attribution import (
    attribution_score,
    memory_receipt_prefixes,
    referenced_memory_prefixes,
    strip_memory_receipt,
)
from cortex.cognition import attention_topics, plan_recall
from cortex.retrieval import MemoryRetriever
from cortex.store import CortexStore
from cortex.tooling import extract_tool_executions


class RecallPlannerTests(unittest.TestCase):
    def test_attention_gate_skips_social_and_self_contained_turns(self) -> None:
        self.assertFalse(plan_recall("Thanks!").needs_memory)
        self.assertFalse(plan_recall("Translate this into French: the door is open.").needs_memory)

    def test_attention_gate_scales_context_to_task(self) -> None:
        focused = plan_recall("What did I decide for my Hermes backend?")
        deep = plan_recall("Why did my Hermes backend change and how is it related to the vault?")
        procedural = plan_recall("Deploy the service and test the API")
        self.assertEqual(focused.mode, "focused")
        self.assertEqual(deep.mode, "deep")
        self.assertGreater(deep.token_budget, focused.token_budget)
        self.assertEqual(procedural.mode, "procedural")
        self.assertGreater(procedural.tool_limit, 0)

    def test_explicit_attention_override_is_deep_and_topics_are_bounded(self) -> None:
        plan = plan_recall("Pay close attention to Plex and Proxmox during this deploy")
        self.assertEqual(plan.mode, "deep")
        self.assertIn("explicit user request", plan.reason)
        self.assertEqual(
            attention_topics("Pay close attention to Plex and Proxmox during this deploy"),
            ("plex", "proxmox", "deploy"),
        )


class HybridMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_transparent_feature_index_catches_modest_paraphrase(self) -> None:
        memory_id, _ = self.store.add_memory(
            "The VPS hosts the Cortex brain.", kind="operational", confidence=0.9, importance=0.8
        )
        self.assertFalse(self.store.fts_search("Which computer contains it?"))
        results = MemoryRetriever(self.store, threshold=0.05).search("Which computer contains it?", limit=3)
        self.assertIn(memory_id, {result.memory["id"] for result in results})

        database_id, _ = self.store.add_memory(
            "Project Amber's verified database backend is SQLite.", kind="decision"
        )
        database_results = MemoryRetriever(self.store, threshold=0.05).search(
            "Which data store was approved for Project Amber?", limit=3
        )
        self.assertIn(database_id, {result.memory["id"] for result in database_results})

    def test_one_rare_token_cannot_outrank_direct_multiword_match(self) -> None:
        incidental_id, _ = self.store.add_memory(
            "Infra changes need approval before applying; recall the unrelated backup checklist first.",
            kind="preference",
            importance=1.0,
            confidence=1.0,
        )
        direct_id, _ = self.store.add_memory(
            "Cortex uses adaptive recall and reversible consolidation.",
            kind="semantic",
            importance=0.5,
            confidence=0.7,
        )
        results = MemoryRetriever(self.store, threshold=0.0).search(
            "Cortex adaptive recall reversible consolidation", limit=2
        )
        self.assertEqual(results[0].memory["id"], direct_id)
        self.assertNotIn(incidental_id, {result.memory["id"] for result in results})

    def test_temporal_recall_prefers_memory_valid_at_target_time(self) -> None:
        old_id, _ = self.store.add_memory(
            "The Cortex dashboard used port 8765 in 2025.",
            kind="operational",
            subject="cortex:dashboard",
            predicate="port",
            object_value="8765",
            valid_from="2025-01-01T00:00:00+00:00",
            valid_to="2025-12-31T23:59:59+00:00",
        )
        current_id, _ = self.store.add_memory(
            "The Cortex dashboard uses port 8787 in 2026.",
            kind="operational",
            subject="cortex:dashboard",
            predicate="port",
            object_value="8787",
            valid_from="2026-01-01T00:00:00+00:00",
            supersedes_id=old_id,
        )
        retriever = MemoryRetriever(self.store, threshold=0.0)
        historical = retriever.search(
            "Cortex dashboard port", temporal_mode="historical", as_of="2025-06-01T00:00:00+00:00", limit=2
        )
        current = retriever.search("Cortex dashboard port", temporal_mode="current", limit=2)
        self.assertEqual(historical[0].memory["id"], old_id)
        self.assertEqual(current[0].memory["id"], current_id)

    def test_consolidation_is_reversible(self) -> None:
        canonical, _ = self.store.add_memory(
            "Cortex dashboard is available at brain.example.test.", kind="semantic", importance=0.8
        )
        duplicate, _ = self.store.add_memory(
            "The Cortex dashboard lives at brain.example.test.", kind="semantic", importance=0.5
        )
        preview = self.store.consolidate(dry_run=True, similarity_threshold=0.3)
        self.assertGreaterEqual(preview["member_count"], 1)
        applied = self.store.consolidate(dry_run=False, similarity_threshold=0.3)
        self.assertEqual(self.store.get_memory(canonical)["state"], "active")
        self.assertEqual(self.store.get_memory(duplicate)["state"], "cold")
        undone = self.store.undo_consolidation(applied["run_id"])
        self.assertEqual(undone["restored"], 1)
        self.assertEqual(self.store.get_memory(duplicate)["state"], "active")

    def test_pruning_regret_can_restore_archived_memory(self) -> None:
        memory_id, _ = self.store.add_memory("Rare but useful Cortex recovery phrase.", kind="procedure")
        self.store.set_state(memory_id, "archived", reason="test pruning")
        self.assertTrue(self.store.record_pruning_regret(memory_id, query="recovery phrase", score=0.8, restore=True))
        self.assertEqual(self.store.get_memory(memory_id)["state"], "active")
        self.assertEqual(self.store.stats()["pruning_regrets"], 1)


class AttributionAndWorkflowTests(unittest.TestCase):
    def test_structured_value_is_strong_attribution(self) -> None:
        memory = {"content": "The dashboard uses port 8787.", "object_value": "8787"}
        self.assertEqual(attribution_score(memory, "Connect to port 8787."), 1.0)

    def test_vague_conceptual_similarity_does_not_receive_credit(self) -> None:
        memory = {"content": "Use the website research tool for current papers.", "object_value": ""}
        self.assertLess(attribution_score(memory, "I completed the task."), 0.18)

    def test_memory_receipt_syntax_is_bounded_and_removable(self) -> None:
        response = (
            "The deployment uses the verified route.\n\n"
            "Cortex memory: M:12ab34cd, M:98ef76ab"
        )
        self.assertEqual(
            memory_receipt_prefixes(response),
            ["12ab34cd", "98ef76ab"],
        )
        self.assertEqual(
            strip_memory_receipt(response),
            "The deployment uses the verified route.",
        )
        self.assertEqual(
            referenced_memory_prefixes("M:12ab34cd was wrong."),
            ["12ab34cd"],
        )
        self.assertEqual(
            memory_receipt_prefixes(
                "Cortex memory: M:12ab34cd, M:98ef76ab, M:11111111, M:22222222"
            ),
            [],
        )

    def test_json_containing_word_error_is_not_automatically_a_failure(self) -> None:
        messages = [
            {"role": "user", "content": "Inspect the logs."},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call-1", "function": {"name": "log_reader", "arguments": "{}"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": json.dumps({"ok": True, "message": "No error records were found"}),
            },
        ]
        execution = extract_tool_executions(messages, session_id="test")[0]
        self.assertTrue(execution.success)


class ProviderEfficiencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.provider = CortexMemoryProvider(
            {"db_path": "$HERMES_HOME/cortex/test.db", "retrieval_threshold": 0.05, "compact_context": True}
        )
        self.provider.initialize("adaptive-session", hermes_home=self.tmp.name, agent_context="primary")

    def tearDown(self) -> None:
        self.provider.shutdown()
        self.tmp.cleanup()

    def test_skipped_recall_is_recorded_and_injects_zero_tokens(self) -> None:
        self.assertEqual(self.provider.prefetch("Hello!", session_id="adaptive-session"), "")
        row = self.provider._store._conn.execute("SELECT * FROM recall_runs ORDER BY created_at DESC LIMIT 1").fetchone()
        self.assertEqual(row["mode"], "none")
        self.assertEqual(row["estimated_tokens"], 0)
        self.assertEqual(row["abstained"], 1)
        trace = self.provider._store.memory_traces(task_id=row["task_id"])[0]
        self.assertFalse(trace["retrieval_used"])
        self.assertEqual(trace["recall_mode"], "none")
        self.assertEqual(trace["candidate_memories"], [])
        self.assertEqual(trace["queries"], [])
        self.assertIn("social turn", trace["retrieval_reason"])

    def test_repeated_workflow_is_learned_only_after_distinct_tasks(self) -> None:
        for index, query in enumerate(
            ("Research current Hermes memory systems online.", "Research psychological memory studies online."), 1
        ):
            messages = [
                {"role": "user", "content": query},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": f"search-{index}",
                            "function": {"name": "web_search", "arguments": '{"query":"safe"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": f"search-{index}", "content": '{"ok":true}'},
            ]
            self.provider.sync_turn(query, "The research is complete.", session_id="adaptive-session", messages=messages)
        context = self.provider.prefetch("Research a current agent-memory paper", session_id="adaptive-session")
        self.assertIn("Reinforced workflows", context)
        self.assertIn("web_search(query)", context)
        self.assertEqual(self.provider._store.stats()["tool_workflows"], 2)


if __name__ == "__main__":
    unittest.main()
