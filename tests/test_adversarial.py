from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


from tests._bootstrap import ROOT

from cortex import CortexMemoryProvider
from cortex.retrieval import MemoryRetriever
from cortex.store import CortexStore


class AdversarialStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_different_time_periods_do_not_contradict(self) -> None:
        old_id, _ = self.store.add_memory(
            "The service used port 3000 during 2025.",
            subject="service:api",
            predicate="port",
            object_value="3000",
            valid_from="2025-01-01T00:00:00+00:00",
            valid_to="2025-12-31T23:59:59+00:00",
        )
        current_id, _ = self.store.add_memory(
            "The service uses port 3001 during 2026.",
            subject="service:api",
            predicate="port",
            object_value="3001",
            valid_from="2026-01-01T00:00:00+00:00",
        )
        self.assertFalse(any(edge["relation"] == "contradicts" for edge in self.store.explain(current_id)["edges"]))
        conflict_id, _ = self.store.add_memory(
            "The service uses port 4000 during July 2026.",
            subject="service:api",
            predicate="port",
            object_value="4000",
            valid_from="2026-07-01T00:00:00+00:00",
            valid_to="2026-07-31T23:59:59+00:00",
        )
        conflicts = [edge for edge in self.store.explain(conflict_id)["edges"] if edge["relation"] == "contradicts"]
        self.assertEqual(len(conflicts), 1)
        self.assertIn(current_id, {conflicts[0]["src_id"], conflicts[0]["dst_id"]})
        self.assertNotIn(old_id, {conflicts[0]["src_id"], conflicts[0]["dst_id"]})

    def test_dependency_invalidation_is_staged_and_reversible(self) -> None:
        evidence_id, _ = self.store.add_memory(
            "DNS failed at 14:03.", kind="episode", source_category="TOOL_VERIFIED", confidence=0.95
        )
        belief_id, _ = self.store.add_memory(
            "Local DNS failure can prevent Hermes provider access.",
            kind="semantic",
            source_category="REFLECTION",
            confidence=0.8,
            evidence_ids=[evidence_id],
        )
        self.store.set_state(evidence_id, "archived", reason="evidence retracted")
        self.assertEqual(self.store.get_memory(belief_id)["dirty"], 1)
        shadow = self.store.repair_dependencies(dry_run=True)
        self.assertEqual(shadow["proposals"][0]["action"], "quarantine")
        self.assertEqual(self.store.get_memory(belief_id)["state"], "active")
        self.store.set_state(evidence_id, "active", reason="evidence restored")
        applied = self.store.repair_dependencies(dry_run=False)
        self.assertEqual(applied["proposals"][0]["action"], "recalculate")
        self.assertEqual(self.store.get_memory(belief_id)["state"], "active")
        self.assertEqual(self.store.get_memory(belief_id)["dirty"], 0)

    def test_protected_prospective_memory_is_not_pruned(self) -> None:
        memory_id, _ = self.store.add_memory(
            "Follow up on the Cortex retrieval evaluation.",
            kind="prospective",
            protected=True,
            importance=0.4,
        )
        old = (datetime.now(timezone.utc) - timedelta(days=500)).isoformat()
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET updated_at=? WHERE id=?", (old, memory_id))
        report = self.store.maintenance(dry_run=True, cold_after_days=30, archive_after_days=60)
        self.assertNotIn(memory_id, report["memory_ids"]["cold"])
        self.assertNotIn(memory_id, report["memory_ids"]["archived"])

    def test_harmful_reinforcement_overcomes_raw_retrieval_frequency(self) -> None:
        bad_id, _ = self.store.add_memory("The edge Hermes service always uses port 9999.", confidence=0.75, importance=0.6)
        good_id, _ = self.store.add_memory("The edge Hermes service uses port 8642.", confidence=0.9, importance=0.6)
        for _ in range(20):
            self.store.log_access(bad_id, "retrieved", query="edge Hermes port")
        for _ in range(4):
            self.store.log_access(bad_id, "wrong", query="edge Hermes port")
        self.store.log_access(good_id, "validated", query="edge Hermes port")
        ranked = MemoryRetriever(self.store, threshold=0.0).search("edge Hermes port", limit=2)
        self.assertEqual(ranked[0].memory["id"], good_id)

    def test_restore_returns_archived_memory_to_retrieval(self) -> None:
        memory_id, _ = self.store.add_memory("Archived recovery procedure for the edge server.", kind="procedure")
        self.store.set_state(memory_id, "archived", reason="test")
        self.assertFalse(MemoryRetriever(self.store).search("edge server recovery procedure"))
        self.store.set_state(memory_id, "active", reason="restored")
        self.assertTrue(MemoryRetriever(self.store).search("edge server recovery procedure"))


class ToolLearningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.provider = CortexMemoryProvider({"db_path": "$HERMES_HOME/cortex/test.db", "retrieval_threshold": 0.05})
        self.provider.initialize("tools-session", hermes_home=self.tmp.name, agent_context="primary")

    def tearDown(self) -> None:
        self.provider.shutdown()
        self.tmp.cleanup()

    def _tool_turn(self, call_id: str, query: str, result: str) -> None:
        messages = [
            {"role": "user", "content": query},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "web_search", "arguments": '{"query":"redacted"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call_id, "content": result},
            {"role": "assistant", "content": "I found the requested sources."},
        ]
        self.provider.sync_turn(query, "I found the requested sources.", session_id="tools-session", messages=messages)

    def _named_tool_turn(self, call_id: str, query: str, tool_name: str, result: str) -> None:
        messages = [
            {"role": "user", "content": query},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": '{"command":"redacted"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call_id, "content": result},
            {"role": "assistant", "content": "The tool attempt completed."},
        ]
        self.provider.sync_turn(query, "The tool attempt completed.", session_id="tools-session", messages=messages)

    def test_repeated_tool_success_becomes_procedural_guidance(self) -> None:
        self._tool_turn(
            "call-1", "Research current Hermes memory systems online.", '{"success":true,"results":["source-a"]}'
        )
        first_context = self.provider.prefetch("Research current agent memory papers", session_id="tools-session")
        self.assertNotIn("Tool-outcome guidance", first_context)
        self._tool_turn(
            "call-2", "Research psychological memory studies online.", '{"success":true,"results":["source-b"]}'
        )
        context = self.provider.prefetch("Research current agent memory papers", session_id="tools-session")
        self.assertIn("Tool-outcome guidance", context)
        self.assertIn("web_search", context)
        self.assertIn("keys=query", context)
        stats = json.loads(self.provider.handle_tool_call("cortex_memory", {"action": "stats"}))["stats"]
        self.assertEqual(stats["tool_executions"], 2)
        search = json.loads(
            self.provider.handle_tool_call("cortex_memory", {"action": "search", "query": "web search successful tool"})
        )
        self.assertTrue(any(row["kind"] == "procedure" for row in search["results"]))

    def test_retrieved_but_unused_memory_gets_no_positive_reinforcement(self) -> None:
        saved = json.loads(
            self.provider.handle_tool_call(
                "cortex_memory",
                {
                    "action": "remember",
                    "content": "Use direct MCP for operations dashboard actions.",
                    "kind": "decision",
                },
            )
        )
        self.provider.prefetch("How should operations dashboard actions work?", session_id="tools-session")
        self.provider.sync_turn(
            "How should operations dashboard actions work?",
            "I cannot determine that from the available context.",
            session_id="tools-session",
        )
        explanation = json.loads(
            self.provider.handle_tool_call("cortex_memory", {"action": "explain", "memory_id": saved["memory_id"]})
        )["explanation"]
        memory = explanation["memory"]
        self.assertGreaterEqual(memory["selected_count"], 1)
        self.assertEqual(memory["used_count"], 0)
        self.assertEqual(memory["helpful_count"], 0)
        self.assertTrue(
            any(
                record["outcome"] == "ignored"
                for record in self.provider._store._conn.execute(
                    "SELECT outcome FROM usage_records WHERE memory_id=?", (saved["memory_id"],)
                ).fetchall()
            )
        )

    def test_repeated_tool_failure_becomes_scoped_warning(self) -> None:
        failure = '{"success":false,"error":"permission denied"}'
        self._named_tool_turn("fail-1", "Run the project build command.", "shell_exec", failure)
        self._named_tool_turn("fail-2", "Run the project test command.", "shell_exec", failure)
        context = self.provider.prefetch("Run the project build command", session_id="tools-session")
        self.assertIn("tool=shell_exec", context)
        self.assertIn("failures=2", context)
        search = json.loads(
            self.provider.handle_tool_call(
                "cortex_memory", {"action": "search", "query": "shell tool repeatedly failed permission"}
            )
        )
        self.assertTrue(any("repeatedly failed" in row["content"] for row in search["results"]))


if __name__ == "__main__":
    unittest.main()
