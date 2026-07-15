from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT

from cortex.client import CortexMemory


class CortexClientTests(unittest.TestCase):
    def test_agent_neutral_remember_recall_feedback_and_sleep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with CortexMemory(Path(tmp) / "cortex.db") as memory:
                memory_id, created = memory.remember(
                    "Production deployment requires a health check and a verified backup.",
                    kind="procedure",
                    source_category="USER_EXPLICIT",
                    session_id="adapter-a",
                )
                self.assertTrue(created)
                batch = memory.recall(
                    "What is required before production deployment?",
                    session_id="adapter-b",
                    task_type="deployment",
                )
                self.assertEqual(batch.memories[0]["id"], memory_id)
                self.assertIn("metacognition", batch.memories[0])
                self.assertIn(batch.memories[0]["metacognition"]["decision"], {"use", "verify", "abstain"})
                self.assertIn("fallible evidence", batch.context())
                affected = batch.finish([memory_id], outcome="helpful")
                self.assertEqual(affected, [memory_id])
                self.assertEqual(memory.store.get_memory(memory_id)["helpful_count"], 1)
                trace = memory.store.memory_traces(task_id=batch.task_id)[0]
                self.assertEqual(trace["goal"], "What is required before production deployment?")
                self.assertTrue(trace["retrieval_used"])
                self.assertEqual(trace["selected_memory_ids"], [memory_id])
                self.assertTrue(trace["influence"][0]["influenced"])
                self.assertEqual(trace["evaluations"][0]["rating"], "Helpful")
                self.assertTrue(trace["evaluations"][0]["improved_outcome"])
                self.assertEqual(trace["memory_actions"][0]["action"], "ignored")
                events = [
                    json.loads(line)
                    for line in memory.store.memory_trace_jsonl(task_id=batch.task_id).splitlines()
                ]
                self.assertEqual(
                    [event["event_type"] for event in events],
                    ["retrieval_decision", "task_evaluation", "outcome_feedback"],
                )
                self.assertNotIn("reasoning", memory.store.memory_trace_jsonl(task_id=batch.task_id))

                report = memory.sleep()
                self.assertEqual(report["mode"], "shadow")
                self.assertEqual(report["reflection_token_budget"], 0)
                self.assertTrue(memory.audit()["ok"])

    def test_recall_batch_cannot_credit_unknown_or_resolve_twice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with CortexMemory(Path(tmp) / "cortex.db") as memory:
                memory.remember("The service listens on port 8123.", kind="operational")
                batch = memory.recall("Which port does the service use?")
                with self.assertRaises(ValueError):
                    batch.finish(["not-from-this-batch"], outcome="helpful")
                batch.finish()
                trace = memory.store.memory_traces(task_id=batch.task_id)[0]
                self.assertEqual(trace["outcome"], "completed_unlabeled")
                self.assertTrue(all(item["rating"] == "Irrelevant" for item in trace["evaluations"]))
                with self.assertRaises(RuntimeError):
                    batch.finish()

    def test_agent_neutral_api_passes_explicit_project_and_system_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with CortexMemory(Path(tmp) / "cortex.db") as memory:
                memory_id, _ = memory.remember(
                    "The private release route is cobalt.",
                    context_mode="context_dependent",
                    scope={"project": "Cortex"},
                    preconditions={"environment": "production"},
                    entities=["Cortex"],
                    source_context="Production release verification.",
                )
                missing = memory.recall("Which private release route is cobalt?")
                self.assertEqual(missing.memories, [])
                missing.finish()
                matched = memory.recall(
                    "Which private release route is cobalt?",
                    active_project="Cortex",
                    system_state={"environment": "production"},
                )
                self.assertEqual(matched.memories[0]["id"], memory_id)
                self.assertIn("project_match", matched.memories[0]["components"])
                matched.finish([memory_id])


if __name__ == "__main__":
    unittest.main()
