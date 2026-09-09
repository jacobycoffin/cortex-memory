from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT

from cortex.client import CortexMemory, RecallBatch, estimate_text_tokens
from cortex.retrieval import MemoryRetriever, RetrievalResult


class CortexClientTests(unittest.TestCase):
    def test_recall_context_carries_origin_and_review_without_claiming_truth(self) -> None:
        result = RetrievalResult(
            memory={
                "id": "memory-provenance-123",
                "kind": "semantic",
                "content": "The release name is Juniper.",
                "state": "active",
                "source_type": "conversation",
                "source_category": "OPERATOR_APPROVED",
                "origin_source_category": "USER_STATED",
                "source_ref": "session-42",
                "approval_state": "operator_approved",
            },
            score=0.74,
            components={"lexical": 0.8},
            estimated_tokens=24,
        )

        memory = result.as_dict()
        batch = RecallBatch(
            task_id="task-provenance",
            query="What is the release name?",
            memories=[memory],
            _store=object(),  # context rendering does not touch storage
        )
        context = batch.context()

        self.assertEqual(memory["source_type"], "conversation")
        self.assertEqual(memory["source_category"], "OPERATOR_APPROVED")
        self.assertEqual(memory["origin_source_category"], "USER_STATED")
        self.assertEqual(memory["source_ref"], "session-42")
        self.assertEqual(memory["approval_state"], "operator_approved")
        self.assertIn("source: user-stated", context)
        self.assertIn("ref: session-42", context)
        self.assertIn("review: approved, not independently verified", context)
        self.assertNotIn("source: operator-approved", context)

    def test_repeated_retrieval_and_injection_do_not_raise_activation(self) -> None:
        baseline = {
            "kind": "semantic",
            "volatility": 0.4,
            "updated_at": "2026-01-01T00:00:00+00:00",
            "last_used_at": None,
            "last_helpful_at": None,
            "last_injected_at": None,
            "retrieved_count": 0,
            "injected_count": 0,
            "used_count": 0,
            "success_count": 0,
            "confirmed_count": 0,
            "helpful_count": 0,
            "validated_count": 0,
        }
        repeatedly_seen = {
            **baseline,
            "retrieved_count": 10_000,
            "injected_count": 10_000,
            "last_injected_at": "2026-07-16T12:00:00+00:00",
            # Some legacy irrelevant-access paths updated this timestamp even
            # though the memory was never actually used.
            "last_used_at": "2026-07-16T12:00:00+00:00",
        }

        self.assertAlmostEqual(
            MemoryRetriever._activation(baseline),
            MemoryRetriever._activation(repeatedly_seen),
            places=7,
        )
        previously_used = {**baseline, "used_count": 1, "last_used_at": baseline["updated_at"]}
        later_ignored = {
            **previously_used,
            "last_used_at": "2026-07-16T12:00:00+00:00",
            "retrieved_count": 1,
            "injected_count": 1,
        }
        self.assertAlmostEqual(
            MemoryRetriever._activation(previously_used),
            MemoryRetriever._activation(later_ignored),
            places=7,
        )

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

    def test_recall_batch_invalid_outcome_writes_nothing_and_stays_retryable(self) -> None:
        """A failed finish() must not partially attribute learning data.

        Regression test: finish() used to commit usage attribution before
        validating the outcome, so finish([id], outcome="bogus") raised AFTER
        marking the memory used — and a retry with no used IDs still rewarded
        that memory via apply_task_outcome.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with CortexMemory(Path(tmp) / "cortex.db") as memory:
                memory_id, _ = memory.remember(
                    "The staging deploy key rotates every Sunday.", kind="operational"
                )
                batch = memory.recall("When does the staging deploy key rotate?")
                self.assertEqual(batch.memories[0]["id"], memory_id)
                with self.assertRaises(ValueError):
                    batch.finish([memory_id], outcome="bogus-outcome")
                rows = memory.store._conn.execute(
                    "SELECT outcome FROM usage_records WHERE task_id=?", (batch.task_id,)
                ).fetchall()
                self.assertTrue(rows)
                self.assertTrue(all(row["outcome"] == "pending" for row in rows))
                affected = batch.finish([memory_id], outcome="helpful")
                self.assertEqual(affected, [memory_id])
                self.assertEqual(memory.store.get_memory(memory_id)["helpful_count"], 1)

    def test_recall_batch_context_enforces_rendered_token_budget(self) -> None:
        """The budget bounds the FINAL rendered block, not just raw content.

        Regression test: retrieval estimated content length plus a fixed
        overhead, but context() rendered variable provenance labels on top,
        so the configured budget never actually bounded injected context.
        """
        memories = [
            {
                "id": f"memory-budget-{index:04d}-abcdef",
                "kind": "semantic",
                "score": 0.9 - index * 0.05,
                "content": (
                    "The production deploy procedure requires a health check, "
                    f"a verified backup, and gateway rotation step {index}."
                ),
                "source_type": "conversation",
                "source_category": "USER_EXPLICIT",
                "source_ref": f"session-42-turn-{index}-with-a-long-reference-tail",
                "approval_state": "operator_approved",
            }
            for index in range(4)
        ]
        batch = RecallBatch(
            task_id="task-budget",
            query="What does production deploy require?",
            memories=memories,
            _store=object(),
            token_budget=120,
        )
        text = batch.context()
        self.assertLessEqual(estimate_text_tokens(text), 120)
        self.assertEqual(batch.context_tokens(), estimate_text_tokens(text))
        dropped = batch.dropped_memory_ids
        self.assertTrue(dropped)
        self.assertIn("withheld", text)
        # Best-score-first: at this budget the top memory survives the cut.
        self.assertNotIn(str(memories[0]["id"]), dropped)
        self.assertIn(str(memories[0]["id"])[:8], text)
        # A repeat render is stable and the default budget fits everything.
        self.assertEqual(batch.context(), text)
        full = RecallBatch(
            task_id="task-budget-full",
            query="What does production deploy require?",
            memories=memories,
            _store=object(),
        )
        full_text = full.context()
        self.assertEqual(full.dropped_memory_ids, [])
        self.assertNotIn("withheld", full_text)

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
