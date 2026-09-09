from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

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

    def test_finish_rejects_withheld_memory_ids(self) -> None:
        """finish() must not credit memories the budget withheld from context.

        Regression: finish() attributed over every retrieved memory, so a
        caller crediting batch.memories could reward a memory the model never
        saw. Withheld IDs are rejected and the batch stays retryable.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with CortexMemory(Path(tmp) / "cortex.db") as memory:
                ids = []
                for content in (
                    "The staging deploy key rotates every Sunday at midnight UTC, "
                    "and the rotation checklist lives with the release captain.",
                    "Production deploys require a verified backup plus a health "
                    "check against staging before any traffic is shifted over.",
                    "Gateway rotation happens after midnight UTC once the deploy "
                    "train has fully cleared the staging environment and checks.",
                ):
                    memory_id, _ = memory.remember(content, kind="operational")
                    ids.append(memory_id)
                batch = memory.recall(
                    "What are the deploy key rotation, production deploy check, "
                    "and gateway rotation procedures?",
                    token_budget=100,
                )
                self.assertGreaterEqual(len(batch.memories), 2)
                batch.context()
                self.assertTrue(batch.dropped_memory_ids)
                withheld = batch.dropped_memory_ids[0]
                rendered = batch.rendered_memory_ids
                self.assertTrue(rendered)
                self.assertNotIn(withheld, rendered)
                with self.assertRaises(ValueError):
                    batch.finish([withheld], outcome="helpful")
                for memory_id in ids:
                    self.assertEqual(
                        memory.store.get_memory(memory_id)["helpful_count"], 0
                    )
                affected = batch.finish([rendered[0]], outcome="helpful")
                self.assertEqual(affected, [rendered[0]])

    def test_finish_marks_withheld_distinct_from_ignored(self) -> None:
        """Budget-withheld memories must not train as shown-but-ignored.

        Regression: withheld rows resolved as "ignored", which feeds the
        context-feedback usefulness penalty for evidence the model never saw.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with CortexMemory(Path(tmp) / "cortex.db") as memory:
                for content in (
                    "The staging deploy key rotates every Sunday at midnight UTC, "
                    "and the rotation checklist lives with the release captain.",
                    "Production deploys require a verified backup plus a health "
                    "check against staging before any traffic is shifted over.",
                    "Gateway rotation happens after midnight UTC once the deploy "
                    "train has fully cleared the staging environment and checks.",
                ):
                    memory.remember(content, kind="operational")
                batch = memory.recall(
                    "What are the deploy key rotation, production deploy check, "
                    "and gateway rotation procedures?",
                    token_budget=100,
                )
                self.assertGreaterEqual(len(batch.memories), 2)
                batch.context()
                self.assertTrue(batch.dropped_memory_ids)
                self.assertTrue(batch.rendered_memory_ids)
                shown_unused = [
                    memory_id
                    for memory_id in batch.rendered_memory_ids[1:]
                ]
                batch.finish([batch.rendered_memory_ids[0]], outcome="helpful")
                rows = {
                    str(row["memory_id"]): str(row["outcome"])
                    for row in memory.store._conn.execute(
                        "SELECT memory_id, outcome FROM usage_records WHERE task_id=?",
                        (batch.task_id,),
                    ).fetchall()
                }
                self.assertEqual(rows[batch.rendered_memory_ids[0]], "helpful")
                for memory_id in batch.dropped_memory_ids:
                    self.assertEqual(rows[memory_id], "withheld")
                for memory_id in shown_unused:
                    self.assertEqual(rows[memory_id], "ignored")

    def test_structured_evidence_explicit_without_context(self) -> None:
        """Callers consuming batch.memories directly keep an explicit path."""
        with tempfile.TemporaryDirectory() as tmp:
            with CortexMemory(Path(tmp) / "cortex.db") as memory:
                memory_id, _ = memory.remember(
                    "The staging deploy key rotates every Sunday.", kind="operational"
                )
                batch = memory.recall("When does the staging deploy key rotate?")
                affected = batch.finish(
                    [memory_id], outcome="helpful", evidence="structured"
                )
                self.assertEqual(affected, [memory_id])
                self.assertEqual(memory.store.get_memory(memory_id)["helpful_count"], 1)

    def _use_event_count(self, memory: CortexMemory, task_id: str) -> int:
        return int(
            memory.store._conn.execute(
                "SELECT COUNT(*) FROM memory_experience_events"
                " WHERE task_id=? AND event_type='memory_used'",
                (task_id,),
            ).fetchone()[0]
        )

    def test_failed_finish_retries_without_duplicate_use_events(self) -> None:
        """A mid-flight finish failure stays retryable with single attribution.

        Failure injection at the client seam: the store raises once, the
        retry succeeds, and exactly one memory_used event exists — the failed
        attempt must not leave a partial reward behind.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with CortexMemory(Path(tmp) / "cortex.db") as memory:
                memory_id, _ = memory.remember(
                    "The staging deploy key rotates every Sunday.", kind="operational"
                )
                batch = memory.recall("When does the staging deploy key rotate?")
                real = memory.store.resolve_usage_and_apply_outcome

                def _boom_once(task_id: str, attribution: object, outcome: str, **kwargs: object) -> object:
                    memory.store.resolve_usage_and_apply_outcome = real  # type: ignore[method-assign]
                    raise RuntimeError("simulated mid-flight failure")

                memory.store.resolve_usage_and_apply_outcome = _boom_once  # type: ignore[method-assign]
                try:
                    with self.assertRaises(RuntimeError):
                        batch.finish([memory_id], outcome="helpful")
                finally:
                    memory.store.resolve_usage_and_apply_outcome = real  # type: ignore[method-assign]
                self.assertEqual(self._use_event_count(memory, batch.task_id), 0)
                affected = batch.finish([memory_id], outcome="helpful")
                self.assertEqual(affected, [memory_id])
                self.assertEqual(memory.store.get_memory(memory_id)["helpful_count"], 1)
                self.assertEqual(self._use_event_count(memory, batch.task_id), 1)

    def test_concurrent_finish_resolves_exactly_once(self) -> None:
        """Two threads racing finish() produce one winner, one RuntimeError."""
        import threading as _threading

        with tempfile.TemporaryDirectory() as tmp:
            with CortexMemory(Path(tmp) / "cortex.db") as memory:
                memory_id, _ = memory.remember(
                    "The racing deploy key rotates every Sunday.", kind="operational"
                )
                for iteration in range(10):
                    batch = memory.recall("When does the racing deploy key rotate?")
                    self.assertEqual(batch.memories[0]["id"], memory_id)
                    barrier = _threading.Barrier(2)
                    outcomes: list[object] = [None, None]

                    def _racer(slot: int) -> None:
                        barrier.wait()
                        try:
                            outcomes[slot] = batch.finish([memory_id], outcome="helpful")
                        except RuntimeError as exc:
                            outcomes[slot] = exc
                        except Exception as exc:  # noqa: BLE001 — surface, never swallow
                            outcomes[slot] = exc

                    threads = [
                        _threading.Thread(target=_racer, args=(slot,)) for slot in (0, 1)
                    ]
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join()
                    winners = [result for result in outcomes if result == [memory_id]]
                    losers = [result for result in outcomes if isinstance(result, RuntimeError)]
                    self.assertEqual(len(winners), 1, f"iteration {iteration}: {outcomes}")
                    self.assertEqual(len(losers), 1, f"iteration {iteration}: {outcomes}")
                    self.assertEqual(self._use_event_count(memory, batch.task_id), 1)
                self.assertEqual(memory.store.get_memory(memory_id)["helpful_count"], 10)

    def test_rendered_memory_ids_partition_batch(self) -> None:
        """Rendered + withheld IDs partition every retrieved memory."""
        memories = [
            {
                "id": f"memory-part-{index:04d}-abcdef",
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
            task_id="task-partition",
            query="What does production deploy require?",
            memories=memories,
            _store=object(),
            token_budget=120,
        )
        batch.context()
        rendered = batch.rendered_memory_ids
        dropped = batch.dropped_memory_ids
        self.assertTrue(rendered)
        self.assertTrue(dropped)
        self.assertEqual(set(rendered) | set(dropped), {str(item["id"]) for item in memories})
        self.assertFalse(set(rendered) & set(dropped))

    def _budget_memories(self) -> list[dict[str, Any]]:
        return [
            {
                "id": f"memory-small-{index:04d}-abcdef",
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

    def test_zero_budget_returns_empty_context_with_metadata(self) -> None:
        """A zero budget carries no envelope: empty text, full metadata."""
        batch = RecallBatch(
            task_id="task-zero",
            query="What does production deploy require?",
            memories=self._budget_memories(),
            _store=object(),
            token_budget=0,
        )
        self.assertEqual(batch.context(), "")
        self.assertEqual(batch.context_tokens(), 0)
        self.assertEqual(batch.rendered_memory_ids, [])
        self.assertEqual(len(batch.dropped_memory_ids), 4)

    def test_tiny_budgets_return_empty_context(self) -> None:
        """Budgets below the header+note envelope emit nothing (0, 1, 13, 20)."""
        for budget in (1, 13, 20):
            with self.subTest(token_budget=budget):
                batch = RecallBatch(
                    task_id=f"task-tiny-{budget}",
                    query="What does production deploy require?",
                    memories=self._budget_memories(),
                    _store=object(),
                    token_budget=budget,
                )
                text = batch.context()
                self.assertEqual(text, "")
                self.assertEqual(batch.context_tokens(), 0)
                self.assertEqual(batch.rendered_memory_ids, [])
                self.assertEqual(len(batch.dropped_memory_ids), 4)
                self.assertLessEqual(estimate_text_tokens(text or " ") - 1, budget)

    def test_exact_boundary_budget(self) -> None:
        """A budget fitting the full render keeps everything; one less cuts."""
        batch = RecallBatch(
            task_id="task-boundary",
            query="What does production deploy require?",
            memories=self._budget_memories(),
            _store=object(),
            token_budget=4000,
        )
        full_text = batch.context()
        self.assertEqual(batch.dropped_memory_ids, [])
        full_budget = estimate_text_tokens(full_text)
        batch.token_budget = full_budget
        self.assertEqual(batch.context(), full_text)
        self.assertEqual(batch.dropped_memory_ids, [])
        batch.token_budget = full_budget - 1
        cut_text = batch.context()
        self.assertTrue(batch.dropped_memory_ids)
        self.assertIn("withheld", cut_text)
        self.assertLessEqual(estimate_text_tokens(cut_text), full_budget - 1)

    def test_negative_budget_rejected(self) -> None:
        """Negative budgets fail fast instead of silently clamping to zero."""
        with self.assertRaises(ValueError):
            RecallBatch(
                task_id="task-negative",
                query="What does production deploy require?",
                memories=self._budget_memories(),
                _store=object(),
                token_budget=-1,
            )
        with tempfile.TemporaryDirectory() as tmp:
            with CortexMemory(Path(tmp) / "cortex.db") as memory:
                memory.remember("The staging deploy key rotates every Sunday.")
                with self.assertRaises(ValueError):
                    memory.recall("When does the key rotate?", token_budget=-5)

    def test_estimate_text_tokens_is_len_over_four_heuristic(self) -> None:
        """The len/4 rule is a documented heuristic, verified behaviorally."""
        self.assertEqual(estimate_text_tokens("abcd"), 1)
        self.assertEqual(estimate_text_tokens("abcde"), 2)
        self.assertEqual(estimate_text_tokens(""), 1)

    def _recall_three(self, memory: CortexMemory) -> Any:
        for content in (
            "The staging deploy key rotates every Sunday at midnight UTC, "
            "and the rotation checklist lives with the release captain.",
            "Production deploys require a verified backup plus a health "
            "check against staging before any traffic is shifted over.",
            "Gateway rotation happens after midnight UTC once the deploy "
            "train has fully cleared the staging environment and checks.",
        ):
            memory.remember(content, kind="operational")
        return memory.recall(
            "What are the deploy key rotation, production deploy check, "
            "and gateway rotation procedures?",
            token_budget=100,
        )

    def _render_events(self, memory: CortexMemory, task_id: str) -> list[dict[str, Any]]:
        import json as _json

        return [
            _json.loads(line)
            for line in memory.store.memory_trace_jsonl(task_id=task_id).splitlines()
            if _json.loads(line)["event_type"] == "render_decision"
        ]

    def test_context_reports_render_metrics_once(self) -> None:
        """The first render records counts/tokens; repeats add no signals."""
        with tempfile.TemporaryDirectory() as tmp:
            with CortexMemory(Path(tmp) / "cortex.db") as memory:
                batch = self._recall_three(memory)
                self.assertGreaterEqual(len(batch.memories), 2)
                first = batch.context()
                second = batch.context()
                self.assertEqual(first, second)
                row = memory.store._conn.execute(
                    "SELECT selected_count, rendered_count, withheld_count, rendered_tokens"
                    " FROM recall_runs WHERE task_id=?",
                    (batch.task_id,),
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row["rendered_count"] + row["withheld_count"], row["selected_count"])
                self.assertEqual(row["rendered_count"], len(batch.rendered_memory_ids))
                self.assertEqual(row["withheld_count"], len(batch.dropped_memory_ids))
                self.assertEqual(row["rendered_tokens"], batch.context_tokens())
                self.assertEqual(len(self._render_events(memory, batch.task_id)), 1)

    def test_render_decision_explains_budget_cut(self) -> None:
        """A cut trace names rendered/withheld IDs and the budget reason."""
        with tempfile.TemporaryDirectory() as tmp:
            with CortexMemory(Path(tmp) / "cortex.db") as memory:
                batch = self._recall_three(memory)
                batch.token_budget = 0
                self.assertEqual(batch.context(), "")
                events = self._render_events(memory, batch.task_id)
                self.assertEqual(len(events), 1)
                payload = events[0]["payload"]
                self.assertEqual(payload["rendered_count"], 0)
                self.assertGreater(payload["withheld_count"], 0)
                self.assertEqual(set(payload["withheld_memory_ids"]), set(batch.dropped_memory_ids))
                self.assertIn("budget", payload["reason"])

    def test_full_render_emits_no_cut_event(self) -> None:
        """Nothing withheld means nothing new to explain in the trace."""
        with tempfile.TemporaryDirectory() as tmp:
            with CortexMemory(Path(tmp) / "cortex.db") as memory:
                memory.remember("The staging deploy key rotates every Sunday.")
                batch = memory.recall("When does the staging deploy key rotate?")
                batch.context()
                self.assertEqual(batch.dropped_memory_ids, [])
                self.assertEqual(self._render_events(memory, batch.task_id), [])
                row = memory.store._conn.execute(
                    "SELECT rendered_count, withheld_count FROM recall_runs WHERE task_id=?",
                    (batch.task_id,),
                ).fetchone()
                self.assertEqual(row["withheld_count"], 0)
                self.assertGreaterEqual(row["rendered_count"], 1)

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
