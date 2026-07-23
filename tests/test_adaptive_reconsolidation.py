from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401

from cortex.adaptive_reconsolidation import run_adaptive_reconsolidation
from cortex.autojudge import AutoJudgeConfig
from cortex.store import CortexStore


class AdaptiveReconsolidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    @staticmethod
    def config() -> AutoJudgeConfig:
        return AutoJudgeConfig(
            enabled=True,
            endpoint="http://127.0.0.1:9999/v1/chat/completions",
            model="synthetic-reconsolidation-judge",
            api_key_env="",
            credential_file=None,
            timeout_seconds=5.0,
            max_output_tokens=1200,
        )

    def task_pair(
        self,
        *,
        old_content: str = "The service listens on port 3000.",
        new_content: str = "The service now listens on port 3001.",
        old_kind: str = "operational",
    ) -> tuple[str, str, str]:
        old_id, _ = self.store.add_memory(old_content, kind=old_kind)
        new_id, _ = self.store.add_memory(new_content, kind="operational")
        task_id = self.store.create_usage_batch(
            [(old_id, 0.8)],
            query="Which port does the service use?",
            session_id="recon-session",
            task_type="debugging",
            recall_mode="focused",
        )
        self.store.record_memory_trace_decision(
            task_id=task_id,
            session_id="recon-session",
            goal="Confirm the current service port",
            context_summary="task_type=debugging",
            task_type="debugging",
            recall_mode="focused",
            retrieval_used=True,
            retrieval_reason="old memory influenced the answer",
            queries=["Which port does the service use?"],
            candidate_memories=[
                {
                    "memory_id": old_id,
                    "kind": old_kind,
                    "selected": True,
                    "score": 0.8,
                    "components": {},
                    "reason": "selected",
                }
            ],
        )
        self.store.resolve_usage(
            task_id,
            {old_id: 1.0},
            memory_actions=[
                {
                    "action": "created",
                    "memory_id": new_id,
                    "reason": "new same-task durable evidence",
                }
            ],
        )
        return task_id, old_id, new_id

    def stage(
        self,
        *,
        action: str,
        replacement_content: str | None = None,
        old_kind: str = "operational",
    ) -> tuple[dict, str, str]:
        task_id, old_id, new_id = self.task_pair(old_kind=old_kind)

        def provider(_endpoint, _key, payload, _timeout):
            candidate = json.loads(payload["messages"][1]["content"])["candidates"][0]
            proposal = {
                "candidate_id": candidate["candidate_id"],
                "action": action,
                "confidence": 0.92,
                "reason": "Same-task evidence establishes the relationship.",
            }
            if replacement_content is not None:
                proposal["replacement_content"] = replacement_content
            return {
                "choices": [
                    {"message": {"content": json.dumps({"proposals": [proposal]})}}
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            }

        report = run_adaptive_reconsolidation(
            self.store,
            self.config(),
            provider_call=provider,
            task_id=task_id,
        )
        proposal = self.store.adaptive_reconsolidation_snapshot()["proposals"][0]
        return {"report": report, "proposal": proposal}, old_id, new_id

    def test_same_task_use_is_required_and_model_result_stays_shadow(self) -> None:
        result, old_id, _new_id = self.stage(
            action="supersede",
            replacement_content="The service listens on port 3001.",
        )

        self.assertEqual(result["report"]["proposals"], 1)
        self.assertEqual(result["proposal"]["status"], "proposed")
        self.assertEqual(
            self.store.get_memory(old_id)["content"],
            "The service listens on port 3000.",
        )

        unrelated, _ = self.store.add_memory("Unrelated evidence.")
        task_id = self.store.create_usage_batch(
            [(old_id, 0.8)],
            query="unused",
            session_id="unused-session",
            task_type="debugging",
            recall_mode="focused",
        )
        self.store.record_memory_trace_decision(
            task_id=task_id,
            session_id="unused-session",
            goal="unused",
            context_summary="unused",
            task_type="debugging",
            recall_mode="focused",
            retrieval_used=True,
            retrieval_reason="selected but unused",
            queries=["unused"],
            candidate_memories=[
                {
                    "memory_id": old_id,
                    "kind": "operational",
                    "selected": True,
                    "score": 0.8,
                    "components": {},
                    "reason": "selected",
                }
            ],
        )
        self.store.resolve_usage(
            task_id,
            {},
            memory_actions=[{"action": "created", "memory_id": unrelated, "reason": "new"}],
        )
        self.assertEqual(
            self.store.adaptive_reconsolidation_candidates(task_id=task_id),
            [],
        )

    def test_supersede_and_undo_preserve_version_history(self) -> None:
        result, old_id, _new_id = self.stage(
            action="supersede",
            replacement_content="The service listens on port 3001.",
        )
        proposal_id = result["proposal"]["proposal_id"]

        applied = self.store.apply_adaptive_reconsolidation(proposal_id)
        versions_after_apply = self.store.explain(old_id)["versions"]
        undone = self.store.undo_adaptive_reconsolidation(proposal_id)
        versions_after_undo = self.store.explain(old_id)["versions"]

        self.assertEqual(applied["status"], "applied")
        self.assertTrue(self.store.audit()["ok"])
        self.assertEqual(self.store.get_memory(old_id)["content"], "The service listens on port 3000.")
        self.assertTrue(undone["reversed"])
        self.assertGreater(len(versions_after_undo), len(versions_after_apply))

    def test_protected_memory_requires_explicit_confirmation(self) -> None:
        result, _old_id, _new_id = self.stage(
            action="supersede",
            replacement_content="The user's preferred port is 3001.",
            old_kind="preference",
        )
        proposal_id = result["proposal"]["proposal_id"]

        with self.assertRaisesRegex(ValueError, "explicit confirmation"):
            self.store.apply_adaptive_reconsolidation(proposal_id)
        applied = self.store.apply_adaptive_reconsolidation(
            proposal_id,
            confirm_protected=True,
        )

        self.assertEqual(applied["status"], "applied")
        self.assertTrue(result["proposal"]["protected_confirmation_required"])

    def test_extend_edge_is_evidenced_and_wrong_feedback_undoes_it(self) -> None:
        result, old_id, new_id = self.stage(action="extend")
        proposal_id = result["proposal"]["proposal_id"]

        self.store.apply_adaptive_reconsolidation(proposal_id)
        evidence = self.store.edge_evidence(new_id, old_id, "extends")
        feedback = self.store.record_adaptive_reconsolidation_feedback(
            proposal_id,
            "wrong",
            reason="human review rejected the relationship",
        )

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["evidence_type"], "reviewed_reconsolidation")
        self.assertTrue(feedback["automatically_undone"])
        self.assertEqual(self.store.edge_evidence(new_id, old_id, "extends"), [])
        self.assertEqual(feedback["feedback"]["correctness"], 0.0)

    def test_wrong_feedback_rejects_unapplied_proposal(self) -> None:
        result, old_id, _new_id = self.stage(
            action="supersede",
            replacement_content="The service listens on port 3001.",
        )
        proposal_id = result["proposal"]["proposal_id"]

        feedback = self.store.record_adaptive_reconsolidation_feedback(
            proposal_id,
            "wrong",
            reason="review found the proposed replacement unsupported",
        )
        proposal = self.store.adaptive_reconsolidation_snapshot()["proposals"][0]

        self.assertFalse(feedback["automatically_undone"])
        self.assertEqual(proposal["status"], "rejected")
        self.assertEqual(
            self.store.get_memory(old_id)["content"],
            "The service listens on port 3000.",
        )
        with self.assertRaisesRegex(ValueError, "not proposed"):
            self.store.apply_adaptive_reconsolidation(proposal_id)


if __name__ == "__main__":
    unittest.main()
