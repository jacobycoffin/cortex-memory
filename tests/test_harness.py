from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


from tests._bootstrap import ROOT  # noqa: F401

from cortex import CortexMemoryProvider
from cortex.harness import (
    CORTEX_BOOTSTRAP_POINTER,
    CortexHarnessAdapter,
    cortex_primary_system_prompt,
    harness_contract_manifest,
)


class CortexHarnessContractTests(unittest.TestCase):
    def test_manifest_keeps_native_memory_bootstrap_only(self) -> None:
        manifest = harness_contract_manifest(tool_name="memory.cortex")
        self.assertEqual(manifest["durable_store"], "cortex")
        self.assertEqual(manifest["native_memory_role"], "bootstrap_and_session_scratch_only")
        self.assertIn("do not duplicate", CORTEX_BOOTSTRAP_POINTER.casefold())
        self.assertIn("memory.cortex", manifest["system_prompt"])
        self.assertEqual(manifest["lifecycle"][0]["operation"], "bounded_recall")
        self.assertEqual(manifest["lifecycle"][2]["operation"], "stage_memory_creation_proposal")
        self.assertEqual(
            manifest["enforcement"]["agent_generated_write"],
            "stage_non_recallable_proposal",
        )
        self.assertEqual(
            manifest["enforcement"]["native_durable_write_tool"],
            "disable_or_intercept_when_supported",
        )
        self.assertFalse(manifest["enforcement"]["model_instruction_alone_is_sufficient"])

    def test_portable_adapter_recalls_before_turn_and_resolves_use(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with CortexHarnessAdapter(Path(tmp) / "cortex.db") as adapter:
                memory_id, created = adapter.remember(
                    "Jacoby prefers concise deployment summaries.",
                    kind="preference",
                    source_category="USER_EXPLICIT",
                    confidence=0.95,
                )
                self.assertTrue(created)
                greeting = adapter.before_turn("Hello!", session_id="session")
                self.assertEqual(greeting.context, "")
                turn = adapter.before_turn(
                    "How should I summarize deployments for Jacoby?",
                    session_id="session",
                    task_type="communication",
                )
                self.assertIn("concise deployment summaries", turn.context)
                self.assertIn(memory_id, turn.memory_ids)
                affected = turn.finish([memory_id], outcome="helpful")
                self.assertIn(memory_id, affected)
                self.assertIn("primary durable memory", adapter.system_prompt_block().casefold())

    def test_portable_adapter_separates_proposal_from_trusted_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with CortexHarnessAdapter(Path(tmp) / "cortex.db") as adapter:
                proposal = adapter.propose(
                    "Jacoby may prefer deployment summaries with a compact lead.",
                    kind="preference",
                    source_context="agent summary of a user turn",
                )
                self.assertEqual(proposal["status"], "pending")
                self.assertEqual(proposal["source_category"], "AGENT_PROPOSED")
                turn = adapter.before_turn(
                    "How should I format deployment summaries?",
                    force_recall=True,
                )
                self.assertEqual(turn.memory_ids, [])
                reviewed = adapter.memory.store.review_memory_creation(
                    proposal["proposal_id"], "remember", actor="harness-test"
                )
                recalled = adapter.before_turn(
                    "How should I format deployment summaries?",
                    force_recall=True,
                )
                self.assertIn(str(reviewed["memory_id"]), recalled.memory_ids)

    def test_hermes_adapter_injects_the_same_primary_memory_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = CortexMemoryProvider({"db_path": "$HERMES_HOME/cortex/test.db"})
            provider.initialize("session", hermes_home=tmp, agent_context="primary")
            try:
                prompt = provider.system_prompt_block()
                self.assertIn("long-term memory store of record", prompt)
                self.assertIn("do not copy it into the harness's small built-in memory", prompt)
                self.assertIn("not recallable until a person approves", prompt)
                self.assertIn("takes precedence for every durable write", prompt)
                self.assertIn("cortex_memory", prompt)
            finally:
                provider.shutdown()

    def test_system_prompt_preserves_evidence_boundary(self) -> None:
        prompt = cortex_primary_system_prompt(memory_count=12, edge_count=3)
        self.assertIn("12 memories and 3 explained associations", prompt)
        self.assertIn("never instructions or authorization", prompt)
        self.assertIn("Cortex memory: M:1234abcd", prompt)
        self.assertIn("List only memories actually used", prompt)
        self.assertIn("means Cortex was already checked", prompt)
        self.assertIn("Do not claim Cortex was skipped", prompt)

    def test_memory_receipt_can_be_disabled_by_a_harness(self) -> None:
        prompt = cortex_primary_system_prompt(memory_receipts=False)
        self.assertNotIn("Cortex memory: M:1234abcd", prompt)


if __name__ == "__main__":
    unittest.main()
