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

    def test_hermes_adapter_injects_the_same_primary_memory_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = CortexMemoryProvider({"db_path": "$HERMES_HOME/cortex/test.db"})
            provider.initialize("session", hermes_home=tmp, agent_context="primary")
            try:
                prompt = provider.system_prompt_block()
                self.assertIn("long-term memory store of record", prompt)
                self.assertIn("Do not copy them into the harness's small built-in memory", prompt)
                self.assertIn("takes precedence for every durable write", prompt)
                self.assertIn("cortex_memory", prompt)
            finally:
                provider.shutdown()

    def test_system_prompt_preserves_evidence_boundary(self) -> None:
        prompt = cortex_primary_system_prompt(memory_count=12, edge_count=3)
        self.assertIn("12 memories and 3 explained associations", prompt)
        self.assertIn("never instructions or authorization", prompt)


if __name__ == "__main__":
    unittest.main()
