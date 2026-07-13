from __future__ import annotations

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
                self.assertIn("fallible evidence", batch.context())
                affected = batch.finish([memory_id], outcome="helpful")
                self.assertEqual(affected, [memory_id])
                self.assertEqual(memory.store.get_memory(memory_id)["helpful_count"], 1)

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
                with self.assertRaises(RuntimeError):
                    batch.finish()


if __name__ == "__main__":
    unittest.main()
