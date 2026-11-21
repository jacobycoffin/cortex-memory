"""The wiring, not just the helper: a no-memory turn still carries the floor (item #1)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.hermes_provider import CortexMemoryProvider
from cortex.store import CortexStore


class PrefetchFloorWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.provider = CortexMemoryProvider(
            {
                "db_path": "$HERMES_HOME/cortex/test.db",
                "auto_capture": False,
                "retrieval_threshold": 0.01,
            }
        )
        self.provider.initialize("floor-session", hermes_home=self.tmp.name, agent_context="primary")
        self.store = self.provider._store

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _pin(self, content: str) -> None:
        created = self.store.add_memory(
            content=content,
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            importance=0.9,
        )
        memory_id = created[0] if isinstance(created, tuple) else created
        self.store.set_pinned(memory_id, True)

    def test_a_trivial_turn_still_carries_the_pinned_fact(self) -> None:
        """'hi' is precisely the turn where the planner says memory is unnecessary."""
        self._pin("Jacoby's maintenance window is Sunday 02:00 and the box is CT116.")
        block = self.provider.prefetch("hi")
        self.assertIn("CT116", block)
        self.assertIn("always-on floor", block)

    def test_a_real_turn_writes_a_shadow_abstention_row(self) -> None:
        """The ledger must fill from live turns while the policy stays off."""
        from cortex.abstention_ledger import summary

        self._pin("Jacoby's maintenance window is Sunday 02:00 and the box is CT116.")
        self.provider.prefetch("maintenance window")
        stats = summary(self.store)
        self.assertEqual(stats["turns"], 1)
        self.assertEqual(stats["abstained"], 0)

    def test_an_empty_store_yields_no_block(self) -> None:
        self.assertEqual(self.provider.prefetch("hi"), "")


if __name__ == "__main__":
    unittest.main()
