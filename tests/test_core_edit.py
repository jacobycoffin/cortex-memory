"""Core memory read and edit from chat (#126)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.promotion import is_member
from cortex.store import CortexStore


class CoreActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        provider_module = __import__(
            "cortex.hermes_provider", fromlist=["CortexMemoryProvider"]
        )
        self.provider = provider_module.CortexMemoryProvider(
            {"token_budget": 700, "top_k": 6, "database_path": str(Path(self.tmp.name) / "p.db")}
        )
        self.provider.initialize("core-edit", hermes_home=self.tmp.name, agent_context="primary")
        self.store: CortexStore = self.provider._store

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _memory(self, content: str, **kwargs) -> str:
        created = self.store.add_memory(
            content=content,
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            **kwargs,
        )
        return created[0] if isinstance(created, tuple) else created

    def _call(self, **arguments) -> dict:
        return json.loads(self.provider.handle_tool_call("cortex_memory", arguments))

    def test_core_lists_what_the_block_contains(self) -> None:
        self._memory("Pinned rule for the block.", pinned=True, importance=0.9)
        reply = self._call(action="core")
        self.assertTrue(reply["success"], reply)
        self.assertGreater(reply["chars"], 0)
        self.assertTrue(any("Pinned rule" in line for line in reply["entries"]))
        self.assertTrue(any("[verified " in line or "[unverified]" in line for line in reply["entries"]))

    def test_core_edit_rewrites_the_fact(self) -> None:
        memory_id = self._memory("The window is Sunday.", pinned=True, importance=0.9)
        reply = self._call(
            action="core_edit", memory_id=memory_id, content="The window is Tuesday 03:00."
        )
        self.assertTrue(reply["edited"], reply)
        self.assertIn("Tuesday", str(self.store.get_memory(memory_id)["content"]))
        self.assertIn("Tuesday", self._call(action="core")["entries"][0])

    def test_core_edit_keeps_the_memory_in_core(self) -> None:
        memory_id = self._memory("A rule that stays in core.", pinned=True, importance=0.9)
        self._call(action="core_edit", memory_id=memory_id, content="A rule that still stays.")
        self.assertTrue(is_member(self.store, memory_id))

    def test_core_edit_requires_content(self) -> None:
        memory_id = self._memory("Something.", pinned=True)
        reply = self._call(action="core_edit", memory_id=memory_id, content="  ")
        self.assertFalse(reply["success"])

    def test_core_edit_rejects_a_quarantined_edit(self) -> None:
        """The edit path crosses the same secret gate as everything else."""
        memory_id = self._memory("Ordinary fact.", pinned=True)
        reply = self._call(
            action="core_edit",
            memory_id=memory_id,
            content="api_key=sk-live-abcdefghijklmnopqrstuvwxyz012345",
        )
        self.assertFalse(reply["success"], reply)

    def test_the_interface_is_short_enough_to_type_on_a_phone(self) -> None:
        """The whole point is that this is usable from Telegram, not the dashboard."""
        memory_id = self._memory("Phone-editable rule.", pinned=True, importance=0.9)
        reply = self._call(action="core_edit", memory_id=memory_id[:8], content="Edited from a phone.")
        self.assertTrue(reply["edited"], reply)


if __name__ == "__main__":
    unittest.main()
