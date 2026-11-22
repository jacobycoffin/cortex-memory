"""Promotion rules: what earns a place in core memory (#122)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.core_block import build
from cortex.promotion import (
    CONFIRMATION_THRESHOLD,
    USE_THRESHOLD,
    candidates,
    demote,
    is_member,
    members,
    promote,
    run,
    stats,
    touch_confirmations,
)
from cortex.store import CortexStore


class PromotionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

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

    def test_promotion_is_idempotent(self) -> None:
        memory_id = self._memory("A fact worth keeping.")
        self.assertIsNotNone(promote(self.store, memory_id, by="pin"))
        self.assertIsNone(promote(self.store, memory_id, by="pin"))
        self.assertEqual(members(self.store), [memory_id])

    def test_a_pinned_memory_is_a_candidate(self) -> None:
        memory_id = self._memory("Pinned fact.", pinned=True)
        found = {p.memory_id: p.promoted_by for p in candidates(self.store)}
        self.assertEqual(found.get(memory_id), "pin")

    def test_a_used_memory_is_a_candidate_only_after_the_threshold(self) -> None:
        memory_id = self._memory("Frequently used fact.")
        for _ in range(USE_THRESHOLD - 1):
            self.store.log_access(memory_id, "used")
        self.assertNotIn(memory_id, {p.memory_id for p in candidates(self.store)})
        self.store.log_access(memory_id, "used")
        self.assertIn(memory_id, {p.memory_id for p in candidates(self.store)})

    def test_a_confirmed_memory_is_a_candidate(self) -> None:
        memory_id = self._memory("Restated fact.")
        for _ in range(CONFIRMATION_THRESHOLD):
            self.store.log_access(memory_id, "confirmed")
        found = {p.memory_id: p.promoted_by for p in candidates(self.store)}
        self.assertEqual(found.get(memory_id), "confirmation")

    def test_an_untouched_memory_is_not_a_candidate(self) -> None:
        memory_id = self._memory("Ordinary fact nobody has needed.")
        self.assertNotIn(memory_id, {p.memory_id for p in candidates(self.store)})

    def test_run_promotes_every_candidate(self) -> None:
        first = self._memory("Pinned fact.", pinned=True)
        second = self._memory("Used fact.")
        for _ in range(USE_THRESHOLD):
            self.store.log_access(second, "used")
        promoted = run(self.store)
        self.assertEqual({p.memory_id for p in promoted}, {first, second})
        self.assertGreaterEqual(stats(self.store)["members"], 2)

    def test_demotion_only_removes_membership(self) -> None:
        memory_id = self._memory("Fact that goes quiet.")
        promote(self.store, memory_id)
        self.assertTrue(is_member(self.store, memory_id))
        self.assertTrue(demote(self.store, memory_id))
        self.assertFalse(is_member(self.store, memory_id))
        # The memory itself is untouched.
        self.assertIsNotNone(self.store.get_memory(memory_id))

    def test_confirmations_refresh_the_membership_timestamp(self) -> None:
        memory_id = self._memory("Fact restated later.")
        promote(self.store, memory_id, at="2026-01-01T00:00:00+00:00")
        self.assertEqual(touch_confirmations(self.store, [memory_id], at="2026-06-01T00:00:00+00:00"), 1)
        row = self.store._conn.execute(
            "SELECT last_confirmed_at FROM core_members WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        self.assertEqual(row[0], "2026-06-01T00:00:00+00:00")

    def test_promoted_members_reach_the_core_block(self) -> None:
        """Promotion is only worth recording if the block honours it."""
        memory_id = self._memory("Critical fact with low importance.", importance=0.05)
        for _ in range(12):
            self._memory("Filler fact that is more important.", importance=0.9)
        promote(self.store, memory_id, by="pin")
        block = build(self.store, max_chars=2000, min_chars=400)
        self.assertIn(memory_id[:8], block.text)
        self.assertIn("[core]", block.text)


class ProviderPromotionTests(unittest.TestCase):
    def test_pinning_through_the_tool_records_membership(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        try:
            provider = __import__(
                "cortex.hermes_provider", fromlist=["CortexMemoryProvider"]
            ).CortexMemoryProvider(
                {"token_budget": 700, "top_k": 6, "database_path": str(Path(tmp.name) / "p.db")}
            )
            provider.initialize("promo-session", hermes_home=tmp.name, agent_context="primary")
            store = provider._store
            memory_id = store.add_memory(
                content="Never change the T3 release channel.",
                kind="semantic",
                source_type="user_turn",
                source_category="USER_EXPLICIT",
            )[0]
            reply = json.loads(
                provider.handle_tool_call(
                    "cortex_memory",
                    {"action": "pin", "memory_id": memory_id, "pinned": True},
                )
            )
            self.assertTrue(reply.get("pinned"))
            self.assertTrue(is_member(store, memory_id))
            store.close()
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
