"""Continuity must be small, derived, and absent when there is nothing to carry (item #18)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.continuity import (
    MAX_TOPICS,
    build,
    open_commitments,
    previous_session,
    session_topics,
    used_memories,
)
from cortex.store import CortexStore


class ContinuityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _memory(self, content: str) -> str:
        created = self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )
        return created[0] if isinstance(created, tuple) else created

    def _log(self, memory_id: str, event: str, query: str, session: str, offset: str = "+0 seconds") -> None:
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO access_log (memory_id, event, query, session_id, score, created_at) "
                "VALUES (?, ?, ?, ?, 0.5, datetime('now', ?))",
                (memory_id, event, query, session, offset),
            )

    def test_no_history_yields_nothing(self) -> None:
        self.assertEqual(build(self.store, current_session="new"), "")

    def test_the_current_session_is_never_its_own_predecessor(self) -> None:
        memory = self._memory("Seerr notifications stopped after the update.")
        self._log(memory, "retrieved", "why did seerr notifications stop", "s1")
        self.assertIsNone(previous_session(self.store, current="s1"))
        self.assertEqual(previous_session(self.store, current="s2"), "s1")

    def test_the_block_carries_the_previous_topics(self) -> None:
        memory = self._memory("Seerr notifications stopped after the update.")
        self._log(memory, "retrieved", "why did seerr notifications stop", "s1")
        block = build(self.store, current_session="s2")
        self.assertIn("Where you left off", block)
        self.assertIn("seerr", block)
        self.assertIn("fallible", block)

    def test_used_memories_appear_and_retrieved_ones_do_not(self) -> None:
        used = self._memory("The telegram bot token lives in the homelab env file.")
        other = self._memory("Cortex ranking prefers used memories.")
        self._log(used, "used", "where is the bot token", "s1")
        self._log(other, "retrieved", "where is the bot token", "s1")
        summaries = used_memories(self.store, "s1")
        self.assertEqual(len(summaries), 1)
        self.assertIn("telegram bot token", summaries[0])

    def test_the_block_stays_small(self) -> None:
        memory = self._memory("Seerr notifications stopped after the update.")
        for index in range(20):
            self._log(memory, "retrieved", f"question number {index} about seerr", "s1", f"+{index} seconds")
        block = build(self.store, current_session="s2")
        self.assertLessEqual(len(block.splitlines()), 6)
        self.assertLessEqual(block.count(";"), MAX_TOPICS)

    def test_filler_queries_are_skipped(self) -> None:
        memory = self._memory("A fact.")
        self._log(memory, "retrieved", "do that", "s1")
        self.assertEqual(session_topics(self.store, "s1"), [])

    def test_open_commitments_are_counted(self) -> None:
        self.assertEqual(open_commitments(self.store), 0)

    def test_a_store_without_a_connection_is_safe(self) -> None:
        class Broken:
            pass

        self.assertEqual(build(Broken(), current_session="s"), "")
        self.assertIsNone(previous_session(Broken(), current="s"))


if __name__ == "__main__":
    unittest.main()
