"""Warm vs cold profile: stable block plus a 24h recent appendage (#129)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.core_block import RECENT_TITLE, build
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


def hours_ago(hours: int) -> str:
    return (NOW - timedelta(hours=hours)).isoformat()


class RecentSectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.store.add_memory(
            content="Jacoby prefers short answers with no preamble.",
            kind="preference",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            pinned=True,
            importance=0.9,
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _memory(self, content: str, **kwargs) -> str:
        created = self.store.add_memory(
            content=content,
            kind=kwargs.pop("kind", "semantic"),
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            **kwargs,
        )
        return created[0] if isinstance(created, tuple) else created

    def _touch(self, memory_id: str, stamp: str) -> None:
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET last_used_at = ? WHERE id = ?", (stamp, memory_id))

    def test_a_fresh_memory_appears_in_recent(self) -> None:
        memory_id = self._memory("Worked on the recall cache this morning.")
        self._touch(memory_id, hours_ago(2))
        block = build(self.store, now=NOW)
        self.assertIn(RECENT_TITLE, block.text)
        self.assertIn(memory_id[:8], block.text)
        self.assertEqual(block.recent, 1)

    def test_an_old_memory_does_not(self) -> None:
        memory_id = self._memory("Worked on the recall cache days ago.")
        self._touch(memory_id, hours_ago(72))
        block = build(self.store, now=NOW)
        self.assertNotIn(RECENT_TITLE, block.text)
        self.assertEqual(block.recent, 0)

    def test_recent_can_be_left_out_entirely(self) -> None:
        memory_id = self._memory("Worked on this an hour ago.")
        self._touch(memory_id, hours_ago(1))
        block = build(self.store, now=NOW, include_recent=False)
        self.assertNotIn(RECENT_TITLE, block.text)
        self.assertEqual(block.recent, 0)

    def test_the_stable_profile_is_not_crowded_out_by_recent(self) -> None:
        """The profile is a promise; the warm part must not eat its budget."""
        for index in range(20):
            fresh = self._memory(f"Fresh fact {index} from the last hour.")
            self._touch(fresh, hours_ago(1))
        block = build(self.store, now=NOW, max_chars=2000)
        self.assertIn("Preferences:", block.text)
        self.assertIn(RECENT_TITLE, block.text)
        self.assertLessEqual(block.chars, 2000)

    def test_a_memory_is_never_shown_twice(self) -> None:
        memory_id = self._memory("A fact that is both pinned and fresh.")
        self.store.set_pinned(memory_id, True)
        self._touch(memory_id, hours_ago(1))
        block = build(self.store, now=NOW)
        self.assertEqual(block.text.count(memory_id[:8]), 1)

    def test_recent_is_ordered_newest_first(self) -> None:
        older = self._memory("Older recent fact.")
        newer = self._memory("Newer recent fact.")
        self._touch(older, hours_ago(20))
        self._touch(newer, hours_ago(1))
        text = build(self.store, now=NOW).text
        self.assertLess(text.index(newer[:8]), text.index(older[:8]))

    def test_pinned_entries_are_not_repeated_in_recent(self) -> None:
        for index in range(4):
            self._memory(f"Another pinned fact {index}.", pinned=True, importance=0.5 + index / 100)
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET last_used_at = ? WHERE pinned = 1", (hours_ago(1),))
        block = build(self.store, now=NOW)
        self.assertEqual(block.recent, 0, block.text)

    def test_the_window_is_a_day(self) -> None:
        """25 hours old may still be in the profile; it must not be in *Recent*."""
        just_in = self._memory("Twenty three hours ago.")
        just_out = self._memory("Twenty five hours ago.")
        self._touch(just_in, hours_ago(23))
        self._touch(just_out, hours_ago(25))
        text = build(self.store, now=NOW).text
        recent_block = text.split(RECENT_TITLE, 1)[1] if RECENT_TITLE in text else ""
        self.assertIn(just_in[:8], recent_block)
        self.assertNotIn(just_out[:8], recent_block)

    def test_rendering_is_deterministic_for_a_fixed_now(self) -> None:
        memory_id = self._memory("A fresh fact.")
        self._touch(memory_id, hours_ago(3))
        self.assertEqual(build(self.store, now=NOW).text, build(self.store, now=NOW).text)


if __name__ == "__main__":
    unittest.main()
