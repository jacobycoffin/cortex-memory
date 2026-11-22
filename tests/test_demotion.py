"""Demotion rules: a core fact that goes quiet leaves (#123)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.promotion import (
    demote_stale,
    history,
    is_member,
    members,
    promote,
    stale_members,
    touch_confirmations,
)
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


def days_ago(days: int) -> str:
    return (NOW - timedelta(days=days)).isoformat()


class DemotionTests(unittest.TestCase):
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

    def test_a_fresh_member_stays(self) -> None:
        memory_id = self._memory("Fresh core fact.")
        promote(self.store, memory_id, at=days_ago(5))
        self.assertEqual(stale_members(self.store, days=90, now=NOW), [])

    def test_a_quiet_member_leaves(self) -> None:
        memory_id = self._memory("Fact nobody has touched in a year.")
        promote(self.store, memory_id, at=days_ago(365))
        stale = stale_members(self.store, days=90, now=NOW)
        self.assertEqual([memory_id for memory_id, _reason in stale], [memory_id])
        self.assertIn("quiet since", stale[0][1])

    def test_a_recent_confirmation_keeps_it(self) -> None:
        memory_id = self._memory("Old fact, restated last week.")
        promote(self.store, memory_id, at=days_ago(365))
        touch_confirmations(self.store, [memory_id], at=days_ago(7))
        self.assertEqual(stale_members(self.store, days=90, now=NOW), [])

    def test_recent_use_keeps_it(self) -> None:
        """Being used is evidence, even when nobody restated it in words."""
        memory_id = self._memory("Old fact, still informing answers.")
        promote(self.store, memory_id, at=days_ago(365))
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE memories SET last_used_at = ? WHERE id = ?", (days_ago(3), memory_id)
            )
        self.assertEqual(stale_members(self.store, days=90, now=NOW), [])

    def test_a_pin_outranks_the_timer(self) -> None:
        memory_id = self._memory("Pinned years ago and never touched since.", pinned=True)
        promote(self.store, memory_id, by="pin", at=days_ago(900))
        self.assertEqual(stale_members(self.store, days=90, now=NOW), [])

    def test_an_archived_memory_leaves_regardless_of_window(self) -> None:
        memory_id = self._memory("Archived yesterday but still a core member.")
        promote(self.store, memory_id, at=days_ago(1))
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET state='archived' WHERE id=?", (memory_id,))
        stale = stale_members(self.store, days=90, now=NOW)
        self.assertEqual([memory_id for memory_id, _reason in stale], [memory_id])
        self.assertIn("archived", stale[0][1])

    def test_dry_run_changes_nothing(self) -> None:
        memory_id = self._memory("Stale fact.")
        promote(self.store, memory_id, at=days_ago(365))
        demote_stale(self.store, days=90, now=NOW, apply=False)
        self.assertTrue(is_member(self.store, memory_id))

    def test_applying_removes_membership_but_not_the_memory(self) -> None:
        memory_id = self._memory("Stale fact that should leave core.")
        promote(self.store, memory_id, at=days_ago(365))
        demote_stale(self.store, days=90, now=NOW, apply=True)
        self.assertFalse(is_member(self.store, memory_id))
        self.assertEqual(members(self.store), [])
        self.assertIsNotNone(self.store.get_memory(memory_id))
        self.assertEqual(str(self.store.get_memory(memory_id)["state"]), "active")

    def test_a_second_pass_is_a_no_op(self) -> None:
        memory_id = self._memory("Stale fact.")
        promote(self.store, memory_id, at=days_ago(365))
        demote_stale(self.store, days=90, now=NOW, apply=True)
        self.assertEqual(demote_stale(self.store, days=90, now=NOW, apply=True), [])

    def test_the_window_comes_from_the_environment(self) -> None:
        import os
        from unittest import mock

        from cortex.promotion import window_days

        with mock.patch.dict(os.environ, {"CORTEX_CORE_WINDOW_DAYS": "30"}):
            self.assertEqual(window_days(), 30)
        with mock.patch.dict(os.environ, {"CORTEX_CORE_WINDOW_DAYS": "not-a-number"}):
            self.assertEqual(window_days(), 90)


class HistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_promotion_and_demotion_are_recorded(self) -> None:
        memory_id = self.store.add_memory(
            content="A fact that enters and leaves core.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]
        promote(self.store, memory_id, by="pin", reason="test")
        demote_stale(
            self.store,
            days=1,
            now=datetime.now(timezone.utc) + timedelta(days=10),
            apply=True,
        )
        events = history(self.store, memory_id)
        self.assertEqual([event["action"] for event in events], ["demote", "promote"])
        self.assertIn("pin", events[1]["reason"])

    def test_tied_timestamps_order_by_insertion(self) -> None:
        """Two events in the same instant must not depend on machine speed.

        CI caught this: `ORDER BY at DESC` alone returned promote-then-demote on a fast box
        and the reverse on a slow one, because both events landed in the same microsecond.
        """
        from cortex.promotion import demote

        memory_id = self.store.add_memory(
            content="A fact with two events in one instant.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]
        stamp = "2026-01-01T00:00:00+00:00"
        promote(self.store, memory_id, at=stamp)
        demote(self.store, memory_id, at=stamp, reason="same instant")
        self.assertEqual(
            [event["action"] for event in history(self.store, memory_id)], ["demote", "promote"]
        )

    def test_history_is_filterable_by_memory(self) -> None:
        first = self.store.add_memory(
            content="First fact.", kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]
        second = self.store.add_memory(
            content="Second fact.", kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]
        promote(self.store, first)
        promote(self.store, second)
        self.assertEqual([event["memory_id"] for event in history(self.store, first)], [first])


if __name__ == "__main__":
    unittest.main()
