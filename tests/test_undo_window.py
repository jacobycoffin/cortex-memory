"""The undo window (#186): short, stated, single-use, and exact or refused."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import undo_window
from cortex.store import CortexStore

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)


class Broken:
    _conn = None


class WindowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str = "the proxmox box runs 8.4") -> str:
        return self.store.add_memory(
            content=text, kind="semantic", source_type="user_turn", source_category="USER_STATED",
        )[0]

    def state(self, memory_id: str) -> str:
        return str(self.store._conn.execute(
            "SELECT state FROM memories WHERE id = ?", (memory_id,)).fetchone()[0])

    def set_state(self, memory_id: str, state: str) -> None:
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET state = ? WHERE id = ?", (state, memory_id))

    def opened(self, memory_id: str, *, before: str = "active", after: str = "archived",
               seconds: int = undo_window.DEFAULT_WINDOW_SECONDS, at: datetime = NOW):
        self.set_state(memory_id, after)
        return undo_window.open_window(
            self.store, target="memory", target_id=memory_id, action="retire",
            restore={"state": before}, seconds=seconds, at=at,
        )

    # -- the window is stated ---------------------------------------------------------

    def test_a_token_states_when_it_expires(self) -> None:
        token = self.opened(self.memory())
        self.assertIn("undo until", token.statement(at=NOW))
        self.assertEqual(token.seconds_left(at=NOW), undo_window.DEFAULT_WINDOW_SECONDS)

    def test_the_default_window_is_short_and_named(self) -> None:
        self.assertEqual(undo_window.DEFAULT_WINDOW_SECONDS, 30)

    def test_the_window_can_be_lowered_but_not_negative(self) -> None:
        token = self.opened(self.memory(), seconds=-5)
        self.assertEqual(token.seconds_left(at=NOW), 0.0)
        self.assertIn("window closed", token.statement(at=NOW))

    # -- undo restores exactly what changed ------------------------------------------

    def test_undo_puts_back_the_previous_value(self) -> None:
        memory_id = self.memory()
        self.set_state(memory_id, "cold")
        token = undo_window.open_window(
            self.store, target="memory", target_id=memory_id, action="mute",
            restore={"state": "active"}, at=NOW)
        self.set_state(memory_id, "cold")
        result = undo_window.undo(self.store, token.token_id, at=NOW)
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(self.state(memory_id), "active")

    def test_undo_restores_the_state_it_was_actually_in_not_a_default(self) -> None:
        """A restore that sets 'active' on a memory that was 'cold' has not undone anything."""
        memory_id = self.memory()
        self.set_state(memory_id, "cold")
        token = undo_window.open_window(
            self.store, target="memory", target_id=memory_id, action="retire",
            restore={"state": "cold"}, at=NOW)
        self.set_state(memory_id, "archived")
        undo_window.undo(self.store, token.token_id, at=NOW)
        self.assertEqual(self.state(memory_id), "cold")

    def test_a_pin_is_restored_as_an_integer_column(self) -> None:
        memory_id = self.memory()
        token = undo_window.open_window(
            self.store, target="memory", target_id=memory_id, action="pin",
            restore={"pinned": 0}, at=NOW)
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET pinned = 1 WHERE id = ?", (memory_id,))
        undo_window.undo(self.store, token.token_id, at=NOW)
        pinned = self.store._conn.execute("SELECT pinned FROM memories WHERE id = ?", (memory_id,)).fetchone()[0]
        self.assertEqual(pinned, 0)

    def test_only_restorable_columns_are_captured(self) -> None:
        token = undo_window.open_window(
            self.store, target="memory", target_id=self.memory(), action="pin",
            restore={"state": "active", "content": "anything at all"}, at=NOW)
        self.assertEqual(token.restore, {"state": "active"},
                         "a token that captured content would restore over a later correction")

    def test_an_empty_restore_opens_no_token(self) -> None:
        token = undo_window.open_window(
            self.store, target="memory", target_id=self.memory(), action="pin", restore={}, at=NOW)
        self.assertEqual(token.token_id, "")
        self.assertEqual(undo_window.pending(self.store, at=NOW), [])

    # -- once, and only on its own row -----------------------------------------------

    def test_a_token_is_single_use(self) -> None:
        memory_id = self.memory()
        token = self.opened(memory_id)
        self.assertTrue(undo_window.undo(self.store, token.token_id, at=NOW).ok)
        second = undo_window.undo(self.store, token.token_id, at=NOW)
        self.assertEqual(second.status, "already")

    def test_a_token_cannot_be_replayed_against_another_row(self) -> None:
        first = self.memory()
        second = self.memory("the box runs 8.3")
        token = self.opened(first)
        self.set_state(second, "archived")
        token.target_id = second  # a view that trusted its own copy of the id
        result = undo_window.undo(self.store, token.token_id, at=NOW)
        self.assertTrue(result.ok)
        self.assertEqual(self.state(first), "active", "the token's own row is the one restored")

    # -- too late, and conflict ------------------------------------------------------

    def test_an_expired_token_refuses_and_names_the_durable_path(self) -> None:
        memory_id = self.memory()
        token = self.opened(memory_id, seconds=30, at=NOW)
        later = NOW + timedelta(seconds=31)
        result = undo_window.undo(self.store, token.token_id, at=later)
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "expired")
        self.assertIn("correction", result.reason)
        self.assertEqual(self.state(memory_id), "archived")

    def test_an_undo_that_would_overwrite_a_later_change_refuses(self) -> None:
        memory_id = self.memory()
        token = undo_window.open_window(
            self.store, target="memory", target_id=memory_id, action="pin",
            restore={"pinned": 1}, at=NOW)
        self.set_state(memory_id, "quarantine")  # somebody else moved it after the action
        result = undo_window.undo(self.store, token.token_id, at=NOW)
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "stale")
        self.assertIn("state", result.reason)

    def test_an_unknown_token_is_named(self) -> None:
        result = undo_window.undo(self.store, "not-a-token", at=NOW)
        self.assertEqual(result.status, "unknown")
        self.assertIn("no undo token", result.reason)

    def test_a_broken_store_refuses_rather_than_raising(self) -> None:
        result = undo_window.undo(Broken(), "anything", at=NOW)
        self.assertEqual(result.status, "refused")

    # -- what a UI can offer ---------------------------------------------------------

    def test_pending_lists_only_open_windows(self) -> None:
        memory_id = self.memory()
        open_token = self.opened(memory_id)
        closed = self.opened(memory_id, at=NOW - timedelta(minutes=5))
        tokens = undo_window.pending(self.store, at=NOW)
        ids = {token.token_id for token in tokens}
        self.assertIn(open_token.token_id, ids)
        self.assertNotIn(closed.token_id, ids)

    def test_a_used_token_leaves_the_pending_list(self) -> None:
        memory_id = self.memory()
        token = self.opened(memory_id)
        undo_window.undo(self.store, token.token_id, at=NOW)
        self.assertEqual(undo_window.pending(self.store, at=NOW), [])

    def test_rendering_names_the_action_and_the_deadline(self) -> None:
        memory_id = self.memory()
        self.opened(memory_id)
        lines = undo_window.render(undo_window.pending(self.store, at=NOW), at=NOW)
        self.assertEqual(len(lines), 1)
        self.assertIn("retire", lines[0])
        self.assertIn("until", lines[0])

    def test_an_edge_weight_can_be_undone(self) -> None:
        src = self.memory()
        dst = self.memory("the box runs 8.3")
        self.store.add_edge(src, dst, "supports", weight=0.4)
        with self.store.transaction() as conn:
            conn.execute("UPDATE edges SET weight = 0.9 WHERE relation = 'supports'")
        token = undo_window.open_window(
            self.store, target="edge", target_id=src, action="reinforce",
            restore={"weight": 0.4}, at=NOW)
        result = undo_window.undo(self.store, token.token_id, at=NOW)
        self.assertTrue(result.ok, result.reason)
        weight = self.store._conn.execute(
            "SELECT weight FROM edges WHERE relation = 'supports'").fetchone()[0]
        self.assertAlmostEqual(weight, 0.4, places=4)


if __name__ == "__main__":
    unittest.main()
