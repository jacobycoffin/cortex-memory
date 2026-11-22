"""Restore once, restore twice, never again: one call then protection (#99)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import restore_ladder as rl
from cortex import suppression as sup
from cortex.forget_ladder import plan
from cortex.store import CortexStore


class RestoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str = "A fact.", *, state: str = "tombstoned") -> str:
        memory_id = self.store.add_memory(
            content=content,
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]
        if state != "active":
            self.store.set_state(memory_id, state, reason="fixture")
        return memory_id


class OneCallTests(RestoreTestCase):
    def test_a_tombstoned_memory_comes_back_in_one_call(self) -> None:
        memory_id = self.memory(state="tombstoned")
        result = rl.restore(self.store, memory_id)
        self.assertTrue(result.restored)
        self.assertEqual(self.store.get_memory(memory_id)["state"], "active")
        self.assertEqual(result.count, 1)

    def test_an_archived_memory_comes_back_too(self) -> None:
        memory_id = self.memory(state="archived")
        self.assertTrue(rl.restore(self.store, memory_id).restored)
        self.assertEqual(self.store.get_memory(memory_id)["state"], "active")

    def test_restoring_an_active_memory_is_harmless(self) -> None:
        memory_id = self.memory(state="active")
        result = rl.restore(self.store, memory_id)
        self.assertTrue(result.restored)
        self.assertEqual(result.count, 0)

    def test_an_unknown_id_reports_rather_than_inventing(self) -> None:
        result = rl.restore(self.store, "00000000-0000-0000-0000-000000000000")
        self.assertFalse(result.restored)
        self.assertIn("nothing to restore", result.summary)

    def test_restoring_also_lifts_a_suppression(self) -> None:
        memory_id = self.memory(state="active")
        sup.suppress(self.store, memory_id, reason="test")
        rl.restore(self.store, memory_id)
        self.assertEqual(self.store.get_memory(memory_id)["state"], "active")
        self.assertEqual(sup.suppressions(self.store), [])


class CountingTests(RestoreTestCase):
    def test_the_count_comes_from_the_lifecycle_history(self) -> None:
        memory_id = self.memory(state="tombstoned")
        rl.restore(self.store, memory_id)
        self.assertEqual(rl.restore_count(self.store, memory_id), 1)
        self.store.set_state(memory_id, "tombstoned", reason="ladder again")
        rl.restore(self.store, memory_id)
        self.assertEqual(rl.restore_count(self.store, memory_id), 2)

    def test_a_suppression_lift_is_not_a_ladder_restore(self) -> None:
        """Only coming back from a demotion counts; otherwise one flag inflates the history."""
        memory_id = self.memory(state="archived")
        sup.suppress(self.store, memory_id)
        rl.restore(self.store, memory_id)
        self.assertGreaterEqual(rl.restore_count(self.store, memory_id), 1)
        self.assertEqual(rl.restore_count(self.store, memory_id), 1)


class ProtectionTests(RestoreTestCase):
    def test_the_first_restore_does_not_pin(self) -> None:
        memory_id = self.memory(state="tombstoned")
        result = rl.restore(self.store, memory_id)
        self.assertFalse(result.protected)
        self.assertFalse(self.store.get_memory(memory_id)["pinned"])
        self.assertIn("one more restore pins it", result.summary)

    def test_the_second_restore_pins_permanently(self) -> None:
        memory_id = self.memory(state="archived")
        rl.restore(self.store, memory_id)
        self.store.set_state(memory_id, "archived", reason="ladder again")
        result = rl.restore(self.store, memory_id)
        self.assertTrue(result.protected)
        self.assertTrue(self.store.get_memory(memory_id)["pinned"])
        self.assertIn("pinned", result.summary)

    def test_a_pinned_memory_never_walks_again(self) -> None:
        memory_id = self.memory(state="archived")
        rl.restore(self.store, memory_id)
        self.store.set_state(memory_id, "archived", reason="ladder again")
        rl.restore(self.store, memory_id)
        self.store.set_state(memory_id, "archived", reason="for the test")
        steps = [step.memory_id for step in plan(self.store).steps]
        self.assertNotIn(memory_id, steps)

    def test_the_threshold_is_configurable(self) -> None:
        memory_id = self.memory(state="tombstoned")
        result = rl.restore(self.store, memory_id, protect_at=1)
        self.assertTrue(result.protected)

    def test_protection_can_be_switched_off(self) -> None:
        memory_id = self.memory(state="tombstoned")
        rl.restore(self.store, memory_id, protect_at=0)
        self.store.set_state(memory_id, "tombstoned", reason="again")
        result = rl.restore(self.store, memory_id, protect_at=0)
        self.assertFalse(result.protected)
        self.assertFalse(self.store.get_memory(memory_id)["pinned"])

    def test_protect_is_available_without_a_restore(self) -> None:
        memory_id = self.memory(state="active")
        self.assertTrue(rl.protect(self.store, memory_id))
        self.assertTrue(self.store.get_memory(memory_id)["pinned"])


class ReportTests(RestoreTestCase):
    def test_the_report_lists_restored_memories_and_pins_due(self) -> None:
        first = self.memory("restored twice", state="archived")
        for _ in range(2):
            rl.restore(self.store, first)
            self.store.set_state(first, "archived", reason="again")
        self.store.set_pinned(first, False)
        described = rl.report(self.store)
        counts = {item["memory_id"]: item["count"] for item in described["restored"]}
        self.assertEqual(counts[first], 2)
        self.assertIn(first, described["would_be_pinned"])

    def test_an_untouched_store_reports_nothing(self) -> None:
        self.memory("never restored", state="active")
        described = rl.report(self.store)
        self.assertEqual(described["restored"], [])
        self.assertEqual(described["would_be_pinned"], [])


if __name__ == "__main__":
    unittest.main()
