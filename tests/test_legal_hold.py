"""A legal hold outranks pins, timers and force (#103)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import legal_hold as lh
from cortex import triage
from cortex import suppression as sup
from cortex.forget_ladder import plan, protections
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class HoldTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str = "A contract clause.", *, days_old: float = 400,
               kind: str = "semantic") -> str:
        memory_id = self.store.add_memory(
            content=content,
            kind=kind,
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            importance=0.4,
        )[0]
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET created_at = ? WHERE id = ?",
                         ((NOW - timedelta(days=days_old)).isoformat(), memory_id))
        return memory_id


class PlaceTests(HoldTestCase):
    def test_a_hold_is_placed_and_visible(self) -> None:
        memory_id = self.memory()
        result = lh.place(self.store, [memory_id], reason="signed contract")
        self.assertEqual(result.placed, [memory_id])
        self.assertTrue(lh.is_held(self.store, memory_id))
        listed = lh.holds(self.store)
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0].reason, "signed contract")

    def test_a_hold_also_pins_so_it_is_visible_in_the_profile(self) -> None:
        memory_id = self.memory()
        lh.place(self.store, [memory_id], reason="medical note")
        self.assertTrue(self.store.get_memory(memory_id)["pinned"])

    def test_unknown_ids_are_refused_not_invented(self) -> None:
        result = lh.place(self.store, ["00000000-0000-0000-0000-000000000000"], reason="x")
        self.assertEqual(result.placed, [])
        self.assertEqual(len(result.refused), 1)

    def test_a_prefix_resolves_to_the_full_id(self) -> None:
        memory_id = self.memory()
        lh.place(self.store, [memory_id[:8]], reason="contract")
        self.assertTrue(lh.is_held(self.store, memory_id))

    def test_placing_twice_updates_rather_than_duplicates(self) -> None:
        memory_id = self.memory()
        lh.place(self.store, [memory_id], reason="first")
        lh.place(self.store, [memory_id], reason="second")
        self.assertEqual(len(lh.holds(self.store)), 1)
        self.assertEqual(lh.holds(self.store)[0].reason, "second")


class ProtectionTests(HoldTestCase):
    def test_a_held_memory_is_in_the_protection_map(self) -> None:
        memory_id = self.memory()
        lh.place(self.store, [memory_id], reason="contract")
        self.assertEqual(protections(self.store).get(memory_id), "legal_hold")

    def test_the_ladder_never_walks_a_held_memory(self) -> None:
        memory_id = self.memory(days_old=2000)
        lh.place(self.store, [memory_id], reason="contract")
        self.assertEqual(plan(self.store, at=NOW).steps, [])

    def test_triage_never_considers_a_held_memory(self) -> None:
        memory_id = self.memory(days_old=90)
        lh.place(self.store, [memory_id], reason="contract")
        self.assertEqual(triage.candidates(self.store, at=NOW), [])

    def test_a_held_memory_is_not_suppressed_by_a_sweep(self) -> None:
        memory_id = self.memory(days_old=2000)
        lh.place(self.store, [memory_id], reason="contract")
        self.store.set_state(memory_id, "active", reason="noop")
        self.assertEqual(self.store.get_memory(memory_id)["state"], "active")


class DeletionTests(HoldTestCase):
    def test_deletion_refuses_a_held_memory(self) -> None:
        memory_id = self.memory()
        lh.place(self.store, [memory_id], reason="contract")
        with self.assertRaises(sup.DeleteRefused) as caught:
            sup.delete(self.store, memory_id, confirm=memory_id)
        self.assertIn("legal hold", str(caught.exception))

    def test_even_allow_protected_cannot_override_a_hold(self) -> None:
        """Otherwise every destructive path carries a force that quietly ignores the hold too."""
        memory_id = self.memory()
        lh.place(self.store, [memory_id], reason="contract")
        with self.assertRaises(sup.DeleteRefused):
            sup.delete(self.store, memory_id, confirm=memory_id, allow_protected=True)
        self.assertIsNotNone(self.store.get_memory(memory_id))

    def test_releasing_the_hold_allows_deletion(self) -> None:
        memory_id = self.memory()
        lh.place(self.store, [memory_id], reason="contract")
        self.assertTrue(lh.release(self.store, memory_id, reason="retention expired"))
        self.assertFalse(lh.is_held(self.store, memory_id))
        result = sup.delete(self.store, memory_id, confirm=memory_id, allow_protected=True)
        self.assertEqual(result["deleted"], memory_id)

    def test_releasing_an_unheld_memory_is_a_no_op(self) -> None:
        memory_id = self.memory()
        self.assertFalse(lh.release(self.store, memory_id))


class ReviewTests(HoldTestCase):
    def test_a_hold_without_a_review_date_never_falls_due(self) -> None:
        memory_id = self.memory()
        lh.place(self.store, [memory_id], reason="contract")
        self.assertEqual(lh.due_for_review(self.store, at=NOW + timedelta(days=3650)), [])

    def test_a_hold_past_its_review_date_is_listed_not_lifted(self) -> None:
        """A review prompt, not a timer: the hold stays until a person releases it."""
        memory_id = self.memory()
        lh.place(self.store, [memory_id], reason="contract",
                 review_at=(NOW + timedelta(days=30)).isoformat())
        self.assertEqual(lh.due_for_review(self.store, at=NOW), [])
        due = lh.due_for_review(self.store, at=NOW + timedelta(days=31))
        self.assertEqual(len(due), 1)
        self.assertTrue(lh.is_held(self.store, memory_id))

    def test_the_report_counts_holds_and_reviews_due(self) -> None:
        first = self.memory("contract")
        second = self.memory("medical note")
        lh.place(self.store, [first], reason="contract",
                 review_at=lh.review_in(-1, at=NOW))
        lh.place(self.store, [second], reason="medical")
        described = lh.report(self.store, at=NOW)
        self.assertEqual(described["held"], 2)
        self.assertEqual(described["due_for_review"], 1)
        self.assertEqual(len(described["holds"]), 2)

    def test_a_released_hold_is_excluded_from_the_report_but_retained_in_history(self) -> None:
        memory_id = self.memory()
        lh.place(self.store, [memory_id], reason="contract")
        lh.release(self.store, memory_id, reason="done with it")
        self.assertEqual(lh.report(self.store)["held"], 0)
        self.assertEqual(len(lh.holds(self.store, include_released=True)), 1)


if __name__ == "__main__":
    unittest.main()
