"""Cold sweep triage: never-recalled memories, archived in batches, undone by batch (#92)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import triage
from cortex.promotion import promote
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class TriageTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str = "A lonely fact.", *, days_old: float = 60,
               kind: str = "semantic", **kwargs) -> str:
        created = self.store.add_memory(
            content=content,
            kind=kind,
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            importance=kwargs.pop("importance", 0.5),
            **kwargs,
        )
        memory_id = created[0] if isinstance(created, tuple) else created
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE memories SET created_at = ? WHERE id = ?",
                ((NOW - timedelta(days=days_old)).isoformat(), memory_id),
            )
        return memory_id


class CandidateTests(TriageTestCase):
    def test_a_never_recalled_memory_is_a_candidate(self) -> None:
        memory_id = self.memory("Never read once.", days_old=60)
        self.assertEqual([item["memory_id"] for item in triage.candidates(self.store, at=NOW)],
                         [memory_id])

    def test_a_retrieved_memory_is_not(self) -> None:
        memory_id = self.memory("Read once.", days_old=60)
        self.store.log_access(memory_id, "retrieved", query="q")
        self.assertEqual(triage.candidates(self.store, at=NOW), [])

    def test_any_access_event_excludes_it_even_with_a_zero_counter(self) -> None:
        """The counter can lag; an event of any kind is evidence somebody touched it."""
        memory_id = self.memory("Touched, not counted.", days_old=60)
        self.store.log_access(memory_id, "used", query="q")
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET retrieved_count = 0 WHERE id = ?", (memory_id,))
        self.assertEqual(triage.candidates(self.store, at=NOW), [])

    def test_a_fresh_memory_is_too_young(self) -> None:
        self.memory("Written this morning.", days_old=1)
        self.assertEqual(triage.candidates(self.store, at=NOW), [])

    def test_a_pinned_memory_is_never_triaged(self) -> None:
        self.memory("Pinned rule.", days_old=60, pinned=True)
        self.assertEqual(triage.candidates(self.store, at=NOW), [])

    def test_a_protected_memory_is_never_triaged(self) -> None:
        memory_id = self.memory("Core fact.", days_old=60)
        promote(self.store, memory_id, by="pin")
        self.assertEqual(triage.candidates(self.store, at=NOW), [])

    def test_the_minimum_age_is_configurable(self) -> None:
        import os
        from unittest import mock

        self.memory("Two days old.", days_old=2)
        self.assertEqual(triage.candidates(self.store, at=NOW), [])
        with mock.patch.dict(os.environ, {"CORTEX_TRIAGE_MIN_AGE_DAYS": "1"}):
            self.assertEqual(len(triage.candidates(self.store, at=NOW)), 1)

    def test_a_high_importance_memory_is_not_batch_triaged(self) -> None:
        """Unread is not worthless: the first live batch led with the dashboard URL and a MAC."""
        self.memory("The dashboard lives at cortex.example.com", days_old=60, importance=0.95)
        self.assertEqual(triage.candidates(self.store, at=NOW), [])

    def test_ordinary_importance_is_still_triaged(self) -> None:
        self.memory("A routine note.", days_old=60, importance=0.4)
        self.assertEqual(len(triage.candidates(self.store, at=NOW)), 1)

    def test_the_batch_starts_with_the_least_important(self) -> None:
        for importance in (0.6, 0.2, 0.4):
            self.memory(f"note {importance}", days_old=60, importance=importance)
        built = triage.batch(self.store, size=3, at=NOW)
        self.assertEqual([round(item["importance"], 2) for item in built.items], [0.2, 0.4, 0.6])

    def test_a_never_walk_kind_is_not_batch_triaged(self) -> None:
        for kind in ("preference", "procedural", "identity", "commitment"):
            self.memory(f"a {kind}", days_old=60, kind=kind, importance=0.3)
        self.assertEqual(triage.candidates(self.store, at=NOW), [])

    def test_the_count_matches_the_queue(self) -> None:
        for index in range(7):
            self.memory(f"fact {index}", days_old=60)
        self.assertEqual(triage.count_candidates(self.store, at=NOW), 7)


class BatchTests(TriageTestCase):
    def test_a_batch_is_bounded_and_does_not_repeat(self) -> None:
        for index in range(30):
            self.memory(f"fact {index}", days_old=60)
        first = triage.batch(self.store, size=10, at=NOW)
        self.assertEqual(len(first.items), 10)
        triage.apply(self.store, first, at=NOW)
        second = triage.batch(self.store, size=10, at=NOW)
        self.assertEqual(len(second.items), 10)
        self.assertEqual(set(first.ids) & set(second.ids), set())

    def test_the_default_action_is_archive(self) -> None:
        self.memory("archive me", days_old=60)
        built = triage.batch(self.store, at=NOW)
        self.assertEqual(built.action, "archived")

    def test_triage_never_tombstones(self) -> None:
        self.memory("archive me", days_old=60)
        with self.assertRaises(ValueError):
            triage.batch(self.store, action="tombstoned", at=NOW)

    def test_applying_archives_the_batch(self) -> None:
        memory_id = self.memory("archive me", days_old=60)
        built = triage.batch(self.store, at=NOW)
        moved = triage.apply(self.store, built, at=NOW)
        self.assertEqual(moved, [memory_id])
        self.assertEqual(self.store.get_memory(memory_id)["state"], "archived")

    def test_a_batch_cannot_be_applied_twice(self) -> None:
        self.memory("archive me", days_old=60)
        built = triage.batch(self.store, at=NOW)
        triage.apply(self.store, built, at=NOW)
        self.assertEqual(triage.apply(self.store, built, at=NOW), [])

    def test_undo_restores_exactly_that_batch(self) -> None:
        for index in range(6):
            self.memory(f"fact {index}", days_old=60)
        built = triage.batch(self.store, size=3, at=NOW)
        triage.apply(self.store, built, at=NOW)
        applied = set(built.ids)
        restored = triage.undo(self.store, built)
        self.assertEqual(set(restored), applied)
        for memory_id in restored:
            self.assertEqual(self.store.get_memory(memory_id)["state"], "active")

    def test_undo_does_not_touch_a_second_batch(self) -> None:
        for index in range(6):
            self.memory(f"fact {index}", days_old=60)
        first = triage.batch(self.store, size=3, at=NOW)
        triage.apply(self.store, first, at=NOW)
        second = triage.batch(self.store, size=3, at=NOW)
        triage.apply(self.store, second, at=NOW)
        triage.undo(self.store, first)
        for memory_id in first.ids:
            self.assertEqual(self.store.get_memory(memory_id)["state"], "active")
        for memory_id in second.ids:
            self.assertEqual(self.store.get_memory(memory_id)["state"], "archived")

    def test_undo_twice_is_a_no_op(self) -> None:
        self.memory("once", days_old=60)
        built = triage.batch(self.store, at=NOW)
        triage.apply(self.store, built, at=NOW)
        self.assertEqual(len(triage.undo(self.store, built)), 1)
        self.assertEqual(triage.undo(self.store, built), [])

    def test_undo_before_apply_does_nothing(self) -> None:
        self.memory("not yet", days_old=60)
        built = triage.batch(self.store, at=NOW)
        self.assertEqual(triage.undo(self.store, built), [])

    def test_the_batch_list_is_reported(self) -> None:
        self.memory("one", days_old=60)
        built = triage.batch(self.store, at=NOW)
        triage.apply(self.store, built, at=NOW)
        listed = triage.batches(self.store)
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["size"], 1)
        self.assertIsNotNone(listed[0]["applied_at"])
        self.assertIsNone(listed[0]["undone_at"])


class KeepTests(TriageTestCase):
    def test_keeping_a_memory_removes_it_from_the_queue_for_good(self) -> None:
        """Without this the same batch comes back next run and triage gets abandoned."""
        memory_id = self.memory("keep me", days_old=60)
        self.assertEqual(len(triage.candidates(self.store, at=NOW)), 1)
        self.assertEqual(triage.keep(self.store, [memory_id]), 1)
        self.assertEqual(triage.candidates(self.store, at=NOW), [])
        self.assertIn(memory_id, triage.kept(self.store))
        self.assertEqual(self.store.get_memory(memory_id)["state"], "active")

    def test_keeping_is_idempotent(self) -> None:
        memory_id = self.memory("keep me", days_old=60)
        triage.keep(self.store, [memory_id])
        triage.keep(self.store, [memory_id])
        self.assertEqual(triage.kept(self.store).count(memory_id), 1)


if __name__ == "__main__":
    unittest.main()
