"""Suppression keeps it, deletion removes it, and both are honoured (#91)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import suppression as sup
from cortex.forget_ladder import restore
from cortex.promotion import promote
from cortex.retrieval import MemoryRetriever
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
CONTENT = "The greenhouse controller runs on the ESP32 in the shed with a 12V supply."


class SuppressionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.memory_id = self.store.add_memory(
            content=CONTENT,
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            importance=0.7,
        )[0]

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def search(self, query: str) -> list[str]:
        results, _diagnostics = MemoryRetriever(self.store).search_detailed(query, limit=5)
        return [str(result.memory["id"]) for result in results]


class SuppressionTests(SuppressionTestCase):
    def test_a_suppressed_memory_stops_being_recalled(self) -> None:
        self.assertIn(self.memory_id, self.search("greenhouse controller ESP32"))
        self.assertTrue(sup.suppress(self.store, self.memory_id, reason="test"))
        self.assertNotIn(self.memory_id, self.search("greenhouse controller ESP32"))

    def test_the_content_survives_suppression(self) -> None:
        sup.suppress(self.store, self.memory_id, reason="test")
        row = self.store.get_memory(self.memory_id)
        self.assertIsNotNone(row)
        self.assertEqual(row["content"], CONTENT)
        self.assertEqual(row["state"], sup.SUPPRESSED)

    def test_unsuppressing_restores_recall(self) -> None:
        sup.suppress(self.store, self.memory_id, reason="test")
        self.assertTrue(sup.unsuppress(self.store, self.memory_id))
        self.assertIn(self.memory_id, self.search("greenhouse controller ESP32"))

    def test_suppression_is_recorded_with_its_reason(self) -> None:
        sup.suppress(self.store, self.memory_id, reason="too noisy")
        listed = sup.suppressions(self.store)
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["reason"], "too noisy")
        self.assertEqual(listed[0]["content"], CONTENT)

    def test_a_timed_suppression_expires(self) -> None:
        sup.suppress(
            self.store, self.memory_id, until=(NOW + timedelta(hours=2)).isoformat()
        )
        self.assertEqual(sup.expired(self.store, at=NOW), [])
        later = NOW + timedelta(hours=3)
        self.assertEqual(sup.expired(self.store, at=later), [self.memory_id])
        self.assertEqual(sup.expire(self.store, at=later), [self.memory_id])
        self.assertIn(self.memory_id, self.search("greenhouse controller ESP32"))

    def test_an_untimed_suppression_never_expires(self) -> None:
        sup.suppress(self.store, self.memory_id)
        later = NOW + timedelta(days=3650)
        self.assertEqual(sup.expired(self.store, at=later), [])

    def test_suppressing_a_missing_memory_is_a_no_op(self) -> None:
        self.assertFalse(sup.suppress(self.store, "00000000-0000-0000-0000-000000000000"))

    def test_suppression_is_not_a_tombstone(self) -> None:
        """Different states because they mean different things: one is reversible by design."""
        sup.suppress(self.store, self.memory_id)
        self.assertEqual(self.store.get_memory(self.memory_id)["state"], "suppressed")
        self.store.set_state(self.memory_id, "tombstoned", reason="ladder")
        self.assertEqual(self.store.get_memory(self.memory_id)["state"], "tombstoned")


class DeletionTests(SuppressionTestCase):
    def test_deletion_requires_the_exact_id(self) -> None:
        with self.assertRaises(sup.DeleteRefused):
            sup.delete(self.store, self.memory_id, confirm="not-the-id")
        self.assertIsNotNone(self.store.get_memory(self.memory_id))

    def test_deletion_removes_the_row_and_records_what_was_lost(self) -> None:
        result = sup.delete(self.store, self.memory_id, confirm=self.memory_id,
                            reason="never existed")
        self.assertEqual(result["deleted"], self.memory_id)
        self.assertIsNone(self.store.get_memory(self.memory_id))
        audit = sup.deletions(self.store)
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["preview"], CONTENT)
        self.assertEqual(audit[0]["reason"], "never existed")
        self.assertTrue(audit[0]["content_hash"])

    def test_deletion_removes_its_edges(self) -> None:
        other = self.store.add_memory(
            content="The shed has a 12V supply for the controller.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]
        self.store.add_edge(self.memory_id, other, "relates_to", evidence_type="fixture")
        result = sup.delete(self.store, self.memory_id, confirm=self.memory_id)
        self.assertGreaterEqual(result["edges_removed"], 1)
        remaining = self.store._conn.execute(
            "SELECT COUNT(*) FROM edges WHERE src_id = ? OR dst_id = ?",
            (self.memory_id, self.memory_id),
        ).fetchone()[0]
        self.assertEqual(remaining, 0)

    def test_a_protected_memory_is_not_deleted_by_default(self) -> None:
        promote(self.store, self.memory_id, by="pin")
        with self.assertRaises(sup.DeleteRefused) as caught:
            sup.delete(self.store, self.memory_id, confirm=self.memory_id)
        self.assertIn("protected", str(caught.exception))
        self.assertIsNotNone(self.store.get_memory(self.memory_id))

    def test_a_protected_memory_can_be_deleted_deliberately(self) -> None:
        self.store.set_pinned(self.memory_id, True)
        result = sup.delete(
            self.store, self.memory_id, confirm=self.memory_id, allow_protected=True
        )
        self.assertEqual(result["deleted"], self.memory_id)

    def test_a_suppression_row_does_not_survive_deletion(self) -> None:
        sup.suppress(self.store, self.memory_id)
        sup.delete(self.store, self.memory_id, confirm=self.memory_id)
        self.assertEqual(sup.suppressions(self.store), [])

    def test_the_deletion_note_is_all_that_is_left(self) -> None:
        sup.delete(self.store, self.memory_id, confirm=self.memory_id, reason="test")
        note = sup.restore_deleted_note(self.store, self.memory_id[:8])
        self.assertIn("was deleted", note)
        self.assertIn(CONTENT, note)

    def test_an_unknown_id_reports_rather_than_inventing(self) -> None:
        self.assertIn("no deletion recorded", sup.restore_deleted_note(self.store, "deadbeef"))


class ReportTests(SuppressionTestCase):
    def test_the_report_distinguishes_all_four_situations(self) -> None:
        second = self.store.add_memory(
            content="A second fact about the greenhouse fans.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]
        third = self.store.add_memory(
            content="A third fact about the greenhouse roof vent.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]
        sup.suppress(self.store, second)
        sup.delete(self.store, third, confirm=third)
        self.store.set_state(self.memory_id, "tombstoned", reason="ladder")
        report = sup.report(self.store)
        self.assertEqual(report.suppressed, 1)
        self.assertEqual(report.tombstoned, 1)
        self.assertEqual(report.deleted, 1)
        lines = "\n".join(report.lines())
        self.assertIn("suppressed now: 1", lines)
        self.assertIn("deleted (audited, content gone): 1", lines)

    def test_a_lifted_suppression_stops_being_counted(self) -> None:
        sup.suppress(self.store, self.memory_id)
        sup.unsuppress(self.store, self.memory_id)
        self.assertEqual(sup.report(self.store).suppressed, 0)


class LadderInteropTests(SuppressionTestCase):
    def test_a_suppressed_memory_is_invisible_to_the_ladder(self) -> None:
        """The ladder walks active/cold/archived; suppression is a different decision."""
        sup.suppress(self.store, self.memory_id)
        from cortex import forget_ladder as ladder

        sweep = ladder.plan(self.store, at=NOW)
        self.assertEqual([step.memory_id for step in sweep.steps], [])

    def test_restoring_a_tombstoned_memory_is_not_suppression(self) -> None:
        self.store.set_state(self.memory_id, "tombstoned", reason="ladder")
        self.assertTrue(restore(self.store, self.memory_id))
        self.assertFalse(sup.is_suppressed(self.store, self.memory_id))


if __name__ == "__main__":
    unittest.main()
