"""Receipts for every prune, and receipts that expire (#100)."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import prune_receipts as rc
from cortex import triage
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class ReceiptTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str = "A fact.", *, days_old: float = 60) -> str:
        memory_id = self.store.add_memory(
            content=content,
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            importance=0.4,
        )[0]
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET created_at = ? WHERE id = ?",
                         ((NOW - timedelta(days=days_old)).isoformat(), memory_id))
        return memory_id


class RecordTests(ReceiptTestCase):
    def test_a_receipt_records_what_was_touched(self) -> None:
        ids = [self.memory("one"), self.memory("two")]
        receipt_id = rc.record(self.store, "triage", "archived", ids, summary="batch x")
        receipt = rc.get(self.store, receipt_id)
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt.count, 2)
        self.assertEqual(set(receipt.ids), set(ids))
        self.assertEqual(receipt.summary, "batch x")

    def test_receipts_are_listed_newest_first(self) -> None:
        first = rc.record(self.store, "triage", "archived", [self.memory("a")], at=NOW)
        second = rc.record(
            self.store, "triage", "archived", [self.memory("b")],
            at=NOW + timedelta(hours=1),
        )
        listed = rc.recent(self.store)
        self.assertEqual([receipt.receipt_id for receipt in listed], [second, first])

    def test_listing_can_be_filtered_by_kind(self) -> None:
        rc.record(self.store, "triage", "archived", [self.memory("a")], at=NOW)
        rc.record(self.store, "forget_ladder", "walk", [self.memory("b")], at=NOW)
        listed = rc.recent(self.store, kind="forget_ladder")
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0].kind, "forget_ladder")

    def test_an_empty_receipt_is_still_a_receipt(self) -> None:
        receipt_id = rc.record(self.store, "triage", "archived", [])
        self.assertEqual(rc.get(self.store, receipt_id).count, 0)


class VerifyTests(ReceiptTestCase):
    def test_verification_confirms_what_is_still_present(self) -> None:
        ids = [self.memory("one"), self.memory("two")]
        receipt_id = rc.record(self.store, "triage", "archived", ids)
        checked = rc.verify(self.store, receipt_id)
        self.assertTrue(checked["found"])
        self.assertEqual(checked["still_present"], 2)
        self.assertEqual(checked["gone"], 0)

    def test_verification_reports_what_is_gone(self) -> None:
        keep = self.memory("keep")
        doomed = self.memory("doomed")
        receipt_id = rc.record(self.store, "triage", "archived", [keep, doomed])
        with self.store.transaction() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (doomed,))
        checked = rc.verify(self.store, receipt_id)
        self.assertEqual(checked["still_present"], 1)
        self.assertEqual(checked["gone"], 1)
        self.assertEqual(checked["gone_ids"], [doomed])

    def test_an_unknown_receipt_says_so(self) -> None:
        self.assertFalse(rc.verify(self.store, "nope")["found"])


class PruneTests(ReceiptTestCase):
    def test_old_receipts_are_dropped_and_recent_ones_kept(self) -> None:
        rc.record(self.store, "triage", "archived", [self.memory("old")],
                  at=NOW - timedelta(days=400))
        rc.record(self.store, "triage", "archived", [self.memory("new")], at=NOW)
        result = rc.prune(self.store, older_than_days=90, keep_last=1, at=NOW)
        self.assertEqual(result["deleted"], 1, "the 400-day-old receipt goes, the fresh one stays")
        self.assertEqual(result["kept"], 1)
        remaining = rc.recent(self.store)
        self.assertEqual(len(remaining), 1)
        # The survivor is inside retention, so a second sweep with keep_last=0 keeps it anyway.
        self.assertEqual(rc.prune(self.store, older_than_days=90, keep_last=0, at=NOW)["deleted"], 0)

    def test_the_newest_receipts_are_never_pruned(self) -> None:
        for index in range(5):
            rc.record(self.store, "triage", "archived", [self.memory(f"f{index}")],
                      at=NOW - timedelta(days=500 + index))
        rc.prune(self.store, older_than_days=90, keep_last=3, at=NOW)
        self.assertEqual(len(rc.recent(self.store)), 3)

    def test_pruning_is_counted_not_receipted(self) -> None:
        """A log of pruning the log is how audit systems grow without bound."""
        for index in range(4):
            rc.record(self.store, "triage", "archived", [self.memory(f"g{index}")],
                      at=NOW - timedelta(days=500))
        rc.prune(self.store, older_than_days=90, keep_last=1, at=NOW)
        described = rc.stats(self.store)
        self.assertEqual(described["pruned_total"], 3)
        self.assertEqual(described["by_kind"].get("triage"), 1)

    def test_retention_is_configurable(self) -> None:
        rc.record(self.store, "triage", "archived", [self.memory("x")],
                  at=NOW - timedelta(days=10))
        rc.prune(self.store, keep_last=0, at=NOW)
        self.assertEqual(len(rc.recent(self.store)), 1)
        with mock.patch.dict(os.environ, {"CORTEX_RECEIPT_RETENTION_DAYS": "5"}):
            rc.prune(self.store, keep_last=0, at=NOW)
        self.assertEqual(len(rc.recent(self.store)), 0)

    def test_an_empty_store_prunes_harmlessly(self) -> None:
        self.assertEqual(rc.prune(self.store, at=NOW)["deleted"], 0)


class ProducerTests(ReceiptTestCase):
    def test_a_triage_batch_writes_a_receipt(self) -> None:
        for index in range(3):
            self.memory(f"fact {index}")
        built = triage.batch(self.store, size=2, at=NOW)
        triage.apply(self.store, built, at=NOW)
        receipts = rc.recent(self.store, kind="triage")
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0].count, len(built.ids))
        self.assertIn(built.batch_id, receipts[0].summary)

    def test_a_deletion_writes_a_receipt(self) -> None:
        from cortex import suppression as sup

        memory_id = self.memory("delete me")
        sup.delete(self.store, memory_id, confirm=memory_id, reason="test")
        receipts = rc.recent(self.store, kind="deletion")
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0].ids, [memory_id])
        self.assertIn("edges", receipts[0].summary)

    def test_a_ladder_sweep_writes_one_receipt(self) -> None:
        from cortex import forget_ladder as ladder

        for index in range(3):
            self.memory(f"sweep {index}", days_old=200)
        planned, applied = ladder.sweep(self.store, at=NOW, apply_steps=True)
        receipts = rc.recent(self.store, kind="forget_ladder")
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0].count, sum(applied.values()))
        self.assertEqual(planned.counts(), applied)

    def test_a_dry_run_writes_no_receipt(self) -> None:
        from cortex import forget_ladder as ladder

        self.memory("dry", days_old=200)
        ladder.sweep(self.store, at=NOW, apply_steps=False)
        self.assertEqual(rc.recent(self.store), [])


if __name__ == "__main__":
    unittest.main()
