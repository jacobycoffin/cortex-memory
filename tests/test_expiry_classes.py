"""Every fact carries its own review date (#64)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import expiry_classes as ec
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class ClassifyTests(unittest.TestCase):
    def test_permanent_kinds_are_never_reviewed(self) -> None:
        for kind in ("identity", "preference", "procedure", "commitment", "correction"):
            self.assertEqual(ec.classify(kind), ec.PERMANENT, kind)

    def test_short_lived_kinds_are_reviewed_in_thirty(self) -> None:
        for kind in ("episode", "operational", "tool"):
            self.assertEqual(ec.classify(kind), ec.REVIEW_30, kind)

    def test_ordinary_semantics_are_reviewed_in_ninety(self) -> None:
        self.assertEqual(ec.classify("semantic"), ec.REVIEW_90)

    def test_volatility_wins_over_kind(self) -> None:
        self.assertEqual(ec.classify("semantic", "ephemeral"), ec.REVIEW_30)
        self.assertEqual(ec.classify("semantic", "durable"), ec.PERMANENT)
        self.assertEqual(ec.classify("semantic", "seasonal"), ec.REVIEW_90)

    def test_a_permanent_kind_stays_permanent_under_volatility(self) -> None:
        """A preference is not made ephemeral by a stray marker."""
        self.assertEqual(ec.classify("preference", "ephemeral"), ec.PERMANENT)

    def test_an_unknown_kind_gets_the_middle(self) -> None:
        self.assertEqual(ec.classify("hiccup"), ec.REVIEW_90)

    def test_review_dates_match_the_class(self) -> None:
        self.assertEqual(ec.review_date(ec.REVIEW_30, at=NOW), NOW + timedelta(days=30))
        self.assertEqual(ec.review_date(ec.REVIEW_90, at=NOW), NOW + timedelta(days=90))
        self.assertIsNone(ec.review_date(ec.PERMANENT, at=NOW))


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def add(self, content: str, kind: str = "semantic") -> str:
        return self.store.add_memory(
            content=content, kind=kind, source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def test_assignment_bakes_the_class_and_date(self) -> None:
        memory_id = self.add("The dashboard runs on port 8100.")
        assigned = ec.assign(self.store, memory_id, kind="semantic", at=NOW)
        self.assertEqual(assigned.expiry_class, ec.REVIEW_90)
        self.assertEqual(assigned.review_after, (NOW + timedelta(days=90)).isoformat())

    def test_a_permanent_fact_has_no_review_date(self) -> None:
        memory_id = self.add("Jacoby prefers short answers.", kind="preference")
        assigned = ec.assign(self.store, memory_id, kind="preference", at=NOW)
        self.assertEqual(assigned.expiry_class, ec.PERMANENT)
        self.assertIsNone(assigned.review_after)

    def test_assignment_is_idempotent(self) -> None:
        memory_id = self.add("A fact.")
        ec.assign(self.store, memory_id, at=NOW)
        ec.assign(self.store, memory_id, at=NOW)
        self.assertEqual(ec.report(self.store, at=NOW)["assigned"], 1)

    def test_an_unassigned_memory_reports_nothing(self) -> None:
        self.assertIsNone(ec.of(self.store, "no-such-id"))

    def test_due_lists_only_past_review(self) -> None:
        soon = self.add("An ephemeral thing.", kind="episode")
        later = self.add("A durable thing.", kind="semantic")
        ec.assign(self.store, soon, kind="episode", at=NOW)
        ec.assign(self.store, later, kind="semantic", at=NOW)
        overdue = ec.due(self.store, at=NOW + timedelta(days=31))
        self.assertEqual([item["memory_id"] for item in overdue], [soon])

    def test_a_permanent_fact_is_never_due(self) -> None:
        memory_id = self.add("Jacoby prefers short answers.", kind="preference")
        ec.assign(self.store, memory_id, kind="preference", at=NOW)
        self.assertEqual(ec.due(self.store, at=NOW + timedelta(days=5000)), [])

    def test_acknowledging_pushes_the_date_out_without_deleting(self) -> None:
        memory_id = self.add("An ephemeral thing.", kind="episode")
        ec.assign(self.store, memory_id, kind="episode", at=NOW)
        self.assertTrue(ec.acknowledge(self.store, memory_id, at=NOW + timedelta(days=31)))
        self.assertEqual(ec.due(self.store, at=NOW + timedelta(days=40)), [])
        self.assertIsNotNone(self.store.get_memory(memory_id))

    def test_acknowledging_an_unassigned_memory_is_false(self) -> None:
        self.assertFalse(ec.acknowledge(self.store, "no-such-id"))

    def test_a_closed_memory_is_not_due(self) -> None:
        memory_id = self.add("An ephemeral thing.", kind="episode")
        ec.assign(self.store, memory_id, kind="episode", at=NOW)
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET state = 'archived' WHERE id = ?", (memory_id,))
        self.assertEqual(ec.due(self.store, at=NOW + timedelta(days=200)), [])

    def test_the_report_counts_the_overdue(self) -> None:
        memory_id = self.add("An ephemeral thing.", kind="episode")
        ec.assign(self.store, memory_id, kind="episode", at=NOW)
        described = ec.report(self.store, at=NOW + timedelta(days=31))
        self.assertEqual(described["due"], 1)
        self.assertEqual(described["by_class"][ec.REVIEW_30], 1)

    def test_a_fresh_store_reports_nothing_due(self) -> None:
        described = ec.report(self.store)
        self.assertEqual((described["assigned"], described["due"]), (0, 0))


if __name__ == "__main__":
    unittest.main()
