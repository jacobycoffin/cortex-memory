"""A new edge must earn two independent witnesses before it scores (#84)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import edge_quarantine as eq
from cortex.store import CortexStore

OLD = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.left = self.memory("The dashboard runs on port 8100.")
        self.right = self.memory("Nginx proxies the dashboard.")
        self.stamp = NOW

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def link(self, *, weight: float = 0.5, at: datetime | None = None,
             evidence: tuple[str, ...] = ()) -> None:
        self.store.add_edge(self.left, self.right, "related", weight=weight)
        if at is not None:
            with self.store.transaction() as conn:
                # add_edge stores the pair in canonical order, so age it from either end.
                conn.execute(
                    "UPDATE edges SET created_at = ? "
                    "WHERE (src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?)",
                    (at.isoformat(), self.left, self.right, self.right, self.left),
                )
        for evidence_type in evidence:
            self.witness(evidence_type)

    def witness(self, evidence_type: str) -> None:
        import uuid

        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO edge_evidence
                   (evidence_id, src_id, dst_id, relation, evidence_type, evidence_key, summary,
                    created_at)
                   VALUES (?, ?, ?, 'related', ?, ?, ?, ?)""",
                (str(uuid.uuid4()), self.left, self.right, evidence_type,
                 f"{evidence_type}:{uuid.uuid4()}", f"{evidence_type} witness",
                 self.stamp.isoformat()),
            )

    def test_without_an_epoch_everything_is_grandfathered(self) -> None:
        self.link()
        self.assertIsNone(eq.epoch(self.store))
        self.assertTrue(eq.status(self.store, self.left, self.right, "related").trusted)

    def test_an_edge_older_than_the_epoch_is_grandfathered(self) -> None:
        self.link(at=OLD)
        eq.adopt(self.store, at=NOW)
        verdict = eq.status(self.store, self.left, self.right, "related")
        self.assertTrue(verdict.trusted)
        self.assertIn("before the adoption epoch", verdict.reason)

    def test_a_new_edge_with_no_witness_is_quarantined(self) -> None:
        eq.adopt(self.store, at=NOW)
        self.link(at=NOW + timedelta(minutes=1))
        verdict = eq.status(self.store, self.left, self.right, "related")
        self.assertFalse(verdict.trusted)
        self.assertIn("0 of 2", verdict.reason)

    def test_a_second_independent_witness_releases_it(self) -> None:
        eq.adopt(self.store, at=NOW)
        self.link(at=NOW + timedelta(minutes=1))
        self.witness("explicit_wikilink")
        self.assertFalse(eq.status(self.store, self.left, self.right, "related").trusted)
        self.witness("co_retrieval")
        self.assertTrue(eq.status(self.store, self.left, self.right, "related").trusted)

    def test_two_rows_of_one_type_are_one_witness(self) -> None:
        eq.adopt(self.store, at=NOW)
        self.link(at=NOW + timedelta(minutes=1))
        self.witness("co_retrieval")
        self.witness("co_retrieval")
        verdict = eq.status(self.store, self.left, self.right, "related")
        self.assertFalse(verdict.trusted)
        self.assertEqual(verdict.witnesses, 1)

    def test_the_unattributed_legacy_row_is_not_a_witness(self) -> None:
        """add_edge writes it itself, and it says it preserves no evidence."""
        eq.adopt(self.store, at=NOW)
        self.link(at=NOW + timedelta(minutes=1))
        self.assertEqual(eq.witnesses(self.store, self.left, self.right, "related"), [])
        self.assertFalse(eq.status(self.store, self.left, self.right, "related").trusted)

    def test_a_person_can_release_a_link(self) -> None:
        eq.adopt(self.store, at=NOW)
        self.link(at=NOW + timedelta(minutes=1))
        self.assertFalse(eq.status(self.store, self.left, self.right, "related").trusted)
        self.assertTrue(eq.release(self.store, self.left, self.right, "related"))
        self.assertTrue(eq.status(self.store, self.left, self.right, "related").trusted)
        self.assertIn("released", eq.status(self.store, self.left, self.right, "related").reason)

    def test_the_release_is_recorded(self) -> None:
        eq.adopt(self.store, at=NOW)
        eq.release(self.store, self.left, self.right, "related", by="jacoby")
        row = self.store._conn.execute(
            "SELECT released_by FROM edge_quarantine WHERE key != '__epoch__'"
        ).fetchone()
        self.assertEqual(str(row["released_by"]), "jacoby")

    def test_a_released_edge_is_not_released_twice_by_accident(self) -> None:
        eq.adopt(self.store, at=NOW)
        eq.release(self.store, self.left, self.right, "related")
        self.assertEqual(eq.release(self.store, self.left, self.right, "related"), True)

    def test_scoring_edges_filters_the_list(self) -> None:
        eq.adopt(self.store, at=NOW)
        self.link(at=NOW + timedelta(minutes=1))
        rows = [{"src_id": self.left, "dst_id": self.right, "relation": "related",
                 "created_at": (NOW + timedelta(minutes=1)).isoformat()}]
        self.assertEqual(eq.scoring_edges(self.store, rows), [])
        self.witness("explicit_wikilink")
        self.witness("co_retrieval")
        self.assertEqual(len(eq.scoring_edges(self.store, rows)), 1)

    def test_the_witness_query_reads_either_direction(self) -> None:
        self.witness("explicit_wikilink")
        self.assertEqual(eq.witnesses(self.store, self.right, self.left, "related"),
                         ["explicit_wikilink"])

    def test_the_epoch_is_idempotent(self) -> None:
        first = eq.adopt(self.store, at=NOW)
        second = eq.adopt(self.store, at=NOW + timedelta(days=1))
        self.assertNotEqual(first, second)
        count = self.store._conn.execute(
            "SELECT COUNT(*) FROM edge_quarantine WHERE key = '__epoch__'"
        ).fetchone()[0]
        self.assertEqual(int(count), 1)

    def test_the_report_splits_grandfathered_from_candidates(self) -> None:
        self.link(at=OLD)
        eq.adopt(self.store, at=NOW)
        described = eq.report(self.store)
        self.assertEqual(described["edges"], 1)
        self.assertEqual(described["grandfathered"], 1)
        self.assertEqual(described["quarantined_candidates"], 0)

    def test_the_status_describes_itself(self) -> None:
        eq.adopt(self.store, at=NOW)
        self.link(at=NOW + timedelta(minutes=1))
        self.assertIn("quarantined",
                      eq.status(self.store, self.left, self.right, "related").describe())

    def test_an_empty_store_is_quiet(self) -> None:
        other = CortexStore(Path(self.tmp.name) / "empty.db")
        try:
            self.assertEqual(eq.report(other)["edges"], 0)
            self.assertTrue(eq.status(other, "a", "b", "related").trusted)
        finally:
            other.close()


if __name__ == "__main__":
    unittest.main()
