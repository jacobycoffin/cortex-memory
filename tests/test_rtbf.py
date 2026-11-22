"""Erasure removes the derivatives too, and proves it without keeping the data (#97)."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import legal_hold as lh
from cortex import rtbf
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class PurgeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str, *, kind: str = "semantic") -> str:
        return self.store.add_memory(
            content=content,
            kind=kind,
            source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def subject_data(self) -> tuple[str, str, str]:
        """A subject with a memory, an access log row, a usage record and an edge."""
        first = self.memory("Jordan Reyes signed the retainer on 2026-03-02.")
        second = self.memory("Notes: Jordan Reyes prefers invoices by email.")
        other = self.memory("The greenhouse controller runs on an ESP32.")
        self.store.log_access(first, "used", query="retainer for Jordan Reyes")
        self.store.add_edge(first, second, "relates_to", evidence_type="fixture")
        self.store.add_edge(second, other, "relates_to", evidence_type="fixture")
        return first, second, other


class DiscoveryTests(PurgeTestCase):
    def test_dependent_tables_are_discovered_not_hard_coded(self) -> None:
        self.subject_data()
        tables = rtbf.dependent_tables(self.store)
        self.assertIn("edges", tables)
        self.assertIn("access_log", tables)
        self.assertIn("memories_version" if "memories_version" in tables else "memory_versions",
                      tables)

    def test_edges_are_matched_on_both_ends(self) -> None:
        self.subject_data()
        self.assertEqual(sorted(rtbf.dependent_tables(self.store)["edges"]), ["dst_id", "src_id"])


class PreviewTests(PurgeTestCase):
    def test_a_preview_counts_the_memories_and_their_derivatives(self) -> None:
        first, second, _other = self.subject_data()
        planned = rtbf.preview(self.store, "Jordan Reyes")
        self.assertEqual(set(planned.memory_ids), {first, second})
        self.assertGreaterEqual(planned.counts.get("access_log", 0), 1)
        self.assertGreaterEqual(planned.counts.get("edges", 0), 2)

    def test_a_preview_changes_nothing(self) -> None:
        first, _second, _other = self.subject_data()
        rtbf.preview(self.store, "Jordan Reyes")
        self.assertIsNotNone(self.store.get_memory(first))

    def test_the_description_names_the_tables(self) -> None:
        self.subject_data()
        described = rtbf.preview(self.store, "Jordan Reyes").describe()
        self.assertIn("access_log", described)
        self.assertIn("memories", described)

    def test_an_unrelated_subject_matches_nothing(self) -> None:
        self.subject_data()
        self.assertEqual(rtbf.preview(self.store, "Nobody At All").memory_ids, [])


class PurgeTests(PurgeTestCase):
    def test_erasure_requires_the_subject_twice(self) -> None:
        self.subject_data()
        with self.assertRaises(rtbf.PurgeRefused):
            rtbf.purge(self.store, "Jordan Reyes", confirm="someone else")
        self.assertEqual(len(rtbf.preview(self.store, "Jordan Reyes").memory_ids), 2)

    def test_erasure_removes_the_memories_and_their_derivatives(self) -> None:
        first, second, other = self.subject_data()
        result = rtbf.purge(self.store, "Jordan Reyes", confirm="Jordan Reyes", at=NOW)
        self.assertEqual(result["matched"], 2)
        self.assertIsNone(self.store.get_memory(first))
        self.assertIsNone(self.store.get_memory(second))
        self.assertIsNotNone(self.store.get_memory(other), "unrelated memory must survive")
        self.assertGreater(result["purged"].get("access_log", 0), 0)
        self.assertGreater(result["purged"].get("edges", 0), 0)

    def test_the_derived_rows_are_actually_gone_(self) -> None:
        first, _second, _other = self.subject_data()
        rtbf.purge(self.store, "Jordan Reyes", confirm="Jordan Reyes", at=NOW)
        remaining = self.store._conn.execute(
            "SELECT COUNT(*) FROM access_log WHERE memory_id = ?", (first,)
        ).fetchone()[0]
        self.assertEqual(remaining, 0)
        dangling = self.store._conn.execute(
            "SELECT COUNT(*) FROM edges WHERE src_id = ? OR dst_id = ?", (first, first)
        ).fetchone()[0]
        self.assertEqual(dangling, 0)

    def test_a_certificate_proves_erasure_without_the_name(self) -> None:
        self.subject_data()
        result = rtbf.purge(self.store, "Jordan Reyes", confirm="Jordan Reyes", at=NOW)
        listed = rtbf.certificates(self.store)
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["certificate"], result["certificate"])
        self.assertEqual(listed[0]["purged"]["memories"], 2)
        blob = json.dumps(listed[0])
        self.assertNotIn("Jordan", blob)
        self.assertNotIn("Reyes", blob)

    def test_fulfilment_can_be_checked_later_from_the_hash(self) -> None:
        self.subject_data()
        rtbf.purge(self.store, "Jordan Reyes", confirm="Jordan Reyes", at=NOW)
        self.assertTrue(rtbf.fulfilled(self.store, "jordan reyes"))
        self.assertFalse(rtbf.fulfilled(self.store, "Someone Else"))

    def test_a_legal_hold_blocks_the_erasure_entirely(self) -> None:
        first, _second, _other = self.subject_data()
        lh.place(self.store, [first], reason="contract")
        with self.assertRaises(rtbf.PurgeRefused) as caught:
            rtbf.purge(self.store, "Jordan Reyes", confirm="Jordan Reyes")
        self.assertIn("legal hold", str(caught.exception))
        self.assertIsNotNone(self.store.get_memory(first))
        self.assertEqual(rtbf.certificates(self.store), [])

    def test_an_empty_subject_still_issues_a_certificate(self) -> None:
        result = rtbf.purge(self.store, "Nobody At All", confirm="Nobody At All", at=NOW)
        self.assertEqual(result["matched"], 0)
        self.assertEqual(len(rtbf.certificates(self.store)), 1)

    def test_a_second_purge_finds_nothing_left(self) -> None:
        self.subject_data()
        rtbf.purge(self.store, "Jordan Reyes", confirm="Jordan Reyes", at=NOW)
        again = rtbf.purge(self.store, "Jordan Reyes", confirm="Jordan Reyes", at=NOW)
        self.assertEqual(again["matched"], 0)
        self.assertEqual(again["rows"], 0)

    def test_the_note_is_recorded_with_the_certificate(self) -> None:
        self.subject_data()
        rtbf.purge(self.store, "Jordan Reyes", confirm="Jordan Reyes",
                   note="ticket #4512", at=NOW)
        self.assertEqual(rtbf.certificates(self.store)[0]["note"], "ticket #4512")


if __name__ == "__main__":
    unittest.main()
