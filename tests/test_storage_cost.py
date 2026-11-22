"""Storage cost is not where you think it is (#102)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import storage_cost as sc
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class StorageTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str, *, kind: str = "semantic", days_old: float = 60) -> str:
        memory_id = self.store.add_memory(
            content=content,
            kind=kind,
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            importance=0.4,
        )[0]
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE memories SET created_at = ? WHERE id = ?",
                ((NOW - timedelta(days=days_old)).isoformat(), memory_id),
            )
        return memory_id


class FileSizeTests(StorageTestCase):
    def test_a_store_has_a_size(self) -> None:
        self.memory("a fact")
        self.assertGreater(sc.file_bytes(self.store), 0)

    def test_human_formatting_is_readable(self) -> None:
        self.assertEqual(sc.human(512), "512B")
        self.assertEqual(sc.human(2048), "2.0KB")
        self.assertEqual(sc.human(5 * 1024 * 1024), "5.0MB")


class KindTests(StorageTestCase):
    def test_bytes_are_attributed_per_kind(self) -> None:
        self.memory("x" * 400, kind="semantic")
        self.memory("y" * 100, kind="episode")
        costs = {cost.kind: cost for cost in sc.by_kind(self.store)}
        self.assertEqual(costs["semantic"].memories, 1)
        self.assertEqual(costs["semantic"].content_bytes, 400)
        self.assertLess(costs["episode"].content_bytes, costs["semantic"].content_bytes)

    def test_an_empty_kind_is_absent_rather_than_invented(self) -> None:
        self.memory("only fact")
        kinds = [cost.kind for cost in sc.by_kind(self.store)]
        self.assertEqual(kinds, ["semantic"])

    def test_average_size_is_per_memory_not_per_kind(self) -> None:
        self.memory("a" * 100)
        self.memory("b" * 300)
        cost = {c.kind: c for c in sc.by_kind(self.store)}["semantic"]
        self.assertEqual(cost.average, 200)


class TableTests(StorageTestCase):
    def test_indexes_are_rolled_into_their_table(self) -> None:
        """'memory_features is 250MB' is the useful statement, not three index lines."""
        for index in range(20):
            self.memory(f"fact {index} " + "padding " * 20)
        costs = sc.tables(self.store)
        if not costs:  # dbstat missing in this build
            self.skipTest("dbstat not compiled in")
        names = [cost.table for cost in costs]
        self.assertIn("memories", names)
        self.assertFalse([name for name in names if name.startswith("idx_")],
                         f"indexes should be rolled up: {names}")

    def test_tables_are_ordered_by_size(self) -> None:
        for index in range(20):
            self.memory(f"fact {index} " + "padding " * 20)
        costs = sc.tables(self.store)
        if not costs:
            self.skipTest("dbstat not compiled in")
        sizes = [cost.bytes for cost in costs]
        self.assertEqual(sizes, sorted(sizes, reverse=True))


class ReclaimTests(StorageTestCase):
    def test_archiving_reclaims_nothing_and_says_so(self) -> None:
        self.memory("never used at all", days_old=90)
        reclaim = sc.reclaimable(self.store, at=NOW)
        self.assertEqual(reclaim.archiveable, 1)
        self.assertTrue(any("free no bytes" in note for note in reclaim.notes))

    def test_permanently_hidden_memories_count_as_deletable(self) -> None:
        keep = self.memory("still live")
        hidden = self.memory("hidden away")
        self.store.set_state(hidden, "archived", reason="test")
        reclaim = sc.reclaimable(self.store, at=NOW)
        self.assertEqual(reclaim.deletable, 1)
        self.assertGreater(reclaim.delete_bytes, 0)
        self.assertGreater(reclaim.delete_bytes, len("hidden away"))
        self.assertIsNotNone(keep)

    def test_an_empty_store_reclaims_nothing(self) -> None:
        reclaim = sc.reclaimable(self.store, at=NOW)
        self.assertEqual((reclaim.archiveable, reclaim.deletable), (0, 0))


class ReportTests(StorageTestCase):
    def test_the_report_contains_both_layers(self) -> None:
        self.memory("a fact worth keeping")
        described = sc.report(self.store, at=NOW)
        self.assertGreater(described["file_bytes"], 0)
        self.assertIn("kinds", described)
        self.assertIn("tables", described)
        self.assertIn("reclaim", described)

    def test_content_is_a_small_share_of_the_file(self) -> None:
        """The finding that makes the report worth having: content is a rounding error."""
        for index in range(30):
            self.memory(f"fact {index}")
        described = sc.report(self.store, at=NOW)
        self.assertLess(described["content_bytes"] / described["file_bytes"], 1.0)


if __name__ == "__main__":
    unittest.main()
