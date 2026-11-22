"""Compaction reclaims derivatives of retired memories and nothing else (#98)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import compaction
from cortex.store import CortexStore


class CompactionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str, *, state: str = "active") -> str:
        memory_id = self.store.add_memory(
            content=content,
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]
        self.store.log_access(memory_id, "used", query="q")
        if state != "active":
            self.store.set_state(memory_id, state, reason="fixture")
        return memory_id


class DiscoveryTests(CompactionTestCase):
    def test_audit_tables_are_never_compacted(self) -> None:
        self.memory("retired", state="archived")
        tables = compaction._tables(self.store)
        for protected in ("lifecycle_events", "memory_versions", "deletions", "prune_receipts"):
            self.assertNotIn(protected, tables, protected)

    def test_history_is_excluded_by_default(self) -> None:
        self.memory("retired", state="archived")
        self.assertNotIn("access_log", compaction._tables(self.store))

    def test_history_can_be_included_explicitly(self) -> None:
        self.memory("retired", state="archived")
        self.assertEqual(compaction.plan(self.store).counts.get("access_log"), None)
        planned = compaction.plan(self.store, include_history=True)
        self.assertEqual(planned.counts.get("access_log"), 1)


class PlanTests(CompactionTestCase):
    def test_a_plan_counts_only_retired_memories(self) -> None:
        self.memory("active one")
        self.memory("archived one", state="archived")
        planned = compaction.plan(self.store)
        self.assertEqual(planned.retired, 1)

    def test_a_plan_is_read_only(self) -> None:
        memory_id = self.memory("archived", state="archived")
        compaction.plan(self.store)
        self.assertIsNotNone(self.store.get_memory(memory_id))

    def test_an_active_memory_contributes_nothing(self) -> None:
        self.memory("active one")
        self.assertEqual(compaction.plan(self.store).rows, 0)

    def test_the_report_carries_the_file_size(self) -> None:
        self.memory("archived", state="archived")
        described = compaction.report(self.store)
        self.assertGreater(described["file_bytes"], 0)
        self.assertEqual(described["retired"], 1)


class CompactTests(CompactionTestCase):
    def test_compaction_removes_derivatives_of_retired_memories(self) -> None:
        memory_id = self.memory("retired", state="archived")
        removed = compaction.compact(self.store)
        self.assertGreaterEqual(removed.get("memory_presentations", 0), 0)
        rows = self.store._conn.execute(
            "SELECT COUNT(*) FROM memory_presentations WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0]
        self.assertEqual(rows, 0)

    def test_the_memory_and_its_content_survive(self) -> None:
        memory_id = self.memory("a fact worth keeping", state="archived")
        compaction.compact(self.store)
        row = self.store.get_memory(memory_id)
        self.assertIsNotNone(row)
        self.assertEqual(row["content"], "a fact worth keeping")

    def test_an_active_memory_keeps_its_derivatives(self) -> None:
        memory_id = self.memory("active one")
        compaction.compact(self.store)
        rows = self.store._conn.execute(
            "SELECT COUNT(*) FROM memory_features WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0]
        self.assertGreater(rows, 0)

    def test_audit_history_survives_compaction(self) -> None:
        memory_id = self.memory("retired", state="archived")
        before = self.store._conn.execute(
            "SELECT COUNT(*) FROM lifecycle_events WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0]
        compaction.compact(self.store)
        after = self.store._conn.execute(
            "SELECT COUNT(*) FROM lifecycle_events WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0]
        self.assertEqual(before, after)
        self.assertGreater(after, 0)

    def test_use_evidence_survives_unless_asked_for(self) -> None:
        memory_id = self.memory("retired", state="archived")
        compaction.compact(self.store)
        rows = self.store._conn.execute(
            "SELECT COUNT(*) FROM access_log WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0]
        self.assertEqual(rows, 1)
        compaction.compact(self.store, include_history=True)
        rows = self.store._conn.execute(
            "SELECT COUNT(*) FROM access_log WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0]
        self.assertEqual(rows, 0)

    def test_a_compaction_is_receipted(self) -> None:
        from cortex import prune_receipts as rc

        self.memory("retired", state="archived")
        compaction.compact(self.store)
        receipts = rc.recent(self.store, kind="compaction")
        self.assertEqual(len(receipts), 1)
        self.assertTrue(receipts[0].summary)

    def test_a_restored_memory_is_whole_again(self) -> None:
        """Compaction removes cache, not capability: the memory restores and is recallable."""
        from cortex.retrieval import MemoryRetriever
        from cortex.restore_ladder import restore

        memory_id = self.memory("the greenhouse runs on an ESP32", state="archived")
        compaction.compact(self.store)
        self.assertTrue(restore(self.store, memory_id).restored)
        results, _diagnostics = MemoryRetriever(self.store).search_detailed(
            "greenhouse ESP32", limit=3
        )
        self.assertIn(memory_id, [str(result.memory["id"]) for result in results])

    def test_compacting_twice_is_harmless(self) -> None:
        self.memory("retired", state="archived")
        compaction.compact(self.store)
        self.assertEqual(compaction.compact(self.store), {})


if __name__ == "__main__":
    unittest.main()
