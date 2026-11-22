from __future__ import annotations

import gzip
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the repo as `cortex`)

from cortex import compaction
from cortex.store import CortexStore


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


class ArchiveCompactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db_path = self.root / "cortex.db"
        self.store = CortexStore(self.db_path)
        self.archived: list[str] = []
        for index in range(6):
            memory_id, _ = self.store.add_memory(
                f"Archived vault note {index} about service {index % 2}.",
                kind="operational",
                source_type="vault",
            )
            self.store.set_state(memory_id, "archived", reason="test fixture")
            self.archived.append(memory_id)
        self.live_id, _ = self.store.add_memory("An active memory that must survive compaction.", kind="identity")
        self.pinned_id, _ = self.store.add_memory("A pinned archived memory that must survive.", kind="preference")
        self.store.set_state(self.pinned_id, "archived", reason="test fixture")
        self.store.set_pinned(self.pinned_id, True)
        self.conn = _connect(self.db_path)

    def tearDown(self) -> None:
        self.conn.close()
        self.store.close()
        self.tmp.cleanup()

    def _ids(self, conn: sqlite3.Connection | None = None) -> set[str]:
        target = conn or self.conn
        return {str(row[0]) for row in target.execute("SELECT id FROM memories")}

    def test_dry_run_reports_plan_and_changes_nothing(self) -> None:
        before = self._ids()
        result = compaction.compact(self.conn, self.root / "cold", apply=False)
        self.assertFalse(result["applied"])
        self.assertEqual(result["candidates"], len(self.archived))
        self.assertEqual(self._ids(), before)
        self.assertFalse((self.root / "cold").exists())
        log = self.conn.execute("SELECT dry_run FROM maintenance_log ORDER BY id DESC LIMIT 1").fetchone()
        self.assertEqual(int(log[0]), 1)

    def test_compact_exports_then_summarises_and_restores_byte_for_byte(self) -> None:
        contents = {
            str(row["id"]): str(row["content"])
            for row in self.conn.execute("SELECT id, content FROM memories")
        }
        result = compaction.compact(self.conn, self.root / "cold", apply=True)
        self.assertTrue(result["applied"])
        self.assertEqual(result["deleted"], len(self.archived))
        self.assertEqual(result["rows"]["memories"], len(self.archived))

        remaining = self._ids()
        for memory_id in self.archived:
            self.assertNotIn(memory_id, remaining)
        self.assertIn(self.live_id, remaining)
        self.assertIn(self.pinned_id, remaining, "pinned archived rows are never compacted")

        summaries = [
            row for row in self.conn.execute(
                "SELECT id, content FROM memories WHERE source_type=?", (compaction.SUMMARY_SOURCE_TYPE,)
            )
        ]
        self.assertEqual(len(summaries), result["summaries"])
        self.assertGreaterEqual(len(summaries), 1)
        self.assertIn("cold storage", str(summaries[0]["content"]))

        manifest_path = Path(result["manifest"])
        manifest = json.loads(manifest_path.read_text())
        cold_path = manifest_path.parent / manifest["cold_file"]
        self.assertEqual(manifest["sha256"], compaction._file_sha256(cold_path))
        with gzip.open(cold_path, "rt", encoding="utf-8") as handle:
            exported = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(sum(1 for entry in exported if entry["table"] == "memories"), len(self.archived))

        # A second compaction must ignore the summaries it just wrote.
        second = compaction.compact(self.conn, self.root / "cold", apply=False)
        self.assertEqual(second["candidates"], 0)

        restored = compaction.restore(self.conn, manifest_path)
        self.assertEqual(restored["restored"]["memories"], len(self.archived))
        self.assertEqual(restored["summaries_removed"], len(summaries))
        after = self._ids()
        for memory_id in self.archived:
            self.assertIn(memory_id, after)
        for memory_id in self.archived:
            row = self.conn.execute("SELECT content, state FROM memories WHERE id=?", (memory_id,)).fetchone()
            self.assertEqual(str(row["content"]), contents[memory_id])
            self.assertEqual(str(row["state"]), "archived")
        self.assertFalse(
            [row for row in self.conn.execute("SELECT id FROM memories WHERE source_type=?", (compaction.SUMMARY_SOURCE_TYPE,))]
        )

    def test_active_and_protected_rows_are_never_candidates(self) -> None:
        protected_id, _ = self.store.add_memory("A protected archived memory.", kind="identity")
        self.store.set_state(protected_id, "archived", reason="test fixture")
        self.conn.execute("UPDATE memories SET protected=1 WHERE id=?", (protected_id,))
        self.conn.commit()
        ids = compaction.plan(self.conn)["ids"]
        self.assertNotIn(protected_id, ids)
        self.assertNotIn(self.live_id, ids)
        self.assertNotIn(self.pinned_id, ids)
        self.assertEqual(set(ids), set(self.archived))

    def test_delete_guard_spares_a_row_reactivated_mid_run(self) -> None:
        target = self.archived[0]
        self.conn.execute("UPDATE memories SET state='active' WHERE id=?", (target,))
        self.conn.commit()
        deleted, skipped = compaction._delete_originals(self.conn, [target])
        self.conn.commit()
        self.assertEqual(deleted, [])
        self.assertEqual(skipped, [target])
        self.assertIn(target, self._ids())

    def test_delete_skips_rows_held_by_a_restrict_reference(self) -> None:
        target = self.archived[1]
        self.conn.execute(
            """INSERT INTO memory_dependencies(memory_id, evidence_id, relation, created_at)
               VALUES(?,?,?,?)""",
            (self.archived[2], target, "depends_on", "2026-01-01T00:00:00+00:00"),
        )
        self.conn.commit()
        result = compaction.compact(self.conn, self.root / "cold", apply=True)
        self.assertEqual(result["skipped_restricted"], 1)
        self.assertIn(target, self._ids(), "a RESTRICT-held memory is skipped, never force-deleted")

    def test_restore_refuses_a_tampered_cold_file(self) -> None:
        result = compaction.compact(self.conn, self.root / "cold", apply=True)
        manifest_path = Path(result["manifest"])
        manifest = json.loads(manifest_path.read_text())
        cold_path = manifest_path.parent / manifest["cold_file"]
        cold_path.write_bytes(cold_path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            compaction.restore(self.conn, manifest_path)

    def test_restore_brings_back_edges_and_access_history(self) -> None:
        src, dst = self.archived[3], self.archived[4]
        self.conn.execute(
            "INSERT INTO edges(src_id,dst_id,relation,weight,evidence_count,created_at,last_reinforced_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (src, dst, "related", 0.5, 1, "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
        )
        self.conn.execute(
            "INSERT INTO access_log(memory_id,event,created_at) VALUES(?,?,?)",
            (src, "used", "2026-01-02T00:00:00+00:00"),
        )
        self.conn.commit()
        result = compaction.compact(self.conn, self.root / "cold", apply=True)
        self.assertIsNone(
            self.conn.execute("SELECT 1 FROM edges WHERE src_id=? AND dst_id=?", (src, dst)).fetchone()
        )
        compaction.restore(self.conn, Path(result["manifest"]))
        self.assertIsNotNone(
            self.conn.execute("SELECT 1 FROM edges WHERE src_id=? AND dst_id=?", (src, dst)).fetchone()
        )
        self.assertIsNotNone(
            self.conn.execute("SELECT 1 FROM access_log WHERE memory_id=? AND event='used'", (src,)).fetchone()
        )

    def test_older_than_days_filters_recent_archives(self) -> None:
        now = datetime.now(timezone.utc)
        aged, recent = self.archived[:3], self.archived[3:]
        self.conn.execute(
            "UPDATE memories SET updated_at=? WHERE id IN (%s)" % ",".join("?" for _ in aged),
            [(now - timedelta(days=400)).isoformat(), *aged],
        )
        self.conn.execute(
            "UPDATE memories SET updated_at=? WHERE id IN (%s)" % ",".join("?" for _ in recent),
            [now.isoformat(), *recent],
        )
        self.conn.commit()
        plan_one = compaction.plan(self.conn, older_than_days=1)
        self.assertEqual(set(plan_one["ids"]), set(aged), "rows updated today are inside the window")
        self.assertEqual(compaction.plan(self.conn, older_than_days=365)["candidates"], len(aged))

    def test_restore_preserves_supersedes_chains(self) -> None:
        older, newer = self.archived[5], self.archived[0]
        self.conn.execute("UPDATE memories SET supersedes_id=? WHERE id=?", (older, newer))
        self.conn.commit()
        result = compaction.compact(self.conn, self.root / "cold", apply=True)
        self.assertGreaterEqual(result["deleted"], 1)
        compaction.restore(self.conn, Path(result["manifest"]))
        row = self.conn.execute("SELECT supersedes_id FROM memories WHERE id=?", (newer,)).fetchone()
        self.assertEqual(str(row["supersedes_id"]), older)

    def test_restore_is_idempotent(self) -> None:
        src = self.archived[2]
        self.conn.execute(
            "INSERT INTO access_log(memory_id,event,created_at) VALUES(?,?,?)",
            (src, "used", "2026-01-03T00:00:00+00:00"),
        )
        self.conn.commit()
        before = self.conn.execute("SELECT count(*) FROM access_log").fetchone()[0]
        result = compaction.compact(self.conn, self.root / "cold", apply=True)
        manifest = Path(result["manifest"])
        compaction.restore(self.conn, manifest)
        once = self.conn.execute("SELECT count(*) FROM access_log").fetchone()[0]
        compaction.restore(self.conn, manifest)
        twice = self.conn.execute("SELECT count(*) FROM access_log").fetchone()[0]
        self.assertEqual(once, before)
        self.assertEqual(twice, before)


if __name__ == "__main__":
    unittest.main()
