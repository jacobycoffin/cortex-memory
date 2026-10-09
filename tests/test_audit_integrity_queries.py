"""Regression tests for audit integrity anti-joins.

The audit uses NOT IN for indexed foreign-key lookups. FTS is intentionally
nullable, so its NULL-key rows must still count as orphaned.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401
from cortex.store import CortexStore


class AuditIntegrityTests(unittest.TestCase):
    def test_fts_null_and_orphan_ids_keep_left_join_semantics(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        store = CortexStore(Path(tmp.name) / "cortex.db")
        try:
            memory_id, _ = store.add_memory("A valid audit memory.")
            with store.transaction() as conn:
                conn.execute(
                    "INSERT INTO memory_fts(memory_id, content) VALUES(?, ?)",
                    (None, "null-key orphan"),
                )
                conn.execute(
                    "INSERT INTO memory_fts(memory_id, content) VALUES(?, ?)",
                    ("purged-memory-id", "purged orphan"),
                )
                conn.execute("DELETE FROM memory_fts WHERE memory_id=?", (memory_id,))

            audit = store.audit()
            self.assertEqual(audit["orphan_fts_rows"], 2)
            self.assertEqual(audit["missing_fts_rows"], 1)
        finally:
            store.close()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
