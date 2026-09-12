"""Open-time repair passes must not run a full recompute on every open.

Regression coverage for the audit's efficiency finding: every store open
rebuilt feature stats and recomputed all deterministic neighborhood
memberships (O(n) per open), even though current write paths maintain both
incrementally and the neighborhood repair is only needed for legacy labels.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests._bootstrap import ROOT  # noqa: F401 - loads the flat ``cortex`` package

from cortex import store as store_module
from cortex.store import CortexStore

FEATURE_STATS_KEY = "feature_stats_backfill_version"


class OpenRepairGatingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "cortex.db"
        seed = CortexStore(self.db)
        seed.add_memory(
            "Synthetic gating sentinel about neighborhood repair behavior.",
            kind="procedure",
        )
        seed.close()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _reopen_counting_neighborhood_refresh(self) -> tuple[CortexStore, int]:
        calls: list[int] = []
        original = CortexStore._refresh_dynamic_neighborhoods_tx

        def counting(self_, conn):  # type: ignore[no-untyped-def]
            calls.append(1)
            return original(self_, conn)

        with mock.patch.object(CortexStore, "_refresh_dynamic_neighborhoods_tx", counting):
            store = CortexStore(self.db)
        return store, len(calls)

    def _reopen_counting_feature_stats_rebuild(self) -> tuple[CortexStore, int]:
        calls: list[int] = []
        original = store_module._rebuild_feature_stats

        def counting(conn):  # type: ignore[no-untyped-def]
            calls.append(1)
            return original(conn)

        with mock.patch.object(store_module, "_rebuild_feature_stats", counting):
            store = CortexStore(self.db)
        return store, len(calls)

    def test_reopen_skips_recompute_when_no_legacy_groupings(self) -> None:
        store, calls = self._reopen_counting_neighborhood_refresh()
        self.assertEqual(calls, 0)
        memberships = store._conn.execute(
            "SELECT COUNT(*) FROM memory_neighborhood_memberships"
        ).fetchone()[0]
        self.assertGreater(memberships, 0)
        store.close()

    def test_legacy_grouping_triggers_recompute_and_is_pruned(self) -> None:
        """A grouping the current evidence does not admit must still be repaired."""
        store = CortexStore(self.db)
        with store.transaction() as conn:
            conn.execute(
                """INSERT INTO memory_neighborhoods(
                     neighborhood_id,slug,label,category,parent_neighborhood_id,
                     description,safety_class,created_at
                   ) VALUES('neighborhood:service-atlas','service-atlas','Service: Atlas',
                            'service','neighborhood:services','legacy heuristic','normal',
                            '2026-07-16T00:00:00+00:00')"""
            )
        store.close()

        store, calls = self._reopen_counting_neighborhood_refresh()
        self.assertEqual(calls, 1)
        remaining = store._conn.execute(
            "SELECT COUNT(*) FROM memory_neighborhoods WHERE slug='service-atlas'"
        ).fetchone()[0]
        self.assertEqual(remaining, 0)
        store.close()

    def test_reopen_skips_recompute_for_admitted_groupings(self) -> None:
        """A grouping the evidence legitimately admits must not trigger a rebuild."""
        admitted_db = Path(self.tmp.name) / "admitted.db"
        store = CortexStore(admitted_db)
        for index in range(5):
            store.add_memory(
                f"Atlas migration step {index} keeps verified rollback notes.",
                scope={"project": "Atlas"},
                source_category="OPERATOR_APPROVED",
                storage_policy="operator_approved",
                session_id=f"context-{index % 2}",
            )
        store.close()

        calls: list[int] = []
        original = CortexStore._refresh_dynamic_neighborhoods_tx

        def counting(self_, conn):  # type: ignore[no-untyped-def]
            calls.append(1)
            return original(self_, conn)

        with mock.patch.object(CortexStore, "_refresh_dynamic_neighborhoods_tx", counting):
            reopened = CortexStore(admitted_db)
        slugs = {
            str(row["slug"])
            for row in reopened._conn.execute(
                "SELECT slug FROM memory_neighborhoods WHERE category IN ('project','service')"
            )
        }
        self.assertEqual(slugs, {"project-atlas"})
        self.assertEqual(len(calls), 0)
        reopened.close()

    def test_reopen_skips_feature_stats_rebuild_within_same_version(self) -> None:
        store, calls = self._reopen_counting_feature_stats_rebuild()
        self.assertEqual(calls, 0)
        stats = store._conn.execute("SELECT COUNT(*) FROM feature_stats").fetchone()[0]
        self.assertGreater(stats, 0)
        store.close()

    def test_feature_stats_rebuild_runs_after_version_bump(self) -> None:
        store = CortexStore(self.db)
        store._conn.execute("UPDATE meta SET value='stale-version' WHERE key=?", (FEATURE_STATS_KEY,))
        store._conn.commit()
        store.close()

        store, calls = self._reopen_counting_feature_stats_rebuild()
        self.assertEqual(calls, 1)
        marker = store._conn.execute(
            "SELECT value FROM meta WHERE key=?", (FEATURE_STATS_KEY,)
        ).fetchone()
        self.assertNotEqual(str(marker["value"]), "stale-version")
        store.close()


if __name__ == "__main__":
    unittest.main()