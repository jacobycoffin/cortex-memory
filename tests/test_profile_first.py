"""Profile-first serving: the profile is a read, not a search (#121)."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.core_block import build
from cortex.store import CortexStore

BUDGET_MS = 100.0


class ProfileCostTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        # Raw inserts in one transaction: this test is about the cost of *reading* a big store
        # with plain SQL, and `add_memory` (2,500 × ~130ms of FTS upkeep) would spend five
        # minutes building a fixture that the profile never touches.
        import uuid as _uuid

        rows = [
            (
                str(_uuid.uuid4()),
                f"Fact number {index} about the homelab, its services and their ports.",
                f"hash-{index}",
                "semantic",
                "active",
                (index % 100) / 100,
                1 if index % 200 == 0 else 0,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            )
            for index in range(12000)
        ]
        with self.store.transaction() as conn:
            conn.executemany(
                "INSERT INTO memories(id, content, content_hash, kind, state, importance, "
                "pinned, created_at, updated_at, observed_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                rows,
            )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_rendering_the_profile_is_a_read_not_a_search(self) -> None:
        build(self.store)  # warm the page cache the way a running process would be
        started = time.perf_counter()
        block = build(self.store)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.assertGreater(block.chars, 0)
        self.assertLess(
            elapsed_ms,
            BUDGET_MS,
            f"profile render took {elapsed_ms:.1f}ms on a 2,500-memory store (budget {BUDGET_MS}ms)",
        )

    def test_the_candidate_pool_is_bounded(self) -> None:
        """An unbounded ORDER BY over the whole store was 300ms; the block renders ~10 lines.

        Asserted on row count rather than wall-clock so it cannot pass on a fast machine.
        """
        import cortex.core_block as module

        sql = module._IMPORTANT_SQL.format(
            recent_exclusion="", core_join="", verified=module._VERIFIED_NO_CORE
        )
        rows = self.store._conn.execute(sql).fetchall()
        self.assertLessEqual(len(rows), 200, f"candidate pool is unbounded: {len(rows)} rows")
        self.assertIn("LIMIT", module._RECENT_SQL)


class ProfileFirstWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        provider_module = __import__(
            "cortex.hermes_provider", fromlist=["CortexMemoryProvider"]
        )
        self.provider = provider_module.CortexMemoryProvider(
            {"token_budget": 700, "top_k": 6, "database_path": str(Path(self.tmp.name) / "p.db")}
        )
        self.provider.initialize("profile", hermes_home=self.tmp.name, agent_context="primary")
        self.store: CortexStore = self.provider._store
        self.store.add_memory(
            content="Pinned rule: never change the release channel.",
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            pinned=True,
        )

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_the_profile_is_rendered_before_recall(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_CORE_MEMORY": "1"}):
            self.provider.prefetch("what is the release channel rule", session_id="profile")
            self.assertTrue(getattr(self.provider, "_core_text", ""), "profile was not prepared")

    def test_the_profile_survives_an_empty_recall(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_CORE_MEMORY": "1"}):
            context = self.provider.prefetch("zzzz no such memory anywhere", session_id="profile")
            self.assertIn("never change the release channel", context)

    def test_the_profile_survives_a_spent_latency_budget(self) -> None:
        """A slow turn must not silently drop the always-on block."""
        self.provider._retriever.latency_budget_ms = 0.0001
        with mock.patch.dict(os.environ, {"CORTEX_CORE_MEMORY": "1"}):
            context = self.provider.prefetch("what is the release channel rule", session_id="profile")
            self.assertIn("never change the release channel", context)

    def test_nothing_changes_when_the_block_is_off(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CORTEX_CORE_MEMORY", None)
            context = self.provider.prefetch("what is the release channel rule", session_id="profile")
            self.assertNotIn("Core memory", context)
            self.assertEqual(getattr(self.provider, "_core_text", ""), "")


if __name__ == "__main__":
    unittest.main()
