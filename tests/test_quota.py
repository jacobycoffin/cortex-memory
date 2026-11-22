"""The quota sweeps instead of refusing writes (#101)."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import quota
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class QuotaTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def fill(self, count: int, *, days_old: float = 200, importance: float = 0.4) -> None:
        import uuid

        rows = [
            (
                str(uuid.uuid4()),
                f"Filler memory {index}",
                f"hash-{index}",
                "semantic",
                "active",
                importance,
                0,
                (NOW - timedelta(days=days_old)).isoformat(),
                (NOW - timedelta(days=days_old)).isoformat(),
                (NOW - timedelta(days=days_old)).isoformat(),
            )
            for index in range(count)
        ]
        with self.store.transaction() as conn:
            conn.executemany(
                "INSERT INTO memories(id, content, content_hash, kind, state, importance, "
                "pinned, created_at, updated_at, observed_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                rows,
            )


class StatusTests(QuotaTestCase):
    def test_an_empty_store_is_ok(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_MAX_ACTIVE_MEMORIES": "100"}):
            self.assertEqual(quota.status(self.store).band, "ok")

    def test_the_warn_band_starts_at_eighty_five_percent(self) -> None:
        self.fill(86)
        with mock.patch.dict(os.environ, {"CORTEX_MAX_ACTIVE_MEMORIES": "100"}):
            self.assertEqual(quota.status(self.store).band, "warn")

    def test_a_full_store_says_so(self) -> None:
        self.fill(100)
        with mock.patch.dict(os.environ, {"CORTEX_MAX_ACTIVE_MEMORIES": "100"}):
            self.assertEqual(quota.status(self.store).band, "full")

    def test_the_limit_is_configurable_and_defaults_sanely(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CORTEX_MAX_ACTIVE_MEMORIES", None)
            self.assertEqual(quota.limit(), quota.DEFAULT_LIMIT)
        for bad in ("", "lots", "0", "-5"):
            with mock.patch.dict(os.environ, {"CORTEX_MAX_ACTIVE_MEMORIES": bad}):
                self.assertEqual(quota.limit(), quota.DEFAULT_LIMIT, bad)

    def test_archived_memories_do_not_count(self) -> None:
        self.fill(10)
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET state = 'archived'")
        with mock.patch.dict(os.environ, {"CORTEX_MAX_ACTIVE_MEMORIES": "10"}):
            self.assertEqual(quota.status(self.store).active, 0)

    def test_check_reports_without_changing_anything(self) -> None:
        self.fill(5)
        with mock.patch.dict(os.environ, {"CORTEX_MAX_ACTIVE_MEMORIES": "100"}):
            described = quota.check(self.store)
        self.assertEqual(described["active"], 5)
        self.assertFalse(described["would_sweep"])


class SweepTests(QuotaTestCase):
    def test_no_sweep_inside_the_quota(self) -> None:
        self.fill(5)
        with mock.patch.dict(os.environ, {"CORTEX_MAX_ACTIVE_MEMORIES": "100"}):
            result = quota.ensure(self.store, at=NOW)
        self.assertIn("no sweep", result.summary)
        self.assertEqual(result.reclaimed, 0)

    def test_a_full_store_is_swept_and_reclaimed(self) -> None:
        self.fill(100)
        with mock.patch.dict(os.environ, {"CORTEX_MAX_ACTIVE_MEMORIES": "100"}):
            result = quota.ensure(self.store, at=NOW)
            after = quota.status(self.store)
        self.assertEqual(result.before.band, "full")
        self.assertGreater(result.reclaimed, 0)
        self.assertLess(after.active, result.before.active)

    def test_the_sweep_goes_through_the_ladder_not_a_new_mechanism(self) -> None:
        self.fill(100)
        with mock.patch.dict(os.environ, {"CORTEX_MAX_ACTIVE_MEMORIES": "100"}):
            result = quota.ensure(self.store, at=NOW)
        self.assertTrue(result.walked, "the ladder should have walked memories")

    def test_a_pinned_memory_survives_the_sweep(self) -> None:
        self.fill(100)
        pinned = self.store._conn.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
        self.store.set_pinned(str(pinned), True)
        with mock.patch.dict(os.environ, {"CORTEX_MAX_ACTIVE_MEMORIES": "100"}):
            quota.ensure(self.store, at=NOW)
        self.assertEqual(self.store.get_memory(str(pinned))["state"], "active")

    def test_a_dry_run_reclaims_nothing(self) -> None:
        self.fill(100)
        with mock.patch.dict(os.environ, {"CORTEX_MAX_ACTIVE_MEMORIES": "100"}):
            result = quota.ensure(self.store, apply=False, at=NOW)
            self.assertEqual(quota.status(self.store).band, "full")
        self.assertEqual(result.reclaimed, 0)

    def test_the_sweep_never_raises_on_an_empty_store(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_MAX_ACTIVE_MEMORIES": "1"}):
            result = quota.ensure(self.store, at=NOW)
        self.assertEqual(result.before.active, 0)


class ProviderWiringTests(QuotaTestCase):
    def test_a_capture_still_succeeds_when_the_store_is_full(self) -> None:
        """The whole point: the quota tidies, it never refuses the write."""
        import json

        provider_module = __import__("cortex.hermes_provider", fromlist=["CortexMemoryProvider"])
        provider = provider_module.CortexMemoryProvider(
            {"token_budget": 700, "top_k": 6, "database_path": str(Path(self.tmp.name) / "p.db")}
        )
        provider.initialize("quota", hermes_home=self.tmp.name, agent_context="primary")
        try:
            with mock.patch.dict(os.environ, {"CORTEX_MAX_ACTIVE_MEMORIES": "1"}):
                reply = json.loads(
                    provider.handle_tool_call(
                        "cortex_memory",
                        {"action": "remember", "content": "A capture that must not be refused."},
                    )
                )
            self.assertTrue(reply.get("success"), reply)
        finally:
            provider._store.close()


if __name__ == "__main__":
    unittest.main()
