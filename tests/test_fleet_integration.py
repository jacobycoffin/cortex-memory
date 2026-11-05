"""End-to-end tests for the layered (multi-tenant) recall path.

These exercise the provider, not just ``fleet.py``: the point of the feature is
that a bot process reads two stores, writes only its own, and reaches the shared
layer exclusively through a promoted, audited record.
"""

from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import CortexMemoryProvider, fleet
from cortex.store import CortexStore


GENERAL_KNOWLEDGE = (
    "Hermes cron jobs deliver output to the channel recorded in the job's "
    "delivery block, and a job with no delivery block stays silent."
)
FINANCE_LEAK = "Chase card balance is 4,210.55 USD with a 19.99% APR."


class FleetIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "profiles" / "teknium"
        self.home.mkdir(parents=True)
        self.shared_path = Path(self.tmp.name) / "fleet" / "shared.db"
        self.shared_path.parent.mkdir(parents=True)
        self.providers: list[CortexMemoryProvider] = []

    def tearDown(self) -> None:
        for provider in self.providers:
            provider.shutdown()
        self.tmp.cleanup()

    # -- helpers -------------------------------------------------------------
    def make_provider(self, **overrides) -> CortexMemoryProvider:
        config = {
            "db_path": str(self.home / "cortex" / "cortex.db"),
            "retrieval_threshold": 0.0,
            "auto_capture": False,
            "memory_receipts": False,
            **overrides,
        }
        provider = CortexMemoryProvider(config)
        provider.initialize("session-1", hermes_home=self.home, agent_context="primary")
        self.providers.append(provider)
        return provider

    def seed_shared(self, content: str, *, tenant: str = "hermes", source_type: str = "fleet_promotion") -> str:
        store = CortexStore(self.shared_path)
        try:
            memory_id, _created = store.add_memory(
                content,
                kind="semantic",
                source_type=source_type,
                source_category="AGENT_INFERENCE",
                context_mode="standalone",
                scope={"layer": "shared", "origin_tenant": tenant},
                confidence=0.9,
                importance=0.7,
            )
            return str(memory_id)
        finally:
            store.close()

    def seed_local(self, provider: CortexMemoryProvider, content: str) -> str:
        memory_id, _created = provider._store.add_memory(
            content,
            kind="semantic",
            source_type="conversation",
            context_mode="standalone",
            confidence=0.9,
            importance=0.9,
        )
        return str(memory_id)

    def call(self, provider: CortexMemoryProvider, **args) -> dict:
        return json.loads(provider.handle_tool_call("cortex_memory", args))

    # -- disabled path -------------------------------------------------------
    def test_no_shared_path_keeps_the_single_store_behaviour(self) -> None:
        provider = self.make_provider()
        self.assertIsNone(provider._fleet)
        self.assertNotIsInstance(provider._retriever, fleet.FleetMemory)
        self.seed_local(provider, GENERAL_KNOWLEDGE)
        context = provider.prefetch("where does cron deliver output", session_id="session-1")
        self.assertIn("delivery block", context)
        self.assertNotIn("[shared from", context)

    # -- read path -----------------------------------------------------------
    def test_shared_hit_is_labelled_and_audited_in_the_bot_database(self) -> None:
        shared_id = self.seed_shared(GENERAL_KNOWLEDGE)
        provider = self.make_provider(shared_db_path=str(self.shared_path))
        self.assertIsNotNone(provider._fleet)
        self.assertIsInstance(provider._retriever, fleet.FleetMemory)

        context = provider.prefetch("where does a cron job deliver its output", session_id="s-1")
        self.assertIn("delivery block", context)
        self.assertIn("[shared from hermes", context)

        rows = provider._store._conn.execute(
            "SELECT memory_id, event FROM fleet_shared_usage"
        ).fetchall()
        self.assertEqual({str(row["memory_id"]) for row in rows}, {shared_id})

        # The local access log must not carry the shared id: it carries a
        # foreign key into this bot's memories table.
        leaked = provider._store._conn.execute(
            "SELECT COUNT(*) FROM access_log WHERE memory_id = ?", (shared_id,)
        ).fetchone()[0]
        self.assertEqual(int(leaked), 0)
        usage = provider._store._conn.execute(
            "SELECT COUNT(*) FROM usage_records WHERE memory_id = ?", (shared_id,)
        ).fetchone()[0]
        self.assertEqual(int(usage), 0)

    def test_shared_database_is_read_only_for_the_bot(self) -> None:
        self.seed_shared(GENERAL_KNOWLEDGE)
        provider = self.make_provider(shared_db_path=str(self.shared_path))
        shared_store = provider._fleet.shared_store
        self.assertIsNotNone(shared_store)
        with self.assertRaises(sqlite3.Error):
            shared_store._conn.execute(
                "INSERT INTO memories (id, content) VALUES ('x', 'y')"
            )

    def test_finance_rows_in_the_shared_layer_are_dropped_at_read_time(self) -> None:
        self.seed_shared(GENERAL_KNOWLEDGE)
        # A row that reached the file out of band (manual SQL, older policy).
        self.seed_shared(FINANCE_LEAK, tenant="ledger")
        provider = self.make_provider(shared_db_path=str(self.shared_path))
        context = provider.prefetch("card balance and APR and cron delivery", session_id="s-1")
        self.assertNotIn("4,210.55", context)

    def test_missing_shared_file_is_bootstrapped_then_read_only(self) -> None:
        missing = Path(self.tmp.name) / "new" / "shared.db"
        provider = self.make_provider(shared_db_path=str(missing))
        self.assertIsNotNone(provider._fleet)
        self.assertTrue(missing.exists())
        self.assertIsNotNone(provider._fleet.shared_store)
        with self.assertRaises(sqlite3.Error):
            provider._fleet.shared_store._conn.execute("DELETE FROM memories")

    # -- promotion path ------------------------------------------------------
    def test_bot_queues_a_request_and_operator_admits_it(self) -> None:
        provider = self.make_provider(
            shared_db_path=str(self.shared_path),
            shared_promotion_enabled=True,
        )
        memory_id = self.seed_local(provider, GENERAL_KNOWLEDGE)

        queued = self.call(provider, action="promote", memory_id=memory_id, reason="general")
        self.assertTrue(queued["success"], queued)
        self.assertEqual(queued["promotion"]["status"], "pending")
        request_id = queued["promotion"]["request_id"]

        # Queuing must not have written the shared file.
        shared_store = CortexStore(self.shared_path)
        try:
            before = shared_store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        finally:
            shared_store.close()
        self.assertEqual(int(before), 0)

        operator = fleet.FleetMemory(provider._fleet.config)
        try:
            receipt = operator.admit_request(
                request_id, actor="jacoby", reason="reusable fleet-wide fact"
            )
        finally:
            operator.close()
        self.assertTrue(receipt.promoted, receipt)
        self.assertEqual(receipt.origin_memory_id, memory_id)

        shared_store = CortexStore(self.shared_path)
        try:
            row = shared_store._conn.execute(
                "SELECT content, scope_json, source_type FROM memories"
            ).fetchone()
        finally:
            shared_store.close()
        self.assertIn("delivery block", str(row["content"]))
        self.assertEqual(str(row["source_type"]), "fleet_promotion")
        self.assertIn("teknium", str(row["scope_json"]))

        state = provider._fleet.promotions(status="admitted")
        self.assertEqual([str(item["request_id"]) for item in state], [request_id])

    def test_finance_memory_cannot_even_be_queued(self) -> None:
        provider = self.make_provider(
            shared_db_path=str(self.shared_path),
            shared_promotion_enabled=True,
        )
        memory_id = self.seed_local(provider, FINANCE_LEAK)
        denied = self.call(provider, action="promote", memory_id=memory_id, reason="share")
        self.assertFalse(denied["success"])
        self.assertIn("denied", denied["error"])

        rows = provider._store._conn.execute(
            "SELECT status, content FROM fleet_promotion_requests"
        ).fetchall()
        self.assertEqual([str(row["status"]) for row in rows], ["denied"])
        self.assertNotIn("4,210.55", str(rows[0]["content"]))

        shared_store = CortexStore(self.shared_path)
        try:
            count = shared_store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        finally:
            shared_store.close()
        self.assertEqual(int(count), 0)

    def test_promotion_is_off_unless_the_tenant_opts_in(self) -> None:
        provider = self.make_provider(shared_db_path=str(self.shared_path))
        memory_id = self.seed_local(provider, GENERAL_KNOWLEDGE)
        refused = self.call(provider, action="promote", memory_id=memory_id)
        self.assertFalse(refused["success"])
        self.assertIn("disabled", refused["error"])

    def test_promote_is_idempotent(self) -> None:
        provider = self.make_provider(
            shared_db_path=str(self.shared_path),
            shared_promotion_enabled=True,
        )
        memory_id = self.seed_local(provider, GENERAL_KNOWLEDGE)
        first = self.call(provider, action="promote", memory_id=memory_id)
        second = self.call(provider, action="promote", memory_id=memory_id)
        self.assertTrue(first["success"])
        self.assertTrue(second["promotion"]["duplicate"])
        self.assertEqual(first["promotion"]["request_id"], second["promotion"]["request_id"])

    # -- operator surfaces ---------------------------------------------------
    def test_fleet_tool_status_reports_the_resolved_config(self) -> None:
        provider = self.make_provider(
            shared_db_path=str(self.shared_path),
            shared_promotion_enabled=True,
        )
        status = self.call(provider, action="fleet")["fleet"]
        self.assertTrue(status["fleet_enabled"])
        self.assertEqual(status["tenant"], self.home.name)
        self.assertEqual(status["shared_db"], str(self.shared_path))

    def test_cli_admits_a_queued_request(self) -> None:
        provider = self.make_provider(
            shared_db_path=str(self.shared_path),
            shared_promotion_enabled=True,
        )
        memory_id = self.seed_local(provider, GENERAL_KNOWLEDGE)
        request_id = self.call(provider, action="promote", memory_id=memory_id)["promotion"][
            "request_id"
        ]

        # The operator CLI reads the deployed config, not the caller's dict.
        (self.home / "config.yaml").write_text(
            "plugins:\n"
            "  cortex:\n"
            f"    shared_db_path: {self.shared_path}\n"
            "    shared_promotion_enabled: true\n",
            encoding="utf-8",
        )

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = fleet._main(
                [
                    "--home",
                    str(self.home),
                    "admit",
                    request_id,
                    "--actor",
                    "jacoby",
                    "--reason",
                    "curated",
                ]
            )
        self.assertEqual(code, 0, buffer.getvalue())
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["status"], "promoted")
        shared_store = CortexStore(self.shared_path)
        try:
            count = shared_store._conn.execute(
                "SELECT COUNT(*) FROM fleet_promotions"
            ).fetchone()[0]
        finally:
            shared_store.close()
        self.assertEqual(int(count), 1)


if __name__ == "__main__":
    unittest.main()
