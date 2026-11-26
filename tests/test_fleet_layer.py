"""The fleet memory layer (#fleet): private by default, promotion gated.

These tests pin the two properties that make the layer safe to ship:

* with no ``shared_db_path`` configured nothing changes — the provider keeps one
  store, the same retriever and a plain revision, so recall behaves exactly as it
  did before the layer existed;
* a row reaches the shared store only through an explicit :class:`Approval`, and
  finance-derived content is refused outright rather than redacted.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import fleet
from cortex.retrieval import MemoryRetriever
from cortex.store import CortexStore

FINANCE_TEXT = "The card balance is 4,321.09 and the minimum payment is due on the 14th."
SHAREABLE_TEXT = "The Cortex dashboard listens on port 8088 behind the local proxy."


class _FleetCase(unittest.TestCase):
    """Temporary local + shared databases, no process-wide state."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._fleets: list[fleet.FleetMemory] = []

    def tearDown(self) -> None:
        for memory in self._fleets:
            memory.close()
        self.tmp.cleanup()

    def fleet_memory(self, tenant: str = "alpha", **overrides) -> fleet.FleetMemory:
        config = fleet.resolve_fleet_config(
            hermes_home=self.root,
            tenant=tenant,
            plugin_config={
                "db_path": str(self.root / f"{tenant}.db"),
                "shared_db_path": str(self.root / "shared.db"),
                "shared_promotion_enabled": True,
                **overrides,
            },
        )
        self.assertTrue(config.fleet_enabled)
        memory = fleet.FleetMemory(config)
        self._fleets.append(memory)
        return memory

    @staticmethod
    def add_local(memory: fleet.FleetMemory, content: str, **kwargs) -> str:
        created = memory.local_store.add_memory(
            content=content,
            kind=kwargs.pop("kind", "semantic"),
            source_type=kwargs.pop("source_type", "user_turn"),
            source_category=kwargs.pop("source_category", "USER_EXPLICIT"),
            **kwargs,
        )
        return created[0] if isinstance(created, tuple) else created


class FleetDisabledTests(unittest.TestCase):
    def test_no_shared_path_means_fleet_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = fleet.resolve_fleet_config(
                hermes_home=tmp,
                tenant="alpha",
                plugin_config={"db_path": str(Path(tmp) / "local.db")},
            )
        self.assertFalse(config.fleet_enabled)
        self.assertIsNone(config.shared_db)
        self.assertIn("shared_db_path", config.disabled_reason)

    def test_provider_keeps_one_store_when_fleet_is_unconfigured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _provider(tmp)
            try:
                self.assertIsNone(provider._fleet)
                self.assertIsInstance(provider._retriever, MemoryRetriever)
                revision = provider._recall_revision()
                self.assertEqual(revision, provider._store.retrieval_revision())
                self.assertNotIsInstance(revision[0], tuple)
            finally:
                provider.shutdown()

    def test_recall_with_fleet_off_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _provider(tmp)
            try:
                provider._store.add_memory(
                    content=SHAREABLE_TEXT,
                    kind="operational",
                    source_type="user_turn",
                    source_category="USER_EXPLICIT",
                )
                context = provider.prefetch("which port does the dashboard use")
                self.assertIn("8088", context)
                self.assertNotIn("[shared from", context)
            finally:
                provider.shutdown()

    def test_fleet_tool_is_a_no_op_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _provider(tmp)
            try:
                payload = json.loads(
                    provider.handle_tool_call("cortex_memory", {"action": "fleet"})
                )
                self.assertIn("error", payload)
                self.assertIn("not enabled", payload["error"])
            finally:
                provider.shutdown()


class PromotionApprovalTests(_FleetCase):
    def test_promotion_without_approval_is_refused(self) -> None:
        memory = self.fleet_memory()
        memory_id = self.add_local(memory, SHAREABLE_TEXT)
        receipt = memory.promote(memory_id, approval=None)
        self.assertEqual(receipt.status, "denied")
        self.assertIn("approval_required", receipt.reasons)
        self.assertEqual(memory.shared_store.stats()["memories"], 0)

    def test_incomplete_approval_is_refused(self) -> None:
        memory = self.fleet_memory()
        memory_id = self.add_local(memory, SHAREABLE_TEXT)
        receipt = memory.promote(memory_id, approval=fleet.Approval(actor="", reason="looks fine"))
        self.assertEqual(receipt.status, "denied")
        self.assertIn("approval_required", receipt.reasons)

    def test_a_memory_cannot_approve_itself(self) -> None:
        memory = self.fleet_memory()
        memory_id = self.add_local(memory, SHAREABLE_TEXT)
        receipt = memory.promote(
            memory_id, approval=fleet.Approval(actor=memory_id, reason="self-approved")
        )
        self.assertEqual(receipt.status, "denied")
        self.assertIn("self_approval_refused", receipt.reasons)

        machine_receipt = memory.promote(
            memory_id, approval=fleet.Approval(actor="system", reason="automatic")
        )
        self.assertEqual(machine_receipt.status, "denied")
        self.assertIn("self_approval_refused", machine_receipt.reasons)
        self.assertEqual(memory.shared_store.stats()["memories"], 0)

    def test_finance_content_is_refused_not_redacted(self) -> None:
        memory = self.fleet_memory()
        memory_id = self.add_local(memory, FINANCE_TEXT)
        receipt = memory.promote(
            memory_id,
            approval=fleet.Approval(actor="jacoby", reason="general budgeting habit"),
        )
        self.assertEqual(receipt.status, "denied")
        self.assertTrue(any(reason.startswith("hard_deny:") for reason in receipt.reasons))
        self.assertEqual(memory.shared_store.stats()["memories"], 0)
        shared_bytes = (self.root / "shared.db").read_bytes()
        self.assertNotIn(b"4,321.09", shared_bytes)
        self.assertNotIn(b"minimum payment", shared_bytes)

    def test_soft_identifiers_are_redacted_and_reported(self) -> None:
        memory = self.fleet_memory()
        memory_id = self.add_local(memory, "Ask ops@example.com for the proxy allowlist.")
        receipt = memory.promote(
            memory_id, approval=fleet.Approval(actor="jacoby", reason="general procedure")
        )
        self.assertEqual(receipt.status, "promoted")
        self.assertIn("email", receipt.redactions)
        shared = memory.shared_store.get_memory(receipt.shared_memory_id)
        self.assertIn("[email redacted]", shared["content"])
        self.assertNotIn("ops@example.com", shared["content"])

    def test_valid_approval_promotes_into_the_shared_store(self) -> None:
        memory = self.fleet_memory("alpha")
        memory_id = self.add_local(memory, SHAREABLE_TEXT)
        receipt = memory.promote(
            memory_id, approval=fleet.Approval(actor="jacoby", reason="general knowledge")
        )
        self.assertEqual(receipt.status, "promoted", receipt.reasons)
        shared = memory.shared_store.get_memory(receipt.shared_memory_id)
        self.assertEqual(shared["content"], SHAREABLE_TEXT)
        self.assertEqual(shared["source_type"], fleet.SHARED_SOURCE_TYPE)
        self.assertEqual(shared["approval_state"], fleet.SHARED_APPROVAL_STATE)

        other = self.fleet_memory("beta")
        hits = other.search("dashboard port proxy", limit=6, token_budget=400)
        shared_hits = [hit for hit in hits if getattr(hit, "layer", "local") == "shared"]
        self.assertTrue(shared_hits, "a promoted row must be recallable by another tenant")
        self.assertEqual(shared_hits[0].origin_tenant, "alpha")
        # Private rows never cross tenants.
        self.assertEqual(other.local_store.stats()["memories"], 0)


class ToolSchemaTests(unittest.TestCase):
    def test_schema_offers_the_fleet_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _provider(tmp)
            try:
                schemas = provider.get_tool_schemas()
                actions = schemas[0]["parameters"]["properties"]["action"]["enum"]
                self.assertIn("fleet", actions)
                self.assertIn("promote", actions)
            finally:
                provider.shutdown()


def _provider(tmp: str):
    provider_module = __import__("cortex.hermes_provider", fromlist=["CortexMemoryProvider"])
    provider = provider_module.CortexMemoryProvider({"db_path": str(Path(tmp) / "provider.db")})
    provider.initialize("fleet-off", hermes_home=tmp, agent_context="primary")
    return provider


class FleetEnabledProviderTests(unittest.TestCase):
    """A configured profile swaps the retriever and offers the fleet tool."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.shared = self.root / "shared.db"
        self.providers: list[object] = []

    def tearDown(self) -> None:
        for provider in self.providers:
            provider.shutdown()
        self.tmp.cleanup()

    def _provider(self, tenant: str):
        home = self.root / "profiles" / tenant
        home.mkdir(parents=True, exist_ok=True)
        provider_module = __import__(
            "cortex.hermes_provider", fromlist=["CortexMemoryProvider"]
        )
        # Explicit config mirrors a profile config block; this keeps the test
        # independent of whether a YAML reader is installed in the test env.
        provider = provider_module.CortexMemoryProvider(
            {
                "db_path": str(home / "cortex.db"),
                "shared_db_path": str(self.shared),
                "shared_promotion_enabled": True,
            }
        )
        provider.initialize("fleet-on", hermes_home=str(home), agent_context="primary")
        self.providers.append(provider)
        return provider

    def test_configured_profile_layers_recall_and_offers_status(self) -> None:
        alpha = self._provider("alpha")
        self.assertIsInstance(alpha._retriever, fleet.FleetMemory)
        self.assertEqual(alpha._fleet.config.tenant, "alpha")
        revision = alpha._recall_revision()
        self.assertEqual(len(revision), 2)
        self.assertIsInstance(revision[0], tuple)

        created = alpha._store.add_memory(
            content=SHAREABLE_TEXT,
            kind="operational",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
        )
        memory_id = created[0] if isinstance(created, tuple) else created
        queued = json.loads(
            alpha.handle_tool_call(
                "cortex_memory",
                {"action": "promote", "memory_id": memory_id, "reason": "general knowledge"},
            )
        )
        self.assertIn("promotion", queued)
        self.assertEqual(queued["promotion"]["status"], "pending")

        # A queued request is not a promotion: the operator admits it.
        receipt = alpha._fleet.admit_request(
            queued["promotion"]["request_id"], actor="jacoby", reason="general knowledge"
        )
        self.assertTrue(receipt.promoted, receipt.reasons)

        beta = self._provider("beta")
        context = beta.prefetch("which port does the dashboard use")
        self.assertIn("8088", context)
        self.assertIn("[shared from alpha", context)

        status = json.loads(beta.handle_tool_call("cortex_memory", {"action": "fleet"}))
        self.assertTrue(status["fleet"]["fleet_enabled"])
        self.assertEqual(status["fleet"]["shared_memories"], 1)


if __name__ == "__main__":
    unittest.main()
