"""Tests for the multi-tenant fleet memory layer.

Run standalone::

    python3 -m unittest tests.test_fleet -v

These tests force the *checkout's* ``cortex`` package to load. The plugin copy
installed under ``~/.hermes/installs`` would otherwise win ``import cortex`` and
the tests would silently exercise different code.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importing the tests package runs _bootstrap, which loads *this checkout* as
# ``cortex``. Without it the installed Hermes plugin copy wins the import and
# these tests would exercise the wrong code.
from tests import _bootstrap  # noqa: E402,F401
import cortex  # noqa: E402

if Path(cortex.__file__).resolve().parent != ROOT:
    raise RuntimeError(f"tests are exercising {cortex.__file__}, not {ROOT}")

from cortex import fleet  # noqa: E402
from cortex.fleet import (  # noqa: E402
    LOCAL_LAYER,
    SHARED_LAYER,
    Approval,
    FleetConfig,
    FleetMemory,
    PromotionPolicy,
    resolve_fleet_config,
)


class FleetTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.local_db = self.tmp / "tenant" / "cortex.db"
        self.shared_db = self.tmp / "shared.db"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def config(self, *, tenant: str = "teknium", shared: bool = True, promotion: bool = True) -> FleetConfig:
        return FleetConfig(
            tenant=tenant,
            local_db=self.local_db,
            shared_db=self.shared_db if shared else None,
            shared_promotion_enabled=promotion,
        )

    def add(self, memory: FleetMemory, content: str, **kwargs) -> str:
        memory_id, _created = memory.local_store.add_memory(content, **kwargs)
        return memory_id

    @staticmethod
    def count(store) -> int:
        return int(store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])


class ResolveConfigTests(FleetTestBase):
    def test_absent_shared_path_keeps_single_store(self) -> None:
        config = resolve_fleet_config(
            hermes_home=self.tmp / "profiles" / "teknium", tenant="teknium", plugin_config={}
        )
        self.assertFalse(config.fleet_enabled)
        self.assertIsNone(config.shared_db)
        self.assertEqual(
            config.local_db, (self.tmp / "profiles" / "teknium" / "cortex" / "cortex.db")
        )

    def test_shared_path_enables_fleet_and_expands_home(self) -> None:
        home = self.tmp / "profiles" / "teknium"
        config = resolve_fleet_config(
            hermes_home=home,
            tenant="teknium",
            plugin_config={
                "db_path": "${HERMES_HOME}/cortex/cortex.db",
                "shared_db_path": "${HERMES_HOME}/../shared/cortex.db",
                "shared_weight": 0.7,
                "shared_promotion_enabled": True,
            },
        )
        self.assertTrue(config.fleet_enabled)
        self.assertEqual(config.local_db, home / "cortex" / "cortex.db")
        self.assertEqual(config.shared_db.resolve(), (home / ".." / "shared" / "cortex.db").resolve())
        self.assertAlmostEqual(config.shared_weight, 0.7)
        self.assertTrue(config.shared_promotion_enabled)


class RetrievalMergeTests(FleetTestBase):
    def test_shared_layer_absent_means_local_only(self) -> None:
        memory = FleetMemory(self.config(shared=False))
        try:
            self.add(memory, "Duelist Road deploys to the staging host on port 3001.")
            hits = memory.recall("duelist road staging port", limit=5)
            self.assertTrue(hits)
            self.assertTrue(all(hit.layer == LOCAL_LAYER for hit in hits))
        finally:
            memory.close()

    def test_shared_hits_appear_tagged_and_lower_ranked(self) -> None:
        # Tenant B promotes a fact tenant A does not have locally: that is the
        # actual cross-bot sharing path, and it must show up as a shared hit.
        config_b = FleetConfig(tenant="hermes", local_db=self.tmp / "hermes" / "cortex.db",
                               shared_db=self.shared_db, shared_promotion_enabled=True)
        with FleetMemory(config_b) as b:
            promoted = self.add(b, "Tailscale exit node for the homelab gateway is the NUC.")
            self.assertEqual(
                b.promote(promoted, approval=Approval(actor="jacoby", reason="general topology")).status,
                "promoted",
            )

        with FleetMemory(self.config(tenant="teknium", promotion=False)) as a:
            self.add(a, "The homelab gateway runs Proxmox on the NUC box.")
            hits = a.recall("homelab gateway tailscale", limit=5)
            layers = {hit.layer for hit in hits}
            self.assertIn(LOCAL_LAYER, layers)
            self.assertIn(SHARED_LAYER, layers)
            shared = [hit for hit in hits if hit.layer == SHARED_LAYER]
            self.assertTrue(all(hit.tenant == "teknium" for hit in shared))
            # Scores are capped by the shared weight, so a shared fact never
            # outranks an equally strong private one.
            for hit in shared:
                self.assertLessEqual(hit.score, a.config.shared_weight + 1e-9)

    def test_merge_applies_shared_weight_and_prefers_local_on_ties(self) -> None:
        with FleetMemory(self.config(shared=False)) as memory:
            strong_shared = fleet.FleetHit(
                memory={"id": "s1", "content": "shared fact about gateways"},
                score=0.9 * memory.config.shared_weight,
                layer=SHARED_LAYER,
                tenant="teknium",
                estimated_tokens=10,
            )
            weak_local = fleet.FleetHit(
                memory={"id": "l1", "content": "private fact about gateways"},
                score=0.5,
                layer=LOCAL_LAYER,
                tenant="teknium",
                estimated_tokens=10,
            )
            ordered = memory._merge([weak_local, strong_shared], limit=5, token_budget=100)
            self.assertEqual([hit.layer for hit in ordered], [SHARED_LAYER, LOCAL_LAYER])

            tied_shared = fleet.FleetHit(
                memory={"id": "s2", "content": "private fact about gateways"},
                score=0.5,
                layer=SHARED_LAYER,
                tenant="teknium",
                estimated_tokens=10,
            )
            merged = memory._merge([tied_shared, weak_local], limit=5, token_budget=100)
            self.assertEqual(len(merged), 1)
            self.assertEqual(merged[0].layer, LOCAL_LAYER)

    def test_duplicate_content_prefers_the_private_copy(self) -> None:
        with FleetMemory(self.config()) as memory:
            text = "Cortex stores memories in SQLite with WAL enabled."
            memory_id = self.add(memory, text)
            self.assertEqual(
                memory.promote(memory_id, approval=Approval(actor="jacoby", reason="general fact")).status,
                "promoted",
            )
            hits = memory.recall("Cortex SQLite WAL", limit=5)
            matching = [h for h in hits if "SQLite with WAL" in str(h.memory.get("content"))]
            self.assertEqual(len(matching), 1)
            self.assertEqual(matching[0].layer, LOCAL_LAYER)

    def test_tenant_isolation_private_memory_never_crosses(self) -> None:
        config_a = self.config(tenant="teknium")
        config_b = FleetConfig(tenant="ledger", local_db=self.tmp / "ledger" / "cortex.db",
                               shared_db=self.shared_db, shared_promotion_enabled=False)
        with FleetMemory(config_a) as a, FleetMemory(config_b) as b:
            self.add(a, "Jacoby prefers dense one-screen engineering answers.")
            self.assertFalse(
                [h for h in b.recall("dense one-screen engineering answers", limit=5)
                 if "dense one-screen" in str(h.memory.get("content"))]
            )

    def test_token_budget_is_respected(self) -> None:
        with FleetMemory(self.config(shared=False)) as memory:
            for index in range(6):
                self.add(memory, f"Hinge calibration note {index} covers the folding posture threshold value.")
            hits = memory.recall("hinge calibration folding posture threshold", limit=6, token_budget=40)
            self.assertLessEqual(sum(max(1, h.estimated_tokens) for h in hits), 40)


class PromotionPolicyTests(FleetTestBase):
    def policy(self, tenant: str = "teknium") -> PromotionPolicy:
        return PromotionPolicy(tenant=tenant)

    def test_currency_and_balance_are_hard_denied(self) -> None:
        verdict = self.policy().inspect(
            {"content": "His card balance is $1,240 and the APR is 24.9%.", "source_type": "conversation"}
        )
        self.assertFalse(verdict.allowed)
        self.assertIn("hard_deny:balance_or_debt", verdict.reasons)

    def test_ledger_tenant_denied_even_without_numbers(self) -> None:
        verdict = self.policy("ledger").inspect(
            {"content": "He wants to clear the card before the mortgage renewal.", "source_type": "conversation"}
        )
        self.assertFalse(verdict.allowed)
        self.assertIn("tenant:ledger", verdict.reasons)

    def test_finance_source_type_denied(self) -> None:
        verdict = self.policy().inspect(
            {"content": "Weekly grocery spend pattern is stable.", "source_type": "transaction"}
        )
        self.assertFalse(verdict.allowed)
        self.assertIn("source_type:transaction", verdict.reasons)

    def test_secret_material_denied(self) -> None:
        verdict = self.policy().inspect(
            {"content": "The deploy token is ghp_abcdefghijklmnopqrstuvwxyz0123.", "source_type": "conversation"}
        )
        self.assertFalse(verdict.allowed)
        self.assertIn("hard_deny:secret_token", verdict.reasons)

    def test_soft_identifiers_are_redacted(self) -> None:
        verdict = self.policy().inspect(
            {"content": "From /Users/jacoby log into 192.168.1.50 and mail ops@example.com.",
             "source_type": "conversation"}
        )
        self.assertTrue(verdict.allowed)
        self.assertIn("home_path", verdict.soft_matches)
        self.assertIn("local_ip", verdict.soft_matches)
        self.assertIn("email", verdict.soft_matches)
        self.assertNotIn("jacoby", verdict.redacted_text or "")
        self.assertNotIn("192.168.1.50", verdict.redacted_text or "")
        self.assertNotIn("ops@example.com", verdict.redacted_text or "")

    def test_clean_procedure_is_allowed(self) -> None:
        verdict = self.policy().inspect(
            {"content": "Rebuild the Swift package with swift build before running the release script.",
             "source_type": "conversation"}
        )
        self.assertTrue(verdict.allowed)
        self.assertEqual(verdict.hard_matches, [])


class PromotionGateTests(FleetTestBase):
    def test_requires_an_explicit_approval(self) -> None:
        with FleetMemory(self.config()) as memory:
            memory_id = self.add(memory, "Duo posture thresholds live in DuoLayout.swift.")
            receipt = memory.promote(memory_id, approval=None)
            self.assertEqual(receipt.status, "denied")
            self.assertIn("approval_required", receipt.reasons)
            self.assertEqual(self.count(memory._shared), 0)

    def test_disabled_tenant_cannot_write_shared(self) -> None:
        with FleetMemory(self.config(promotion=False)) as memory:
            memory_id = self.add(memory, "Duo posture thresholds live in DuoLayout.swift.")
            receipt = memory.promote(memory_id, approval=Approval(actor="jacoby", reason="general"))
            self.assertEqual(receipt.status, "disabled")
            self.assertEqual(self.count(memory._shared), 0)

    def test_finance_promotion_writes_nothing(self) -> None:
        with FleetMemory(self.config()) as memory:
            memory_id = self.add(
                memory,
                "Card balance $2,300, minimum payment due on the 14th, APR 26.99%.",
                source_type="conversation",
            )
            receipt = memory.promote(memory_id, approval=Approval(actor="jacoby", reason="try"))
            self.assertEqual(receipt.status, "denied")
            self.assertEqual(self.count(memory._shared), 0)

    def test_promotion_is_idempotent(self) -> None:
        with FleetMemory(self.config()) as memory:
            memory_id = self.add(memory, "Xcode Cloud workflow named dev builds the development branch.")
            first = memory.promote(memory_id, approval=Approval(actor="jacoby", reason="general build fact"))
            second = memory.promote(memory_id, approval=Approval(actor="jacoby", reason="again"))
            self.assertEqual(first.status, "promoted")
            self.assertEqual(second.status, "duplicate")
            self.assertEqual(second.shared_memory_id, first.shared_memory_id)
            self.assertEqual(self.count(memory._shared), 1)

    def test_promotion_writes_audit_row_and_redacts(self) -> None:
        with FleetMemory(self.config()) as memory:
            memory_id = self.add(memory, "Publish the dashboard at 192.168.1.50 and notify ops@example.com.")
            receipt = memory.promote(
                memory_id, approval=Approval(actor="jacoby", reason="homelab runbook")
            )
            self.assertEqual(receipt.status, "promoted")
            self.assertCountEqual(receipt.redactions, ["local_ip", "email"])
            row = memory._shared._conn.execute(
                "SELECT origin_tenant, actor, redactions_json FROM fleet_promotions"
            ).fetchone()
            self.assertEqual(row[0], "teknium")
            self.assertEqual(row[1], "jacoby")
            self.assertIn("local_ip", row[2])
            shared_text = memory._shared.get_memory(receipt.shared_memory_id)["content"]
            self.assertNotIn("192.168.1.50", shared_text)
            self.assertNotIn("ops@example.com", shared_text)

    def test_dry_run_reports_without_writing(self) -> None:
        with FleetMemory(self.config()) as memory:
            memory_id = self.add(memory, "Temple uses Xcode Cloud with a dev workflow.")
            receipt = memory.promote(
                memory_id, approval=Approval(actor="jacoby", reason="check"), dry_run=True
            )
            self.assertEqual(receipt.status, "dry_run")
            self.assertEqual(self.count(memory._shared), 0)

    def test_unknown_memory_is_denied(self) -> None:
        with FleetMemory(self.config()) as memory:
            receipt = memory.promote("does-not-exist", approval=Approval(actor="jacoby", reason="x"))
            self.assertEqual(receipt.status, "denied")
            self.assertIn("unknown_memory", receipt.reasons)


class AuditTests(FleetTestBase):
    def test_audit_counts_promotable_and_denied_without_writing(self) -> None:
        with FleetMemory(self.config()) as memory:
            self.add(memory, "Run the duo posture check script before merging layout work.")
            self.add(memory, "The Temple repository keeps three branches: development, testing, main.")
            self.add(memory, "His checking balance is $910.22 and the card payment is due Friday.")
            report = memory.audit()
            self.assertEqual(report["scanned"], 3)
            self.assertGreaterEqual(report["promotable"], 2)
            self.assertTrue(report["denied"])
            self.assertTrue(any(reason.startswith("hard_deny:") for reason in report["denied"]))
            self.assertEqual(self.count(memory._shared), 0)


if __name__ == "__main__":
    unittest.main()
