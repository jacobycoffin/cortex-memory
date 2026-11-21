"""Multi-tenant isolation and promotion-gate tests (backlog item #195).

Every test builds throwaway databases under pytest's ``tmp_path``; nothing here
touches a live Cortex database. The tests are plain pytest functions so the
module can also be imported by ``python -m unittest discover`` on hosts where
pytest is not installed (unittest simply ignores them).

Coverage:

1. a memory written to tenant A never comes back from tenant B's recall/search;
2. the shared layer is readable by every tenant and writable by none of them
   except through :class:`cortex.fleet.PromotionGate`;
3. promotion without an explicit :class:`Approval` (actor + reason) is refused;
4. finance-derived content is refused outright, never redacted, including when
   a caller forges an "allowed" promotion verdict or a finance-sourced request
   reaches the gate out of band;
5. tenants with overlapping names (``temple`` / ``temple-work``) do not bleed
   private memories or policy decisions into each other.
"""

from __future__ import annotations

import sqlite3
import uuid

from cortex.fleet import (
    LOCAL_LAYER,
    SHARED_LAYER,
    Approval,
    FleetConfig,
    FleetMemory,
    PromotionPolicy,
    PromotionVerdict,
    resolve_fleet_config,
)
from cortex.store import CortexStore

SECRET = "quasar-lantern-7731"
FINANCE_TEXT = "Card balance is $2,300 and the APR is 26.99%; mail statements to ops@example.com."


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def tenant_memory(tmp_path, tenant: str, shared_db, *, promotion: bool = False) -> FleetMemory:
    """A tenant whose private database lives in its own directory."""

    return FleetMemory(
        FleetConfig(
            tenant=tenant,
            local_db=tmp_path / tenant / "cortex.db",
            shared_db=shared_db,
            shared_promotion_enabled=promotion,
        )
    )


def assert_db_refuses(action, label: str) -> None:
    """A write against the read-only shared layer must raise a sqlite error."""

    try:
        action()
    except sqlite3.Error:
        return
    raise AssertionError(f"shared layer accepted a write: {label}")


def count_memories(store) -> int:
    return int(store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])


def contents(hits) -> list[str]:
    return [str(getattr(hit, "memory", hit).get("content") or "") for hit in hits]


def pend_request(memory: FleetMemory, memory_id: str, text: str, *, source_type: str) -> str:
    """Insert a pending outbox row, simulating data written by an older build."""

    request_id = str(uuid.uuid4())
    with memory.local_store.transaction():
        memory.local_store._conn.execute(
            "INSERT INTO fleet_promotion_requests("
            "request_id,tenant,origin_memory_id,content_hash,content,kind,source_type,"
            "verdict_json,status,requested_at,reason) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                request_id,
                memory.config.tenant,
                memory_id,
                "out-of-band",
                text,
                "semantic",
                source_type,
                "{}",
                "pending",
                "2026-01-01T00:00:00Z",
                "",
            ),
        )
    return request_id


# ---------------------------------------------------------------------------
# 1. private memories never cross tenants
# ---------------------------------------------------------------------------


def test_private_memory_never_returned_to_another_tenant(tmp_path):
    shared = tmp_path / "shared.db"
    private = f"{SECRET}: the vault key rotation runs on the second Tuesday."

    with tenant_memory(tmp_path, "temple", shared) as a, tenant_memory(
        tmp_path, "temple-work", shared
    ) as b:
        memory_id, _ = a.local_store.add_memory(private)

        for query in (
            f"{SECRET} vault key rotation",
            "vault key rotation second Tuesday",
            "quasar lantern",
        ):
            for hit in b.recall(query, limit=8):
                assert SECRET not in hit.memory["content"], f"recall leaked {hit.layer} hit"
            for result in b.search(query, limit=8):
                assert SECRET not in result.memory["content"], "search leaked a private hit"

        # Knowing the private id is not enough either.
        assert b.local_store.get_memory(memory_id) is None

        # Nothing from A's private store was copied into B's database.
        for row in b.local_store._conn.execute("SELECT content FROM memories").fetchall():
            assert SECRET not in str(row[0])

        # Sanity check: A itself can still recall its own fact.
        assert any(SECRET in text for text in contents(a.recall("vault key rotation", limit=8)))


def test_private_memory_is_invisible_in_the_other_direction(tmp_path):
    shared = tmp_path / "shared.db"

    with tenant_memory(tmp_path, "temple", shared) as a, tenant_memory(
        tmp_path, "temple-work", shared
    ) as b:
        b.local_store.add_memory(f"{SECRET} belongs only to temple-work.")

        assert not [
            hit
            for hit in a.recall(f"{SECRET} belongs temple-work", limit=8)
            if SECRET in hit.memory["content"]
        ]
        assert SECRET not in "\n".join(contents(a.search("temple work secret", limit=8)))


# ---------------------------------------------------------------------------
# 2. shared layer: readable by all, writable only through the gate
# ---------------------------------------------------------------------------


def test_shared_layer_is_readable_by_every_tenant(tmp_path):
    shared = tmp_path / "shared.db"
    fact = "The homelab gateway runs on the nuc chassis."

    with tenant_memory(tmp_path, "temple", shared, promotion=True) as author:
        memory_id, _ = author.local_store.add_memory(fact)
        receipt = author.promote(
            memory_id, approval=Approval(actor="jacoby", reason="general homelab topology")
        )
        assert receipt.status == "promoted"

    for reader in ("temple-work", "another-bot"):
        with tenant_memory(tmp_path, reader, shared) as tenant:
            hits = [
                hit
                for hit in tenant.recall("homelab gateway nuc chassis", limit=8)
                if hit.layer == SHARED_LAYER and "homelab gateway" in hit.memory["content"]
            ]
            assert hits, f"{reader} could not read the shared layer"


def test_shared_layer_rejects_direct_writes_from_tenant_handles(tmp_path):
    shared = tmp_path / "shared.db"
    fact = "The homelab gateway runs on the nuc chassis."

    with tenant_memory(tmp_path, "temple", shared, promotion=True) as author:
        memory_id, _ = author.local_store.add_memory(fact)
        assert (
            author.promote(
                memory_id, approval=Approval(actor="jacoby", reason="general homelab topology")
            ).status
            == "promoted"
        )
        assert author.shared_store is not None

        assert_db_refuses(
            lambda: author.shared_store.add_memory(f"smuggled {SECRET}"),
            "add_memory on the tenant's shared handle",
        )
        assert_db_refuses(
            lambda: author.shared_store._conn.execute("DELETE FROM memories"),
            "DELETE through the tenant's shared connection",
        )
        assert_db_refuses(
            lambda: author.shared_store._conn.execute(
                "UPDATE memories SET content='overwritten'"
            ),
            "UPDATE through the tenant's shared connection",
        )

        def write_in_transaction() -> None:
            with author.shared_store.transaction():
                author.shared_store._conn.execute(
                    "INSERT INTO memories(id, content) VALUES('smuggled', 'x')"
                )

        assert_db_refuses(write_in_transaction, "INSERT inside transaction()")

        assert count_memories(author.shared_store) == 1
        assert SECRET not in "\n".join(
            contents(author.shared_store.get_memory(row[0]) for row in
                     author.shared_store._conn.execute("SELECT id FROM memories").fetchall())
        )


def test_second_tenant_cannot_write_shared_either(tmp_path):
    shared = tmp_path / "shared.db"
    with tenant_memory(tmp_path, "temple", shared, promotion=True) as author:
        memory_id, _ = author.local_store.add_memory("Shared gateway fact for the fleet.")
        assert (
            author.promote(memory_id, approval=Approval(actor="jacoby", reason="topology")).status
            == "promoted"
        )

    with tenant_memory(tmp_path, "temple-work", shared, promotion=False) as reader:
        assert reader.shared_store is not None
        assert_db_refuses(
            lambda: reader.shared_store.add_memory(f"smuggled {SECRET}"),
            "add_memory by a non-author tenant",
        )
        assert count_memories(reader.shared_store) == 1
        # Without the promotion flag even the gate refuses.
        local_id, _ = reader.local_store.add_memory("A clean fact that could be shared.")
        receipt = reader.promote(
            local_id, approval=Approval(actor="jacoby", reason="wants to share")
        )
        assert receipt.status == "disabled"
        assert count_memories(reader.shared_store) == 1


def test_only_the_gate_leaves_an_audit_row(tmp_path):
    shared = tmp_path / "shared.db"
    with tenant_memory(tmp_path, "temple", shared, promotion=True) as author:
        memory_id, _ = author.local_store.add_memory("Publish dashboards from the edge proxy.")
        receipt = author.promote(
            memory_id, approval=Approval(actor="jacoby", reason="operator approved runbook")
        )
        assert receipt.status == "promoted"
        row = author.shared_store._conn.execute(
            "SELECT origin_tenant, actor, reason, shared_memory_id FROM fleet_promotions"
        ).fetchone()
        assert row is not None
        assert row["origin_tenant"] == "temple"
        assert row["actor"] == "jacoby"
        assert row["shared_memory_id"] == receipt.shared_memory_id
        assert count_memories(author.shared_store) == 1


# ---------------------------------------------------------------------------
# 3. approval is mandatory and cannot be faked
# ---------------------------------------------------------------------------


def test_promotion_without_explicit_approval_is_refused(tmp_path):
    shared = tmp_path / "shared.db"
    with tenant_memory(tmp_path, "temple", shared, promotion=True) as author:
        memory_id, _ = author.local_store.add_memory(
            "Rebuild the Swift package with swift build before the release script."
        )
        bad_approvals = [
            None,
            Approval(actor="", reason="because"),
            Approval(actor="   ", reason="because"),
            Approval(actor="jacoby", reason=""),
            Approval(actor="jacoby", reason="   "),
        ]
        for approval in bad_approvals:
            receipt = author.promote(memory_id, approval=approval)
            assert receipt.status == "denied", approval
            assert "approval_required" in receipt.reasons, approval

        assert count_memories(author.shared_store) == 0
        assert (
            author.shared_store._conn.execute("SELECT COUNT(*) FROM fleet_promotions").fetchone()[0]
            == 0
        )


def test_gate_refuses_content_without_approval_even_when_policy_allows(tmp_path):
    shared = tmp_path / "shared.db"
    with tenant_memory(tmp_path, "temple", shared, promotion=True) as author:
        gate, owned = author._gate()
        try:
            receipt = gate.promote_content(
                "A clean general fact about port 8080.",
                origin_memory_id="local-1",
                approval=None,
            )
        finally:
            if owned and gate.shared is not None:
                gate.shared.close()
        assert receipt.status == "denied"
        assert "approval_required" in receipt.reasons
        assert count_memories(author.shared_store) == 0


def test_forged_verdict_cannot_bypass_the_promotion_gate(tmp_path):
    """A caller-supplied "allowed" verdict must not skip the finance scan.

    ``promote_content`` used to trust ``approved_verdict`` outright, so a forged
    verdict (or a verdict computed before a policy tightened) admitted denied
    content into the shared layer. Admission re-inspects the text itself.
    """

    shared = tmp_path / "shared.db"
    with tenant_memory(tmp_path, "temple", shared, promotion=True) as author:
        memory_id, _ = author.local_store.add_memory(FINANCE_TEXT)
        gate, owned = author._gate()
        try:
            forged = PromotionVerdict(
                allowed=True, reasons=["clean"], redacted_text=None
            )
            receipt = gate.promote_content(
                FINANCE_TEXT,
                origin_memory_id=memory_id,
                approval=Approval(actor="jacoby", reason="forged verdict says it is fine"),
                approved_verdict=forged,
            )
        finally:
            if owned and gate.shared is not None:
                gate.shared.close()

        assert receipt.status == "denied"
        assert any(reason.startswith("hard_deny:") for reason in receipt.reasons)
        assert count_memories(author.shared_store) == 0
        assert "2,300" not in "\n".join(
            contents(author.shared_store.get_memory(row[0]) for row in
                     author.shared_store._conn.execute("SELECT id FROM memories").fetchall())
        )


def test_pending_request_with_finance_source_type_cannot_be_admitted(tmp_path):
    shared = tmp_path / "shared.db"
    with tenant_memory(tmp_path, "temple", shared, promotion=True) as author:
        memory_id, _ = author.local_store.add_memory("Weekly grocery spend pattern is stable.")
        request_id = pend_request(
            author, memory_id, "Weekly grocery spend pattern is stable.", source_type="statement"
        )

        receipt = author.admit_request(request_id, actor="jacoby", reason="looks harmless")

        assert receipt.status == "denied"
        assert "source_type:statement" in receipt.reasons
        assert count_memories(author.shared_store) == 0


# ---------------------------------------------------------------------------
# 4. finance is denied outright, never redacted
# ---------------------------------------------------------------------------


def test_finance_content_is_denied_outright_not_redacted(tmp_path):
    shared = tmp_path / "shared.db"
    with tenant_memory(tmp_path, "temple", shared, promotion=True) as author:
        memory_id, _ = author.local_store.add_memory(FINANCE_TEXT)
        memory = author.local_store.get_memory(memory_id)

        verdict = author.policy.inspect(memory)
        assert verdict.allowed is False
        assert verdict.redacted_text is None, "finance must not take the redaction path"
        assert any(reason.startswith("hard_deny:") for reason in verdict.reasons)

        receipt = author.promote(
            memory_id, approval=Approval(actor="jacoby", reason="operator asked")
        )
        assert receipt.status == "denied"
        assert receipt.shared_memory_id is None
        assert receipt.redactions == []
        assert count_memories(author.shared_store) == 0

        # The outbox must not keep a second copy of the denied content either.
        outcome = author.request_promotion(memory_id, reason="bot asked to share")
        assert outcome["success"] is False
        assert outcome["status"] == "denied"
        row = author.promotions(status=None)[0]
        assert row["status"] == "denied"
        assert "2,300" not in str(row["content"])
        assert "26.99" not in str(row["content"])


def test_finance_derived_source_type_is_denied_without_reading_numbers(tmp_path):
    # A Ledger tenant is finance-derived by identity: no digits are needed for
    # the record-level denial to fire.
    ledger_policy = PromotionPolicy(tenant="ledger")
    verdict = ledger_policy.inspect(
        {"content": "He is paying down the card first.", "source_type": "conversation"}
    )
    assert verdict.allowed is False
    assert "tenant:ledger" in verdict.reasons

    policy = PromotionPolicy(tenant="temple")
    sourced = policy.inspect(
        {"content": "Weekly grocery spend pattern is stable.", "source_type": "transaction"}
    )
    assert sourced.allowed is False
    assert "source_type:transaction" in sourced.reasons


def test_out_of_band_finance_rows_are_dropped_at_recall(tmp_path):
    """Defence in depth: a row inserted directly into the shared file is not served."""

    shared = tmp_path / "shared.db"
    seed = CortexStore(shared)
    try:
        seed.add_memory(
            "Weekly grocery spend pattern is stable.",
            source_type="statement",
            scope={"layer": SHARED_LAYER, "origin_tenant": "temple"},
        )
        seed.add_memory(
            "He prefers weekly reviews on Sunday.",
            source_type="fleet_promotion",
            scope={"layer": SHARED_LAYER, "origin_tenant": "ledger"},
        )
        seed.add_memory(
            "The staging proxy listens on port 3001.",
            source_type="fleet_promotion",
            scope={"layer": SHARED_LAYER, "origin_tenant": "temple"},
        )
    finally:
        seed.close()

    with tenant_memory(tmp_path, "temple-work", shared) as reader:
        finance = reader.recall("weekly grocery spend pattern", limit=8)
        assert not any("grocery" in text for text in contents(finance))

        ledger = reader.recall("weekly reviews Sunday", limit=8)
        assert not any("weekly reviews" in text for text in contents(ledger))

        # The clean row from a non-denied tenant still comes through, so the
        # filter is not simply dropping everything.
        assert any("staging proxy" in text for text in contents(reader.recall(
            "staging proxy port 3001", limit=8
        )))


# ---------------------------------------------------------------------------
# 5. overlapping tenant names do not bleed
# ---------------------------------------------------------------------------


def test_overlapping_tenant_names_do_not_bleed(tmp_path):
    shared = tmp_path / "shared.db"
    home_a = tmp_path / "profiles" / "temple"
    home_b = tmp_path / "profiles" / "temple-work"

    config_a = resolve_fleet_config(
        hermes_home=home_a, tenant="", plugin_config={"shared_db_path": str(shared)}
    )
    config_b = resolve_fleet_config(
        hermes_home=home_b, tenant="", plugin_config={"shared_db_path": str(shared)}
    )

    assert config_a.tenant == "temple"
    assert config_b.tenant == "temple-work"
    assert config_a.local_db != config_b.local_db
    assert config_a.local_db.parent != config_b.local_db.parent

    with FleetMemory(config_a) as a, FleetMemory(config_b) as b:
        a.local_store.add_memory(
            "temple-only: the api gateway listens on port 8080 in the temple namespace."
        )
        b.local_store.add_memory(
            "temple-work-only: the batch job runs at 03:00 in the temple-work namespace."
        )

        # A substring query for the longer name must not reach the shorter one.
        assert not any(
            "temple-work-only" in hit.memory["content"]
            for hit in a.recall("temple-work batch job 03:00 namespace", limit=8)
        )
        assert not any(
            "temple-only" in hit.memory["content"]
            for hit in b.recall("temple api gateway 8080 namespace", limit=8)
        )

        # Each still sees its own memory with the overlapping query.
        assert any(
            "temple-only" in hit.memory["content"]
            for hit in a.recall("temple api gateway 8080", limit=8)
        )
        assert any(
            "temple-work-only" in hit.memory["content"]
            for hit in b.recall("temple-work batch job 03:00", limit=8)
        )

        # And neither database contains the other tenant's text.
        for row in a.local_store._conn.execute("SELECT content FROM memories").fetchall():
            assert "temple-work-only" not in str(row[0])
        for row in b.local_store._conn.execute("SELECT content FROM memories").fetchall():
            assert "temple-only" not in str(row[0])


def test_overlapping_names_do_not_share_a_deny_decision():
    """Deny-listed tenant ids match exactly; 'ledger-work' is not 'ledger'."""

    neighbour = PromotionPolicy(tenant="ledger-work")
    verdict = neighbour.inspect(
        {"content": "Refactor the ledger import script.", "source_type": "conversation"}
    )
    assert verdict.allowed is True

    ledger = PromotionPolicy(tenant="ledger")
    denied = ledger.inspect(
        {"content": "Refactor the ledger import script.", "source_type": "conversation"}
    )
    assert denied.allowed is False
    assert "tenant:ledger" in denied.reasons


def test_local_layer_tag_is_always_local(tmp_path):
    """No hit from a private store may claim to be a shared-layer hit."""

    shared = tmp_path / "shared.db"
    with tenant_memory(tmp_path, "temple", shared) as tenant:
        tenant.local_store.add_memory(f"{SECRET} private note about namespaces.")
        hits = tenant.search(f"{SECRET} private note namespaces", limit=8)
        assert hits
        assert all(getattr(hit, "layer", LOCAL_LAYER) == LOCAL_LAYER for hit in hits)
        assert all(getattr(hit, "origin_tenant", "") == "" for hit in hits)
