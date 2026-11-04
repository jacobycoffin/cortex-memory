"""Multi-tenant fleet memory.

Cortex normally runs one SQLite brain per Hermes profile. This module adds a
second, narrow layer on top of that: each bot keeps its own local database
(unchanged, private), and a *shared* database holds general knowledge that every
bot may read. Nothing moves between the two automatically.

Design rules encoded here:

* **Private by default.** A local store is only ever read by its own tenant. The
  shared layer is read-only to tenants except through :class:`PromotionGate`.
* **No silent promotion.** ``promote()`` requires an explicit :class:`Approval`
  (actor + reason). A memory record can never approve itself.
* **Finance is denied, not redacted.** Ledger-derived content — balances, debts,
  rates, statements, transaction sources — is refused outright, because a
  redaction bug that leaves one number behind is a leak. Soft identifiers
  (home paths, emails, phone numbers) are redacted and reported instead.
* **Fleet off is the default.** With no ``shared_db_path`` configured, resolve
  returns a config whose ``fleet_enabled`` is False and callers keep using the
  single store exactly as before. This is what makes the rollout staged.

The module deliberately does not import ``hermes_provider`` or any harness code:
it composes two :class:`~cortex.store.CortexStore` instances and can be tested
against temporary databases.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .retrieval import MemoryRetriever, RetrievalContext, RetrievalResult
from .store import CortexStore

LOCAL_LAYER = "local"
SHARED_LAYER = "shared"

DEFAULT_SHARED_WEIGHT = 0.85
DEFAULT_LOCAL_LIMIT = 6
DEFAULT_SHARED_LIMIT = 4
DEFAULT_EXCLUDE_ARCHIVED = True

#: Tenants whose *content* is finance-derived no matter how it is phrased.
#: A Ledger memory is refused even when the text carries no number at all:
#: "he is paying down the card first" is finance knowledge without digits.
DENY_TENANTS: frozenset[str] = frozenset({"ledger", "finance", "budget"})

#: Source types that mark a row as derived from account data.
DENY_SOURCE_TYPES: frozenset[str] = frozenset(
    {"statement", "transaction", "account_summary", "plaid", "bank_sync", "invoice"}
)

#: ``approval_state`` values Cortex accepts (see cortex_records.add_memory).
SHARED_APPROVAL_STATE = "operator_approved"

# ---------------------------------------------------------------------------
# Promotion policy
# ---------------------------------------------------------------------------

#: Hard denies. A single match refuses the promotion; there is no redaction
#: path for these because a partially-redacted financial fact is still a leak.
_HARD_DENY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("currency_amount", re.compile(r"(?:[$€£¥]\s?\d|\b\d[\d,.]*\s?(?:usd|eur|gbp|cad|dollars?)\b)", re.I)),
    ("balance_or_debt", re.compile(
        r"\b(?:balance|balances|owed|owes|owing|debt|debts|loan|loans|mortgage|principle|interest rate|apr|"
        r"minimum payment|payoff|pay off|net worth|credit score|credit limit|statement balance)\b", re.I)),
    ("income", re.compile(r"\b(?:salary|paycheck|paychecks|income|take-home pay|hourly rate)\b", re.I)),
    ("account_identifiers", re.compile(
        r"\b(?:account|acct|routing|iban|swift|sort code|card)\s*(?:number|no\.?|#)\b", re.I)),
    ("card_number", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("secret_token", re.compile(r"\b(?:sk-[A-Za-z0-9_\-]{12,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9\-]{10,}|AKIA[0-9A-Z]{16})\b")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("password", re.compile(r"\b(?:password|passwd|api[_ ]?key|secret)\s*[:=]\s*\S", re.I)),
)

#: Soft identifiers. Matching text is masked before it can reach the shared
#: layer, and the receipt reports exactly what was masked.
_SOFT_REDACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("phone", re.compile(r"(?<![\d-])(?:\+?1[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}(?![\d-])")),
    ("home_path", re.compile(r"(?:/(?:Users|home)/|C:\\\\Users\\\\)([\w.\-]+)")),
    ("local_ip", re.compile(r"\b(?:192\.168|10\.|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b")),
)

_REDACTION_BY_KIND = {
    "email": "[email redacted]",
    "phone": "[phone redacted]",
    "home_path": "/[path redacted]",
    "local_ip": "[address redacted]",
}


def _normalize(text: str) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def content_fingerprint(text: str) -> str:
    """Stable identity for dedupe/audit. Not a security boundary."""

    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PromotionVerdict:
    """Outcome of inspecting one candidate for the shared layer."""

    allowed: bool
    reasons: list[str] = field(default_factory=list)
    hard_matches: list[str] = field(default_factory=list)
    soft_matches: list[str] = field(default_factory=list)
    redacted_text: str | None = None

    @property
    def redactions(self) -> list[str]:
        return list(self.soft_matches)

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reasons": list(self.reasons),
            "hard_matches": list(self.hard_matches),
            "soft_matches": list(self.soft_matches),
            "redacted_text": self.redacted_text,
        }


@dataclass(frozen=True)
class Approval:
    """Explicit human/operator authorization to promote one memory."""

    actor: str
    reason: str
    approved_at: str | None = None

    def is_valid(self) -> bool:
        return bool(self.actor and self.actor.strip()) and bool(self.reason and self.reason.strip())


class PromotionPolicy:
    """Decides whether a local memory may be copied into the shared layer."""

    def __init__(
        self,
        *,
        tenant: str,
        deny_tenants: Iterable[str] | None = None,
        deny_source_types: Iterable[str] | None = None,
    ) -> None:
        self.tenant = str(tenant or "").strip().lower()
        self.deny_tenants = frozenset(str(t).lower() for t in (deny_tenants or DENY_TENANTS))
        self.deny_source_types = frozenset(
            str(s).lower() for s in (deny_source_types or DENY_SOURCE_TYPES)
        )

    # -- tenant / record level ------------------------------------------------
    def inspect_record(self, memory: dict[str, Any]) -> list[str]:
        """Reasons the *record itself* can never be shared, independent of text."""

        reasons: list[str] = []
        if self.tenant in self.deny_tenants:
            reasons.append(f"tenant:{self.tenant}")
        source_type = str(memory.get("source_type") or "").strip().lower()
        if source_type in self.deny_source_types:
            reasons.append(f"source_type:{source_type}")
        scope = memory.get("scope") or {}
        if isinstance(scope, str):
            try:
                scope = json.loads(scope)
            except (TypeError, ValueError):
                scope = {}
        if isinstance(scope, dict):
            for key, value in scope.items():
                if "financ" in str(key).lower() or "financ" in str(value).lower():
                    reasons.append(f"scope:{key}")
        for value in memory.get("entities") or []:
            if "financ" in str(value).lower():
                reasons.append("entity:finance")
                break
        return reasons

    # -- text level -----------------------------------------------------------
    def inspect_text(self, text: str) -> PromotionVerdict:
        hard = sorted({name for name, pattern in _HARD_DENY_PATTERNS if pattern.search(text)})
        if hard:
            return PromotionVerdict(
                allowed=False,
                reasons=[f"hard_deny:{name}" for name in hard],
                hard_matches=hard,
            )
        redacted = text
        soft: list[str] = []
        for name, pattern in _SOFT_REDACT_PATTERNS:
            if pattern.search(redacted):
                soft.append(name)
                redacted = pattern.sub(_REDACTION_BY_KIND.get(name, "[redacted]"), redacted)
        return PromotionVerdict(
            allowed=True,
            reasons=[f"redacted:{name}" for name in soft],
            soft_matches=soft,
            redacted_text=redacted if soft else text,
        )

    def inspect(self, memory: dict[str, Any]) -> PromotionVerdict:
        record_reasons = self.inspect_record(memory)
        text = str(memory.get("content") or "")
        verdict = self.inspect_text(text)
        if record_reasons:
            return PromotionVerdict(
                allowed=False,
                reasons=sorted(set(record_reasons) | set(verdict.reasons)),
                hard_matches=verdict.hard_matches,
                soft_matches=verdict.soft_matches,
                redacted_text=None,
            )
        return verdict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FleetConfig:
    """Resolved tenant configuration. ``shared_db is None`` means fleet off."""

    tenant: str
    local_db: Path
    shared_db: Path | None = None
    shared_weight: float = DEFAULT_SHARED_WEIGHT
    shared_promotion_enabled: bool = False
    retrieval_threshold: float = 0.16
    semantic_weight: float = 0.0
    semantic_pool: int = 20

    @property
    def fleet_enabled(self) -> bool:
        return self.shared_db is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "tenant": self.tenant,
            "local_db": str(self.local_db),
            "shared_db": str(self.shared_db) if self.shared_db else None,
            "shared_weight": self.shared_weight,
            "shared_promotion_enabled": self.shared_promotion_enabled,
            "fleet_enabled": self.fleet_enabled,
        }


def resolve_fleet_config(
    *,
    hermes_home: str | Path,
    tenant: str,
    plugin_config: dict[str, Any] | None = None,
) -> FleetConfig:
    """Build a tenant config from a Hermes profile + plugin config block.

    ``plugin_config`` is the ``plugins.entries.cortex`` mapping. Absent
    ``shared_db_path`` keeps the pre-existing single-store behaviour.
    """

    home = Path(hermes_home).expanduser()
    config = dict(plugin_config or {})

    raw_local = str(config.get("db_path") or "").strip()
    if not raw_local:
        # Default Hermes layout: $HERMES_HOME/cortex/cortex.db. A profile's
        # HERMES_HOME is the profile directory, so this is already per-bot.
        local_db = home / "cortex" / "cortex.db"
    else:
        raw_local = raw_local.replace("${HERMES_HOME}", str(home)).replace("$HERMES_HOME", str(home))
        local_db = Path(raw_local).expanduser()

    raw_shared = config.get("shared_db_path")
    shared_db: Path | None = None
    if raw_shared not in (None, "", False):
        raw_shared = str(raw_shared).replace("${HERMES_HOME}", str(home)).replace("$HERMES_HOME", str(home))
        shared_db = Path(raw_shared).expanduser()

    return FleetConfig(
        tenant=str(tenant or "default").strip(),
        local_db=local_db,
        shared_db=shared_db,
        shared_weight=float(config.get("shared_weight") or DEFAULT_SHARED_WEIGHT),
        shared_promotion_enabled=bool(config.get("shared_promotion_enabled", False)),
        retrieval_threshold=float(config.get("retrieval_threshold") or 0.16),
        semantic_weight=float(config.get("semantic_fusion_weight") or 0.0),
        semantic_pool=int(config.get("semantic_fusion_pool") or 20),
    )


# ---------------------------------------------------------------------------
# Retrieval across layers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FleetHit:
    """One recall hit with its provenance."""

    memory: dict[str, Any]
    score: float
    layer: str
    tenant: str
    components: dict[str, float] = field(default_factory=dict)
    estimated_tokens: int = 0

    def as_dict(self) -> dict[str, Any]:
        record = dict(self.memory)
        record["score"] = round(self.score, 5)
        record["layer"] = self.layer
        record["tenant"] = self.tenant
        if self.components:
            record["components"] = {k: round(v, 5) for k, v in self.components.items()}
        return record


class FleetMemory:
    """Reads a tenant's private store plus the shared layer, merged by score.

    Only :meth:`promote` writes to the shared layer, and it delegates to
    :class:`PromotionGate`, which requires an :class:`Approval`. The shared
    store is kept private on purpose: there is no supported way to call
    ``add_memory`` on it from tenant code.
    """

    def __init__(
        self,
        config: FleetConfig,
        *,
        policy: PromotionPolicy | None = None,
        open_shared: bool = True,
    ) -> None:
        self.config = config
        self.policy = policy or PromotionPolicy(tenant=config.tenant)
        self._local = CortexStore(config.local_db)
        self._local_retriever = MemoryRetriever(
            self._local,
            threshold=config.retrieval_threshold,
            semantic_weight=config.semantic_weight,
            semantic_pool=config.semantic_pool,
        )
        self._shared: CortexStore | None = None
        self._shared_retriever: MemoryRetriever | None = None
        if config.fleet_enabled and open_shared:
            self._shared = CortexStore(config.shared_db)
            self._ensure_audit_table(self._shared)
            self._shared_retriever = MemoryRetriever(
                self._shared,
                threshold=config.retrieval_threshold,
                semantic_weight=0.0,  # embeddings are local-only for now
                semantic_pool=config.semantic_pool,
            )

    # -- lifecycle ------------------------------------------------------------
    @property
    def local_store(self) -> CortexStore:
        return self._local

    def close(self) -> None:
        for store in (self._local, self._shared):
            if store is not None:
                store.close()

    def __enter__(self) -> "FleetMemory":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- recall ---------------------------------------------------------------
    def recall(
        self,
        query: str,
        *,
        limit: int = DEFAULT_LOCAL_LIMIT,
        token_budget: int = 700,
        local_limit: int | None = None,
        shared_limit: int = DEFAULT_SHARED_LIMIT,
        include_archived: bool = False,
        context: RetrievalContext | None = None,
        graph_depth: int = 1,
    ) -> list[FleetHit]:
        if not query or limit <= 0 or token_budget <= 0:
            return []

        local_n = int(local_limit or limit)
        local_results = self._search(self._local_retriever, query, local_n, token_budget,
                                     include_archived, context, graph_depth)
        hits = [
            FleetHit(
                memory=dict(result.memory),
                score=float(result.score),
                layer=LOCAL_LAYER,
                tenant=self.config.tenant,
                components=dict(result.components),
                estimated_tokens=int(result.estimated_tokens),
            )
            for result in local_results
        ]

        if self._shared_retriever is not None and shared_limit > 0:
            shared_results = self._search(self._shared_retriever, query, shared_limit, token_budget,
                                          include_archived, context, graph_depth)
            for result in shared_results:
                hits.append(
                    FleetHit(
                        memory=dict(result.memory),
                        score=float(result.score) * self.config.shared_weight,
                        layer=SHARED_LAYER,
                        tenant=self.config.tenant,
                        components=dict(result.components),
                        estimated_tokens=int(result.estimated_tokens),
                    )
                )

        return self._merge(hits, limit=limit, token_budget=token_budget)

    def _search(
        self,
        retriever: MemoryRetriever,
        query: str,
        limit: int,
        token_budget: int,
        include_archived: bool,
        context: RetrievalContext | None,
        graph_depth: int,
    ) -> list[RetrievalResult]:
        if limit <= 0:
            return []
        results, _diagnostics = retriever.search_detailed(
            query,
            limit=limit,
            token_budget=token_budget,
            include_archived=include_archived,
            context=context,
            graph_depth=graph_depth,
        )
        return list(results)

    def _merge(self, hits: Sequence[FleetHit], *, limit: int, token_budget: int) -> list[FleetHit]:
        """Dedupe by content, prefer local, order by score, then fit the budget."""

        by_fingerprint: dict[str, FleetHit] = {}
        for hit in sorted(hits, key=lambda h: (-h.score, 0 if h.layer == LOCAL_LAYER else 1)):
            fingerprint = content_fingerprint(str(hit.memory.get("content") or ""))
            existing = by_fingerprint.get(fingerprint)
            if existing is None:
                by_fingerprint[fingerprint] = hit
                continue
            # Same text in both layers: the private copy wins, and the shared
            # copy never inflates the budget by appearing twice.
            if existing.layer == SHARED_LAYER and hit.layer == LOCAL_LAYER:
                by_fingerprint[fingerprint] = hit

        ordered = sorted(
            by_fingerprint.values(), key=lambda h: (-h.score, 0 if h.layer == LOCAL_LAYER else 1)
        )

        selected: list[FleetHit] = []
        spent = 0
        for hit in ordered:
            cost = max(1, int(hit.estimated_tokens or 1))
            if len(selected) >= limit:
                break
            if spent + cost > token_budget:
                continue
            selected.append(hit)
            spent += cost
        return selected

    # -- promotion ------------------------------------------------------------
    def promote(
        self,
        memory_id: str,
        *,
        approval: Approval | None,
        dry_run: bool = False,
    ) -> "PromotionReceipt":
        gate = PromotionGate(
            config=self.config,
            policy=self.policy,
            local_store=self._local,
            shared_store=self._shared,
        )
        return gate.promote(memory_id, approval=approval, dry_run=dry_run)

    # -- audit ----------------------------------------------------------------
    def audit(self, *, limit: int | None = None) -> dict[str, Any]:
        """Read-only classification of this tenant's promotable memories."""

        rows = self._local._conn.execute(
            "SELECT id FROM memories WHERE state IN ('active','cold') ORDER BY created_at DESC"
            + (" LIMIT ?" if limit else ""),
            ((limit,) if limit else ()),
        ).fetchall()
        promotable = 0
        denied: dict[str, int] = {}
        redacted = 0
        samples: list[dict[str, Any]] = []
        for row in rows:
            memory = self._local.get_memory(row[0])
            if not memory:
                continue
            verdict = self.policy.inspect(memory)
            if verdict.allowed:
                promotable += 1
                if verdict.soft_matches:
                    redacted += 1
                if len(samples) < 5:
                    samples.append(
                        {
                            "memory_id": memory.get("id"),
                            "text": str(memory.get("content") or "")[:160],
                            "redactions": verdict.soft_matches,
                        }
                    )
            else:
                for reason in verdict.reasons:
                    denied[reason] = denied.get(reason, 0) + 1
        return {
            "tenant": self.config.tenant,
            "scanned": len(rows),
            "promotable": promotable,
            "requiring_redaction": redacted,
            "denied": dict(sorted(denied.items(), key=lambda kv: -kv[1])),
            "samples": samples,
        }

    @staticmethod
    def _ensure_audit_table(store: CortexStore) -> None:
        store._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS fleet_promotions (
                id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL UNIQUE,
                origin_memory_id TEXT NOT NULL,
                origin_tenant TEXT NOT NULL,
                shared_memory_id TEXT NOT NULL,
                promoted_at TEXT NOT NULL,
                actor TEXT NOT NULL,
                reason TEXT NOT NULL,
                redactions_json TEXT NOT NULL DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_fleet_promotions_tenant
                ON fleet_promotions(origin_tenant, promoted_at);
            """
        )
        store._conn.commit()


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromotionReceipt:
    status: str  # promoted | denied | duplicate | dry_run | disabled
    origin_memory_id: str | None = None
    shared_memory_id: str | None = None
    reasons: list[str] = field(default_factory=list)
    redactions: list[str] = field(default_factory=list)
    content_preview: str | None = None

    @property
    def promoted(self) -> bool:
        return self.status == "promoted"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "origin_memory_id": self.origin_memory_id,
            "shared_memory_id": self.shared_memory_id,
            "reasons": list(self.reasons),
            "redactions": list(self.redactions),
            "content_preview": self.content_preview,
        }


class PromotionGate:
    """The only supported writer into the shared layer."""

    def __init__(
        self,
        *,
        config: FleetConfig,
        policy: PromotionPolicy,
        local_store: CortexStore,
        shared_store: CortexStore | None,
    ) -> None:
        self.config = config
        self.policy = policy
        self.local = local_store
        self.shared = shared_store

    def promote(
        self,
        memory_id: str,
        *,
        approval: Approval | None,
        dry_run: bool = False,
    ) -> PromotionReceipt:
        memory = self.local.get_memory(str(memory_id))
        if memory is None:
            return PromotionReceipt(status="denied", origin_memory_id=str(memory_id),
                                    reasons=["unknown_memory"])

        text = str(memory.get("content") or "")
        verdict = self.policy.inspect(memory)
        preview = text[:160]

        if not verdict.allowed:
            return PromotionReceipt(
                status="denied",
                origin_memory_id=str(memory_id),
                reasons=verdict.reasons or ["policy_denied"],
                content_preview=preview,
            )

        if approval is None or not approval.is_valid():
            return PromotionReceipt(
                status="denied",
                origin_memory_id=str(memory_id),
                reasons=["approval_required"],
                content_preview=preview,
            )

        if self.shared is None or not self.config.fleet_enabled:
            return PromotionReceipt(status="disabled", origin_memory_id=str(memory_id),
                                    reasons=["shared_layer_not_configured"])

        if not self.config.shared_promotion_enabled:
            return PromotionReceipt(status="disabled", origin_memory_id=str(memory_id),
                                    reasons=["promotion_not_enabled_for_tenant"])

        final_text = verdict.redacted_text if verdict.redacted_text is not None else text
        fingerprint = content_fingerprint(final_text)

        existing = self._audit_lookup(fingerprint)
        if existing is not None:
            return PromotionReceipt(
                status="duplicate",
                origin_memory_id=str(memory_id),
                shared_memory_id=existing,
                reasons=["already_shared"],
                content_preview=preview,
            )

        if dry_run:
            return PromotionReceipt(
                status="dry_run",
                origin_memory_id=str(memory_id),
                reasons=["would_promote"],
                redactions=verdict.soft_matches,
                content_preview=str(final_text)[:160],
            )

        shared_id, _created = self.shared.add_memory(
            final_text,
            kind=str(memory.get("kind") or "semantic"),
            source_type="fleet_promotion",
            source_category=str(memory.get("source_category") or "AGENT_INFERENCE"),
            source_ref=str(memory_id),
            approval_state=SHARED_APPROVAL_STATE,
            context_mode=str(memory.get("context_mode") or "standalone"),
            scope={
                "layer": SHARED_LAYER,
                "origin_tenant": self.config.tenant,
                "origin_memory_id": str(memory_id),
            },
            entities=list(memory.get("entities") or []),
            applicable_systems=list(memory.get("applicable_systems") or []),
            confidence=float(memory.get("confidence") or 0.65),
            importance=float(memory.get("importance") or 0.5),
            trust=float(memory.get("trust") or 0.7),
        )

        self._audit_write(
            fingerprint=fingerprint,
            origin_memory_id=str(memory_id),
            shared_memory_id=str(shared_id),
            actor=str(approval.actor).strip(),
            reason=str(approval.reason).strip(),
            redactions=verdict.soft_matches,
        )

        return PromotionReceipt(
            status="promoted",
            origin_memory_id=str(memory_id),
            shared_memory_id=str(shared_id),
            reasons=["promoted"],
            redactions=verdict.soft_matches,
            content_preview=str(final_text)[:160],
        )

    # -- audit table ----------------------------------------------------------
    def _audit_lookup(self, fingerprint: str) -> str | None:
        if self.shared is None:
            return None
        row = self.shared._conn.execute(
            "SELECT shared_memory_id FROM fleet_promotions WHERE content_hash = ?",
            (fingerprint,),
        ).fetchone()
        return str(row[0]) if row else None

    def _audit_write(
        self,
        *,
        fingerprint: str,
        origin_memory_id: str,
        shared_memory_id: str,
        actor: str,
        reason: str,
        redactions: Sequence[str],
    ) -> None:
        assert self.shared is not None
        import uuid
        from datetime import datetime, timezone

        self.shared._conn.execute(
            "INSERT INTO fleet_promotions"
            " (id, content_hash, origin_memory_id, origin_tenant, shared_memory_id,"
            "  promoted_at, actor, reason, redactions_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                fingerprint,
                origin_memory_id,
                self.config.tenant,
                shared_memory_id,
                datetime.now(timezone.utc).isoformat(),
                actor,
                reason,
                json.dumps(list(redactions)),
            ),
        )
        self.shared._conn.commit()
