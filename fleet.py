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

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .preload import RecallCache
from .retrieval import (
    MemoryRetriever,
    RetrievalContext,
    RetrievalDiagnostics,
    RetrievalResult,
)
from .store import CortexStore

LOCAL_LAYER = "local"
SHARED_LAYER = "shared"

DEFAULT_SHARED_WEIGHT = 0.85
DEFAULT_LOCAL_LIMIT = 6
DEFAULT_SHARED_LIMIT = 4
DEFAULT_SHARED_MAX_SHARE = 0.5
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


def _origin_tenant(memory: dict[str, Any]) -> str:
    """Recover the promoting tenant recorded in a shared memory's scope."""

    raw = memory.get("scope")
    if raw is None:
        raw = memory.get("scope_json")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "{}")
        except ValueError:
            return ""
    if isinstance(raw, dict):
        return str(raw.get("origin_tenant") or "")
    return ""


def _shared_origin_memory_id(memory: dict[str, Any]) -> str:
    """The memory this shared row was promoted from, if it is recorded.

    Redaction rewrites the text on the way into the shared layer, so the content
    fingerprint changes and a bot can no longer recognise its own promoted fact
    by text alone. The provenance id survives that rewrite.
    """

    raw = memory.get("scope")
    if raw is None:
        raw = memory.get("scope_json")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "{}")
        except ValueError:
            raw = {}
    if isinstance(raw, dict):
        origin = str(raw.get("origin_memory_id") or "").strip()
        if origin:
            return origin
    return str(memory.get("source_ref") or "").strip()


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

    def denied_for_read(self, text: str, *, origin_tenant: str = "") -> bool:
        """Read-side filter for the shared layer.

        Promotion is the gate, but a shared row can still arrive out of band
        (manual SQL, a future writer, a policy tightened after admission). Any
        finance or credential hit — or content whose originating tenant is
        deny-listed — is dropped at recall time, so a bad row can never reach a
        prompt even if it reached the file.
        """

        if str(origin_tenant or "").strip().lower() in self.deny_tenants:
            return True
        return bool(self.inspect_text(text).hard_matches)


#: Tables this module owns. They are created in the *bot's* database (outbox +
#: shared-usage ledger) and in the shared database (admission audit), never by
#: ``cortex_schema``: that keeps the multi-tenant feature from forcing a schema
#: migration on any existing brain.
LOCAL_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS fleet_promotion_requests (
    request_id TEXT PRIMARY KEY,
    tenant TEXT NOT NULL,
    origin_memory_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    content TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'semantic',
    source_type TEXT NOT NULL DEFAULT 'conversation',
    verdict_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending'
      CHECK(status IN ('pending','denied','admitted','failed')),
    requested_at TEXT NOT NULL,
    decided_at TEXT,
    actor TEXT,
    reason TEXT,
    shared_memory_id TEXT,
    note TEXT
);
CREATE INDEX IF NOT EXISTS idx_fleet_requests_status
  ON fleet_promotion_requests(tenant,status,requested_at DESC);
CREATE INDEX IF NOT EXISTS idx_fleet_requests_origin
  ON fleet_promotion_requests(tenant,origin_memory_id);

CREATE TABLE IF NOT EXISTS fleet_shared_usage (
    usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    query TEXT,
    score REAL,
    event TEXT NOT NULL DEFAULT 'injected',
    session_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fleet_usage_memory
  ON fleet_shared_usage(tenant,memory_id,created_at DESC);
"""

SHARED_AUDIT_TABLE_SQL = """
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

SHARED_SOURCE_TYPE = "fleet_promotion"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _ensure_local_tables(store: CortexStore) -> None:
    with store.transaction():
        store._conn.executescript(LOCAL_TABLES_SQL)


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
    shared_max_share: float = DEFAULT_SHARED_MAX_SHARE
    shared_pool: int = DEFAULT_SHARED_LIMIT
    shared_min_score: float = 0.0
    shared_promotion_enabled: bool = False
    retrieval_threshold: float = 0.16
    semantic_weight: float = 0.0
    semantic_pool: int = 20
    disabled_reason: str = ""

    @property
    def fleet_enabled(self) -> bool:
        return self.shared_db is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "tenant": self.tenant,
            "local_db": str(self.local_db),
            "shared_db": str(self.shared_db) if self.shared_db else None,
            "shared_weight": self.shared_weight,
            "shared_max_share": self.shared_max_share,
            "shared_pool": self.shared_pool,
            "shared_min_score": self.shared_min_score,
            "shared_promotion_enabled": self.shared_promotion_enabled,
            "fleet_enabled": self.fleet_enabled,
            "disabled_reason": self.disabled_reason,
        }


def _annotate_result(result: RetrievalResult, *, layer: str, origin_tenant: str) -> RetrievalResult:
    """Tag a retrieval hit with the layer that answered.

    Newer builds declare ``layer``/``origin_tenant`` on the dataclass; older ones
    do not, and a deployment running the older retrieval module must still be
    able to say where a hit came from. Setting the attributes afterwards keeps
    both generations working, and every reader uses ``getattr`` with a default.
    """

    try:
        result.layer = layer
        result.origin_tenant = origin_tenant
    except (AttributeError, TypeError):
        # Some builds ship a frozen dataclass; object.__setattr__ still works and
        # the tag is what keeps a shared id out of the FK-bound local tables.
        try:
            object.__setattr__(result, "layer", layer)
            object.__setattr__(result, "origin_tenant", origin_tenant)
        except (AttributeError, TypeError):  # pragma: no cover - slotted class
            pass
    return result


def open_read_only_store(path: Path) -> CortexStore:
    """Open a store read-only, with a fallback for stores that predate the kwarg.

    ``CortexStore(read_only=...)`` only exists in newer builds; deployments that
    still ship the older store must not fail to start the shared layer. Those
    get a normal store with SQLite's query_only pragma set, which refuses writes
    from this connection just the same.
    """

    try:
        return CortexStore(path, read_only=True)
    except TypeError:
        store = CortexStore(path, initialize_schema=False)
        try:
            store._conn.execute("PRAGMA query_only=ON")
        except sqlite3.Error:
            pass
        return store


def tenant_from_home(hermes_home: str | Path) -> str:
    """``~/.hermes/profiles/teknium`` -> ``teknium``; the shared home -> ``default``."""

    home = Path(hermes_home).expanduser()
    return home.name if home.parent.name == "profiles" else "default"


def _clamp(value: Any, default: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


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
    disabled_reason = ""
    if raw_shared not in (None, "", False):
        raw_shared = str(raw_shared).replace("${HERMES_HOME}", str(home)).replace("$HERMES_HOME", str(home))
        candidate = Path(raw_shared).expanduser()
        if candidate.resolve() == local_db.resolve():
            # Two roles on one file is the failure that looks like it works:
            # every "shared" write is really private and every private write is
            # fleet-visible. Refuse the layered path instead of faking it.
            disabled_reason = "shared_db_path resolves to the local database"
        elif candidate.parent == candidate:
            disabled_reason = "shared_db_path is not a file path"
        else:
            shared_db = candidate
    else:
        disabled_reason = "shared_db_path is not configured"

    return FleetConfig(
        tenant=str(tenant or tenant_from_home(home)).strip(),
        local_db=local_db,
        shared_db=shared_db,
        shared_weight=_clamp(config.get("shared_weight"), DEFAULT_SHARED_WEIGHT, 0.0, 1.5),
        shared_max_share=_clamp(
            config.get("shared_max_share"), DEFAULT_SHARED_MAX_SHARE, 0.0, 1.0
        ),
        shared_pool=int(_clamp(config.get("shared_pool"), DEFAULT_SHARED_LIMIT, 1, 20)),
        shared_min_score=_clamp(config.get("shared_min_score"), 0.0, 0.0, 1.0),
        shared_promotion_enabled=bool(config.get("shared_promotion_enabled", False)),
        retrieval_threshold=_clamp(config.get("retrieval_threshold"), 0.16, 0.0, 1.0),
        semantic_weight=_clamp(config.get("semantic_fusion_weight"), 0.0, 0.0, 1.0),
        semantic_pool=int(_clamp(config.get("semantic_fusion_pool"), 20, 1, 200)),
        disabled_reason=disabled_reason,
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
    #: Tenant that promoted the shared row (empty for local hits).
    origin_tenant: str = ""

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
        self._lock = threading.RLock()
        self._last_shared_ids: set[str] = set()
        _ensure_local_tables(self._local)
        if config.fleet_enabled and open_shared and config.shared_db is not None:
            self._shared = self._open_shared(config.shared_db)
            if self._shared is not None:
                self._shared_retriever = MemoryRetriever(
                    self._shared,
                    threshold=config.retrieval_threshold,
                    semantic_weight=0.0,  # embeddings are local-only for now
                    semantic_pool=config.semantic_pool,
                )

    @staticmethod
    def _open_shared(path: Path) -> CortexStore | None:
        """Open the shared layer read-only, bootstrapping an empty file once.

        A fresh fleet has no shared database yet, so the first bot to start must
        create it. That happens exactly once (file absent), and every later open
        is read-only. A broken shared file must never take recall down, hence
        the ``None`` return instead of an exception.
        """

        if not path.exists():
            try:
                bootstrap = CortexStore(path)
                bootstrap.close()
            except (sqlite3.Error, OSError):
                pass
        try:
            return open_read_only_store(path)
        except (sqlite3.Error, OSError):
            return None

    # -- lifecycle ------------------------------------------------------------
    @property
    def local_store(self) -> CortexStore:
        return self._local

    @property
    def shared_store(self) -> CortexStore | None:
        return self._shared

    @property
    def recall_cache(self) -> RecallCache:
        # One cache, keyed on both layer revisions; the provider sets the TTL
        # through this property exactly as it does for a single retriever.
        return self._local_retriever.recall_cache

    @property
    def store(self) -> CortexStore:
        return self._local

    @property
    def threshold(self) -> float:
        return self._local_retriever.threshold

    @property
    def semantic_weight(self) -> float:
        return self._local_retriever.semantic_weight

    @property
    def semantic_pool(self) -> int:
        return self._local_retriever.semantic_pool

    @property
    def semantic_model_id(self) -> str | None:
        return self._local_retriever.semantic_model_id

    @property
    def semantic_floor(self) -> float:
        return self._local_retriever.semantic_floor

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
        hits, _diagnostics = self._layered(
            query,
            limit=int(local_limit or limit),
            token_budget=token_budget,
            include_archived=include_archived,
            context=context,
            graph_depth=graph_depth,
        )
        return hits

    # -- retriever-compatible surface ----------------------------------------
    def search(self, query: str, **options: Any) -> list[RetrievalResult]:
        results, _diagnostics = self.search_detailed(query, **options)
        return results

    def search_detailed(
        self, query: str, **options: Any
    ) -> tuple[list[RetrievalResult], RetrievalDiagnostics]:
        limit = int(options.pop("limit", DEFAULT_LOCAL_LIMIT) or DEFAULT_LOCAL_LIMIT)
        token_budget = int(options.pop("token_budget", 700) or 700)
        hits, diagnostics = self._layered(
            query, limit=limit, token_budget=token_budget, **options
        )
        results = [
            _annotate_result(
                RetrievalResult(
                    memory=hit.memory,
                    score=hit.score,
                    components=hit.components,
                    estimated_tokens=hit.estimated_tokens,
                ),
                layer=hit.layer,
                origin_tenant=hit.origin_tenant,
            )
            for hit in hits
        ]
        return results, diagnostics

    def search_cached(
        self, query: str, *, speculative: bool = False, **options: Any
    ) -> tuple[list[RetrievalResult], RetrievalDiagnostics]:
        cache = self.recall_cache
        settings = (
            "fleet",
            self.config.tenant,
            self.config.shared_weight,
            self.config.shared_pool,
            self.config.shared_max_share,
            self.config.shared_min_score,
            self.threshold,
        )
        key = cache.key(query, options, settings)
        started = time.perf_counter()
        computed = False

        def revision() -> tuple[Any, Any]:
            shared: Any = (0, 0)
            if self._shared is not None:
                try:
                    shared = self._shared.retrieval_revision()
                except sqlite3.Error:
                    shared = (0, 0)
            return (self._local.retrieval_revision(), shared)

        def compute():
            nonlocal computed
            computed = True
            results, diagnostics = self.search_detailed(query, **options)
            return results, replace(diagnostics, preloaded=speculative)

        results, diagnostics = cache.search(key, revision, compute, speculative=speculative)
        if not computed:
            diagnostics = replace(
                diagnostics,
                cache_hit=True,
                stage_ms={
                    **dict.fromkeys(diagnostics.stage_ms, 0.0),
                    "cache_lookup": (time.perf_counter() - started) * 1000,
                },
            )
        return results, diagnostics

    # -- merge ----------------------------------------------------------------
    def _layered(
        self,
        query: str,
        *,
        limit: int,
        token_budget: int,
        **options: Any,
    ) -> tuple[list[FleetHit], RetrievalDiagnostics]:
        started = time.perf_counter()
        local_results, local_diagnostics = self._local_retriever.search_detailed(
            query, limit=limit, token_budget=token_budget, **options
        )
        hits: list[FleetHit] = [
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
        shared_diagnostics = RetrievalDiagnostics(0, 0, 0, True)
        if self._shared_retriever is not None and limit > 0:
            shared_limit = max(1, min(limit, int(self.config.shared_pool)))
            base_threshold = options.get("threshold")
            base_threshold = (
                self.threshold if base_threshold is None else float(base_threshold)
            )
            shared_options = {k: v for k, v in options.items() if k != "threshold"}
            try:
                shared_results, shared_diagnostics = self._shared_retriever.search_detailed(
                    query,
                    limit=shared_limit,
                    token_budget=token_budget,
                    threshold=max(base_threshold, self.config.shared_min_score),
                    **shared_options,
                )
            except (sqlite3.Error, OSError, ValueError):
                # The shared layer never takes recall down with it.
                shared_results = []
            local_ids = {str(result.memory.get("id")) for result in local_results}
            local_fingerprints = {
                content_fingerprint(str(result.memory.get("content") or ""))
                for result in local_results
            }
            for result in shared_results:
                text = str(result.memory.get("content") or "")
                origin = _origin_tenant(result.memory)
                if str(result.memory.get("id")) in local_ids:
                    continue
                if content_fingerprint(text) in local_fingerprints:
                    continue
                # This tenant's own fact, already held privately. Redaction can
                # change the shared text (an address, an email), so the content
                # fingerprint no longer matches — match the provenance instead,
                # otherwise the bot sees its own fact twice, once labelled
                # foreign.
                origin_memory_id = _shared_origin_memory_id(result.memory)
                if origin_memory_id and origin_memory_id in local_ids:
                    continue
                if origin_memory_id:
                    try:
                        owns_it = self._local.get_memory(origin_memory_id) is not None
                    except (sqlite3.Error, ValueError):
                        owns_it = False
                    if owns_it:
                        continue
                if self.policy.denied_for_read(text, origin_tenant=origin):
                    continue
                hits.append(
                    FleetHit(
                        memory=dict(result.memory),
                        score=float(result.score) * self.config.shared_weight,
                        layer=SHARED_LAYER,
                        tenant=self.config.tenant,
                        components={**dict(result.components), "layer_weight": self.config.shared_weight},
                        estimated_tokens=int(result.estimated_tokens),
                        origin_tenant=origin,
                    )
                )
        merged = self._merge(hits, limit=limit, token_budget=token_budget)
        with self._lock:
            self._last_shared_ids = {
                str(hit.memory.get("id")) for hit in merged if hit.layer == SHARED_LAYER
            }
        stage_ms = {
            **{f"local.{key}": value for key, value in local_diagnostics.stage_ms.items()},
            **{f"shared.{key}": value for key, value in shared_diagnostics.stage_ms.items()},
            "layer_merge": (time.perf_counter() - started) * 1000,
        }
        diagnostics = RetrievalDiagnostics(
            candidate_count=local_diagnostics.candidate_count + shared_diagnostics.candidate_count,
            selected_count=len(merged),
            estimated_tokens=sum(max(1, hit.estimated_tokens) for hit in merged),
            abstained=not merged,
            candidate_decisions=local_diagnostics.candidate_decisions,
            stage_ms=stage_ms,
        )
        return merged, diagnostics

    def _merge(self, hits: Sequence[FleetHit], *, limit: int, token_budget: int) -> list[FleetHit]:
        """Dedupe by content, prefer local, cap the shared share, fit the budget."""

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

        max_shared = max(0, int(round(limit * max(0.0, min(1.0, self.config.shared_max_share)))))
        selected: list[FleetHit] = []
        spent = 0
        shared_taken = 0
        for hit in ordered:
            cost = max(1, int(hit.estimated_tokens or 1))
            if len(selected) >= limit:
                break
            if hit.layer == SHARED_LAYER and shared_taken >= max_shared:
                # A large shared corpus must not crowd out a bot's own memory.
                continue
            if spent + cost > token_budget:
                continue
            selected.append(hit)
            spent += cost
            if hit.layer == SHARED_LAYER:
                shared_taken += 1
        return selected

    # -- audit helpers --------------------------------------------------------
    def is_shared(self, memory_id: str) -> bool:
        with self._lock:
            return str(memory_id) in self._last_shared_ids

    def shared_revision(self) -> Any:
        """Cheap change marker for the shared layer (used as a cache key part)."""

        if self._shared is None:
            return (0, 0)
        try:
            return self._shared.retrieval_revision()
        except sqlite3.Error:
            return (0, 0)

    def audit_results(
        self,
        results: Sequence[RetrievalResult],
        *,
        query: str = "",
        session_id: str = "",
    ) -> list[RetrievalResult]:
        """Record shared hits locally and return the FK-safe local subset.

        ``access_log`` and ``usage_records`` carry a foreign key to ``memories``,
        so writing a shared memory id there fails. Shared usage is recorded in
        this module's own table instead — which also means the bot's statistics
        never require write access to the shared file.
        """

        shared = [result for result in results if getattr(result, "layer", LOCAL_LAYER) != LOCAL_LAYER]
        if not shared:
            return list(results)
        rows = [
            (
                self.config.tenant,
                str(result.memory["id"]),
                query,
                float(result.score),
                "injected",
                session_id,
                _now(),
            )
            for result in shared
        ]
        try:
            with self._local.transaction():
                self._local._conn.executemany(
                    "INSERT INTO fleet_shared_usage(tenant,memory_id,query,score,event,session_id,created_at) "
                    "VALUES(?,?,?,?,?,?,?)",
                    rows,
                )
        except sqlite3.Error:
            # Accounting must never break a turn.
            pass
        return [result for result in results if getattr(result, "layer", LOCAL_LAYER) == LOCAL_LAYER]

    # -- promotion ------------------------------------------------------------
    def _gate(self) -> tuple["PromotionGate", bool]:
        if self.config.shared_db is None:
            return (
                PromotionGate(
                    config=self.config,
                    policy=self.policy,
                    local_store=self._local,
                    shared_store=None,
                ),
                False,
            )
        # Operator action: open the shared file read-write for the length of one
        # promotion, then close it. Recall never holds this handle.
        shared = CortexStore(self.config.shared_db)
        with shared.transaction():
            shared._conn.executescript(SHARED_AUDIT_TABLE_SQL)
        return (
            PromotionGate(
                config=self.config,
                policy=self.policy,
                local_store=self._local,
                shared_store=shared,
            ),
            True,
        )

    def promote(
        self,
        memory_id: str,
        *,
        approval: Approval | None,
        dry_run: bool = False,
    ) -> "PromotionReceipt":
        gate, owned = self._gate()
        try:
            return gate.promote(memory_id, approval=approval, dry_run=dry_run)
        finally:
            if owned and gate.shared is not None:
                gate.shared.close()

    def request_promotion(self, memory_id: str, *, reason: str = "") -> dict[str, Any]:
        """Bot-side promotion path: queue a request, never write the shared file."""

        if not self.config.shared_promotion_enabled:
            return {
                "success": False,
                "status": "disabled",
                "error": "promotion is disabled for this tenant "
                         "(set shared_promotion_enabled: true under plugins.entries.cortex)",
            }
        memory = self._local.get_memory(str(memory_id))
        if memory is None:
            return {"success": False, "status": "unknown_memory", "error": f"unknown memory {memory_id}"}
        verdict = self.policy.inspect(memory)
        allowed = bool(verdict.allowed)
        if allowed:
            final_text = (
                verdict.redacted_text
                if verdict.redacted_text is not None
                else str(memory.get("content") or "")
            )
        else:
            # Denied content is never copied into the outbox: a second copy of a
            # balance is a second place to leak it from.
            final_text = "[withheld: denied by promotion policy]"
        fingerprint = content_fingerprint(final_text)
        existing = self._local._conn.execute(
            "SELECT request_id,status FROM fleet_promotion_requests "
            "WHERE tenant=? AND content_hash=? AND status IN ('pending','admitted') "
            "ORDER BY requested_at DESC LIMIT 1",
            (self.config.tenant, fingerprint),
        ).fetchone()
        if existing is not None:
            return {
                "success": True,
                "status": str(existing["status"]),
                "request_id": str(existing["request_id"]),
                "duplicate": True,
            }
        request_id = str(uuid.uuid4())
        status = "pending" if verdict.allowed else "denied"
        with self._local.transaction():
            self._local._conn.execute(
                "INSERT INTO fleet_promotion_requests("
                "request_id,tenant,origin_memory_id,content_hash,content,kind,source_type,"
                "verdict_json,status,requested_at,reason,note) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    request_id,
                    self.config.tenant,
                    str(memory_id),
                    fingerprint,
                    final_text,
                    str(memory.get("kind") or "semantic"),
                    str(memory.get("source_type") or "conversation"),
                    json.dumps(verdict.as_dict(), ensure_ascii=False),
                    status,
                    _now(),
                    str(reason or ""),
                    "" if verdict.allowed else "; ".join(verdict.reasons),
                ),
            )
        return {
            "success": bool(verdict.allowed),
            "status": status,
            "request_id": request_id,
            "verdict": verdict.as_dict(),
            "duplicate": False,
        }

    def promotions(self, *, status: str | None = "pending", limit: int = 50) -> list[dict[str, Any]]:
        sql = "SELECT * FROM fleet_promotion_requests WHERE tenant=?"
        args: list[Any] = [self.config.tenant]
        if status:
            sql += " AND status=?"
            args.append(status)
        sql += " ORDER BY requested_at DESC LIMIT ?"
        args.append(max(1, int(limit)))
        return [dict(row) for row in self._local._conn.execute(sql, args).fetchall()]

    def admit_request(
        self,
        request_id: str,
        *,
        actor: str,
        reason: str,
        dry_run: bool = False,
    ) -> "PromotionReceipt":
        row = self._local._conn.execute(
            "SELECT * FROM fleet_promotion_requests WHERE request_id=? AND tenant=?",
            (request_id, self.config.tenant),
        ).fetchone()
        if row is None:
            return PromotionReceipt(status="denied", reasons=["unknown_request"])
        if str(row["status"]) != "pending":
            return PromotionReceipt(status=str(row["status"]), reasons=["request_not_pending"])
        approval = Approval(actor=actor, reason=reason)
        gate, owned = self._gate()
        try:
            receipt = gate.promote_content(
                str(row["content"]),
                origin_memory_id=str(row["origin_memory_id"]),
                kind=str(row["kind"] or "semantic"),
                source_type=str(row["source_type"] or ""),
                approval=approval,
                dry_run=dry_run,
            )
        finally:
            if owned and gate.shared is not None:
                gate.shared.close()
        if receipt.promoted:
            with self._local.transaction():
                self._local._conn.execute(
                    "UPDATE fleet_promotion_requests SET status='admitted',decided_at=?,"
                    "actor=?,reason=?,shared_memory_id=? WHERE request_id=?",
                    (_now(), approval.actor, approval.reason, receipt.shared_memory_id, request_id),
                )
        return receipt

    def status(self) -> dict[str, Any]:
        payload: dict[str, Any] = {**self.config.as_dict()}
        try:
            rows = self._local._conn.execute(
                "SELECT status,COUNT(*) FROM fleet_promotion_requests WHERE tenant=? GROUP BY status",
                (self.config.tenant,),
            ).fetchall()
            payload["requests"] = {str(status): int(count) for status, count in rows}
        except sqlite3.Error:
            payload["requests"] = {}
        shared_memories = 0
        if self._shared is not None:
            try:
                row = self._shared._conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE source_type=?", (SHARED_SOURCE_TYPE,)
                ).fetchone()
                shared_memories = int(row[0]) if row else 0
            except sqlite3.Error:
                shared_memories = -1
        payload["shared_memories"] = shared_memories
        return payload

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
        with store.transaction():
            store._conn.executescript(SHARED_AUDIT_TABLE_SQL)


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
        verdict = self.policy.inspect(memory)
        if not verdict.allowed:
            return PromotionReceipt(
                status="denied",
                origin_memory_id=str(memory_id),
                reasons=verdict.reasons or ["policy_denied"],
                content_preview=str(memory.get("content") or "")[:160],
            )
        final_text = verdict.redacted_text if verdict.redacted_text is not None else str(memory.get("content") or "")
        return self.promote_content(
            final_text,
            origin_memory_id=str(memory_id),
            kind=str(memory.get("kind") or "semantic"),
            source_type=str(memory.get("source_type") or ""),
            approval=approval,
            dry_run=dry_run,
            approved_verdict=verdict,
            context_mode=str(memory.get("context_mode") or "standalone"),
            entities=list(memory.get("entities") or []),
            applicable_systems=list(memory.get("applicable_systems") or []),
            confidence=float(memory.get("confidence") or 0.65),
            importance=float(memory.get("importance") or 0.5),
            trust=float(memory.get("trust") or 0.7),
            source_category=str(memory.get("source_category") or "AGENT_INFERENCE"),
        )

    def promote_content(
        self,
        content: str,
        *,
        origin_memory_id: str,
        approval: Approval | None,
        kind: str = "semantic",
        source_type: str = "",
        dry_run: bool = False,
        approved_verdict: PromotionVerdict | None = None,
        context_mode: str = "standalone",
        entities: Sequence[str] = (),
        applicable_systems: Sequence[str] = (),
        confidence: float = 0.65,
        importance: float = 0.5,
        trust: float = 0.7,
        source_category: str = "AGENT_INFERENCE",
    ) -> PromotionReceipt:
        """Admit one piece of text. The policy is re-applied here even when the
        caller already inspected it: admission is the security boundary, so it
        never trusts an earlier verdict from another process."""

        text = str(content or "")
        preview = text[:160]
        verdict = self.policy.inspect_text(text) if approved_verdict is None else approved_verdict
        if not verdict.allowed:
            return PromotionReceipt(
                status="denied",
                origin_memory_id=str(origin_memory_id),
                reasons=verdict.reasons or ["policy_denied"],
                content_preview=preview,
            )
        if verdict.redacted_text is not None and verdict.redacted_text != text:
            text = verdict.redacted_text
            preview = text[:160]

        if approval is None or not approval.is_valid():
            return PromotionReceipt(
                status="denied",
                origin_memory_id=str(origin_memory_id),
                reasons=["approval_required"],
                content_preview=preview,
            )

        if self.shared is None or not self.config.fleet_enabled:
            return PromotionReceipt(status="disabled", origin_memory_id=str(origin_memory_id),
                                    reasons=["shared_layer_not_configured"])

        if not self.config.shared_promotion_enabled:
            return PromotionReceipt(status="disabled", origin_memory_id=str(origin_memory_id),
                                    reasons=["promotion_not_enabled_for_tenant"])

        fingerprint = content_fingerprint(text)
        existing = self._audit_lookup(fingerprint)
        if existing is not None:
            return PromotionReceipt(
                status="duplicate",
                origin_memory_id=str(origin_memory_id),
                shared_memory_id=existing,
                reasons=["already_shared"],
                content_preview=preview,
            )

        if dry_run:
            return PromotionReceipt(
                status="dry_run",
                origin_memory_id=str(origin_memory_id),
                reasons=["would_promote"],
                redactions=verdict.soft_matches,
                content_preview=preview,
            )

        shared_id, _created = self.shared.add_memory(
            text,
            kind=str(kind or "semantic"),
            source_type=SHARED_SOURCE_TYPE,
            source_category=source_category,
            origin_source_category="OPERATOR_APPROVED",
            source_ref=str(origin_memory_id),
            approval_state=SHARED_APPROVAL_STATE,
            context_mode=context_mode,
            scope={
                "layer": SHARED_LAYER,
                "origin_tenant": self.config.tenant,
                "origin_memory_id": str(origin_memory_id),
            },
            entities=list(entities or ()),
            applicable_systems=list(applicable_systems or ()),
            confidence=confidence,
            importance=importance,
            trust=trust,
            uniqueness=0.6,
            storage_policy="trusted",
            extraction_method="fleet_promotion",
        )

        self._audit_write(
            fingerprint=fingerprint,
            origin_memory_id=str(origin_memory_id),
            shared_memory_id=str(shared_id),
            actor=str(approval.actor).strip(),
            reason=str(approval.reason).strip(),
            redactions=verdict.soft_matches,
        )

        return PromotionReceipt(
            status="promoted",
            origin_memory_id=str(origin_memory_id),
            shared_memory_id=str(shared_id),
            reasons=["promoted"],
            redactions=verdict.soft_matches,
            content_preview=preview,
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


# ---------------------------------------------------------------------------
# Operator CLI: the curator side of the promotion path
# ---------------------------------------------------------------------------


def _load_fleet(hermes_home: Path, tenant: str | None = None) -> FleetMemory:
    from .hermes_config import DEFAULTS, _read_config

    block = {**DEFAULTS, **_read_config(hermes_home)}
    resolved_tenant = str(tenant or block.get("fleet_tenant") or tenant_from_home(hermes_home))
    return FleetMemory(
        resolve_fleet_config(
            hermes_home=hermes_home, tenant=resolved_tenant, plugin_config=block
        )
    )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cortex.fleet",
        description="Inspect and curate Cortex's shared fleet memory layer.",
    )
    parser.add_argument(
        "--home",
        default=os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes"),
        help="Hermes home/profile directory (default: $HERMES_HOME or ~/.hermes)",
    )
    parser.add_argument("--tenant", default=None, help="Override the tenant id")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show resolved config, request counts, shared size")
    audit = sub.add_parser("audit", help="Classify local memories for promotability")
    audit.add_argument("--limit", type=int, default=None)

    requests = sub.add_parser("requests", help="List queued promotion requests")
    requests.add_argument("--status", default="pending")
    requests.add_argument("--limit", type=int, default=50)

    promote = sub.add_parser("promote", help="Promote one memory by id (operator path)")
    promote.add_argument("memory_id")
    promote.add_argument("--actor", required=True)
    promote.add_argument("--reason", required=True)
    promote.add_argument("--dry-run", action="store_true")

    admit = sub.add_parser("admit", help="Admit a bot's queued promotion request")
    admit.add_argument("request_id")
    admit.add_argument("--actor", required=True)
    admit.add_argument("--reason", required=True)
    admit.add_argument("--dry-run", action="store_true")

    deny = sub.add_parser("deny", help="Reject a bot's queued promotion request")
    deny.add_argument("request_id")
    deny.add_argument("--actor", required=True)
    deny.add_argument("--reason", required=True)

    args = parser.parse_args(list(argv) if argv is not None else None)
    memory = _load_fleet(Path(args.home).expanduser(), args.tenant)
    try:
        if args.command == "status":
            payload = memory.status()
        elif args.command == "audit":
            payload = memory.audit(limit=args.limit)
        elif args.command == "requests":
            payload = {"requests": memory.promotions(status=args.status, limit=args.limit)}
        elif args.command == "promote":
            payload = memory.promote(
                args.memory_id,
                approval=Approval(actor=args.actor, reason=args.reason),
                dry_run=args.dry_run,
            ).as_dict()
        elif args.command == "admit":
            payload = memory.admit_request(
                args.request_id,
                actor=args.actor,
                reason=args.reason,
                dry_run=args.dry_run,
            ).as_dict()
        elif args.command == "deny":
            with memory.local_store.transaction():
                cursor = memory.local_store._conn.execute(
                    "UPDATE fleet_promotion_requests SET status='denied',decided_at=?,actor=?,reason=? "
                    "WHERE request_id=? AND tenant=? AND status='pending'",
                    (_now(), args.actor, args.reason, args.request_id, memory.config.tenant),
                )
            payload = {"status": "denied" if cursor.rowcount else "not_pending",
                       "request_id": args.request_id}
        else:  # pragma: no cover - argparse enforces the set
            parser.error(f"unknown command {args.command}")
            return 2
    finally:
        memory.close()
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    if isinstance(payload, dict) and payload.get("status") in {"denied", "disabled", "failed"}:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
