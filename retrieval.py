"""Fast deterministic retrieval and utility scoring for Cortex."""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Iterable

from .cortex_trace import (
    elapsed_ms as _trace_elapsed_ms,
    phase_start as _trace_phase_start,
    record_phase as _trace_record,
    trace_enabled as _trace_enabled,
)
from .precompute import Precompute, revision_of
from .store_read import select
from .preload import RecallCache
from .source_authority import cap_conflicts
from .abstention import should_abstain
from .cognition import query_class
from .density import by_confidence
from .diversity import enforce_distinct
from .entity_anchor import (
    entity_memories,
    entity_memory_ids,
    extract_entities,
    known_entities,
)
from .stability import apply_stability_blend
from .latency_budget import budget_from
from .namespace import channel_namespaces, filter_results
from .negative_cues import excluded_terms, suppress
from .usage_weight import apply_usage_boost
from . import decay, edge_quarantine, near_miss, serial_position  # (#80, #82, #84)
from .rerank import DEFAULT_TOP_N, rerank
from .validity import apply_validity_penalty, superseded_ids
from .semantics import feature_similarity
from .store import CortexStore, query_tokens


_HALF_LIFE_DAYS = {
    "identity": 3650.0,
    "preference": 1825.0,
    "decision": 365.0,
    "procedure": 365.0,
    "semantic": 180.0,
    "episode": 60.0,
    "operational": 21.0,
    "prospective": 365.0,
    "schema": 730.0,
}

_SOURCE_RELIABILITY = {
    "TOOL_VERIFIED": 0.95,
    "USER_EXPLICIT": 0.90,
    "USER_STATED": 0.84,
    "DOCUMENT_EXTRACTED": 0.82,
    # Review is a useful governance signal, but it is not independent
    # verification of the underlying claim.
    "OPERATOR_APPROVED": 0.76,
    "AUTOMATIC_APPROVED": 0.70,
    "REFLECTION": 0.62,
    "AGENT_INFERENCE": 0.52,
    "AGENT_PROPOSED": 0.48,
}

_MEMORY_TYPE_PRIOR = {
    "identity": 0.95,
    "preference": 0.90,
    "procedure": 0.86,
    "prospective": 0.82,
    "decision": 0.76,
    "semantic": 0.66,
    "operational": 0.58,
    "episode": 0.48,
    "schema": 0.82,
}

# Keep scoring policies explicit and versioned so historical traces can be
# interpreted honestly. The semantic/graph policy is shadow-only: live recall
# continues to use the current policy until outcome evidence supports a switch.
LIVE_SCORING_POLICY_VERSION = "retrieval_live_v2"
SHADOW_SCORING_POLICY_VERSION = "semantic_graph_shadow_v1"

# Budgets at or above this are preload-scale and are spent by density.
DENSITY_MIN_BUDGET = 2000

# Association is weaker than lexical evidence, so the entity bonus stays small.
# Measured on the frozen set: 0.02/0.03 give 0.6167/0.2917/0.4389,
# 0.06 gives 0.6167/0.2903/0.4410, 0.10 gives 0.6208/0.2934/0.4403. The spread is one case
# of hit@6, i.e. noise on 240 labels; 0.10 is kept because it is not worse and it closes a
# real hole (query entities were never extracted at all).
try:
    ENTITY_BOOST = float(os.environ.get("CORTEX_ENTITY_BOOST", "0.10"))
except ValueError:
    ENTITY_BOOST = 0.10
SCORE_SIGNAL_WEIGHTS = {
    "lexical": 0.20,
    "phrase": 0.06,
    "semantic": 0.06,
    "activation": 0.09,
    "utility": 0.11,
    "importance": 0.06,
    "confidence": 0.05,
    "source_reliability": 0.06,
    "currentness": 0.06,
    "memory_type": 0.03,
    "uniqueness": 0.02,
    "graph": 0.06,
    "neighborhood": 0.04,
}
SHADOW_SCORE_SIGNAL_WEIGHTS = {
    **SCORE_SIGNAL_WEIGHTS,
    "lexical": 0.15,
    "semantic": 0.11,
    "importance": 0.05,
    "confidence": 0.04,
    "graph": 0.08,
}
SCORE_PENALTIES = {
    "relevance": 0.12,
    "stale_risk": 0.08,
    "wrong_rate": 0.12,
}
SHADOW_SCORE_PENALTIES = {
    **SCORE_PENALTIES,
    "relevance": 0.16,
}
_GRAPH_EXPANSION_WEIGHT = 0.10
_SHADOW_GRAPH_EXPANSION_WEIGHT = 0.14

# Reciprocal-rank-fusion constant for combining the semantic ranking with the
# native scoring. 60 is the standard value from the RRF literature and is what
# the evaluation that justified this feature used (see plans/Cortex Semantic
# Fusion results, 2026-09-10).
_SEMANTIC_RRF_K = 60

# Shadow-only role tiering. This version string names the exact proposed
# policy compared against live retrieval; nothing here changes live results.
SHADOW_ROLE_POLICY_VERSION = "role_tier_shadow_v1"
_TECHNICAL_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)+|[a-z0-9]+_[a-z0-9_]+")


@dataclass(frozen=True)
class RetrievalContext:
    """Explicit task context used to validate scoped memory applicability."""

    active_project: str | None = None
    goal: str | None = None
    entities: tuple[str, ...] = ()
    scope: dict[str, str] = field(default_factory=dict)
    system_state: dict[str, str] = field(default_factory=dict)
    applicable_systems: tuple[str, ...] = ()
    applicable_versions: tuple[str, ...] = ()

    def as_record(self) -> dict[str, Any]:
        return {
            "active_project": self.active_project,
            "entities": list(self.entities),
            "scope": dict(self.scope),
            "system_state": dict(self.system_state),
            "applicable_systems": list(self.applicable_systems),
            "applicable_versions": list(self.applicable_versions),
        }


@dataclass(frozen=True)
class RetrievalResult:
    memory: dict[str, Any]
    score: float
    components: dict[str, float]
    estimated_tokens: int

    def as_dict(self) -> dict[str, Any]:
        source_category = str(self.memory.get("source_category") or "AGENT_INFERENCE")
        origin_source_category = str(
            self.memory.get("origin_source_category") or source_category
        )
        record = {
            "id": self.memory["id"],
            "kind": self.memory["kind"],
            "content": self.memory["content"],
            "state": self.memory["state"],
            "source_type": self.memory.get("source_type"),
            "source_category": source_category,
            "origin_source_category": origin_source_category,
            "source_ref": self.memory.get("source_ref"),
            "context_mode": str(self.memory.get("context_mode") or "standalone"),
            "scope": _memory_context_map(self.memory, "scope", "scope_json"),
            "entities": _memory_context_list(self.memory, "entities", "entities_json"),
            "preconditions": _memory_context_map(
                self.memory, "preconditions", "preconditions_json"
            ),
            "source_context": self.memory.get("source_context"),
            "applicable_systems": _memory_context_list(
                self.memory, "applicable_systems", "applicable_systems_json"
            ),
            "applicable_versions": _memory_context_list(
                self.memory, "applicable_versions", "applicable_versions_json"
            ),
            "score": round(self.score, 5),
            "components": {k: round(v, 5) for k, v in self.components.items()},
            "estimated_tokens": self.estimated_tokens,
        }
        if self.memory.get("approval_state") is not None:
            record["approval_state"] = str(self.memory["approval_state"])
        return record


@dataclass(frozen=True)
class RetrievalDiagnostics:
    candidate_count: int
    selected_count: int
    estimated_tokens: int
    abstained: bool
    candidate_decisions: tuple[dict[str, Any], ...] = ()
    # Live-path per-stage timings in milliseconds, keyed by pipeline stage
    # (fts, feature, context, neighborhood, prospective, merge, scoring,
    # graph, select). Informational only: summed stages approximate the
    # provider-measured prepare_ms for the same turn and let operators see
    # which stage dominates tail latency on their own host and corpus.
    stage_ms: dict[str, float] = field(default_factory=dict)
    cache_hit: bool = False
    preloaded: bool = False


@dataclass(frozen=True)
class _QuerySignals:
    tokens: frozenset[str]
    folded: str
    active_scope: dict[str, str]
    available_state: dict[str, str]
    entities: frozenset[str]
    systems: frozenset[str]
    versions: frozenset[str]
    specificity: float

    @classmethod
    def prepare(cls, query: str, context: RetrievalContext) -> "_QuerySignals":
        scope = _clean_context_map(context.scope)
        if context.active_project:
            scope["project"] = context.active_project
        entities = {item.casefold() for item in context.entities}
        if context.active_project:
            entities.add(context.active_project.casefold())
        entities.update(str(value).casefold() for value in scope.values())
        return cls(
            frozenset(query_tokens(query)),
            " ".join((query or "").casefold().split()),
            scope,
            {**scope, **_clean_context_map(context.system_state)},
            frozenset(entities),
            frozenset(item.casefold() for item in context.applicable_systems),
            frozenset(item.casefold() for item in context.applicable_versions),
            float(bool(re.search(
                r"\b(?:specific|specifics|example|examples|episode|episodes|"
                r"exact|detail|details|instance|instances|when exactly)\b", query, re.I
            ))),
        )


class MemoryRetriever:
    """FTS + utility + one-hop associative graph retrieval."""

    def __init__(
        self,
        store: CortexStore,
        *,
        threshold: float = 0.16,
        semantic_weight: float = 0.0,
        semantic_pool: int = 20,
        semantic_model_id: str | None = None,
        semantic_floor: float = 0.0,
        abstain_threshold: float | None = None,
        latency_budget_ms: float | None = None,
    ):
        self.store = store
        self.threshold = threshold
        # Off by default: no measured signal separates hits from misses yet
        # (see docs/ABSTENTION.md). Enabling this is an explicit choice.
        self.abstain_threshold = abstain_threshold
        # Deadline for candidate work; unset keeps recall fully deterministic.
        self.latency_budget_ms = latency_budget_ms
        # Stage-two lexical rerank over the top slice; deterministic, no model. OFF by
        # default: measured on the frozen set it gains 0.008 hit@6 at weight 0.05 and
        # costs 0.012 MRR, and degrades monotonically as the weight rises (0.20 -> -3.0
        # points of hit@6). Stage one already carries a 0.20 lexical weight with phrase
        # matching, so BM25 over full content mostly re-weights note length.
        self.rerank_enabled = False
        # Spend preload-scale budgets by density (CORTEX_DENSITY_BUDGET=0 to A/B).
        self.density_budget_enabled = os.environ.get("CORTEX_DENSITY_BUDGET", "1") != "0"
        # Anchor recall on entity names found in the query (CORTEX_ENTITY_ANCHOR=0 to A/B).
        self.entity_anchor_enabled = os.environ.get("CORTEX_ENTITY_ANCHOR", "1") != "0"
        # Semantic fusion is OFF by default: 0.0 reproduces the previous
        # behaviour exactly, so enabling it is an explicit, reversible choice.
        self.semantic_weight = max(0.0, float(semantic_weight))
        self.semantic_pool = max(1, int(semantic_pool))
        self.semantic_model_id = semantic_model_id
        # RRF rank is useful only after the embedding says the candidate is
        # directionally related. A zero floor excludes anti-correlated and
        # orthogonal vectors; deployments may raise it for stricter precision.
        self.semantic_floor = max(-1.0, min(1.0, float(semantic_floor)))
        self.recall_cache = RecallCache()
        # Query-independent inputs, shared across turns. Every entry here is a pure function
        # of the store revision, so there is no TTL: a time-based expiry would make recall
        # time-dependent, and a cache that changes an answer because five seconds passed is
        # a correctness bug wearing an optimisation's clothes.
        self.precompute = Precompute(ttl_seconds=float("inf"))

    def _edge_sets(self, relation: str) -> tuple[set[str], set[str]]:
        """(sources, destinations) of one edge relation, cached per store revision."""

        def compute() -> tuple[set[str], set[str]]:
            rows = select(
                self.store,
                "SELECT src_id, dst_id, relation, created_at FROM edges WHERE relation = ?",
                (relation,),
            )
            # A quarantined edge is recorded but does not score (#84). With no adoption epoch every
            # edge is grandfathered, so this is inert until somebody adopts one - the rule is about
            # the next thousand edges, not about rewriting the graph today.
            trusted = edge_quarantine.scoring_edges(
                self.store,
                [
                    {"src_id": str(row[0]), "dst_id": str(row[1]), "relation": str(row[2]),
                     "created_at": str(row[3]) if len(row) > 3 else None}
                    for row in rows
                ],
            )
            return ({row["src_id"] for row in trusted}, {row["dst_id"] for row in trusted})

        return self.precompute.get(
            f"edges:{relation}", compute, revision=revision_of(self.store)
        )

    def _anchor_names(self, query: str) -> list[str]:
        """Entities this query anchors on, memoised until the store revision changes.

        The name list is ~489 rows and only changes when memories change, so it is fetched
        per revision rather than per call: the cache key must not cost more than the work
        it exists to avoid.
        """
        try:
            revision = self.store.retrieval_revision()
        except Exception:  # noqa: BLE001 - a cache key must never break recall
            revision = None
        cached = getattr(self, "_anchor_cache", None)
        if cached and revision is not None and cached[0] == revision:
            names = cached[1]
        else:
            names = known_entities(self.store)
            self._anchor_cache = (revision, names)
        return extract_entities(query, names)

    def search_cached(
        self, query: str, *, speculative: bool = False, **options: Any
    ) -> tuple[list[RetrievalResult], RetrievalDiagnostics]:
        """Reuse exact recall only while both context and store revision match."""

        # Everything that can change a result belongs in the key: the scoring settings,
        # the on/off switches, the query's class and the entities it anchors on. Two turns
        # with the same words but a different class or anchor set must not share an entry.
        settings = (
            self.threshold, self.semantic_weight, self.semantic_pool,
            self.semantic_model_id, self.semantic_floor,
            bool(getattr(self, "entity_anchor_enabled", True)),
            float(ENTITY_BOOST),
            bool(getattr(self, "rerank_enabled", False)),
            bool(getattr(self, "density_budget_enabled", True)),
            getattr(self, "latency_budget_ms", None),
            getattr(self, "abstain_threshold", None),
            # Every policy switch belongs in the key, or an A/B comparison serves the
            # baseline's entry to the challenger (#25).
            bool(getattr(self, "usage_boost_enabled", True)),
            bool(getattr(self, "negative_cues_enabled", True)),
            bool(getattr(self, "session_scope_enabled", True)),
        )
        scoped = dict(options)
        scoped["__class__"] = query_class(query)
        if getattr(self, "entity_anchor_enabled", True) and not (
            getattr(options.get("context"), "entities", None)
        ):
            scoped["__entities__"] = self._anchor_names(query)
        key = self.recall_cache.key(query, scoped, settings)
        started = time.perf_counter()
        computed = False
        def compute():
            nonlocal computed
            computed = True
            results, diagnostics = self.search_detailed(query, **options)
            return results, replace(diagnostics, preloaded=speculative)
        results, diagnostics = self.recall_cache.search(
            key, self.store.retrieval_revision, compute, speculative=speculative,
        )
        if not computed:
            diagnostics = replace(
                diagnostics, cache_hit=True,
                stage_ms={**dict.fromkeys(diagnostics.stage_ms, 0.0),
                          "cache_lookup": (time.perf_counter() - started) * 1000},
            )
        return results, diagnostics

    def search(
        self,
        query: str,
        *,
        limit: int = 6,
        token_budget: int = 700,
        include_archived: bool = False,
        temporal_mode: str = "current",
        as_of: str | None = None,
        graph_depth: int = 1,
        threshold: float | None = None,
        context: RetrievalContext | None = None,
        evidence_lookup: bool = False,
    ) -> list[RetrievalResult]:
        results, _diagnostics = self.search_detailed(
            query,
            limit=limit,
            token_budget=token_budget,
            include_archived=include_archived,
            temporal_mode=temporal_mode,
            as_of=as_of,
            graph_depth=graph_depth,
            threshold=threshold,
            context=context,
            evidence_lookup=evidence_lookup,
        )
        return results

    def search_detailed(
        self,
        query: str,
        *,
        limit: int = 6,
        token_budget: int = 700,
        include_archived: bool = False,
        temporal_mode: str = "current",
        as_of: str | None = None,
        graph_depth: int = 1,
        threshold: float | None = None,
        context: RetrievalContext | None = None,
        evidence_lookup: bool = False,
    ) -> tuple[list[RetrievalResult], RetrievalDiagnostics]:
        # One traced line per retrieval, only when CORTEX_TRACE=1. With tracing
        # off this is a single boolean test before the real work.
        if not _trace_enabled():
            return self._search_detailed(
                query,
                limit=limit,
                token_budget=token_budget,
                include_archived=include_archived,
                temporal_mode=temporal_mode,
                as_of=as_of,
                graph_depth=graph_depth,
                threshold=threshold,
                context=context,
                evidence_lookup=evidence_lookup,
            )
        started = _trace_phase_start()
        results, diagnostics = self._search_detailed(
            query,
            limit=limit,
            token_budget=token_budget,
            include_archived=include_archived,
            temporal_mode=temporal_mode,
            as_of=as_of,
            graph_depth=graph_depth,
            threshold=threshold,
            context=context,
            evidence_lookup=evidence_lookup,
        )
        _trace_record(
            "recall",
            _trace_elapsed_ms(started),
            counts={
                "candidates": diagnostics.candidate_count,
                "returned": diagnostics.selected_count,
            },
            session_id=(context.scope or {}).get("session_id") if context else None,
        )
        return results, diagnostics

    def _search_detailed(
        self,
        query: str,
        *,
        limit: int = 6,
        token_budget: int = 700,
        include_archived: bool = False,
        temporal_mode: str = "current",
        as_of: str | None = None,
        graph_depth: int = 1,
        threshold: float | None = None,
        context: RetrievalContext | None = None,
        evidence_lookup: bool = False,
    ) -> tuple[list[RetrievalResult], RetrievalDiagnostics]:
        if not query or limit <= 0 or token_budget <= 0:
            return [], RetrievalDiagnostics(0, 0, 0, True)
        # A query with no usable tokens — blank, punctuation-only, stopword-only
        # or a single character — carries no retrieval signal. Without this guard
        # the store's no-token FTS fallback answers it with the newest memories,
        # which then clear the score threshold and reach the model as confident
        # context (measured 2026-09-14: `"   "`, `"?!..."`, `",,,"`, `"a"` and
        # `"the and of to"` each selected 3 unrelated memories at ~0.29 with
        # `abstained=False`). Non-Latin scripts are unaffected: CJK tokenizes into
        # a single token, so `日本語のテキスト` still recalls.
        if not query_tokens(query):
            return [], RetrievalDiagnostics(0, 0, 0, True)
        expanded_query = _expand_query(query)
        retrieval_context = _normalize_retrieval_context(context, goal=query)
        scoring_profile = self.precompute.get(
            f"weights:{retrieval_context.scope.get('task_type')}",
            lambda: self.store.active_scoring_weights(
                retrieval_context.scope.get("task_type")
            ),
            revision=revision_of(self.store),
        )
        candidate_limit = max(40, limit * 8)
        stage_ms: dict[str, float] = {}
        budget = budget_from(self.latency_budget_ms)

        # Anchor on the names in the query before anything else runs: entity_match is the
        # strongest signal the store has, and in a real chat nothing fills it in.
        anchored_entities: list[str] = list(retrieval_context.entities or ())
        if not anchored_entities and self.entity_anchor_enabled:
            anchored_entities = extract_entities(query, known_entities(self.store))
            if anchored_entities:
                retrieval_context = replace(retrieval_context, entities=tuple(anchored_entities))
                stage_ms["entities_anchored"] = float(len(anchored_entities))
        stage_start = time.perf_counter()
        lexical_candidates = self.store.fts_search(
            expanded_query,
            limit=candidate_limit,
            include_archived=include_archived,
            evidence_lookup=evidence_lookup,
        )
        stage_ms["fts"] = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        feature_candidates = self.store.feature_search(
            expanded_query, limit=candidate_limit, include_archived=include_archived
        )
        stage_ms["feature"] = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        context_candidates = self.store.context_search(
            active_project=retrieval_context.active_project,
            entities=retrieval_context.entities,
            scope=retrieval_context.scope,
            system_state=retrieval_context.system_state,
            applicable_systems=retrieval_context.applicable_systems,
            applicable_versions=retrieval_context.applicable_versions,
            limit=candidate_limit,
            include_archived=include_archived,
            evidence_lookup=evidence_lookup,
        )
        stage_ms["context"] = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        neighborhood_candidates = self.store.neighborhood_search(
            expanded_query,
            limit=candidate_limit,
            include_archived=include_archived,
            evidence_lookup=evidence_lookup,
        )
        stage_ms["neighborhood"] = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        # Deliberately NOT memoised: "which commitments are due" depends on the clock, not
        # on the store revision, so caching it would make a deterministic retriever
        # time-dependent — the exact mistake the latency budget (#22) is careful about. It
        # costs 0.3ms, and that is cheaper than a selection that changes with the wall clock.
        prospective_candidates = self.store.due_prospective_memories(limit=2)[: min(2, limit)]
        candidates_by_id: dict[str, dict[str, Any]] = {}
        for candidate in lexical_candidates:
            candidates_by_id[str(candidate["id"])] = candidate
        for candidate in feature_candidates:
            memory_id = str(candidate["id"])
            if memory_id in candidates_by_id:
                candidates_by_id[memory_id]["feature_score"] = candidate.get("feature_score", 0.0)
                candidates_by_id[memory_id]["feature_matches"] = candidate.get("feature_matches", 0)
            else:
                candidates_by_id[memory_id] = candidate
        for candidate in context_candidates:
            memory_id = str(candidate["id"])
            if memory_id in candidates_by_id:
                candidates_by_id[memory_id]["context_candidate_score"] = candidate.get(
                    "context_candidate_score", 0.0
                )
            else:
                candidates_by_id[memory_id] = candidate
        for candidate in neighborhood_candidates:
            memory_id = str(candidate["id"])
            if memory_id in candidates_by_id:
                candidates_by_id[memory_id]["neighborhood_candidate_score"] = candidate.get(
                    "neighborhood_candidate_score", 0.0
                )
                candidates_by_id[memory_id]["neighborhood_labels"] = candidate.get(
                    "neighborhood_labels", ""
                )
            else:
                candidates_by_id[memory_id] = candidate
        # Entity expansion: memories attached to a name the query used, even when no
        # lexical signal would have surfaced them. This is the "expand before rank" half
        # of entity anchoring; filling the context alone changed no ranking.
        if anchored_entities:
            expanded = 0
            entity_ids = entity_memory_ids(
                self.store, anchored_entities, limit=candidate_limit
            )
            missing = [mid for mid in entity_ids if mid not in candidates_by_id]
            for memory_id in entity_ids:
                if memory_id in candidates_by_id:
                    candidates_by_id[memory_id]["entity_candidate"] = 1.0
            for row in entity_memories(self.store, missing):
                candidates_by_id[str(row["id"])] = {**row, "entity_candidate": 1.0}
                expanded += 1
            if expanded:
                stage_ms["entity_expanded_count"] = float(expanded)
        for candidate in prospective_candidates:
            memory_id = str(candidate["id"])
            if memory_id in candidates_by_id:
                candidates_by_id[memory_id]["prospective_due_score"] = 1.0
            else:
                candidates_by_id[memory_id] = candidate
        experience_strengths = self.store.memory_experience_strengths(list(candidates_by_id))
        for memory_id, evidence in experience_strengths.items():
            if memory_id in candidates_by_id:
                candidates_by_id[memory_id].update(evidence)
        candidates = list(candidates_by_id.values())
        # Query-independent (7.6ms/turn measured) and cached per store revision: the set
        # of memories serving as schema examples changes only when dependencies change.
        all_schema_sources = self.precompute.get(
            "schema_sources",
            lambda: {
                str(row[0])
                for row in select(
                    self.store,
                    "SELECT DISTINCT evidence_id FROM memory_dependencies "
                    "WHERE relation = 'schema_source' AND active = 1",
                )
            },
            revision=revision_of(self.store),
        )
        schema_source_ids = {cid for cid in candidates_by_id if cid in all_schema_sources}
        for candidate in candidates:
            candidate["schema_example"] = str(candidate["id"]) in schema_source_ids
        # Both of these are edge lookups over the candidate list; the edge table is small
        # and changes only when memories are written, so the global sets are cached per
        # revision and intersected in memory (~17ms of every turn otherwise).
        superseded = (
            {cid for cid in candidates_by_id if cid in self._edge_sets("supersedes")[1]}
            if temporal_mode == "current"
            else set()
        )
        contradicts = self._edge_sets("contradicts")
        contradicted = {
            cid for cid in candidates_by_id if cid in contradicts[0] or cid in contradicts[1]
        }
        stage_ms["prospective_merge"] = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        prepared = _QuerySignals.prepare(expanded_query, retrieval_context)
        feedback = self.store.context_feedback_many(list(candidates_by_id), retrieval_context.as_record())
        scored: dict[str, RetrievalResult] = {}
        for candidate in candidates:
            result = self._score(
                expanded_query,
                candidate,
                temporal_mode=temporal_mode,
                as_of=as_of,
                superseded=candidate["id"] in superseded,
                contradicted=candidate["id"] in contradicted,
                context=retrieval_context,
                scoring_weights=scoring_profile["weights"],
                prepared=prepared,
                context_feedback=feedback[str(candidate["id"])],
            )
            scored[candidate["id"]] = result

        stage_ms["scoring"] = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()
        seeds = sorted(scored.values(), key=lambda r: r.score, reverse=True)[: min(10, len(scored))]
        if seeds:
            graph_scores = self.store.association_scores(
                {str(result.memory["id"]): result.score for result in seeds}, depth=graph_depth
            )
            # Same cached edge sets the selection pass uses: these are per-turn queries
            # over a handful of ids, but they were two of them (#219).
            graph_superseded = (
                {mid for mid in graph_scores if mid in self._edge_sets("supersedes")[1]}
                if temporal_mode == "current"
                else set()
            )
            graph_contradicts = self._edge_sets("contradicts")
            graph_contradicted = {
                mid for mid in graph_scores
                if mid in graph_contradicts[0] or mid in graph_contradicts[1]
            }
            new_neighbors = [memory_id for memory_id in graph_scores if memory_id not in scored]
            graph_feedback = self.store.context_feedback_many(new_neighbors, retrieval_context.as_record())
            neighbor_memories = {str(memory["id"]): memory for memory in self.store.get_memories(new_neighbors)}
            graph_eligible = self.store.recall_eligible_ids(new_neighbors, include_archived=include_archived)
            for neighbor_id, graph_boost in graph_scores.items():
                if neighbor_id in scored:
                    old = scored[neighbor_id]
                    components = dict(old.components)
                    components["graph"] = max(components.get("graph", 0.0), graph_boost)
                    score = min(1.0, old.score + _GRAPH_EXPANSION_WEIGHT * graph_boost)
                    shadow_score = min(
                        1.0,
                        float(components.get("shadow_score") or old.score)
                        + _SHADOW_GRAPH_EXPANSION_WEIGHT * graph_boost,
                    )
                    components["shadow_score"] = shadow_score
                    components["shadow_score_delta"] = shadow_score - score
                    scored[neighbor_id] = RetrievalResult(old.memory, score, components, old.estimated_tokens)
                    continue
                memory = neighbor_memories.get(neighbor_id)
                if not memory or memory["state"] not in (
                    {"active", "cold", "archived"} if include_archived else {"active", "cold"}
                ):
                    continue
                if neighbor_id not in graph_eligible:
                    continue
                result = self._score(
                    expanded_query,
                    memory,
                    graph=graph_boost,
                    temporal_mode=temporal_mode,
                    as_of=as_of,
                    superseded=neighbor_id in graph_superseded,
                    contradicted=neighbor_id in graph_contradicted,
                    context=retrieval_context,
                    scoring_weights=scoring_profile["weights"],
                    prepared=prepared,
                    context_feedback=graph_feedback[neighbor_id],
                )
                scored[neighbor_id] = result

        stage_ms["graph"] = (time.perf_counter() - stage_start) * 1000
        stage_start = time.perf_counter()

        def _mark(label: str) -> None:
            """Record one sub-stage's elapsed time and restart the clock.

            `select` used to be a single number covering nine passes, so a 33ms reading could
            not say which pass owned it. Every pass now reports itself (#219).
            """
            nonlocal stage_start
            stage_ms[label] = (time.perf_counter() - stage_start) * 1000
            stage_start = time.perf_counter()

        # Optional semantic fusion. This AUGMENTS the candidate pool rather than
        # merely re-ranking it: the measured gain comes from embeddings finding
        # memories the lexical signals never surfaced, which re-ranking alone
        # cannot do. Inert (empty dict) unless semantic_weight > 0.
        if self.semantic_weight > 0.0 and not budget.expired():
            self._inject_semantic_candidates(
                query,
                scored,
                limit=limit,
                temporal_mode=temporal_mode,
                as_of=as_of,
                retrieval_context=retrieval_context,
                scoring_profile=scoring_profile,
                include_archived=include_archived,
                evidence_lookup=evidence_lookup,
            )
        else:
            if self.semantic_weight > 0.0 and budget.expired():
                stage_ms["semantic_skipped"] = 1.0
        _mark("semantic")

        # Entity evidence: a memory reachable *only* because the query named an entity is
        # relevant by association, and no lexical signal will ever speak for it.
        if anchored_entities:
            entity_ids = set(
                entity_memory_ids(self.store, anchored_entities, limit=candidate_limit)
            )
            entity_boosted = 0
            for memory_id, result in list(scored.items()):
                if memory_id not in entity_ids:
                    continue
                components = dict(result.components or {})
                components["entity_anchored"] = 1.0
                scored[memory_id] = replace(
                    result, score=float(result.score) + ENTITY_BOOST, components=components
                )
                entity_boosted += 1
            if entity_boosted:
                stage_ms["entity_boosted_count"] = float(entity_boosted)
        _mark("entity")

        # Stage two: rerank the top slice with BM25 over the actual contents, so a memory
        # that contains the query's exact terms beats one that merely shares a category.
        if getattr(self, "rerank_enabled", True):
            ordered = sorted(scored.values(), key=lambda r: (-r.score, r.memory["id"]))
            head, tail = ordered[:DEFAULT_TOP_N], ordered[DEFAULT_TOP_N:]
            reranked_head, reranked = rerank(head, query, top_n=DEFAULT_TOP_N)
            if reranked:
                # Only the head was rescored; the tail keeps its stage-one scores so the
                # candidate pool is never silently shrunk to the rerank window.
                scored = {r.memory["id"]: r for r in [*reranked_head, *tail]}
                stage_ms["reranked"] = float(reranked)
        _mark("rerank")

        # A memory that something else supersedes is history: reachable, never winning a
        # tie against its replacement.
        # Historical recall wants the old value — that is the question being asked. Demote
        # superseded memories only when the turn is about what is true now.
        # `superseded_ids(store)` is the whole superseded set and costs ~12ms per turn; it
        # only changes when an edge changes, so it is cached per store revision (#219).
        all_superseded = (
            self.precompute.get(
                "superseded_all",
                lambda: superseded_ids(self.store),
                revision=revision_of(self.store),
            )
            if temporal_mode != "historical"
            else set()
        )
        scored_list, superseded_demoted = (
            apply_validity_penalty(scored.values(), all_superseded)
            if temporal_mode != "historical"
            else (list(scored.values()), 0)
        )
        if superseded_demoted:
            scored = {r.memory["id"]: r for r in scored_list}
            stage_ms["superseded_demoted"] = float(superseded_demoted)
        _mark("validity")

        # A fact restated over months should settle near-ties against one fresh sentence.
        scored_list, stability_boosted = apply_stability_blend(scored.values())
        if stability_boosted:
            scored = {r.memory["id"]: r for r in scored_list}
            stage_ms["stability_boosted"] = float(stability_boosted)
        _mark("stability")

        # If the query said what it does not want, do not return it confidently.
        if getattr(self, "negative_cues_enabled", True):
            scored_list, negative_cue_suppressed = suppress(
                scored.values(), excluded_terms(query)
            )
        else:
            scored_list, negative_cue_suppressed = list(scored.values()), 0
        if negative_cue_suppressed:
            scored = {r.memory["id"]: r for r in scored_list}
            stage_ms["negative_cue_suppressed"] = float(negative_cue_suppressed)
        _mark("negative_cues")

        # Rank on use, not on retrieval count: a memory that informed an answer is worth
        # more than one that merely matched a query.
        # One cached global map instead of an IN(...) query per turn (measured 29.6ms).
        if getattr(self, "usage_boost_enabled", True):
            scored_list, usage_boosted = apply_usage_boost(self.store, scored.values())
        else:
            scored_list, usage_boosted = list(scored.values()), 0
        if usage_boosted:
            scored = {r.memory["id"]: r for r in scored_list}
            stage_ms["usage_boosted_count"] = float(usage_boosted)

        # A document must not outrank the user's own words on the same fact. Applied
        # before the sort, so it can only cap a competitor -- never lift anything.
        scored_list, authority_capped = cap_conflicts(scored.values())
        if authority_capped:
            scored = {r.memory["id"]: r for r in scored_list}
            stage_ms["authority_capped"] = float(authority_capped)

        ranked = rank_by_score(scored.values())

        # Namespace isolation: a finance/work memory must not reach a turn that declared a
        # different channel. Applied before eligibility so a blocked memory never even
        # consumes a slot or a token.
        channel = channel_namespaces(context)
        if channel:
            ranked, namespace_blocked = filter_results(ranked, channel)
            if namespace_blocked:
                stage_ms["namespace_blocked"] = float(namespace_blocked)
        eligible_ids = self.store.recall_eligible_ids(
            [str(result.memory["id"]) for result in ranked],
            evidence_lookup=evidence_lookup, include_archived=include_archived,
        )
        selected: list[RetrievalResult] = []
        rejection_reasons: dict[str, str] = {}
        consumed = 0
        remaining = list(ranked)
        # `_memory_similarity` is a pure function of the two memory bodies, and a
        # candidate's diversified value depends only on the SELECTED set. Instead
        # of re-scoring every remaining candidate against every selected memory on
        # every pick (quadratic work: 47 candidates used to cost ~1,200 set
        # intersections), track each remaining candidate's maximum similarity to
        # the selected set and refresh it only when a memory is actually picked.
        # The computed values are identical to the re-computation; only the
        # repeated pair work is removed.
        max_similarity_to_selected: dict[str, float] = {}
        selected_kinds: set[str] = set()
        while remaining and len(selected) < limit:
            if budget.expired():
                # Best-so-far beats late-and-complete for an interactive turn.
                stage_ms["budget_exhausted"] = 1.0
                break

            def diversified_value(result: RetrievalResult) -> float:
                similarity = max_similarity_to_selected.get(str(result.memory["id"]), 0.0)
                # A result whose kind is not yet represented earns the diversity
                # bonus; the bonus can only disappear once a same-kind memory is
                # selected, so tracking selected kinds matches the scan it
                # replaces.
                type_bonus = (
                    0.035
                    if selected and result.memory["kind"] not in selected_kinds
                    else 0.0
                )
                # The semantic contribution is already folded into result.score
                # by _inject_semantic_candidates; do not count it twice here.
                return result.score + type_bonus - 0.16 * similarity

            if selected:
                best_index = 0
                best_value = diversified_value(remaining[0])
                for index in range(1, len(remaining)):
                    # `remaining` stays in descending score order and the kind
                    # bonus is bounded by 0.035, so once even the best achievable
                    # value cannot beat the incumbent, every later candidate is
                    # out of contention. The scan is otherwise exact: the first
                    # maximum wins, exactly as `max(..., key=...)` chose it.
                    occurrence = remaining[index]
                    if occurrence.score + 0.035 <= best_value:
                        break
                    value = diversified_value(occurrence)
                    if value > best_value:
                        best_index, best_value = index, value
                result = remaining.pop(best_index)
            else:
                # Nothing selected yet, so every candidate's diversified value is
                # exactly its score (no kind bonus, no similarity penalty) and
                # `remaining` is in descending score order: the maximum is the
                # first element. Rescanning the whole pool on every rejection made
                # this quadratic — a 320-candidate pool whose results all failed
                # the eligibility or score gates cost 51,360 comparisons to
                # decide the same thing.
                result = remaining.pop(0)
            if str(result.memory["id"]) not in eligible_ids:
                rejection_reasons[str(result.memory["id"])] = "outside the active recall set"
                continue
            effective_threshold = self.threshold if threshold is None else float(threshold)
            if result.components.get("context_gate", 1.0) < 1.0:
                rejection_reasons[str(result.memory["id"])] = (
                    "required memory scope or preconditions are unavailable in the active task context"
                )
                continue
            if (
                result.components.get("relevance_penalty", 0.0) > 0.0
                and graph_depth <= 1
                and result.memory.get("kind") not in {"procedure", "prospective"}
                # A candidate surfaced by the semantic signal is direct evidence
                # of relevance in its own right — it has low lexical overlap by
                # construction, which is precisely why the embedding found it and
                # the lexical pass did not. Without this exemption the fusion
                # could never contribute a memory, only reorder existing ones.
                and not result.components.get("semantic_fusion")
            ):
                # Focused/lean recall requires direct evidence. Indirect low-
                # overlap associations are reserved for the planner's deep,
                # multi-hop mode where they can add useful context on purpose.
                rejection_reasons[str(result.memory["id"])] = "insufficient direct relevance for focused recall"
                continue
            if result.score < effective_threshold and not result.memory["pinned"]:
                rejection_reasons[str(result.memory["id"])] = "score below the active retrieval threshold"
                continue
            if max_similarity_to_selected.get(str(result.memory["id"]), 0.0) >= 0.86:
                rejection_reasons[str(result.memory["id"])] = "near-duplicate of a stronger selected memory"
                continue
            family_count = sum(
                1
                for prior in selected
                if result.memory.get("subject")
                and result.memory.get("predicate")
                and prior.memory.get("subject") == result.memory.get("subject")
                and prior.memory.get("predicate") == result.memory.get("predicate")
            )
            if family_count >= 2:
                rejection_reasons[str(result.memory["id"])] = "same claim family already has enough evidence"
                continue
            if consumed + result.estimated_tokens > token_budget:
                rejection_reasons[str(result.memory["id"])] = "would exceed the task context budget"
                continue
            selected.append(result)
            consumed += result.estimated_tokens
            selected_kinds.add(result.memory["kind"])
            if len(selected) < limit:
                # Refresh the running similarity maxima for the next picks; this
                # is the only place similarity is computed during selection.
                for other in remaining:
                    other_key = str(other.memory["id"])
                    similarity = _memory_similarity(other, result)
                    if similarity > max_similarity_to_selected.get(other_key, 0.0):
                        max_similarity_to_selected[other_key] = similarity
        # Soft similarity penalties still let four copies of one note through when every
        # alternative scores lower. Enforce the hard ceiling, and say so in the report.
        abstain, abstain_reason = should_abstain(selected, threshold=self.abstain_threshold)
        if abstain:
            stage_ms["abstained"] = 1.0
            stage_ms["abstained_top_score"] = round(float(selected[0].score), 6)
            selected = []

        # A recorded near-miss pair must not surface together (#80): the weaker half is dropped and
        # the reason reported. Neither memory is hidden from its own queries - the rule is about the
        # pair, not about either memory. Empty by default: nothing is dropped until a pair is
        # recorded, so this is inert until somebody judges two memories to be confusable.
        try:
            near_miss_drop = near_miss.demote(
                self.store, selected,
                score_of=lambda item: float(item.score),
                id_of=lambda item: str(item.memory["id"]),
            )
        except Exception:  # noqa: BLE001 - a judgement about a pair must never break recall
            near_miss_drop = []
        if near_miss_drop:
            for index in near_miss_drop:
                rejection_reasons[str(selected[index].memory["id"])] = (
                    "near-miss of a stronger selected memory (#80)"
                )
            dropped = set(near_miss_drop)
            selected = [result for index, result in enumerate(selected) if index not in dropped]
            stage_ms["near_miss_dropped_count"] = float(len(dropped))

        # Serial position (#82): where in the session a memory was last seen. Off by default -- the
        # effect is measured from the access log (late use-rate 0.313 vs early 0.223), which is
        # enough to build the weight and not enough to change every recall.
        try:
            if serial_position.enabled():
                session = str((getattr(context, "scope", None) or {}).get("session_id") or "")
                turns = serial_position.turn_index(self.store, session)
                counters = serial_position.apply(self.store, selected, session_id=session,
                                                 turns=turns)
                for name, value in (counters or {}).items():
                    stage_ms[name] = float(value)
        except Exception:  # noqa: BLE001 - a weight must never break recall
            pass

        selected, duplicates_replaced = enforce_distinct(selected, remaining, limit=limit)
        if duplicates_replaced:
            stage_ms["duplicates_replaced"] = float(duplicates_replaced)
            for result in selected:
                rejection_reasons.setdefault(
                    str(result.memory["id"]),
                    "selected after replacing a near-duplicate",
                )
        # Preload-scale budgets are spent by density instead of by rank: the same tokens
        # then carry more, sharper evidence. Inline recall (<=700) is untouched.
        if (
            getattr(self, "density_budget_enabled", True)
            and token_budget >= DENSITY_MIN_BUDGET
            and ranked
        ):
            # Same gates the selection loop applies: the threshold and pinned protection
            # are not optional just because the budget is large.
            eligible = [
                r for r in ranked
                if float(r.score) >= effective_threshold or r.memory.get("pinned")
            ]
            by_confidence_selection = by_confidence(eligible, token_budget, max_items=limit)
            if by_confidence_selection:
                selected = by_confidence_selection
                stage_ms["density_selected"] = float(len(selected))
        stage_ms["select"] = (time.perf_counter() - stage_start) * 1000
        selected_ids = {str(result.memory["id"]) for result in selected}
        candidate_decisions: list[dict[str, Any]] = []
        for rank, result in enumerate(ranked, 1):
            memory_id = str(result.memory["id"])
            was_selected = memory_id in selected_ids
            reason = rejection_reasons.get(memory_id)
            if was_selected:
                reason = "selected by score, context budget, and diversity constraints"
            elif reason is None and len(selected) >= limit:
                reason = "lower diversified rank than the selected result limit"
            elif reason is None:
                reason = "not selected after bounded diversification"
            candidate_decisions.append(
                {
                    "memory_id": memory_id,
                    "kind": str(result.memory.get("kind") or "semantic"),
                    "context_mode": str(result.memory.get("context_mode") or "standalone"),
                    "scope": _memory_context_map(result.memory, "scope", "scope_json"),
                    "entities": _memory_context_list(result.memory, "entities", "entities_json"),
                    "preconditions": _memory_context_map(
                        result.memory, "preconditions", "preconditions_json"
                    ),
                    "applicable_systems": _memory_context_list(
                        result.memory, "applicable_systems", "applicable_systems_json"
                    ),
                    "applicable_versions": _memory_context_list(
                        result.memory, "applicable_versions", "applicable_versions_json"
                    ),
                    "content_preview": " ".join(str(result.memory.get("content") or "").split())[:180],
                    "rank": rank,
                    "selected": was_selected,
                    "pinned": bool(result.memory.get("pinned")),
                    "score": round(float(result.score), 6),
                    "estimated_tokens": int(result.estimated_tokens),
                    "scoring_policy_version": scoring_profile["policy_version"],
                    "shadow_scoring_policy_version": SHADOW_SCORING_POLICY_VERSION,
                    "components": {
                        key: round(float(value), 6) for key, value in result.components.items()
                    },
                    "reason": reason,
                }
            )
        return selected, RetrievalDiagnostics(
            candidate_count=len(scored),
            selected_count=len(selected),
            estimated_tokens=consumed,
            abstained=not selected,
            candidate_decisions=tuple(candidate_decisions),
            stage_ms=stage_ms,
        )

    def shadow_tiered_comparison(
        self,
        query: str,
        *,
        limit: int = 6,
        token_budget: int = 700,
        include_archived: bool = False,
        temporal_mode: str = "current",
        as_of: str | None = None,
        graph_depth: int = 1,
        threshold: float | None = None,
        context: RetrievalContext | None = None,
    ) -> dict[str, Any]:
        """Compare live retrieval with the proposed role-tiered policy, shadow-only.

        The live selection is computed by the unchanged pipeline and returned
        untouched; the shadow selection re-walks the same ranked candidates
        with role gates applied. Nothing here is injected into context and no
        counters or logs are updated. Activation stays gated behind paired
        evaluation.
        """

        selected, diagnostics = self.search_detailed(
            query,
            limit=limit,
            token_budget=token_budget,
            include_archived=include_archived,
            temporal_mode=temporal_mode,
            as_of=as_of,
            graph_depth=graph_depth,
            threshold=threshold,
            context=context,
        )
        live_ids = [str(result.memory["id"]) for result in selected]
        candidate_ids = [str(row["memory_id"]) for row in diagnostics.candidate_decisions]
        roles: dict[str, str] = {}
        if candidate_ids:
            for memory in self.store.get_memories(candidate_ids):
                roles[str(memory["id"])] = str(memory.get("record_role") or "canonical")
        query_technical_tokens = {
            token.casefold() for token in _TECHNICAL_TOKEN.findall(query or "")
        }
        effective_threshold = self.threshold if threshold is None else float(threshold)
        shadow_ids: list[str] = []
        consumed = 0
        gate_decisions: list[dict[str, Any]] = []
        for row in diagnostics.candidate_decisions:
            memory_id = str(row["memory_id"])
            components = dict(row.get("components") or {})
            role = roles.get(memory_id, "canonical")
            score = float(row.get("score") or 0.0)
            pinned = bool(row.get("pinned"))
            if len(shadow_ids) >= max(1, int(limit)):
                break
            if components.get("context_gate", 1.0) < 1.0:
                continue
            if score < effective_threshold and not pinned:
                continue
            gate = self._shadow_role_gate(
                role,
                components,
                content_preview=str(row.get("content_preview") or ""),
                query_technical_tokens=query_technical_tokens,
            )
            if not gate["eligible"]:
                gate_decisions.append({"memory_id": memory_id, "role": role, **gate})
                continue
            estimated = int(row.get("estimated_tokens") or 0)
            if consumed + estimated > token_budget:
                continue
            shadow_ids.append(memory_id)
            consumed += estimated
            gate_decisions.append({"memory_id": memory_id, "role": role, **gate})
        live_set, shadow_set = set(live_ids), set(shadow_ids)
        return {
            "policy_version": SHADOW_ROLE_POLICY_VERSION,
            "query_length": len(query or ""),
            "live_selected_ids": live_ids,
            "shadow_selected_ids": shadow_ids,
            "identical": live_ids == shadow_ids,
            "only_live_ids": [memory_id for memory_id in live_ids if memory_id not in shadow_set],
            "only_shadow_ids": [memory_id for memory_id in shadow_ids if memory_id not in live_set],
            "role_gates": gate_decisions[:50],
            "live_estimated_tokens": diagnostics.estimated_tokens,
            "shadow_estimated_tokens": consumed,
            "claim_boundary": (
                "Shadow results never affect injected context. Activation requires labeled cases and a "
                "paired evaluation showing no unacceptable loss."
            ),
        }

    @staticmethod
    def _shadow_role_gate(
        role: str,
        components: dict[str, Any],
        *,
        content_preview: str,
        query_technical_tokens: set[str],
    ) -> dict[str, Any]:
        """Deterministic per-role eligibility under the proposed tiered policy."""

        lexical = float(components.get("lexical") or 0.0)
        phrase = float(components.get("phrase") or 0.0)
        graph = float(components.get("graph") or 0.0)
        currentness = float(components.get("currentness") or 0.0)
        context_support = max(
            float(components.get("scope_match") or 0.0) * float(components.get("has_scope") or 0.0),
            float(components.get("entity_match") or 0.0) * float(components.get("has_entities") or 0.0),
            float(components.get("system_match") or 0.0) * float(components.get("has_systems") or 0.0),
            float(components.get("version_match") or 0.0) * float(components.get("has_versions") or 0.0),
            float(components.get("precondition_match") or 0.0)
            * float(components.get("has_preconditions") or 0.0),
        )
        preview_fold = content_preview.casefold()
        technical_match = any(token in preview_fold for token in query_technical_tokens)
        if role == "reference":
            direct_support = max(lexical, phrase) >= 0.22 or context_support > 0.0 or technical_match
            graph_only = graph > 0.0 and max(lexical, phrase) < 0.10 and context_support <= 0.0
            eligible = direct_support and not graph_only
            reason = (
                "reference evidence has direct lexical, scope, entity, system, version, or exact "
                "technical support"
                if eligible
                else "reference evidence lacks direct support and cannot ride graph expansion alone"
            )
        elif role == "event":
            eligible = currentness >= 0.35 or max(lexical, phrase) >= 0.30
            reason = (
                "event remains temporally relevant or directly requested"
                if eligible
                else "event fell outside its temporal relevance window"
            )
        elif role == "claim":
            eligible = False
            reason = "unsupported claims stay gated until reviewed or evidence-backed"
        else:
            eligible = True
            reason = "canonical records keep normal eligibility"
        return {"eligible": eligible, "reason": reason}

    def _inject_semantic_candidates(
        self,
        query: str,
        scored: dict[str, RetrievalResult],
        *,
        limit: int,
        temporal_mode: str,
        as_of: str | None,
        retrieval_context: RetrievalContext | None,
        scoring_profile: dict[str, Any],
        include_archived: bool,
        evidence_lookup: bool,
    ) -> dict[str, int]:
        """Union the semantic top-N into the candidate pool; return ``id -> rank``.

        Only the semantic top-``semantic_pool`` ids get a fused rank, and only
        ones that pass the same eligibility gates as every other candidate are
        scored. Returns ``{}`` — meaning "no semantic signal" — whenever the
        model or the stored vectors are unavailable, so retrieval falls back to
        exactly the previous behaviour instead of failing.
        """

        try:
            from .embeddings import get_embedder

            embedder = get_embedder()
            if not embedder.available:
                return {}
            model_id = self.semantic_model_id or embedder.model_id
            vector = embedder.embed_one(query)
            if not vector:
                return {}
            hits = self.store.semantic_top_ids(
                vector, model_id, limit=max(int(self.semantic_pool), int(limit))
            )
        except Exception:                                        # noqa: BLE001
            return {}
        if not hits:
            return {}

        # Keep the raw query for embedding; use the same expanded lexical query
        # as the main candidate pool when scoring the semantic-only candidate.
        expanded_query = _expand_query(query)
        allowed_states = (
            {"active", "cold", "archived"} if include_archived else {"active", "cold"}
        )
        semantic_rank: dict[str, int] = {}
        for rank, (memory_id, similarity) in enumerate(hits):
            # Rank fusion orders semantically related candidates; it must not
            # turn orthogonal/anti-correlated vectors into evidence merely
            # because the pool is sparse. Keep the configured boundary strict.
            if float(similarity) <= self.semantic_floor:
                continue
            # The fusion contribution must land in the SCORE, not only in the
            # selector's ordering value: the threshold and relevance gates below
            # read `result.score`, so a boost applied later is invisible to them
            # and a semantically-found memory gets rejected for having — by
            # definition — little lexical overlap. (Caught by
            # tests/test_semantic_fusion_integration.py.)
            bonus = self.semantic_weight * (1.0 / (_SEMANTIC_RRF_K + rank + 1))

            existing = scored.get(memory_id)
            if existing is not None:
                components = dict(existing.components)
                components["semantic_fusion"] = bonus
                scored[memory_id] = RetrievalResult(
                    existing.memory,
                    min(1.0, existing.score + bonus),
                    components,
                    existing.estimated_tokens,
                )
                semantic_rank[memory_id] = rank
                continue

            try:
                memory = self.store.get_memory(memory_id)
            except Exception:                                    # noqa: BLE001
                continue
            if not memory or str(memory.get("state")) not in allowed_states:
                continue
            if not self.store.is_memory_recall_eligible(
                memory_id,
                evidence_lookup=evidence_lookup,
                include_archived=include_archived,
            ):
                continue
            result = self._score(
                expanded_query,
                memory,
                graph=0.0,
                temporal_mode=temporal_mode,
                as_of=as_of,
                context=retrieval_context,
                scoring_weights=scoring_profile["weights"],
            )
            components = dict(result.components)
            components["semantic_fusion"] = bonus
            scored[memory_id] = RetrievalResult(
                result.memory,
                min(1.0, result.score + bonus),
                components,
                result.estimated_tokens,
            )
            semantic_rank[memory_id] = rank
        return semantic_rank

    def _with_volatility(self, memory: dict[str, Any]) -> dict[str, Any]:
        """Attach the volatility class so decay can use it (#54). One cheap lookup per memory.

        Decay is off by default, so this costs nothing on the normal path, and a missing table or a
        reader error leaves the memory exactly as it was: a scoring hint must never change a result
        by raising.
        """
        if not decay.enabled():
            return memory
        if "volatility" in memory:
            return memory
        memory_id = str(memory.get("id") or "")
        if not memory_id:
            return memory
        cache = getattr(self, "_volatility_class_cache", None)
        if cache is None:
            cache = {}
            self._volatility_class_cache = cache
        if memory_id not in cache:
            try:
                row = self.store._conn.execute(
                    "SELECT volatility FROM memory_volatility WHERE memory_id = ?",
                    (memory_id,),
                ).fetchone()
                cache[memory_id] = str(row["volatility"]) if row else None
            except Exception:  # noqa: BLE001 - no table means no class
                cache[memory_id] = None
        assigned = cache[memory_id]
        if not assigned:
            return memory
        enriched = dict(memory)
        enriched["volatility"] = assigned
        return enriched

    def _score(
        self,
        query: str,
        memory: dict[str, Any],
        *,
        graph: float = 0.0,
        temporal_mode: str = "current",
        as_of: str | None = None,
        superseded: bool = False,
        contradicted: bool = False,
        context: RetrievalContext | None = None,
        scoring_weights: dict[str, float] | None = None,
        prepared: _QuerySignals | None = None,
        context_feedback: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        prepared = prepared or _QuerySignals.prepare(query, context or RetrievalContext(goal=query))
        q_tokens = prepared.tokens
        m_tokens = set(_content_token_set(memory["content"]))
        intersection = len(q_tokens & m_tokens)
        union = max(1, len(q_tokens | m_tokens))
        overlap = intersection / union
        coverage = intersection / max(1, len(q_tokens))
        phrase = 1.0 if len(query.strip()) >= 8 and query.casefold() in memory["content"].casefold() else 0.0

        fts_relevance = _fts_relevance(memory.get("fts_rank", 8.0))
        # FTS5 BM25 can assign a very strong score to one rare query token.
        # Temper that signal by query coverage so a well-connected memory that
        # happens to mention one word (for example, "recall") cannot outrank a
        # document matching the actual multi-word request. Single-word queries
        # retain the full BM25 signal because their coverage is 1.0.
        coverage_weighted_fts = fts_relevance * min(1.0, 0.20 + 0.80 * coverage)
        lexical = max(overlap, coverage * 0.8, coverage_weighted_fts)
        raw_feature_score = max(0.0, float(memory.get("feature_score", 0.0) or 0.0))
        semantic = raw_feature_score / (2.0 + raw_feature_score)
        direct_relevance = max(lexical, semantic * 0.8, phrase)
        # Associative activation is useful, but it is not evidence that a
        # memory answers the current request. On multi-term queries, charge a
        # small explicit penalty when neither the text nor semantic features
        # provide enough direct support. Strong graph neighbors can still
        # surface; they simply cannot win on connectivity alone.
        relevance_penalty = (
            SCORE_PENALTIES["relevance"]
            if len(q_tokens) >= 3 and direct_relevance < 0.28
            else 0.0
        )
        activation = self._activation(memory)
        historical_usefulness = self._utility(memory)
        confidence = float(memory["confidence"])
        source_reliability = _source_reliability(memory)
        currentness = self._currentness(memory, temporal_mode=temporal_mode, as_of=as_of)
        importance = float(memory["importance"])
        uniqueness = float(memory.get("uniqueness", 1.0))
        memory_type = _MEMORY_TYPE_PRIOR.get(str(memory.get("kind") or "semantic"), 0.60)
        stale_risk = self._stale_risk(memory)
        harmful = int(memory.get("harmful_count", 0)) + int(memory["false_positive_count"])
        # Only judged use is evidence about correctness. Merely injecting the
        # same item again must not dilute its observed error rate.
        judged = int(memory.get("used_count", 0)) + harmful
        wrong_rate = harmful / max(1, judged)
        context_components = _context_components(memory, context or RetrievalContext(goal=query), query, prepared=prepared)
        context_feedback = context_feedback if context_feedback is not None else self.store.context_feedback(
            str(memory["id"]),
            (context or RetrievalContext()).as_record(),
        )
        context_feedback_weight = min(1.0, float(context_feedback["observations"]) / 3.0)
        context_adaptation = (
            (2.0 * float(context_feedback["usefulness"]) - 1.0) * context_feedback_weight
        )
        if (
            context_components["is_context_dependent"]
            and context_components["context_gate"] >= 1.0
            and context_components["context_completeness"] >= 1.0
        ):
            relevance_penalty = 0.0
        context_boost = (
            0.08 * context_components["project_match"] * context_components["has_project_scope"]
            + 0.06 * context_components["goal_match"] * context_components["has_goal_scope"]
            + 0.05 * context_components["entity_match"] * context_components["has_entities"]
            + 0.07 * context_components["scope_match"] * context_components["has_scope"]
            + 0.07 * context_components["precondition_match"] * context_components["has_preconditions"]
            + 0.04 * context_components["system_match"] * context_components["has_systems"]
            + 0.03 * context_components["version_match"] * context_components["has_versions"]
            + 0.05
            * context_components["context_completeness"]
            * context_components["is_context_dependent"]
        )
        neighborhood = min(1.0, float(memory.get("neighborhood_candidate_score", 0.0) or 0.0))

        operator_policy = self.store.active_policy_adjustment(
            "retrieval",
            {
                "kind": str(memory.get("kind") or "semantic"),
                "source_type": str(memory.get("source_type") or "conversation"),
                "source_category": str(memory.get("source_category") or "AGENT_INFERENCE"),
            },
        )
        score_signals = {
            "lexical": lexical,
            "phrase": phrase,
            "semantic": semantic,
            "activation": activation,
            "utility": historical_usefulness,
            "importance": importance,
            "confidence": confidence,
            "source_reliability": source_reliability,
            "currentness": currentness,
            "memory_type": memory_type,
            "uniqueness": uniqueness,
            "graph": graph,
            "neighborhood": neighborhood,
        }
        common_adjustments = {
            "context_boost": context_boost,
            "context_adaptation": context_adaptation,
            "relevance_penalty": relevance_penalty,
            "stale_risk": stale_risk,
            "wrong_rate": min(1.0, wrong_rate),
            "contradiction_risk": float(contradicted),
            "prospective_due": float(memory.get("prospective_due_score") or 0.0),
            "superseded": float(superseded),
            "pinned": float(bool(memory["pinned"])),
            "cold": float(memory["state"] == "cold"),
            "archived": float(memory["state"] == "archived"),
            "stranded": float(bool(memory.get("stranded"))),
            "schema_example": float(bool(memory.get("schema_example"))),
            "specificity_request": prepared.specificity,
            "operator_policy": float(operator_policy.get("score_adjustment") or 0.0),
            "context_gate": float(context_components["context_gate"]),
            # Off unless CORTEX_DECAY is set (#93): episodic retention decays exponentially,
            # everything else reports 0.0 here and scoring is untouched. When on, a memory that
            # knows how fast it goes stale decays on that schedule instead of on its kind (#54).
            "decay": decay.penalty(self._with_volatility(memory)),
        }
        score = _weighted_score(
            score_signals,
            scoring_weights or SCORE_SIGNAL_WEIGHTS,
            SCORE_PENALTIES,
            common_adjustments,
        )
        shadow_score = _weighted_score(
            score_signals,
            SHADOW_SCORE_SIGNAL_WEIGHTS,
            SHADOW_SCORE_PENALTIES,
            common_adjustments,
        )

        components = {
            **score_signals,
            "phrase": phrase,
            "semantic_similarity": semantic,
            "historical_usefulness": historical_usefulness,
            "relevance_penalty": relevance_penalty,
            "stale_risk": stale_risk,
            "wrong_rate": min(1.0, wrong_rate),
            "superseded": float(superseded),
            "contradiction_risk": float(contradicted),
            "operator_policy": float(operator_policy.get("score_adjustment") or 0.0),
            "prospective_due": float(memory.get("prospective_due_score") or 0.0),
            # Reported, not scored (#48): a dispute says the store has not resolved which source
            # is right, which is not evidence that this memory is worse.
            "disputed": float(bool(memory.get("disputed"))),
            "context_candidate": min(1.0, float(memory.get("context_candidate_score", 0.0)) / 4.0),
            "context_historical_usefulness": float(context_feedback["usefulness"]),
            "context_feedback_observations": min(
                1.0, float(context_feedback["observations"]) / 10.0
            ),
            "context_adaptation": context_adaptation,
            "context_boost": context_boost,
            "shadow_score": shadow_score,
            "shadow_score_delta": shadow_score - score,
            "spaced_reinforcement": float(memory.get("spaced_reinforcement") or 0.0),
            "selected_unused": min(1.0, float(memory.get("selected_unused") or 0.0) / 5.0),
            "stranded": float(bool(memory.get("stranded"))),
            "schema_example": float(bool(memory.get("schema_example"))),
            "specificity_request": common_adjustments["specificity_request"],
            **{
                f"live_weight_{signal}": float(
                    (scoring_weights or SCORE_SIGNAL_WEIGHTS)[signal]
                )
                for signal in SCORE_SIGNAL_WEIGHTS
            },
            **context_components,
        }
        estimated_tokens = max(12, math.ceil(len(memory["content"]) / 4) + 18)
        return RetrievalResult(memory, max(0.0, min(1.0, score)), components, estimated_tokens)

    @staticmethod
    def _activation(memory: dict[str, Any]) -> float:
        # Retrieval and injection are observations, not successful rehearsal.
        # An "irrelevant" access can also update last_used_at, so outcome-free
        # selection cannot safely provide recency credit. Positive outcome
        # events have their own timestamp; actual use still earns frequency.
        age = _age_days(memory.get("last_helpful_at") or memory.get("updated_at"))
        half_life = _HALF_LIFE_DAYS.get(memory["kind"], 120.0)
        half_life *= max(0.2, 1.15 - float(memory["volatility"]))
        recency = 1.0 / (1.0 + age / max(1.0, half_life))
        weighted_uses = (
            0.25 * min(2, int(memory["used_count"]))
            + 1.5 * int(memory["success_count"])
            + 2.0 * int(memory["confirmed_count"])
            + 1.5 * int(memory.get("helpful_count", 0))
            + 2.0 * int(memory.get("validated_count", 0))
        )
        frequency = min(1.0, math.log1p(weighted_uses) / math.log(12.0))
        spaced = max(0.0, min(1.0, float(memory.get("spaced_reinforcement") or 0.0)))
        selected_unused = max(0, int(memory.get("selected_unused") or 0))
        inhibition = 0.10 * min(1.0, selected_unused / 5.0)
        return max(0.0, min(1.0, 0.54 * recency + 0.28 * frequency + 0.18 * spaced - inhibition))

    @staticmethod
    def _utility(memory: dict[str, Any]) -> float:
        used = int(memory["used_count"])
        successes = (
            int(memory["success_count"])
            + int(memory["confirmed_count"])
            + int(memory.get("helpful_count", 0))
            + int(memory.get("validated_count", 0))
        )
        false = int(memory["false_positive_count"]) + int(memory.get("harmful_count", 0))
        # Bayesian smoothing prevents one early success from dominating.
        positive = (successes + 1.5) / (used + 3.0)
        # Injection is not an outcome and cannot wash out negative feedback.
        penalty = false / max(3.0, used + false + 2.0)
        return max(0.0, min(1.0, positive - 0.7 * penalty))

    @staticmethod
    def _stale_risk(memory: dict[str, Any]) -> float:
        age = _age_days(memory["updated_at"])
        volatility = float(memory["volatility"])
        half_life = _HALF_LIFE_DAYS.get(memory["kind"], 120.0)
        return min(1.0, volatility * age / max(1.0, half_life))

    @staticmethod
    def _currentness(
        memory: dict[str, Any],
        *,
        temporal_mode: str = "current",
        as_of: str | None = None,
    ) -> float:
        target = datetime.now(timezone.utc)
        if temporal_mode == "historical" and as_of:
            try:
                target = datetime.fromisoformat(as_of)
                if target.tzinfo is None:
                    target = target.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        base = float(memory.get("currentness_confidence", 0.7))
        try:
            valid_from = datetime.fromisoformat(memory["valid_from"]) if memory.get("valid_from") else None
            valid_to = datetime.fromisoformat(memory["valid_to"]) if memory.get("valid_to") else None
            if valid_from and valid_from.tzinfo is None:
                valid_from = valid_from.replace(tzinfo=timezone.utc)
            if valid_to and valid_to.tzinfo is None:
                valid_to = valid_to.replace(tzinfo=timezone.utc)
            if valid_from and target < valid_from:
                return base * 0.25
            if valid_to and target > valid_to:
                return base * 0.10
        except ValueError:
            return base * 0.6
        return base


def _weighted_score(
    signals: dict[str, float],
    weights: dict[str, float],
    penalties: dict[str, float],
    adjustments: dict[str, float],
) -> float:
    """Apply one explicit scoring policy to observable component values."""

    score = sum(
        float(weights.get(component, 0.0)) * float(signals.get(component, 0.0))
        for component in weights
    )
    score += float(adjustments.get("context_boost", 0.0))
    score += 0.08 * float(adjustments.get("context_adaptation", 0.0))
    if float(adjustments.get("relevance_penalty", 0.0)) > 0.0:
        score -= float(penalties["relevance"])
    score -= float(penalties["stale_risk"]) * float(adjustments.get("stale_risk", 0.0))
    score -= float(penalties["wrong_rate"]) * float(adjustments.get("wrong_rate", 0.0))
    score -= 0.15 * float(adjustments.get("contradiction_risk", 0.0))
    score += 0.18 * float(adjustments.get("prospective_due", 0.0))
    score -= 0.18 * float(adjustments.get("superseded", 0.0))
    score += 0.08 * float(adjustments.get("pinned", 0.0))
    score -= 0.03 * float(adjustments.get("cold", 0.0))
    score -= 0.10 * float(adjustments.get("archived", 0.0))
    score -= 0.12 * float(adjustments.get("stranded", 0.0))
    if float(adjustments.get("schema_example", 0.0)) > 0.0:
        score -= 0.10 * (1.0 - float(adjustments.get("specificity_request", 0.0)))
    score += float(adjustments.get("operator_policy", 0.0))
    score -= decay.WEIGHT * float(adjustments.get("decay", 0.0))
    if float(adjustments.get("context_gate", 1.0)) < 1.0:
        score = min(score, 0.01)
    return max(0.0, min(1.0, score))


def rank_by_score(results: Iterable["RetrievalResult"]) -> list["RetrievalResult"]:
    """Rank by score, then pinned, then memory id.

    The id tie-break is not cosmetic: equal scores used to order by whatever order the dict
    happened to hold, which is where a "fails once, passes on re-run" suite flake came from
    (#220). Ranking is now a pure function of the results.
    """
    return sorted(
        results,
        key=lambda result: (
            -float(result.score),
            -float(bool(result.memory.get("pinned"))),
            str(result.memory.get("id") or ""),
        ),
    )


def token_overlap(left: str, right: str) -> float:
    a = set(query_tokens(left))
    b = set(query_tokens(right))
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _normalize_retrieval_context(
    context: RetrievalContext | None,
    *,
    goal: str,
) -> RetrievalContext:
    source = context or RetrievalContext()
    return RetrievalContext(
        active_project=_clean_context_value(source.active_project),
        goal=_clean_context_value(source.goal) or _clean_context_value(goal),
        entities=tuple(_clean_context_list(source.entities)),
        scope=_clean_context_map(source.scope),
        system_state=_clean_context_map(source.system_state),
        applicable_systems=tuple(_clean_context_list(source.applicable_systems)),
        applicable_versions=tuple(_clean_context_list(source.applicable_versions)),
    )


def _context_components(
    memory: dict[str, Any],
    context: RetrievalContext,
    query: str,
    *, prepared: _QuerySignals | None = None,
) -> dict[str, float]:
    mode = str(memory.get("context_mode") or "standalone").casefold()
    scope = _memory_context_map(memory, "scope", "scope_json")
    preconditions = _memory_context_map(memory, "preconditions", "preconditions_json")
    entities = _memory_context_list(memory, "entities", "entities_json")
    systems = _memory_context_list(memory, "applicable_systems", "applicable_systems_json")
    versions = _memory_context_list(memory, "applicable_versions", "applicable_versions_json")

    prepared = prepared or _QuerySignals.prepare(query, context)
    active_scope = prepared.active_scope
    available_state = prepared.available_state
    query_folded = prepared.folded
    context_entities = prepared.entities
    context_systems = prepared.systems
    context_versions = prepared.versions

    scope_results: list[float] = []
    scope_gate_results: list[bool] = []
    for key, required in scope.items():
        actual = active_scope.get(key)
        if key == "goal":
            actual = context.goal
            result = feature_similarity(required, actual or "") if actual else 0.0
            scope_results.append(result)
            scope_gate_results.append(result >= 0.55)
        else:
            result = _context_value_match(required, actual)
            scope_results.append(result)
            scope_gate_results.append(result >= 0.999)
    scope_match = sum(scope_results) / len(scope_results) if scope_results else 1.0
    project_match = (
        _context_value_match(scope["project"], context.active_project or active_scope.get("project"))
        if scope.get("project")
        else 1.0
    )
    goal_match = (
        feature_similarity(scope["goal"], context.goal or "") if scope.get("goal") else 1.0
    )

    query_token_set = prepared.tokens

    def _query_mentions(value: str) -> bool:
        """True when the query carries the value as whole words."""

        tokens = _context_tokens(value)
        return bool(tokens) and tokens <= query_token_set

    entity_hits = sum(
        1
        for entity in entities
        if entity.casefold() in context_entities
        or entity.casefold() in query_folded
        or bool(set(query_tokens(entity)))
        and set(query_tokens(entity)) <= query_token_set
    )
    entity_match = entity_hits / len(entities) if entities else 1.0

    precondition_results = [
        _context_value_match(required, available_state.get(key))
        for key, required in preconditions.items()
    ]
    precondition_match = (
        sum(precondition_results) / len(precondition_results) if precondition_results else 1.0
    )
    system_hits = sum(
        1
        for system in systems
        if _context_label_in(system, context_systems) or _query_mentions(system)
    )
    system_match = system_hits / len(systems) if systems else 1.0
    version_hits = sum(
        1
        for version in versions
        if _context_label_in(version, context_versions) or _query_mentions(version)
    )
    version_match = version_hits / len(versions) if versions else 1.0

    requirement_scores: list[float] = []
    if scope:
        requirement_scores.append(scope_match)
    if entities:
        requirement_scores.append(entity_match)
    if preconditions:
        requirement_scores.append(precondition_match)
    if systems:
        requirement_scores.append(system_match)
    if versions:
        requirement_scores.append(version_match)
    context_completeness = (
        sum(requirement_scores) / len(requirement_scores) if requirement_scores else (0.0 if mode == "context_dependent" else 1.0)
    )
    context_gate = 1.0
    if mode == "context_dependent":
        hard_requirements = bool(scope or preconditions or systems or versions)
        hard_failed = (
            (bool(scope) and not all(scope_gate_results))
            or (bool(preconditions) and precondition_match < 0.999)
            or (bool(systems) and system_match < 0.999)
            or (bool(versions) and version_match < 0.999)
        )
        if hard_failed or (not hard_requirements and (not entities or entity_match <= 0.0)):
            context_gate = 0.0
    return {
        "project_match": max(0.0, min(1.0, project_match)),
        "goal_match": max(0.0, min(1.0, goal_match)),
        "entity_match": max(0.0, min(1.0, entity_match)),
        "scope_match": max(0.0, min(1.0, scope_match)),
        "precondition_match": max(0.0, min(1.0, precondition_match)),
        "system_match": max(0.0, min(1.0, system_match)),
        "version_match": max(0.0, min(1.0, version_match)),
        "context_completeness": max(0.0, min(1.0, context_completeness)),
        "metadata_completeness": max(0.0, min(1.0, float(memory.get("metadata_completeness") or 0.0))),
        "context_gate": context_gate,
        "has_project_scope": float(bool(scope.get("project"))),
        "has_goal_scope": float(bool(scope.get("goal"))),
        "has_entities": float(bool(entities)),
        "has_scope": float(bool(scope)),
        "has_preconditions": float(bool(preconditions)),
        "has_systems": float(bool(systems)),
        "has_versions": float(bool(versions)),
        "is_context_dependent": float(mode == "context_dependent"),
    }


def _source_reliability(memory: dict[str, Any]) -> float:
    source_category = str(
        memory.get("origin_source_category")
        or memory.get("source_category")
        or "AGENT_INFERENCE"
    )
    prior = _SOURCE_RELIABILITY.get(source_category, 0.64)
    trust = max(0.0, min(1.0, float(memory.get("trust") or 0.0)))
    return max(0.0, min(1.0, 0.55 * prior + 0.45 * trust))


def _memory_context_map(memory: dict[str, Any], parsed_key: str, json_key: str) -> dict[str, str]:
    parsed = memory.get(parsed_key)
    if not isinstance(parsed, dict):
        try:
            parsed = json.loads(str(memory.get(json_key) or "{}"))
        except (json.JSONDecodeError, TypeError, ValueError):
            parsed = {}
    return _clean_context_map(parsed if isinstance(parsed, dict) else {})


def _memory_context_list(memory: dict[str, Any], parsed_key: str, json_key: str) -> list[str]:
    parsed = memory.get(parsed_key)
    if not isinstance(parsed, (list, tuple)):
        try:
            parsed = json.loads(str(memory.get(json_key) or "[]"))
        except (json.JSONDecodeError, TypeError, ValueError):
            parsed = []
    return _clean_context_list(parsed if isinstance(parsed, (list, tuple)) else [])


def _clean_context_map(values: dict[str, Any] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in (values or {}).items():
        clean_key = _clean_context_value(key).casefold().replace(" ", "_")
        clean_value = _clean_context_value(value)
        if clean_key and clean_value:
            result[clean_key] = clean_value
    return result


def _clean_context_list(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    result: list[str] = []
    seen: set[str] = set()
    for value in values or ():
        clean = _clean_context_value(value)
        if clean and clean.casefold() not in seen:
            seen.add(clean.casefold())
            result.append(clean)
    return result


def _clean_context_value(value: Any) -> str:
    return " ".join(str(value or "").split())[:300]


def _context_value_match(required: str, actual: str | None) -> float:
    """Return 1.0 only when the active context genuinely carries the requirement.

    Containment is deliberately NOT a match: "nonproduction" must not satisfy
    "production", "not enabled" must not satisfy "enabled", and "v10" must not
    satisfy "v1".  Values are compared as casefolded word tokens so casing,
    punctuation, and whitespace differences that do not change the words still
    match.
    """

    if not actual:
        return 0.0
    left = _clean_context_value(required).casefold()
    right = _clean_context_value(actual).casefold()
    if left == right:
        return 1.0
    left_tokens = _context_tokens(left)
    right_tokens = _context_tokens(right)
    if left_tokens and left_tokens == right_tokens:
        return 1.0
    return 0.0


_CONTEXT_TOKEN = re.compile(r"[a-z0-9]+")


def _context_tokens(value: str) -> set[str]:
    """Casefolded word tokens for a context label, with no stopword removal."""

    return set(_CONTEXT_TOKEN.findall(str(value or "").casefold()))


def _context_label_in(value: str, candidates: Iterable[str]) -> bool:
    """True when ``value``'s tokens exactly equal some candidate's tokens."""

    tokens = _context_tokens(value)
    return bool(tokens) and any(tokens == _context_tokens(candidate) for candidate in candidates)


def _age_days(timestamp: str | None) -> float:
    if not timestamp:
        return 3650.0
    try:
        dt = datetime.fromisoformat(timestamp)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)
    except (TypeError, ValueError):
        return 3650.0


def _expand_query(query: str) -> str:
    """Add a small set of intent terms without an LLM or embedding call."""
    tokens = set(query_tokens(query))
    additions: set[str] = set()
    groups = (
        (
            {"format", "response", "respond", "style", "answer"},
            {"prefer", "preference", "concise", "summary", "detail"},
        ),
        ({"fix", "broken", "error", "issue", "problem"}, {"workaround", "resolved", "verified", "root", "cause"}),
        ({"where", "location", "path", "host", "hosted"}, {"runs", "stored", "server", "backend"}),
        ({"decision", "choose", "choice", "approach"}, {"decided", "chose", "picked", "uses"}),
    )
    for triggers, related in groups:
        if tokens & triggers:
            additions.update(related)
    return query if not additions else f"{query} {' '.join(sorted(additions))}"


def _fts_relevance(raw_rank: Any) -> float:
    """Convert SQLite FTS5 BM25 rank to a bounded higher-is-better value.

    SQLite deliberately negates BM25: better matches are numerically lower
    (usually more negative). Taking ``abs(rank)`` and then an inverse, as the
    prototype did, reverses that ordering. Positive values are retained as a
    defensive fallback for non-FTS or legacy candidates.
    """

    try:
        rank = float(raw_rank if raw_rank is not None else 8.0)
    except (TypeError, ValueError):
        rank = 8.0
    if rank <= 0:
        magnitude = abs(rank)
        return magnitude / (1.0 + magnitude)
    return 1.0 / (1.0 + rank)


def _memory_similarity(left: RetrievalResult, right: RetrievalResult) -> float:
    a = _content_token_set(left.memory["content"])
    b = _content_token_set(right.memory["content"])
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@lru_cache(maxsize=4096)
def _content_token_set(content: str) -> frozenset[str]:
    """Tokenize a memory body once.

    ``_memory_similarity`` is called O(candidates x selected) times per recall
    and always with the same handful of memory bodies, so tokenizing on every
    call re-ran the tokenizer ~27,000 times for a single recall and dominated
    live prepare latency (profiled 2026-09-10: 95% of the select stage, ~3.9M
    ``casefold`` calls). The token set for a given body is immutable, so cache
    it. Returns a ``frozenset`` for hashing; memory-body similarity deliberately
    uses the full distinct-token set rather than the query-side cap.
    """
    return frozenset(query_tokens(content, limit=None))
