"""Fast deterministic retrieval and utility scoring for Cortex."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

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
}

_SOURCE_RELIABILITY = {
    "TOOL_VERIFIED": 0.95,
    "USER_EXPLICIT": 0.90,
    "DOCUMENT_EXTRACTED": 0.82,
    "REFLECTION": 0.62,
    "AGENT_INFERENCE": 0.52,
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
}

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
        return {
            "id": self.memory["id"],
            "kind": self.memory["kind"],
            "content": self.memory["content"],
            "state": self.memory["state"],
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


@dataclass(frozen=True)
class RetrievalDiagnostics:
    candidate_count: int
    selected_count: int
    estimated_tokens: int
    abstained: bool
    candidate_decisions: tuple[dict[str, Any], ...] = ()


class MemoryRetriever:
    """FTS + utility + one-hop associative graph retrieval."""

    def __init__(self, store: CortexStore, *, threshold: float = 0.16):
        self.store = store
        self.threshold = threshold

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
    ) -> tuple[list[RetrievalResult], RetrievalDiagnostics]:
        if not query or limit <= 0 or token_budget <= 0:
            return [], RetrievalDiagnostics(0, 0, 0, True)
        expanded_query = _expand_query(query)
        retrieval_context = _normalize_retrieval_context(context, goal=query)
        candidate_limit = max(40, limit * 8)
        lexical_candidates = self.store.fts_search(
            expanded_query, limit=candidate_limit, include_archived=include_archived
        )
        feature_candidates = self.store.feature_search(
            expanded_query, limit=candidate_limit, include_archived=include_archived
        )
        context_candidates = self.store.context_search(
            active_project=retrieval_context.active_project,
            entities=retrieval_context.entities,
            scope=retrieval_context.scope,
            system_state=retrieval_context.system_state,
            applicable_systems=retrieval_context.applicable_systems,
            applicable_versions=retrieval_context.applicable_versions,
            limit=candidate_limit,
            include_archived=include_archived,
        )
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
        candidates = list(candidates_by_id.values())
        superseded = self.store.superseded_ids(list(candidates_by_id)) if temporal_mode == "current" else set()
        contradicted = self.store.contradicted_ids(list(candidates_by_id))
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
            )
            scored[candidate["id"]] = result

        seeds = sorted(scored.values(), key=lambda r: r.score, reverse=True)[: min(10, len(scored))]
        if seeds:
            graph_scores = self.store.association_scores(
                {str(result.memory["id"]): result.score for result in seeds}, depth=graph_depth
            )
            graph_superseded = (
                self.store.superseded_ids(list(graph_scores)) if temporal_mode == "current" else set()
            )
            graph_contradicted = self.store.contradicted_ids(list(graph_scores))
            for neighbor_id, graph_boost in graph_scores.items():
                if neighbor_id in scored:
                    old = scored[neighbor_id]
                    components = dict(old.components)
                    components["graph"] = max(components.get("graph", 0.0), graph_boost)
                    score = min(1.0, old.score + 0.10 * graph_boost)
                    scored[neighbor_id] = RetrievalResult(old.memory, score, components, old.estimated_tokens)
                    continue
                memory = self.store.get_memory(neighbor_id)
                if not memory or memory["state"] not in (
                    {"active", "cold", "archived"} if include_archived else {"active", "cold"}
                ):
                    continue
                result = self._score(
                    query,
                    memory,
                    graph=graph_boost,
                    temporal_mode=temporal_mode,
                    as_of=as_of,
                    superseded=neighbor_id in graph_superseded,
                    contradicted=neighbor_id in graph_contradicted,
                    context=retrieval_context,
                )
                scored[neighbor_id] = result

        ranked = sorted(scored.values(), key=lambda r: (r.score, r.memory["pinned"]), reverse=True)
        selected: list[RetrievalResult] = []
        rejection_reasons: dict[str, str] = {}
        consumed = 0
        remaining = list(ranked)
        while remaining and len(selected) < limit:

            def diversified_value(result: RetrievalResult) -> float:
                similarity = max((_memory_similarity(result, prior) for prior in selected), default=0.0)
                type_bonus = (
                    0.035
                    if selected and all(prior.memory["kind"] != result.memory["kind"] for prior in selected)
                    else 0.0
                )
                return result.score + type_bonus - 0.16 * similarity

            result = max(remaining, key=diversified_value)
            remaining.remove(result)
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
            ):
                # Focused/lean recall requires direct evidence. Indirect low-
                # overlap associations are reserved for the planner's deep,
                # multi-hop mode where they can add useful context on purpose.
                rejection_reasons[str(result.memory["id"])] = "insufficient direct relevance for focused recall"
                continue
            if result.score < effective_threshold and not result.memory["pinned"]:
                rejection_reasons[str(result.memory["id"])] = "score below the active retrieval threshold"
                continue
            if any(_memory_similarity(result, prior) >= 0.86 for prior in selected):
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
    ) -> RetrievalResult:
        q_tokens = set(query_tokens(query))
        m_tokens = set(query_tokens(memory["content"]))
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
        relevance_penalty = 0.12 if len(q_tokens) >= 3 and direct_relevance < 0.28 else 0.0
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
        wrong_rate = harmful / max(1, int(memory["injected_count"]))
        context_components = _context_components(memory, context or RetrievalContext(goal=query), query)
        context_feedback = self.store.context_feedback(
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

        score = (
            0.20 * lexical
            + 0.06 * phrase
            + 0.06 * semantic
            + 0.09 * activation
            + 0.11 * historical_usefulness
            + 0.06 * importance
            + 0.05 * confidence
            + 0.06 * source_reliability
            + 0.06 * currentness
            + 0.03 * memory_type
            + 0.02 * uniqueness
            + 0.06 * graph
            + context_boost
            + 0.08 * context_adaptation
            - relevance_penalty
            - 0.08 * stale_risk
            - 0.12 * min(1.0, wrong_rate)
            - 0.15 * float(contradicted)
        )
        if superseded:
            score -= 0.18
        if memory["pinned"]:
            score += 0.08
        if memory["state"] == "cold":
            score -= 0.03
        if memory["state"] == "archived":
            score -= 0.10
        operator_policy = self.store.active_policy_adjustment(
            "retrieval",
            {
                "kind": str(memory.get("kind") or "semantic"),
                "source_type": str(memory.get("source_type") or "conversation"),
                "source_category": str(memory.get("source_category") or "AGENT_INFERENCE"),
            },
        )
        score += float(operator_policy.get("score_adjustment") or 0.0)
        if context_components["context_gate"] < 1.0:
            score = min(score, 0.01)

        components = {
            "lexical": lexical,
            "phrase": phrase,
            "semantic": semantic,
            "semantic_similarity": semantic,
            "activation": activation,
            "utility": historical_usefulness,
            "historical_usefulness": historical_usefulness,
            "importance": importance,
            "confidence": confidence,
            "source_reliability": source_reliability,
            "currentness": currentness,
            "memory_type": memory_type,
            "uniqueness": uniqueness,
            "graph": graph,
            "relevance_penalty": relevance_penalty,
            "stale_risk": stale_risk,
            "wrong_rate": min(1.0, wrong_rate),
            "superseded": float(superseded),
            "contradiction_risk": float(contradicted),
            "operator_policy": float(operator_policy.get("score_adjustment") or 0.0),
            "context_candidate": min(1.0, float(memory.get("context_candidate_score", 0.0)) / 4.0),
            "context_historical_usefulness": float(context_feedback["usefulness"]),
            "context_feedback_observations": min(
                1.0, float(context_feedback["observations"]) / 10.0
            ),
            "context_adaptation": context_adaptation,
            **context_components,
        }
        estimated_tokens = max(12, math.ceil(len(memory["content"]) / 4) + 18)
        return RetrievalResult(memory, max(0.0, min(1.0, score)), components, estimated_tokens)

    @staticmethod
    def _activation(memory: dict[str, Any]) -> float:
        age = _age_days(memory["last_used_at"] or memory["last_injected_at"] or memory["updated_at"])
        half_life = _HALF_LIFE_DAYS.get(memory["kind"], 120.0)
        half_life *= max(0.2, 1.15 - float(memory["volatility"]))
        recency = 1.0 / (1.0 + age / max(1.0, half_life))
        weighted_uses = (
            0.03 * int(memory["retrieved_count"])
            + 0.08 * int(memory["injected_count"])
            + 0.75 * int(memory["used_count"])
            + 1.5 * int(memory["success_count"])
            + 2.0 * int(memory["confirmed_count"])
            + 1.5 * int(memory.get("helpful_count", 0))
            + 2.0 * int(memory.get("validated_count", 0))
        )
        frequency = min(1.0, math.log1p(weighted_uses) / math.log(12.0))
        return min(1.0, 0.58 * recency + 0.42 * frequency)

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
        penalty = false / max(3.0, int(memory["injected_count"]) + 2.0)
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
) -> dict[str, float]:
    mode = str(memory.get("context_mode") or "standalone").casefold()
    scope = _memory_context_map(memory, "scope", "scope_json")
    preconditions = _memory_context_map(memory, "preconditions", "preconditions_json")
    entities = _memory_context_list(memory, "entities", "entities_json")
    systems = _memory_context_list(memory, "applicable_systems", "applicable_systems_json")
    versions = _memory_context_list(memory, "applicable_versions", "applicable_versions_json")

    active_scope = _clean_context_map(context.scope)
    if context.active_project:
        active_scope["project"] = context.active_project
    available_state = {**active_scope, **_clean_context_map(context.system_state)}
    query_folded = " ".join((query or "").casefold().split())
    context_entities = {item.casefold() for item in context.entities}
    if context.active_project:
        context_entities.add(context.active_project.casefold())
    context_entities.update(str(value).casefold() for value in active_scope.values())
    context_systems = {item.casefold() for item in context.applicable_systems}
    context_versions = {item.casefold() for item in context.applicable_versions}

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

    query_token_set = set(query_tokens(query))
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
    system_hits = sum(1 for system in systems if system.casefold() in context_systems or system.casefold() in query_folded)
    system_match = system_hits / len(systems) if systems else 1.0
    version_hits = sum(
        1 for version in versions if version.casefold() in context_versions or version.casefold() in query_folded
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
    prior = _SOURCE_RELIABILITY.get(str(memory.get("source_category") or "AGENT_INFERENCE"), 0.64)
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
    if not actual:
        return 0.0
    left = _clean_context_value(required).casefold()
    right = _clean_context_value(actual).casefold()
    if left == right:
        return 1.0
    if left in right or right in left:
        return 1.0
    return 0.0


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
    a = set(query_tokens(left.memory["content"]))
    b = set(query_tokens(right.memory["content"]))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
