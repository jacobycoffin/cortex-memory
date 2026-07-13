"""Fast deterministic retrieval and utility scoring for Cortex."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

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
    ) -> tuple[list[RetrievalResult], RetrievalDiagnostics]:
        if not query or limit <= 0 or token_budget <= 0:
            return [], RetrievalDiagnostics(0, 0, 0, True)
        expanded_query = _expand_query(query)
        candidate_limit = max(40, limit * 8)
        lexical_candidates = self.store.fts_search(
            expanded_query, limit=candidate_limit, include_archived=include_archived
        )
        feature_candidates = self.store.feature_search(
            expanded_query, limit=candidate_limit, include_archived=include_archived
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
        candidates = list(candidates_by_id.values())
        superseded = self.store.superseded_ids(list(candidates_by_id)) if temporal_mode == "current" else set()
        scored: dict[str, RetrievalResult] = {}
        for candidate in candidates:
            result = self._score(
                expanded_query,
                candidate,
                temporal_mode=temporal_mode,
                as_of=as_of,
                superseded=candidate["id"] in superseded,
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
                )
                scored[neighbor_id] = result

        ranked = sorted(scored.values(), key=lambda r: (r.score, r.memory["pinned"]), reverse=True)
        selected: list[RetrievalResult] = []
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
            if (
                result.components.get("relevance_penalty", 0.0) > 0.0
                and graph_depth <= 1
                and result.memory.get("kind") not in {"procedure", "prospective"}
            ):
                # Focused/lean recall requires direct evidence. Indirect low-
                # overlap associations are reserved for the planner's deep,
                # multi-hop mode where they can add useful context on purpose.
                continue
            if result.score < effective_threshold and not result.memory["pinned"]:
                continue
            if any(_memory_similarity(result, prior) >= 0.86 for prior in selected):
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
                continue
            if consumed + result.estimated_tokens > token_budget:
                continue
            selected.append(result)
            consumed += result.estimated_tokens
        return selected, RetrievalDiagnostics(
            candidate_count=len(scored),
            selected_count=len(selected),
            estimated_tokens=consumed,
            abstained=not selected,
        )

    def _score(
        self,
        query: str,
        memory: dict[str, Any],
        *,
        graph: float = 0.0,
        temporal_mode: str = "current",
        as_of: str | None = None,
        superseded: bool = False,
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
        utility = self._utility(memory)
        confidence = float(memory["confidence"]) * float(memory["trust"])
        currentness = self._currentness(memory, temporal_mode=temporal_mode, as_of=as_of)
        importance = float(memory["importance"])
        uniqueness = float(memory.get("uniqueness", 1.0))
        stale_risk = self._stale_risk(memory)
        harmful = int(memory.get("harmful_count", 0)) + int(memory["false_positive_count"])
        wrong_rate = harmful / max(1, int(memory["injected_count"]))

        score = (
            0.36 * lexical
            + 0.08 * phrase
            + 0.02 * semantic
            + 0.13 * activation
            + 0.14 * utility
            + 0.09 * importance
            + 0.07 * confidence
            + 0.07 * currentness
            + 0.04 * uniqueness
            + 0.10 * graph
            - relevance_penalty
            - 0.08 * stale_risk
            - 0.12 * min(1.0, wrong_rate)
        )
        if superseded:
            score -= 0.18
        if memory["pinned"]:
            score += 0.08
        if memory["state"] == "cold":
            score -= 0.03
        if memory["state"] == "archived":
            score -= 0.10

        components = {
            "lexical": lexical,
            "phrase": phrase,
            "semantic": semantic,
            "activation": activation,
            "utility": utility,
            "importance": importance,
            "confidence": confidence,
            "currentness": currentness,
            "uniqueness": uniqueness,
            "graph": graph,
            "relevance_penalty": relevance_penalty,
            "stale_risk": stale_risk,
            "wrong_rate": min(1.0, wrong_rate),
            "superseded": float(superseded),
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
