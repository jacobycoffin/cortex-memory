"""Pure rank-fusion primitives: Reciprocal Rank Fusion and semantic cosine ranking.

Cortex ranks memories with a lexical/heuristic scorer, while a local embedding
model provides an independent ranking. Neither is best on its own, so the two
orders are combined here.

Reciprocal Rank Fusion (RRF) is used because it is *scale-free*: it only looks
at the position a document occupies in each ranking, so a BM25-ish score never
has to be normalised against a cosine similarity.

This module is deliberately pure: no I/O, no database access, and no imports
outside the standard library plus ``numpy``.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

__all__ = ["reciprocal_rank_fusion", "cosine_ranking", "fuse_semantic"]


def _validate_weights(weights: Sequence[float] | None, count: int) -> list[float]:
    """Resolve ``weights`` to a list of ``count`` non-negative floats."""
    if weights is None:
        return [1.0] * count
    resolved = [float(weight) for weight in weights]
    if len(resolved) != count:
        raise ValueError(
            f"weights has length {len(resolved)} but {count} ranking(s) were given"
        )
    for index, weight in enumerate(resolved):
        if weight < 0.0:
            raise ValueError(f"weights[{index}] is negative ({weight}); RRF weights must be >= 0")
    return resolved


def _rank_order(scores: Mapping[str, float], first_seen: Mapping[str, int]) -> list[str]:
    """Sort ids by descending score, breaking ties on first appearance then id.

    ``first_seen`` holds the position at which each id was first encountered
    while the rankings were scanned, which makes the output fully deterministic
    and reproducible.
    """
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], first_seen[doc_id], doc_id))


def _fused_scores(
    rankings: Sequence[Sequence[str]],
    k: int,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Raw RRF score for every id, without sorting.

    Score(id) = sum over rankings of ``weight_i / (k + rank + 1)`` where ``rank``
    is the id's 0-based index in that ranking. Only the first occurrence of a
    duplicated id inside a single ranking contributes a score term; later ids
    keep their own index in the supplied ranking (duplicates are not re-packed).
    """
    resolved = _validate_weights(weights, len(rankings))

    scores: dict[str, float] = {}
    for index, ranking in enumerate(rankings):
        weight = resolved[index]
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank + 1)
    return scores


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[str]:
    """Fuse ranked id lists into a single ranking with Reciprocal Rank Fusion.

    Args:
        rankings: Ranked lists of memory ids, best first. Ids may repeat inside
            one list (only the first occurrence counts) and may be missing from
            some lists.
        k: RRF damping constant. Larger ``k`` flattens the influence of rank.
        weights: Optional per-ranking weight, same length as ``rankings``.
            Defaults to all ``1.0``. Must be non-negative.

    Returns:
        Ids sorted by descending fused score. Ties are broken deterministically
        by first-appearance order, then lexicographically by id.
    """
    if not rankings:
        # Still validate weights so a bad call stays loud.
        _validate_weights(weights, 0)
        return []

    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for ranking in rankings:
        for doc_id in ranking:
            if doc_id not in first_seen:
                first_seen[doc_id] = len(first_seen)

    scores = _fused_scores(rankings, k, weights)
    return _rank_order(scores, first_seen)


def cosine_ranking(
    query_vec: Sequence[float],
    candidates: Mapping[str, Sequence[float]],
) -> list[str]:
    """Rank candidate ids by cosine similarity to ``query_vec`` using numpy.

    Defensive by design: a candidate whose vector has a different length than
    the query, or whose vector (or the query) has zero norm, is skipped instead
    of raising. Malformed/non-finite vectors are skipped too.

    Ties are broken deterministically by mapping order, then lexicographically
    by id.
    """
    try:
        query = np.asarray(query_vec, dtype=float).ravel()
    except (TypeError, ValueError):
        return []
    if query.size == 0:
        return []
    query_norm = float(np.linalg.norm(query))
    if not np.isfinite(query_norm) or query_norm <= 0.0:
        return []

    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for candidate_id, candidate_vec in candidates.items():
        try:
            vector = np.asarray(candidate_vec, dtype=float).ravel()
        except (TypeError, ValueError):
            continue
        if vector.shape != query.shape:
            continue
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0.0:
            continue
        similarity = float(np.dot(query, vector) / (query_norm * norm))
        if not np.isfinite(similarity):
            continue
        scores[candidate_id] = similarity
        first_seen[candidate_id] = len(first_seen)
    return _rank_order(scores, first_seen)


def fuse_semantic(
    cortex_ranked: Sequence[str],
    semantic_ranked: Sequence[str],
    k: int = 60,
    semantic_weight: float = 1.0,
) -> list[str]:
    """Fuse Cortex's own ranking with an embedding-model ranking via RRF.

    Thin convenience wrapper: RRF over exactly the two rankings with
    ``weights=[1.0, semantic_weight]``.
    """
    return reciprocal_rank_fusion(
        [list(cortex_ranked), list(semantic_ranked)],
        k=k,
        weights=[1.0, semantic_weight],
    )
