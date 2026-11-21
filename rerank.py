"""Two-stage recall: wide and cheap first, then a proper lexical rerank (#6).

Stage one gathers candidates with FTS, features, context and graph — broad and fast, and
deliberately cheap enough to over-collect. Its scores are a blend of many weak signals, so
the ordering inside the candidate pool is noisy: a memory that contains the query's exact
terms can sit below one that merely shares a category.

Stage two reranks the top slice with BM25 over the actual contents — term frequency,
saturation, and length normalisation, with IDF computed *within the candidate set*. That is
the classic second stage, and it is deterministic: no model, no network, no training.

The contribution is additive and small (max +0.05), so the rerank only reorders candidates
that stage one could not separate. A convex 50/50 blend was measured and rejected: it
replaced stage one's ordering rather than refining it, costing 6.7 points of hit@6.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import replace
from typing import Any, Iterable, Sequence

DEFAULT_TOP_N = 50
# Additive, not a convex blend: measured, a 50/50 blend REPLACED stage one's ordering and
# cost 6.7 points of hit@6 (0.6167 -> 0.5500), because normalising BM25 to its own peak
# drowns the calibrated stage-one scores it was supposed to refine.
try:  # tunable for measurement; the shipped default is 0.05
    RERANK_WEIGHT = float(os.environ.get("CORTEX_RERANK_WEIGHT", "0.05"))
except ValueError:
    RERANK_WEIGHT = 0.05
K1 = 1.2
B = 0.75

_WORD = re.compile(r"[a-z0-9][\w'’-]*")
_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "into", "over", "under", "was",
    "were", "are", "you", "your", "our", "its", "it", "a", "an", "of", "to", "in", "on",
    "is", "be", "as", "at", "by", "or", "if", "we", "i", "me", "my", "do", "did", "does",
}


def terms(text: str) -> list[str]:
    return [w for w in _WORD.findall(str(text or "").lower()) if w not in _STOP and len(w) > 1]


def bm25(
    query: str,
    documents: Sequence[Sequence[str]],
    *,
    k1: float = K1,
    b: float = B,
) -> list[float]:
    """BM25 scores for the given tokenised documents, in order."""
    query_terms = terms(query)
    if not query_terms or not documents:
        return [0.0 for _ in documents]

    lengths = [len(doc) for doc in documents]
    average = sum(lengths) / len(lengths) if lengths else 0.0
    if average <= 0:
        return [0.0 for _ in documents]

    document_frequency: Counter[str] = Counter()
    for doc in documents:
        unique = set(doc)
        for term in query_terms:
            if term in unique:
                document_frequency[term] += 1

    total = len(documents)
    scores: list[float] = []
    for doc, length in zip(documents, lengths):
        counts = Counter(doc)
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            norm = frequency + k1 * (1 - b + b * (length / average))
            score += idf * (frequency * (k1 + 1)) / norm
        scores.append(score)
    return scores


def rerank(
    results: Iterable[Any],
    query: str,
    *,
    top_n: int = DEFAULT_TOP_N,
    weight: float = RERANK_WEIGHT,
) -> tuple[list[Any], int]:
    """Blend BM25 into the top `top_n` results. Returns (results, reranked_count).

    Results beyond `top_n` are returned untouched: they were never candidates for the
    final block, and rescoring them would make the rerank's cost grow with the pool.
    """
    results = list(results)
    if not results or not terms(query):
        return results, 0

    head = results[: max(0, int(top_n))]
    tail = results[max(0, int(top_n)) :]
    documents = [terms((getattr(r, "memory", {}) or {}).get("content", "")) for r in head]
    scores = bm25(query, documents)
    # Min-max, not peak-normalisation: dividing by the maximum gives the top candidate a
    # constant full weight, which shifts every score equally and therefore reorders
    # nothing (measured: identical frozen results at weights 0.05, 0.10 and 0.20).
    low = min(scores) if scores else 0.0
    peak = (max(scores) - low) if scores else 0.0

    out: list[Any] = []
    changed = 0
    for result, score in zip(head, scores):
        normalised = ((score - low) / peak) if peak > 0 else 0.0
        old = float(getattr(result, "score", 0.0) or 0.0)
        blended = old + weight * normalised
        components = dict(getattr(result, "components", {}) or {})
        components["bm25"] = round(score, 6)
        components["rerank_delta"] = round(blended - old, 6)
        if abs(blended - old) > 1e-9:
            changed += 1
        out.append(replace(result, score=blended, components=components))
    return out + tail, changed
