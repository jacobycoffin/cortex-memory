"""Capture confirmation when a fact is restated (backlog item #216).

Only 82 of 3,829 active memories (2.1%) carried any corroboration, so the recency-vs-
stability blend (#9) was inert on live data: nothing ever incremented `confirmed_count`
because nothing looked at a *second* statement of the same fact.

This module does the looking. When a fact arrives that an existing memory already states,
that is evidence — the user said it twice — and the existing memory gets a `confirmed`
event instead of a new near-duplicate row. Three things follow from that:

1. `confirmed_count` starts moving, so stability can settle ties.
2. The store stops growing near-duplicates (redundancy #11 measured 34 retirable copies
   already; this is the tap that was filling the bucket).
3. Nothing is destroyed: the original memory is untouched apart from its counters.

The dangerous case is a **correction**, which is also a restatement: "the window moved to
Tuesday" mentions the same words as "the window is Sunday". Corrections must create a new
memory (and supersede the old one), so contrast markers in the new text disqualify it from
being a confirmation. Getting this backwards would silently refuse to record updates, which
is far worse than a duplicate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .redundancy import (
    RESTATED_CONTAINMENT,
    RESTATED_THRESHOLD,
    VERBATIM_THRESHOLD,
    containment,
    jaccard,
    tokens,
)

# Text that signals "this is a change, not an agreement". Deliberately broad: a missed
# confirmation costs one duplicate row, a false confirmation loses a correction.
CONTRAST = re.compile(
    r"\b(?:no longer|instead of|rather than|moved to|changed to|changed from|switched to|"
    r"replaced by|supersed\w*|correction\b|actually it'?s|actually,? it is|was wrong|"
    r"that'?s wrong|now it'?s|now it is|deprecated|obsolete|stopped using|"
    r"not anymore|is now|are now)\b",
    re.I,
)

MAX_CANDIDATES = 200


@dataclass(frozen=True)
class Confirmation:
    memory_id: str
    similarity: float
    containment: float
    reason: str

    @property
    def verbatim(self) -> bool:
        return self.similarity >= VERBATIM_THRESHOLD


def is_correction(text: str) -> bool:
    """Does the new statement contradict rather than repeat?"""
    return bool(CONTRAST.search(str(text or "")))


def _score(candidate: str, existing: str) -> tuple[float, float]:
    a, b = tokens(candidate), tokens(existing)
    return jaccard(a, b), containment(a, b)


def _candidates(store: Any, text: str, limit: int) -> list[dict[str, Any]]:
    """Two lexical lookups, because one is not enough.

    A long memory used as a query ranks its own near-copies below other long notes, and a
    handful of leading words ranks by the boilerplate prefix every note shares. Sorted
    distinctive tokens behave like a blocking key and surface the actual duplicates, so both
    are fetched and merged.
    """
    queries = [text]
    distinctive = " ".join(sorted(tokens(text))[:16])
    if distinctive and distinctive != text:
        queries.insert(0, distinctive)
    found: dict[str, dict[str, Any]] = {}
    for query in queries:
        try:
            rows = store.fts_search(
                query, limit=limit, include_archived=False, evidence_lookup=False
            )
        except Exception:  # noqa: BLE001 - a failed lookup must not block a capture
            continue
        for row in rows:
            found.setdefault(str(row["id"]), row)
    return list(found.values())


def find_confirmation(
    store: Any,
    content: str,
    *,
    kind: str | None = None,
    exclude_id: str | None = None,
    limit: int = MAX_CANDIDATES,
) -> Confirmation | None:
    """The existing memory this content restates, if any.

    Candidate generation reuses the store's own lexical search, so the call costs one query
    rather than a scan of the store. The decision then uses the same two rules redundancy
    uses — near-verbatim, or restatement with high containment — because either alone
    mis-fires in one direction.
    """
    text = str(content or "").strip()
    if not text or is_correction(text):
        return None
    candidates = _candidates(store, text, limit)

    best: Confirmation | None = None
    for candidate in candidates:
        if exclude_id is not None and str(candidate.get("id")) == str(exclude_id):
            continue
        if kind and str(candidate.get("kind")) != kind:
            continue
        existing = str(candidate.get("content") or "")
        if not existing:
            continue
        similarity, contain = _score(text, existing)
        if similarity >= VERBATIM_THRESHOLD:
            reason = "near-verbatim restatement"
        elif similarity >= RESTATED_THRESHOLD and contain >= RESTATED_CONTAINMENT:
            reason = "restated with high containment"
        else:
            continue
        if best is None or similarity > best.similarity:
            best = Confirmation(
                memory_id=str(candidate["id"]),
                similarity=round(similarity, 4),
                containment=round(contain, 4),
                reason=reason,
            )
    return best


def record_confirmation(
    store: Any,
    content: str,
    *,
    session_id: str | None = None,
    kind: str | None = None,
) -> Confirmation | None:
    """Record a `confirmed` event on the restated memory, if one exists.

    The event goes through `log_access`, which already knows what a `confirmed` event means
    (`confirmed_count` and `validated_count`, plus the timestamp) — so there is one
    definition of confirmation in the codebase, not two.
    """
    confirmation = find_confirmation(store, content, kind=kind)
    if confirmation is None:
        return None
    store.log_access(
        confirmation.memory_id,
        "confirmed",
        query=str(content or "")[:500],
        session_id=session_id,
    )
    return confirmation


def corroboration_stats(store: Any) -> dict[str, int]:
    """How many active memories carry any corroboration at all (the item's headline)."""
    row = store._conn.execute(
        """SELECT COUNT(*) AS total,
                  SUM(CASE WHEN confirmed_count + validated_count + helpful_count > 0
                           THEN 1 ELSE 0 END) AS corroborated,
                  SUM(confirmed_count) AS confirmations,
                  SUM(validated_count) AS validations,
                  SUM(helpful_count) AS helpful
           FROM memories WHERE state = 'active'"""
    ).fetchone()
    return {key: int(row[key] or 0) for key in row.keys()}
