"""Rank on what the agent used, not on what it happened to retrieve (backlog item #3).

`access_log` separates two very different events: `retrieved` (the memory was injected)
and `used` (the memory informed the answer). Only the second is evidence that a memory is
worth surfacing, and until now ranking ignored both.

Retrieval counts are popularity — they reward memories that match many queries, which is
how a note about VR games became the most-retrieved memory in the store. Use counts are
outcomes. This adds a small, capped boost from use, deliberately smaller than any real
relevance signal: it breaks ties toward memories that have actually worked, and it can
never lift an irrelevant memory over a relevant one.

The boost is log-scaled so the tenth use is worth much less than the second, and capped so
a frequently used memory cannot dominate the ranking forever.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Iterable

try:  # package import when installed as a plugin
    from .store_read import select
except ImportError:  # standalone checkout / tests
    from store_read import select

PER_USE = 0.03
MAX_BOOST = 0.09


def use_boost(used_count: int) -> float:
    """Boost for a memory used `used_count` times. Zero for never used."""
    if used_count <= 0:
        return 0.0
    return min(MAX_BOOST, PER_USE * math.log1p(used_count))


def used_counts(store: Any, memory_ids: Iterable[str]) -> dict[str, int]:
    """Used-event counts for the given memories, in one query."""
    ids = [str(m) for m in memory_ids if m]
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = select(
        store,
        f"""SELECT memory_id, COUNT(*) FROM access_log
            WHERE event = 'used' AND memory_id IN ({placeholders})
            GROUP BY memory_id""",
        tuple(ids),
    )
    return {str(row[0]): int(row[1]) for row in rows}


def apply_usage_boost(store: Any, results: Iterable[Any]) -> tuple[list[Any], int]:
    """Add the use boost to each result's score in place.

    Returns (results, boosted_count). Records `use_boost` and `use_count` on the
    components so a ranking decision can be explained after the fact.
    """
    results = list(results)
    counts = used_counts(
        store,
        [str((getattr(r, "memory", {}) or {}).get("id") or "") for r in results],
    )
    updated: list[Any] = []
    boosted = 0
    for result in results:
        memory_id = str((getattr(result, "memory", {}) or {}).get("id") or "")
        count = counts.get(memory_id, 0)
        boost = use_boost(count)
        if boost <= 0.0:
            updated.append(result)
            continue
        components = dict(getattr(result, "components", {}) or {})
        components["use_count"] = count
        components["use_boost"] = round(boost, 6)
        # RetrievalResult is a frozen dataclass: replace, do not mutate.
        updated.append(
            replace(result, score=float(result.score) + boost, components=components)
        )
        boosted += 1
    return updated, boosted
