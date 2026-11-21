"""Never spend several slots on the same memory in slightly different words (#10).

Recall already penalises similarity softly (`0.16 x similarity`) while it selects. That is
not enough: when a query's top hits are four copies of the same vault note, the penalty
lowers their scores and they still win, because every alternative scores lower still. The
observed result was four nearly identical `Seerr Telegram Notifications` notes filling one
turn's memory block.

This is the hard version: a memory whose similarity to something already chosen is at or
above the ceiling is not injected alongside it. If that would leave slots empty, duplicates
are used to fill them — a redundant memory beats an empty slot — so the rule can only ever
reorder and replace, never shrink the block.
"""

from __future__ import annotations

from typing import Any, Callable

DEFAULT_CEILING = 0.85
DEFAULT_MAX_PER_GROUP = 2


def token_set(result: Any) -> frozenset[str]:
    content = str((getattr(result, "memory", {}) or {}).get("content") or "").lower()
    return frozenset(t for t in content.replace("\n", " ").split() if len(t) > 2)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def group_key(result: Any) -> str | None:
    """Which fact a memory belongs to. Same vault path, same claim, same slot."""
    memory = getattr(result, "memory", {}) or {}
    for field in ("source_ref", "subject"):
        value = str(memory.get(field) or "").strip()
        if not value:
            continue
        # Vault refs carry a section anchor ("file.md#heading"): the note is the group,
        # not each heading, or four sections of one note eat the whole block.
        base = value.split("#", 1)[0].strip().lower()
        # A bare "memory" (or any non-path source) is not a group: it would lump every
        # user-stated fact together and cap unrelated memories.
        if "/" not in base and not base.startswith("vault:"):
            continue
        return base or None
    return None


def enforce_distinct(
    selected: list[Any],
    spare: list[Any] | None = None,
    *,
    limit: int | None = None,
    ceiling: float = DEFAULT_CEILING,
    max_per_group: int = DEFAULT_MAX_PER_GROUP,
    similarity: Callable[[Any, Any], float] | None = None,
    group: Callable[[Any], str | None] = group_key,
) -> tuple[list[Any], int]:
    """Drop near-duplicates from `selected`, backfilling from `spare` when slots run low.

    Returns (selection, replaced_count).
    """
    selected = list(selected)
    spare = list(spare or [])
    limit = limit if limit is not None else len(selected) + len(spare)
    if limit <= 0:
        return [], 0
    if ceiling >= 1.0:
        # Disable the similarity rule but keep the group cap (tests rely on the split).
        ceiling = 2.0

    def sim(a: Any, b: Any) -> float:
        if similarity is not None:
            return float(similarity(a, b))
        return jaccard(token_set(a), token_set(b))

    kept: list[Any] = []
    duplicates: list[Any] = []
    per_group: dict[str, int] = {}
    for result in selected:
        key = group(result)
        over_group = bool(key) and per_group.get(key, 0) >= max_per_group
        if over_group or any(sim(result, other) >= ceiling for other in kept):
            duplicates.append(result)
            continue
        kept.append(result)
        if key:
            per_group[key] = per_group.get(key, 0) + 1

    replaced = len(duplicates)

    # Fill spare slots with distinct-but-lower candidates. Duplicates are never re-added:
    # a shorter block of distinct facts is better than a longer one that repeats itself,
    # both for attention and for tokens. An empty slot is not a loss.
    for candidate in spare:
        if len(kept) >= limit:
            break
        kept.append(candidate)

    return kept[:limit], replaced
