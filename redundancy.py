"""Redundancy collapse: find memories that say the same thing, and plan one copy of each.

Exact duplicates are already impossible — `content_hash` is unique at write time. What
accumulates is *near* duplication: the same fact stated again days later in slightly
different words, usually from a vault sync re-ingesting a note. Each copy costs tokens,
dilutes attention, and gives the injection four ways to say one thing.

This is the planner, not the executioner. It groups memories by content similarity, picks a
canonical copy (most-used, then most-important, then oldest), and reports what collapsing
would save. Collapsing itself is a destructive store operation and is deliberately not
performed here: the plan is produced for a human to approve.

Deterministic: cluster order, canonical choice and output are all stable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Sequence

# Measured on real restatements: the same fact said twice in different words scores
# ~0.64 Jaccard / ~0.82 containment, while unrelated facts score ~0.04. A near-verbatim
# re-ingest scores 0.85+. Two rules, because either one alone mis-fires in one direction.
VERBATIM_THRESHOLD = 0.85
RESTATED_THRESHOLD = 0.60
RESTATED_CONTAINMENT = 0.80
_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "into", "over", "under", "vault",
    "note", "path", "section", "notes",
}
_WORD = re.compile(r"[a-z0-9][\w'-]*")


@lru_cache(maxsize=8192)
def tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD.findall(str(text or "").lower()) if w not in _STOP)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(a: frozenset[str], b: frozenset[str]) -> float:
    """How much of the smaller text is present in the larger one."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def is_duplicate(a: frozenset[str], b: frozenset[str], *, same_source: bool = False) -> bool:
    """Near-verbatim anywhere; a restatement only within the same note.

    Two different vault notes that share phrasing are two facts, not one — measured false
    positives were a TeslaMate note merged into a Docker note. Restatement tolerances only
    apply inside one source.
    """
    if not a or not b:
        return False
    if jaccard(a, b) >= VERBATIM_THRESHOLD:
        return True
    return same_source and (
        jaccard(a, b) >= RESTATED_THRESHOLD and containment(a, b) >= RESTATED_CONTAINMENT
    )


@dataclass(frozen=True)
class Memory:
    id: str
    content: str
    importance: float = 0.0
    used_count: int = 0
    created_at: str = ""
    pinned: bool = False
    state: str = "active"
    source_ref: str = ""

    @property
    def source(self) -> str:
        """The note a memory came from, with section anchors folded."""
        return str(self.source_ref or "").split("#", 1)[0].strip().lower()

    @property
    def tokens(self) -> frozenset[str]:
        return tokens(self.content)


def memory_from_row(row: dict[str, Any]) -> Memory:
    return Memory(
        id=str(row.get("id")),
        content=str(row.get("content") or ""),
        importance=float(row.get("importance") or 0.0),
        used_count=int(row.get("used_count") or 0),
        created_at=str(row.get("created_at") or ""),
        pinned=bool(row.get("pinned")),
        state=str(row.get("state") or "active"),
        source_ref=str(row.get("source_ref") or ""),
    )


def canonical(cluster: Sequence[Memory]) -> Memory:
    """The copy to keep: most used, then most important, then oldest, then lowest id."""
    return sorted(
        cluster,
        key=lambda m: (-m.used_count, -m.importance, m.created_at or "9999", m.id),
    )[0]


def cluster(memories: Iterable[Memory]) -> list[list[Memory]]:
    """Greedy clustering by similarity. Pinned memories are never absorbed."""
    ordered = sorted(
        memories, key=lambda m: (-m.used_count, -m.importance, m.created_at or "9999", m.id)
    )
    clusters: list[list[Memory]] = []
    # Inverted index over cluster representatives, so a memory is only compared against
    # clusters that share a token with it. Without this, 3,800 memories cost 7 million
    # pairwise comparisons and the report never finishes.
    index: dict[str, set[int]] = {}
    for memory in ordered:
        memory_tokens = memory.tokens
        if not memory_tokens:
            continue
        candidates: set[int] = set()
        for token in memory_tokens:
            candidates |= index.get(token, set())
        placed = False
        for position in sorted(candidates):
            group = clusters[position]
            if memory.pinned and group[0].pinned:
                # Two pinned memories are two decisions: never merge them.
                continue
            same_source = bool(memory.source) and memory.source == group[0].source
            if is_duplicate(memory_tokens, group[0].tokens, same_source=same_source):
                group.append(memory)
                placed = True
                break
        if not placed:
            clusters.append([memory])
            position = len(clusters) - 1
            for token in memory_tokens:
                index.setdefault(token, set()).add(position)
    return [group for group in clusters if len(group) > 1]


@dataclass(frozen=True)
class Plan:
    keep: Memory
    drop: tuple[Memory, ...]
    chars_saved: int

    @property
    def tokens_saved(self) -> int:
        return self.chars_saved // 4


def plan(memories: Iterable[Memory]) -> list[Plan]:
    plans: list[Plan] = []
    for group in cluster(memories):
        keep = canonical(group)
        drop = tuple(m for m in group if m.id != keep.id)
        if not drop:
            continue
        plans.append(
            Plan(keep=keep, drop=drop, chars_saved=sum(len(m.content) for m in drop))
        )
    return sorted(plans, key=lambda p: -p.chars_saved)
