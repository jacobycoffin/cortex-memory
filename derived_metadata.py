"""Regenerating a neighbour may touch its metadata, never its content (#69).

Graph work has a tempting shortcut: to improve a neighbour's searchability, rewrite it. Add a
keyword, tidy the phrasing, refresh the summary. Every one of those edits looks small and each one
is a silent content change — an unnamed, unattributed rewrite of something the user said, with no
date, no author and no diff.

So the split is explicit in code:

| field | may be regenerated |
|---|---|
| `keywords`, `tags`, `features`, derived vectors | **yes**, any time |
| `content`, `confidence`, `kind`, `source`, `created_at`, `supersedes_id` | **never** |

`regenerate()` recomputes the derived rows and then **verifies the content fingerprint is unchanged**
before returning; if it is not, the whole operation is reported as a refusal rather than a partial
success. `assert_immutable()` exists for callers that already hold two versions of a row — a
consolidation pass, an import, a replay — so the check is a library call instead of a review habit.

This is the difference between "the store got better at finding this" and "the store quietly edited
something Jacoby said". Only the first is allowed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable

# Recomputable from content at any time. Rewriting these loses nothing.
DERIVED_FIELDS: tuple[str, ...] = (
    "features",
    "keywords",
    "tags",
    "tokens",
    "normalized_content",
    "fts_row",
)

# The record itself. A regeneration that changes any of these is not a regeneration.
AUTHORITATIVE_FIELDS: tuple[str, ...] = (
    "content",
    "confidence",
    "kind",
    "source_type",
    "source_category",
    "created_at",
    "observed_at",
    "supersedes_id",
    "entities_json",
)


def fingerprint(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:16]


@dataclass
class Result:
    memory_id: str
    regenerated: bool
    changed_fields: list[str] = field(default_factory=list)
    refused: str = ""

    def describe(self) -> str:
        if self.refused:
            return f"refused: {self.refused}"
        if not self.regenerated:
            return "nothing to regenerate"
        return "derived metadata regenerated"


def snapshot(row: dict[str, Any]) -> dict[str, str]:
    """The fields that must not move, as comparable strings."""
    return {
        name: str(row.get(name) if row.get(name) is not None else "")
        for name in AUTHORITATIVE_FIELDS
    }


def changed(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Which authoritative fields differ between two versions of a row."""
    left = snapshot(before)
    right = snapshot(after)
    return [name for name in AUTHORITATIVE_FIELDS if left[name] != right[name]]


def assert_immutable(before: dict[str, Any], after: dict[str, Any]) -> None:
    """Raise if a regeneration (or any caller) moved something it may not move."""
    moved = changed(before, after)
    if moved:
        raise ValueError(
            "refusing to change authoritative fields: " + ", ".join(sorted(moved))
        )


def regenerate(store: Any, memory_id: str) -> Result:
    """Recompute derived metadata for one memory, verifying content stays put."""
    row = store.get_memory(memory_id)
    if not row:
        return Result(str(memory_id), False, refused="no such memory")
    before = snapshot(row)
    try:
        store._backfill_memory_features()
    except Exception as error:  # noqa: BLE001 - a failed rebuild is reported, not raised
        return Result(str(memory_id), False, refused=f"rebuild failed: {type(error).__name__}")
    after_row = store.get_memory(memory_id) or {}
    moved = changed(before, after_row)
    if moved:
        return Result(str(memory_id), False, changed_fields=moved,
                      refused="authoritative fields moved: " + ", ".join(sorted(moved)))
    return Result(str(memory_id), True)


def regenerate_many(store: Any, memory_ids: Iterable[str]) -> dict[str, Any]:
    results = [regenerate(store, memory_id) for memory_id in memory_ids]
    return {
        "regenerated": sum(1 for item in results if item.regenerated),
        "refused": [item.describe() for item in results if item.refused],
        "results": results,
    }


def report(store: Any) -> dict[str, Any]:
    """How the store is split between immutable records and derived rows."""
    memories = store._conn.execute(
        "SELECT COUNT(*) FROM memories WHERE state NOT IN ('tombstoned','deleted')"
    ).fetchone()[0]
    try:
        features = store._conn.execute("SELECT COUNT(*) FROM memory_features").fetchone()[0]
    except Exception:  # noqa: BLE001 - the table is optional in a stripped store
        features = 0
    return {
        "memories": int(memories or 0),
        "derived_rows": int(features or 0),
        "derived_fields": list(DERIVED_FIELDS),
        "authoritative_fields": list(AUTHORITATIVE_FIELDS),
    }
