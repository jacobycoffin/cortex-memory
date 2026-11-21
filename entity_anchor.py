"""Entity-anchored recall: find the names in the query, then expand through them (#8).

`context_search` can already rank by `entity_match`, but the entity list came from the
caller's `RetrievalContext` — which in a real chat is empty. So the strongest anchor the
store has (6,644 memories carry entities; 625 distinct ones) was never used.

This extracts entities from the query by matching them against the names the store actually
holds, so "how is Mack's access set up?" anchors on `Mack` and expands through every memory
attached to it.

Matching is whole-name and case-insensitive, longest name first, so `Cortex Brain Mechanics`
wins over a bare `Cortex` when both are present. Dates and bare numbers are excluded: they
appear as entities in vault imports and anchor nothing.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Iterable

MAX_ENTITIES = 5
_MIN_LEN = 3
_DATE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$|^\d+$")


def usable(name: str) -> bool:
    text = str(name or "").strip()
    if len(text) < _MIN_LEN or _DATE.match(text):
        return False
    if "›" in text:  # a heading, not an entity
        return False
    return any(ch.isalpha() for ch in text)


def known_entities(store: Any, *, limit: int = 2000) -> list[str]:
    """Distinct entity names in the store, longest first. Never raises."""
    conn = getattr(store, "_conn", None)
    if conn is None:
        return []
    try:
        rows = conn.execute(
            """
            SELECT value, COUNT(*) AS n
            FROM memories m, json_each(m.entities_json)
            WHERE json_valid(m.entities_json)
            GROUP BY value
            ORDER BY n DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    except sqlite3.Error:
        return []
    names = [str(row[0]) for row in rows if usable(row[0])]
    return sorted(set(names), key=lambda n: (-len(n), n.lower()))


def extract_entities(
    query: str, names: Iterable[str], *, max_entities: int = MAX_ENTITIES
) -> list[str]:
    """Names from `names` that appear in the query, longest first."""
    text = f" {str(query or '').lower()} "
    if not text.strip():
        return []
    padded = re.sub(r"[^0-9a-z]+", " ", text)
    found: list[str] = []
    for name in sorted({str(n) for n in names if usable(n)}, key=lambda n: (-len(n), n.lower())):
        needle = re.sub(r"[^0-9a-z]+", " ", f" {name.lower()} ").strip()
        if needle and f" {needle} " in padded:
            if any(needle in existing.lower() for existing in found):
                continue  # already covered by a longer match
            found.append(name)
            if len(found) >= max_entities:
                break
    return found


def entity_memory_ids(store: Any, entities: Iterable[str], *, limit: int = 200) -> list[str]:
    """Memories attached to any of these entities, for graph expansion. Never raises."""
    names = [str(e) for e in entities if str(e).strip()]
    if not names:
        return []
    conn = getattr(store, "_conn", None)
    if conn is None:
        return []
    placeholders = ",".join("?" for _ in names)
    try:
        rows = conn.execute(
            f"""
            SELECT m.id
            FROM memories m, json_each(m.entities_json)
            WHERE json_valid(m.entities_json)
              AND value IN ({placeholders})
              AND m.state IN ('active','cold')
            LIMIT ?
            """,
            (*names, int(limit)),
        ).fetchall()
    except sqlite3.Error:
        return []
    return [str(row[0]) for row in rows]


def parse_entities(raw: Any) -> list[str]:
    """Entities as stored on a memory row (JSON string or list)."""
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    try:
        loaded = json.loads(raw or "[]")
    except (ValueError, TypeError):
        return []
    return [str(x) for x in loaded] if isinstance(loaded, list) else []
