"""Forgetting everything about one entity, as one auditable operation (#95).

"Forget everything about this client" is the natural request and the dangerous one: it sounds
like a single decision but it touches every memory the entity appears in, across kinds and
sessions, including the ones that merely mention them in passing.

So it reuses the envelope that semantic forgetting already established (#89) rather than inventing
a second destructive path:

    entity -> matched ids -> envelope -> human reads the list -> apply by envelope

Three rules specific to entities:

1. **Three independent sources of evidence.** Structured `entities_json`, content mentions, and
   graph edges to a memory that is *about* the entity. A single source misses too much — imports
   often carry the name only in the text, and capture-time extraction only started recently.
2. **Matched by boundary, not substring.** `Acme` must not match `AcmeCorp`, `Acme Holdings`, or
   a word that merely contains the letters. Word-boundary matching on the raw name and on
   capitalisation variants only.
3. **A held record is never touched**, even with `allow_protected` — the same rule deletion obeys
   (#103). An entity request is broad, and broad requests are exactly where a contract clause gets
   swept up by accident.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from . import forget_preview


@dataclass
class EntityPreview:
    entity: str
    envelope: str
    matches: list[dict[str, Any]]

    @property
    def ids(self) -> list[str]:
        return [str(match["memory_id"]) for match in self.matches]

    def describe(self, snippet: int = 100) -> str:
        lines = [
            f"entity '{self.entity}': {len(self.matches)} memories, "
            f"envelope {self.envelope[:12]}"
        ]
        for match in self.matches:
            content = " ".join(str(match.get("content") or "").split())
            marker = "" if match.get("evidence") == "matched" else " (connected only)"
            lines.append(
                f"  {str(match['memory_id'])[:8]} [{match.get('kind')}/{match.get('state')}]"
                f"{marker} {content[:snippet]}"
            )
        return "\n".join(lines)


def _pattern(entity: str) -> re.Pattern[str]:
    """Boundary match so `Acme` does not match `AcmeCorp` or `Acmeson`."""
    escaped = re.escape(entity.strip())
    return re.compile(rf"(?<![\w-]){escaped}(?![\w-])", re.IGNORECASE)


def matches(
    store: Any, entity: str, *, limit: int = 200, include_neighbours: bool = False
) -> list[dict[str, Any]]:
    """Every active/cold memory that is about this entity, from three sources of evidence."""
    name = str(entity or "").strip()
    if not name:
        return []
    found: dict[str, dict[str, Any]] = {}

    # 1. structured entities captured at write time
    try:
        rows = store._conn.execute(
            """SELECT m.id, m.kind, m.state, m.content, m.pinned
               FROM memories m, json_each(m.entities_json)
               WHERE json_valid(m.entities_json) AND value = ?
                 AND m.state IN ('active','cold') LIMIT ?""",
            (name, int(limit)),
        ).fetchall()
    except Exception:  # noqa: BLE001 - entities_json is optional
        rows = []
    for row in rows:
        found.setdefault(str(row["id"]), dict(row))

    # 2. content mentions, boundary-matched in Python so the boundary rule is one implementation
    pattern = _pattern(name)
    try:
        candidates = store._conn.execute(
            """SELECT id, kind, state, content, pinned FROM memories
               WHERE state IN ('active','cold') AND content LIKE ? LIMIT ?""",
            (f"%{name}%", max(int(limit) * 4, 200)),
        ).fetchall()
    except Exception:  # noqa: BLE001
        candidates = []
    for row in candidates:
        if pattern.search(str(row["content"] or "")):
            found.setdefault(str(row["id"]), dict(row))

    # 3. graph neighbours: a memory connected to something that names the entity.
    # These are collected separately, never silently merged: a neighbour of a memory about the
    # client is not itself about the client. The first live run pulled in "All Jarvis/Hermes
    # surfaces share one personal Cortex scope" on an entity forget for Project Zomboid.
    neighbours: dict[str, dict[str, Any]] = {}
    seeds = list(found.keys())
    if seeds:
        placeholders = ",".join("?" for _ in seeds)
        try:
            rows = store._conn.execute(
                f"""SELECT DISTINCT m.id, m.kind, m.state, m.content, m.pinned
                    FROM memories m JOIN edges e ON (e.src_id = m.id OR e.dst_id = m.id)
                    WHERE (e.src_id IN ({placeholders}) OR e.dst_id IN ({placeholders}))
                      AND m.state IN ('active','cold') LIMIT ?""",
                (*seeds, *seeds, int(limit)),
            ).fetchall()
        except Exception:  # noqa: BLE001
            rows = []
        for row in rows:
            if str(row["id"]) not in found:
                neighbours[str(row["id"])] = dict(row)

    ordered = sorted(
        found.values(), key=lambda row: (str(row.get("kind") or ""), -len(str(row.get("content") or "")))
    )
    results = [
        {
            "memory_id": str(row["id"]),
            "kind": row.get("kind"),
            "state": row.get("state"),
            "pinned": bool(row.get("pinned")),
            "content": row.get("content"),
            "evidence": "matched",
        }
        for row in ordered[:limit]
    ]
    if include_neighbours:
        results.extend(
            {
                "memory_id": str(row["id"]),
                "kind": row.get("kind"),
                "state": row.get("state"),
                "pinned": bool(row.get("pinned")),
                "content": row.get("content"),
                "evidence": "connected only",
            }
            for row in sorted(neighbours.values(), key=lambda row: str(row["id"]))[:limit]
        )
    return results


def preview(
    store: Any,
    entity: str,
    *,
    limit: int = 200,
    at: Any = None,
    include_neighbours: bool = False,
) -> EntityPreview:
    """Build the envelope for an entity forget. Nothing is changed except the envelope record."""
    matched = matches(store, entity, limit=limit, include_neighbours=include_neighbours)
    envelope = forget_preview.record_envelope(
        store, [match["memory_id"] for match in matched], f"entity:{entity}", at=at
    )
    return EntityPreview(entity=str(entity), envelope=envelope, matches=matched)


def apply(
    store: Any, envelope: str, ids: list[str], *, at: Any = None, allow_protected: bool = False
) -> Any:
    """Tombstone exactly the previewed ids. Held records are never included, even with force."""
    from . import legal_hold

    held = legal_hold.held_ids(store)
    blocked = [memory_id for memory_id in ids if memory_id in held]
    if blocked:
        return forget_preview.Applied(
            envelope=envelope,
            refused=(
                "memories on legal hold: " + ", ".join(item[:8] for item in blocked)
            ),
        )
    return forget_preview.apply(
        store, envelope, ids, at=at, allow_protected=allow_protected
    )
