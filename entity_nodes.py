"""Entities as records with a page, not just strings on memories (#49).

A name that appears on two hundred memories is a thing the store knows something about, and today the
only way to ask about it is to search for the string and read the results. This gives each entity a
row of its own — a kind, a canonical name, its aliases, the facts that mention it, when it was first
and last seen, and the other entities it is connected to — so "what do we know about the Proxmox box"
and "what changed about it this week" are lookups rather than searches.

Two deliberate limits:

- **a kind is a guess and says so.** Classification uses a small lexicon and the shape of the name
  (a host-shaped token is a device, an email is an account); anything unclear stays `unknown`.
  A wrong kind is worse than no kind, because every later question filters on it.
- **the page is assembled, never stored.** Facts, aliases and neighbours are read from the tables that
  already own them, so a page cannot drift from the store it describes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

TABLE = "entity_nodes"

KINDS = ("person", "project", "service", "device", "account", "place", "unknown")

PEOPLE = ("jacoby", "hermes", "ledger", "teknium", "kaya", "jarvis", "rowan")
PROJECTS = (
    "cortex", "cortex-memory", "duelist-road", "temple", "temple-ios", "website", "arttools",
    "rigtools", "camtools", "scratch", "homebox",
)
SERVICES = (
    "nginx", "proxmox", "jellyfin", "immich", "homebox", "opencode", "t3code", "cloudflare",
    "tailscale", "docker", "sqlite", "postgres", "redis",
)
HOST = re.compile(r"^(?=[a-z0-9._-]*\d)[a-z][a-z0-9._-]{1,30}$")
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.IGNORECASE)
ADDRESS = re.compile(
    r"^\d+\s+\S+.*\b(ave|avenue|st|street|rd|road|ln|lane|dr|drive|blvd|way)\b", re.IGNORECASE
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    entity_id TEXT PRIMARY KEY,
                    canonical_name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
        )


def classify(name: str) -> str:
    """The kind a name most likely is — `unknown` when the evidence does not say."""
    value = str(name or "").strip()
    if not value:
        return "unknown"
    lowered = value.casefold()
    if EMAIL.match(value):
        return "account"
    if lowered in PEOPLE:
        return "person"
    if lowered in PROJECTS or lowered.startswith(("cortex-", "temple-")):
        return "project"
    if lowered in SERVICES:
        return "service"
    if HOST.match(lowered) and any(char.isdigit() for char in lowered):
        return "device"
    if ADDRESS.match(value):
        return "place"
    return "unknown"


@dataclass
class Node:
    entity_id: str
    canonical_name: str
    kind: str
    summary: str = ""
    created_at: str = ""
    updated_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "kind": self.kind,
            "summary": self.summary,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def create(store: Any, entity_id: str, name: str = "", *, kind: str = "",
           summary: str = "", at: datetime | None = None) -> Node | None:
    """Create a node, or return the existing one. Never reclassifies by force."""
    identifier = str(entity_id or "").strip()
    if not identifier:
        return None
    _ensure(store)
    existing = get(store, identifier)
    if existing is not None:
        if summary and not existing.summary:
            return set_summary(store, identifier, summary, at=at) or existing
        return existing
    canonical = str(name or identifier).strip()
    resolved_kind = str(kind or "").strip().lower()
    if resolved_kind not in KINDS:
        resolved_kind = classify(canonical)
    stamp = (at or _now()).isoformat()
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT OR IGNORE INTO {TABLE}
                (entity_id, canonical_name, kind, summary, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
            (identifier, canonical, resolved_kind, str(summary or ""), stamp, stamp),
        )
    return get(store, identifier)


def get(store: Any, entity_id: str) -> Node | None:
    _ensure(store)
    try:
        row = store._conn.execute(
            f"SELECT * FROM {TABLE} WHERE entity_id = ?", (str(entity_id),)
        ).fetchone()
    except Exception:  # noqa: BLE001
        return None
    return Node(**dict(row)) if row else None


def set_kind(store: Any, entity_id: str, kind: str, *, at: datetime | None = None) -> Node | None:
    """A person correcting the kind outranks the lexicon."""
    resolved = str(kind or "").strip().lower()
    if resolved not in KINDS:
        return None
    _ensure(store)
    with store.transaction() as conn:
        conn.execute(
            f"UPDATE {TABLE} SET kind = ?, updated_at = ? WHERE entity_id = ?",
            (resolved, (at or _now()).isoformat(), str(entity_id)),
        )
    return get(store, entity_id)


def set_summary(store: Any, entity_id: str, summary: str, *,
                at: datetime | None = None) -> Node | None:
    _ensure(store)
    with store.transaction() as conn:
        conn.execute(
            f"UPDATE {TABLE} SET summary = ?, updated_at = ? WHERE entity_id = ?",
            (str(summary or ""), (at or _now()).isoformat(), str(entity_id)),
        )
    return get(store, entity_id)


def known_names(store: Any, *, limit: int = 2000) -> dict[str, str]:
    """Every entity name the store has ever seen, mapped to itself (or its alias target)."""
    names: dict[str, str] = {}
    try:
        from . import aliases

        for row in store._conn.execute(
            f"SELECT entity_id, alias FROM {aliases.TABLE} LIMIT ?", (int(limit),)
        ).fetchall():
            names[str(row["alias"])] = str(row["entity_id"])
    except Exception:  # noqa: BLE001
        pass
    try:
        rows = store._conn.execute(
            """SELECT entities_json FROM memories WHERE state IN ('active','cold')
               AND json_valid(entities_json) AND json_array_length(entities_json) > 0
               LIMIT ?""",
            (int(limit),),
        ).fetchall()
    except Exception:  # noqa: BLE001
        rows = []
    for row in rows:
        try:
            loaded = json.loads(str(row["entities_json"]))
        except (TypeError, ValueError):
            continue
        for name in loaded if isinstance(loaded, list) else []:
            text = str(name).strip()
            if text and text not in names:
                names[text] = text
    return names


def sync(store: Any, *, limit: int = 2000, at: datetime | None = None) -> dict[str, Any]:
    """Give every known entity a node. Idempotent: a second pass creates nothing."""
    names = known_names(store, limit=limit)
    created: dict[str, int] = {}
    for name, entity_id in names.items():
        before = get(store, entity_id)
        if before is not None:
            continue
        node = create(store, entity_id, name, at=at)
        if node is not None:
            created[node.kind] = created.get(node.kind, 0) + 1
    return {"known": len(names), "created": sum(created.values()), "by_kind": created}


def _facts(store: Any, name: str, *, limit: int = 5) -> list[dict[str, Any]]:
    try:
        rows = store._conn.execute(
            """SELECT id, content, created_at FROM memories
               WHERE state = 'active' AND (entities_json LIKE ? OR content LIKE ?)
               ORDER BY created_at DESC LIMIT ?""",
            (f'%"{name}"%', f"%{name}%", int(limit)),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    return [dict(row) for row in rows]


def _neighbours(store: Any, entity_id: str, *, limit: int = 5) -> list[str]:
    """Entities that share at least one memory with this one."""
    try:
        mine = store._conn.execute(
            """SELECT entities_json FROM memories
               WHERE state = 'active' AND entities_json LIKE ? LIMIT 40""",
            (f"%{entity_id}%",),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    counts: dict[str, int] = {}
    for row in mine:
        try:
            loaded = json.loads(str(row["entities_json"]))
        except (TypeError, ValueError):
            continue
        for name in loaded if isinstance(loaded, list) else []:
            text = str(name).strip()
            if text and text != entity_id:
                counts[text] = counts.get(text, 0) + 1
    return [name for name, _ in sorted(counts.items(), key=lambda item: -item[1])[:limit]]


def page(store: Any, entity_id: str) -> dict[str, Any] | None:
    """Everything the store knows about one entity, assembled from the tables that own it."""
    node = get(store, entity_id)
    if node is None:
        return None
    name = node.canonical_name
    try:
        from . import aliases

        names = [alias.alias for alias in aliases.aliases_for(store, node.entity_id)]
    except Exception:  # noqa: BLE001
        names = []
    facts = _facts(store, name)
    first_seen = min((str(fact.get("created_at") or "") for fact in facts), default="")
    last_seen = max((str(fact.get("created_at") or "") for fact in facts), default="")
    return {
        "entity_id": node.entity_id,
        "canonical_name": node.canonical_name,
        "kind": node.kind,
        "summary": node.summary,
        "aliases": sorted({alias for alias in names if alias}),
        "facts": facts,
        "fact_count": len(facts),
        "first_seen": first_seen,
        "last_seen": last_seen,
        "neighbours": _neighbours(store, node.entity_id),
    }


def report(store: Any, *, limit: int = 5000) -> dict[str, Any]:
    _ensure(store)
    try:
        rows = store._conn.execute(
            f"SELECT kind, COUNT(*) AS count FROM {TABLE} GROUP BY kind LIMIT ?", (int(limit),)
        ).fetchall()
    except Exception:  # noqa: BLE001
        return {"nodes": 0, "by_kind": {}}
    by_kind = {str(row["kind"]): int(row["count"]) for row in rows}
    return {"nodes": sum(by_kind.values()), "by_kind": by_kind}
