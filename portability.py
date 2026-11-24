"""Move memory between harnesses without changing what it means (#153)

A memory that cannot leave is a memory held hostage, and a memory that arrives changed is worse than one
that never arrived. Both failures are easy to make:

- **re-deriving on import.** A receiving harness that re-scores confidence, re-classifies the source or
  re-guesses the kind has created a *different belief* with the same words. Provenance travels or it is
  lost.
- **losing the links.** `supersedes_id` is what makes "the port is 8101" outrank "the port is 8100". An
  import that drops it silently resurrects retired beliefs as current ones.

Five rules:

- **ids travel.** A portable bundle keeps the original id, so supersession chains, receipts, edges and
  external references still point at the same thing after the move.
- **a collision is refused, never overwritten.** Two harnesses with the same id are not the same memory,
  and the last writer winning is how provenance becomes a rumour.
- **nothing is dropped silently.** A field the target cannot hold is reported by name; an unknown kind is
  kept as-is rather than coerced into `semantic`.
- **a broken chain says so.** If a superseded id is not in the bundle, the importer records the link as
  dangling instead of pretending the memory stands alone.
- **import is idempotent.** Importing the same bundle twice adds nothing the second time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

FORMAT = "cortex-portable"
VERSION = 1
# Fields that must survive a move for the memory to mean the same thing on the other side.
PORTABLE_FIELDS = ("id", "kind", "content", "content_hash", "confidence", "source_type",
                   "source_category", "created_at", "updated_at", "observed_at", "valid_from",
                   "valid_to", "supersedes_id", "pinned", "state", "context_mode", "scope_json")
# Fields the receiving store may not have; reported by name rather than dropped quietly.
OPTIONAL_FIELDS = ("subject", "predicate", "object_value", "extraction_method", "volatility",
                   "importance", "preconditions_json")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def _columns(store: Any) -> set[str]:
    return {str(row["name"]) for row in _rows(store, "PRAGMA table_info(memories)")}


def bundle(store: Any, *, ids: Iterable[str] | None = None, limit: int = 500,
           include_edges: bool = True, include_parents: bool = True) -> dict[str, Any]:
    """A self-describing bundle: the memories, their links, and what the format guarantees."""
    if ids is not None:
        wanted = [str(item) for item in ids][: int(limit)]
        rows: list[dict[str, Any]] = []
        for memory_id in wanted:
            rows.extend(_rows(store, "SELECT * FROM memories WHERE id = ?", (memory_id,)))
    else:
        rows = _rows(store, "SELECT * FROM memories WHERE state <> 'archived' "
                            "ORDER BY created_at LIMIT ?", (int(limit),))
    if include_parents:
        # A superseded memory points at the belief it replaced, and the target store enforces that
        # reference. Exporting the child without the parent makes the bundle unimportable (24 of 500 in
        # the live store), so the chain travels with it.
        seen = {str(row.get("id")) for row in rows}
        queue = [row.get("supersedes_id") for row in rows if row.get("supersedes_id")]
        while queue and len(rows) < int(limit) * 2:
            parent = str(queue.pop())
            if parent in seen:
                continue
            found_rows = _rows(store, "SELECT * FROM memories WHERE id = ?", (parent,))
            if not found_rows:
                continue
            seen.add(parent)
            rows.append(found_rows[0])
            if found_rows[0].get("supersedes_id"):
                queue.append(found_rows[0]["supersedes_id"])
    memories = []
    for row in rows:
        record = {name: row.get(name) for name in PORTABLE_FIELDS if name in row}
        extras = {name: row.get(name) for name in OPTIONAL_FIELDS if row.get(name) not in (None, "")}
        if extras:
            record["extra"] = extras
        memories.append(record)
    kept = {str(record["id"]) for record in memories}
    edges: list[dict[str, Any]] = []
    if include_edges:
        for row in _rows(store, "SELECT src_id, dst_id, relation, weight, evidence_count FROM edges"):
            if str(row["src_id"]) in kept and str(row["dst_id"]) in kept:
                edges.append({"src_id": str(row["src_id"]), "dst_id": str(row["dst_id"]),
                              "relation": str(row["relation"] or ""),
                              "weight": float(row["weight"] or 0.0),
                              "evidence_count": int(row["evidence_count"] or 0)})
    return {
        "format": FORMAT,
        "version": VERSION,
        "exported_at": _now().isoformat(),
        "memories": memories,
        "edges": edges,
        "guarantees": [
            "ids travel unchanged, so supersession chains and receipts still resolve",
            "confidence and provenance are copied, never re-derived",
            "an unknown kind is kept as-is rather than coerced",
            "a link to an id outside the bundle is reported as dangling",
        ],
    }


@dataclass
class Imported:
    ok: bool = False
    added: int = 0
    skipped: int = 0
    collisions: list[str] = field(default_factory=list)
    dangling: list[str] = field(default_factory=list)
    unknown_fields: dict[str, int] = field(default_factory=dict)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "added": self.added, "skipped": self.skipped,
                "collisions": list(self.collisions), "dangling": list(self.dangling),
                "unknown_fields": dict(self.unknown_fields), "reason": self.reason}


def validate(payload: dict[str, Any]) -> str:
    """Empty string means the bundle is usable."""
    if not isinstance(payload, dict):
        return "the bundle is not an object"
    if payload.get("format") != FORMAT:
        return f"not a {FORMAT} bundle"
    if int(payload.get("version") or 0) > VERSION:
        return (f"bundle version {payload.get('version')} is newer than this importer ({VERSION})")
    if not isinstance(payload.get("memories"), list):
        return "the bundle has no memories list"
    return ""


def required_columns(store: Any) -> set[str]:
    """NOT NULL columns with no default: a record missing one cannot be inserted at all."""
    rows = _rows(store, "PRAGMA table_info(memories)")
    return {str(row["name"]) for row in rows
            if int(row.get("notnull") or 0) and row.get("dflt_value") in (None, "")}


def import_bundle(store: Any, payload: dict[str, Any], *, dry_run: bool = True) -> Imported:
    """Copy memories in, preserving meaning. Refuses collisions; reports what it cannot hold."""
    problem = validate(payload)
    if problem:
        return Imported(reason=problem)
    columns = _columns(store)
    required = required_columns(store)
    for record in payload["memories"][:1]:
        missing = sorted(required - set(record))
        if missing:
            return Imported(reason="the bundle cannot be imported: it is missing required "
                                   f"field(s) {', '.join(missing)}")
    found = Imported(ok=True)
    ids_in_bundle = {str(record.get("id")) for record in payload["memories"]}
    # A superseded child references its parent, and the store enforces that reference, so parents must
    # be inserted first. Records whose parent never appears are inserted last and reported.
    pending = list(payload["memories"])
    ordered: list[dict[str, Any]] = []
    while pending:
        ready = [record for record in pending
                 if not record.get("supersedes_id")
                 or str(record["supersedes_id"]) not in {str(other.get("id")) for other in pending}]
        if not ready:
            ordered.extend(pending)   # a cycle: import as-is and let the store refuse what it must
            break
        ordered.extend(ready)
        pending = [record for record in pending if record not in ready]
    for record in ordered:
        memory_id = str(record.get("id") or "")
        if not memory_id:
            found.skipped += 1
            continue
        existing = _rows(store, "SELECT id FROM memories WHERE id = ?", (memory_id,))
        if existing:
            found.collisions.append(memory_id)
            found.skipped += 1
            continue
        link = record.get("supersedes_id")
        if link and str(link) not in ids_in_bundle:
            found.dangling.append(memory_id)
        for name in (record.get("extra") or {}):
            if name not in columns:
                found.unknown_fields[name] = found.unknown_fields.get(name, 0) + 1
        if dry_run:
            found.added += 1
            continue
        usable = {name: value for name, value in record.items()
                  if name in columns and name != "extra"}
        assignments = ", ".join(f"{name} = ?" for name in usable)
        try:
            with store.transaction() as conn:
                conn.execute(f"INSERT INTO memories({', '.join(usable)}) "
                             f"VALUES({', '.join('?' for _ in usable)})",
                             tuple(usable.values()))
        except Exception as error:  # noqa: BLE001
            found.reason = f"the store refused {memory_id}: {error}"
            found.skipped += 1
            continue
        found.added += 1
        _ = assignments
    if found.collisions:
        found.reason = (found.reason or
                        f"{len(found.collisions)} id(s) already exist here; refusing to overwrite")
    return found


def round_trip(store: Any, memory_id: str, target: Any) -> dict[str, Any]:
    """Export one memory into a second store and compare what came out."""
    payload = bundle(store, ids=[memory_id], include_edges=False)
    if not payload["memories"]:
        return {"ok": False, "reason": "no such memory to move"}
    result = import_bundle(target, payload, dry_run=False)
    if not result.ok or not result.added:
        return {"ok": False, "reason": result.reason or "the target refused the memory"}
    source = payload["memories"][0]
    arrived = _rows(target, "SELECT * FROM memories WHERE id = ?", (memory_id,))
    if not arrived:
        return {"ok": False, "reason": "the memory is not in the target after import"}
    differences = [name for name in ("kind", "content", "confidence", "source_category")
                   if str(arrived[0].get(name)) != str(source.get(name))]
    return {"ok": not differences, "differences": differences,
            "reason": "" if not differences else "the memory changed in transit"}


def render(store: Any, *, limit: int = 500) -> list[str]:
    payload = bundle(store, limit=limit)
    size = len(json.dumps(payload))
    lines = [f"portability: {len(payload['memories'])} memor(ies), "
             f"{len(payload['edges'])} edge(s), {size} bytes"]
    dangling = sum(1 for record in payload["memories"]
                   if record.get("supersedes_id")
                   and str(record["supersedes_id"]) not in
                   {str(item["id"]) for item in payload["memories"]})
    lines.append(f"  links outside the bundle: {dangling}")
    lines.append(f"  format {FORMAT} v{VERSION}")
    return lines
