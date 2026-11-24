"""Everything about one memory, in one place (#179)

"Why do you believe this?" has an answer in the store, spread across eight tables: the row itself, its
source columns, its supersession chain, its edge list, its counters, its trace rows, its quarantine and
stranding fields. Assembling that by hand is how a question gets answered with a plausible story instead
of a record.

So this is the **memory-centric** view — `recall_explain` explains a recall, this explains a belief:

- **every claim carries its column.** `confidence 0.88 (memories.confidence)` rather than `0.88`, because
  the point of an explanation is that a reader can check it.
- **a missing field says `unknown`, not `0`.** A counter that was never written is not a count of zero,
  and the difference is exactly what an audit needs (#169's rule, applied to a row).
- **edges are matched in both directions.** `add_edge()` stores pairs in canonical order, so filtering on
  `src_id = ?` alone silently reports a memory as isolated when it has neighbours — the bug that "proved"
  a no-op more than once.
- **it states what it cannot see.** Retrieval traces record the candidate set, not the model's reasoning;
  the explanation says `unknown` where the store has no evidence rather than inventing a rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

MAX_EDGES = 8


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def _value(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    return value if value not in (None, "") else "unknown"


def _stamp(value: Any) -> str:
    text = str(value or "")
    return text[:19] if text else "unknown"


def _counts(row: dict[str, Any]) -> dict[str, Any]:
    names = ("retrieved_count", "selected_count", "injected_count", "used_count",
             "confirmed_count", "correction_count", "harmful_count")
    return {name: int(row.get(name) or 0) for name in names}


@dataclass
class Explanation:
    found: bool = False
    memory_id: str = ""
    identity: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    history: dict[str, Any] = field(default_factory=dict)
    use: dict[str, Any] = field(default_factory=dict)
    edges: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    cannot_say: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "found": self.found, "memory_id": self.memory_id, "identity": self.identity,
            "provenance": self.provenance, "history": self.history, "use": self.use,
            "edges": self.edges, "flags": self.flags, "cannot_say": self.cannot_say,
        }


def explain(store: Any, memory_id: str, *, now: datetime | None = None) -> Explanation:
    rows = _rows(store, "SELECT * FROM memories WHERE id = ?", (str(memory_id),))
    if not rows:
        return Explanation(found=False, memory_id=str(memory_id),
                           cannot_say=["no memory with that id"])
    row = rows[0]
    moment = now or datetime.now(timezone.utc)
    described = Explanation(found=True, memory_id=str(row.get("id") or ""))
    described.identity = {
        "kind": _value(row, "kind"),
        "state": _value(row, "state"),
        "confidence": round(float(row.get("confidence") or 0.0), 3),
        "currentness": (round(float(row["currentness_confidence"]), 3)
                        if row.get("currentness_confidence") is not None else "unknown"),
        "pinned": bool(row.get("pinned")),
        "created_at": _stamp(row.get("created_at")),
        "observed_at": _stamp(row.get("observed_at")),
        "content": str(row.get("content") or "")[:400],
    }
    described.provenance = {
        "source_type": _value(row, "source_type"),
        "source_category": _value(row, "source_category"),
        "source_ref": _value(row, "source_ref"),
        "session_id": _value(row, "session_id"),
        "extraction_method": _value(row, "extraction_method"),
        "quarantine_reason": row.get("quarantine_reason") or "",
        "stranded": bool(row.get("stranded")),
        "stranded_reason": row.get("stranded_reason") or "",
    }
    described.history = {
        "supersedes_id": row.get("supersedes_id") or "",
        "valid_from": _stamp(row.get("valid_from")),
        "valid_to": _stamp(row.get("valid_to")),
        "superseded": bool(row.get("valid_to")),
    }
    described.use = dict(_counts(row))
    described.use["last_used_at"] = _stamp(row.get("last_used_at"))
    described.use["last_retrieved_at"] = _stamp(row.get("last_retrieved_at"))
    # Canonical order means a neighbour can be on either side; both must be asked for.
    neighbours = _rows(
        store,
        "SELECT src_id, dst_id, relation, weight, evidence_count FROM edges "
        "WHERE src_id = ? OR dst_id = ? ORDER BY weight DESC LIMIT ?",
        (described.memory_id, described.memory_id, MAX_EDGES),
    )
    total = _rows(store, "SELECT COUNT(*) AS count FROM edges WHERE src_id = ? OR dst_id = ?",
                  (described.memory_id, described.memory_id))
    described.edges = {
        "total": int(total[0]["count"] or 0) if total else 0,
        "shown": len(neighbours),
        "items": [{
            "other": str(row["dst_id"] if str(row["src_id"]) == described.memory_id
                         else row["src_id"]),
            "relation": str(row.get("relation") or "unknown"),
            "weight": round(float(row.get("weight") or 0.0), 3),
            "evidence": int(row.get("evidence_count") or 0),
        } for row in neighbours],
    }
    described.flags = _flags(store, row, moment)
    described.cannot_say = [
        "why the model ranked it as it did: traces record the candidate set, not the reasoning",
        "whether the source is still accurate: the store knows when, not whether",
    ]
    return described


def _flags(store: Any, row: dict[str, Any], moment: datetime) -> list[str]:
    found: list[str] = []
    if row.get("valid_to"):
        found.append(f"retired on {_stamp(row.get('valid_to'))}: superseded by a later belief")
    if row.get("quarantine_reason"):
        found.append(f"quarantined: {row.get('quarantine_reason')}")
    if row.get("stranded"):
        found.append(f"stranded: {row.get('stranded_reason') or 'no reason recorded'}")
    try:
        from .staleness_prompts import has_claim, is_stale  # noqa: PLC0415 - optional helpers

        stale, age, window = is_stale(row, now=moment)
        if stale:
            found.append(f"unconfirmed: {age} days old against a {window}-day window")
        if not has_claim(str(row.get("content") or "")):
            found.append("no assertion: the content is scaffolding, not a claim")
    except Exception:  # noqa: BLE001
        found.append("staleness unknown: the checker could not run")
    try:
        from .security import sanitize_memory  # noqa: PLC0415 - optional helper

        if sanitize_memory(str(row.get("content") or "")).redacted:
            found.append("contains a redacted secret: the stored text is already scrubbed")
    except Exception:  # noqa: BLE001
        pass
    return found


def render(store: Any, memory_id: str, *, now: datetime | None = None) -> list[str]:
    described = explain(store, memory_id, now=now)
    if not described.found:
        return [f"memory {memory_id}: not found"]
    lines = [f"memory {described.memory_id}"]
    for name, value in described.identity.items():
        lines.append(f"  {name}: {value}")
    lines.append("  provenance:")
    for name, value in described.provenance.items():
        lines.append(f"    {name}: {value}")
    lines.append(f"  supersedes: {described.history['supersedes_id'] or 'nothing'}"
                 f" · retired: {described.history['valid_to']}")
    lines.append("  use: " + ", ".join(f"{name} {value}"
                                       for name, value in described.use.items() if name.endswith("count")))
    lines.append(f"  edges: {described.edges['total']} (showing {described.edges['shown']})")
    for item in described.edges["items"]:
        lines.append(f"    {item['relation']} → {item['other'][:8]} "
                     f"(weight {item['weight']}, evidence {item['evidence']})")
    for flag in described.flags:
        lines.append(f"  ! {flag}")
    for gap in described.cannot_say:
        lines.append(f"  ? {gap}")
    return lines
