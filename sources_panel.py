"""Where a memory came from, and whether anybody ever used it (#175).

Every panel that shows a memory can show its text; this one shows the **sources**: which path wrote it,
what evidence its edges carry, when it was retrieved, selected, injected and used, what superseded it,
and what was refused or redacted on the way in. It exists because "why is this in my prompt" and "why do
I not trust it" are questions about provenance, not about content.

Five rules, each the way a provenance panel misleads:

- **a missing field is `unknown`, never zero and never blank.** `retrieved_count = 0` and "this column is
  absent" are different claims, and a panel that prints `0` for both turns *not measured* into *measured
  as never used*. Every value carries the column it came from, so a reader can check the claim.
- **the panel is bounded and says what it cut.** A memory with 400 edges does not get a 400-line panel;
  it gets the strongest ten and the number left.
- **it says what it cannot know.** The model's reasoning for choosing this memory, whether the original
  source is still true, and what a document looked like before extraction are not in this store. Stating
  the boundary is the difference between a provenance panel and a story.
- **a redacted value is reported as `redacted`, never echoed.** If the write path redacted content, the
  panel says so and names the pattern, not the secret. A source panel is exactly where a leaked key
  would be quoted back at a reader.
- **unreadable is its own answer.** A store that cannot be queried returns `readable: False` with no
  counts at all, rather than an empty panel that reads as "this memory has no sources".

Read-only: every statement is a SELECT, and a test snapshots the schema and row counts to prove it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

MEMORY_TABLE = "memories"
EDGE_TABLE = "edges"
USAGE_TABLE = "usage_records"
MEMBERSHIP_TABLE = "memory_recall_memberships"

# Provenance columns, as (label, column). Named explicitly so the panel can say "unknown" per field and a
# reader can find each one in `PRAGMA table_info(memories)` rather than trusting a blob.
PROVENANCE: tuple[tuple[str, str], ...] = (
    ("source type", "source_type"),
    ("source category", "source_category"),
    ("source ref", "source_ref"),
    ("session", "session_id"),
    ("extraction method", "extraction_method"),
    ("record role", "record_role"),
    ("approval state", "approval_state"),
    ("origin category", "origin_source_category"),
)

# Counters the harness writes as it recalls. `unknown` when the column is absent: a zero here would be a
# measurement, and an absent column is not one.
COUNTERS: tuple[tuple[str, str], ...] = (
    ("retrieved", "retrieved_count"),
    ("selected", "selected_count"),
    ("injected", "injected_count"),
    ("used", "used_count"),
    ("confirmed", "confirmed_count"),
    ("corrections", "correction_count"),
)

STAMPS: tuple[tuple[str, str], ...] = (
    ("last retrieved", "last_retrieved_at"),
    ("last injected", "last_injected_at"),
    ("last used", "last_used_at"),
)

MAX_EDGES = 10
MAX_USAGE = 5
MAX_CHAIN = 8


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]] | None:
    """Rows, or None when the store cannot answer. The caller must not read None as 'nothing found'."""
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return None


def _columns(store: Any, table: str) -> set[str] | None:
    rows = _rows(store, f"PRAGMA table_info({table})")
    if rows is None:
        return None
    return {str(row.get("name")) for row in rows}


UNKNOWN = "unknown"


def _value(row: dict[str, Any], column: str) -> Any:
    if column not in row:
        return UNKNOWN
    value = row.get(column)
    if value is None or (isinstance(value, str) and not value.strip()):
        return UNKNOWN
    return value


@dataclass
class Field:
    label: str
    column: str
    value: Any

    @property
    def known(self) -> bool:
        return self.value != UNKNOWN

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "column": self.column, "value": self.value, "known": self.known}


@dataclass
class Panel:
    memory_id: str = ""
    found: bool = False
    readable: bool = True
    reason: str = ""
    provenance: list[Field] = field(default_factory=list)
    counters: list[Field] = field(default_factory=list)
    stamps: list[Field] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    edges_total: int = 0
    usage: list[dict[str, Any]] = field(default_factory=list)
    usage_total: int = 0
    chain: list[dict[str, Any]] = field(default_factory=list)
    chain_total: int = 0
    flags: list[str] = field(default_factory=list)
    cannot_know: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "found": self.found,
            "readable": self.readable,
            "reason": self.reason,
            "provenance": [f.as_dict() for f in self.provenance],
            "counters": [f.as_dict() for f in self.counters],
            "stamps": [f.as_dict() for f in self.stamps],
            "edges": self.edges,
            "edges_total": self.edges_total,
            "usage": self.usage,
            "usage_total": self.usage_total,
            "chain": self.chain,
            "chain_total": self.chain_total,
            "flags": self.flags,
            "cannot_know": self.cannot_know,
        }


CANNOT_KNOW = (
    "why the model chose this memory - only what was selected and used is recorded",
    "whether the original source is still true - the store records what was extracted, not the world",
    "what a document looked like before extraction ran",
)


# The write path leaves a marker where it removed a secret. The marker names the *pattern*, which is
# what a reader needs to know how the memory was altered, and never the value.
REDACTION_MARKER = re.compile(r"\[REDACTED:([A-Za-z0-9_\-]{1,40})\]")


def _redaction_flags(columns: set[str], row: dict[str, Any]) -> list[str]:
    """What the write path refused or clipped, named without the value."""
    flags: list[str] = []
    for pattern in sorted(set(REDACTION_MARKER.findall(str(row.get("content") or "")))):
        flags.append(f"redacted at write time: {pattern} pattern removed from the content")
    if "quarantine_reason" in columns and _value(row, "quarantine_reason") != UNKNOWN:
        flags.append(f"quarantined: {_value(row, 'quarantine_reason')}")
    if "redaction_reason" in columns and _value(row, "redaction_reason") != UNKNOWN:
        flags.append(f"redacted: {_value(row, 'redaction_reason')}")
    if "stranded" in columns and int(row.get("stranded") or 0):
        flags.append(f"stranded: {row.get('stranded_reason') or 'no reason recorded'}")
    if "protected" in columns and int(row.get("protected") or 0):
        flags.append("protected: automated paths must not change it")
    return flags


def payload(store: Any, memory_id: str) -> Panel:
    """The sources behind one memory. Never raises: an unreadable store is a panel that says so."""
    panel = Panel(memory_id=str(memory_id or ""), cannot_know=list(CANNOT_KNOW))
    columns = _columns(store, MEMORY_TABLE)
    if columns is None:
        panel.readable = False
        panel.reason = "the store could not be read: memories could not be inspected"
        return panel
    rows = _rows(store, f"SELECT * FROM {MEMORY_TABLE} WHERE id = ?", (str(memory_id),))
    if rows is None:
        panel.readable = False
        panel.reason = "the store could not be read: the row query failed"
        return panel
    if not rows:
        panel.reason = f"unknown memory: no row in {MEMORY_TABLE} with that id"
        return panel
    row = rows[0]
    panel.found = True

    panel.provenance = [Field(label, column, _value(row, column)) for label, column in PROVENANCE]
    panel.counters = [Field(label, column, _value(row, column)) for label, column in COUNTERS]
    panel.stamps = [Field(label, column, _value(row, column)) for label, column in STAMPS]
    panel.flags = _redaction_flags(columns, row)

    # Edges: strongest first, bounded, with the number left behind stated.
    edge_rows = _rows(
        store,
        f"SELECT src_id, dst_id, relation, weight FROM {EDGE_TABLE} "
        "WHERE src_id = ? OR dst_id = ? ORDER BY weight DESC",
        (str(memory_id), str(memory_id)),
    )
    if edge_rows is None:
        panel.flags.append("edges could not be read")
    else:
        panel.edges_total = len(edge_rows)
        panel.edges = [
            {"relation": entry.get("relation"), "weight": entry.get("weight"),
             "other": entry.get("dst_id") if entry.get("src_id") == str(memory_id) else entry.get("src_id"),
             "direction": "out" if entry.get("src_id") == str(memory_id) else "in"}
            for entry in edge_rows[:MAX_EDGES]
        ]

    # Usage: the record of it being recalled. Absent table is "not measured", not "never used".
    usage_rows = _rows(
        store,
        f"SELECT created_at, selected, used, attribution FROM {USAGE_TABLE} "
        "WHERE memory_id = ? ORDER BY created_at DESC",
        (str(memory_id),),
    )
    if usage_rows is None:
        panel.flags.append(f"{USAGE_TABLE} could not be read, so use is unknown")
    else:
        panel.usage_total = len(usage_rows)
        panel.usage = usage_rows[:MAX_USAGE]

    # Supersession chain: what this replaced and what replaced it, bounded.
    chain_rows = _rows(
        store,
        f"SELECT id, created_at, content FROM {MEMORY_TABLE} "
        "WHERE id IN (?, ?) OR supersedes_id = ? ORDER BY created_at",
        (str(memory_id), str(row.get("supersedes_id") or ""), str(memory_id)),
    )
    if chain_rows:
        panel.chain_total = len(chain_rows)
        panel.chain = [
            {"id": entry.get("id"), "when": entry.get("created_at"),
             "excerpt": " ".join(str(entry.get("content") or "").split())[:80]}
            for entry in chain_rows[:MAX_CHAIN]
        ]
    return panel


def render(panel: Panel, *, limit: int = 12) -> list[str]:
    """The panel as lines. A field that is unknown prints `unknown`, and says which column it wanted."""
    if not panel.readable:
        return [panel.reason]
    if not panel.found:
        return [panel.reason]
    lines: list[str] = [f"sources for {panel.memory_id[:8]}"]
    known = [f for f in panel.provenance if f.known]
    unknown = [f for f in panel.provenance if not f.known]
    for item in known[:limit]:
        lines.append(f"  {item.label}: {item.value}  ({MEMORY_TABLE}.{item.column})")
    for item in unknown[: max(0, limit - len(known))]:
        lines.append(f"  {item.label}: unknown  ({MEMORY_TABLE}.{item.column} not recorded)")
    counters = ", ".join(f"{f.label} {f.value}" for f in panel.counters[:6])
    lines.append(f"  {counters}  ({MEMORY_TABLE})")
    if panel.edges_total:
        shown = len(panel.edges)
        lines.append(f"  edges {shown} of {panel.edges_total}  ({EDGE_TABLE})")
    if panel.usage_total:
        lines.append(f"  usage records {len(panel.usage)} of {panel.usage_total}  ({USAGE_TABLE})")
    elif panel.usage_total == 0 and not any("could not be read" in flag for flag in panel.flags):
        lines.append(f"  usage records: none for this memory  ({USAGE_TABLE})")
    for flag in panel.flags:
        lines.append(f"  flag: {flag}")
    if panel.cannot_know:
        lines.append(f"  cannot know: {panel.cannot_know[0]}")
    return lines
