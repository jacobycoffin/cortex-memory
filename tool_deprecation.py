"""Procedures that depend on a tool's shape, and what happens when the shape moves (#138).

A procedure is only as good as the tool it was written against. `t3` 0.0.45 speaks V1 and quietly lies about a V2
store; Omarchy moved its bar from Waybar to Quickshell; a CLI renames `--json` and every recorded recipe that used it
is now a guess. Nothing in a store of procedures notices that — they keep reading as true because they were true once.

This module keeps the link explicit: a procedure names the tool **and the shape it was verified against**, and when the
tool reports a new shape the procedures behind it are *demoted*, not deleted.

Six rules:

- **invalidation demotes, never deletes.** A stale procedure is still evidence of intent; deleting it throws away the
  only record that someone once solved this.
- **invalidation is per tool.** A change in `gh` must not taint a `pytest` recipe — over-broad invalidation trains
  people to ignore the flag.
- **the previous shape is kept**, so a rollback can re-validate the procedures it restores.
- **a procedure with no recorded shape is `unpinned`, not current.** "Nobody wrote down what this was verified
  against" is a different fact from "verified against the current shape", and only one of them is safe.
- **re-verification is an explicit act** with a timestamp and the evidence that justified it; nothing becomes current
  by passing time.
- **observing the same shape twice is a no-op** — a heartbeat is not a change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

SHAPE_TABLE = "tool_shapes"
PROCEDURE_TABLE = "tool_procedures"

CURRENT = "current"
STALE = "stale"
UNPINNED = "unpinned"
STATES = (CURRENT, STALE, UNPINNED)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]] | None:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return None


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {SHAPE_TABLE} (
                    tool TEXT PRIMARY KEY,
                    shape TEXT NOT NULL,
                    previous TEXT,
                    first_seen TEXT NOT NULL,
                    changed_at TEXT NOT NULL
                )"""
        )
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {PROCEDURE_TABLE} (
                    procedure_id TEXT PRIMARY KEY,
                    tool TEXT NOT NULL,
                    shape TEXT,
                    state TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    verified_at TEXT,
                    invalidated_at TEXT
                )"""
        )


@dataclass
class Observation:
    tool: str
    shape: str
    changed: bool = False
    previous: str = ""
    invalidated: list[str] = field(default_factory=list)
    already_known: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "shape": self.shape, "changed": self.changed,
                "previous": self.previous, "invalidated": self.invalidated,
                "already_known": self.already_known}

    def render(self) -> str:
        if self.already_known:
            return f"{self.tool} {self.shape}: unchanged"
        if not self.changed:
            return f"{self.tool} {self.shape}: first observation, nothing to invalidate"
        return (f"{self.tool} {self.previous} -> {self.shape}: {len(self.invalidated)} procedure(s) demoted "
                f"to stale" + (f" ({', '.join(self.invalidated[:4])})" if self.invalidated else ""))


def observe(store: Any, tool: str, shape: str) -> Observation:
    """Record the shape a tool reports now. Demotes the procedures written against the old one."""
    name = str(tool or "").strip()
    value = str(shape or "").strip()
    if not name or not value:
        return Observation(tool=name, shape=value, changed=False)
    _ensure(store)
    existing = _rows(store, f"SELECT shape FROM {SHAPE_TABLE} WHERE tool = ?", (name,))
    stamp = _now()
    if existing:
        previous = str(existing[0]["shape"])
        if previous == value:
            with store.transaction() as conn:
                conn.execute(f"UPDATE {SHAPE_TABLE} SET changed_at = changed_at WHERE tool = ?", (name,))
            return Observation(tool=name, shape=value, already_known=True, previous=previous)
        stale = _rows(store, f"SELECT procedure_id FROM {PROCEDURE_TABLE} WHERE tool = ? AND state = ?",
                      (name, CURRENT)) or []
        ids = [str(row["procedure_id"]) for row in stale]
        with store.transaction() as conn:
            conn.execute(f"UPDATE {SHAPE_TABLE} SET shape = ?, previous = ?, changed_at = ? WHERE tool = ?",
                         (value, previous, stamp, name))
            conn.execute(
                f"UPDATE {PROCEDURE_TABLE} SET state = ?, invalidated_at = ? "
                "WHERE tool = ? AND state = ?",
                (STALE, stamp, name, CURRENT),
            )
        return Observation(tool=name, shape=value, changed=True, previous=previous, invalidated=ids)
    with store.transaction() as conn:
        conn.execute(f"INSERT INTO {SHAPE_TABLE} (tool, shape, previous, first_seen, changed_at) "
                     "VALUES (?, ?, NULL, ?, ?)", (name, value, stamp, stamp))
    return Observation(tool=name, shape=value)


def record_procedure(store: Any, procedure_id: str, tool: str, *, shape: str | None = None,
                     note: str = "") -> dict[str, Any]:
    """Register a procedure and the shape it was verified against. No shape means `unpinned`."""
    pid = str(procedure_id or "").strip()
    name = str(tool or "").strip()
    if not pid or not name:
        return {"ok": False, "reason": "a procedure id and a tool are required"}
    _ensure(store)
    known = _rows(store, f"SELECT shape FROM {SHAPE_TABLE} WHERE tool = ?", (name,))
    current = str(known[0]["shape"]) if known else None
    pinned = str(shape).strip() if shape else None
    if pinned is None:
        state = UNPINNED
    elif current is not None and pinned != current:
        state = STALE
    else:
        state = CURRENT
    stamp = _now() if state == CURRENT else None
    with store.transaction() as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO {PROCEDURE_TABLE} "
            "(procedure_id, tool, shape, state, note, verified_at, invalidated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (pid, name, pinned, state, str(note or ""), stamp, None if state == CURRENT else _now()),
        )
    return {"ok": True, "procedure_id": pid, "tool": name, "shape": pinned, "state": state}


def revalidate(store: Any, procedure_id: str, *, evidence: str = "") -> dict[str, Any]:
    """Re-verify a procedure against the tool's current shape. Evidence is required to mean anything."""
    pid = str(procedure_id or "").strip()
    rows = _rows(store, f"SELECT tool, state FROM {PROCEDURE_TABLE} WHERE procedure_id = ?", (pid,))
    if not rows:
        return {"ok": False, "reason": f"no procedure {pid}"}
    tool = str(rows[0]["tool"])
    known = _rows(store, f"SELECT shape FROM {SHAPE_TABLE} WHERE tool = ?", (tool,))
    if not known:
        return {"ok": False, "reason": f"no observed shape for {tool}: re-verify against something first"}
    if not str(evidence or "").strip():
        return {"ok": False, "reason": "re-verification needs evidence: what did you run, and what did it say"}
    stamp = _now()
    with store.transaction() as conn:
        conn.execute(
            f"UPDATE {PROCEDURE_TABLE} SET state = ?, shape = ?, verified_at = ?, invalidated_at = NULL, "
            "note = ? WHERE procedure_id = ?",
            (CURRENT, str(known[0]["shape"]), stamp, str(evidence), pid),
        )
    return {"ok": True, "procedure_id": pid, "tool": tool, "shape": str(known[0]["shape"]), "state": CURRENT}


@dataclass
class Status:
    readable: bool = True
    reason: str = ""
    tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"readable": self.readable, "reason": self.reason, "tools": self.tools, "counts": self.counts}

    def render(self) -> list[str]:
        if not self.readable:
            return [self.reason]
        if not self.tools:
            return ["No tool shape has been observed yet."]
        lines = [f"{len(self.tools)} tool(s): " + ", ".join(f"{state} {count}" for state, count in
                                                            sorted(self.counts.items()))]
        for name, info in sorted(self.tools.items()):
            shape = info.get("shape") or "(unknown)"
            counts = ", ".join(f"{state} {count}" for state, count in sorted(info["procedures"].items()))
            lines.append(f"- {name} @ {shape}: {counts or 'no procedures'}")
        return lines


def _tables_exist(store: Any) -> bool | None:
    """True/False when the store answered, None when it could not be asked at all."""
    rows = _rows(store, "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (?, ?)",
                 (SHAPE_TABLE, PROCEDURE_TABLE))
    if rows is None:
        return None
    return len(rows) == 2


def status(store: Any) -> Status:
    # Reading never creates schema: a store that has never seen a tool has no tables, and that is "nothing
    # observed", not "the store is broken" — the two must not look the same.
    exists = _tables_exist(store)
    if exists is None:
        return Status(readable=False, reason="the tool tables could not be read")
    if not exists:
        return Status()
    shapes = _rows(store, f"SELECT tool, shape, previous, changed_at FROM {SHAPE_TABLE}")
    procedures = _rows(store, f"SELECT tool, state FROM {PROCEDURE_TABLE}")
    if shapes is None or procedures is None:
        return Status(readable=False, reason="the tool tables could not be read")
    out = Status()
    for row in shapes:
        out.tools[str(row["tool"])] = {"shape": str(row["shape"]), "previous": row["previous"],
                                       "changed_at": row["changed_at"], "procedures": {}}
    for row in procedures:
        tool = str(row["tool"])
        info = out.tools.setdefault(tool, {"shape": "", "previous": None, "changed_at": "",
                                           "procedures": {}})
        state = str(row["state"])
        info["procedures"][state] = info["procedures"].get(state, 0) + 1
        out.counts[state] = out.counts.get(state, 0) + 1
    return out
