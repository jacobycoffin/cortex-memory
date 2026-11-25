"""What the next session must know before running a procedure (#143).

A procedure that worked yesterday can fail today for a reason nobody wrote down: the machine suspends when its owner
walks away, a PR shows no checks while it is conflicted, an inhibit call needs interactive authentication over SSH.
The next session re-learns each of those by hitting it. A handoff note is the cheap fix — provided it is short,
typed, and tied to the thing that can invalidate it.

Five rules:

- **a note has a type**, because the four kinds are read differently: `precondition` (check before starting),
  `gotcha` (this looks like a failure and is not), `env` (this host differs), `last_failure` (what broke last time).
  An untyped note is a sentence nobody can act on.
- **a note is bound to the tool shape it was written against** (the same `tool` + `shape` pair as `#138`), so a shape
  change marks its notes stale instead of letting them mislead quietly.
- **the newest failure outranks old advice.** A `last_failure` recorded today beats a `precondition` from March when
  the handoff is cut to fit a screen.
- **the handoff is bounded and says what it left out.** Five lines is a handoff; fifty is a wiki.
- **notes are superseded, never deleted** — the history is how you learn that a gotcha was fixed.

Read-only rendering: `handoff()` returns lines, it does not write.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

NOTE_TABLE = "procedure_handoff_notes"

PRECONDITION = "precondition"
GOTCHA = "gotcha"
ENV = "env"
LAST_FAILURE = "last_failure"
KINDS = (LAST_FAILURE, PRECONDITION, GOTCHA, ENV)

# When the handoff is cut, what survives: a fresh failure first, then what to check, then what misleads, then context.
PRIORITY = {LAST_FAILURE: 0, PRECONDITION: 1, GOTCHA: 2, ENV: 3}

# One note per kind, because a new note of the same kind supersedes the old one — so a handoff is at most four
# lines by construction, not by truncation. A limit below that still states what it left out.
MAX_LINES = len(KINDS)


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
            f"""CREATE TABLE IF NOT EXISTS {NOTE_TABLE} (
                    note_id TEXT PRIMARY KEY,
                    procedure_id TEXT NOT NULL,
                    tool TEXT NOT NULL DEFAULT '',
                    shape TEXT,
                    kind TEXT NOT NULL,
                    body TEXT NOT NULL,
                    written_at TEXT NOT NULL,
                    superseded_at TEXT
                )"""
        )


@dataclass
class Note:
    note_id: str
    procedure_id: str
    kind: str
    body: str
    written_at: str
    tool: str = ""
    shape: str | None = None
    superseded: bool = False
    stale: bool = False

    def render(self) -> str:
        # "superseded" is history (a newer note replaced it); "stale" is a warning (the tool moved under it). They
        # are different facts and a reader must not have to guess which one applies.
        mark = " (superseded)" if self.superseded else (" (stale: written against an older shape)" if self.stale else "")
        return f"[{self.kind}] {self.body}{mark}"

    def as_dict(self) -> dict[str, Any]:
        return {"note_id": self.note_id, "procedure_id": self.procedure_id, "kind": self.kind,
                "body": self.body, "written_at": self.written_at, "tool": self.tool,
                "shape": self.shape, "superseded": self.superseded, "stale": self.stale}


def note(store: Any, procedure_id: str, kind: str, body: str, *, tool: str = "",
         shape: str | None = None, note_id: str | None = None) -> dict[str, Any]:
    """Write a handoff note. Typed, bound to a shape when one is given, and never a bare sentence."""
    pid = str(procedure_id or "").strip()
    text = " ".join(str(body or "").split())
    if not pid:
        return {"ok": False, "reason": "a procedure id is required"}
    if kind not in KINDS:
        return {"ok": False, "reason": f"kind must be one of {', '.join(KINDS)}: an untyped note cannot be acted on"}
    if len(text) < 12:
        return {"ok": False, "reason": "a note must say something checkable; 12 characters is the floor"}
    _ensure(store)
    stamp = _now()
    nid = note_id or f"hn-{pid}-{abs(hash((pid, kind, text))) % 10**10}"
    # A newer note of the same kind supersedes the older one rather than piling up: a precondition list that keeps
    # every historical version is a list nobody reads.
    with store.transaction() as conn:
        conn.execute(f"UPDATE {NOTE_TABLE} SET superseded_at = ? WHERE procedure_id = ? AND kind = ? "
                     "AND superseded_at IS NULL", (stamp, pid, kind))
        conn.execute(
            f"INSERT OR REPLACE INTO {NOTE_TABLE} (note_id, procedure_id, tool, shape, kind, body, written_at, "
            "superseded_at) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
            (nid, pid, str(tool or ""), shape, kind, text, stamp),
        )
    return {"ok": True, "note_id": nid, "procedure_id": pid, "kind": kind, "body": text}


def notes(store: Any, procedure_id: str, *, include_superseded: bool = False) -> list[Note]:
    if _rows(store, "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (NOTE_TABLE,)) is None:
        return []
    sql = f"SELECT * FROM {NOTE_TABLE} WHERE procedure_id = ?"
    if not include_superseded:
        sql += " AND superseded_at IS NULL"
    rows = _rows(store, sql + " ORDER BY written_at DESC", (str(procedure_id or "").strip(),))
    return [
        Note(note_id=str(row["note_id"]), procedure_id=str(row["procedure_id"]), kind=str(row["kind"]),
             body=str(row["body"]), written_at=str(row["written_at"]), tool=str(row["tool"]),
             shape=row["shape"], superseded=bool(row["superseded_at"]))
        for row in rows or []
    ]


def stale_notes(store: Any, procedure_id: str, current_shapes: dict[str, str]) -> list[Note]:
    """Notes written against a tool shape that is no longer current. Empty when the shape still matches."""
    out: list[Note] = []
    for item in notes(store, procedure_id):
        if not item.tool:
            continue
        current = current_shapes.get(item.tool)
        if current is not None and item.shape and item.shape != current:
            out.append(item)
    return out


def handoff(store: Any, procedure_id: str, *, limit: int = MAX_LINES,
            current_shapes: dict[str, str] | None = None) -> list[str]:
    """The bounded handoff: newest failure first, stale notes marked, and the cut stated."""
    items = notes(store, procedure_id)
    if current_shapes:
        stale_ids = {item.note_id for item in stale_notes(store, procedure_id, current_shapes)}
        for item in items:
            item.stale = item.note_id in stale_ids
    if not items:
        return [f"No handoff notes for {procedure_id}."]
    items.sort(key=lambda item: (PRIORITY.get(item.kind, 9), item.written_at), reverse=False)
    head = f"{procedure_id}: {len(items)} note(s)"
    lines = [head] + [f"- {item.render()}" for item in items[: max(0, int(limit))]]
    if len(items) > limit:
        lines.append(f"… and {len(items) - limit} more (handoff(limit=…))")
    return lines
