"""What a memory actually taught the agent (#81).

Every other view of a memory describes it: its content, its scope, its evidence, its trend. None of
them answer the question a person asks when deciding whether to keep it — *what did this change?*

This reads that answer out of what the store already recorded in a window: the turns it was selected
for and used in, the tool calls it informed (#70), how those tasks turned out (#71), the corrections
that replaced it, and the associations it formed. Nothing is written; this is a lens, not a ledger.

Two rules:

- **an effect is a recorded event, not a plausible one.** A memory that was injected forty times and
  never used taught nothing, and the count says zero rather than forty.
- **the window is stated.** "In the last 90 days" is part of the answer; a lifetime number would
  flatter an old memory and hide a dormant one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

DEFAULT_DAYS = 90


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001 - a lens that cannot read reports nothing
        return []


def _window(days: int, at: datetime | None = None) -> tuple[str, str]:
    end = at or _now()
    start = end - timedelta(days=max(1, int(days)))
    return start.isoformat(), end.isoformat()


def _table_exists(store: Any, name: str) -> bool:
    return bool(
        _rows(store, "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (name,))
    )


@dataclass
class Effects:
    memory_id: str
    days: int = DEFAULT_DAYS
    selected: int = 0
    injected: int = 0
    used: int = 0
    sessions: int = 0
    tool_calls_informed: int = 0
    tasks_succeeded: int = 0
    tasks_failed: int = 0
    corrections_replacing_it: int = 0
    associations_formed: int = 0
    queries: list[str] = field(default_factory=list)

    @property
    def taught_something(self) -> bool:
        return self.used > 0 or self.tool_calls_informed > 0 or self.associations_formed > 0

    @property
    def injected_but_unused(self) -> int:
        return max(0, self.injected - self.used)

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "days": self.days,
            "selected": self.selected,
            "injected": self.injected,
            "used": self.used,
            "injected_but_unused": self.injected_but_unused,
            "sessions": self.sessions,
            "tool_calls_informed": self.tool_calls_informed,
            "tasks_succeeded": self.tasks_succeeded,
            "tasks_failed": self.tasks_failed,
            "corrections_replacing_it": self.corrections_replacing_it,
            "associations_formed": self.associations_formed,
            "queries": self.queries,
            "taught_something": self.taught_something,
            "sentence": self.describe(),
        }

    def describe(self) -> str:
        if not self.taught_something:
            if self.injected:
                base = (f"in the last {self.days} days: offered {self.injected} time(s) and never "
                        f"used — it taught nothing")
            else:
                base = f"in the last {self.days} days: never retrieved, never used"
            if self.corrections_replacing_it:
                # Being corrected is information even when the memory was never retrieved: it was
                # wrong, and the store found out. Saying "taught nothing" would hide that.
                base += f", and was corrected {self.corrections_replacing_it} time(s)"
            return base
        parts = [f"used in {self.used} turn(s) across {self.sessions} session(s)"]
        if self.tool_calls_informed:
            outcome = f", {self.tasks_succeeded} succeeded"
            if self.tasks_failed:
                outcome += f" and {self.tasks_failed} failed"
            parts.append(f"informed {self.tool_calls_informed} tool call(s){outcome}")
        if self.associations_formed:
            parts.append(f"formed {self.associations_formed} association(s)")
        if self.corrections_replacing_it:
            parts.append(f"was corrected {self.corrections_replacing_it} time(s)")
        return f"in the last {self.days} days: " + "; ".join(parts)


def effects(store: Any, memory_id: str, *, days: int = DEFAULT_DAYS,
            at: datetime | None = None, query_limit: int = 3) -> Effects:
    """Everything this memory caused inside the window, from recorded events only."""
    identifier = str(memory_id or "").strip()
    start, end = _window(days, at)
    found = Effects(memory_id=identifier, days=int(days))
    for row in _rows(
        store,
        """SELECT event, COUNT(*) AS count, COUNT(DISTINCT session_id) AS sessions
           FROM access_log WHERE memory_id = ? AND created_at >= ? AND created_at <= ?
           GROUP BY event""",
        (identifier, start, end),
    ):
        event = str(row["event"])
        count = int(row["count"] or 0)
        if event == "selected":
            found.selected = count
        elif event == "injected":
            found.injected = count
        elif event == "used":
            found.used = count
        found.sessions = max(found.sessions, int(row["sessions"] or 0))
    if found.used:
        found.queries = [
            str(row["query"] or "")
            for row in _rows(
                store,
                """SELECT query FROM access_log
                   WHERE memory_id = ? AND event = 'used' AND created_at >= ?
                     AND query IS NOT NULL AND query <> ''
                   ORDER BY created_at DESC LIMIT ?""",
                (identifier, start, int(query_limit)),
            )
        ]
    if _table_exists(store, "memory_tool_links"):
        found.tool_calls_informed = len(_rows(
            store,
            """SELECT execution_id FROM memory_tool_links
               WHERE memory_id = ? AND created_at >= ?""",
            (identifier, start),
        ))
    if _table_exists(store, "memory_outcomes"):
        for row in _rows(
            store,
            """SELECT outcome, COUNT(*) AS count FROM memory_outcomes
               WHERE memory_id = ? AND created_at >= ? GROUP BY outcome""",
            (identifier, start),
        ):
            if str(row["outcome"]) == "success":
                found.tasks_succeeded = int(row["count"] or 0)
            elif str(row["outcome"]) == "failure":
                found.tasks_failed = int(row["count"] or 0)
    if _table_exists(store, "self_corrections"):
        found.corrections_replacing_it = len(_rows(
            store,
            """SELECT correction_id FROM self_corrections
               WHERE superseded_id = ? AND created_at >= ?""",
            (identifier, start),
        ))
    if _table_exists(store, "edges"):
        found.associations_formed = len(_rows(
            store,
            """SELECT 1 AS found FROM edges
               WHERE (src_id = ? OR dst_id = ?) AND created_at >= ? LIMIT 200""",
            (identifier, identifier, start),
        ))
    return found


def digest(store: Any, memory_id: str, *, days: int = DEFAULT_DAYS,
           at: datetime | None = None) -> dict[str, Any]:
    """The effects plus the memory's own text, so the sentence has a subject."""
    row = _rows(store, "SELECT id, content, kind FROM memories WHERE id = ?", (str(memory_id),))
    found = effects(store, memory_id, days=days, at=at)
    payload = found.as_dict()
    payload["content"] = str(row[0]["content"]) if row else ""
    payload["kind"] = str(row[0]["kind"]) if row else ""
    return payload


def most_influential(store: Any, *, days: int = DEFAULT_DAYS, limit: int = 10,
                     at: datetime | None = None, sample: int = 4000) -> list[dict[str, Any]]:
    """Memories that changed the most in the window, by recorded effects."""
    start, _end = _window(days, at)
    rows = _rows(
        store,
        """SELECT memory_id, SUM(event = 'used') AS used,
                  SUM(event = 'injected') AS injected
           FROM access_log WHERE created_at >= ?
           GROUP BY memory_id HAVING used > 0 ORDER BY used DESC LIMIT ?""",
        (start, int(sample)),
    )
    ranked: list[dict[str, Any]] = []
    for row in rows:
        memory_id = str(row["memory_id"])
        found = effects(store, memory_id, days=days, at=at, query_limit=0)
        ranked.append({
            "memory_id": memory_id,
            "used": found.used,
            "injected": found.injected,
            "tool_calls_informed": found.tool_calls_informed,
            "score": found.used + 2 * found.tool_calls_informed + found.associations_formed,
            "sentence": found.describe(),
        })
    ranked.sort(key=lambda item: (-item["score"], item["memory_id"]))
    return ranked[: max(0, int(limit))]


def silently_useless(store: Any, *, days: int = DEFAULT_DAYS, limit: int = 20,
                     at: datetime | None = None) -> list[dict[str, Any]]:
    """Offered repeatedly, never used — the honest counterweight to the influential list."""
    start, _end = _window(days, at)
    rows = _rows(
        store,
        """SELECT memory_id, SUM(event = 'injected') AS injected, SUM(event = 'used') AS used
           FROM access_log WHERE created_at >= ?
           GROUP BY memory_id HAVING injected >= 5 AND used = 0
           ORDER BY injected DESC LIMIT ?""",
        (start, int(limit)),
    )
    return [
        {"memory_id": str(row["memory_id"]), "injected": int(row["injected"] or 0),
         "used": int(row["used"] or 0)}
        for row in rows
    ]


def report(store: Any, *, days: int = DEFAULT_DAYS, at: datetime | None = None) -> dict[str, Any]:
    _ = re  # keep the module import surface honest if the lens grows
    start, _end = _window(days, at)
    rows = _rows(
        store,
        """SELECT COUNT(DISTINCT memory_id) AS memories,
                  SUM(event = 'injected') AS injected, SUM(event = 'used') AS used
           FROM access_log WHERE created_at >= ?""",
        (start,),
    )
    row = rows[0] if rows else {}
    injected = int(row.get("injected") or 0)
    used = int(row.get("used") or 0)
    return {
        "days": int(days),
        "memories": int(row.get("memories") or 0),
        "injected": injected,
        "used": used,
        "use_share": round(used / injected, 4) if injected else 0.0,
        "note": "an effect is a recorded event; offered is not used",
    }
