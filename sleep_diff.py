"""What changed overnight, in plain English (#106).

The store changes while nobody is watching: memories retired by the sweep, corrections written, promotions into
the core, contradictions resolved. This module answers one question in the morning — *what is different about
what you know* — and it is deliberately not a metrics page. A digest that reports "605 maintenance actions" tells
you the machine was busy; a digest that reports "eleven vault notes were retired as duplicates" tells you what you
would have wanted to know.

Five rules, each a way an overnight report becomes noise:

- **changes only, never metrics.** A count of actions taken is in `weekly_digest`; this one lists what is now
  different about the content.
- **a quiet night is a valid answer.** "Nothing changed" is a fact and it is reported as one — an empty report
  that looks like a broken one teaches people to ignore the whole feature.
- **every line names its table and the memory.** A change you cannot trace is a rumour, and the eight-character
  prefix is enough to look up.
- **the report is bounded, and the cut is stated.** A night with four hundred changes needs one line and a number,
  not four hundred lines nobody reads on a phone.
- **a change this module cannot classify is reported as `other`, with its table** — never dropped, because "the
  sweep did something I do not have a word for" is exactly the thing worth reading.

Read-only: every statement is a SELECT, and a test snapshots the store to prove the report changes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

MEMORY_TABLE = "memories"
LIFECYCLE_TABLE = "lifecycle_events"
CORE_TABLE = "core_history"

DEFAULT_HOURS = 12
MAX_LINES = 12

# Plain English per lifecycle transition. A transition not in this table is not hidden: it is reported as
# "changed state" with the two states, which is honest and still readable.
TRANSITIONS: dict[tuple[str, str], str] = {
    ("active", "archived"): "retired",
    ("active", "cold"): "set aside",
    ("cold", "active"): "brought back",
    ("active", "quarantine"): "quarantined",
    ("quarantine", "active"): "released from quarantine",
    ("active", "tombstoned"): "tombstoned",
    ("", "active"): "restored",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]] | None:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return None


def _iso(moment: datetime) -> str:
    return moment.isoformat()


def _short(text: Any, limit: int = 70) -> str:
    body = " ".join(str(text or "").split())
    return body if len(body) <= limit else body[: limit - 1] + "…"


@dataclass
class Change:
    kind: str
    memory_id: str
    when: str
    sentence: str
    table: str = ""
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "memory_id": self.memory_id, "when": self.when,
                "sentence": self.sentence, "table": self.table, "detail": self.detail}


@dataclass
class Report:
    window_hours: int = DEFAULT_HOURS
    since: str = ""
    until: str = ""
    readable: bool = True
    reason: str = ""
    changes: list[Change] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    more: int = 0

    @property
    def quiet(self) -> bool:
        return self.readable and not self.changes and not self.more

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_hours": self.window_hours,
            "since": self.since,
            "until": self.until,
            "readable": self.readable,
            "reason": self.reason,
            "quiet": self.quiet,
            "counts": self.counts,
            "changes": [change.as_dict() for change in self.changes],
            "more": self.more,
        }


def report(store: Any, *, hours: int = DEFAULT_HOURS, until: datetime | None = None,
           limit: int = MAX_LINES, previews: bool = True) -> Report:
    """What is different since `hours` ago. Never raises."""
    window = max(1, int(hours))
    moment = until or _now()
    start = moment - timedelta(hours=window)
    out = Report(window_hours=window, since=_iso(start), until=_iso(moment))

    lifecycle = _rows(
        store,
        f"SELECT memory_id, from_state, to_state, reason, created_at FROM {LIFECYCLE_TABLE} "
        "WHERE created_at >= ? AND created_at <= ? ORDER BY created_at DESC",
        (_iso(start), _iso(moment)),
    )
    added = _rows(
        store,
        f"SELECT id, kind, content, source_type, created_at FROM {MEMORY_TABLE} "
        "WHERE created_at >= ? AND created_at <= ? ORDER BY created_at DESC",
        (_iso(start), _iso(moment)),
    )
    promoted = _rows(
        store,
        f"SELECT memory_id, action, reason, at FROM {CORE_TABLE} WHERE at >= ? AND at <= ? "
        "ORDER BY at DESC",
        (_iso(start), _iso(moment)),
    )
    if lifecycle is None and added is None and promoted is None:
        out.readable = False
        out.reason = "the store could not be read"
        return out

    changes: list[Change] = []
    for row in added or []:
        changes.append(Change(
            kind="added",
            memory_id=str(row.get("id")),
            when=str(row.get("created_at") or ""),
            sentence=f"learned: {_short(row.get('content'))}" if previews else "learned a new memory",
            table=MEMORY_TABLE,
            detail=str(row.get("source_type") or ""),
        ))
    for row in lifecycle or []:
        from_state = str(row.get("from_state") or "")
        to_state = str(row.get("to_state") or "")
        verb = TRANSITIONS.get((from_state, to_state)) or f"changed state {from_state or 'new'} -> {to_state}"
        changes.append(Change(
            kind=verb.split()[0],
            memory_id=str(row.get("memory_id")),
            when=str(row.get("created_at") or ""),
            sentence=f"{verb} ({row.get('reason') or 'no reason recorded'})",
            table=LIFECYCLE_TABLE,
            detail=f"{from_state}->{to_state}",
        ))
    for row in promoted or []:
        changes.append(Change(
            kind="promoted",
            memory_id=str(row.get("memory_id")),
            when=str(row.get("at") or ""),
            sentence=f"promoted to the core ({row.get('action') or 'move'})",
            table=CORE_TABLE,
            detail=str(row.get("reason") or ""),
        ))

    changes.sort(key=lambda change: change.when, reverse=True)
    out.counts = {}
    for change in changes:
        out.counts[change.kind] = out.counts.get(change.kind, 0) + 1
    out.more = max(0, len(changes) - max(0, int(limit)))
    out.changes = changes[: max(0, int(limit))]
    return out


def render(report: Report, *, limit: int = MAX_LINES) -> list[str]:
    """Chat-sized lines. Quiet is stated as a sentence, not as an empty page."""
    if not report.readable:
        return [report.reason]
    if report.quiet:
        return [f"Nothing changed in the last {report.window_hours}h."]
    head = (f"{sum(report.counts.values())} change(s) in the last {report.window_hours}h: "
            + ", ".join(f"{count} {kind}" for kind, count in sorted(report.counts.items())))
    lines = [head]
    for change in report.changes[:limit]:
        lines.append(f"- {change.memory_id[:8]} {change.sentence}  [{change.table}]")
    if report.more:
        lines.append(f"… and {report.more} more (see sleep_diff.report(limit=…))")
    return lines
