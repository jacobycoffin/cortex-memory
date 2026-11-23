"""Answer "what did I believe on this date" from the version log (#55).

The store keeps two clocks in `memory_versions`: **system time** (`system_from`/`system_to`, when the
store held a version) and **valid time** (`valid_from`/`valid_to`, when the fact was true in the
world). They are different questions, and conflating them is how a history feature ends up confidently
wrong:

- *"What did I believe on 2026-03-01?"* is system time — the store's own state that day, mistakes
  included;
- *"What was true on 2026-03-01?"* is valid time — the world's state, which may have been recorded
  later.

This module answers the first, and says so. It reads the version log rather than reconstructing belief
from current rows, because a reconstruction cannot distinguish "we did not know this yet" from "we knew
it and it was wrong".

Nothing is written: `believed()` is a query over a log that already exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def instant(value: Any) -> str:
    """A comparable ISO timestamp. A bare date means the start of that day, in UTC."""
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc).isoformat()
    text = str(value or "").strip()
    if not text:
        return datetime.now(timezone.utc).isoformat()
    if _ISO.match(text):
        return f"{text}T00:00:00+00:00"
    return text


@dataclass
class Belief:
    memory_id: str
    content: str
    kind: str = ""
    state: str = ""
    confidence: float = 0.0
    since: str = ""
    until: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "kind": self.kind,
            "state": self.state,
            "confidence": self.confidence,
            "since": self.since,
            "until": self.until,
        }

    def describe(self) -> str:
        window = f"{self.since[:10]} -> {self.until[:10] if self.until else 'now'}"
        return f"[{self.kind or 'fact'} {window}] {self.content[:90]}"


def _versions(store: Any, *, limit: int = 100000) -> list[dict[str, Any]]:
    return _rows(
        store,
        """SELECT version_id, memory_id, content, confidence, state, valid_from, valid_to,
                  system_from, system_to
           FROM memory_versions ORDER BY memory_id, system_from LIMIT ?""",
        (int(limit),),
    )


def believed(store: Any, when: Any, *, limit: int = 50) -> list[Belief]:
    """What the store held as current at that instant, one version per memory."""
    moment = instant(when)
    rows = _rows(
        store,
        """SELECT version_id, memory_id, content, confidence, state, system_from, system_to
           FROM memory_versions
           WHERE system_from <= ? AND (system_to IS NULL OR system_to = '' OR system_to > ?)
           ORDER BY system_from""",
        (moment, moment),
    )
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest[str(row["memory_id"])] = row  # the last version opened at or before the instant
    kinds = {
        str(row["id"]): str(row.get("kind") or "")
        for row in _rows(store, "SELECT id, kind FROM memories")
    }
    beliefs = [
        Belief(
            memory_id=memory_id,
            content=str(row["content"] or ""),
            kind=kinds.get(memory_id, ""),
            state=str(row["state"] or ""),
            confidence=float(row["confidence"] or 0.0),
            since=str(row["system_from"] or ""),
            until=str(row["system_to"] or ""),
        )
        for memory_id, row in latest.items()
    ]
    beliefs.sort(key=lambda belief: (belief.since, belief.memory_id))
    return beliefs[: max(0, int(limit))]


def answer(store: Any, when: Any, *, query: str = "", limit: int = 20) -> dict[str, Any]:
    """The same question, phrased for a person, optionally narrowed to a subject."""
    moment = instant(when)
    beliefs = believed(store, moment, limit=100000)
    words = {word for word in re.split(r"[^a-z0-9]+", str(query or "").lower()) if len(word) > 2}
    if words:
        beliefs = [
            belief for belief in beliefs
            if words & {word for word in re.split(r"[^a-z0-9]+", belief.content.lower())}
        ]
    earliest = _earliest(store)
    note = ""
    if earliest and moment < earliest:
        note = f"the version log starts at {earliest[:10]}; before that the store cannot say"
    return {
        "when": moment,
        "facts": [belief.as_dict() for belief in beliefs[: max(0, int(limit))]],
        "count": len(beliefs),
        "earliest_version": earliest,
        "note": note,
        "sentence": (f"{len(beliefs)} fact(s) held on {moment[:10]}" + (f" about {query}" if query else "")
                     + (f" - {note}" if note else "")),
    }


def _earliest(store: Any) -> str:
    rows = _rows(store, "SELECT MIN(system_from) AS earliest FROM memory_versions")
    return str(rows[0]["earliest"]) if rows and rows[0].get("earliest") else ""


def changes(store: Any, start: Any, end: Any, *, limit: int = 100) -> dict[str, Any]:
    """What the store started and stopped believing between two instants."""
    first, last = instant(start), instant(end)
    opened = _rows(
        store,
        """SELECT DISTINCT memory_id, content, system_from FROM memory_versions
           WHERE system_from > ? AND system_from <= ? ORDER BY system_from LIMIT ?""",
        (first, last, int(limit)),
    )
    closed = _rows(
        store,
        """SELECT DISTINCT memory_id, content, system_to FROM memory_versions
           WHERE system_to IS NOT NULL AND system_to <> '' AND system_to > ? AND system_to <= ?
           ORDER BY system_to LIMIT ?""",
        (first, last, int(limit)),
    )
    return {
        "start": first,
        "end": last,
        "learned": [dict(row) for row in opened],
        "retired": [dict(row) for row in closed],
        "sentence": (f"{len(opened)} fact(s) learned and {len(closed)} retired between "
                     f"{first[:10]} and {last[:10]}"),
    }


def history(store: Any, memory_id: str) -> list[dict[str, Any]]:
    """Every version of one fact, with the window it was believed for."""
    return _rows(
        store,
        """SELECT version_id, content, confidence, state, system_from, system_to, reason
           FROM memory_versions WHERE memory_id = ? ORDER BY system_from""",
        (str(memory_id),),
    )


def coverage(store: Any) -> dict[str, Any]:
    rows = _rows(
        store,
        """SELECT COUNT(*) AS versions,
                  SUM(CASE WHEN system_to IS NOT NULL AND system_to <> '' THEN 1 ELSE 0 END) AS closed,
                  MIN(system_from) AS earliest, MAX(system_from) AS latest,
                  COUNT(DISTINCT memory_id) AS memories
           FROM memory_versions""",
    )
    row = rows[0] if rows else {}
    versions = int(row.get("versions") or 0)
    closed = int(row.get("closed") or 0)
    return {
        "versions": versions,
        "closed": closed,
        "open": versions - closed,
        "memories": int(row.get("memories") or 0),
        "earliest": str(row.get("earliest") or ""),
        "latest": str(row.get("latest") or ""),
        "closed_share": round(closed / versions, 4) if versions else 0.0,
    }


@dataclass
class Interval:
    content: str
    start: str
    end: str = ""

    @property
    def current(self) -> bool:
        return not self.end


class Timeline:
    """A memory's belief intervals, in order."""

    def __init__(self, intervals: list[Interval] | None = None) -> None:
        self.intervals: list[Interval] = intervals or []

    @property
    def current(self) -> Interval | None:
        for interval in self.intervals:
            if interval.current:
                return interval
        return None

    def as_list(self) -> list[dict[str, Any]]:
        return [
            {"content": item.content, "start": item.start, "end": item.end or None,
             "current": item.current}
            for item in self.intervals
        ]


def timeline(store: Any, memory_id: str) -> Timeline:
    return Timeline([
        Interval(content=str(row["content"] or ""), start=str(row["system_from"] or ""),
                 end=str(row["system_to"] or ""))
        for row in history(store, memory_id)
    ])
