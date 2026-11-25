"""What the night noticed and did not act on (#111).

Most of what a sleep cycle finds, it should not act on: a near-duplicate that is not close enough, a contradiction
whose sides are both cited, a stale memory that ordinary use will reach, a memory nothing can classify. Today each of
those is dropped silently — the cycle looks clean in the morning because its doubts were never written down, and the
same near-misses are re-discovered and re-dropped every night.

A dream journal fixes the *silence*, not the inaction. Five rules:

- **a notice is not an action.** Writing a journal entry changes no memory, ever; a test snapshots the store to prove
  it. The journal is a record of restraint, and restraint must not quietly become a queue.
- **every entry says why it was not acted on** — `below_evidence_bar`, `contradicted`, `budget`, `needs_decision`,
  `not_scheduled`. An entry without a reason is refused, because "the night noticed something" with no verdict is
  exactly the noise that makes people stop reading journals.
- **it is per night** (the same 06:00 UTC boundary as the sleep budget) **and deduplicated**: noticing the same thing
  three times in one night increments `times` instead of writing three rows.
- **it is bounded**, and the bound is stated: a night that noticed 6,771 unclassifiable memories writes one
  aggregated entry, not 6,771.
- **it keeps the ids, not the content.** A journal that quotes memories is a second copy of the store with weaker
  rules; ids let a reader look up what they are allowed to see.

Retention is by night: `prune()` drops nights older than `keep_nights`, because a journal nobody prunes becomes a
second store of record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

JOURNAL_TABLE = "sleep_journal"

NIGHT_START_HOUR = 6
MAX_ENTRIES_PER_NIGHT = 40
KEEP_NIGHTS = 30

# Why the night left something alone. A closed set, because a free-text reason cannot be counted, and the count is
# what tells you whether a bar is set wrong.
WHY_NOT_ACTED = (
    "below_evidence_bar",
    "contradicted",
    "budget",
    "needs_decision",
    "not_scheduled",
)

KINDS = (
    "near_duplicate",
    "contradiction",
    "stale",
    "unclassifiable",
    "low_confidence",
    "expansion_declined",
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def night_of(moment: datetime | None = None) -> str:
    return ((moment or _now()) - timedelta(hours=NIGHT_START_HOUR)).date().isoformat()


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]] | None:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return None


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {JOURNAL_TABLE} (
                    entry_id TEXT PRIMARY KEY,
                    night TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    why TEXT NOT NULL,
                    note TEXT NOT NULL,
                    memory_ids TEXT NOT NULL DEFAULT '',
                    times INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
        )


@dataclass
class Entry:
    entry_id: str
    night: str
    kind: str
    why: str
    note: str
    memory_ids: list[str] = field(default_factory=list)
    times: int = 1

    def render(self) -> str:
        more = f" x{self.times}" if self.times > 1 else ""
        ids = ", ".join(item[:8] for item in self.memory_ids[:3])
        if len(self.memory_ids) > 3:
            ids += f" +{len(self.memory_ids) - 3}"
        return f"[{self.kind}/{self.why}]{more} {self.note}" + (f" ({ids})" if ids else "")

    def as_dict(self) -> dict[str, Any]:
        return {"entry_id": self.entry_id, "night": self.night, "kind": self.kind, "why": self.why,
                "note": self.note, "memory_ids": self.memory_ids, "times": self.times}


def notice(store: Any, *, kind: str, note: str, why: str, memory_ids: Iterable[str] = (),
           night: str | None = None, limit: int = MAX_ENTRIES_PER_NIGHT) -> dict[str, Any]:
    """Record something the night noticed and did not act on. Never touches a memory."""
    if why not in WHY_NOT_ACTED:
        return {"ok": False, "reason": f"unknown why '{why}': an entry with no verdict is noise "
                                      f"(one of {', '.join(WHY_NOT_ACTED)})"}
    if kind not in KINDS:
        return {"ok": False, "reason": f"unknown kind '{kind}' (one of {', '.join(KINDS)})"}
    body = " ".join(str(note or "").split())
    if not body:
        return {"ok": False, "reason": "an entry needs a note saying what was noticed"}
    night_key = night or night_of()
    ids = sorted({str(item) for item in memory_ids if str(item or "").strip()})
    _ensure(store)

    existing = _rows(store, f"SELECT entry_id, kind, memory_ids, times FROM {JOURNAL_TABLE} "
                            "WHERE night = ? AND kind = ? AND why = ? AND note = ? ORDER BY entry_id",
                     (night_key, kind, why, body)) or []
    for row in existing:
        if ",".join(ids) == (row["memory_ids"] or ""):
            with store.transaction() as conn:
                conn.execute(f"UPDATE {JOURNAL_TABLE} SET times = times + 1, updated_at = ? WHERE entry_id = ?",
                             (_now().isoformat(), row["entry_id"]))
            return {"ok": True, "entry_id": row["entry_id"], "deduplicated": True,
                    "times": int(row["times"]) + 1}

    count = len(_rows(store, f"SELECT entry_id FROM {JOURNAL_TABLE} WHERE night = ?", (night_key,)) or [])
    if count >= max(0, int(limit)):
        return {"ok": False, "reason": f"the night's journal is full ({count} of {limit}): aggregate instead of "
                                      f"writing one entry per memory", "full": True, "count": count}

    entry_id = f"sj-{night_key}-{count + 1:03d}"
    stamp = _now().isoformat()
    with store.transaction() as conn:
        conn.execute(
            f"INSERT INTO {JOURNAL_TABLE} (entry_id, night, kind, why, note, memory_ids, times, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)",
            (entry_id, night_key, kind, why, body, ",".join(ids), stamp, stamp),
        )
    return {"ok": True, "entry_id": entry_id, "deduplicated": False, "times": 1}


def journal(store: Any, *, night: str | None = None) -> list[Entry]:
    rows = _rows(store, f"SELECT * FROM {JOURNAL_TABLE} WHERE night = ? ORDER BY entry_id",
                 (night or night_of(),))
    return [
        Entry(entry_id=str(row["entry_id"]), night=str(row["night"]), kind=str(row["kind"]),
              why=str(row["why"]), note=str(row["note"]),
              memory_ids=[part for part in (row["memory_ids"] or "").split(",") if part],
              times=int(row["times"]))
        for row in rows or []
    ]


def render(entries: list[Entry], *, limit: int = 12) -> list[str]:
    if not entries:
        return ["The night noticed nothing it needed to leave alone."]
    head = f"{len(entries)} notice(s) left alone: " + ", ".join(
        f"{count} {why}" for why, count in sorted(_count_by(entries, "why").items()))
    lines = [head]
    lines.extend(f"- {entry.render()}" for entry in entries[:limit])
    if len(entries) > limit:
        lines.append(f"… and {len(entries) - limit} more")
    return lines


def _count_by(entries: list[Entry], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        key = getattr(entry, attr)
        counts[key] = counts.get(key, 0) + 1
    return counts


def prune(store: Any, *, keep_nights: int = KEEP_NIGHTS, now: datetime | None = None) -> int:
    """Drop nights older than the retention window. Returns how many nights were dropped."""
    _ensure(store)
    cutoff = night_of((now or _now()) - timedelta(days=max(1, int(keep_nights))))
    rows = _rows(store, f"SELECT DISTINCT night FROM {JOURNAL_TABLE} WHERE night < ?", (cutoff,)) or []
    if not rows:
        return 0
    with store.transaction() as conn:
        conn.execute(f"DELETE FROM {JOURNAL_TABLE} WHERE night < ?", (cutoff,))
    return len(rows)
