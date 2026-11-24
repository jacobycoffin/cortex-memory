"""A receipt per answer: what this reply actually used (#174)

An answer that cites nothing and an answer that cites three memories look the same from the outside,
which is why "the brain helped" is usually a feeling rather than a fact. A receipt makes it a fact for
one turn:

- **used** — memories whose content this reply repeats (a declared marker, or a shared phrase), from
  `citations.detect`, which is the same detector the citation counters use;
- **offered** — memories the pipeline injected or ranked for the turn, cited or not;
- **unused** — offered and not cited. This is the number worth watching: a memory injected twenty times
  and never used is not evidence of a working brain, it is evidence of a noisy one (#81).

Rules, because a receipt that flatters is worse than none:

- **evidence, not decoration.** A memory appears under *used* only if the reply actually repeats it.
  Being injected is not use, and a receipt that counts injections as cites would make every answer look
  grounded.
- **a missing trace is not an empty answer.** If the turn was never recorded, the receipt says so
  instead of reporting zero citations, which would read as "the answer used nothing".
- **the receipt is persisted with the turn**, so the question "what did yesterday's answer rest on" is
  answerable after the reply is gone from context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from . import citations

TABLE = "recall_receipts"
SNIPPET = 120


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001 - a receipt must never break a turn
        return []


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    turn_id TEXT PRIMARY KEY,
                    offered INTEGER NOT NULL DEFAULT 0,
                    used INTEGER NOT NULL DEFAULT 0,
                    cited_ids TEXT NOT NULL DEFAULT '',
                    unused_ids TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )"""
        )


def _snippet(store: Any, memory_id: str) -> tuple[str, str]:
    rows = _rows(store, "SELECT content, kind FROM memories WHERE id = ?", (memory_id,))
    if not rows:
        return "", ""
    body = " ".join(str(rows[0]["content"] or "").split())
    return (body[: SNIPPET - 1] + "…" if len(body) > SNIPPET else body), str(rows[0]["kind"] or "")


def offered_ids(store: Any, task_id: str) -> list[str]:
    """The memories this turn was offered: injected candidates plus the final selection."""
    try:
        traces = store.memory_traces(limit=1, task_id=str(task_id)) or []
    except Exception:  # noqa: BLE001
        traces = []
    if not traces:
        return []
    trace = traces[0]
    ids: list[str] = []
    for candidate in trace.get("candidate_memories") or []:
        memory_id = str((candidate or {}).get("memory_id") or "")
        if memory_id and (candidate or {}).get("selected"):
            ids.append(memory_id)
    for memory_id in trace.get("selected_memory_ids") or []:
        text = str(memory_id)
        if text and text not in ids:
            ids.append(text)
    return ids


@dataclass
class Receipt:
    turn_id: str = ""
    recorded: bool = False
    offered: int = 0
    used: list[dict[str, Any]] = field(default_factory=list)
    unused: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    @property
    def used_count(self) -> int:
        return len(self.used)

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "recorded": self.recorded,
            "offered": self.offered,
            "used": self.used,
            "unused": self.unused,
            "note": self.note,
            "sentence": self.describe(),
        }

    def describe(self) -> str:
        if not self.recorded:
            return f"turn {self.turn_id}: not recorded, so this answer has no receipt"
        if not self.offered:
            return f"turn {self.turn_id}: nothing was offered to this answer"
        return (f"turn {self.turn_id}: this answer used {self.used_count} of {self.offered} "
                f"offered memor{'y' if self.offered == 1 else 'ies'}")

    def render(self, limit: int = 5) -> list[str]:
        lines = [self.describe()]
        for item in self.used[:limit]:
            how = item.get("method") or "?"
            lines.append(f"  ✓ {str(item['memory_id'])[:8]} ({how}) {item.get('snippet', '')[:70]}")
        if len(self.used) > limit:
            lines.append(f"  ✓ … {len(self.used) - limit} more used")
        if self.unused:
            lines.append(f"  · {len(self.unused)} offered but not used")
        if self.note:
            lines.append(f"  note: {self.note}")
        return lines


def build(store: Any, task_id: str, answer: str, *, min_words: int = citations.MIN_WORDS
          ) -> Receipt:
    """What this answer used, out of what it was offered."""
    found = Receipt(turn_id=str(task_id))
    offered = offered_ids(store, task_id)
    try:
        traces = store.memory_traces(limit=1, task_id=str(task_id)) or []
    except Exception:  # noqa: BLE001
        traces = []
    if not traces:
        found.note = "the turn was not recorded, so nothing can be said about what it used"
        return found
    found.recorded = True
    found.offered = len(offered)
    if not offered:
        found.note = "no memory was offered to this turn"
        return found
    detected = citations.detect(store, str(answer or ""), offered, min_words=min_words)
    cited = {item.memory_id: item for item in detected.cited}
    for memory_id in offered:
        snippet, kind = _snippet(store, memory_id)
        citation = cited.get(memory_id)
        row = {
            "memory_id": memory_id,
            "kind": kind,
            "snippet": snippet,
            "method": citation.method if citation else "",
            "phrase": citation.phrase if citation else "",
        }
        (found.used if citation else found.unused).append(row)
    if not found.used:
        found.note = ("none of the offered memories is repeated in the answer; offered is not used"
                      if answer else "no answer text was supplied to compare against")
    return found


def record(store: Any, task_id: str, answer: str, *, at: datetime | None = None,
           min_words: int = citations.MIN_WORDS) -> Receipt:
    """Persist the receipt for a turn, idempotently, and the citations it found."""
    moment = at or _now()
    found = build(store, task_id, answer, min_words=min_words)
    if not found.recorded or not found.offered:
        return found
    citations.record(store, found.turn_id, answer, offered_ids(store, task_id),
                     at=moment, min_words=min_words)
    _ensure(store)
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT OR REPLACE INTO {TABLE}
                (turn_id, offered, used, cited_ids, unused_ids, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
            (
                found.turn_id,
                found.offered,
                found.used_count,
                ",".join(item["memory_id"] for item in found.used),
                ",".join(item["memory_id"] for item in found.unused),
                moment.isoformat(),
            ),
        )
    return found


def show(store: Any, task_id: str) -> dict[str, Any]:
    """A stored receipt, without needing the answer any more."""
    _ensure(store)
    rows = _rows(store, f"SELECT * FROM {TABLE} WHERE turn_id = ?", (str(task_id),))
    if not rows:
        return {"turn_id": str(task_id), "recorded": False,
                "sentence": f"turn {task_id}: no receipt was recorded"}
    row = rows[0]
    cited = [item for item in str(row["cited_ids"] or "").split(",") if item]
    unused = [item for item in str(row["unused_ids"] or "").split(",") if item]
    used = []
    for memory_id in cited:
        snippet, kind = _snippet(store, memory_id)
        used.append({"memory_id": memory_id, "kind": kind, "snippet": snippet, "method": "recorded"})
    return {
        "turn_id": str(row["turn_id"]),
        "recorded": True,
        "offered": int(row["offered"] or 0),
        "used": used,
        "unused": [{"memory_id": memory_id} for memory_id in unused],
        "created_at": str(row["created_at"] or ""),
        "sentence": f"turn {row['turn_id']}: this answer used {len(cited)} of "
                    f"{int(row['offered'] or 0)} offered",
    }


def stats(store: Any, limit: int = 200) -> dict[str, Any]:
    """How much of what was offered is actually used, across recorded receipts."""
    _ensure(store)
    rows = _rows(store, f"SELECT * FROM {TABLE} ORDER BY created_at DESC LIMIT ?", (int(limit),))
    offered = sum(int(row["offered"] or 0) for row in rows)
    used = sum(int(row["used"] or 0) for row in rows)
    return {
        "turns": len(rows),
        "offered": offered,
        "used": used,
        "used_share": round(used / offered, 4) if offered else 0.0,
        "turns_with_no_use": sum(1 for row in rows if int(row["used"] or 0) == 0),
        "note": "offered is not used: an injected memory that never appears in an answer taught nothing",
    }


def unused_memories(store: Any, limit: int = 200) -> list[dict[str, Any]]:
    """Memories that were offered and never used — the #81 question, per turn."""
    _ensure(store)
    counts: dict[str, int] = {}
    for row in _rows(store, f"SELECT unused_ids FROM {TABLE} ORDER BY created_at DESC LIMIT ?",
                     (int(limit),)):
        for memory_id in str(row["unused_ids"] or "").split(","):
            if memory_id:
                counts[memory_id] = counts.get(memory_id, 0) + 1
    ranked: Iterable[tuple[str, int]] = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [{"memory_id": memory_id, "offered_unused": count} for memory_id, count in ranked]
