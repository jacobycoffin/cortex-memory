"""Which memories the answer actually used, not which ones were in the room (#201).

#70 records co-use: a memory was in context for a call that happened. That is circumstantial — a
memory can sit in the prompt while the answer comes from somewhere else entirely, and a store that
treats presence as use learns the wrong lesson from its own history (this is exactly how the note
about VR games became the most-retrieved memory in the store).

A citation is different: the reply repeats something that is only in the memory. This module looks for
that, conservatively, and records it per turn:

- **explicit markers win.** `[m:<id>]` or `<memory id>` in the reply is a citation by declaration; no
  guessing needed.
- **otherwise a run of words, not a word.** A citation needs a contiguous phrase of at least
  `MIN_WORDS` words shared with the memory, because a single matching word ("portal", "backup") is
  coincidence, and short overlaps are how a store convinces itself it was quoted.
- **the memory must be a candidate.** A phrase match against a memory that was never in context is a
  coincidence, not a citation, so only the ids the caller passed can be cited.

Nothing is inferred about *influence* — this records that a phrase from the memory appears in the
answer, which is checkable. Whether the answer would have been worse without it is a different and
much harder question, and this module does not pretend to answer it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

TABLE = "memory_citations"
MIN_WORDS = 5
_MARKER = re.compile(r"[\[<]\s*(?:m:|memory[:\s]+)?([0-9a-fA-F-]{8,64})\s*[\]>]")
_WORD = re.compile(r"[a-z0-9]+")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001 - a report that cannot read reports nothing
        return []


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    memory_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    method TEXT NOT NULL,
                    phrase TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (memory_id, turn_id)
                )"""
        )


def words(text: str) -> list[str]:
    return _WORD.findall(str(text or "").lower())


def phrases(text: str, *, size: int = MIN_WORDS) -> set[str]:
    """Every contiguous word run of at least `size`, as a set of space-joined strings.

    A run of words is the smallest unit that can be evidence: any single word is noise, and any
    scoring over bag-of-words cannot tell "the port is 8100" from "the port is not 8100".
    """
    tokens = words(text)
    found: set[str] = set()
    for index in range(0, max(0, len(tokens) - size + 1)):
        found.add(" ".join(tokens[index:index + size]))
    return found


def markers(reply: str) -> list[str]:
    """Ids the reply cites explicitly."""
    seen: list[str] = []
    for match in _MARKER.finditer(str(reply or "")):
        value = match.group(1)
        if value not in seen:
            seen.append(value)
    return seen


@dataclass
class Citation:
    memory_id: str
    method: str
    phrase: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"memory_id": self.memory_id, "method": self.method, "phrase": self.phrase}


@dataclass
class Result:
    turn_id: str = ""
    cited: list[Citation] = field(default_factory=list)
    considered: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "considered": self.considered,
            "cited": [item.as_dict() for item in self.cited],
            "sentence": self.describe(),
        }

    def describe(self) -> str:
        if not self.cited:
            return f"nothing cited out of {self.considered} candidate(s)"
        return f"{len(self.cited)} cited out of {self.considered}: " + ", ".join(
            f"{item.memory_id[:8]} ({item.method})" for item in self.cited[:5]
        )


def detect(store: Any, reply: str, candidate_ids: Iterable[str], *,
           min_words: int = MIN_WORDS) -> Result:
    """Which of the candidates this reply actually repeats — by marker or by phrase."""
    ids = [str(item) for item in candidate_ids if item]
    result = Result(considered=len(ids))
    if not ids:
        return result
    text = str(reply or "")
    declared = set(markers(text))
    reply_phrases = phrases(text, size=min_words)
    for memory_id in ids:
        if memory_id in declared or any(
            memory_id.startswith(value) for value in declared if len(value) >= 8
        ):
            result.cited.append(Citation(memory_id=memory_id, method="marker"))
            continue
        rows = _rows(store, "SELECT content FROM memories WHERE id = ?", (memory_id,))
        if not rows:
            continue
        shared = reply_phrases & phrases(str(rows[0]["content"] or ""), size=min_words)
        if shared:
            result.cited.append(
                Citation(memory_id=memory_id, method="phrase",
                         phrase=sorted(shared, key=len, reverse=True)[0][:120])
            )
    return result


def record(store: Any, turn_id: str, reply: str, candidate_ids: Iterable[str], *,
           at: datetime | None = None, min_words: int = MIN_WORDS) -> Result:
    """Detect and persist citations for one turn, idempotently per (memory, turn)."""
    moment = at or _now()
    found = detect(store, reply, candidate_ids, min_words=min_words)
    found.turn_id = str(turn_id)
    if not found.cited:
        return found
    _ensure(store)
    with store.transaction() as conn:
        for citation in found.cited:
            conn.execute(
                f"""INSERT OR IGNORE INTO {TABLE}
                    (memory_id, turn_id, method, phrase, created_at) VALUES (?, ?, ?, ?, ?)""",
                (citation.memory_id, found.turn_id, citation.method, citation.phrase,
                 moment.isoformat()),
            )
    return found


def citations(store: Any, memory_id: str, limit: int = 50) -> list[dict[str, Any]]:
    _ensure(store)
    return _rows(
        store,
        f"SELECT * FROM {TABLE} WHERE memory_id = ? ORDER BY created_at DESC LIMIT ?",
        (str(memory_id), int(limit)),
    )


def report(store: Any, *, sample: int = 500) -> dict[str, Any]:
    _ensure(store)
    rows = _rows(
        store,
        f"""SELECT method, COUNT(*) AS count, COUNT(DISTINCT memory_id) AS memories
            FROM {TABLE} GROUP BY method""",
    )
    by_method = {str(row["method"]): int(row["count"] or 0) for row in rows}
    totals = _rows(store, f"SELECT COUNT(*) AS total FROM {TABLE}")
    considered = _rows(
        store, "SELECT COUNT(*) AS total FROM memories WHERE state IN ('active', 'cold') LIMIT ?",
        (int(sample),),
    )
    return {
        "citations": int((totals[0] if totals else {}).get("total") or 0),
        "by_method": by_method,
        "memories_available": int((considered[0] if considered else {}).get("total") or 0),
        "min_words": MIN_WORDS,
        "note": "a phrase in the answer is checkable; influence is not",
    }
