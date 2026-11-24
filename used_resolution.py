"""Which injected memories did the reply actually use (#145)

`citations.py` asks "did the reply mention this memory". This asks the narrower, countable question the
counters need: **of the memories injected into this turn, which ones did the answer actually lean on** —
and it refuses to guess when the evidence does not point at one.

Five rules, and the last two are the ones that keep `used_count` meaningful:

- **only injected ids are candidates.** A memory the reply happens to echo but that was never injected is
  not evidence of use; counting it would credit the store for something it did not supply.
- **distinctiveness decides, not overlap.** A word that appears in 80% of the store ("hermes", "dashboard")
  proves nothing. Overlap is weighted by inverse document frequency against the live store, so a shared
  rare term counts and a shared ubiquitous one does not.
- **ambiguity is reported, never tie-broken.** If two injected memories match the reply equally, the turn
  is `ambiguous` and *neither* gets a `used` — picking the higher-scoring one because it sorts first is
  how a counter becomes a fiction.
- **unresolved is not unused.** No match means "cannot tell", so the counter is left alone. `used_count`
  is evidence of use, not evidence of absence.
- **resolution is idempotent.** The same turn resolved twice must not double-count, so each
  (turn, memory) pair is recorded once and `apply()` skips what it has already seen.

Live context: 519 traces, **515 of them record selected ids and influence data**, and 1,064 of 3,843
memories carry a `used_count` — so the counter exists, the selections are recorded, and what has been
missing is the step in between.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

MIN_DISTINCT_WORDS = 2
IDF_FLOOR = 0.35          # a term in more than ~65% of the store cannot carry a match on its own
_WORD = re.compile(r"[a-z0-9][a-z0-9'_-]{2,}")
_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about", "there", "their", "them",
    "then", "than", "when", "what", "which", "while", "would", "could", "should", "have", "has",
    "had", "was", "were", "are", "you", "your", "his", "her", "she", "him", "they", "will", "not",
    "but", "because", "just", "like", "also", "only", "some", "such", "over", "under", "after",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def terms(text: str) -> set[str]:
    return {word for word in _WORD.findall(str(text or "").lower()) if word not in _STOP}


def _idf(store: Any) -> tuple[dict[str, float], int]:
    """Inverse document frequency over the live store, computed once per resolution."""
    rows = _rows(store, "SELECT content FROM memories WHERE state <> 'archived'")
    total = max(1, len(rows))
    seen: dict[str, int] = {}
    for row in rows:
        for word in terms(row.get("content") or ""):
            seen[word] = seen.get(word, 0) + 1
    return ({word: math.log(total / count) for word, count in seen.items()}, total)


def _weight(word: str, table: dict[str, float], total: int) -> float:
    value = table.get(word)
    if value is None:
        return math.log(max(2, total))       # a term the store has never seen is maximally distinctive
    return max(IDF_FLOOR, value)


@dataclass
class Resolution:
    turn_id: str = ""
    resolved: list[str] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"turn_id": self.turn_id, "resolved": list(self.resolved),
                "ambiguous": list(self.ambiguous), "unresolved": list(self.unresolved),
                "scores": {key: round(value, 3) for key, value in self.scores.items()},
                "note": self.note}


def resolve(store: Any, reply: str, candidate_ids: Iterable[str], *, turn_id: str = "",
            threshold: float = 0.5) -> Resolution:
    """Score each injected memory against the reply, weighted by how distinctive its terms are."""
    found = Resolution(turn_id=turn_id)
    wanted = [str(item) for item in candidate_ids]
    if not wanted:
        found.note = "no memories were injected, so there is nothing to resolve"
        return found
    rows = _rows(store, "SELECT id, content FROM memories WHERE id IN ("
                        + ",".join("?" for _ in wanted) + ")", tuple(wanted))
    if not rows:
        found.unresolved = list(wanted)
        found.note = "none of the injected ids are memories in this store"
        return found
    table, total = _idf(store)
    reply_terms = terms(reply)
    if not reply_terms:
        found.unresolved = [str(row["id"]) for row in rows]
        found.note = "the reply has no content words to match"
        return found
    scored: list[tuple[str, float, int]] = []
    for row in rows:
        memory_terms = terms(row.get("content") or "")
        shared = memory_terms & reply_terms
        if not shared:
            continue
        weights = sorted((_weight(word, table, total) for word in shared), reverse=True)
        # Two distinctive shared terms are the bar: one is a coincidence, and a pile of common words is
        # worse than one rare one. Distinctiveness is judged on the RAW idf, before the scoring floor —
        # otherwise the floor makes every ubiquitous word pass its own test.
        raw = [table.get(word, math.log(max(2, total))) for word in shared]
        distinct = [value for value in raw if value >= IDF_FLOOR]
        if len(distinct) < MIN_DISTINCT_WORDS:
            continue
        scored.append((str(row["id"]), sum(weights), len(distinct)))
    if not scored:
        found.unresolved = [str(row["id"]) for row in rows]
        found.note = "no injected memory shares enough distinctive terms with the reply"
        return found
    # Every injected id has to land somewhere: a memory that shares nothing with the reply is
    # unresolved, not invisible.
    seen = {item[0] for item in scored}
    for memory_id in wanted:
        if memory_id not in seen and memory_id not in found.unresolved:
            found.unresolved.append(memory_id)
    scored.sort(key=lambda item: -item[1])
    best_id, best_score, _count = scored[0]
    tied = [item[0] for item in scored
            if abs(item[1] - best_score) < 1e-9 and item[0] != best_id]
    found.scores = {item[0]: item[1] for item in scored}
    for item in scored:
        if item[1] >= best_score * threshold:
            found.resolved.append(item[0])
        elif item[0] not in found.unresolved:
            found.unresolved.append(item[0])
    if tied:
        found.ambiguous = sorted([best_id] + tied)
        found.resolved = [item for item in found.resolved if item not in found.ambiguous]
        found.note = (f"{len(found.ambiguous)} memories matched the reply equally; "
                      "none of them is counted as used")
    return found


def ensure_schema(store: Any) -> bool:
    try:
        with store.transaction() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS used_resolutions ("
                "turn_id TEXT NOT NULL, memory_id TEXT NOT NULL, resolved_at TEXT NOT NULL, "
                "score REAL NOT NULL, PRIMARY KEY(turn_id, memory_id))"
            )
        return True
    except Exception:  # noqa: BLE001
        return False


def apply(store: Any, resolution: Resolution, *, dry_run: bool = False,
          now: datetime | None = None) -> dict[str, Any]:
    """Count the resolved ids once per turn. Ambiguous and unresolved ids are never counted."""
    ensure_schema(store)
    moment = (now or _now()).isoformat()
    applied: list[str] = []
    skipped: list[str] = []
    if dry_run:
        return {"applied": list(resolution.resolved), "skipped": [], "dry_run": True}
    for memory_id in resolution.resolved:
        try:
            with store.transaction() as conn:
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO used_resolutions(turn_id, memory_id, resolved_at, score) "
                    "VALUES(?,?,?,?)",
                    (resolution.turn_id, memory_id, moment,
                     float(resolution.scores.get(memory_id, 0.0))),
                )
                if not cursor.rowcount:
                    skipped.append(memory_id)
                    continue
                conn.execute("UPDATE memories SET used_count = COALESCE(used_count, 0) + 1, "
                             "last_used_at = ? WHERE id = ?", (moment, memory_id))
            applied.append(memory_id)
        except Exception:  # noqa: BLE001
            skipped.append(memory_id)
    return {"applied": applied, "skipped": skipped, "dry_run": False}


def summary(store: Any, *, limit: int = 200) -> dict[str, Any]:
    """How much of the recent record the resolver could actually account for."""
    rows = _rows(store, "SELECT trace_id, selected_memory_ids_json FROM memory_traces "
                        "ORDER BY created_at DESC LIMIT ?", (max(1, int(limit)),))
    with_selection = 0
    total_ids = 0
    for row in rows:
        try:
            ids = json.loads(row.get("selected_memory_ids_json") or "[]")
        except Exception:  # noqa: BLE001
            ids = []
        if ids:
            with_selection += 1
            total_ids += len(ids)
    resolved = _rows(store, "SELECT COUNT(*) AS count FROM used_resolutions")
    return {
        "traces": len(rows),
        "with_selection": with_selection,
        "injected_ids": total_ids,
        "resolutions_recorded": int(resolved[0]["count"] or 0) if resolved else 0,
        "note": ("the store records what was injected, not what was said, so resolution has to run "
                 "at turn time; a reply is not stored to re-resolve later"),
    }


def render(store: Any, *, limit: int = 200) -> list[str]:
    described = summary(store, limit=limit)
    lines = [f"used-id resolution: {described['resolutions_recorded']} recorded across "
             f"{described['with_selection']} of {described['traces']} recent trace(s)"]
    lines.append(f"  injected ids seen: {described['injected_ids']}")
    lines.append(f"  {described['note']}")
    return lines
