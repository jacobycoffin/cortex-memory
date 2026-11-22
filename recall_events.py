"""One immutable row per recall, and the query never ends up inside a memory (#66).

Recall is currently known only through counters (`retrieved_count`, `selected_count`) and an
`access_log` whose rows carry the query text. That answers "how often" and cannot answer the
questions that matter later:

- at what rank was this memory, with what score, from which retriever?
- was it **cited in the answer**, or merely injected into the context?
- what did the store say on turn 41 that it no longer says on turn 9,000?

So a recall event is recorded per memory per turn, with the score components that produced the
ranking. Two design choices:

1. **The table is append-only, enforced by triggers.** An event log that can be edited is not an
   event log; the triggers make "immutable" a property of the database rather than a convention.
2. **`used_in` is derived, not mutated.** Because rows cannot be updated, a citation is its own
   append (an event of type `citation` for the same turn and memory). `used_in` is then a query —
   `citation exists?` — which keeps both facts (ranked, then cited) instead of overwriting one with
   the other.

And the title's other half: recording a recall must never touch the memory's text. The query is a
**cue**, not content; a test asserts the content is byte-identical before and after recording, and
that the query string appears in no `memories.content` anywhere.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

_TABLE = "recall_events"
RANKED = "ranked"
CITATION = "citation"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {_TABLE} (
                    event_id TEXT PRIMARY KEY,
                    turn_id TEXT NOT NULL,
                    session_id TEXT,
                    memory_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    rank INTEGER,
                    score REAL,
                    components_json TEXT,
                    retriever TEXT,
                    created_at TEXT NOT NULL
                )"""
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_recall_events_turn ON {_TABLE}(turn_id, event_type)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_recall_events_memory ON {_TABLE}(memory_id, created_at)"
        )
        # Immutability as a database property, not a promise in a docstring.
        for operation in ("UPDATE", "DELETE"):
            conn.execute(
                f"""CREATE TRIGGER IF NOT EXISTS recall_events_no_{operation.lower()}
                    BEFORE {operation} ON {_TABLE}
                    BEGIN SELECT RAISE(ABORT, 'recall events are immutable'); END"""
            )


def _event_id(turn_id: str, memory_id: str, event_type: str, stamp: str) -> str:
    payload = f"{turn_id}:{memory_id}:{event_type}:{stamp}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


@dataclass
class Event:
    event_id: str
    turn_id: str
    memory_id: str
    event_type: str
    rank: int | None
    score: float | None
    retriever: str
    created_at: str


def record(
    store: Any,
    turn_id: str,
    results: list[Any],
    *,
    session_id: str | None = None,
    retriever: str = "hybrid",
    at: datetime | None = None,
) -> list[str]:
    """Record one row per recalled memory. Returns the event ids."""
    _ensure(store)
    stamp = (at or _utc_now()).isoformat()
    written: list[str] = []
    with store.transaction() as conn:
        for rank, result in enumerate(results, start=1):
            memory = getattr(result, "memory", None) or result
            memory_id = str(memory.get("id"))
            if not memory_id:
                continue
            components = getattr(result, "components", None) or {}
            score = getattr(result, "score", None)
            event_id = _event_id(turn_id, memory_id, RANKED, stamp + str(rank))
            conn.execute(
                f"""INSERT OR IGNORE INTO {_TABLE}
                    (event_id, turn_id, session_id, memory_id, event_type, rank, score,
                     components_json, retriever, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (event_id, str(turn_id), session_id, memory_id, RANKED, rank,
                 float(score) if score is not None else None,
                 json.dumps({str(key): float(value) for key, value in components.items()
                             if isinstance(value, (int, float))}),
                 retriever, stamp),
            )
            written.append(event_id)
    return written


def cite(store: Any, turn_id: str, memory_ids: list[str], *, at: datetime | None = None) -> list[str]:
    """Append a citation event for each memory that appeared in the answer."""
    _ensure(store)
    stamp = (at or _utc_now()).isoformat()
    written: list[str] = []
    with store.transaction() as conn:
        for memory_id in memory_ids:
            event_id = _event_id(str(turn_id), str(memory_id), CITATION, stamp)
            conn.execute(
                f"""INSERT OR IGNORE INTO {_TABLE}
                    (event_id, turn_id, session_id, memory_id, event_type, rank, score,
                     components_json, retriever, created_at)
                    VALUES (?, ?, NULL, ?, ?, NULL, NULL, NULL, NULL, ?)""",
                (event_id, str(turn_id), str(memory_id), CITATION, stamp),
            )
            written.append(event_id)
    return written


def events(store: Any, turn_id: str) -> list[Event]:
    _ensure(store)
    rows = store._conn.execute(
        f"""SELECT event_id, turn_id, memory_id, event_type, rank, score, retriever, created_at
            FROM {_TABLE} WHERE turn_id = ? ORDER BY created_at ASC, rank ASC""",
        (str(turn_id),),
    ).fetchall()
    return [
        Event(str(row["event_id"]), str(row["turn_id"]), str(row["memory_id"]),
              str(row["event_type"]), row["rank"], row["score"],
              str(row["retriever"] or ""), str(row["created_at"]))
        for row in rows
    ]


def used_in(store: Any, turn_id: str) -> set[str]:
    """Which memories were cited rather than merely injected — the `used_in` half of the row."""
    _ensure(store)
    rows = store._conn.execute(
        f"SELECT memory_id FROM {_TABLE} WHERE turn_id = ? AND event_type = ?",
        (str(turn_id), CITATION),
    ).fetchall()
    return {str(row["memory_id"]) for row in rows}


def detail(store: Any, turn_id: str) -> dict[str, Any]:
    """The turn as a reader wants it: rank, score, components, retriever, cited or not."""
    recorded = events(store, turn_id)
    cited = used_in(store, turn_id)
    ranked = [event for event in recorded if event.event_type == RANKED]
    return {
        "turn_id": turn_id,
        "ranked": len(ranked),
        "cited": len(cited),
        "results": [
            {
                "memory_id": event.memory_id,
                "rank": event.rank,
                "score": event.score,
                "retriever": event.retriever,
                "used_in": event.memory_id in cited,
            }
            for event in ranked
        ],
    }


def summary(store: Any, *, limit: int = 50) -> dict[str, Any]:
    _ensure(store)
    row = store._conn.execute(
        f"""SELECT COUNT(*) AS n,
                   COUNT(DISTINCT turn_id) AS turns,
                   SUM(CASE WHEN event_type = '{RANKED}' THEN 1 ELSE 0 END) AS ranked,
                   SUM(CASE WHEN event_type = '{CITATION}' THEN 1 ELSE 0 END) AS citations
            FROM {_TABLE}"""
    ).fetchone()
    return {
        "events": int(row["n"] or 0),
        "turns": int(row["turns"] or 0),
        "ranked": int(row["ranked"] or 0),
        "citations": int(row["citations"] or 0),
        "recent_turns": [
            str(item["turn_id"])
            for item in store._conn.execute(
                f"SELECT DISTINCT turn_id FROM {_TABLE} ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        ],
    }


def query_leaked_into_memory(store: Any, query: str) -> list[str]:
    """Guard for the title's other half: is this query text stored as a memory anywhere?"""
    text = str(query or "").strip()
    if not text:
        return []
    rows = store._conn.execute(
        "SELECT id FROM memories WHERE content LIKE ? LIMIT 20", (f"%{text}%",)
    ).fetchall()
    return [str(row["id"]) for row in rows]
