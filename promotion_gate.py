"""A semantic claim needs independent episodes before it is allowed to exist (#105).

Episodes record what happened; a semantic fact is the store deciding that something is **generally
true**. The jump between the two is the riskiest write in the system, because a generalisation made
from one conversation is a guess with a citation attached, and once it is a semantic fact it will be
retrieved, quoted and acted on without the conversation in front of anyone.

So the gate is explicit and measurable:

- **episodes, not rows.** Evidence comes from the `episodes` table (real conversations), not from how
  many times a memory was edited.
- **independence is a different session.** Three paraphrases inside one conversation are one
  observation, not three: they were said by one person at one moment about one thing. A cluster needs
  `MIN_SESSIONS` distinct sessions and `MIN_DAYS` distinct days before it counts.
- **citations are the point.** A promoted fact carries the episode ids it came from, so the claim can
  be audited later — and retracted by pointing at the same evidence.

`ready()` returns a verdict with its reasons whether or not the gate passes, because "not enough
evidence yet" is information the caller should be able to show.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

EVIDENCE = "promotion_evidence"
MIN_EPISODES = 3
MIN_SESSIONS = 2
MIN_DAYS = 2

_STOP = frozenset(
    "the a an and or but if then than so just very now this that these those i you he she it we they "
    "is are was were be been being of to in on for with from by as at do does did not no yes".split()
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {EVIDENCE} (
                    memory_id TEXT NOT NULL,
                    episode_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL DEFAULT '',
                    observed_at TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (memory_id, episode_id)
                )"""
        )


def tokens(text: str) -> set[str]:
    return {
        word for word in re.split(r"[^a-z0-9]+", str(text or "").lower())
        if len(word) > 2 and word not in _STOP
    }


@dataclass
class Support:
    episodes: list[int] = field(default_factory=list)
    sessions: list[str] = field(default_factory=list)
    days: list[str] = field(default_factory=list)

    @property
    def independent(self) -> int:
        return max(len(self.sessions), len(self.days))

    def as_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "sessions": self.sessions,
            "days": self.days,
            "independent": self.independent,
        }


@dataclass
class Verdict:
    ready: bool
    statement: str = ""
    support: Support = field(default_factory=Support)
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "statement": self.statement,
            "support": self.support.as_dict(),
            "reasons": self.reasons,
            "sentence": self.describe(),
        }

    def describe(self) -> str:
        if self.ready:
            return (f"ready: {len(self.support.episodes)} episode(s) across "
                    f"{len(self.support.sessions)} session(s) and {len(self.support.days)} day(s)")
        return "not ready: " + ("; ".join(self.reasons) or "no evidence")


def episodes_for(store: Any, statement: str, *, overlap: int = 2,
                 limit: int = 5000) -> list[dict[str, Any]]:
    """Episodes that actually discuss the claim, by token overlap rather than substring luck."""
    wanted = tokens(statement)
    if not wanted:
        return []
    rows = _rows(
        store,
        """SELECT id, session_id, user_content, created_at FROM episodes
           ORDER BY created_at DESC LIMIT ?""",
        (int(limit),),
    )
    found: list[dict[str, Any]] = []
    for row in rows:
        have = tokens(str(row["user_content"] or ""))
        if len(wanted & have) >= overlap:
            found.append(row)
    return found


def support(store: Any, statement: str, *, overlap: int = 2) -> Support:
    """Which episodes back this claim, and how independent they are."""
    rows = episodes_for(store, statement, overlap=overlap)
    return Support(
        episodes=sorted(int(row["id"]) for row in rows),
        sessions=sorted({str(row["session_id"] or "") for row in rows if row["session_id"]}),
        days=sorted({str(row["created_at"] or "")[:10] for row in rows if row["created_at"]}),
    )


def ready(store: Any, statement: str, *, overlap: int = 2,
          min_episodes: int = MIN_EPISODES, min_sessions: int = MIN_SESSIONS,
          min_days: int = MIN_DAYS) -> Verdict:
    """Would this generalisation be allowed, and why not if not."""
    found = support(store, statement, overlap=overlap)
    reasons: list[str] = []
    if len(found.episodes) < min_episodes:
        reasons.append(f"{len(found.episodes)} episode(s), need {min_episodes}")
    if len(found.sessions) < min_sessions:
        reasons.append(f"{len(found.sessions)} session(s), need {min_sessions} - paraphrases in one "
                       f"conversation are one observation")
    if len(found.days) < min_days:
        reasons.append(f"{len(found.days)} day(s), need {min_days}")
    return Verdict(ready=not reasons, statement=str(statement), support=found, reasons=reasons)


def promote(store: Any, statement: str, *, kind: str = "semantic", confidence: float = 0.7,
            overlap: int = 2, at: datetime | None = None,
            allow_unready: bool = False) -> dict[str, Any]:
    """Write the semantic claim with its citations, or refuse and say what is missing."""
    verdict = ready(store, statement, overlap=overlap)
    if not verdict.ready and not allow_unready:
        return {"promoted": False, "reason": verdict.describe(), **verdict.as_dict()}
    moment = at or _now()
    memory_id = store.add_memory(
        content=str(statement),
        kind=kind,
        source_type="promotion",
        source_category="AGENT_INFERENCE",
        confidence=float(confidence),
        source_context=("promoted from episodes " +
                        ", ".join(str(item) for item in verdict.support.episodes[:10])),
    )[0]
    _ensure(store)
    with store.transaction() as conn:
        for row in episodes_for(store, statement, overlap=overlap):
            conn.execute(
                f"""INSERT OR IGNORE INTO {EVIDENCE}
                    (memory_id, episode_id, session_id, observed_at, created_at)
                    VALUES (?, ?, ?, ?, ?)""",
                (memory_id, int(row["id"]), str(row["session_id"] or ""),
                 str(row["created_at"] or ""), moment.isoformat()),
            )
    return {
        "promoted": True,
        "ready": verdict.ready,
        "memory_id": memory_id,
        "citations": verdict.support.episodes,
        "support": verdict.support.as_dict(),
        "sentence": f"promoted with {len(verdict.support.episodes)} citation(s) across "
                    f"{len(verdict.support.sessions)} session(s)",
    }


def citations(store: Any, memory_id: str) -> list[dict[str, Any]]:
    _ensure(store)
    return _rows(
        store,
        f"""SELECT episode_id, session_id, observed_at FROM {EVIDENCE}
            WHERE memory_id = ? ORDER BY observed_at""",
        (str(memory_id),),
    )


def report(store: Any, *, limit: int = 200) -> dict[str, Any]:
    _ensure(store)
    rows = _rows(
        store,
        f"""SELECT COUNT(*) AS citations, COUNT(DISTINCT memory_id) AS memories FROM {EVIDENCE}""",
    )
    row = rows[0] if rows else {}
    episodes = _rows(store, "SELECT COUNT(*) AS total FROM episodes")
    return {
        "citations": int(row.get("citations") or 0),
        "promoted": int(row.get("memories") or 0),
        "episodes_available": int((episodes[0] if episodes else {}).get("total") or 0),
        "thresholds": {"episodes": MIN_EPISODES, "sessions": MIN_SESSIONS, "days": MIN_DAYS},
        "note": "paraphrases in one conversation are one observation, not three",
    }
