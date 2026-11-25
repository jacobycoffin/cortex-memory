"""Which memories to replay tonight, and why each one is on the list (#107).

Retrieval only touches what a query happens to surface, so a memory that is never asked about is never revisited:
its confidence does not decay, its contradictions are not noticed, and its facts are not re-derived. A nightly
replay is the only mechanism that reaches the long tail — but "replay everything" is just a full scan, and a full
scan at 8,371 memories is not a night's work.

So the schedule is a *ranked, bounded* list with a stated reason per entry:

- **the budget is a number and it is printed.** `per_night` either fits or the schedule says what did not fit. A
  replay plan with no ceiling is a wish.
- **frequency is not value.** A memory retrieved 131 times is already being tested every day; the ones that need
  replay are the ones nothing asks about — so a high use count *lowers* replay priority, and "never used" is the
  single strongest signal.
- **regret outranks everything.** A memory that was pruned and later needed is the definition of a wrong call;
  those are scheduled first, with the regret row cited.
- **only replayable states are scheduled.** `archived`, `quarantined` and `tombstoned` are decisions, not
  oversights; replaying them would churn against the operator.
- **the order is deterministic.** Ties break on the memory id, so tonight's schedule and tomorrow's are comparable
  and a replay that ran can be checked against the plan that produced it.

Read-only: `schedule()` computes, it does not write. Nothing here promotes, refreshes or edits a memory; it answers
"what should tonight look like" for the job that does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

MEMORY_TABLE = "memories"
REGRET_TABLE = "pruning_regret"

DEFAULT_PER_NIGHT = 50
# A memory untouched for a month is stale enough that a replay tells us something; below this it was probably
# exercised by ordinary use and is not worth the budget.
STALE_DAYS = 30
REPLAYABLE_STATES = ("active", "cold")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]] | None:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return None


def _age_days(stamp: Any, *, now: datetime) -> float | None:
    if not stamp:
        return None
    try:
        moment = datetime.fromisoformat(str(stamp))
    except Exception:  # noqa: BLE001
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return max(0.0, (now - moment).total_seconds() / 86400.0)


@dataclass
class Entry:
    memory_id: str
    score: float
    reason: str
    signals: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"memory_id": self.memory_id, "score": round(self.score, 4),
                "reason": self.reason, "signals": self.signals}

    def render(self) -> str:
        return f"{self.memory_id[:8]} {self.score:.2f} — {self.reason}"


@dataclass
class Schedule:
    per_night: int = DEFAULT_PER_NIGHT
    readable: bool = True
    reason: str = ""
    entries: list[Entry] = field(default_factory=list)
    candidates: int = 0
    deferred: int = 0
    excluded: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "per_night": self.per_night,
            "readable": self.readable,
            "reason": self.reason,
            "candidates": self.candidates,
            "deferred": self.deferred,
            "excluded": self.excluded,
            "entries": [entry.as_dict() for entry in self.entries],
        }

    def render(self) -> list[str]:
        if not self.readable:
            return [self.reason]
        if not self.entries:
            return [f"nothing to replay tonight ({self.candidates} candidate(s) considered)"]
        lines = [f"{len(self.entries)} of {self.candidates} candidate(s), budget {self.per_night}"
                 f" ({self.deferred} deferred)"]
        lines.extend(f"- {entry.render()}" for entry in self.entries)
        return lines


def schedule(store: Any, *, per_night: int = DEFAULT_PER_NIGHT, now: datetime | None = None,
             stale_days: int = STALE_DAYS) -> Schedule:
    """Rank what tonight should replay. Never writes."""
    budget = max(0, int(per_night))
    moment = now or _now()
    out = Schedule(per_night=budget)

    rows = _rows(
        store,
        f"SELECT id, state, kind, used_count, retrieved_count, importance, confidence, "
        f"last_used_at, last_retrieved_at, created_at FROM {MEMORY_TABLE}",
    )
    if rows is None:
        out.readable = False
        out.reason = "the store could not be read"
        return out
    regret_rows = _rows(store, f"SELECT memory_id, score, created_at FROM {REGRET_TABLE}")
    if regret_rows is None:
        out.excluded["regret table unreadable"] = 0
        regret_rows = []
    regret = {}
    for row in regret_rows:
        key = str(row.get("memory_id"))
        regret[key] = regret.get(key, 0.0) + float(row.get("score") or 0.0)

    excluded: dict[str, int] = {}
    scored: list[Entry] = []
    for row in rows:
        state = str(row.get("state") or "")
        if state not in REPLAYABLE_STATES:
            excluded[state or "unknown state"] = excluded.get(state or "unknown state", 0) + 1
            continue
        memory_id = str(row.get("id"))
        used = int(row.get("used_count") or 0)
        retrieved = int(row.get("retrieved_count") or 0)
        age = _age_days(row.get("last_used_at") or row.get("last_retrieved_at") or row.get("created_at"),
                        now=moment)
        # Frequency is the *absence* of need: something tested 131 times is being exercised every day.
        unexercised = 1.0 / (1.0 + used + 0.25 * retrieved)
        staleness = 0.0 if age is None else min(1.0, age / max(1, stale_days))
        regret_score = min(3.0, regret.get(memory_id, 0.0))
        importance = float(row.get("importance") or 0.0)
        confidence = float(row.get("confidence") or 0.0)

        score = (2.0 * regret_score) + (1.5 * unexercised) + (1.0 * staleness) + (
            0.5 * importance) + (0.5 * (1.0 - confidence))
        if score <= 0.0:
            continue
        signals = {"regret": round(regret_score, 3), "unexercised": round(unexercised, 3),
                   "staleness": round(staleness, 3), "importance": round(importance, 3),
                   "uncertainty": round(1.0 - confidence, 3)}
        dominant = max(signals, key=lambda key: signals[key])
        scored.append(Entry(memory_id=memory_id, score=score,
                            reason=f"{dominant} ({signals[dominant]})", signals=signals))

    scored.sort(key=lambda entry: (-entry.score, entry.memory_id))
    out.candidates = len(scored)
    out.entries = scored[:budget]
    out.deferred = max(0, len(scored) - budget)
    out.excluded = excluded
    return out


def report(store: Any, *, per_night: int = DEFAULT_PER_NIGHT, now: datetime | None = None) -> dict[str, Any]:
    """The numbers behind a schedule, for a log line rather than a list."""
    plan = schedule(store, per_night=per_night, now=now)
    if not plan.readable:
        return {"readable": False, "reason": plan.reason}
    regret_first = sum(1 for entry in plan.entries if entry.signals.get("regret", 0) > 0)
    return {
        "readable": True,
        "per_night": plan.per_night,
        "candidates": plan.candidates,
        "scheduled": len(plan.entries),
        "deferred": plan.deferred,
        "with_regret": regret_first,
        "excluded": plan.excluded,
        "never_used_scheduled": sum(1 for entry in plan.entries if entry.signals.get("unexercised", 0) >= 0.9),
    }
