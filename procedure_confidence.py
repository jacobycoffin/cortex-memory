"""How much a procedure has earned (#142).

A procedure is a claim that doing X gets you Y. Some have been run forty times without a miss; some were written
once and never tried; some failed the last two times and nobody noticed. Treating all three as equally trustworthy is
how a store of procedures rots — the bad ones stay in rotation because nothing in the record distinguishes them.

The number here is deliberately pessimistic and fully explained:

    confidence = smoothed_success x evidence_weight x recency_decay

- **smoothed_success** = `(successes + 1) / (successes + 1.5*failures + 2)`. Laplace smoothing so one lucky run is
  not 100%, and failures weigh **1.5x** because a failure is more informative than a success: it tells you the
  procedure is wrong *somewhere*, while a success only tells you it worked that time.
- **evidence_weight** = `min(1, outcomes / 4)`. Below four outcomes a procedure is a hypothesis, and its confidence is
  scaled down rather than left to float on one good day.
- **recency_decay** = `0.5 ** (idle_days / 45)`. An untested procedure is not a proven one; a year idle leaves ~0.4%
  of the claim standing.

Five rules:

- **retrieval is not evidence.** Reading a procedure, or having it injected, does not raise its confidence. Only
  recorded outcomes do — the same discipline as "a memory never becomes true by being retrieved".
- **the number never travels alone.** Every score reports successes, failures and idle days next to it.
- **a failed run is recorded as a failure, not as silence.** Silence is what makes a broken procedure look fresh.
- **decay is a function of idleness, not of wall time.** Two runs today and none for a year is not the same as two
  runs spread over a year.
- **the same outcome at the same moment is idempotent** — a retry that reports the same event must not double-count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

OUTCOME_TABLE = "procedure_outcomes"

SUCCESS = "success"
FAILURE = "failure"
OUTCOMES = (SUCCESS, FAILURE)

FAILURE_WEIGHT = 1.5
EVIDENCE_FULL_AT = 4
HALF_LIFE_DAYS = 45.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]] | None:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return None


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {OUTCOME_TABLE} (
                    procedure_id TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    at TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (procedure_id, outcome, at)
                )"""
        )


@dataclass
class Confidence:
    procedure_id: str
    successes: int = 0
    failures: int = 0
    idle_days: float = 0.0
    confidence: float = 0.0
    verdict: str = "untested"

    def parts(self) -> dict[str, float]:
        smoothed = (self.successes + 1.0) / (self.successes + FAILURE_WEIGHT * self.failures + 2.0)
        evidence = min(1.0, (self.successes + self.failures) / EVIDENCE_FULL_AT)
        decay = 0.5 ** (self.idle_days / HALF_LIFE_DAYS)
        return {"smoothed_success": round(smoothed, 3), "evidence": round(evidence, 3),
                "decay": round(decay, 3)}

    def render(self) -> str:
        return (f"{self.procedure_id}: {self.confidence:.2f} {self.verdict} "
                f"({self.successes} ok, {self.failures} failed, {self.idle_days:.0f}d idle)")

    def as_dict(self) -> dict[str, Any]:
        return {"procedure_id": self.procedure_id, "confidence": round(self.confidence, 4),
                "verdict": self.verdict, "successes": self.successes, "failures": self.failures,
                "idle_days": round(self.idle_days, 1), "parts": self.parts()}


def record(store: Any, procedure_id: str, outcome: str, *, at: datetime | None = None,
           note: str = "") -> dict[str, Any]:
    """Record one run. Idempotent on (procedure, outcome, moment)."""
    pid = str(procedure_id or "").strip()
    if not pid:
        return {"ok": False, "reason": "a procedure id is required"}
    if outcome not in OUTCOMES:
        return {"ok": False, "reason": f"outcome must be one of {', '.join(OUTCOMES)}"}
    moment = (at or _now()).replace(microsecond=0).isoformat()
    _ensure(store)
    with store.transaction() as conn:
        conn.execute(
            f"INSERT OR IGNORE INTO {OUTCOME_TABLE} (procedure_id, outcome, at, note) VALUES (?, ?, ?, ?)",
            (pid, outcome, moment, str(note or "")),
        )
    return {"ok": True, "procedure_id": pid, "outcome": outcome, "at": moment}


def _table_exists(store: Any) -> bool | None:
    """True/False when the store answered, None when it could not be asked at all."""
    rows = _rows(store, "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (OUTCOME_TABLE,))
    if rows is None:
        return None
    return bool(rows)


def confidence(store: Any, procedure_id: str, *, now: datetime | None = None) -> Confidence:
    pid = str(procedure_id or "").strip()
    # Reading never creates schema: a store that has never recorded a run has no table, and that is "untested",
    # not "unreadable" — the two must not look the same.
    exists = _table_exists(store)
    if exists is None:
        return Confidence(procedure_id=pid, verdict="unreadable")
    if not exists:
        return Confidence(procedure_id=pid, verdict="untested")
    rows = _rows(store, f"SELECT outcome, at FROM {OUTCOME_TABLE} WHERE procedure_id = ?", (pid,))
    if rows is None:
        return Confidence(procedure_id=pid, verdict="unreadable")
    moment = now or _now()
    successes = sum(1 for row in rows if row["outcome"] == SUCCESS)
    failures = sum(1 for row in rows if row["outcome"] == FAILURE)
    last: datetime | None = None
    for row in rows:
        try:
            stamp = datetime.fromisoformat(str(row["at"]))
        except Exception:  # noqa: BLE001
            continue
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        if last is None or stamp > last:
            last = stamp
    idle = 0.0 if last is None else max(0.0, (moment - last).total_seconds() / 86400.0)
    out = Confidence(procedure_id=pid, successes=successes, failures=failures, idle_days=idle)
    parts = out.parts()
    out.confidence = round(max(0.0, min(1.0, parts["smoothed_success"] * parts["evidence"] * parts["decay"])), 4)
    if successes + failures == 0:
        out.verdict = "untested"
    elif failures > successes:
        out.verdict = "suspect"
    elif out.confidence >= 0.6:
        out.verdict = "trusted"
    elif idle > 180:
        out.verdict = "stale"
    else:
        out.verdict = "learning"
    return out


def rank(store: Any, *, now: datetime | None = None, limit: int = 20) -> list[Confidence]:
    """Every procedure that has a record, most trustworthy first. Nothing is invented for the rest."""
    if _table_exists(store) is not True:
        return []
    rows = _rows(store, f"SELECT DISTINCT procedure_id FROM {OUTCOME_TABLE}")
    if rows is None:
        return []
    scored = [confidence(store, str(row["procedure_id"]), now=now) for row in rows]
    scored.sort(key=lambda item: (-item.confidence, item.procedure_id))
    return scored[: max(0, int(limit))]


def render(scored: list[Confidence]) -> list[str]:
    if not scored:
        return ["No procedure has been run yet."]
    lines = [f"{len(scored)} procedure(s) with a record:"]
    lines.extend(f"- {item.render()}" for item in scored)
    return lines


def idle_days_cutoff(days: int, *, now: datetime | None = None) -> str:
    """ISO timestamp `days` before now, for callers that want to query idleness directly."""
    return ((now or _now()) - timedelta(days=max(0, int(days)))).isoformat()
