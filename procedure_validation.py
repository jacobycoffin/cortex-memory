"""Procedure validation (#115): a candidate procedure must succeed N times before it is offered as guidance.

`procedure_confidence.py` (#142) says how much a procedure is trusted; `argument_memory.py` (#134) keeps the failed
invocations. Neither of them gates anything: a procedure with one lucky run was still available to be recommended, and
the recommendation looked the same as one earned over a month.

This is the gate. A procedure is **offered** only after `REQUIRED_SUCCESSES` **consecutive** successes; everything else
is withheld with the count and the reason.

Six rules:

1. **Consecutive successes, not a total.** A procedure that worked three times and then failed is not validated - the
   failure is the newer evidence, and summing successes lets a broken procedure inherit its own history.
2. **N is stated in every verdict.** "withheld, 2 of 3" reads as progress; "withheld" alone reads as breakage.
3. **No evidence is its own verdict** (`unvalidated`), never silently "offered" and never "broken".
4. **The last failure travels with the verdict**, because *why* it is withheld is the actionable part.
5. **Reading never creates schema** - a store that has never recorded a run has no table, and that is not an error.
6. **The gate is a read.** It decides what may be *offered*; it never writes, and it never retires a procedure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from . import procedure_confidence

# Three in a row is the bar: high enough that one accidental success cannot carry a procedure, low enough that a
# useful procedure is not hidden for weeks. The number is printed in every verdict so it is never a mystery.
REQUIRED_SUCCESSES = 3

OFFERED = "offered"
WITHHELD = "withheld"
UNVALIDATED = "unvalidated"
UNREADABLE = "unreadable"


@dataclass(frozen=True)
class Verdict:
    procedure_id: str
    verdict: str
    required: int = REQUIRED_SUCCESSES
    streak: int = 0
    successes: int = 0
    failures: int = 0
    last_failure_at: str = ""
    last_failure_note: str = ""
    reason: str = ""

    @property
    def may_offer(self) -> bool:
        return self.verdict == OFFERED

    def as_dict(self) -> dict[str, Any]:
        return {
            "procedure_id": self.procedure_id,
            "verdict": self.verdict,
            "required": self.required,
            "streak": self.streak,
            "successes": self.successes,
            "failures": self.failures,
            "last_failure_at": self.last_failure_at,
            "last_failure_note": self.last_failure_note,
            "reason": self.reason,
        }

    def render(self) -> str:
        if self.verdict == OFFERED:
            return f"{self.procedure_id}: offered ({self.streak} consecutive successes, {self.failures} failure(s) older)"
        if self.verdict == WITHHELD:
            return f"{self.procedure_id}: withheld — {self.reason}"
        if self.verdict == UNVALIDATED:
            return f"{self.procedure_id}: unvalidated — no recorded runs (needs {self.required} in a row)"
        return f"{self.procedure_id}: unreadable — {self.reason}"


@dataclass
class Gate:
    verdicts: list[Verdict] = field(default_factory=list)

    @property
    def offered(self) -> list[Verdict]:
        return [item for item in self.verdicts if item.verdict == OFFERED]

    @property
    def withheld(self) -> list[Verdict]:
        return [item for item in self.verdicts if item.verdict == WITHHELD]

    def render(self) -> list[str]:
        lines = [f"{len(self.verdicts)} procedure(s): {len(self.offered)} offered, "
                 f"{len(self.withheld)} withheld, "
                 f"{len([v for v in self.verdicts if v.verdict == UNVALIDATED])} unvalidated"]
        for verdict in self.verdicts[:10]:
            lines.append(f"  - {verdict.render()}")
        if len(self.verdicts) > 10:
            lines.append(f"  … and {len(self.verdicts) - 10} more")
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {"verdicts": [verdict.as_dict() for verdict in self.verdicts],
                "offered": len(self.offered), "withheld": len(self.withheld),
                "required": REQUIRED_SUCCESSES}


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]] | None:
    try:
        cursor = store._conn.execute(sql, params)
    except Exception:
        return None
    try:
        return [dict(row) for row in cursor.fetchall()]
    except Exception:
        return None


def validate(store: Any, procedure_id: str, *, required: int = REQUIRED_SUCCESSES) -> Verdict:
    """Decide whether a procedure may be offered. Reads only."""
    pid = str(procedure_id or "").strip()
    bar = max(1, int(required))
    if not pid:
        return Verdict(procedure_id="", verdict=UNVALIDATED, required=bar,
                       reason="a procedure needs an id")
    # A store that has never recorded a run has no table at all. That is "no evidence", not "unreadable" - the two
    # must not look the same, or a fresh install reads as a broken one.
    exists = _rows(store, "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                   (procedure_confidence.OUTCOME_TABLE,))
    if exists is None:
        return Verdict(procedure_id=pid, verdict=UNREADABLE, required=bar,
                       reason="the store refused the schema query")
    if not exists:
        return Verdict(procedure_id=pid, verdict=UNVALIDATED, required=bar)
    rows = _rows(
        store,
        f"SELECT outcome, at, note FROM {procedure_confidence.OUTCOME_TABLE} "
        "WHERE procedure_id = ? ORDER BY at",
        (pid,),
    )
    if rows is None:
        return Verdict(procedure_id=pid, verdict=UNREADABLE, required=bar,
                       reason="the store refused the outcome query")
    if not rows:
        return Verdict(procedure_id=pid, verdict=UNVALIDATED, required=bar)

    successes = sum(1 for row in rows if row.get("outcome") == procedure_confidence.SUCCESS)
    failures = sum(1 for row in rows if row.get("outcome") == procedure_confidence.FAILURE)
    # The streak is counted backwards from the newest run: anything before the last failure is history, not evidence.
    streak = 0
    last_failure_at = ""
    last_failure_note = ""
    for row in reversed(rows):
        if row.get("outcome") == procedure_confidence.SUCCESS:
            streak += 1
            continue
        last_failure_at = str(row.get("at") or "")
        last_failure_note = str(row.get("note") or "")
        break
    if streak >= bar:
        return Verdict(procedure_id=pid, verdict=OFFERED, required=bar, streak=streak,
                       successes=successes, failures=failures,
                       last_failure_at=last_failure_at, last_failure_note=last_failure_note)
    reason = f"{streak} of {bar} consecutive successes"
    if last_failure_at:
        reason += f"; last failure {last_failure_at}"
        if last_failure_note:
            reason += f" ({last_failure_note})"
    return Verdict(procedure_id=pid, verdict=WITHHELD, required=bar, streak=streak,
                   successes=successes, failures=failures,
                   last_failure_at=last_failure_at, last_failure_note=last_failure_note, reason=reason)


def gate(store: Any, procedure_ids: Iterable[str], *, required: int = REQUIRED_SUCCESSES) -> Gate:
    """Validate a batch, preserving the caller's order and dropping duplicates."""
    seen: set[str] = set()
    verdicts: list[Verdict] = []
    for procedure_id in procedure_ids:
        pid = str(procedure_id or "").strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        verdicts.append(validate(store, pid, required=required))
    return Gate(verdicts=verdicts)


def known_procedures(store: Any, *, limit: int = 200) -> list[str]:
    """Every procedure the store has evidence about, most recently run first."""
    rows = _rows(
        store,
        f"SELECT procedure_id, MAX(at) AS latest FROM {procedure_confidence.OUTCOME_TABLE} "
        "GROUP BY procedure_id ORDER BY latest DESC LIMIT ?",
        (max(1, int(limit)),),
    )
    if not rows:
        return []
    return [str(row["procedure_id"]) for row in rows]
