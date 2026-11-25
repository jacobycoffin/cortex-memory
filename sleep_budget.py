"""The ceiling on a night's work, and how it stops without leaving a mess (#110).

A sleep cycle runs unattended: replay, consolidation, summary regeneration, pruning. Unattended work with no
ceiling runs until something else stops it — a timeout, a lock, or the user's morning. A *number* makes it stop
deliberately, and a **graceful** stop finishes the unit in flight instead of abandoning it half-written.

Five rules, each a way a budget makes things worse:

- **the budget is per night and it persists.** A restart at 03:20 must not hand the cycle a fresh 600 seconds:
  the spend belongs to the night, not to the process. The night is the date in UTC, recorded with every row.
- **the soft stop is at ninety percent, and it means "finish, do not start".** A hard stop mid-unit is how a
  half-written summary happens; stopping *new* work at 90% leaves room to complete what is running.
- **a stop is recorded with its reason, and it can be resumed — inside what is left.** `resume()` clears the
  stop only if there is budget remaining; it never resets the spend.
- **a ceiling can be lowered mid-night, never raised.** A budget you can raise when you are close to it is not a
  budget, and the refusal says so.
- **unmeasured work is not zero.** `consume()` with neither seconds nor tokens records that it happened and marks
  the spend as estimated, so a night whose costs nobody measured does not look free.

Nothing here schedules or runs anything; it answers "may I start another unit, and if not, why not".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

BUDGET_TABLE = "sleep_budgets"

DEFAULT_SECONDS = 600
DEFAULT_TOKENS = 200_000
SOFT_STOP = 0.90

REASON_CEILING = "the night's ceiling was reached"
REASON_SOFT = "the soft stop was reached; finish what is running and start nothing new"
REASON_MANUAL = "stopped by the caller"


def _now() -> datetime:
    return datetime.now(timezone.utc)


# The night runs from 06:00 UTC to 05:59 the next day, so a cycle that starts at 23:50 and finishes at 00:20 is
# one night's work rather than two. Getting this wrong the obvious way (plain UTC date) splits every overnight run
# across two budgets and halves both.
NIGHT_START_HOUR = 6


def night_of(moment: datetime | None = None) -> str:
    """The night a moment belongs to: the date of the most recent 06:00 UTC boundary."""
    current = moment or _now()
    return (current - timedelta(hours=NIGHT_START_HOUR)).date().isoformat()


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]] | None:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return None


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {BUDGET_TABLE} (
                    night TEXT PRIMARY KEY,
                    seconds_limit REAL NOT NULL,
                    tokens_limit REAL NOT NULL,
                    spent_seconds REAL NOT NULL DEFAULT 0,
                    spent_tokens REAL NOT NULL DEFAULT 0,
                    estimated INTEGER NOT NULL DEFAULT 0,
                    stopped_at TEXT,
                    stop_reason TEXT,
                    updated_at TEXT NOT NULL
                )"""
        )


@dataclass
class Budget:
    night: str = ""
    seconds_limit: float = DEFAULT_SECONDS
    tokens_limit: float = DEFAULT_TOKENS
    spent_seconds: float = 0.0
    spent_tokens: float = 0.0
    estimated: bool = False
    stop_reason: str = ""
    stopped_at: str = ""

    # -- the two questions a cycle asks ------------------------------------------------

    @property
    def seconds_share(self) -> float:
        return self.spent_seconds / self.seconds_limit if self.seconds_limit else 1.0

    @property
    def tokens_share(self) -> float:
        return self.spent_tokens / self.tokens_limit if self.tokens_limit else 1.0

    @property
    def used_share(self) -> float:
        """The worst of the two: a night is as spent as its most-spent resource."""
        return max(self.seconds_share, self.tokens_share)

    @property
    def exhausted(self) -> bool:
        return self.used_share >= 1.0

    def may_start(self) -> bool:
        """False once the soft stop is reached, or once the cycle has been stopped for any reason."""
        if self.stopped_at:
            return False
        return self.used_share < SOFT_STOP

    def remaining_seconds(self) -> float:
        return max(0.0, self.seconds_limit - self.spent_seconds)

    def statement(self) -> str:
        state = ("stopped: " + self.stop_reason) if self.stopped_at else (
            "soft stop" if not self.may_start() else "running")
        share = f"{self.used_share * 100:.0f}%"
        return (f"night {self.night}: {self.spent_seconds:.0f}s of {self.seconds_limit:.0f}s, "
                f"{self.spent_tokens:.0f} of {self.tokens_limit:.0f} tokens ({share}, {state})"
                + (" [estimated]" if self.estimated else ""))

    def as_dict(self) -> dict[str, Any]:
        return {
            "night": self.night,
            "seconds_limit": self.seconds_limit,
            "tokens_limit": self.tokens_limit,
            "spent_seconds": self.spent_seconds,
            "spent_tokens": self.spent_tokens,
            "estimated": self.estimated,
            "used_share": round(self.used_share, 4),
            "may_start": self.may_start(),
            "stopped_at": self.stopped_at,
            "stop_reason": self.stop_reason,
            "statement": self.statement(),
        }


def _load(store: Any, night: str) -> Budget:
    rows = _rows(store, f"SELECT * FROM {BUDGET_TABLE} WHERE night = ?", (night,))
    if not rows:
        return Budget(night=night)
    row = rows[0]
    return Budget(
        night=str(row["night"]),
        seconds_limit=float(row["seconds_limit"]),
        tokens_limit=float(row["tokens_limit"]),
        spent_seconds=float(row["spent_seconds"]),
        spent_tokens=float(row["spent_tokens"]),
        estimated=bool(row["estimated"]),
        stop_reason=str(row["stop_reason"] or ""),
        stopped_at=str(row["stopped_at"] or ""),
    )


def _save(store: Any, budget: Budget) -> None:
    _ensure(store)
    with store.transaction() as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO {BUDGET_TABLE} (night, seconds_limit, tokens_limit, spent_seconds, "
            "spent_tokens, estimated, stopped_at, stop_reason, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (budget.night, budget.seconds_limit, budget.tokens_limit, budget.spent_seconds,
             budget.spent_tokens, int(budget.estimated), budget.stopped_at or None,
             budget.stop_reason or None, _now().isoformat()),
        )


def open_night(store: Any, *, seconds: int = DEFAULT_SECONDS, tokens: int = DEFAULT_TOKENS,
               at: datetime | None = None) -> Budget:
    """Load tonight's budget, creating it if this is the first call. Existing spend is preserved."""
    night = night_of(at)
    budget = _load(store, night)
    if budget.seconds_limit in (0.0, DEFAULT_SECONDS) and budget.tokens_limit in (0.0, DEFAULT_TOKENS):
        # A fresh night takes the caller's limits; an existing one keeps its own, because a restart must not
        # silently hand the cycle a new ceiling.
        budget.seconds_limit = float(max(0, int(seconds)))
        budget.tokens_limit = float(max(0, int(tokens)))
        _save(store, budget)
    return budget


def consume(store: Any, *, seconds: float = 0.0, tokens: float = 0.0,
            at: datetime | None = None) -> Budget:
    """Record work done. An unmeasured unit is recorded as estimated rather than free."""
    budget = _load(store, night_of(at))
    budget.spent_seconds += max(0.0, float(seconds))
    budget.spent_tokens += max(0.0, float(tokens))
    if seconds <= 0 and tokens <= 0:
        budget.estimated = True
    if budget.exhausted and not budget.stopped_at:
        budget.stop_reason = REASON_CEILING
        budget.stopped_at = _now().isoformat()
    elif not budget.may_start() and not budget.stopped_at:
        budget.stop_reason = REASON_SOFT
        budget.stopped_at = _now().isoformat()
    _save(store, budget)
    return budget


def stop(store: Any, reason: str = REASON_MANUAL, *, at: datetime | None = None) -> Budget:
    """Stop the night deliberately. The reason is kept: 'why did it end at 04:10' must be answerable."""
    budget = _load(store, night_of(at))
    budget.stopped_at = _now().isoformat()
    budget.stop_reason = str(reason or REASON_MANUAL)
    _save(store, budget)
    return budget


def resume(store: Any, *, at: datetime | None = None) -> tuple[Budget, bool]:
    """Clear a stop only if budget remains. Returns (budget, resumed)."""
    budget = _load(store, night_of(at))
    if not budget.stopped_at:
        return budget, True
    if budget.exhausted:
        return budget, False
    budget.stopped_at = ""
    budget.stop_reason = ""
    _save(store, budget)
    return budget, True


def lower(store: Any, *, seconds: int | None = None, tokens: int | None = None,
          at: datetime | None = None) -> tuple[Budget, str]:
    """Lower a ceiling. Raising is refused; the refusal carries the current limit."""
    budget = _load(store, night_of(at))
    refused = ""
    if seconds is not None:
        wanted = float(max(0, int(seconds)))
        if wanted > budget.seconds_limit:
            refused = f"seconds cannot be raised mid-night (now {budget.seconds_limit:.0f}s)"
        else:
            budget.seconds_limit = wanted
    if tokens is not None:
        wanted = float(max(0, int(tokens)))
        if wanted > budget.tokens_limit:
            refused = f"tokens cannot be raised mid-night (now {budget.tokens_limit:.0f})"
        else:
            budget.tokens_limit = wanted
    if not refused:
        _save(store, budget)
        if not budget.may_start() and not budget.stopped_at:
            budget.stop_reason = REASON_SOFT
            budget.stopped_at = _now().isoformat()
            _save(store, budget)
    return budget, refused


def history(store: Any, *, limit: int = 14) -> list[dict[str, Any]]:
    _ensure(store)
    rows = _rows(store, f"SELECT * FROM {BUDGET_TABLE} ORDER BY night DESC LIMIT ?", (int(limit),))
    return rows or []
