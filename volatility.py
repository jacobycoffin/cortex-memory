"""Volatility at write: durable, seasonal or ephemeral (#31).

A fact's kind says what it is about; volatility says how fast it goes stale, and the two are
independent. "The dashboard runs on port 8100" is a `semantic` memory that will be true for years.
"On call this week" is also semantic and is wrong on Monday.

Getting this wrong is expensive in both directions: treating an ephemeral fact as durable leaves the
store confidently asserting last month's rota, and treating a durable fact as ephemeral deletes the
port number. So volatility is decided **at write**, from evidence in the text rather than from the
kind alone, and the ladder uses it as a window multiplier.

Precedence matters more than any single signal:

1. a **durable marker** (identity, preference, procedure, "always", "never", a credential, an
   address) wins — it is cheaper to keep a stale fact than to lose a stable one;
2. then an **ephemeral marker** ("right now", "today", "this week", "temporarily");
3. then **seasonal** ("during the school year", "in winter", "every summer");
4. otherwise the kind decides, and an unremarkable semantic fact is durable.

Ephemeral shortens the window, durable lengthens it, seasonal barely moves it — all three are
multipliers on the declared policy, so #87's windows stay the authority and this only tunes them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

DURABLE = "durable"
SEASONAL = "seasonal"
EPHEMERAL = "ephemeral"
CLASSES = (DURABLE, SEASONAL, EPHEMERAL)

MULTIPLIERS = {EPHEMERAL: 0.5, SEASONAL: 0.75, DURABLE: 1.5}
_TABLE = "memory_volatility"

# Kinds whose own meaning implies volatility, used only when no marker fires.
_KIND_DEFAULT = {
    "episode": EPHEMERAL,
    "episodic": EPHEMERAL,
    "tool": SEASONAL,
    "prospective": EPHEMERAL,
}

# Kinds that are durable by definition: an ephemeral marker must not shorten them below this.
_NEVER_EPHEMERAL_KINDS = frozenset(
    {"identity", "preference", "procedure", "procedural", "commitment", "correction"}
)

_DURABLE_PATTERNS = (
    r"\balways\b",
    r"\bnever\b",
    r"\bprefers?\b",
    r"\bpassword\b",
    r"\bapi key\b",
    r"\bport \d+",
    r"\bruns? on\b",
    r"\blives? at\b",
    r"\binstalled at\b",
    r"\bdefault is\b",
    r"\bconvention\b",
    r"\brequires?\b",
)

_EPHEMERAL_PATTERNS = (
    r"\bright now\b",
    r"\bat the moment\b",
    r"\btoday\b",
    r"\btonight\b",
    r"\bthis (?:week|morning|afternoon|evening|weekend)\b",
    r"\btemporar(?:y|ily)\b",
    r"\bfor now\b",
    r"\bcurrently\b",
    r"\bon call\b",
    r"\bin progress\b",
    r"\buntil (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow)\b",
    r"\bjust (?:now|finished|started)\b",
    r"\bthis session\b",
)

_SEASONAL_PATTERNS = (
    r"\bschool year\b",
    r"\bsemester\b",
    r"\bin (?:winter|spring|summer|autumn|fall)\b",
    r"\bevery (?:summer|winter|spring|fall|autumn|month|year)\b",
    r"\bseason(?:al|ally)?\b",
    r"\bholiday(?:s)?\b",
    r"\bduring the week\b",
    r"\bweekends? only\b",
    r"\bquarter(?:ly)?\b",
)


def _hits(text: str, patterns: tuple[str, ...]) -> list[str]:
    return [match.group(0) for pattern in patterns
            for match in [re.search(pattern, text, re.IGNORECASE)] if match]


@dataclass
class Assessment:
    volatility: str
    reason: str
    evidence: tuple[str, ...]

    def describe(self) -> str:
        evidence = ", ".join(self.evidence[:3])
        return f"{self.volatility} ({self.reason}{': ' + evidence if evidence else ''})"


def classify(content: str, kind: str | None = None) -> Assessment:
    """Decide durability from the text first, the kind second."""
    text = str(content or "")
    lowered = text.lower()
    kind_name = str(kind or "").strip().lower()

    durable = _hits(text, _DURABLE_PATTERNS)
    if durable and kind_name not in _NEVER_EPHEMERAL_KINDS:
        # A durable marker outranks an ephemeral one: losing a stable fact costs more.
        return Assessment(DURABLE, "durable marker", tuple(durable))
    if durable:
        return Assessment(DURABLE, "durable kind and marker", tuple(durable))

    ephemeral = _hits(text, _EPHEMERAL_PATTERNS)
    if ephemeral and kind_name not in _NEVER_EPHEMERAL_KINDS:
        return Assessment(EPHEMERAL, "time-bound marker", tuple(ephemeral))
    if ephemeral:
        return Assessment(SEASONAL, "ephemeral marker on a durable kind", tuple(ephemeral))

    seasonal = _hits(text, _SEASONAL_PATTERNS)
    if seasonal:
        return Assessment(SEASONAL, "recurring marker", tuple(seasonal))

    if kind_name in _NEVER_EPHEMERAL_KINDS:
        return Assessment(DURABLE, "durable kind", ())
    if kind_name in _KIND_DEFAULT:
        return Assessment(_KIND_DEFAULT[kind_name], "kind default", (kind_name,))
    return Assessment(DURABLE, "no volatility marker", ())


def multiplier(volatility: str) -> float:
    return MULTIPLIERS.get(str(volatility).strip().lower(), 1.0)


def apply_to(dwell: tuple[int, int, int], volatility: str) -> tuple[int, int, int]:
    """Scale a declared window triple. Never returns a zero-day window."""
    scale = multiplier(volatility)
    if scale == 1.0:
        return dwell
    # Floor, not round: 30 * 0.75 is 22.5, and "round half to even" would make the window depend
    # on which side of .5 a number happens to land. Truncation is dull and predictable.
    return tuple(max(1, int(days * scale)) for days in dwell)  # type: ignore[return-value]


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {_TABLE} (
                    memory_id TEXT PRIMARY KEY,
                    volatility TEXT NOT NULL,
                    reason TEXT,
                    evidence TEXT,
                    assigned_at TEXT NOT NULL
                )"""
        )


def assign(
    store: Any,
    memory_id: str,
    content: str,
    *,
    kind: str | None = None,
    at: datetime | None = None,
) -> Assessment:
    """Classify and persist. Idempotent per memory id."""
    _ensure(store)
    assessment = classify(content, kind)
    stamp = (at or datetime.now(timezone.utc)).isoformat()
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT INTO {_TABLE} (memory_id, volatility, reason, evidence, assigned_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    volatility = excluded.volatility,
                    reason = excluded.reason,
                    evidence = excluded.evidence,
                    assigned_at = excluded.assigned_at""",
            (str(memory_id), assessment.volatility, assessment.reason,
             ", ".join(assessment.evidence[:3]), stamp),
        )
    return assessment


def lookup(store: Any, memory_id: str) -> str | None:
    try:
        row = store._conn.execute(
            f"SELECT volatility FROM {_TABLE} WHERE memory_id = ?", (str(memory_id),)
        ).fetchone()
    except Exception:  # noqa: BLE001 - a missing table must not break a sweep
        return None
    return str(row["volatility"]) if row else None


def report(store: Any) -> dict[str, Any]:
    _ensure(store)
    rows = store._conn.execute(
        f"SELECT volatility, COUNT(*) AS n FROM {_TABLE} GROUP BY volatility"
    ).fetchall()
    counts = {str(row["volatility"]): int(row["n"]) for row in rows}
    return {
        "assigned": sum(counts.values()),
        "counts": {name: counts.get(name, 0) for name in CLASSES},
        "multipliers": dict(MULTIPLIERS),
    }
