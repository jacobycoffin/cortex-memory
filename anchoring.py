"""Anchor relative time at write, because "yesterday" expires (#40).

*"I'm on call yesterday"* is meaningless a week later, and *"last Tuesday the deploy broke"* is
worse than meaningless — it is confidently wrong, because every reader resolves it against their own
Tuesday. So the relative phrase is kept **and** pinned to a date when the memory is written:

    last Tuesday the deploy broke
    last Tuesday (2026-11-17) the deploy broke

The phrase stays on purpose. Rewriting it to a bare date would lose how the user said it, and the
quote is what makes the memory readable to a person; the parenthetical is what makes it correct for
a machine.

Two rules keep this honest:

- **only resolve when the resolution is unambiguous** — "last Tuesday" said on a Wednesday is the
  Tuesday six days back, a real date; "the other week" is not, and is left untouched;
- **idempotent** — an already-anchored phrase is not re-anchored, so writing the same text twice
  cannot produce `(2026-11-17) (2026-11-17)`.

This is not `temporal.py`: that module classifies whether a record's *values* are point-in-time. This
one fixes the *reference frame* of a phrase. A record can need either, both, or neither.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_ALREADY = re.compile(r"\((\d{4}-\d{2}-\d{2})\)")
_WEEKDAY = "|".join(WEEKDAYS)
_MONTH = "|".join(sorted(MONTHS, key=len, reverse=True))

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bthe day before yesterday\b", re.IGNORECASE), "d-2"),
    (re.compile(r"\byesterday\b", re.IGNORECASE), "d-1"),
    (re.compile(r"\btoday\b", re.IGNORECASE), "d0"),
    (re.compile(r"\btomorrow\b", re.IGNORECASE), "d+1"),
    (re.compile(r"\blast week\b", re.IGNORECASE), "w-1"),
    (re.compile(r"\bnext week\b", re.IGNORECASE), "w+1"),
    (re.compile(r"\blast month\b", re.IGNORECASE), "m-1"),
    (re.compile(r"\bnext month\b", re.IGNORECASE), "m+1"),
    (re.compile(r"\blast year\b", re.IGNORECASE), "y-1"),
    (re.compile(r"\b(\d{1,2})\s+days?\s+ago\b", re.IGNORECASE), "n-"),
    (re.compile(r"\bin\s+(\d{1,2})\s+days?\b", re.IGNORECASE), "n+"),
    (re.compile(rf"\blast\s+({_WEEKDAY})\b", re.IGNORECASE), "last-weekday"),
    (re.compile(rf"\bnext\s+({_WEEKDAY})\b", re.IGNORECASE), "next-weekday"),
    (re.compile(rf"\b({_MONTH})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.IGNORECASE), "month-day"),
)


@dataclass
class Anchor:
    text: str
    resolved: list[tuple[str, str]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.resolved)

    def describe(self) -> str:
        if not self.resolved:
            return "nothing to anchor"
        return ", ".join(f"{phrase} -> {value}" for phrase, value in self.resolved)


def _as_date(at: datetime | date | None) -> date:
    if isinstance(at, datetime):
        return at.astimezone(timezone.utc).date() if at.tzinfo else at.date()
    if isinstance(at, date):
        return at
    return datetime.now(timezone.utc).date()


def _shift_months(day: date, months: int) -> date:
    month_index = day.month - 1 + months
    year = day.year + month_index // 12
    month = month_index % 12 + 1
    for candidate in (day.day, 28, 29, 30, 31):
        try:
            return date(year, month, candidate)
        except ValueError:
            continue
    return day


def resolve(kind: str, match: re.Match[str], at: date) -> date | None:
    """Turn one matched phrase into a date. None means "not unambiguous"."""
    if kind == "d0":
        return at
    if kind == "d-1":
        return at - timedelta(days=1)
    if kind == "d-2":
        return at - timedelta(days=2)
    if kind == "d+1":
        return at + timedelta(days=1)
    if kind == "w-1":
        return at - timedelta(days=at.weekday() + 7)
    if kind == "w+1":
        return at - timedelta(days=at.weekday()) + timedelta(days=7)
    if kind == "m-1":
        return _shift_months(at, -1)
    if kind == "m+1":
        return _shift_months(at, 1)
    if kind == "y-1":
        try:
            return at.replace(year=at.year - 1)
        except ValueError:  # 29 February
            return at.replace(year=at.year - 1, day=28)
    if kind == "n-":
        return at - timedelta(days=int(match.group(1)))
    if kind == "n+":
        return at + timedelta(days=int(match.group(1)))
    if kind == "last-weekday":
        target = WEEKDAYS[str(match.group(1)).lower()]
        back = (at.weekday() - target) % 7 or 7
        return at - timedelta(days=back)
    if kind == "next-weekday":
        target = WEEKDAYS[str(match.group(1)).lower()]
        ahead = (target - at.weekday()) % 7 or 7
        return at + timedelta(days=ahead)
    if kind == "month-day":
        month = MONTHS[str(match.group(1)).lower()]
        day = int(match.group(2))
        year = at.year if (month, day) <= (at.month, at.day) else at.year - 1
        try:
            return date(year, month, day)
        except ValueError:
            return None
    return None


def anchor(text: str, *, at: datetime | date | None = None) -> Anchor:
    """Append an absolute date after each relative phrase that resolves cleanly."""
    original = str(text or "")
    today = _as_date(at)
    result = original
    resolved: list[tuple[str, str]] = []

    for pattern, kind in _PATTERNS:
        def replace(match: re.Match[str]) -> str:
            tail = result[match.end():match.end() + 16].lstrip()
            if _ALREADY.match(tail):
                return match.group(0)  # already anchored: leave it alone
            when = resolve(kind, match, today)
            if when is None:
                return match.group(0)
            resolved.append((match.group(0), when.isoformat()))
            return f"{match.group(0)} ({when.isoformat()})"

        result = pattern.sub(replace, result)
    return Anchor(result, resolved)


def anchored(text: str, *, at: datetime | date | None = None) -> str:
    return anchor(text, at=at).text


def needs_anchor(text: str) -> bool:
    return bool(anchor(text).changed)


def phrases(text: str) -> list[str]:
    return [phrase for phrase, _value in anchor(text).resolved]
