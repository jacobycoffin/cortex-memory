"""Forget what you learned in a period, with a preview (#96).

"Forget what you learned in March" is a request about *time of learning*, not about content, so the
selector is the observation date — `observed_at` with `created_at` as the fallback — and nothing
else. Content is deliberately not consulted: a March conversation that mentions the greenhouse is
not "what you learned in March" any less than one that does not.

The period parser accepts the shapes a person actually types and refuses everything else loudly:

    2026-03                  the calendar month
    2026-03-01..2026-03-31   an explicit range, both ends inclusive
    2026-03-01               that single day

Everything else — "March", "last month", a typo — raises rather than guessing, because a forgotten
memory is not a moot point. `parse_range` exists separately from `preview` so a caller can validate
a spec without touching the store.

Like the other forget paths this reuses the envelope: preview, read the list, then apply by exact
id. Held records block the whole request even with force (#103).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from . import forget_preview

_MONTH = re.compile(r"^(\d{4})-(\d{2})$")
_DAY = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


class RangeError(ValueError):
    """Raised when a period spec cannot be understood. Never guessed around."""


def parse_range(spec: str, *, now: datetime | None = None) -> tuple[datetime, datetime]:
    """Turn a period spec into an inclusive `(start, end)` pair."""
    raw = str(spec or "").strip()
    if not raw:
        raise RangeError("a period is required, e.g. 2026-03 or 2026-03-01..2026-03-31")
    if ".." in raw:
        left, _, right = raw.partition("..")
        start = _day(left.strip())
        end = _day(right.strip()) + timedelta(days=1)
        if end <= start:
            raise RangeError(f"period ends before it starts: {raw}")
        return _as_utc(start), _as_utc(end)
    month = _MONTH.match(raw)
    if month:
        year, month_number = int(month.group(1)), int(month.group(2))
        if not 1 <= month_number <= 12:
            raise RangeError(f"not a month: {raw}")
        start = datetime(year, month_number, 1, tzinfo=timezone.utc)
        end = datetime(year + (month_number == 12), month_number % 12 + 1, 1,
                       tzinfo=timezone.utc)
        return start, end
    if _DAY.match(raw):
        start = _day(raw)
        return _as_utc(start), _as_utc(start + timedelta(days=1))
    raise RangeError(
        f"unrecognised period '{raw}'; use 2026-03, 2026-03-01 or 2026-03-01..2026-03-31"
    )


def _day(text: str) -> datetime:
    match = _DAY.match(text)
    if not match:
        raise RangeError(f"expected YYYY-MM-DD, got '{text}'")
    year, month, day = (int(part) for part in match.groups())
    try:
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError as error:
        raise RangeError(f"not a real date: {text} ({error})") from error


def _as_utc(stamp: datetime) -> datetime:
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


@dataclass
class RangePreview:
    spec: str
    start: str
    end: str
    envelope: str
    matches: list[dict[str, Any]]

    @property
    def ids(self) -> list[str]:
        return [str(match["memory_id"]) for match in self.matches]

    def describe(self, snippet: int = 100) -> str:
        lines = [
            f"learned {self.start[:10]} .. {self.end[:10]}: {len(self.matches)} memories, "
            f"envelope {self.envelope[:12]}"
        ]
        for match in self.matches:
            content = " ".join(str(match.get("content") or "").split())
            lines.append(
                f"  {str(match['memory_id'])[:8]} [{match.get('kind')}] "
                f"{str(match.get('observed_at') or '')[:10]} {content[:snippet]}"
            )
        return "\n".join(lines)


def matches(store: Any, start: datetime, end: datetime, *, limit: int = 500) -> list[dict[str, Any]]:
    """Active/cold memories *observed* inside the period."""
    rows = store._conn.execute(
        """SELECT id, kind, content, observed_at, created_at FROM memories
           WHERE state IN ('active','cold')
             AND COALESCE(observed_at, created_at) >= ?
             AND COALESCE(observed_at, created_at) < ?
           ORDER BY COALESCE(observed_at, created_at) ASC, id ASC LIMIT ?""",
        (start.isoformat(), end.isoformat(), int(limit)),
    ).fetchall()
    return [
        {
            "memory_id": str(row["id"]),
            "kind": row["kind"],
            "content": row["content"],
            "observed_at": row["observed_at"] or row["created_at"],
        }
        for row in rows
    ]


def preview(store: Any, spec: str, *, limit: int = 500, at: Any = None) -> RangePreview:
    start, end = parse_range(spec)
    found = matches(store, start, end, limit=limit)
    envelope = forget_preview.record_envelope(
        store, [match["memory_id"] for match in found], f"range:{spec}", at=at
    )
    return RangePreview(
        spec=spec,
        start=start.isoformat(),
        end=end.isoformat(),
        envelope=envelope,
        matches=found,
    )


def apply(
    store: Any, envelope: str, ids: list[str], *, at: Any = None, allow_protected: bool = False
) -> Any:
    """Tombstone exactly the previewed ids. Held records block the request entirely."""
    from . import legal_hold

    held = legal_hold.held_ids(store)
    blocked = [memory_id for memory_id in ids if memory_id in held]
    if blocked:
        return forget_preview.Applied(
            envelope=envelope,
            refused="memories on legal hold: " + ", ".join(item[:8] for item in blocked),
        )
    return forget_preview.apply(store, envelope, ids, at=at, allow_protected=allow_protected)


def summary(store: Any, found: list[dict[str, Any]]) -> dict[str, int]:
    """Kinds touched by a preview — the part that tells you if the period is the one you meant."""
    counts: dict[str, int] = {}
    for match in found:
        kind = str(match.get("kind") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return counts
