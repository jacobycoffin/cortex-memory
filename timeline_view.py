"""The shape of the store over time (#170)

A list answers *what do I know*. This answers *how did I come to know it* — which months were busy, which
weeks were silent, and what kind of thing was being written when.

Three rules that make a time series trustworthy:

- **a gap is data.** A day with no memories is reported as `0`, not omitted. Omitting it makes a quiet
  period disappear, which is precisely the period worth noticing — and a chart that hides its silences
  flatters the store.
- **counts come from a group-by**, not from filtering a truncated page of rows. A series built from "the
  last 500 memories" measures the page, not the store.
- **the window is part of the answer.** Every series reports its span and its bucket size, because
  "1,500 memories" means nothing without "in September".

The store's own history is lumpy — this one has 3,520 memories in 2026-09 and 165 so far in 2026-11 — and
that lumpiness is the most useful thing the view shows: it is the difference between a store that grew
and a store that was imported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

BUCKETS = ("day", "week", "month")
DEFAULT_DAYS = 90
SPARK = " ▁▂▃▄▅▆▇█"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def _parse(stamp: Any) -> date | None:
    try:
        return datetime.fromisoformat(str(stamp)).date()
    except Exception:  # noqa: BLE001
        return None


def _bucket_start(day: date, bucket: str) -> date:
    if bucket == "week":
        return day - timedelta(days=day.weekday())
    if bucket == "month":
        return day.replace(day=1)
    return day


def _advance(start: date, bucket: str) -> date:
    if bucket == "week":
        return start + timedelta(days=7)
    if bucket == "month":
        return (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    return start + timedelta(days=1)


@dataclass
class Series:
    bucket: str = "day"
    start: str = ""
    end: str = ""
    points: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"bucket": self.bucket, "start": self.start, "end": self.end,
                "points": self.points, "note": self.note}


def series(store: Any, *, days: int = DEFAULT_DAYS, bucket: str = "day",
           now: datetime | None = None) -> Series:
    """Counts per bucket over the last `days`, with empty buckets present as zeros."""
    if bucket not in BUCKETS:
        bucket = "day"
    span = max(1, int(days))
    today = (now or _now()).date()
    first = _bucket_start(today - timedelta(days=span - 1), bucket)
    rows = _rows(
        store,
        "SELECT substr(created_at, 1, 10) AS day, kind, COUNT(*) AS count FROM memories "
        "WHERE state <> 'archived' AND created_at >= ? GROUP BY day, kind",
        (first.isoformat(),),
    )
    counts: dict[date, dict[str, int]] = {}
    for row in rows:
        day = _parse(row["day"])
        if day is None:
            continue
        slot = counts.setdefault(_bucket_start(day, bucket), {})
        slot[str(row["kind"] or "unknown")] = slot.get(str(row["kind"] or "unknown"), 0) + int(
            row["count"] or 0
        )
    points: list[dict[str, Any]] = []
    cursor = first
    while cursor <= today:
        kinds = counts.get(cursor, {})
        points.append({
            "bucket_start": cursor.isoformat(),
            "count": sum(kinds.values()),
            "kinds": dict(sorted(kinds.items())),
        })
        cursor = _advance(cursor, bucket)
    found = Series(bucket=bucket, start=first.isoformat(), end=today.isoformat(), points=points)
    if not any(point["count"] for point in points):
        found.note = "nothing was written in this window"
    return found


def summary(store: Any, *, days: int = DEFAULT_DAYS, now: datetime | None = None) -> dict[str, Any]:
    """Totals, extremes and silence — the numbers worth quoting."""
    found = series(store, days=days, bucket="day", now=now)
    counts = [point["count"] for point in found.points]
    if not counts:
        return {"window": f"{found.start} → {found.end}", "total": 0, "note": "no window"}
    busiest = max(found.points, key=lambda point: point["count"])
    seen = _rows(store, "SELECT MIN(created_at) AS first, MAX(created_at) AS last, COUNT(*) AS count "
                        "FROM memories WHERE state <> 'archived'")
    streak = longest_quiet = 0
    for count in counts:
        streak = streak + 1 if count == 0 else 0
        longest_quiet = max(longest_quiet, streak)
    return {
        "window": f"{found.start} → {found.end}",
        "days": len(counts),
        "total": sum(counts),
        "per_day": round(sum(counts) / len(counts), 2),
        "busiest": {"day": busiest["bucket_start"], "count": busiest["count"]},
        "quiet_days": sum(1 for count in counts if count == 0),
        "longest_quiet_streak": longest_quiet,
        "first_memory": str(seen[0]["first"] or "")[:10] if seen else "",
        "last_memory": str(seen[0]["last"] or "")[:10] if seen else "",
        "all_time": int(seen[0]["count"] or 0) if seen else 0,
    }


def months(store: Any, *, limit: int = 12) -> list[dict[str, Any]]:
    """Monthly counts across the store's whole life — the shape nobody sees in a page of rows."""
    rows = _rows(
        store,
        "SELECT substr(created_at, 1, 7) AS month, COUNT(*) AS count FROM memories "
        "WHERE state <> 'archived' AND created_at <> '' GROUP BY month ORDER BY month DESC LIMIT ?",
        (max(1, int(limit)),),
    )
    return [{"month": str(row["month"]), "count": int(row["count"] or 0)}
            for row in reversed(rows)]


def sparkline(points: list[dict[str, Any]], *, width: int = 60) -> str:
    """A block-character chart. Cheap, honest, and readable in a terminal at 2am."""
    values = [int(point.get("count") or 0) for point in points]
    if not values:
        return ""
    if len(values) > width:
        # Downsample by taking the maximum in each slot: a spike must not vanish in a bucket.
        stride = len(values) / float(width)
        reduced = []
        for index in range(width):
            low = int(index * stride)
            high = max(low + 1, int((index + 1) * stride))
            reduced.append(max(values[low:high]))
        values = reduced
    top = max(values) or 1
    return "".join(SPARK[min(len(SPARK) - 1, int(round(value / top * (len(SPARK) - 1))))]
                   for value in values)


def render(store: Any, *, days: int = DEFAULT_DAYS, now: datetime | None = None) -> list[str]:
    found = series(store, days=days, bucket="day", now=now)
    described = summary(store, days=days, now=now)
    lines = [f"{described['window']}: {described['total']} memories, "
             f"{described['per_day']}/day, {described['quiet_days']} quiet day(s)"]
    lines.append("  " + sparkline(found.points))
    if described.get("busiest"):
        lines.append(f"  busiest {described['busiest']['day']} with {described['busiest']['count']}")
    if described.get("longest_quiet_streak"):
        lines.append(f"  longest silence {described['longest_quiet_streak']} day(s)")
    monthly = months(store, limit=6)
    if monthly:
        lines.append("  by month: " + ", ".join(f"{row['month']} {row['count']}"
                                                for row in monthly))
    return lines


def kinds_over(store: Any, *, days: int = DEFAULT_DAYS, now: datetime | None = None
               ) -> dict[str, int]:
    """Which kinds were written in the window — the answer to *what was I doing then*."""
    found = series(store, days=days, bucket="day", now=now)
    totals: dict[str, int] = {}
    for point in found.points:
        for kind, count in (point["kinds"] or {}).items():
            totals[kind] = totals.get(kind, 0) + int(count)
    return dict(sorted(totals.items(), key=lambda item: (-item[1], item[0])))
