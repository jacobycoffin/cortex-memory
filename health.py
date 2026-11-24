"""One screen that says whether this brain is healthy (#169)

Every number here already exists somewhere else — in `recall_runs`, in `memory_traces`, in the queue
tables, in the file on disk. What was missing is a single read that answers *is anything wrong*, and
does it without the flattery that health dashboards usually ship with:

- **a check that cannot be measured says `unknown`, never `0`.** A missing table is not a healthy zero;
  it is an absence of evidence, and reporting it as zero is how a broken instrument reads as a fixed
  problem.
- **every check names its source and its window.** "p50 971 ms" is not a fact until it says *over the
  last 500 recorded runs*.
- **every threshold is printed with the reading.** A warning whose bar is invisible cannot be argued
  with, and a bar that moves silently is worse.
- **the worst check decides the verdict.** A green average over a failing tail is the specific lie this
  screen exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

OK = "ok"
WARN = "warn"
FAIL = "fail"
UNKNOWN = "unknown"
_ORDER = {OK: 0, UNKNOWN: 1, WARN: 2, FAIL: 3}

LATENCY_SLO_P50 = 150.0
LATENCY_SLO_P95 = 250.0
GROWTH_MB_PER_DAY = 5.0
WINDOW = 500


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


@dataclass
class Check:
    name: str
    status: str = UNKNOWN
    value: Any = None
    unit: str = ""
    source: str = ""
    window: str = ""
    threshold: str = ""
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "status": self.status, "value": self.value, "unit": self.unit,
            "source": self.source, "window": self.window, "threshold": self.threshold,
            "note": self.note,
        }

    def line(self) -> str:
        mark = {OK: "✓", WARN: "!", FAIL: "✗", UNKNOWN: "?"}[self.status]
        value = "—" if self.value is None else f"{self.value}{self.unit}"
        bits = [f"{mark} {self.name}: {value}"]
        if self.threshold:
            bits.append(f"[{self.threshold}]")
        if self.source:
            bits.append(f"({self.source}{', ' + self.window if self.window else ''})")
        if self.note:
            bits.append(f"— {self.note}")
        return "  ".join(bits)


def latency(store: Any, *, window: int = WINDOW) -> Check:
    rows = _rows(
        store,
        "SELECT prepare_ms FROM recall_runs WHERE prepare_ms IS NOT NULL "
        "ORDER BY created_at DESC, recall_id DESC LIMIT ?",
        (int(window),),
    )
    values = [float(row["prepare_ms"]) for row in rows if row["prepare_ms"] is not None]
    if not values:
        return Check("latency", UNKNOWN, None, "ms", "recall_runs", f"last {window}",
                     f"p50 ≤ {LATENCY_SLO_P50:g}, p95 ≤ {LATENCY_SLO_P95:g}",
                     "no timing recorded yet")
    p50, p95 = _percentile(values, 0.5), _percentile(values, 0.95)
    over = sum(1 for value in values if value > LATENCY_SLO_P50)
    status = FAIL if p95 > LATENCY_SLO_P95 * 4 else (WARN if p95 > LATENCY_SLO_P95 or p50 > LATENCY_SLO_P50 else OK)
    return Check("latency", status, f"p50 {p50:.0f} / p95 {p95:.0f}", "ms", "recall_runs",
                 f"last {len(values)}",
                 f"p50 ≤ {LATENCY_SLO_P50:g}, p95 ≤ {LATENCY_SLO_P95:g}",
                 f"{over} of {len(values)} runs over the p50 target")


def storage(store: Any, *, growth_mb_per_day: float = GROWTH_MB_PER_DAY) -> Check:
    path = ""
    try:
        path = str(store._conn.execute("PRAGMA database_list").fetchone()["file"] or "")
    except Exception:  # noqa: BLE001
        path = ""
    if not path:
        return Check("storage", UNKNOWN, None, "MB", "PRAGMA database_list", "", "", "no file to measure")
    try:
        size = Path(path).stat().st_size / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return Check("storage", UNKNOWN, None, "MB", path, "", "", "the file could not be read")
    rows = _rows(
        store,
        "SELECT MIN(created_at) AS first, MAX(created_at) AS last, COUNT(*) AS count FROM memories",
    )
    per_day = None
    if rows and rows[0]["first"] and rows[0]["last"]:
        try:
            first = datetime.fromisoformat(str(rows[0]["first"]))
            last = datetime.fromisoformat(str(rows[0]["last"]))
            days = max(1.0, (last - first).total_seconds() / 86400)
            per_day = round(size / days, 2)
        except Exception:  # noqa: BLE001
            per_day = None
    if per_day is None:
        return Check("storage", UNKNOWN, round(size, 1), "MB", "memories", "", "", "growth unknown")
    status = FAIL if per_day > growth_mb_per_day * 3 else (WARN if per_day > growth_mb_per_day else OK)
    return Check("storage", status, round(size, 1), "MB", "the database file, first→last memory",
                 "", f"≤ {growth_mb_per_day:g} MB/day", f"≈ {per_day} MB/day across the store's life")


def abstention(store: Any, *, window: int = WINDOW) -> Check:
    rows = _rows(
        store,
        "SELECT COUNT(*) AS total, SUM(CASE WHEN abstained THEN 1 ELSE 0 END) AS abstained "
        "FROM (SELECT abstained FROM recall_runs ORDER BY created_at DESC LIMIT ?)",
        (int(window),),
    )
    if not rows or not rows[0]["total"]:
        return Check("abstention", UNKNOWN, None, "%", "recall_runs", f"last {window}", "≤ 60%")
    total = int(rows[0]["total"] or 0)
    share = round(100.0 * int(rows[0]["abstained"] or 0) / total, 1)
    status = FAIL if share > 85 else (WARN if share > 60 else OK)
    note = "retrieval is declining most questions" if share > 60 else ""
    return Check("abstention", status, share, "%", "recall_runs", f"last {total}", "≤ 60%", note)


def error_turns(store: Any, *, window: int = WINDOW) -> Check:
    rows = _rows(
        store,
        "SELECT COUNT(*) AS count FROM memory_traces WHERE retrieval_used = 0",
    )
    if not rows:
        return Check("turns without retrieval", UNKNOWN, None, "", "memory_traces", "", "≤ 5% of turns")
    total = _rows(store, "SELECT COUNT(*) AS count FROM memory_traces")
    count = int(rows[0]["count"] or 0)
    all_turns = int(total[0]["count"] or 0) if total else 0
    if not all_turns:
        return Check("turns without retrieval", UNKNOWN, None, "", "memory_traces", "", "≤ 5% of turns")
    share = round(100.0 * count / all_turns, 1)
    return Check("turns without retrieval", WARN if share > 5 else OK, share, "%", "memory_traces",
                 f"all {all_turns}", "≤ 5% of turns",
                 f"{count} turn(s) recorded no retrieval at all")


def queue_health(store: Any) -> Check:
    rows = _rows(
        store,
        "SELECT state, COUNT(*) AS count FROM consolidation_queue GROUP BY state",
    )
    if not rows:
        return Check("consolidation queue", UNKNOWN, None, "", "consolidation_queue", "", "0 parked",
                     "nothing has been queued yet")
    counts = {str(row["state"]): int(row["count"] or 0) for row in rows}
    parked = counts.get("parked", 0)
    queued = counts.get("queued", 0)
    return Check("consolidation queue", FAIL if parked else (WARN if queued > 500 else OK),
                 f"{queued} queued / {parked} parked", "", "consolidation_queue", "", "0 parked",
                 "parked items failed three times and stopped")


def data_quality(store: Any) -> Check:
    """Is the text the model reads clean, or is it still carrying the importer's template?"""
    rows = _rows(
        store,
        "SELECT COUNT(*) AS total, SUM(CASE WHEN content LIKE 'Vault note:%' THEN 1 ELSE 0 END) "
        "AS headered FROM memories WHERE state = 'active'",
    )
    if not rows or not rows[0]["total"]:
        return Check("template text", UNKNOWN, None, "%", "memories", "", "≤ 5%")
    total = int(rows[0]["total"] or 0)
    headered = int(rows[0]["headered"] or 0)
    share = round(100.0 * headered / total, 1)
    return Check("template text", FAIL if share > 50 else (WARN if share > 5 else OK), share, "%",
                 "memories", f"{total} active", "≤ 5%",
                 "vault header still inside stored content (#223)" if share else "")


def coverage(store: Any) -> Check:
    rows = _rows(store, "SELECT COUNT(*) AS count FROM memories WHERE state = 'active' AND pinned = 1")
    if not rows:
        return Check("pinned facts", UNKNOWN, None, "", "memories", "", "≥ 3")
    pinned = int(rows[0]["count"] or 0)
    return Check("pinned facts", WARN if pinned < 3 else OK, pinned, "", "memories", "", "≥ 3",
                 "" if pinned >= 3 else "the session preamble (#150) has almost nothing to open with")


def checks(store: Any, *, window: int = WINDOW) -> list[Check]:
    return [
        latency(store, window=window),
        storage(store),
        abstention(store, window=window),
        error_turns(store, window=window),
        queue_health(store),
        data_quality(store),
        coverage(store),
    ]


def verdict(items: list[Check]) -> str:
    worst = OK
    for item in items:
        if _ORDER.get(item.status, 1) > _ORDER.get(worst, 0):
            worst = item.status
    return worst


def render(store: Any, *, window: int = WINDOW) -> list[str]:
    items = checks(store, window=window)
    overall = verdict(items)
    lines = [f"health: {overall.upper()}  ({len(items)} checks)"]
    for item in items:
        lines.append(item.line())
    failing = [item.name for item in items if item.status in (FAIL, WARN)]
    if failing:
        lines.append(f"  needs attention: {', '.join(failing)}")
    return lines


def report(store: Any, *, window: int = WINDOW) -> dict[str, Any]:
    items = checks(store, window=window)
    return {
        "verdict": verdict(items),
        "checks": [item.as_dict() for item in items],
        "generated_at": _now().isoformat(),
        "note": "a check that cannot be measured says unknown, never zero",
    }
