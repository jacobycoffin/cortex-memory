"""Span-level traces for the four operations that can quietly get slow.

Recall, write, prune and consolidate each append one row to ``operation_spans``:
wall-clock duration, an ok flag, and a small detail payload. Nothing here decides
what the memory does — the spans are evidence, and ``scripts/slow_path_alerts.py``
is the only thing that turns them into a page.

Recording is best-effort by design: a telemetry failure must never lose a memory,
abort a recall, or fail a pruning run. ``record_span`` swallows and logs its own
errors; ``trace_operation`` re-raises whatever the wrapped function raised.

Budgets are per-op wall-clock ceilings evaluated on the p95 once a window holds at
least 20 spans, and on a gross 3x outlier below that (at small n the p95 *is* the
slowest call, so p95 alone would page on ordinary variance). Prune and consolidate
call a model, so their ceilings are minutes, not milliseconds.
"""

from __future__ import annotations

import functools
import json
import logging
import time
import uuid
from typing import Any, Callable, Mapping, TypeVar

LOG = logging.getLogger(__name__)

#: Per-op p95 ceilings in milliseconds. Recall/write are interactive paths;
#: prune/consolidate run on a timer with a model in the loop.
#:
#: These are measured, not aspirational. On a scratch copy of the live 8.4k-memory
#: database, steady-state recall ran 105-128ms (cold first recall 1.8s) and write
#: ran ~1.6s, because add_memory computes features/embeddings inline. The write
#: ceiling therefore sits above that measured cost so the alert reports a
#: regression, not the normal price of a write. Tighten via --budget once the
#: write path gets cheaper; do not tighten it to make the dashboard green.
OP_BUDGETS_MS: dict[str, float] = {
    "recall": 250.0,
    "write": 2500.0,
    "prune": 120_000.0,
    "consolidate": 180_000.0,
}

F = TypeVar("F", bound=Callable[..., Any])


def _json_detail(detail: Mapping[str, Any] | None) -> str:
    if not detail:
        return "{}"
    try:
        return json.dumps(
            {str(k): v for k, v in dict(detail).items()},
            ensure_ascii=True,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):
        return "{}"


def record_span_tx(
    conn: Any,
    op: str,
    duration_ms: float,
    *,
    ok: bool = True,
    task_id: str | None = None,
    detail: Mapping[str, Any] | None = None,
) -> str:
    """Append one span inside an already-open transaction."""

    span_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO operation_spans(span_id,op,duration_ms,ok,task_id,detail_json,created_at)
           VALUES(?,?,?,?,?,?,datetime('now'))""",
        (
            span_id,
            str(op),
            max(0.0, float(duration_ms)),
            int(bool(ok)),
            task_id,
            _json_detail(detail),
        ),
    )
    return span_id


def record_span(
    store: Any,
    op: str,
    duration_ms: float,
    *,
    ok: bool = True,
    task_id: str | None = None,
    detail: Mapping[str, Any] | None = None,
) -> str | None:
    """Append one span in its own short transaction. Never raises."""

    try:
        with store.transaction() as conn:
            return record_span_tx(
                conn, op, duration_ms, ok=ok, task_id=task_id, detail=detail
            )
    except Exception:  # telemetry is evidence, never load-bearing
        LOG.warning("span not recorded (op=%s, %.1fms)", op, duration_ms, exc_info=True)
        return None


def trace_operation(op: str) -> Callable[[F], F]:
    """Record one span per call of a ``f(store, ...)`` job.

    Used by the timer-driven jobs (prune, consolidate) that cannot be wrapped
    inline without indenting their whole body. The run_id from a returned report
    is kept in the detail so a slow span can be joined to its run row.
    """

    def decorate(func: F) -> F:
        @functools.wraps(func)
        def wrapper(store: Any, *args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            ok = False
            result: Any = None
            try:
                result = func(store, *args, **kwargs)
                ok = True
                return result
            finally:
                detail: dict[str, Any] = {}
                if isinstance(result, Mapping) and result.get("run_id"):
                    detail["run_id"] = result["run_id"]
                if isinstance(result, Mapping) and result.get("mode"):
                    detail["mode"] = result["mode"]
                record_span(
                    store,
                    op,
                    (time.perf_counter() - started) * 1000.0,
                    ok=ok,
                    detail=detail,
                )

        return wrapper  # type: ignore[return-value]

    return decorate


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(len(ordered) * pct)) - 1))
    return ordered[index]


#: Below this many spans, a single slow call *is* the p95, so p95 alone would page
#: on ordinary variance. Small samples are judged by a gross-outlier rule instead.
MIN_SAMPLES_FOR_P95 = 20
OUTLIER_FACTOR = 3.0


def slow_path_report(
    conn: Any,
    *,
    window_hours: float = 24.0,
    budgets: Mapping[str, float] | None = None,
    min_samples: int = MIN_SAMPLES_FOR_P95,
    outlier_factor: float = OUTLIER_FACTOR,
) -> dict[str, Any]:
    """Summarise spans in the window and name the ops that breach.

    An op breaches when it has at least ``min_samples`` spans and its p95 is over
    budget, or — with fewer spans — when one span is over ``budget * outlier_factor``.
    """

    effective = dict(OP_BUDGETS_MS)
    if budgets:
        effective.update({str(k): float(v) for k, v in budgets.items()})
    try:
        rows = conn.execute(
            """SELECT op,duration_ms,ok FROM operation_spans
               WHERE created_at >= datetime('now',?)""",
            (f"-{float(window_hours)} hours",),
        ).fetchall()
    except Exception as exc:  # table absent in an undeployed database
        return {
            "window_hours": window_hours,
            "spans": 0,
            "ops": {},
            "breaches": [],
            "error": str(exc),
        }

    by_op: dict[str, dict[str, Any]] = {}
    for op, duration_ms, ok in rows:
        entry = by_op.setdefault(
            str(op),
            {"count": 0, "errors": 0, "durations": []},
        )
        entry["count"] += 1
        entry["durations"].append(float(duration_ms or 0.0))
        if not ok:
            entry["errors"] += 1

    ops: dict[str, Any] = {}
    breaches: list[str] = []
    for op, entry in sorted(by_op.items()):
        durations = entry.pop("durations")
        p95 = percentile(durations, 0.95)
        budget = effective.get(op)
        detail = {
            **entry,
            "p50_ms": round(percentile(durations, 0.50), 1),
            "p95_ms": round(p95, 1),
            "max_ms": round(max(durations), 1),
            "budget_ms": budget,
        }
        if budget is not None:
            if entry["count"] >= min_samples:
                if p95 > budget:
                    breaches.append(op)
                    detail["breach"] = "p95"
            elif max(durations) > budget * outlier_factor:
                breaches.append(op)
                detail["breach"] = "outlier"
        ops[op] = detail

    return {
        "window_hours": window_hours,
        "spans": len(rows),
        "ops": ops,
        "breaches": breaches,
    }
