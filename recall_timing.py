"""Per-stage recall latency, measured from the store instead of from one-off probes.

The store could already answer *how long a recall took* (`recall_runs.prepare_ms`) and *roughly
where it went* (`recall_runs.stage_ms_json`), but three things made that unusable as a report:

* **the embedding model load had no stage of its own.** `semantic` covered the lazy ONNX load
  (~800 ms on the first prefetch of a process), the query embedding and the vector search as one
  number, so a slow *first* turn and a slow *steady-state* turn looked identical;
* **rendering was not timed at all.** Composing the injected block (dedupe, budget fitting, the
  provenance labels) happens after retrieval and never appeared in any stage;
* **a value in `stage_ms` was not necessarily a duration.** Counters, flags and even a score were
  written into the same map (`reranked`, `abstained`, `abstained_top_score`), and a reader that
  took every key at face value reported "abstained: 1.0 ms" as a pipeline stage. Cached runs made
  it worse: every stage was stored as `0.0`, so a p50 blended real work with cache hits.

This module is the contract for that column, in one place:

- `stage_ms` is ``dict[str, float]`` and **every value is milliseconds, except keys ending in
  ``_count``** (counters) — `STAGE_DURATIONS` names the durations, and anything else is reported
  as a counter rather than silently ranked as a stage;
- writing is *total*: `StageSink` drops a non-finite, negative or non-numeric value instead of
  raising, and `sanitize_stage_ms()` is what the store applies before JSON encoding, so a broken
  timer can never cost a recall its row (or its other stages);
- reading is *total* too: `stage_percentiles()` returns an empty report with a reason rather than
  an exception, because a report about recall must not be able to break recall.

**Storage:** no new table. `recall_runs.stage_ms_json` already exists, is written on every recall
path and is keyed by `task_id`, which is also the key of `recall_budget_observations` — so a report
can join stage timings to budget/what-was-rendered without duplicating either. Adding a second
table would have meant two sources of truth for one turn.

**Overhead:** a handful of `perf_counter()` calls per recall (sub-microsecond each) plus one JSON
encode that the store was already doing.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

# --------------------------------------------------------------------------- the contract

#: Keys of `stage_ms` that are milliseconds. Ordered as the pipeline runs, so a report reads
#: top-to-bottom. A key that is NOT in here is reported as a counter, never as a stage.
STAGE_DURATIONS: tuple[str, ...] = (
    "fts",
    "feature",
    "context",
    "neighborhood",
    "scoring",
    "graph",
    "prospective_merge",
    "embed_load",
    "embed",
    "semantic",
    "entity",
    "rerank",
    "validity",
    "stability",
    "negative_cues",
    "select",
    "render",
)

#: A cached recall records `cache_lookup` and nothing else: it did not re-run the pipeline, so its
#: zeros are not measurements and must not be averaged in with real runs.
CACHE_MARKER = "cache_lookup"

#: Suffix that marks a value as a counter rather than a duration.
COUNT_SUFFIX = "_count"

#: Default read window, matching `latency_reconcile`.
DEFAULT_WINDOW = 512


def is_duration(key: str) -> bool:
    """True when `key` is a measured stage rather than a counter or a flag."""

    return str(key) in STAGE_DURATIONS


def is_counter(key: str) -> bool:
    return str(key).endswith(COUNT_SUFFIX)


# --------------------------------------------------------------------------- writing (never raises)


def clock() -> float:
    """`time.perf_counter()`, or 0.0 if the clock itself is failing."""

    try:
        return float(time.perf_counter())
    except Exception:  # noqa: BLE001 - a timer may never break a recall
        return 0.0


def elapsed_ms(started: float) -> float:
    """Milliseconds since `clock()`, clamped at zero, or 0.0 on any failure."""

    try:
        value = (float(time.perf_counter()) - float(started)) * 1000.0
    except Exception:  # noqa: BLE001
        return 0.0
    if not math.isfinite(value) or value < 0.0:
        return 0.0
    return value


def _clean(value: Any) -> float | None:
    """A finite, non-negative float, or None when the value cannot be one."""

    try:
        number = float(value)
    except Exception:  # noqa: BLE001 - a broken value is dropped, not raised
        return None
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


class StageSink(dict):
    """A ``dict[str, float]`` that refuses anything that is not a millisecond.

    Dropping garbage here is the difference between a missing stage and a failed recall: the
    pipeline keeps assigning `stage_ms["fts"] = ...` in twenty places, and none of those
    assignments may be able to raise. Bad keys and values simply never enter the map, so the
    stored payload stays valid JSON with no ``NaN`` in it.
    """

    def __setitem__(self, key: Any, value: Any) -> None:  # type: ignore[override]
        try:
            name = str(key)
            number = _clean(value)
            if number is None:
                return
            super().__setitem__(name, number)
        except Exception:  # noqa: BLE001
            return

    def count(self, name: str, value: Any) -> None:
        """Record a counter with the required ``_count`` suffix."""

        base = str(name)
        if not base.endswith(COUNT_SUFFIX):
            base = f"{base}{COUNT_SUFFIX}"
        self[base] = value

    def as_dict(self) -> dict[str, float]:
        return dict(self)


def sanitize_stage_ms(stage_ms: Mapping[str, Any] | None) -> dict[str, float]:
    """Copy a stage map, keeping only finite non-negative numbers.

    Applied at the store boundary so a caller that built a plain dict (or a test that passes a
    hostile one) cannot write ``NaN``/``Infinity`` into `stage_ms_json`.
    """

    cleaned: dict[str, float] = {}
    if not stage_ms:
        return cleaned
    try:
        items = list(dict(stage_ms).items())
    except Exception:  # noqa: BLE001 - an unreadable mapping contributes nothing
        return cleaned
    for key, value in items:
        number = _clean(value)
        if number is None:
            continue
        cleaned[str(key)] = number
    return cleaned


def add_stage(
    stage_ms: Mapping[str, Any] | None, name: str, value: Any
) -> dict[str, float]:
    """A new stage map with one measurement merged in. Returns a copy, never raises."""

    merged = sanitize_stage_ms(stage_ms)
    number = _clean(value)
    if number is not None:
        merged[str(name)] = number
    return merged


# --------------------------------------------------------------------------- reading (never raises)


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Empty input is 0.0, matching `latency_reconcile`."""

    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


@dataclass
class TimingReport:
    """p50/p95 per stage over a window, plus the counters kept out of the percentiles."""

    window: int = DEFAULT_WINDOW
    runs: int = 0
    cached_runs: int = 0
    prepare_p50: float = 0.0
    prepare_p95: float = 0.0
    prepare_max: float = 0.0
    stages: list[dict[str, Any]] = field(default_factory=list)
    counters: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "runs": self.runs,
            "cached_runs": self.cached_runs,
            "prepare_ms": {
                "p50": round(self.prepare_p50, 1),
                "p95": round(self.prepare_p95, 1),
                "max": round(self.prepare_max, 1),
            },
            "stages": [dict(item) for item in self.stages],
            "counters": [dict(item) for item in self.counters],
            "note": self.note,
        }

    def stage(self, name: str) -> dict[str, Any]:
        """One stage's row, or an empty dict when it was never recorded."""

        for item in self.stages:
            if item["stage"] == name:
                return item
        return {}

    def lines(self) -> list[str]:
        return format_report(self)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def stage_percentiles(store: Any, *, window: int = DEFAULT_WINDOW) -> TimingReport:
    """p50/p95/p99 per stage over the newest `window` recorded recalls.

    Cached recalls are counted and excluded: they store zeros for every stage they did not run,
    which would drag the real distribution toward a latency nobody measured.
    """

    try:
        limit = max(1, int(window))
    except Exception:  # noqa: BLE001
        limit = DEFAULT_WINDOW
    report = TimingReport(window=limit)
    try:
        return _profile(store, report)
    except Exception as error:  # noqa: BLE001 - a report may never break its caller
        report.note = f"could not read recall timings: {type(error).__name__}"
        return report


def _profile(store: Any, report: TimingReport) -> TimingReport:
    rows = _rows(
        store,
        # Order by insertion, not by recall_id: two recalls inside the same second share created_at, and recall_id is a
        # random UUID, so a tie on the timestamp would pick an arbitrary run and quietly report the wrong window.
        "SELECT prepare_ms, stage_ms_json FROM recall_runs WHERE prepare_ms IS NOT NULL "
        "ORDER BY created_at DESC, rowid DESC LIMIT ?",
        (int(report.window),),
    )
    if not rows:
        report.note = "no timed recalls recorded"
        return report

    prepare: list[float] = []
    per_stage: dict[str, list[float]] = {}
    per_counter: dict[str, list[float]] = {}
    for row in rows:
        try:
            stages = json.loads(row["stage_ms_json"] or "{}")
        except Exception:  # noqa: BLE001 - one unreadable row does not hide the others
            stages = {}
        if not isinstance(stages, dict):
            stages = {}
        if CACHE_MARKER in stages:
            report.cached_runs += 1
            continue
        value = _clean(row["prepare_ms"])
        if value is None:
            continue
        prepare.append(value)
        for key, raw in stages.items():
            number = _clean(raw)
            if number is None:
                continue
            if is_duration(str(key)):
                per_stage.setdefault(str(key), []).append(number)
            else:
                per_counter.setdefault(str(key), []).append(number)

    report.runs = len(prepare)
    if not prepare:
        report.note = (
            "every run in the window was a cache hit or unreadable: no pipeline stage was measured"
        )
        return report
    report.prepare_p50 = percentile(prepare, 0.5)
    report.prepare_p95 = percentile(prepare, 0.95)
    report.prepare_max = max(prepare)

    order = {name: index for index, name in enumerate(STAGE_DURATIONS)}
    report.stages = [
        {
            "stage": name,
            "p50": round(percentile(values, 0.5), 1),
            "p95": round(percentile(values, 0.95), 1),
            "p99": round(percentile(values, 0.99), 1),
            "max": round(max(values), 1),
            "runs": len(values),
        }
        for name, values in sorted(
            per_stage.items(), key=lambda item: (order.get(item[0], len(order)), item[0])
        )
    ]
    report.counters = [
        {
            "counter": name,
            "p50": round(percentile(values, 0.5), 1),
            "max": round(max(values), 1),
            "runs": len(values),
        }
        for name, values in sorted(per_counter.items())
    ]
    return report


def format_report(report: TimingReport) -> list[str]:
    """Printable lines for a report. Never raises, even on an empty report."""

    try:
        skipped = f", {report.cached_runs} cache hit(s) excluded" if report.cached_runs else ""
        lines = [
            f"recall latency, last {report.window} run(s): {report.runs} timed{skipped}"
        ]
        if report.note:
            lines.append(report.note)
        if report.runs:
            lines.append(
                f"prepare_ms  p50 {report.prepare_p50:.1f}  "
                f"p95 {report.prepare_p95:.1f}  max {report.prepare_max:.1f}"
            )
            lines.append(f"{'stage':<18}{'p50':>9}{'p95':>9}{'p99':>9}{'runs':>7}")
            for item in report.stages:
                lines.append(
                    f"{item['stage']:<18}{item['p50']:>9.1f}{item['p95']:>9.1f}"
                    f"{item['p99']:>9.1f}{item['runs']:>7}"
                )
            if report.counters:
                names = ", ".join(item["counter"] for item in report.counters)
                lines.append(f"counters (not milliseconds): {names}")
        return lines
    except Exception as error:  # noqa: BLE001
        return [f"could not format recall timings: {type(error).__name__}"]
