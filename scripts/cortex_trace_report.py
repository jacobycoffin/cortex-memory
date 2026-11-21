#!/usr/bin/env python3
"""Summarise ``cortex-trace.jsonl``: p50/p95 per phase plus budget breaches.

Cortex emits one JSONL line per traced phase when ``CORTEX_TRACE=1`` (see
``cortex_trace.py``). This reader answers the two questions the raw file cannot:
which phase owns the tail latency, and which individual runs blew their budget.

    python3 scripts/cortex_trace_report.py
    python3 scripts/cortex_trace_report.py --path /tmp/cortex-trace.jsonl
    python3 scripts/cortex_trace_report.py --budget recall=100 --budget prune=2000

Budgets are milliseconds per phase and default to 250 / 500 / 5000 / 60000 for
recall / write / prune / consolidate. Percentiles are nearest-rank
(``ceil(pct/100 * n) - 1`` over the sorted durations), so p95 is always an
observed duration and never an interpolation. Exit codes: 0 every traced run is
within budget, 1 at least one run exceeded its budget, 2 the trace file or the
budget arguments are unusable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULT_BUDGETS_MS = {
    "recall": 250.0,
    "write": 500.0,
    "prune": 5_000.0,
    "consolidate": 60_000.0,
}


def default_trace_path() -> Path:
    """Same resolution as ``cortex_trace.trace_path`` (kept import-free on purpose).

    This script must run from a bare checkout where the ``cortex`` package is not
    importable under that name, so it resolves the path itself rather than
    importing the module; ``tests/test_cortex_trace.py`` pins the two together.
    """

    explicit = (os.environ.get("CORTEX_TRACE_PATH", "") or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    home = (os.environ.get("CORTEX_HOME", "") or "").strip()
    base = Path(home).expanduser() if home else Path.home() / ".cortex"
    return base / "cortex-trace.jsonl"


def percentile(values: Iterable[float], pct: float) -> float:
    """Nearest-rank percentile; 0.0 for an empty sample."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if pct <= 0:
        return ordered[0]
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * pct / 100.0) - 1))
    return ordered[index]


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read JSONL trace lines, skipping (never crashing on) malformed rows."""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or "phase" not in record or "duration_ms" not in record:
                continue
            try:
                record["duration_ms"] = float(record["duration_ms"])
            except (TypeError, ValueError):
                continue
            records.append(record)
    return records


def summarize(
    records: Iterable[dict[str, Any]],
    budgets_ms: Mapping[str, float],
) -> dict[str, Any]:
    """Per-phase p50/p95/max and every run strictly over its phase budget."""

    durations: dict[str, list[float]] = {}
    breaches: list[dict[str, Any]] = []
    for record in records:
        phase = str(record.get("phase"))
        duration = float(record.get("duration_ms") or 0.0)
        durations.setdefault(phase, []).append(duration)
        budget = budgets_ms.get(phase)
        if budget is not None and duration > float(budget):
            breaches.append(
                {
                    "phase": phase,
                    "duration_ms": duration,
                    "budget_ms": float(budget),
                    "ts": record.get("ts"),
                    "run_id": record.get("run_id"),
                    "session_id": record.get("session_id"),
                    "counts": record.get("counts") or {},
                }
            )

    phases = []
    for phase in sorted(durations):
        values = durations[phase]
        phases.append(
            {
                "phase": phase,
                "count": len(values),
                "p50_ms": percentile(values, 50),
                "p95_ms": percentile(values, 95),
                "max_ms": max(values),
                "budget_ms": budgets_ms.get(phase),
            }
        )
    breaches.sort(key=lambda item: str(item.get("ts") or ""))
    return {"phases": phases, "breaches": breaches, "runs": sum(len(v) for v in durations.values())}


def format_report(summary: Mapping[str, Any]) -> str:
    lines = [
        f"{'phase':<12}{'runs':>6}{'p50_ms':>10}{'p95_ms':>10}{'max_ms':>10}{'budget_ms':>12}",
    ]
    for phase in summary["phases"]:
        budget = phase["budget_ms"]
        budget_text = f"{budget:.0f}" if budget is not None else "-"
        lines.append(
            f"{phase['phase']:<12}{phase['count']:>6}{phase['p50_ms']:>10.2f}"
            f"{phase['p95_ms']:>10.2f}{phase['max_ms']:>10.2f}{budget_text:>12}"
        )
    breaches = summary["breaches"]
    if breaches:
        lines.append("")
        lines.append(f"breaches: {len(breaches)} run(s) over budget")
        for item in breaches:
            counts = item.get("counts") or {}
            detail = " ".join(f"{key}={counts[key]}" for key in sorted(counts))
            session = item.get("session_id") or "-"
            lines.append(
                f"  {item.get('ts') or '?'}  {item['phase']:<11} {item['duration_ms']:>9.1f}ms"
                f" (budget {item['budget_ms']:.0f}ms) run={item.get('run_id') or '-'}"
                f" session={session} {detail}".rstrip()
            )
    else:
        lines.append("")
        lines.append("breaches: none")
    return "\n".join(lines)


def parse_budgets(specs: Iterable[str], base: Mapping[str, float] | None = None) -> dict[str, float]:
    """Merge ``phase=ms`` overrides over the defaults; raises ValueError."""

    budgets = dict(DEFAULT_BUDGETS_MS if base is None else base)
    for spec in specs:
        phase, _, value = spec.partition("=")
        phase = phase.strip()
        if not phase or not value:
            raise ValueError(f"budget must be phase=milliseconds, got {spec!r}")
        try:
            milliseconds = float(value)
        except ValueError as exc:
            raise ValueError(f"budget for {phase!r} is not a number: {value!r}") from exc
        if milliseconds < 0:
            raise ValueError(f"budget for {phase!r} cannot be negative")
        budgets[phase] = milliseconds
    return budgets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="trace JSONL file (default: CORTEX_TRACE_PATH, else $CORTEX_HOME/cortex-trace.jsonl)",
    )
    parser.add_argument(
        "--budget",
        action="append",
        default=[],
        metavar="PHASE=MS",
        help="override a phase budget (repeatable)",
    )
    args = parser.parse_args(argv)

    try:
        budgets = parse_budgets(args.budget)
    except ValueError as exc:
        parser.error(str(exc))
    path = args.path or default_trace_path()
    try:
        records = load_records(path)
    except OSError as exc:
        print(f"cannot read trace file {path}: {exc}", file=sys.stderr)
        return 2
    if not records:
        print(f"no trace records in {path}", file=sys.stderr)
        return 0
    summary = summarize(records, budgets)
    print(format_report(summary))
    return 1 if summary["breaches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
