#!/usr/bin/env python3
"""Page on a slow path: read the operation_spans traces and compare to budget.

Recall, write, prune and consolidate each record one span per operation (see
``cortex/telemetry.py``). This script is the only thing that turns those spans
into an alert, so the thresholds stay inspectable and the jobs that write spans
stay dumb.

    python3 scripts/slow_path_alerts.py --db ~/.hermes/cortex/cortex.db --window 24
    python3 scripts/slow_path_alerts.py --json            # for cron mail / metrics
    python3 scripts/slow_path_alerts.py --budget my.json  # {"recall": 300, ...}

An operation breaches when its p95 spends more than its budget, or when any span
in the window failed. Exit codes: 0 clean, 1 breach, 2 database or table unreadable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    _spec = importlib.util.spec_from_file_location(
        "cortex", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    if _spec is None or _spec.loader is None:
        raise RuntimeError("could not load the Cortex checkout")
    _package = importlib.util.module_from_spec(_spec)
    sys.modules["cortex"] = _package
    _spec.loader.exec_module(_package)

from cortex.telemetry import OP_BUDGETS_MS, slow_path_report  # noqa: E402

DEFAULT_DB = Path(os.environ.get("CORTEX_DB", Path.home() / ".hermes" / "cortex" / "cortex.db"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--window", type=float, default=24.0, help="hours to look back")
    parser.add_argument("--budget", type=Path, help="JSON file overriding the per-op p95 ceilings")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    budgets = None
    if args.budget:
        try:
            budgets = json.loads(args.budget.read_text())
        except (OSError, ValueError) as exc:
            print(f"slow_path_alerts: cannot read budget: {exc}", file=sys.stderr)
            raise SystemExit(2)
        if not isinstance(budgets, dict):
            print("slow_path_alerts: budget must be a JSON object", file=sys.stderr)
            raise SystemExit(2)

    try:
        conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        print(f"slow_path_alerts: cannot open {args.db}: {exc}", file=sys.stderr)
        raise SystemExit(2)

    report = slow_path_report(conn, window_hours=args.window, budgets=budgets)
    conn.close()

    if "error" in report:
        print(f"slow_path_alerts: no spans table: {report['error']}", file=sys.stderr)
        raise SystemExit(2)

    failed = sorted(op for op, row in report["ops"].items() if row.get("errors"))
    breaches = sorted(set(report["breaches"]) | set(failed))
    payload = {**report, "breaches": breaches, "failed_ops": failed}

    if args.json:
        print(json.dumps(payload, indent=1))
    else:
        print(f"window {report['window_hours']}h  spans {report['spans']}")
        for op, row in report["ops"].items():
            mark = "FAIL" if op in breaches else "ok"
            print(
                f"  {mark:4} {op:12} n={row['count']:<5} p50={row['p50_ms']:>8.1f} "
                f"p95={row['p95_ms']:>8.1f} max={row['max_ms']:>9.1f} "
                f"budget={row['budget_ms']} errors={row['errors']}"
            )
        if not report["ops"]:
            print("  no spans in window (nothing ran, or the plugin is not deployed yet)")
        if breaches:
            print("breaches: " + ", ".join(breaches))
        elif report["ops"]:
            print("within budget")

    raise SystemExit(1 if breaches else 0)


if __name__ == "__main__":
    main()
