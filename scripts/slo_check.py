#!/usr/bin/env python3
"""Check a frozen-eval report against the retrieval latency budget.

Latency is reported by the eval runner but never gated there, because it moves with
machine load. It still needs a budget, or "it got slower" is discovered by feel months
later. This is the separate, explicit check.

    python3 scripts/slo_check.py --report ~/cortex-evaluations/frozen/frozen-v1-report.json

Budgets live in --budget (JSON) and default to the shipped slo.json. Exit codes:
0 within budget, 1 over budget, 2 report or budget unreadable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULTS = {
    "retrieval_p50_ms": 150.0,
    "retrieval_p95_ms": 250.0,
    "retrieval_max_ms": 900.0,
    # A case that returns nothing is slow for the user in a different way.
    "zero_result_share": 0.05,
}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(len(ordered) * pct)) - 1))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--budget", type=Path, help="JSON file overriding the defaults")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    try:
        report = json.loads(args.report.read_text())
        cases = report["cases"]
    except (OSError, KeyError, ValueError) as exc:
        print(f"slo_check: cannot read report: {exc}", file=sys.stderr)
        raise SystemExit(2)

    budget = dict(DEFAULTS)
    if args.budget:
        try:
            budget.update(json.loads(args.budget.read_text()))
        except (OSError, ValueError) as exc:
            print(f"slo_check: cannot read budget: {exc}", file=sys.stderr)
            raise SystemExit(2)

    latencies = [float(c.get("retrieval_latency_ms") or 0.0) for c in cases]
    observed = {
        "retrieval_p50_ms": percentile(latencies, 0.50),
        "retrieval_p95_ms": percentile(latencies, 0.95),
        "retrieval_max_ms": max(latencies) if latencies else 0.0,
        "zero_result_share": (
            sum(1 for c in cases if not c.get("returned_memories")) / len(cases) if cases else 0.0
        ),
    }

    breaches = [key for key, ceiling in budget.items() if observed.get(key, 0.0) > ceiling]
    payload = {"cases": len(cases), "observed": observed, "budget": budget, "breaches": breaches}

    if args.json:
        print(json.dumps(payload, indent=1))
    else:
        print(f"cases {len(cases)}")
        for key, ceiling in budget.items():
            mark = "FAIL" if key in breaches else "ok"
            print(f"  {mark:4} {key:24} {observed[key]:>8.1f}  budget {ceiling}")

    raise SystemExit(1 if breaches else 0)


if __name__ == "__main__":
    main()
