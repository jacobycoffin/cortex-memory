#!/usr/bin/env python3
"""Run the reproducible Cortex-versus-built-in benchmark."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

try:
    from Brain.benchmarks.core import render_markdown, run_benchmark, write_report
except ModuleNotFoundError:
    from cortex.benchmarks.core import render_markdown, run_benchmark, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Cortex scalable recall with Hermes's bounded built-in memory snapshot."
    )
    parser.add_argument("--sizes", default="100,500,2000", help="Comma-separated corpus sizes.")
    parser.add_argument("--queries", type=int, default=200, help="Maximum labeled queries per corpus size.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic random seed.")
    parser.add_argument("--top-k", type=int, default=6, help="Cortex retrieval limit.")
    parser.add_argument("--token-budget", type=int, default=700, help="Cortex approximate recall token budget.")
    parser.add_argument("--default-char-limit", type=int, default=2200, help="Hermes MEMORY.md character limit.")
    parser.add_argument("--output", type=Path, help="JSON output path; a Markdown report is written beside it.")
    parser.add_argument("--no-write", action="store_true", help="Print only; do not create result files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sizes = [int(value.strip()) for value in args.sizes.split(",") if value.strip()]
    if not sizes or min(sizes) < 1 or args.queries < 1:
        raise SystemExit("sizes and queries must be positive")
    report = run_benchmark(
        sizes,
        query_count=args.queries,
        seed=args.seed,
        top_k=args.top_k,
        token_budget=args.token_budget,
        default_char_limit=args.default_char_limit,
    )
    markdown = render_markdown(report)
    print(markdown)
    if not args.no_write:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = args.output or ROOT / "benchmark-results" / f"cortex-compare-{timestamp}.json"
        json_path, markdown_path = write_report(report, output)
        print(f"JSON: {json_path}")
        print(f"Report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
