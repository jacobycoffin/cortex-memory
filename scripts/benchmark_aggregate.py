#!/usr/bin/env python3
"""Aggregate repeated schema-v2 Cortex live benchmark runs."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

try:
    from Brain.benchmarks.core import summarize_latencies
except ModuleNotFoundError:
    from cortex.benchmarks.core import summarize_latencies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate repeated Cortex live paired benchmark JSON files.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    aggregate = aggregate_reports(reports, samples=args.bootstrap_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = args.output.with_suffix(".md")
    markdown = render_markdown(aggregate)
    markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"JSON: {args.output}")
    print(f"Report: {markdown_path}")
    return 0


def aggregate_reports(reports: Sequence[dict[str, Any]], *, samples: int = 10000) -> dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("aggregate at least two independent runs")
    first = reports[0]
    identity = (
        first["model"],
        first["corpus_memories"],
        first["paired_queries"],
        first.get("cortex_mode", "unknown"),
    )
    for report in reports:
        if report.get("schema_version") != 2:
            raise ValueError("all inputs must use live benchmark schema version 2")
        current = (
            report["model"],
            report["corpus_memories"],
            report["paired_queries"],
            report.get("cortex_mode", "unknown"),
        )
        if current != identity:
            raise ValueError("model, corpus size, paired queries, and Cortex mode must match across runs")

    seeds = [int(report["seed"]) for report in reports]
    bootstrap_seed = seeds[0]

    all_observations: list[dict[str, Any]] = []
    run_deltas: list[list[dict[str, float]]] = []
    run_summaries: list[dict[str, Any]] = []
    for index, report in enumerate(reports, start=1):
        observations = report["observations"]
        all_observations.extend(observations)
        pairs: dict[str, dict[str, dict[str, Any]]] = {}
        for row in observations:
            pairs.setdefault(row["label"], {})[row["condition"]] = row
        deltas = []
        for pair in pairs.values():
            if len(pair) != 2:
                continue
            deltas.append(
                {
                    "agent_ttft_ms": pair["cortex"]["agent_ttft_ms"]
                    - pair["default_built_in"]["agent_ttft_ms"],
                    "agent_total_ms": pair["cortex"]["agent_total_ms"]
                    - pair["default_built_in"]["agent_total_ms"],
                }
            )
        run_deltas.append(deltas)
        run_summaries.append(
            {
                "run": index,
                "run_at": report["run_at"],
                "seed": report["seed"],
                "cortex_accuracy": report["conditions"]["cortex"]["accuracy"],
                "default_accuracy": report["conditions"]["default_built_in"]["accuracy"],
                "cortex_prompt_tokens_median": report["conditions"]["cortex"][
                    "reported_prompt_tokens_median"
                ],
                "default_prompt_tokens_median": report["conditions"]["default_built_in"][
                    "reported_prompt_tokens_median"
                ],
                "agent_ttft_delta_median_ms": round(
                    statistics.median(delta["agent_ttft_ms"] for delta in deltas), 3
                ),
                "agent_total_delta_median_ms": round(
                    statistics.median(delta["agent_total_ms"] for delta in deltas), 3
                ),
            }
        )

    conditions: dict[str, Any] = {}
    for name in ("default_built_in", "cortex"):
        rows = [row for row in all_observations if row["condition"] == name]
        prompt_tokens = [row["reported_prompt_tokens"] for row in rows if row["reported_prompt_tokens"] is not None]
        conditions[name] = {
            "requests": len(rows),
            "accuracy": round(statistics.fmean(row["correct"] for row in rows), 6),
            "answer_availability": round(
                statistics.fmean(row["answer_available_in_context"] for row in rows), 6
            ),
            "memory_prepare": asdict(summarize_latencies([row["memory_prepare_ms"] for row in rows])),
            "agent_ttft": asdict(summarize_latencies([row["agent_ttft_ms"] for row in rows])),
            "agent_total_latency": asdict(summarize_latencies([row["agent_total_ms"] for row in rows])),
            "reported_prompt_tokens_median": statistics.median(prompt_tokens) if prompt_tokens else None,
        }

    pooled_ttft = [delta["agent_ttft_ms"] for run in run_deltas for delta in run]
    pooled_total = [delta["agent_total_ms"] for run in run_deltas for delta in run]
    return {
        "schema_version": 1,
        "benchmark": "cortex-live-paired-repeated-aggregate",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": identity[0],
        "corpus_memories": identity[1],
        "paired_queries_per_run": identity[2],
        "seeds": seeds,
        "cortex_mode": identity[3],
        "runs": len(reports),
        "total_pairs": sum(len(run) for run in run_deltas),
        "conditions": conditions,
        "paired_deltas_cortex_minus_default": {
            "agent_ttft_ms_pooled_median": round(statistics.median(pooled_ttft), 3),
            "agent_ttft_ms_hierarchical_bootstrap_median_95_ci": _hierarchical_bootstrap_ci(
                run_deltas, "agent_ttft_ms", samples=samples, seed=bootstrap_seed, statistic="median"
            ),
            "agent_ttft_ms_hierarchical_bootstrap_mean_95_ci": _hierarchical_bootstrap_ci(
                run_deltas, "agent_ttft_ms", samples=samples, seed=bootstrap_seed, statistic="mean"
            ),
            "agent_total_ms_pooled_median": round(statistics.median(pooled_total), 3),
            "agent_total_ms_hierarchical_bootstrap_median_95_ci": _hierarchical_bootstrap_ci(
                run_deltas, "agent_total_ms", samples=samples, seed=bootstrap_seed + 1, statistic="median"
            ),
            "agent_total_ms_hierarchical_bootstrap_mean_95_ci": _hierarchical_bootstrap_ci(
                run_deltas, "agent_total_ms", samples=samples, seed=bootstrap_seed + 1, statistic="mean"
            ),
            "interpretation": "Negative latency values favor Cortex; positive values favor built-in memory.",
        },
        "run_summaries": run_summaries,
    }


def render_markdown(report: dict[str, Any]) -> str:
    default = report["conditions"]["default_built_in"]
    cortex = report["conditions"]["cortex"]
    delta = report["paired_deltas_cortex_minus_default"]
    prompt_change = (cortex["reported_prompt_tokens_median"] / default["reported_prompt_tokens_median"]) - 1.0
    prompt_change_text = (
        f"Cortex used **{abs(prompt_change):.1%} fewer median prompt tokens**."
        if prompt_change < 0
        else f"Cortex used **{prompt_change:.1%} more median prompt tokens**."
    )
    lines = [
        "# Cortex repeated live benchmark",
        "",
        f"Model: `{report['model']}` · Cortex mode: `{report['cortex_mode']}` · runs: {report['runs']} · "
        f"{report['paired_queries_per_run']} paired questions/run · {report['total_pairs']} total pairs · "
        f"corpus: {report['corpus_memories']:,} memories",
        "",
        "| Condition | Accuracy | Answer available | Median prompt tokens | p50 agent TTFT | p95 agent TTFT | p50 agent total |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        _condition_row("Hermes built-in", default),
        _condition_row("Cortex", cortex),
        "",
        prompt_change_text,
        "",
        f"Pooled paired median agent TTFT difference: **{delta['agent_ttft_ms_pooled_median']:+.1f} ms**. "
        f"Hierarchical bootstrap median 95% CI: "
        f"{delta['agent_ttft_ms_hierarchical_bootstrap_median_95_ci'][0]:+.1f} to "
        f"{delta['agent_ttft_ms_hierarchical_bootstrap_median_95_ci'][1]:+.1f} ms.",
        "",
        f"Pooled paired median agent total-latency difference: "
        f"**{delta['agent_total_ms_pooled_median']:+.1f} ms**. Hierarchical bootstrap median 95% CI: "
        f"{delta['agent_total_ms_hierarchical_bootstrap_median_95_ci'][0]:+.1f} to "
        f"{delta['agent_total_ms_hierarchical_bootstrap_median_95_ci'][1]:+.1f} ms.",
        "",
        f"All requests are retained. Maximum observed total latency was "
        f"{cortex['agent_total_latency']['maximum_ms']:.1f} ms for Cortex and "
        f"{default['agent_total_latency']['maximum_ms']:.1f} ms for built-in memory; the median interval "
        "is reported because latency is outlier-prone. Mean intervals remain available in the raw JSON.",
        "",
        "## Per-run paired medians",
        "",
        "| Run | Cortex accuracy | Default accuracy | Agent TTFT delta | Agent total delta |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["run_summaries"]:
        lines.append(
            f"| {row['run']} | {row['cortex_accuracy']:.1%} | {row['default_accuracy']:.1%} | "
            f"{row['agent_ttft_delta_median_ms']:+.1f} ms | {row['agent_total_delta_median_ms']:+.1f} ms |"
        )
    lines.extend(
        [
            "",
            "Negative latency differences favor Cortex. The hierarchical bootstrap resamples runs and paired "
            "questions; with only three runs, treat the interval as a stability check rather than a universal "
            "provider-performance estimate.",
            "",
        ]
    )
    return "\n".join(lines)


def _condition_row(label: str, row: dict[str, Any]) -> str:
    return (
        f"| {label} | {row['accuracy']:.1%} | {row['answer_availability']:.1%} | "
        f"{row['reported_prompt_tokens_median']:,.1f} | {row['agent_ttft']['p50_ms']:.1f} ms | "
        f"{row['agent_ttft']['p95_ms']:.1f} ms | {row['agent_total_latency']['p50_ms']:.1f} ms |"
    )


def _hierarchical_bootstrap_ci(
    run_deltas: Sequence[Sequence[dict[str, float]]],
    metric: str,
    *,
    samples: int,
    seed: int,
    statistic: str,
) -> list[float]:
    rng = random.Random(seed)
    bootstrap_statistics: list[float] = []
    for _ in range(samples):
        sampled_values: list[float] = []
        for _ in run_deltas:
            selected_run = run_deltas[rng.randrange(len(run_deltas))]
            sampled_values.extend(
                selected_run[rng.randrange(len(selected_run))][metric] for _ in selected_run
            )
        if statistic == "median":
            bootstrap_statistics.append(statistics.median(sampled_values))
        elif statistic == "mean":
            bootstrap_statistics.append(statistics.fmean(sampled_values))
        else:
            raise ValueError(f"unsupported bootstrap statistic: {statistic}")
    bootstrap_statistics.sort()
    return [
        round(bootstrap_statistics[int(samples * 0.025)], 3),
        round(bootstrap_statistics[min(samples - 1, int(samples * 0.975))], 3),
    ]


if __name__ == "__main__":
    raise SystemExit(main())
