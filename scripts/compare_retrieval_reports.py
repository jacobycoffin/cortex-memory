#!/usr/bin/env python3
"""Compare aggregate retrieval quality and latency; never activate a policy.

Use reports produced from the same corpus, labels, settings, and host. A passing
comparison is evidence for review, not permission to apply Sleep proposals.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("report metrics must be finite numbers")
    return float(value)


def compare_reports(baseline: dict[str, Any], candidate: dict[str, Any], *,
                    max_latency_regression_percent: float = 20) -> dict[str, Any]:
    tolerance = _number(max_latency_regression_percent)
    if tolerance < 0:
        raise ValueError("latency regression tolerance must be nonnegative")
    evaluation = baseline.get("evaluation")
    if evaluation != candidate.get("evaluation"):
        raise ValueError("evaluation types must match")
    checks = []
    if evaluation == "synthetic-recall-performance":
        for key in ("schema_version", "memory_count", "seed", "repetitions"):
            if baseline.get(key) != candidate.get(key):
                raise ValueError("synthetic corpus and repetition settings must match")
        if int(candidate["repetitions"]) < 20:
            raise ValueError("performance comparisons require at least 20 repetitions")
        for condition in ("raw", "core", "hermes"):
            old, new = baseline["conditions"][condition], candidate["conditions"][condition]
            checks.append(_quality(condition, "synthetic_hit_at_k", old, new))
            for metric in ("p50_ms", "p95_ms"):
                checks.append(_latency(condition, metric, old, new, tolerance))
        for name, matched in candidate.get("correctness", {}).get("preloaded_selection_matches_uncached", {}).items():
            checks.append(dict(condition=name, metric="preloaded_selection_parity", passed=matched is True))
        sample = "synthetic"
    elif evaluation == "cortex-private-real-history-retrieval":
        for report in (baseline, candidate):
            if report.get("privacy", {}).get("raw_private_text_omitted") is not True:
                raise ValueError("private evaluation reports must pass their privacy self-check")
        for key in ("runner_version", "label_schema_version", "case_count", "policy", "max_top_k", "max_token_budget"):
            if baseline["reproducibility"].get(key) != candidate["reproducibility"].get(key):
                raise ValueError("private evaluation settings and case counts must match")
        if int(candidate["summary"]["cases"]) < 8:
            raise ValueError("private comparisons require at least eight labeled cases")
        for metric in ("hit_at_k", "mean_recall_at_k", "mean_precision_at_k", "mrr"):
            checks.append(_quality("retrieval", metric, baseline["summary"], candidate["summary"]))
        for metric in ("p50_ms", "p95_ms"):
            checks.append(_latency("retrieval", metric, baseline["summary"]["retrieval_latency_ms"],
                                   candidate["summary"]["retrieval_latency_ms"], tolerance))
        sample = "private_labeled_retrieval"
    else:
        raise ValueError("unsupported retrieval report type")
    return dict(evaluation="retrieval-report-comparison", evidence_type=sample,
                passed=all(check["passed"] for check in checks), checks=checks,
                claim_boundary="Same inputs and host required. Aggregate checks do not establish answer accuracy or authorize policy activation.")


def _quality(condition: str, metric: str, old: dict, new: dict) -> dict:
    before, after = _number(old[metric]), _number(new[metric])
    if not (0 <= before <= 1 and 0 <= after <= 1):
        raise ValueError("retrieval quality rates must be between zero and one")
    return dict(condition=condition, metric=metric, baseline=before, candidate=after,
                passed=after + 1e-9 >= before)


def _latency(condition: str, metric: str, old: dict, new: dict, tolerance: float) -> dict:
    before, after = _number(old[metric]), _number(new[metric])
    if before < 0 or after < 0:
        raise ValueError("latencies must be nonnegative")
    return dict(condition=condition, metric=metric, baseline=before, candidate=after,
                passed=after <= before * (1 + tolerance / 100))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--max-latency-regression-percent", type=float, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output and args.output.exists() and not args.overwrite:
        parser.error("output exists; pass --overwrite to replace it")
    try:
        report = compare_reports(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()),
                                 max_latency_regression_percent=args.max_latency_regression_percent)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
