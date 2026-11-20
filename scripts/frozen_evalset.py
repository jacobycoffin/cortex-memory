#!/usr/bin/env python3
"""Rerun the frozen retrieval eval set in one command and gate on the baseline.

    python3 scripts/frozen_evalset.py run --db ~/.hermes/cortex/cortex.db

The labels and baseline stay outside the repository (private queries). The
runner refuses to score a set whose bytes no longer match its manifest, so a
"frozen" number cannot silently change under a moving case file. Cases whose
expected memories were archived or deleted since the freeze are retired and
counted, not scored as misses; if too few cases survive, the run fails instead
of reporting a shrunken, flattering number.

Exit codes: 0 = within tolerance (or baseline first pinned), 1 = regression,
2 = frozen set unusable (hash mismatch, too few cases, missing files).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn


ROOT = Path(__file__).resolve().parents[1]
RUNNER_VERSION = 1
DEFAULT_SET_DIR = Path(
    os.environ.get("CORTEX_FROZEN_DIR", str(Path.home() / "cortex-evaluations" / "frozen"))
)
DEFAULT_MIN_SURVIVING_CASES = 200

# A drop larger than this fails the run. Retrieval timing is reported but never
# gated: it varies with machine load and is not a quality signal.
TOLERANCES = {
    "hit_at_k": 0.01,
    "mean_recall_at_k": 0.01,
    "mrr": 0.02,
    "zero_result_rate": 0.01,  # increase, not decrease
}
HIGHER_IS_BETTER = ("hit_at_k", "mean_recall_at_k", "mrr")


def _load_cortex_package() -> None:
    if "cortex" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "cortex", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    if spec is None or spec.loader is None:
        raise SystemExit("could not load the Cortex package")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cortex"] = module
    spec.loader.exec_module(module)


_load_cortex_package()

from cortex.scripts.evaluate_real_history import (  # noqa: E402
    RetrievalLabel,
    evaluate,
    load_labels,
    private_database_snapshot,
)
from cortex.store import CortexStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen Cortex retrieval eval set and compare it to its pinned baseline."
    )
    parser.add_argument("run", nargs="?", default="run", choices=["run"], help="Action (only 'run').")
    parser.add_argument("--db", type=Path, required=True, help="Cortex database to evaluate (read-only snapshot).")
    parser.add_argument("--set-id", default="frozen-v1", help="Frozen set id (default %(default)s).")
    parser.add_argument("--set-dir", type=Path, default=DEFAULT_SET_DIR, help="Directory holding the frozen set.")
    parser.add_argument("--output", type=Path, help="Sanitized report path (default: <set-dir>/<set-id>-report.json).")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--token-budget", type=int, default=700)
    parser.add_argument("--policy", choices=("fixed", "adaptive"), default="adaptive")
    parser.add_argument(
        "--min-cases",
        type=int,
        default=DEFAULT_MIN_SURVIVING_CASES,
        help="Fail if fewer cases survive retirement than this (default %(default)s).",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Overwrite the baseline with this run. Only do this deliberately, never to silence a regression.",
    )
    return parser.parse_args()


def _abort(message: str, code: int = 2) -> "NoReturn":
    print(message, file=sys.stderr)
    raise SystemExit(code)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_set(set_dir: Path, set_id: str) -> tuple[list[RetrievalLabel], dict[str, Any], Path]:
    labels_path = set_dir / f"{set_id}.jsonl"
    manifest_path = set_dir / f"{set_id}.manifest.json"
    for path in (labels_path, manifest_path):
        if not path.is_file():
            _abort(f"frozen set file missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = _sha256(labels_path)
    if actual != manifest.get("labels_sha256"):
        _abort(
            f"frozen set {set_id} does not match its manifest "
            f"(expected {str(manifest.get('labels_sha256'))[:16]}..., found {actual[:16]}...); "
            "rebuild it with scripts/build_frozen_evalset.py and pin a new baseline"
        )
    labels = load_labels(labels_path)
    if len(labels) != manifest.get("case_count"):
        _abort(
            f"frozen set {set_id} has {len(labels)} cases but its manifest pins "
            f"{manifest.get('case_count')}"
        )
    return labels, manifest, labels_path


def retire_unretrievable(store: CortexStore, labels: list[RetrievalLabel]) -> tuple[list[RetrievalLabel], int]:
    """Drop cases whose expected memories are gone, archived, or not active."""

    kept: list[RetrievalLabel] = []
    retired = 0
    for label in labels:
        targets = [store.get_memory(memory_id) for memory_id in label.relevant_memory_ids]
        if any(target is None or str(target.get("state")) != "active" for target in targets):
            retired += 1
            continue
        kept.append(label)
    return kept, retired


def compare_to_baseline(summary: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    pinned = baseline.get("metrics", {})
    for metric in HIGHER_IS_BETTER:
        before, now = pinned.get(metric), summary.get(metric)
        if before is None or now is None:
            continue
        drop = float(before) - float(now)
        if drop > TOLERANCES[metric]:
            problems.append(f"{metric} {float(before):.4f} -> {float(now):.4f} (-{drop:.4f} > {TOLERANCES[metric]})")
    before_zero, now_zero = pinned.get("zero_result_rate"), summary.get("zero_result_rate")
    if before_zero is not None and now_zero is not None:
        rise = float(now_zero) - float(before_zero)
        if rise > TOLERANCES["zero_result_rate"]:
            problems.append(
                f"zero_result_rate {float(before_zero):.4f} -> {float(now_zero):.4f} "
                f"(+{rise:.4f} > {TOLERANCES['zero_result_rate']})"
            )
    return problems


def main() -> int:
    args = parse_args()
    set_dir = args.set_dir.expanduser()
    labels, manifest, labels_path = load_frozen_set(set_dir, args.set_id)
    baseline_path = set_dir / f"{args.set_id}.baseline.json"
    output_path = args.output or set_dir / f"{args.set_id}-report.json"

    database = args.db.expanduser()
    if not database.is_file():
        _abort(f"Cortex database does not exist: {database}")

    with private_database_snapshot(database) as snapshot_path:
        store = CortexStore(snapshot_path)
        try:
            kept, retired = retire_unretrievable(store, labels)
            if len(kept) < args.min_cases:
                _abort(
                    f"only {len(kept)} of {len(labels)} frozen cases are still retrievable "
                    f"(need {args.min_cases}); rebuild the frozen set before trusting a number"
                )
            report = evaluate(
                store,
                kept,
                top_k=args.top_k,
                token_budget=args.token_budget,
                policy=args.policy,
                min_cases=args.min_cases,
            )
        finally:
            store.close()

    report["frozen_set"] = {
        "set_id": args.set_id,
        "labels_sha256": manifest.get("labels_sha256"),
        "built_at": manifest.get("built_at"),
        "cases_pinned": len(labels),
        "cases_scored": len(kept),
        "cases_retired": retired,
        "runner_version": RUNNER_VERSION,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = report["summary"]
    print(
        f"{args.set_id}: {len(kept)}/{len(labels)} cases scored ({retired} retired), policy={args.policy} "
        f"hit@{args.top_k}={summary['hit_at_k']:.4f} recall@{args.top_k}={summary['mean_recall_at_k']:.4f} "
        f"mrr={summary['mrr']:.4f} zero={summary['zero_result_rate']:.4f}"
    )

    current = {
        "set_id": args.set_id,
        "labels_sha256": manifest.get("labels_sha256"),
        "pinned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "case_count": len(kept),
        "metrics": {
            key: summary[key] for key in ("hit_at_k", "mean_recall_at_k", "mean_precision_at_k", "mrr", "zero_result_rate")
        },
        "context_tokens_p50": summary["approximate_context_tokens"]["p50_ms"],
        "retrieval_latency_ms_p95": report["summary"]["retrieval_latency_ms"]["p95_ms"],
    }

    if args.update_baseline or not baseline_path.is_file():
        baseline_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        baseline_path.chmod(0o600)
        print(f"baseline pinned -> {baseline_path} ({'explicit update' if args.update_baseline else 'first run'})")
        return 0

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("labels_sha256") != manifest.get("labels_sha256"):
        print(
            f"baseline belongs to a different frozen set ({str(baseline.get('labels_sha256'))[:16]}... "
            f"vs {str(manifest.get('labels_sha256'))[:16]}...); pass --update-baseline for the new set",
            file=sys.stderr,
        )
        return 2
    problems = compare_to_baseline(summary, baseline)
    if problems:
        print("REGRESSION vs pinned baseline:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"within tolerance of baseline pinned {baseline.get('pinned_at')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
