#!/usr/bin/env python3
"""Model-swap drill: memory quality must not depend on which chat model is running.

The thesis Cortex rests on is that a cheap model plus excellent memory still feels
smart. That is only true if recall is independent of the chat model. This drill proves
it rather than asserting it: it runs the frozen set twice — once under the current model
configuration, once under an alternate — and compares the metrics.

Retrieval is lexical plus embeddings; it never calls the chat model. So the two runs must
agree exactly. If they ever diverge, something has quietly become model-dependent and
recall now costs quality when the model changes.

    python3 scripts/model_swap_drill.py --db ~/.hermes/cortex/cortex.db \
        --set-dir ~/cortex-evaluations/frozen \
        --alt-model some-other-model

Exit codes: 0 invariants held, 1 the swap changed recall, 2 could not run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

METRICS = ("hit_at_k", "mean_recall_at_k", "mrr")
# Recall is deterministic here, so anything above float noise means model dependence.
TOLERANCE = 1e-9


def compare(baseline: dict, swapped: dict) -> tuple[bool, list[str]]:
    """Return (held, differences). Pure so it can be tested without running anything."""
    problems: list[str] = []
    for metric in METRICS:
        a = float(baseline.get(metric) or 0.0)
        b = float(swapped.get(metric) or 0.0)
        if abs(a - b) > TOLERANCE:
            problems.append(f"{metric}: {a:.6f} -> {b:.6f} (delta {b - a:+.6f})")
    return (not problems, problems)


def run_eval(
    db: Path, set_dir: Path, output: Path, model: str | None, provider: str | None
) -> dict:
    env = dict(os.environ)
    if model:
        # Both names exist in the codebase: the bench harness reads CORTEX_BENCH_MODEL,
        # the judge reads CORTEX_AUTO_JUDGE_MODEL. Set both so a swap is a real swap.
        env["CORTEX_BENCH_MODEL"] = model
        env["CORTEX_AUTO_JUDGE_MODEL"] = model
    if provider:
        env["CORTEX_BENCH_PROVIDER"] = provider
        env["CORTEX_AUTO_JUDGE_PROVIDER"] = provider

    runner = Path(__file__).with_name("frozen_evalset.py")
    proc = subprocess.run(
        [sys.executable, str(runner), "run", "--db", str(db), "--set-dir", str(set_dir),
         "--output", str(output)],
        env=env, capture_output=True, text=True,
    )
    if not output.exists():
        raise RuntimeError(proc.stderr.strip()[-400:] or "eval produced no report")
    report = json.loads(output.read_text())
    cases = report.get("cases") or []
    # The runner nests the numbers under "summary"; an empty metrics dict would make
    # every comparison pass vacuously, which is the one failure mode this drill must not have.
    metrics = report.get("summary") or report.get("metrics") or {}
    if not metrics:
        raise RuntimeError("eval report carried no summary metrics")
    return {
        "metrics": metrics,
        "cases": len(cases),
        "model": model or "(current)",
        "stderr_tail": proc.stderr.strip().splitlines()[-2:] if proc.stderr else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path.home() / ".hermes/cortex/cortex.db")
    parser.add_argument("--set-dir", type=Path,
                        default=Path.home() / "cortex-evaluations" / "frozen")
    parser.add_argument("--alt-model", required=True, help="Model to swap in")
    parser.add_argument("--alt-provider", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.db.exists() or not (args.set_dir / "frozen-v1.jsonl").exists():
        print("model_swap_drill: need a database and a frozen set", file=sys.stderr)
        raise SystemExit(2)

    with tempfile.TemporaryDirectory() as tmp:
        base_path = Path(tmp) / "baseline.json"
        swap_path = Path(tmp) / "swapped.json"
        try:
            base = run_eval(args.db, args.set_dir, base_path, None, None)
            swapped = run_eval(args.db, args.set_dir, swap_path, args.alt_model, args.alt_provider)
        except RuntimeError as exc:
            print(f"model_swap_drill: {exc}", file=sys.stderr)
            raise SystemExit(2)

    held, differences = compare(base["metrics"], swapped["metrics"])
    payload = {
        "baseline": {"model": base["model"], **base["metrics"], "cases": base["cases"]},
        "swapped": {"model": swapped["model"], **swapped["metrics"], "cases": swapped["cases"]},
        "held": held,
        "differences": differences,
    }

    if args.json:
        print(json.dumps(payload, indent=1))
    else:
        print(f"baseline  {base['model']:<28} {base['cases']} cases  "
              + "  ".join(f"{m}={float(base['metrics'].get(m, 0)):.4f}" for m in METRICS))
        print(f"swapped   {swapped['model']:<28} {swapped['cases']} cases  "
              + "  ".join(f"{m}={float(swapped['metrics'].get(m, 0)):.4f}" for m in METRICS))
        if held:
            print("HELD: recall is identical under a different chat model — memory does not "
                  "depend on it")
        else:
            print("CHANGED: recall moved when the model changed; something is model-dependent:")
            for line in differences:
                print(f"  - {line}")

    raise SystemExit(0 if held else 1)


if __name__ == "__main__":
    main()
