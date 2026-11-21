#!/usr/bin/env python3
"""Calibrate an abstention threshold against the frozen set.

For every frozen case, record the top-1 score and whether any relevant memory made the
top-k. Then report, for a range of thresholds, how many cases would be suppressed and how
many of those were hits — a threshold is only safe if the cases it silences were misses.

    python3 scripts/calibrate_abstention.py --db ~/.hermes/cortex/cortex.db --set ~/cortex-evaluations/frozen/frozen-v1.jsonl

Read-only against whatever database it is pointed at; use a copy of the live store.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))


def _load_package() -> None:
    spec = importlib.util.spec_from_file_location(
        "cortex", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the Cortex package")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cortex"] = module
    spec.loader.exec_module(module)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--set", type=Path, required=True)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--json", dest="json_out", type=Path)
    args = parser.parse_args()

    _load_package()
    from cortex.retrieval import MemoryRetriever  # noqa: PLC0415
    from cortex.store import CortexStore  # noqa: PLC0415

    cases = [json.loads(line) for line in args.set.read_text().splitlines() if line.strip()]
    store = CortexStore(args.db)
    retriever = MemoryRetriever(store)
    rows: list[dict[str, object]] = []
    try:
        for case in cases:
            results, _diag = retriever.search_detailed(case["query"], limit=args.k)
            relevant = set(case.get("relevant_memory_ids") or [])
            ids = [str(r.memory["id"]) for r in results]
            rows.append(
                {
                    "case_id": case["case_id"],
                    "top_score": float(results[0].score) if results else 0.0,
                    "hit": any(i in relevant for i in ids),
                }
            )
    finally:
        store.close()

    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=1))

    hits = [r for r in rows if r["hit"]]
    misses = [r for r in rows if not r["hit"]]
    print(f"cases {len(rows)}  hits {len(hits)}  misses {len(misses)}")
    print(
        f"top-1 score: hits mean {mean([float(r['top_score']) for r in hits]):.4f} "
        f"min {min(float(r['top_score']) for r in hits):.4f} | "
        f"misses mean {mean([float(r['top_score']) for r in misses]):.4f} "
        f"max {max(float(r['top_score']) for r in misses):.4f}"
    )
    print("\nthreshold  suppressed  of which hits  precision-of-silence")
    for threshold in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        below = [r for r in rows if float(r["top_score"]) < threshold]
        below_hits = [r for r in below if r["hit"]]
        precision = 1 - (len(below_hits) / len(below)) if below else 0.0
        print(
            f"{threshold:9.2f}  {len(below):10d}  {len(below_hits):13d}  {precision:19.3f}"
        )


if __name__ == "__main__":
    main()
