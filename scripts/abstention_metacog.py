#!/usr/bin/env python3
"""Does the metacognitive assessment separate good recalls from bad ones?

The retrieval score does not (AUC 0.56 on the frozen set). `assess_retrieval` claims to
answer "should this evidence be trusted", which is closer to the question abstention needs.
Measure it before building anything on it.

    python3 scripts/abstention_metacog.py --db <copy.db> --set <frozen.jsonl>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

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


def auc(values: list[tuple[float, bool]]) -> float:
    positives = [v for v, hit in values if hit]
    negatives = [v for v, hit in values if not hit]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(positives) * len(negatives))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--set", type=Path, required=True)
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    _load_package()
    from cortex.metacognition import assess_retrieval  # noqa: PLC0415
    from cortex.retrieval import MemoryRetriever  # noqa: PLC0415
    from cortex.store import CortexStore  # noqa: PLC0415

    cases = [json.loads(l) for l in args.set.read_text().splitlines() if l.strip()]
    store = CortexStore(args.db)
    retriever = MemoryRetriever(store)
    rows = []
    try:
        for case in cases:
            results, _diag = retriever.search_detailed(case["query"], limit=args.k)
            if not results:
                continue
            relevant = set(case.get("relevant_memory_ids") or [])
            hit = any(str(r.memory["id"]) in relevant for r in results)
            assessments = [assess_retrieval(r) for r in results]
            rows.append(
                {
                    "hit": hit,
                    "top_confidence": float(assessments[0].calibrated_probability or 0.0),
                    "max_confidence": max(
                        float(a.calibrated_probability or 0.0) for a in assessments
                    ),
                    "top_raw": float(assessments[0].raw_probability or 0.0),
                }
            )
    finally:
        store.close()

    print(f"cases {len(rows)}  hits {sum(1 for r in rows if r['hit'])}")
    for feature in ("top_confidence", "max_confidence", "top_raw"):
        values = [(r[feature], r["hit"]) for r in rows]
        hits = [v for v, h in values if h]
        misses = [v for v, h in values if not h]
        print(
            f"{feature:16} AUC {auc(values):.3f}  hits {sum(hits)/max(1,len(hits)):.3f}"
            f"  misses {sum(misses)/max(1,len(misses)):.3f}"
        )


if __name__ == "__main__":
    main()
