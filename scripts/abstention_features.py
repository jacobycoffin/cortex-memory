#!/usr/bin/env python3
"""Which signal, if any, can tell a good recall from a bad one?

Score alone does not work (hits' top-1 mean 0.5894, misses' 0.5802, overlapping ranges). So
measure candidate features per frozen case and report AUC for each — 0.5 means useless, 1.0
means a clean separator. Only a feature that actually separates is worth building abstention
on.

    python3 scripts/abstention_features.py --db <copy.db> --set <frozen.jsonl> --out features.json
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

STOP = {
    "the", "and", "for", "with", "that", "this", "what", "when", "where", "which", "how",
    "does", "did", "was", "were", "are", "you", "your", "our", "about", "from", "have",
    "has", "had", "can", "could", "should", "would", "there", "here", "they", "them",
    "then", "than", "into", "onto", "over", "under", "just", "like", "want", "need",
}


def _load_package() -> None:
    spec = importlib.util.spec_from_file_location(
        "cortex", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the Cortex package")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cortex"] = module
    spec.loader.exec_module(module)


def tokens(text: str) -> set[str]:
    out = set()
    for raw in str(text or "").lower().replace("\n", " ").split():
        clean = "".join(ch for ch in raw if ch.isalnum() or ch in "-_")
        if len(clean) > 3 and clean not in STOP:
            out.add(clean)
    return out


def auc(values: list[tuple[float, bool]]) -> float:
    """Rank-based AUC: probability a hit scores above a miss."""
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    _load_package()
    from cortex.retrieval import MemoryRetriever  # noqa: PLC0415
    from cortex.store import CortexStore  # noqa: PLC0415

    cases = [json.loads(l) for l in args.set.read_text().splitlines() if l.strip()]
    store = CortexStore(args.db)
    retriever = MemoryRetriever(store)
    rows = []
    try:
        for case in cases:
            results, diag = retriever.search_detailed(case["query"], limit=args.k)
            relevant = set(case.get("relevant_memory_ids") or [])
            ids = [str(r.memory["id"]) for r in results]
            query_tokens = tokens(case["query"])
            coverages = []
            for result in results:
                body = tokens((result.memory or {}).get("content", ""))
                coverages.append(
                    len(query_tokens & body) / len(query_tokens) if query_tokens else 0.0
                )
            rows.append(
                {
                    "case_id": case["case_id"],
                    "query": case["query"],
                    "hit": any(i in relevant for i in ids),
                    "top_score": float(results[0].score) if results else 0.0,
                    "max_coverage": max(coverages) if coverages else 0.0,
                    "top_coverage": coverages[0] if coverages else 0.0,
                    "n_results": len(results),
                    "span": float(results[0].score - results[-1].score) if len(results) > 1 else 0.0,

                }
            )
    finally:
        store.close()

    args.out.write_text(json.dumps(rows, indent=1))

    print(f"cases {len(rows)}  hits {sum(1 for r in rows if r['hit'])}")
    for feature in ("top_score", "max_coverage", "top_coverage", "span", "n_results"):
        values = [(float(r[feature]), bool(r["hit"])) for r in rows]
        hits = [v for v, h in values if h]
        misses = [v for v, h in values if not h]
        print(
            f"{feature:14} AUC {auc(values):.3f}   hits mean {sum(hits)/len(hits):.3f}"
            f"   misses mean {sum(misses)/len(misses):.3f}"
        )


if __name__ == "__main__":
    main()
