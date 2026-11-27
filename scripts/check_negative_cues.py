#!/usr/bin/env python3
"""End-to-end check of negative-cue suppression against a copy of the live store.

    python3 scripts/check_negative_cues.py --db /path/to/copy.db --term bench --query "bench press form"

Prints the top 5 for the plain query and for the same query with "not the <term>", so the
demotion is visible. Read-only against the copy; writes nothing back.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "cortex", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
assert _spec is not None and _spec.loader is not None
_pkg = importlib.util.module_from_spec(_spec)
sys.modules["cortex"] = _pkg
_spec.loader.exec_module(_pkg)

from cortex.retrieval import MemoryRetriever  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def run(store: CortexStore, retriever: MemoryRetriever, query: str) -> list[tuple[str, float, str]]:
    results, diagnostics = retriever.search_detailed(query, limit=5)
    suppressed = (getattr(diagnostics, "stage_ms", {}) or {}).get("negative_cue_suppressed_count")
    rows = []
    for result in results:
        content = str((result.memory or {}).get("content") or "")[:60].replace("\n", " ")
        rows.append((str((result.memory or {}).get("id"))[:8], float(result.score), content))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--term", required=True)
    args = parser.parse_args()

    store = CortexStore(args.db)
    retriever = MemoryRetriever(store)
    try:
        plain = run(store, retriever, args.query)
        negated = run(store, retriever, f"{args.query} (not the {args.term})")
    finally:
        store.close()

    def show(title: str, rows: list[tuple[str, float, str]]) -> None:
        print(f"\n{title}")
        for memory_id, score, content in rows:
            mark = "  <-- excluded term" if args.term.lower() in content.lower() else ""
            print(f"  {memory_id}  {score:.4f}  {content}{mark}")

    show(f"plain: {args.query}", plain)
    show(f"negated: {args.query} (not the {args.term})", negated)


if __name__ == "__main__":
    main()
