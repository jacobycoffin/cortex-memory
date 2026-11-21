"""Compare two ranking policies on answer quality, not retrieval metrics (#25).

    python3 scripts/ab_recall.py --db /root/.hermes/cortex/cortex.db --challenger no_entity_anchor

Cases come from the frozen set, which is private: real queries with real memory ids. It
lives outside the repository by policy and must stay there — this script reads it from
`--set` (default `~/cortex-evaluations/frozen/frozen-v1.jsonl`) and never writes case text
into the repo, a log, or CI.

The judge is a *coverage* judge by default: it asks whether the block that would be
injected actually carries the fact the case needs. That is an upper bound on answer
quality — it proves the model had what it needed, not that it used it — but it is free,
deterministic, and sensitive enough to catch a policy that drops the right memory.

Both policies run every case (paired), so case difficulty cancels out.

Exit codes: 0 no difference, 1 challenger worse, 2 challenger better, 3 could not run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex.ab_policy import POLICIES, compare, coverage_judge  # noqa: E402
from cortex.retrieval import MemoryRetriever  # noqa: E402
from cortex.store import CortexStore  # noqa: E402

OPTION_TARGETS = {
    "entity_anchor": "entity_anchor_enabled",
    "rerank": "rerank_enabled",
    "density_budget": "density_budget_enabled",
    "usage_boost": "usage_boost_enabled",
    "negative_cues": "negative_cues_enabled",
    "session_scope": "session_scope_enabled",
}


def apply_policy(retriever: MemoryRetriever, options: dict[str, object]) -> None:
    """Reset to the *shipped* defaults, then apply one policy's overrides.

    Resetting to "all True" would silently benchmark a configuration the product does not
    ship — rerank is off by default — so every reset reads the same environment switch the
    retriever reads.
    """
    retriever.entity_anchor_enabled = os.environ.get("CORTEX_ENTITY_ANCHOR", "1") != "0"
    retriever.density_budget_enabled = os.environ.get("CORTEX_DENSITY_BUDGET", "1") != "0"
    retriever.rerank_enabled = False
    retriever.usage_boost_enabled = True
    retriever.negative_cues_enabled = True
    retriever.session_scope_enabled = True
    for key, value in options.items():
        target = OPTION_TARGETS.get(key, key)
        setattr(retriever, target, value)


def gold_snippets(store: CortexStore, memory_ids: list[str], *, limit: int = 3) -> list[str]:
    """The text a correct answer needs, taken from the case's own relevant memories."""
    snippets = []
    for memory_id in memory_ids[:limit]:
        memory = store.get_memory(memory_id)
        if memory:
            snippets.append(str(memory["content"]))
    return snippets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--set", default=str(Path.home() / "cortex-evaluations" / "frozen" / "frozen-v1.jsonl"))
    parser.add_argument("--challenger", default="no_entity_anchor")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--token-budget", type=int, default=700)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    set_path = Path(args.set).expanduser()
    if not set_path.exists():
        print(f"frozen set not found: {set_path}", file=sys.stderr)
        return 3
    if not Path(args.db).exists():
        print(f"database not found: {args.db}", file=sys.stderr)
        return 3

    cases = [json.loads(line) for line in set_path.read_text().splitlines() if line.strip()]
    store = CortexStore(Path(args.db))
    retriever = MemoryRetriever(store)

    def run(policy: str, case: dict) -> tuple[float, str, float]:
        apply_policy(retriever, dict(POLICIES[policy]))
        started = time.perf_counter()
        results, _ = retriever.search_cached(
            case["query"], limit=args.limit, token_budget=args.token_budget
        )
        elapsed = (time.perf_counter() - started) * 1000
        block = "\n".join(str(result.memory["content"]) for result in results)
        gold = gold_snippets(store, case.get("relevant_memory_ids", []))
        score = coverage_judge(gold)(block) if gold else 0.0
        return score, block, elapsed

    # Warm the page cache and the revision-keyed memos on both policies first, otherwise
    # whichever policy runs first pays the cold reads and the latency comparison is rubbish.
    for warm in cases[:5]:
        for policy in (args.baseline, args.challenger):
            run(policy, warm)

    try:
        result = compare(
            run,
            [(case["case_id"], case) for case in cases],
            baseline=args.baseline,
            challenger=args.challenger,
        )
    finally:
        store.close()

    if args.json:
        print(
            json.dumps(
                {
                    "baseline": result.baseline,
                    "challenger": result.challenger,
                    "wins": result.wins,
                    "losses": result.losses,
                    "ties": result.ties,
                    "mean_delta": round(result.mean_delta, 4),
                    "ci": [round(result.ci_low, 4), round(result.ci_high, 4)],
                    "decisive": result.decisive,
                }
            )
        )
    else:
        print(result.summary())
        base_latency = sum(o.latency_ms[result.baseline] for o in result.per_case) / len(result.per_case)
        chal_latency = sum(o.latency_ms[result.challenger] for o in result.per_case) / len(result.per_case)
        print(f"latency: {result.baseline} {base_latency:.1f}ms, {result.challenger} {chal_latency:.1f}ms")
        covered = sum(o.scores[result.baseline] for o in result.per_case) / len(result.per_case)
        print(f"coverage: {result.baseline} {covered:.1%} of cases had the needed fact in the block")

    if not result.decisive:
        return 0
    return 2 if result.ci_low > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
