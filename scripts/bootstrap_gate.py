"""Does a policy change beat noise? Paired bootstrap over recorded traces (#210)

    python3 scripts/bootstrap_gate.py --db DB                        # every policy vs the recorded run
    python3 scripts/bootstrap_gate.py --db DB --policy no_entity_boost
    python3 scripts/bootstrap_gate.py --a 0.1,0.2,0.3 --b 0.2,0.2,0.4

The per-case metric when replaying history is **stability**: did the trace's recorded top-1 survive
the policy change? That is not quality - it cannot be, the record does not say which memory was
*right* - and the script says so rather than dressing stability up as accuracy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401

from cortex import bootstrap_gate, shadow_rank  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def _stability(store, remove, limit: int, top_k: int) -> list[float]:
    """One case per trace: 1.0 if the recorded top-1 survives scoring with `remove`."""
    out: list[float] = []
    for trace in store.memory_traces(limit=int(limit)):
        candidates = shadow_rank._candidates(trace)
        if len(candidates) < 2:
            continue
        recorded = tuple(
            str(item["memory_id"]) for item in sorted(
                candidates, key=lambda item: (int(item.get("rank") or 0), str(item["memory_id"]))
            )
            if item.get("selected")
        )
        if not recorded:
            continue
        for item in candidates:
            item["_adjusted"] = shadow_rank._score(item, remove)
        shadow = shadow_rank._selected(candidates, len(recorded) if len(recorded) > 1 else top_k)
        out.append(1.0 if shadow_rank.record_top(recorded) == shadow_rank.record_top(shadow) else 0.0)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--policy", default="")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=shadow_rank.DEFAULT_TOP_K)
    parser.add_argument("--a", default="")
    parser.add_argument("--b", default="")
    args = parser.parse_args()

    if args.a and args.b:
        left = [float(part) for part in args.a.split(",") if part.strip()]
        right = [float(part) for part in args.b.split(",") if part.strip()]
        print(bootstrap_gate.paired_gate(left, right).describe())
        return 0

    store = CortexStore(Path(args.db))
    try:
        chosen = shadow_rank.policies()
        if args.policy:
            chosen = [item for item in chosen if item.name == args.policy]
            if not chosen:
                print(f"unknown policy; known: {[item.name for item in shadow_rank.policies()]}")
                return 2
        for policy in chosen:
            left = _stability(store, (), args.limit, args.top_k)
            right = _stability(store, policy.remove, args.limit, args.top_k)
            verdict = bootstrap_gate.paired_gate(left, right)
            print(f"{policy.name}: {verdict.describe()}")
        print("note: the metric is stability against the recorded run, not quality")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
