"""Inspect or run the forget ladder (#86)

    python3 scripts/forget_ladder.py --db "$HOME/.hermes/cortex/cortex.db"        # dry run
    python3 scripts/forget_ladder.py --db ... --apply                              # walk one rung

Prints the states, then what would move this sweep and what is being held back and why.
`--apply` writes through `store.set_state`, so every move lands in `lifecycle_events`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import forget_ladder as ladder  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--apply", action="store_true", help="actually move memories")
    parser.add_argument("--limit", type=int, default=500, help="max per sweep")
    parser.add_argument("--per-transition", type=int, default=100,
                        help="max moves for any one transition in this sweep")
    parser.add_argument("--show", type=int, default=15, help="rows to print")
    parser.add_argument("--why", help="explain one memory id (prefix ok): why it is held or warm")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.why:
            import json as _json

            resolved = store.resolve_id(args.why) or args.why
            print(_json.dumps(ladder.explain(store, resolved), indent=2, sort_keys=True))
            return 0
        described = ladder.policy()
        print("policy:", described["version"], "| never walks:", ", ".join(described["never_walk"]))
        print(
            "windows:",
            ", ".join(
                f"{kind} {'/'.join(str(day) for day in dwell)}d"
                for kind, dwell in described["dwell_days"].items()
            ),
        )
        from cortex import regret as _regret

        print("regret guard:", _regret.revivals(store) or "no revivals recorded")
        print("states:", ladder.state_counts(store))
        sweep = ladder.plan(store, limit=args.limit, max_per_transition=args.per_transition)
        print("planned:", sweep.summary)
        if sweep.protected:
            print("held:", ", ".join(f"{k} {v}" for k, v in sorted(sweep.protected.items())))
        if sweep.skipped:
            print("skipped:", ", ".join(f"{k} {v}" for k, v in sorted(sweep.skipped.items())))
        if sweep.protection_failures:
            print(
                "PROTECTION FAILURES (these are not protecting anything):",
                "; ".join(f"{k}: {v}" for k, v in sweep.protection_failures.items()),
            )
        for step in sweep.steps[: args.show]:
            print("  ", step.describe())
        if args.apply:
            print("applied:", ladder.apply(store, sweep) or "nothing")
            print("states:", ladder.state_counts(store))
        else:
            print("dry run - pass --apply to move them")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
