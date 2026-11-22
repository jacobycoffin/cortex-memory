"""Backfill `scope.active_project` onto existing memories (#217)

    python3 scripts/project_backfill.py --db "$HOME/.hermes/cortex/cortex.db"
    python3 scripts/project_backfill.py --db ... --apply

Read-only by default: it prints how many memories it could place, and in which project, so
the plan can be eyeballed before anything is written. `--apply` performs the write.

The backfill exists because a capture-time fix only helps memories written after it ships,
and the store reached 3,829 active memories with **zero** project scopes.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex.project_scope import (  # noqa: E402
    apply_backfill,
    backfill_plan,
    project_coverage,
)
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--project", action="append", default=[], help="known project name")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--apply", action="store_true", help="write the plan")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        before = project_coverage(store)
        plan = backfill_plan(store, configured=args.project, limit=args.limit)
        print(
            f"active memories {before['total']}, scoped {before['scoped']}, "
            f"unscoped {before['missing']}"
        )
        print(f"plannable: {len(plan)} ({len(plan) / max(1, before['missing']):.1%} of unscoped)")
        for project, count in collections.Counter(p for _, p, _ in plan).most_common(10):
            print(f"  {project}: {count}")
        if args.apply and plan:
            changed = apply_backfill(store, plan)
            after = project_coverage(store)
            print(f"applied: {changed} rows; scoped {after['scoped']}/{after['total']}")
        elif plan:
            print("dry run - pass --apply to write")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
