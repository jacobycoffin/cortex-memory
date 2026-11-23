"""Show the question a change actually changes (#211)

    python3 scripts/lesson_demo.py --db DB --baseline BEFORE.db   # a real demonstration
    python3 scripts/lesson_demo.py --db DB                        # drift, clearly labelled
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import lesson_demo  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--baseline", default="",
                        help="a copy of the store from before the change")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    before = CortexStore(Path(args.baseline)) if args.baseline else None
    try:
        basis = "baseline comparison" if before is not None else "drift"
        demos = lesson_demo.demonstrations(store, limit=args.limit, only_changed=False,
                                           baseline=before)
        changed = [demo for demo in demos if demo.changed]
        print(f"{len(demos)} recorded turns, {len(changed)} answer differently ({basis})")
        if before is None:
            print("no --baseline given: differences are drift since the turn, not proof of a change")
        for demo in (demos if args.all else changed):
            print()
            print(demo.describe(store))
        return 0
    finally:
        store.close()
        if before is not None:
            before.close()


if __name__ == "__main__":
    raise SystemExit(main())
