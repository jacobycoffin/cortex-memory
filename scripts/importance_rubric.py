"""Importance from the rubric, and a backfill for memories stuck on the flat default (#32)

    python3 scripts/importance_rubric.py --db DB                  # active distribution
    python3 scripts/importance_rubric.py --db DB --fill [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import importance_rubric  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--fill", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.fill:
            print("filled:", importance_rubric.fill(store, limit=args.limit))
        described = importance_rubric.distribution(store)
        print(f"active {described['active']}, mean {described['mean']}, "
              f"flat {described['flat']}")
        for kind, value in described["by_kind"].items():
            print(f"  {kind:<12} {value}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
