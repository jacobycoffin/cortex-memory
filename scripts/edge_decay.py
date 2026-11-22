"""Unused links fade; evidence does not (#72)

    python3 scripts/edge_decay.py --db DB            # what would fade
    python3 scripts/edge_decay.py --db DB --apply    # fade, recording the base weight
    python3 scripts/edge_decay.py --db DB --restore  # put every weight back
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import edge_decay  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.restore:
            print(f"restored {edge_decay.restore(store)} edge weights")
            return 0
        described = edge_decay.report(store)
        print(
            f"{described['edges']} edges, {described['would_fade']} stale beyond a half-life "
            f"(half-life {described['half_life_days']:.0f}d, floor {described['floor_ratio']}), "
            f"{described['evidence_rows']} evidence rows kept"
        )
        for line in described["steepest"]:
            print("  " + line)
        if args.apply:
            print("applied:", edge_decay.apply(store))
        else:
            print("nothing has changed; pass --apply")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
