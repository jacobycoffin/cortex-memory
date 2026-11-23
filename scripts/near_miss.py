"""Near-miss pairs: similar on purpose, wrong together (#80)

    python3 scripts/near_miss.py --db DB                  # recorded pairs and proposals
    python3 scripts/near_miss.py --db DB --record A B --reason "staging vs prod"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import near_miss, path_query  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--record", nargs=2, metavar=("LEFT", "RIGHT"), default=None)
    parser.add_argument("--reason", default="")
    parser.add_argument("--scan", type=int, default=400,
                        help="recent memories to compare; detection is O(pairs sharing tokens)")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.record:
            left = path_query.resolve(store, args.record[0]) or args.record[0]
            right = path_query.resolve(store, args.record[1]) or args.record[1]
            print("recorded:", near_miss.record(store, left, right, reason=args.reason))
        described = near_miss.report(store, scan=args.scan)
        print(f"{described['recorded']} recorded pairs ({described['by_user']} by a person), "
              f"{described['proposed']} proposed in the band {described['band']}")
        for example in described["examples"]:
            print("  propose: " + example)
        for row in near_miss.pairs(store, limit=10):
            print(f"  recorded: {row['left_id'][:8]} ~ {row['right_id'][:8]} "
                  f"[{row['recorded_by']}] {str(row['reason'])[:50]}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
