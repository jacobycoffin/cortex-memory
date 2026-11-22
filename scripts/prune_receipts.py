"""Receipts for prunes: list, verify, or expire them (#100)

    python3 scripts/prune_receipts.py --db DB --recent 10
    python3 scripts/prune_receipts.py --db DB --verify <receipt>
    python3 scripts/prune_receipts.py --db DB --prune [--keep 10] [--days 90]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import prune_receipts as rc  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--recent", type=int, default=10)
    parser.add_argument("--kind", help="filter by kind")
    parser.add_argument("--verify", help="receipt id to check against the store")
    parser.add_argument("--prune", action="store_true", help="expire old receipts")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--keep", type=int, default=rc.KEEP_LAST)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.verify:
            checked = rc.verify(store, args.verify)
            for key, value in checked.items():
                print(f"{key}: {value}")
            return 0
        if args.prune:
            print(rc.prune(store, older_than_days=args.days, keep_last=args.keep))
            print(rc.stats(store))
            return 0
        described = rc.stats(store)
        print(
            f"receipts: {described['receipts']} "
            f"({', '.join(f'{k} {v}' for k, v in described['by_kind'].items()) or 'none'}); "
            f"pruned so far: {described['pruned_total']}"
        )
        for receipt in rc.recent(store, limit=args.recent, kind=args.kind):
            print("  " + receipt.describe())
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
