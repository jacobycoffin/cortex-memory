"""Automatic write caps (#36)

    python3 scripts/write_throttle.py --db DB            # caps and busiest buckets
    python3 scripts/write_throttle.py --db DB --prune     # drop buckets older than 30 days
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import write_throttle  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.prune:
            print(f"pruned {write_throttle.prune(store, days=args.days)} buckets")
            return 0
        described = write_throttle.report(store)
        caps = described["caps"]
        print(f"caps: {caps['turn']}/turn, {caps['hour']}/hour, {caps['session']}/session")
        print(f"buckets: {described['buckets']}")
        for row in described["top"][:10]:
            print(f"  {row['writes']:>3}  {row['bucket']}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
