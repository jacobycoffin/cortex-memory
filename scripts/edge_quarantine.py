"""Association quarantine (#84)

    python3 scripts/edge_quarantine.py --db DB            # is anything quarantined?
    python3 scripts/edge_quarantine.py --db DB --adopt    # set the epoch: grandfather what exists
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import edge_quarantine  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--adopt", action="store_true")
    parser.add_argument("--release", nargs=3, metavar=("SRC", "DST", "RELATION"), default=None)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.adopt:
            print("adopted at:", edge_quarantine.adopt(store))
        if args.release:
            print("released:", edge_quarantine.release(store, *args.release))
        described = edge_quarantine.report(store)
        print(f"epoch {described['adopted_at'] or '(none)'}: {described['edges']} edges, "
              f"{described['grandfathered']} grandfathered, "
              f"{described['quarantined_candidates']} created after the epoch, "
              f"witness bar {described['witness_bar']}")
        if not described["adopted_at"]:
            print("nothing is quarantined until an epoch is adopted; pass --adopt")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
