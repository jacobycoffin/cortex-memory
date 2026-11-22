"""Current, superseded or disputed (#48)

    python3 scripts/dispute.py --db DB                       # report
    python3 scripts/dispute.py --db DB --mark a,b --reason "vendor says otherwise"
    python3 scripts/dispute.py --db DB --resolve a,b [--winner b]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import dispute  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--mark", help="two ids: left,right")
    parser.add_argument("--resolve", help="two ids: left,right")
    parser.add_argument("--winner", help="id that wins the dispute (supersedes the other)")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.mark:
            left, _, right = args.mark.partition(",")
            record = dispute.mark(store, left.strip(), right.strip(), reason=args.reason)
            print(record.describe() if record else "could not mark that pair")
            return 0 if record else 1
        if args.resolve:
            left, _, right = args.resolve.partition(",")
            ok = dispute.resolve(store, left.strip(), right.strip(),
                                 winner=args.winner, reason=args.reason)
            print("resolved" if ok else "no open dispute for that pair")
            return 0 if ok else 1
        described = dispute.report(store)
        print(f"open disputes: {described['open']} ({described['memories']} memories)")
        for line in described["pairs"]:
            print("  " + line)
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
