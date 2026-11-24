"""A summary is a new node, never a replacement (#205)

    python3 scripts/summaries.py --db DB                         # report
    python3 scripts/summaries.py --db DB --memory <id>           # members + staleness
    python3 scripts/summaries.py --db DB --create --members a,b --statement "..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401

from cortex import summaries  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--memory", default="")
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--members", default="")
    parser.add_argument("--statement", default="")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.create:
            ids = [part.strip() for part in args.members.split(",") if part.strip()]
            print(summaries.create(store, ids, args.statement))
            return 0
        if args.memory:
            found = summaries.stale(store, args.memory)
            print(f"{len(summaries.members_of(store, args.memory))} member(s); "
                  f"{'stale: ' + found.reason if found.dirty else 'current'}")
            return 0
        described = summaries.report(store)
        print(f"{described['summaries']} summary(ies), {described['stale']} stale")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
