"""Conflicts as review items (#52)

    python3 scripts/contradiction_cluster.py --db DB             # stats
    python3 scripts/contradiction_cluster.py --db DB --pending
    python3 scripts/contradiction_cluster.py --db DB --resolve <topic> <winner-id> --by jacoby
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import contradiction_cluster  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--pending", action="store_true")
    parser.add_argument("--resolve", nargs=2, metavar=("TOPIC", "WINNER_ID"), default=None)
    parser.add_argument("--by", default="")
    parser.add_argument("--detect", action="store_true", help="report conflicts the detectors see")
    parser.add_argument("--record", action="store_true", help="file them as disputes")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.resolve:
            topic, winner = args.resolve
            match = [c for c in contradiction_cluster.clusters(store)
                     if c.topic == topic or any(m.startswith(topic) for m in c.members)]
            if len(match) != 1:
                print(f"no single cluster matches {topic!r}")
                return 1
            print(contradiction_cluster.resolve(store, match[0], winner, by=args.by))
            return 0
        if args.detect or args.record:
            outcome = contradiction_cluster.record_conflicts(store, apply=args.record)
            print(f"{outcome['found']} conflict(s) detected, {outcome['recorded']} recorded")
            for proposal in outcome["proposals"][:5]:
                print(f"  {proposal['left_id'][:8]} vs {proposal['right_id'][:8]}: "
                      f"{proposal['reason']}")
        described = contradiction_cluster.stats(store)
        print(f"{described['pairs_open']} open pair(s) in {described['clusters']} review item(s); "
              f"largest {described['largest']}, collapsed {described['collapsed']}")
        if args.pending:
            for cluster in contradiction_cluster.pending(store):
                print(f"  [{cluster['topic'] or 'unnamed'}] {cluster['size']} members - "
                      f"{cluster['recommendation']}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
