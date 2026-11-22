"""Inspect the immutable recall log (#66)

    python3 scripts/recall_events.py --db DB                 # summary
    python3 scripts/recall_events.py --db DB --turn TURN     # one turn in detail
    python3 scripts/recall_events.py --db DB --check "query" # is this query text stored as memory?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import recall_events  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--turn", help="turn id to show")
    parser.add_argument("--check", help="query text that must not appear as a memory")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.check:
            leaked = recall_events.query_leaked_into_memory(store, args.check)
            print("leaked into: " + (", ".join(item[:8] for item in leaked) or "nothing"))
            return 1 if leaked else 0
        if args.turn:
            detail = recall_events.detail(store, args.turn)
            print(f"turn {detail['turn_id']}: ranked {detail['ranked']}, cited {detail['cited']}")
            for row in detail["results"]:
                mark = "cited" if row["used_in"] else "injected"
                print(f"  #{row['rank']} {row['memory_id'][:8]} score={row['score']} "
                      f"{row['retriever']} {mark}")
            return 0
        described = recall_events.summary(store)
        print(
            f"events {described['events']} across {described['turns']} turns "
            f"({described['ranked']} ranked, {described['citations']} citations)"
        )
        for turn in described["recent_turns"][:10]:
            print("  " + turn)
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
