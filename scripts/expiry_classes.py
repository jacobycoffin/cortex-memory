"""Which facts are due a second look? (#64)

    python3 scripts/expiry_classes.py --db DB            # counts
    python3 scripts/expiry_classes.py --db DB --due      # what is overdue, with content
    python3 scripts/expiry_classes.py --db DB --backfill # assign classes to existing memories
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import expiry_classes  # noqa: E402
from cortex import volatility  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--due", action="store_true")
    parser.add_argument("--backfill", action="store_true")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.backfill:
            expiry_classes._ensure(store)
            rows = store._conn.execute(
                """SELECT m.id, m.kind FROM memories m
                   LEFT JOIN memory_expiry e ON e.memory_id = m.id
                   WHERE e.memory_id IS NULL AND m.state IN ('active','cold')"""
            ).fetchall()
            for row in rows:
                expiry_classes.assign(
                    store, str(row["id"]), kind=str(row["kind"]),
                    volatility_class=volatility.lookup(store, str(row["id"])),
                )
            print(f"assigned {len(rows)} memories")
        if args.due:
            items = expiry_classes.due(store, limit=20)
            print(f"{len(items)} due for review")
            for item in items:
                print(f"  {item['review_after'][:10]} [{item['expiry_class']}] "
                      f"{item['content'][:70]}")
            return 0
        described = expiry_classes.report(store)
        print(f"{described['assigned']} assigned, {described['due']} due: "
              f"{described['by_class']}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
