"""Assign volatility to memories that predate the column (#31)

    python3 scripts/volatility.py --db DB            # report
    python3 scripts/volatility.py --db DB --backfill # classify every memory missing a class
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import volatility  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.backfill:
            volatility._ensure(store)
            query = (
                "SELECT m.id, m.content, m.kind FROM memories m "
                "LEFT JOIN memory_volatility v ON v.memory_id = m.id "
                "WHERE v.memory_id IS NULL ORDER BY m.created_at ASC"
            )
            if args.limit:
                query += f" LIMIT {int(args.limit)}"
            rows = store._conn.execute(query).fetchall()
            counts = {name: 0 for name in volatility.CLASSES}
            for row in rows:
                assigned = volatility.assign(
                    store, str(row["id"]), str(row["content"]), kind=str(row["kind"])
                )
                counts[assigned.volatility] += 1
            print(f"assigned {len(rows)}: {counts}")
            return 0
        described = volatility.report(store)
        print(f"assigned {described['assigned']}: {described['counts']}")
        print(f"window multipliers: {described['multipliers']}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
