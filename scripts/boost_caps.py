"""Popularity may break a tie; it may not become truth (#203)

    python3 scripts/boost_caps.py --db DB
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import boost_caps  # noqa: E402
from cortex.store import CortexStore  # noqa: E402
from cortex.usage_weight import use_boost  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        described = boost_caps.report()
        for family, ceiling in described["caps"].items():
            print(f"  {family:<12} <= {ceiling}")
        print(f"  {'total':<12} <= {described['total_cap']} "
              f"(sum of families {described['sum_of_families']})")

        rows = store._conn.execute(
            """SELECT memory_id, COUNT(*) AS n FROM access_log
               WHERE event = 'used' GROUP BY memory_id ORDER BY n DESC LIMIT 200"""
        ).fetchall()
        if not rows:
            print("no recorded uses yet, so no usage boost is in play")
            return 0
        boosts = [use_boost(int(row["n"])) for row in rows]
        print(f"{len(rows)} most-used memories: largest usage boost {max(boosts):.3f} "
              f"(median {sorted(boosts)[len(boosts) // 2]:.3f}), most uses "
              f"{int(rows[0]['n'])}")
        stacked = max(boosts) + boost_caps.CAPS["entity"] + boost_caps.CAPS["session"]
        print(f"usage + entity + session stacked = {stacked:.3f}; the total cap binds at "
              f"{described['total_cap']} once a third family is in play")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
