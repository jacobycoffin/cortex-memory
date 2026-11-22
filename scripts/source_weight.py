"""Source authority as a confidence ceiling (#41)

    python3 scripts/source_weight.py --db DB        # sources in the store and their ceilings
    python3 scripts/source_weight.py --over         # memories above their source's ceiling
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import source_weight  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--over", action="store_true", help="list rows above their ceiling")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.over:
            rows = store._conn.execute(
                """SELECT id, source_category, confidence, state FROM memories
                   WHERE state NOT IN ('tombstoned','deleted')"""
            ).fetchall()
            over = []
            for row in rows:
                ceiling = source_weight.authority(str(row["source_category"]))
                if float(row["confidence"]) > ceiling + 1e-9:
                    over.append((str(row["source_category"]), float(row["confidence"]), ceiling,
                                 str(row["state"])))
            print(f"{len(over)} of {len(rows)} memories above their source's ceiling")
            tally: dict[str, int] = {}
            for category, _value, _ceiling, _state in over:
                tally[category] = tally.get(category, 0) + 1
            for category, count in sorted(tally.items(), key=lambda item: -item[1]):
                print(f"  {category:<20} {count}")
            return 0
        described = source_weight.report(store)
        print(f"{described['total']} memories across {len(described['by_source'])} sources")
        for name, count in described["by_source"].items():
            ceiling = described["ceilings"][name]
            print(f"  {name:<22} {count:>6}  ceiling {ceiling:.2f}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
