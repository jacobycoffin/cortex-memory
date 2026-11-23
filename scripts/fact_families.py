"""Fact families: variant phrasings of one fact (#56)

    python3 scripts/fact_families.py --db DB               # report
    python3 scripts/fact_families.py --db DB --build
    python3 scripts/fact_families.py --db DB --family <memory-id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import fact_families  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--scan", type=int, default=400)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--family", default="")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.family:
            found = fact_families.anchor(store, args.family)
            print(f"{found['size']} phrasing(s); freshest use {found['last_used'][:10] or 'never'}; "
                  f"{found['uses']} use(s) across the family")
            for member in found["members"]:
                row = store._conn.execute(
                    "SELECT content FROM memories WHERE id = ?", (member,)
                ).fetchone()
                print("  -", str(row["content"])[:90] if row else member)
            return 0
        if args.build or args.list:
            families = fact_families.build(store, scan=args.scan)
            grouped = [family for family in families if family.size > 1]
            print(f"{len(grouped)} family(ies) covering {sum(f.size for f in grouped)} memories")
            for family in grouped[:10]:
                print("  ", family.describe())
            return 0
        described = fact_families.report(store, scan=args.scan)
        print(f"{described['families']} family(ies) over {described['memories']} memories "
              f"(largest {described['largest']}, {described['singletons']} singletons); "
              f"aging {'on' if described['aging_enabled'] else 'off'}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
