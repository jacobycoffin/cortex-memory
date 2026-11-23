"""Backfill derived fields for rows written before they existed (#45)

    python3 scripts/backfill.py --db DB                  # plan only
    python3 scripts/backfill.py --db DB --run --limit 2000
    python3 scripts/backfill.py --db DB --verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import backfill  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--only", default="")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--bump", type=int, default=0)
    args = parser.parse_args()

    only = [name.strip() for name in args.only.split(",") if name.strip()]
    store = CortexStore(Path(args.db))
    try:
        if args.verify:
            checked = backfill.verify(store)
            for name, count in sorted(checked["remaining"].items()):
                print(f"  {name:<20} {count}")
            print("clean" if checked["clean"] else "not clean: something is still missing")
            return 0 if checked["clean"] else 1
        planned = backfill.plan(store, only=only, limit=args.limit)
        print(f"schema version {planned['schema_version']}, {planned['total']} row(s) to fill")
        for line in planned["derivations"]:
            needs = "n/a" if line.get("needs") is None else line["needs"]
            flag = " (not derivable)" if line.get("not_derivable") else ""
            print(f"  {line['name']:<20} fills {line['fills']:<20} needs {needs}{flag}")
        if args.run:
            outcome = backfill.run(store, only=only, limit=args.limit)
            print(outcome.describe())
        if args.bump:
            backfill.set_schema_version(store, args.bump)
            print(f"schema version now {args.bump}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
