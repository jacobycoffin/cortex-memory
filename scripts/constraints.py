"""Budgets, allergies and hard nos, always in view (#60)

    python3 scripts/constraints.py --db DB            # how many, by kind, and the block
    python3 scripts/constraints.py --db DB --backfill # promote existing memories
    python3 scripts/constraints.py --db DB --check "book a shellfish restaurant"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import constraints  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--check", metavar="TEXT")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.backfill:
            rows = store._conn.execute(
                """SELECT id, content FROM memories
                   WHERE state IN ('active','cold') AND length(content) <= 400"""
            ).fetchall()
            promoted = 0
            for row in rows:
                if constraints.promote_from_strength(store, str(row["id"]),
                                                     str(row["content"])):
                    promoted += 1
            print(f"promoted {promoted} of {len(rows)} scanned")
        if args.check:
            found = constraints.relevant(store, args.check)
            print(f"{len(found)} constraint(s) bear on that")
            for item in found:
                print(f"  [{item.kind}] {item.content[:80]}")
            return 0
        described = constraints.report(store)
        print(f"{described['constraints']} constraints: {described['by_kind']}")
        block = constraints.context_block(store)
        if block:
            print(block)
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
