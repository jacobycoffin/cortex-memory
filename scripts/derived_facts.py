"""Derived facts and their dependencies (#57)

    python3 scripts/derived_facts.py --db DB                     # report
    python3 scripts/derived_facts.py --db DB --dirty             # what needs re-checking
    python3 scripts/derived_facts.py --db DB --revalidate ID
    python3 scripts/derived_facts.py --db DB --retract ID --reason "no longer follows"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import derived_facts  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--dirty", action="store_true")
    parser.add_argument("--revalidate", default="")
    parser.add_argument("--retract", default="")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.revalidate:
            print(derived_facts.revalidate(store, args.revalidate, note=args.reason))
            return 0
        if args.retract:
            print(derived_facts.retract(store, args.retract, reason=args.reason))
            return 0
        if args.dirty:
            items = derived_facts.dirty(store)
            print(f"{len(items)} inference(s) need re-checking")
            for item in items:
                print("  -", item.describe())
            return 0
        described = derived_facts.report(store)
        print(f"{described['derived']} inference(s): {described['dirty']} dirty, "
              f"{described['clean']} clean")
        for reason, count in sorted(described["reasons"].items()):
            print(f"  {reason:<10} {count}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
