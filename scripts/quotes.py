"""The sentence a fact came from (#42)

    python3 scripts/quotes.py --db DB                      # coverage report
    python3 scripts/quotes.py --db DB --find "nginx"        # which facts came from this phrase
    python3 scripts/quotes.py --db DB --verify <memory>     # does it still match its quote?
    python3 scripts/quotes.py --db DB --drifted             # audit shortlist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import quotes  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--find")
    parser.add_argument("--verify", metavar="MEMORY_ID")
    parser.add_argument("--drifted", action="store_true")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.find:
            found = quotes.find(store, args.find)
            print(f"{len(found)} quote(s) mention that")
            for item in found[:10]:
                print(f"  {item.subject_id[:8]} {item.describe()[:120]}")
            return 0
        if args.verify:
            print(quotes.verify(store, args.verify).describe())
            for quote in quotes.quotes_for(store, args.verify):
                print("  " + quote.describe()[:160])
            return 0
        if args.drifted:
            found = quotes.drifted(store)
            print(f"{len(found)} quoted memor(ies) drifted from their source sentence")
            for item in found[:10]:
                print(f"  {item['memory_id'][:8]} overlap {item['overlap']}")
            return 0
        described = quotes.report(store)
        print(f"{described['quotes']} quotes: {described['by_kind']}")
        print(f"coverage: {described['memories_with_quotes']}/{described['memories']} "
              f"memories ({described['coverage']:.1%})")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
