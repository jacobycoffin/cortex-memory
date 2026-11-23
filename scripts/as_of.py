"""What the store believed on a date (#55)

    python3 scripts/as_of.py --db DB --when 2026-03-01
    python3 scripts/as_of.py --db DB --when 2026-03-01 --about dashboard
    python3 scripts/as_of.py --db DB --changes 2026-01-01 2026-03-01
    python3 scripts/as_of.py --db DB --coverage
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import as_of  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--when", default="")
    parser.add_argument("--about", default="")
    parser.add_argument("--changes", nargs=2, metavar=("FROM", "TO"), default=None)
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.coverage:
            described = as_of.coverage(store)
            print(f"{described['versions']} version(s) across {described['memories']} memories: "
                  f"{described['closed']} closed ({described['closed_share'] * 100:.1f}%), "
                  f"{described['open']} open; log from {described['earliest'][:10] or 'never'} "
                  f"to {described['latest'][:10] or 'never'}")
            return 0
        if args.changes:
            window = as_of.changes(store, args.changes[0], args.changes[1])
            print(window["sentence"])
            for row in window["learned"][: args.limit]:
                print("  +", str(row["content"])[:80])
            for row in window["retired"][: args.limit]:
                print("  -", str(row["content"])[:80])
            return 0
        if not args.when:
            parser.error("give --when, --changes or --coverage")
        answered = as_of.answer(store, args.when, query=args.about, limit=args.limit)
        print(answered["sentence"])
        for fact in answered["facts"]:
            print("  -", str(fact["content"])[:90])
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
