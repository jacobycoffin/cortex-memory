"""Which memories the answer actually repeated (#201)

    python3 scripts/citations.py --db DB                        # report
    python3 scripts/citations.py --db DB --memory <id>           # one memory's citations
    python3 scripts/citations.py --db DB --reply "..." --ids a,b  # detect, write nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401

from cortex import citations  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--memory", default="")
    parser.add_argument("--reply", default="")
    parser.add_argument("--ids", default="")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.reply and args.ids:
            ids = [part.strip() for part in args.ids.split(",") if part.strip()]
            print(citations.detect(store, args.reply, ids).describe())
            return 0
        if args.memory:
            rows = citations.citations(store, args.memory)
            print(f"{len(rows)} citation(s)")
            for row in rows[:5]:
                print(f"  {row['method']:6s} {row['created_at'][:19]} {row['phrase'][:60]}")
            return 0
        described = citations.report(store)
        print(f"{described['citations']} citation(s) {described['by_method']} "
              f"(min {described['min_words']} words)")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
