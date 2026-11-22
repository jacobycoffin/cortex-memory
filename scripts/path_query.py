"""How are these two related? (#75)

    python3 scripts/path_query.py --db DB "<phrase or id>" "<phrase or id>"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import path_query  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--depth", type=int, default=path_query.DEFAULT_DEPTH)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        path = path_query.between(store, args.left, args.right, depth=args.depth)
        if not path.found:
            print("no path:", path.explain())
            for neighbour in path.shared_neighbours[:5]:
                print(f"  shared neighbour: {neighbour}")
            return 1
        print(f"{len(path.hops)} hop(s)")
        for index, hop in enumerate(path.hops, start=1):
            print(f"  {index}. {hop.content_src[:60]}")
            print(f"     --{hop.describe()}")
            print(f"     -> {hop.content_dst[:60]}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
