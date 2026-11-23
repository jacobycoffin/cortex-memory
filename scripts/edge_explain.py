"""Why are these two connected? (#85)

    python3 scripts/edge_explain.py --db DB LEFT RIGHT [--relation supports]
    python3 scripts/edge_explain.py --db DB --memory LEFT [--limit 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import edge_explain  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("left", nargs="?", default="")
    parser.add_argument("right", nargs="?", default="")
    parser.add_argument("--memory", default="")
    parser.add_argument("--relation", default="")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.memory:
            for explained in edge_explain.for_memory(store, args.memory, limit=args.limit):
                print("- " + explained.sentence)
            return 0
        if not (args.left and args.right):
            parser.error("give LEFT and RIGHT, or --memory")
        explained = edge_explain.explain(store, args.left, args.right, args.relation or None)
        print(explained.sentence)
        return 0 if explained.status != "absent" else 1
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
