"""Two-pass extraction review queue (#27)

    python3 scripts/two_pass.py --db DB                  # what the cheap pass got wrong
    python3 scripts/two_pass.py --db DB --resolve ID accept
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import two_pass  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--resolve", default="")
    parser.add_argument("--resolution", default="accept")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.resolve:
            print("resolved:", two_pass.resolve(store, args.resolve, args.resolution))
        described = two_pass.report(store)
        print(f"{described['total']} disagreements recorded, {described['queued']} queued "
              f"({described['over_reach']} over-reach, {described['missed']} missed), "
              f"enabled={described['enabled']}")
        for row in two_pass.queue(store, limit=10):
            print(f"  [{row['kind']}] {row['disagreement_id']} {str(row['text'])[:90]}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
