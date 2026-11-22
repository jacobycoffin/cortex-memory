"""No memory may hold more than N edges (#73)

    python3 scripts/degree_caps.py --db DB            # hubs and pending overflow
    python3 scripts/degree_caps.py --db DB --cap 32   # try a different cap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import degree_caps  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--cap", type=int, default=0)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        maximum = args.cap or None
        described = degree_caps.report(store, maximum=maximum)
        print(
            f"cap {described['cap']}, {described['edges']} edges, "
            f"{described['over_cap']} memories over it, {described['pending']} pending overflow"
        )
        for degree, content in described["worst"]:
            print(f"  {degree:>5}  {content}")
        for group in degree_caps.groups(store):
            print(f"  adoptable: {group['memory_id'][:8]} {group['relation']} "
                  f"{group['pending']} edges (mass {group['mass']})")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
