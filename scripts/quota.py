"""Check the active-memory quota, or run the guard (#101)

    python3 scripts/quota.py --db DB            # report only
    python3 scripts/quota.py --db DB --sweep    # sweep if inside the warn band
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import quota  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--limit", type=int, help="override CORTEX_MAX_ACTIVE_MEMORIES")
    parser.add_argument("--sweep", action="store_true", help="run the guard")
    parser.add_argument("--max-triage", type=int, default=50)
    args = parser.parse_args()

    if args.limit:
        os.environ["CORTEX_MAX_ACTIVE_MEMORIES"] = str(args.limit)

    store = CortexStore(Path(args.db))
    try:
        print(quota.status(store).describe())
        if args.sweep:
            print(quota.ensure(store, max_triage=args.max_triage).summary)
            print(quota.status(store).describe())
        else:
            print("report only - pass --sweep to tidy")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
