"""Who / where / when a fact applies (#30)

    python3 scripts/scope_capture.py --db DB              # coverage
    python3 scripts/scope_capture.py --db DB --backfill 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import scope_capture  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--backfill", type=int, default=0)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.backfill:
            print("backfilled:", scope_capture.backfill(store, limit=args.backfill))
        described = scope_capture.report(store)
        total = described["memories"] or 1
        print(f"{described['memories']} memories: {described['global']} global "
              f"({described['global_share'] * 100:.1f}%), who {described['who']}, "
              f"where {described['where']}, when {described['when']}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
