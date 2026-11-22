"""Report memories that are offered many times and never used (#94)

    python3 scripts/dormancy.py --db DB [--limit 20]

Dormancy shortens the forget ladder's windows for these memories; this script only reports.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import dormancy  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        described = dormancy.report(store, limit=args.limit)
        print(
            f"dormant: {described['count']} memories "
            f"(min {described['min_retrieved']} offers, ratio >= {described['dormant_ratio']}, "
            f"windows x{described['scale']})"
        )
        for row in described["worst"]:
            print(
                f"  {row['memory_id'][:8]} r={row['retrieved']:3} u={row['used']:2} "
                f"ratio={row['ratio']:.2f} [{row['kind']}] {row['content'][:70]}"
            )
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
