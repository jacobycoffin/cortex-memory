"""Co-retrieval edges, gated on three independent sessions (#67)

    python3 scripts/co_retrieval.py --db DB            # report
    python3 scripts/co_retrieval.py --db DB --apply    # materialise the edges that qualify
    python3 scripts/co_retrieval.py --db DB --prune --days 365
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import co_retrieval  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.prune:
            print(f"pruned {co_retrieval.prune_stale(store, days=args.days)} stale observations")
            return 0
        described = co_retrieval.report(store)
        print(
            f"pairs observed {described['pairs_observed']}, eligible {described['eligible']} "
            f"(needs {described['sessions_required']} sessions, cap {described['weight_cap']}, "
            f"half-life {described['half_life_days']:.0f}d)"
        )
        for row in described["top"]:
            print(f"  {row['pair']} sessions={row['sessions']} weight={row['weight']}")
        if args.apply:
            print("materialised:", co_retrieval.materialize(store))
        else:
            print("nothing has changed yet; pass --apply")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
