"""Demote core facts that have gone quiet (#123)

    python3 scripts/core_demote.py --db "$HOME/.hermes/cortex/cortex.db"
    python3 scripts/core_demote.py --db ... --apply [--days 90]

Dry run by default. Demotion removes `core_members` membership only: the memory stays active
and recallable, it just stops costing tokens on every turn.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex.promotion import demote_stale, history, stats, window_days  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--days", type=int, default=None, help="window (default CORTEX_CORE_WINDOW_DAYS or 90)")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        window = args.days if args.days is not None else window_days()
        print(f"window: {window} days; members {stats(store)}")
        stale = demote_stale(store, days=window, apply=False)
        print(f"stale: {len(stale)}")
        for reason, count in collections.Counter(reason.split(" since")[0] for _id, reason in stale).most_common():
            print(f"  {reason}: {count}")
        for memory_id, reason in stale[:10]:
            print(f"  {memory_id[:8]} {reason}")
        if args.apply and stale:
            demote_stale(store, days=window, apply=True)
            print(f"demoted {len(stale)}; members now {stats(store)}")
        elif stale:
            print("dry run - pass --apply to demote")
        recent = history(store, limit=5)
        if recent:
            print("recent membership events:")
            for row in recent:
                print(f"  {row['at'][:19]} {row['action']:7} {row['memory_id'][:8]} {row['reason'][:60]}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
