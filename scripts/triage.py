"""Batch triage for never-recalled memories (#92)

    python3 scripts/triage.py --db DB                     # queue size and the next batch
    python3 scripts/triage.py --db DB --apply BATCH        # archive that batch (the default action)
    python3 scripts/triage.py --db DB --undo BATCH         # one tap: restore exactly that batch
    python3 scripts/triage.py --db DB --keep id1,id2       # "I looked, they stay"

Archive is the default and the only destructive-ish option here; nothing is tombstoned.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import triage  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--size", type=int, default=triage.BATCH_SIZE)
    parser.add_argument("--apply", help="batch id to archive")
    parser.add_argument("--undo", help="batch id to restore")
    parser.add_argument("--keep", help="comma-separated ids to keep out of the queue")
    parser.add_argument("--history", action="store_true", help="recent batches")
    parser.add_argument("--action", default="archived", choices=["archived", "cold"])
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.history:
            rows = triage.batches(store)
            if not rows:
                print("no batches yet")
            for row in rows:
                state = "undone" if row["undone_at"] else ("applied" if row["applied_at"] else "pending")
                print(f"{row['batch_id']} {row['created_at'][:16]} {row['action']:9} "
                      f"{row['size']:3} memories {state}")
            return 0
        if args.apply:
            moved = triage.apply(store, args.apply)
            print(f"archived {len(moved)} memories; undo with --undo {args.apply}")
            return 0 if moved else 1
        if args.undo:
            restored = triage.undo(store, args.undo)
            print(f"restored {len(restored)} memories from {args.undo}")
            return 0 if restored else 1
        if args.keep:
            ids = [store.resolve_id(part.strip()) or part.strip()
                   for part in args.keep.split(",") if part.strip()]
            print(f"kept {triage.keep(store, ids)} memories out of the queue")
            return 0
        queue = triage.count_candidates(store)
        built = triage.batch(store, size=args.size, action=args.action)
        print(f"never-recalled queue: {queue} memories (minimum age {triage._min_age_days()}d)")
        if not built.items:
            print("nothing to triage")
            return 0
        print(built.describe())
        print(f"\napply with: --apply {built.batch_id}   (nothing has changed yet)")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
