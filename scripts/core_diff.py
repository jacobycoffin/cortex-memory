"""Show how core memory changed, and when (#124)

    python3 scripts/core_diff.py --db "$HOME/.hermes/cortex/cortex.db"
    python3 scripts/core_diff.py --db ... --history 10
    python3 scripts/core_diff.py --db ... --record      # snapshot now if it changed

Read-only unless `--record` is passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex.core_diff import diff, history, record  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--history", type=int, default=0, help="list the N most recent snapshots")
    parser.add_argument("--record", action="store_true", help="snapshot the block if it changed")
    parser.add_argument("--older", type=int, default=1, help="compare against the Nth older snapshot")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.record:
            written = record(store)
            print("recorded a new snapshot" if written else "block unchanged; nothing recorded")
        if args.history:
            rows = history(store, limit=args.history)
            if not rows:
                print("no snapshots recorded yet")
            for row in rows:
                print(
                    f"  {row.at[:19]}  {row.chars:5d} chars  "
                    f"{len(row.memory_ids):2d} memories  {row.block_hash[:12]}"
                )
            return 0
        change = diff(store, older_index=args.older)
        print(change.summary())
        for line in change.removed_lines:
            print(f"  - {line}")
        for line in change.added_lines:
            print(f"  + {line}")
        if change.removed_ids:
            print(f"  removed ids: {', '.join(i[:8] for i in change.removed_ids)}")
        if change.added_ids:
            print(f"  added ids:   {', '.join(i[:8] for i in change.added_ids)}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
