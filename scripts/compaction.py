"""Compact the derivative rows of retired memories (#98)

    python3 scripts/compaction.py --db DB              # report only
    python3 scripts/compaction.py --db DB --apply      # delete the caches
    python3 scripts/compaction.py --db DB --apply --history   # also drop use evidence
    python3 scripts/compaction.py --db DB --vacuum     # return the pages (needs exclusive access)

Content, audit history and the structural recall rows are never touched; a restored memory is
re-indexed automatically, so compaction removes cache and not capability.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import compaction, storage_cost  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--history", action="store_true", help="also drop access/usage evidence")
    parser.add_argument("--vacuum", action="store_true")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        described = compaction.report(store)
        print(f"file {storage_cost.human(described['file_bytes'])}")
        print(
            f"{described['retired']} retired memories hold {described['rows']} derivative rows "
            f"(~{storage_cost.human(described['bytes'])})"
        )
        for table, count in described["tables"]:
            print(f"  {table:30} {count}")
        if args.apply:
            removed = compaction.compact(store, include_history=args.history)
            print("removed: " + (", ".join(f"{k} {v}" for k, v in sorted(removed.items())) or "nothing"))
        if args.vacuum:
            ok = compaction.vacuum(store)
            print("vacuum:", "done" if ok else "skipped (database busy)")
            print("file now", storage_cost.human(storage_cost.file_bytes(store)))
        if not args.apply:
            print("\nnothing has changed yet; pass --apply")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
