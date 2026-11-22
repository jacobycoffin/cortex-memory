"""Restore memories, count restores, protect the ones that keep coming back (#99)

    python3 scripts/restore_ladder.py --db DB --restore <id>
    python3 scripts/restore_ladder.py --db DB --report

A memory restored twice is pinned permanently: two restores are not two opinions, they are a
policy that is wrong about that memory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import restore_ladder as rl  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--restore", help="memory id or prefix")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.restore:
            resolved = store.resolve_id(args.restore) or args.restore
            result = rl.restore(store, resolved)
            print(result.summary)
            return 0 if result.restored else 1
        described = rl.report(store, limit=args.limit)
        print(f"protect after {described['protect_at']} restores")
        for row in described["restored"]:
            flag = "PINNED" if row["pinned"] else ("due" if row["count"] >= described["protect_at"] else "")
            print(f"  {row['memory_id'][:8]} restores={row['count']} state={row['state']} "
                  f"{flag:6} {row['content'][:70]}")
        if not described["restored"]:
            print("no restores recorded")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
