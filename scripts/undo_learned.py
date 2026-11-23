"""Undo a learned change (#213)

    python3 scripts/undo_learned.py --db DB              # what can be taken back
    python3 scripts/undo_learned.py --db DB --undo       # take back the latest step
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import undo_learned  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--undo", action="store_true")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.undo:
            result = undo_learned.undo(store, args.run_id or None)
            print(result.get("sentence") or result.get("reason"))
            for memory_id in result.get("restored") or []:
                print(f"  restored {memory_id[:8]}")
            return 0
        described = undo_learned.report(store)
        print(f"{described['steps']} steps recorded, {described['undoable']} undoable, "
              f"{described['undone']} already undone")
        for row in undo_learned.steps(store, limit=10):
            print(f"  {row['run_id']:<40} {row['module']:<24} {row['created_at'][:19]}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
