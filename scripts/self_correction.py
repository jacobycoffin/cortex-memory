"""A user correcting themselves supersedes immediately (#33)

    python3 scripts/self_correction.py --db DB --message "No, it's port 8200 on the dashboard."
    python3 scripts/self_correction.py --db DB --message "I meant 8300." --target <memory-id>
    python3 scripts/self_correction.py --db DB --undo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import self_correction  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--message", default="")
    parser.add_argument("--target", default="")
    parser.add_argument("--undo", action="store_true")
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.undo:
            result = self_correction.undo(store)
            print(result.get("restored_id") or result.get("reason"))
            return 0
        if args.history:
            for entry in self_correction.history(store, limit=10):
                print(f"{entry['correction_id']} {entry['created_at'][:19]} "
                      f"replaced {str(entry['superseded_id'])[:8] or '-'}  "
                      f"{entry['statement'][:60]}")
            return 0
        if not args.message:
            print(self_correction.report(store))
            return 0
        found = self_correction.detect(args.message)
        print("detection:", found.describe())
        if not found.is_correction:
            return 0
        result = self_correction.intercept(store, args.message, target_id=args.target)
        print("result:", result.get("sentence") or result.get("reason"))
        return 0 if result.get("applied") else 1
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
