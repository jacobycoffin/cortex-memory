"""Merge two memories with a diff, consent and undo (#29)

    python3 scripts/merge_review.py --db DB LEFT RIGHT [--merged "text"]
    python3 scripts/merge_review.py --db DB LEFT RIGHT --merged "text" --consent jacoby
    python3 scripts/merge_review.py --db DB --undo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import merge_review, path_query  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("left", nargs="?", default="")
    parser.add_argument("right", nargs="?", default="")
    parser.add_argument("--merged", default="")
    parser.add_argument("--consent", default="", help="who agreed; empty means preview only")
    parser.add_argument("--undo", action="store_true")
    parser.add_argument("--merge-id", default="")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.undo:
            result = merge_review.undo(store, args.merge_id or None)
            print(result.get("sentence") or result.get("reason"))
            return 0
        if not (args.left and args.right):
            parser.error("give LEFT and RIGHT, or --undo")
        left = path_query.resolve(store, args.left) or args.left
        right = path_query.resolve(store, args.right) or args.right
        proposal = merge_review.propose(store, left, right, merged_text=args.merged)
        if proposal is None:
            print("one of the memories does not exist")
            return 1
        print(proposal.describe())
        if not args.consent:
            print("\npreview only: pass --consent <name> to merge")
            return 0
        result = merge_review.apply(store, proposal, consent=True, consented_by=args.consent)
        print("\n" + result.describe())
        return 0 if result.ok else 1
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
