"""Write receipts and the inline edits they invite (#37)

    python3 scripts/write_receipts.py --db DB --ids <id>,<id>   # the receipt line
    python3 scripts/write_receipts.py --db DB --reply "fix a1b2 new words"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import write_receipts  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--ids", help="comma separated memory ids to receipt")
    parser.add_argument("--reply", help="a message to interpret as inline edits (dry run)")
    parser.add_argument("--apply", action="store_true", help="with --reply, actually apply")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.reply:
            if not args.apply:
                print("parsed:", [edit.describe() for edit in write_receipts.parse(args.reply)])
                print("nothing has changed; pass --apply")
                return 0
            edits = write_receipts.apply_edits(store, args.reply)
            for edit in edits:
                print("  " + edit.describe())
            print(write_receipts.summarise(edits))
            return 0
        ids = [part.strip() for part in (args.ids or "").split(",") if part.strip()]
        if not ids:
            print("pass --ids or --reply")
            return 2
        receipt = write_receipts.build(store, ids)
        print(receipt.text or "(nothing to receipt)")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
