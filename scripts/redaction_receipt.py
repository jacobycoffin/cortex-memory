"""Redaction receipts (#44)

    python3 scripts/redaction_receipt.py --db DB                # what has been redacted, by label
    python3 scripts/redaction_receipt.py --db DB --subject ID   # receipts for one proposal/memory
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import redaction_receipt  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--subject")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.subject:
            receipts = redaction_receipt.for_subject(store, args.subject)
            print(f"{len(receipts)} receipt(s)")
            for receipt in receipts:
                print("  " + receipt.describe())
            return 0
        described = redaction_receipt.report(store)
        print(f"{described['redactions']} redaction(s) across {described['receipts']} receipt(s)")
        for label, count in described["by_label"].items():
            print(f"  {label:<20} {count}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
