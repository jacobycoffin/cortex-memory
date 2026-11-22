"""Place, release and report legal holds (#103)

    python3 scripts/legal_hold.py --db DB --hold <id>[,<id>] --reason "signed contract"
    python3 scripts/legal_hold.py --db DB --hold <id> --reason "medical" --review-in 365
    python3 scripts/legal_hold.py --db DB --release <id> --reason "retention expired"
    python3 scripts/legal_hold.py --db DB --report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import legal_hold as lh  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--hold", help="comma-separated memory ids")
    parser.add_argument("--reason", default="legal hold")
    parser.add_argument("--authority", default="user")
    parser.add_argument("--review-in", type=int, default=0, help="days until review (0 = none)")
    parser.add_argument("--release", help="memory id to release")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.hold:
            ids = [part.strip() for part in args.hold.split(",") if part.strip()]
            review = lh.review_in(args.review_in) if args.review_in else None
            result = lh.place(store, ids, reason=args.reason, authority=args.authority,
                              review_at=review)
            print(result.summary)
            return 0 if result.placed else 1
        if args.release:
            resolved = store.resolve_id(args.release) or args.release
            print("released" if lh.release(store, resolved, reason=args.reason)
                  else "that memory is not on hold")
            return 0
        described = lh.report(store)
        print(f"held: {described['held']}, reviews due: {described['due_for_review']}")
        for line in described["holds"]:
            print("  " + line)
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
