"""Right-to-be-forgotten: preview, then purge a data subject (#97)

    python3 scripts/rtbf.py --db DB --subject "Jordan Reyes"        # preview only
    python3 scripts/rtbf.py --db DB --purge "Jordan Reyes"          # erase + certificate
    python3 scripts/rtbf.py --db DB --certificates

Erasure removes the memories **and every row derived from them** (features, traces, access logs,
edges, usage records, version history). It refuses if any affected memory is on legal hold. The
certificate stores a salted hash of the subject and the per-table counts, never the data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import rtbf  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--subject", help="the person or entity to erase")
    parser.add_argument("--purge", help="erase this subject (irreversible)")
    parser.add_argument("--note", default="", help="recorded on the certificate")
    parser.add_argument("--certificates", action="store_true")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.certificates:
            rows = rtbf.certificates(store)
            if not rows:
                print("no erasure certificates")
            for row in rows:
                print(f"{row['certificate']} {row['created_at'][:19]} "
                      f"{row['matched']} memories {row['purged']} {row['note']}")
            return 0
        if args.purge:
            try:
                result = rtbf.purge(store, args.purge, confirm=args.purge, note=args.note)
            except rtbf.PurgeRefused as error:
                print(f"refused: {error}")
                return 1
            print(
                f"erased {result['matched']} memories, {result['rows']} rows total; "
                f"certificate {result['certificate']}"
            )
            for table, count in sorted(result["purged"].items(), key=lambda item: -item[1]):
                print(f"  {table:28} {count}")
            return 0
        if not args.subject:
            print("--subject, --purge or --certificates is required")
            return 2
        planned = rtbf.preview(store, args.subject)
        print(planned.describe())
        print("\nnothing has changed yet; erase with --purge \"...\"")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
