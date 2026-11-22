"""Where the bytes are, per kind and per table, and what pruning would reclaim (#102)

    python3 scripts/storage_cost.py --db DB [--tables 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import storage_cost as sc  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--kinds", type=int, default=8)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        described = sc.report(store, limit=args.tables)
        total = described["file_bytes"]
        print(f"file {described['file_human']} · content {sc.human(described['content_bytes'])}")
        print("\nby table (indexes rolled in):")
        for table, size, share in described["tables"]:
            print(f"  {table:28} {sc.human(size):>10}  {share:5.1%}")
        print("\nby kind (content only):")
        for row in described["kinds"][: args.kinds]:
            print(
                f"  {row['kind']:16} {row['memories']:5} memories "
                f"{sc.human(row['bytes']):>9} avg {sc.human(row['average']):>7}"
            )
        reclaim = described["reclaim"]
        print(
            f"\nreclaim: {reclaim.archiveable} archiveable "
            f"({sc.human(reclaim.archive_bytes)} of content), "
            f"{reclaim.deletable} permanently hidden ({sc.human(reclaim.delete_bytes)} of content)"
        )
        for note in reclaim.notes:
            print(f"  note: {note}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
