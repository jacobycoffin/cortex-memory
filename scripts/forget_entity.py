"""Forget everything about one entity, as one auditable operation (#95)

    python3 scripts/forget_entity.py --db DB --entity "Acme Corp"            # preview only
    python3 scripts/forget_entity.py --db DB --apply ENVELOPE --ids a,b,c    # apply by exact id

An apply only accepts ids the envelope previewed, and refuses if any of them is on legal hold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import forget_entity as fe  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--entity", help="the entity to forget, e.g. 'Acme Corp'")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--apply", help="envelope to apply")
    parser.add_argument("--ids", help="comma-separated ids (prefixes resolved)")
    parser.add_argument("--force", action="store_true", help="allow protected memories")
    parser.add_argument("--neighbours", action="store_true",
                        help="also list graph-connected memories (marked, not usually wanted)")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.apply:
            if not args.ids:
                print("--ids is required with --apply")
                return 2
            ids = [store.resolve_id(part.strip()) or part.strip()
                   for part in args.ids.split(",") if part.strip()]
            result = fe.apply(store, args.apply, ids, allow_protected=args.force)
            print(result.summary)
            return 0 if result.ok else 1
        if not args.entity:
            print("--entity or --apply is required")
            return 2
        preview = fe.preview(
            store, args.entity, limit=args.limit, include_neighbours=args.neighbours
        )
        print(preview.describe())
        print(f"\n{len(preview.matches)} memories would be tombstoned - nothing has changed yet")
        print(f"apply with: --apply {preview.envelope} --ids <ids>")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
