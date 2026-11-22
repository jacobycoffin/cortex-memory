"""An entity's history, oldest first (#50)

    python3 scripts/entity_timeline.py --db DB CT116
    python3 scripts/entity_timeline.py --db DB CT116 --changes    # only what changed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import entity_timeline  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("entity", nargs="?", help="entity name (e.g. CT116)")
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--changes", action="store_true")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    if not args.entity:
        print("pass an entity name")
        return 2

    store = CortexStore(Path(args.db))
    try:
        if args.changes:
            pairs = entity_timeline.changes(store, args.entity)
            print(f"{len(pairs)} change(s) for {args.entity}")
            for pair in pairs[: args.limit]:
                when = pair["when"].date().isoformat() if pair["when"] else "unknown"
                print(f"  {when}: {pair['was'][:60]!r} -> {pair['now'][:60]!r}")
            return 0
        print(entity_timeline.render(store, args.entity, limit=args.limit))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
