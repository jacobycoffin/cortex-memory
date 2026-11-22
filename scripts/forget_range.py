"""Forget a period, with a preview (#96)

    python3 scripts/forget_range.py --db DB --period 2026-03
    python3 scripts/forget_range.py --db DB --apply ENVELOPE --ids a,b,c

Accepts 2026-03, 2026-03-01 or 2026-03-01..2026-03-31. Anything else is refused rather than
guessed. An apply only accepts ids the envelope previewed, and refuses on any legal hold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import forget_range as fr  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--period", help="2026-03, 2026-03-01 or 2026-03-01..2026-03-31")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--apply", help="envelope to apply")
    parser.add_argument("--ids", help="comma-separated ids (prefixes resolved)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.apply:
            if not args.ids:
                print("--ids is required with --apply")
                return 2
            ids = [store.resolve_id(part.strip()) or part.strip()
                   for part in args.ids.split(",") if part.strip()]
            result = fr.apply(store, args.apply, ids, allow_protected=args.force)
            print(result.summary)
            return 0 if result.ok else 1
        if not args.period:
            print("--period or --apply is required")
            return 2
        try:
            preview = fr.preview(store, args.period, limit=args.limit)
        except fr.RangeError as error:
            print(f"refused: {error}")
            return 2
        print(preview.describe())
        kinds = fr.summary(store, preview.matches)
        print("\nkinds: " + (", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) or "none"))
        print(f"{len(preview.matches)} memories would be tombstoned - nothing has changed yet")
        print(f"apply with: --apply {preview.envelope} --ids <ids>")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
