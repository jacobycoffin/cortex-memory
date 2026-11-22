"""Preview and apply a semantic forget (#89)

    python3 scripts/forget_preview.py --db DB --query "the old NAS"          # preview only
    python3 scripts/forget_preview.py --db DB --apply ENVELOPE --ids a,b,c   # apply by exact id
    python3 scripts/forget_preview.py --db DB --history                      # audit trail

An apply only accepts ids that the named envelope previewed, before it expires (30 minutes by
default), and refuses protected memories unless `--force`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import forget_preview as fp  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--query", help="what to forget, in words")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--apply", help="envelope to apply")
    parser.add_argument("--ids", help="comma-separated ids (prefixes resolved)")
    parser.add_argument("--force", action="store_true", help="allow protected memories")
    parser.add_argument("--history", action="store_true", help="recent envelopes")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.history:
            for row in fp.envelopes(store):
                print(
                    f"{row['envelope'][:12]} {row['created_at'][:16]} "
                    f"previewed {row['previewed']:2} applied {row['applied']:2}"
                    f"{' EXPIRED' if row['expired'] else ''}  {row['query'][:60]}"
                )
            return 0
        if args.apply:
            if not args.ids:
                print("--ids is required with --apply")
                return 2
            ids = []
            for raw in args.ids.split(","):
                token = raw.strip()
                if not token:
                    continue
                ids.append(store.resolve_id(token) or token)
            result = fp.apply(store, args.apply, ids, allow_protected=args.force)
            print(result.summary)
            return 0 if result.ok else 1
        if not args.query:
            print("--query, --apply or --history is required")
            return 2
        preview = fp.preview(store, args.query, limit=args.limit)
        print(preview.describe())
        print(f"\n{len(preview.matches)} memories would be tombstoned - nothing has changed yet")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
