"""Promote memories into core, or report who would be (#122)

    python3 scripts/core_promote.py --db "$HOME/.hermes/cortex/cortex.db"
    python3 scripts/core_promote.py --db ... --apply

Dry run by default. Promotion is reversible (there is no data loss — the memory itself is
untouched), but it is also what makes a fact cost ~460 tokens on every future turn, so the
list gets looked at before it is applied.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex.promotion import CONFIRMATION_THRESHOLD, USE_THRESHOLD, candidates, run, stats  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        print(f"thresholds: uses >= {USE_THRESHOLD}, confirmations >= {CONFIRMATION_THRESHOLD}")
        print(f"current members: {stats(store)}")
        pending = candidates(store)
        print(f"candidates: {len(pending)}")
        for kind, count in collections.Counter(p.promoted_by for p in pending).most_common():
            print(f"  {kind}: {count}")
        for promotion in pending[:10]:
            print(f"  {promotion.promoted_by:12} {promotion.memory_id[:8]} {promotion.reason}")
        if args.apply and pending:
            promoted = run(store, limit=args.limit)
            print(f"promoted: {len(promoted)}; members now {stats(store)}")
        elif pending:
            print("dry run - pass --apply to promote")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
