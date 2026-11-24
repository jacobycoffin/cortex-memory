"""Context as a retrieval key (#83)

    python3 scripts/context_tags.py --db DB                 # coverage report
    python3 scripts/context_tags.py --db DB --project cortex --limit 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import context_tags as ct  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--project", default="")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.project:
            ids = [str(row[0]) for row in store._conn.execute(
                """SELECT id FROM memories WHERE state = 'active' AND scope_json LIKE '%"project"%'
                   ORDER BY updated_at DESC LIMIT ?""", (int(args.limit),)).fetchall()]
            for row in ct.rank(store, ids, ct.context(project=args.project)):
                print(f"  x{row['multiplier']:5.2f} {row['memory_id'][:8]} {row['sentence']}")
            return 0
        described = ct.report(store)
        print(f"{described['sampled']} active memories sampled: "
              f"project {described['scoped']['project']}, who {described['scoped']['who']}, "
              f"where {described['scoped']['where']}, when {described['scoped']['when']}, "
              f"unscoped {described['unscoped']} ({described['coverage']:.1%} coverage)")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
