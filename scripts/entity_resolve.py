"""Entity resolution at write (#38)

    python3 scripts/entity_resolve.py --db DB                 # coverage + alias report
    python3 scripts/entity_resolve.py --db DB --seed 2000     # learn names from existing entities
    python3 scripts/entity_resolve.py --db DB --audit 1000    # how many mentions resolve today
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import aliases, entity_resolve  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--audit", type=int, default=0)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.seed:
            print("seeded:", aliases.from_memories(store, limit=args.seed))
        if args.audit:
            rows = store._conn.execute(
                """SELECT content FROM memories WHERE state = 'active'
                   ORDER BY created_at DESC LIMIT ?""",
                (int(args.audit),),
            ).fetchall()
            measured = entity_resolve.audit(store, [str(row["content"]) for row in rows])
            share = measured.get("resolved_share", 0.0)
            print(f"{measured['mentions']} mentions in {measured['texts']} memories: "
                  f"{measured['resolved']} resolved ({share * 100:.1f}%), "
                  f"{measured['ambiguous']} ambiguous, {measured['unknown']} unknown")
            return 0
        described = entity_resolve.report(store)
        print(f"{described['memories']} memories: {described['with_entities']} carry an entity "
              f"({described['empty']} do not); aliases: {described['aliases']}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
