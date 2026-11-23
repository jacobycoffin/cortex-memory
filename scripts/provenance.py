"""The evidence behind one fact (#51)

    python3 scripts/provenance.py --db DB --memory <id>
    python3 scripts/provenance.py --db DB --recent 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import provenance  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--memory", default="")
    parser.add_argument("--recent", type=int, default=0)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.recent:
            rows = store._conn.execute(
                "SELECT id, content FROM memories WHERE state = 'active' "
                "ORDER BY created_at DESC LIMIT ?",
                (int(args.recent),),
            ).fetchall()
            for row in rows:
                chain = provenance.chain(store, str(row["id"]))
                print(f"{str(row['content'])[:60]}")
                print(f"   {chain.describe()[:160]}")
            return 0
        if not args.memory:
            described = provenance.report(store)
            print(f"{described['steps']} linked step(s) across {described['memories']} memories:",
                  described["by_kind"])
            return 0
        chain = provenance.chain(store, args.memory)
        print(chain.describe())
        for step in chain.steps:
            print(f"  {step['at'][:19]}  {step['kind']:<13} {step['detail'][:80]}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
