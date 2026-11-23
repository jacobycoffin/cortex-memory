"""Session dedupe for associations (#204)

    python3 scripts/session_dedupe.py --db DB
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import session_dedupe  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        described = session_dedupe.report(store)
        print(f"bar: {described['min_sessions']} distinct sessions "
              f"(co_retrieval uses {described['strict_sessions']})")
        if not described["observations_table"]:
            print("no co-retrieval observations recorded at all")
        else:
            print(f"{described['pairs']} observed pairs: {described['single_session']} from one "
                  f"session, {described['multi_session']} from two or more")
            for example in described["busiest_single"][:5]:
                print(f"  one session: {example['left']} ~ {example['right']} "
                      f"({example['turns']} turns)")
        rows = store._conn.execute(
            """SELECT e.relation, ev.evidence_type, COUNT(DISTINCT e.src_id || e.dst_id) AS pairs
               FROM edges e JOIN edge_evidence ev
                 ON ev.src_id = e.src_id AND ev.dst_id = e.dst_id AND ev.relation = e.relation
               GROUP BY e.relation, ev.evidence_type
               ORDER BY pairs DESC LIMIT 8"""
        ).fetchall()
        print("where existing associations came from:")
        for row in rows:
            print(f"  {str(row['relation']):<18} {str(row['evidence_type']):<28} {row['pairs']}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
