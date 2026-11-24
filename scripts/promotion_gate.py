"""Would this generalisation be allowed (#105)

    python3 scripts/promotion_gate.py --db DB                          # report
    python3 scripts/promotion_gate.py --db DB --statement "..."        # verdict
    python3 scripts/promotion_gate.py --db DB --audit 20               # do existing semantic facts qualify?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import promotion_gate  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--statement", default="")
    parser.add_argument("--audit", type=int, default=0)
    parser.add_argument("--overlap", type=int, default=4)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.statement:
            verdict = promotion_gate.ready(store, args.statement, overlap=args.overlap)
            print(verdict.describe())
            return 0 if verdict.ready else 1
        if args.audit:
            rows = store._conn.execute(
                """SELECT content FROM memories WHERE kind = 'semantic' AND state = 'active'
                   AND length(content) BETWEEN 40 AND 160 ORDER BY created_at DESC LIMIT ?""",
                (int(args.audit),),
            ).fetchall()
            passed = 0
            for row in rows:
                verdict = promotion_gate.ready(store, str(row[0]), overlap=args.overlap)
                passed += verdict.ready
                print(f"  {'OK ' if verdict.ready else 'no '} {str(row[0])[:60]}")
            print(f"{passed}/{len(rows)} semantic facts would pass the gate today")
            return 0
        described = promotion_gate.report(store)
        print(f"{described['promoted']} promoted claim(s) with {described['citations']} citation(s); "
              f"{described['episodes_available']} episodes available")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
