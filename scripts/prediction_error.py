"""Should this observation touch a memory at all (#68)

    python3 scripts/prediction_error.py --db DB --against <memory-id> --text "..."
    python3 scripts/prediction_error.py --db DB --text "..."     # gate against recent memories
    python3 scripts/prediction_error.py --db DB --audit          # how recent version pairs judge
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import prediction_error as pe  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--text", default="")
    parser.add_argument("--against", default="")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.audit:
            rows = store._conn.execute(
                """SELECT memory_id, content, system_from FROM memory_versions
                   WHERE content <> '' ORDER BY memory_id, system_from"""
            ).fetchall()
            previous: dict[str, str] = {}
            counts: Counter[str] = Counter()
            for memory_id, content, _at in rows:
                if memory_id in previous:
                    counts[pe.compare(previous[memory_id], str(content)).verdict] += 1
                previous[memory_id] = str(content)
            total = sum(counts.values())
            for verdict, count in counts.most_common():
                print(f"  {verdict:11s} {count:6d}  {count / max(1, total) * 100:5.1f}%")
            return 0
        if not args.text:
            print("give --text (and optionally --against <memory-id>) or --audit")
            return 2
        if args.against:
            found = pe.judge(store, args.against, args.text)
            print(found.describe())
            return 0
        decided = pe.gate(store, args.text)
        print(f"{decided['verdict']}: {decided['sentence']}")
        for candidate in decided["candidates"][:3]:
            print(f"  {candidate['verdict']:11s} {candidate['similarity']:.2f} {candidate['memory_id'][:8]}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
