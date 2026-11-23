"""Write-time dedup against the nearest memories (#28)

    python3 scripts/write_dedup.py --db DB --text "a fact about to be written"
    python3 scripts/write_dedup.py --db DB --survey 200   # what the live store would have refused
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import write_dedup  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--text", default="")
    parser.add_argument("--survey", type=int, default=0)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.text:
            verdict = write_dedup.decide(store, args.text)
            print(verdict.describe())
            for neighbour in verdict.neighbours[:3]:
                print(f"  neighbour {neighbour['memory_id'][:8]} {neighbour['similarity']:.2f} "
                      f"{neighbour['content'][:70]}")
            return 0
        if args.survey:
            rows = store._conn.execute(
                """SELECT id, content FROM memories
                   WHERE state = 'active' AND LENGTH(content) >= 24
                   ORDER BY created_at DESC LIMIT ?""",
                (int(args.survey),),
            ).fetchall()
            counts: dict[str, int] = {}
            for row in rows:
                verdict = write_dedup.decide(store, str(row["content"]),
                                            exclude=str(row["id"]))
                counts[verdict.action] = counts.get(verdict.action, 0) + 1
            total = sum(counts.values()) or 1
            print(f"{len(rows)} active memories re-checked against their neighbours:")
            for action, count in sorted(counts.items(), key=lambda item: -item[1]):
                print(f"  {action:<8} {count:5d} ({100.0 * count / total:5.1f}%)")
            return 0
        described = write_dedup.report(store)
        print("thresholds:", described["thresholds"])
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
