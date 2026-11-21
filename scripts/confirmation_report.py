"""How much corroboration was the store missing? (backlog item #216)

    python3 scripts/confirmation_report.py --db "$HOME/.hermes/cortex/cortex.db"

Reads only. For every active memory, asks the confirmation rules whether some *other*
memory restates it. Each pair found is a confirmation the old code would have thrown away
as a duplicate, and shows how much `confirmed_count` would move if capture had recorded it.

Slow by nature (one lexical lookup per memory), so it samples by default.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex.confirmation import corroboration_stats, find_confirmation, is_correction  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--sample", type=int, default=400, help="memories to test (0 = all)")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        stats = corroboration_stats(store)
        rows = [
            (str(row[0]), str(row[1]))
            for row in store._conn.execute(
                "SELECT id, content FROM memories WHERE state = 'active' ORDER BY id"
            ).fetchall()
        ]
        total = len(rows)
        if args.sample and args.sample < total:
            rows = random.Random(args.seed).sample(rows, args.sample)

        corroborated = 0
        corrections = 0
        by_reason: Counter[str] = Counter()
        for memory_id, content in rows:
            if is_correction(content):
                corrections += 1
                continue
            match = find_confirmation(store, content, exclude_id=memory_id)
            if match is not None and match.memory_id != memory_id:
                corroborated += 1
                by_reason[match.reason] += 1
    finally:
        store.close()

    share = corroborated / len(rows) if rows else 0.0
    print(f"active memories: {total} ({len(rows)} tested)")
    print(
        f"corroborated today: {stats['corroborated']}/{stats['total']} "
        f"({stats['corroborated'] / max(1, stats['total']):.1%}) "
        f"- confirmed {stats['confirmations']}, validated {stats['validations']}, helpful {stats['helpful']}"
    )
    print(f"restatements found: {corroborated}/{len(rows)} ({share:.1%})")
    for reason, count in by_reason.most_common():
        print(f"  {reason}: {count}")
    print(f"skipped as corrections: {corrections}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
