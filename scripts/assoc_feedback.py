"""Association feedback (#79)

    python3 scripts/assoc_feedback.py --db DB --statement "X and Y are not related"
    python3 scripts/assoc_feedback.py --db DB                    # recorded judgements
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import assoc_feedback  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--statement", default="")
    parser.add_argument("--left", default="")
    parser.add_argument("--right", default="")
    parser.add_argument("--polarity", choices=("related", "unrelated"), default="")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.statement:
            applied = assoc_feedback.apply_statement(store, args.statement)
            print(applied.describe())
            return 0 if applied.ok else 1
        if args.left and args.right and args.polarity:
            applied = assoc_feedback.apply(store, args.left, args.right, args.polarity)
            print(applied.describe())
            return 0 if applied.ok else 1
        described = assoc_feedback.report(store)
        print(f"{described['judgements']} judgements: {described['related']} related, "
              f"{described['unrelated']} unrelated")
        for row in assoc_feedback.feedback(store, limit=10):
            print(f"  [{row['polarity']}] {row['left_id'][:8]} <-> {row['right_id'][:8]} "
                  f"{str(row['note'])[:60]}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
