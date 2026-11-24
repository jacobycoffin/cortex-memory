"""Failures as questions, not rewrites (#208)

    python3 scripts/outcome_lessons.py --db DB                # open questions
    python3 scripts/outcome_lessons.py --db DB --persist      # record them
    python3 scripts/outcome_lessons.py --db DB --state open
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401

from cortex import outcome_lessons as ol  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--state", default="")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.persist:
            print(f"recorded {ol.persist(store)} lesson(s)")
        if args.state:
            for row in ol.lessons(store, state=args.state):
                print(f"  {row['state']:11s} {row['memory_id'][:8]}  {row['note']}")
            return 0
        for lesson in ol.as_lessons(store):
            print(f"  {lesson.question()}")
            print(f"    {lesson.content[:100]}")
        described = ol.report(store)
        print(f"{described['lessons']} lesson(s), {described['open']} open, "
              f"{described['outcomes_recorded']} outcome(s) recorded")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
