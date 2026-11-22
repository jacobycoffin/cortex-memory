"""How hard is a preference? (#59)

    python3 scripts/preference_strength.py --db DB "Never deploy on Fridays."
    python3 scripts/preference_strength.py --db DB --hard      # rules the agent may not deviate from
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import preference_strength as ps  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="*")
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--hard", action="store_true")
    args = parser.parse_args()

    if args.text:
        print(ps.classify(" ".join(args.text)).describe())
        return 0

    store = CortexStore(Path(args.db))
    try:
        if args.hard:
            rules = ps.hard_rules(store)
            print(f"{len(rules)} hard rule(s)")
            for rule in rules[:10]:
                print(f"  {rule['memory_id'][:8]} [{rule['signals']}] {rule['content'][:70]}")
            return 0
        described = ps.report(store)
        print(f"{described['recorded']} recorded: {described['hard']} hard, "
              f"{described['firm']} firm, {described['soft']} soft")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
