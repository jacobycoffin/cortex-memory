"""Is this statement believed, denied, or unknown? (#58)

    python3 scripts/negative_facts.py --db DB "The box runs Debian."
    python3 scripts/negative_facts.py --db DB --list        # recorded negatives
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import negative_facts  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("statement", nargs="*")
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.list:
            described = negative_facts.report(store)
            print(f"{described['recorded']} recorded: {described['negative']} negative, "
                  f"{described['affirmative']} affirmative")
            for item in negative_facts.negatives(store)[:10]:
                print(f"  {item['memory_id'][:8]} [{item['negator']}] {item['content'][:70]}")
            return 0
        statement = " ".join(args.statement)
        if not statement:
            print("pass a statement or --list")
            return 2
        answer = negative_facts.answer(store, statement)
        print(f"{answer['verdict']}: {statement}")
        if answer["support"]:
            print("  supported by: " + ", ".join(item[:8] for item in answer["support"]))
        if answer["deny"]:
            print("  denied by:    " + ", ".join(item[:8] for item in answer["deny"]))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
