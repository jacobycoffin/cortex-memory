"""Refuse credentials instead of scrubbing them (#43)

    python3 scripts/secret_refusal.py "password: hunter2000"    # what would happen
    python3 scripts/secret_refusal.py --db DB                   # refusals recorded so far
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import secret_refusal  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="*")
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    args = parser.parse_args()

    if args.text:
        verdict = secret_refusal.assess(" ".join(args.text))
        print(verdict.action + ": " + verdict.describe())
        return 1 if verdict.refused else 0

    store = CortexStore(Path(args.db))
    try:
        described = secret_refusal.report(store)
        print(f"refusals: {described['refusals']}")
        if described["reasons"]:
            print("last reasons: " + ", ".join(described["reasons"]))
        if described["last"]:
            print("last at: " + described["last"])
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
