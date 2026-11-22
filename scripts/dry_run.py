"""What would you remember from this? (#46)

    python3 scripts/dry_run.py --db DB "some conversation text"
    python3 scripts/dry_run.py --db DB --file turn.txt --source AGENT_PROPOSED
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import dry_run  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="*")
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--file")
    parser.add_argument("--source", default="USER_EXPLICIT")
    parser.add_argument("--role", default="user")
    args = parser.parse_args()

    text = Path(args.file).read_text() if args.file else " ".join(args.text)
    if not text.strip():
        print("pass some text or --file")
        return 2

    store = CortexStore(Path(args.db))
    try:
        result = dry_run.project(
            store, text, role=args.role, source_category=args.source
        )
        print(dry_run.render(result))
        print("nothing was stored")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
