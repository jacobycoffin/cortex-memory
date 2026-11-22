"""Derived metadata may be regenerated; content may not (#69)

    python3 scripts/derived_metadata.py --db DB              # the split, and counts
    python3 scripts/derived_metadata.py --db DB --check ID    # fingerprint before/after a rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import derived_metadata  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--check", metavar="MEMORY_ID")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.check:
            before = derived_metadata.snapshot(store.get_memory(args.check) or {})
            result = derived_metadata.regenerate(store, args.check)
            print(result.describe())
            print("content fingerprint before:", derived_metadata.fingerprint(before["content"]))
            after = derived_metadata.snapshot(store.get_memory(args.check) or {})
            print("content fingerprint after: ", derived_metadata.fingerprint(after["content"]))
            return 0 if result.regenerated else 1
        described = derived_metadata.report(store)
        print(f"{described['memories']} memories, {described['derived_rows']} derived rows")
        print("regenerable: " + ", ".join(described["derived_fields"]))
        print("immutable:   " + ", ".join(described["authoritative_fields"]))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
