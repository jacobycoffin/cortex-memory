"""Machine-edge provenance (#206)

    python3 scripts/edge_provenance.py --db DB
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import edge_provenance  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        described = edge_provenance.report(store)
        total = described["evidence_rows"] or 1
        print(f"{described['evidence_rows']} evidence rows:")
        for origin in ("human", "document", "machine"):
            count = described["by_origin"].get(origin, 0)
            share = 100.0 * count / total
            state = "citable" if origin in described["citable_origins"] else "not citable"
            print(f"  {origin:<9} {count:6d} ({share:5.1f}%)  {state}")
        print("machine types: " + ", ".join(described["machine_types"]))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
