"""Is the box quiet enough for the expensive pass (#212)

    python3 scripts/idle_window.py --db DB                 # verdict + history
    python3 scripts/idle_window.py --db DB --min-idle 900
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401

from cortex import idle_window  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--min-idle", type=int, default=idle_window.DEFAULT_IDLE_SECONDS)
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        described = idle_window.report(store, min_idle_seconds=args.min_idle)
        idle = described["activity"]["idle_seconds"]
        print(f"{'RUN' if described['run'] else 'WAIT'}: {described['reason']}")
        print(f"  last activity {idle:.0f}s ago ({described['activity']['last_seen'] or 'never'})")
        print(f"  {described['runs']} recorded run(s) {described['by_status']}")
        return 0 if described["run"] else 1
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
