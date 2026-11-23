"""Serial position weighting (#82)

    python3 scripts/serial_position.py --db DB            # the measured use-rate by quartile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import serial_position  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        measured = serial_position.measure(store)
        print(f"{measured['sessions']} sessions, {measured['events']} access events")
        for bucket in measured["buckets"]:
            print(f"  quartile {bucket['bucket']}: selected {bucket['selected']:5d} "
                  f"used {bucket['used']:5d} use-rate {bucket['use_rate']:.3f}")
        described = serial_position.report(store)
        print(f"weight: late >= {described['late_turns']} turns, window "
              f"{described['late_window']}, boost {described['late_boost']}, "
              f"penalty {described['stale_penalty']}, enabled={described['enabled']}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
