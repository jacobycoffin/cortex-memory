"""Is the running plugin the code in the repository? (#221)

    python3 scripts/plugin_drift.py
    python3 scripts/plugin_drift.py --deployed "$HOME/.hermes/plugins/cortex" --limit 200

Read-only. It never copies, deploys or restarts anything: a redeploy changes what the assistant does
and belongs to whoever owns the machine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401

from cortex import plugin_drift  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(ROOT))
    parser.add_argument("--deployed", default=str(Path.home() / ".hermes" / "plugins" / "cortex"))
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    found = plugin_drift.compare(args.repo, args.deployed, limit=args.limit)
    print(found.describe())
    heads = plugin_drift.samples(found, count=args.samples)
    for bucket, names in heads.items():
        if names:
            print(f"  {bucket}: {', '.join(names)}")
    print("read-only: nothing was copied, deployed or restarted")
    return 0 if found.verdict in {plugin_drift.CURRENT, plugin_drift.AHEAD} else 1


if __name__ == "__main__":
    raise SystemExit(main())
