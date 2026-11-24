"""Which memories informed which tool calls (#70)

    python3 scripts/tool_links.py --db DB                  # report by tool
    python3 scripts/tool_links.py --db DB --memory <id>
    python3 scripts/tool_links.py --db DB --tool ssh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401  (loads the checkout as ``cortex``)

from cortex import tool_links  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--memory", default="")
    parser.add_argument("--tool", default="")
    parser.add_argument("--executions", action="store_true")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.executions:
            row = store._conn.execute("SELECT COUNT(*) FROM tool_executions").fetchone()
            print(f"{row[0]} tool execution(s) recorded; "
                  f"{tool_links.report(store)['links']} linked to a memory")
            return 0
        if args.memory:
            for use in tool_links.for_memory(store, args.memory):
                print(" ", use.describe())
            return 0
        if args.tool:
            for memory_id in tool_links.for_tool(store, args.tool):
                print(" ", memory_id)
            return 0
        described = tool_links.report(store)
        print(f"{described['links']} link(s) across {described['memories']} memories and "
              f"{described['tools']} tools")
        for tool, detail in sorted(described["by_tool"].items()):
            print(f"  {tool or '(unnamed)':<16} {detail['calls']:4d} call(s)  "
                  f"{detail['success_share'] * 100:5.1f}% succeeded")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
