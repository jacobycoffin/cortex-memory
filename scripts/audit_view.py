"""“Why did you say that?” — the audit view (#158)

    python3 scripts/audit_view.py --turn <task_id>
    python3 scripts/audit_view.py --last 3
    python3 scripts/audit_view.py --memory <memory_id>

Read-only: it prints what the store recorded and refuses to invent a reason where none was written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import _bootstrap  # noqa: E402,F401

from cortex import audit_view  # noqa: E402
from cortex.store import CortexStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / ".hermes" / "cortex" / "cortex.db"))
    parser.add_argument("--turn", default="")
    parser.add_argument("--last", type=int, default=1)
    parser.add_argument("--memory", default="")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    store = CortexStore(Path(args.db))
    try:
        if args.memory:
            found = audit_view.why_memory(store, args.memory)
            print(found["sentence"])
            for reason, count in sorted(found["rejected"].items(), key=lambda item: -item[1]):
                print(f"  ✗ {count} — {reason}")
            return 0
        turn_ids = [args.turn] if args.turn else [
            str(trace.get("task_id") or "")
            for trace in (store.memory_traces(limit=int(args.last)) or [])
        ]
        if not turn_ids:
            print("no recorded turns")
            return 1
        for task_id in turn_ids:
            found = audit_view.audit_turn(store, task_id)
            if args.json:
                print(json.dumps(found.as_dict(), indent=1)[:4000])
            else:
                print(audit_view.render_lines(found.render(limit=int(args.limit))))
                print()
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
