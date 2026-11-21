#!/usr/bin/env python3
"""What memory costs per turn.

Cortex decides how many tokens of recalled context go into every model call. That
number is invisible in normal use, and it is the one part of the bill an agent can
actually control. This reports it from `recall_runs`, prices it, and projects the
monthly cost.

    python3 scripts/cost_per_turn.py --db ~/.hermes/cortex/cortex.db
    python3 scripts/cost_per_turn.py --db ... --price-in 0.15 --price-out 0.60 --json

Prices are USD per million tokens for the model doing the chat completion. Defaults are
OpenCode Go's DeepSeek V4.1 Flash rates, which is what this profile uses; pass your own.
Only the INPUT side is charged here, because recalled memory and the injected context
are prompt tokens.

Exit codes: 0 fine, 1 over `--max-tokens-per-turn`, 2 unreadable database.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from pathlib import Path

DEFAULT_PRICE_IN = 0.15  # USD per 1M input tokens
DEFAULT_PRICE_OUT = 0.60  # USD per 1M output tokens


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(len(ordered) * pct)) - 1))
    return ordered[index]


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        print(f"cost_per_turn: no such database: {path}", file=sys.stderr)
        raise SystemExit(2)
    try:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        print(f"cost_per_turn: cannot open: {exc}", file=sys.stderr)
        raise SystemExit(2)


def collect(conn: sqlite3.Connection, days: int | None) -> dict:
    where, params = "", ()
    if days:
        where = "WHERE created_at >= datetime('now', ?)"
        params = (f"-{days} day",)

    rows = conn.execute(
        f"""SELECT session_id, estimated_tokens, rendered_tokens, abstained,
                   candidate_count, selected_count, created_at
            FROM recall_runs {where}""",
        params,
    ).fetchall()
    if not rows:
        return {"turns": 0, "warning": "no recall_runs in the window"}

    estimated = [float(r[1] or 0) for r in rows]
    rendered = [float(r[2] or 0) for r in rows]
    sessions = {r[0] for r in rows if r[0]}

    rendered_written = sum(1 for v in rendered if v > 0)
    warning = None
    if rendered_written == 0:
        warning = (
            "rendered_tokens is zero for every row: nothing writes it, so this "
            "reports estimated_tokens instead. See backlog item 215."
        )
        tokens = estimated
        token_source = "estimated_tokens"
    else:
        tokens = rendered
        token_source = "rendered_tokens"

    auxiliary = conn.execute(
        """SELECT COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0),
                  COUNT(*)
           FROM review_copilot_interpretations"""
    ).fetchone()

    return {
        "turns": len(rows),
        "sessions": len(sessions),
        "token_source": token_source,
        "tokens_total": int(sum(tokens)),
        "tokens_mean": round(statistics.fmean(tokens), 1),
        "tokens_p50": round(percentile(tokens, 0.50), 1),
        "tokens_p95": round(percentile(tokens, 0.95), 1),
        "tokens_max": int(max(tokens)),
        "abstained": sum(1 for r in rows if r[3]),
        "candidates_mean": round(statistics.fmean([float(r[4] or 0) for r in rows]), 1),
        "selected_mean": round(statistics.fmean([float(r[5] or 0) for r in rows]), 1),
        "first_at": min(r[6] for r in rows),
        "last_at": max(r[6] for r in rows),
        "aux_input_tokens": int(auxiliary[0]),
        "aux_output_tokens": int(auxiliary[1]),
        "aux_calls": int(auxiliary[2]),
        "warning": warning,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path.home() / ".hermes/cortex/cortex.db")
    parser.add_argument("--days", type=int, default=None, help="Window; default all time")
    parser.add_argument("--price-in", type=float, default=DEFAULT_PRICE_IN,
                        help="USD per 1M input tokens")
    parser.add_argument("--price-out", type=float, default=DEFAULT_PRICE_OUT,
                        help="USD per 1M output tokens")
    parser.add_argument("--turns-per-day", type=float, default=100.0,
                        help="Assumed turns/day for the monthly projection")
    parser.add_argument("--max-tokens-per-turn", type=float, default=None,
                        help="Exit 1 when the mean exceeds this")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    stats = collect(connect(args.db), args.days)
    if not stats.get("turns"):
        print(json.dumps(stats) if args.json else "cost_per_turn: no recalls to price")
        raise SystemExit(0)

    per_turn_cost = stats["tokens_mean"] / 1_000_000 * args.price_in
    aux_cost = (
        stats["aux_input_tokens"] / 1_000_000 * args.price_in
        + stats["aux_output_tokens"] / 1_000_000 * args.price_out
    )
    monthly = per_turn_cost * args.turns_per_day * 30

    payload = {
        **stats,
        "price_in_per_m": args.price_in,
        "price_out_per_m": args.price_out,
        "memory_usd_per_turn": round(per_turn_cost, 6),
        "auxiliary_usd_total": round(aux_cost, 4),
        "projected_usd_per_month": round(monthly, 2),
        "assumed_turns_per_day": args.turns_per_day,
    }

    if args.json:
        print(json.dumps(payload, indent=1))
    else:
        print(f"turns   {stats['turns']} across {stats['sessions']} sessions "
              f"({stats['first_at']} -> {stats['last_at']})")
        print(f"tokens  mean {stats['tokens_mean']:.0f}  p50 {stats['tokens_p50']:.0f}  "
              f"p95 {stats['tokens_p95']:.0f}  max {stats['tokens_max']}  "
              f"(source: {stats['token_source']})")
        print(f"shape   {stats['candidates_mean']:.1f} candidates -> "
              f"{stats['selected_mean']:.1f} selected, {stats['abstained']} abstentions")
        print(f"cost    ${per_turn_cost:.6f} of memory per turn at "
              f"${args.price_in}/M input")
        print(f"aux     {stats['aux_calls']} judge calls, "
              f"{stats['aux_input_tokens'] + stats['aux_output_tokens']} tokens, "
              f"${aux_cost:.4f} total")
        print(f"month   ${monthly:.2f} at {args.turns_per_day:g} turns/day")
        if stats.get("warning"):
            print(f"warning {stats['warning']}")

    if args.max_tokens_per_turn and stats["tokens_mean"] > args.max_tokens_per_turn:
        print(f"BREACH mean {stats['tokens_mean']:.0f} tokens/turn > "
              f"{args.max_tokens_per_turn:g}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
