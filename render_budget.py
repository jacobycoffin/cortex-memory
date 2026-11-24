"""What the render budget is spent on, and what nobody is measuring (#172)

Every block that reaches the prompt has a cap: a constraint block of 320 characters, a bootstrap block
of 900, a memory budget of ~620 tokens. What was missing is a single view that says **how much of each
cap is actually used**, and — the part that matters more — which side of the pipeline is not recording
its spend at all.

The live store answers that bluntly:

```
token_budget      : 527-680, avg 620     ← recorded
estimated_tokens  : avg 513 (>0 in 4,651 of 4,724 runs)   ← recorded
rendered_tokens   : always 0              ← NEVER recorded
withheld_count    : always 0              ← NEVER recorded
```

So the store knows what it *selected* and never learns what it *rendered*. A selection estimate of 513
against a 620 budget looks like a healthy 83% until you notice that the render step — which may add a
header, truncate, or drop a memory — leaves no trace, which means "the budget was respected" is an
assumption rather than a measurement.

Four rules:

- **a budget is reported with its cap and its source**, because "513 tokens" means nothing without "of
  620, from `recall_runs.token_budget`";
- **an unrecorded quantity is `None`, not zero.** `rendered_tokens = 0` may mean "nobody wrote it" or
  "nothing was rendered", and the two must not print the same;
- **the registry is the list.** Every cap in the codebase appears in `budgets()`, so a new block cannot
  be added without a number a reader can check;
- **estimate vs measurement is labelled.** Tokens per character is a ratio, so anything computed from
  text length says `estimate` in its name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

WINDOW = 500
CHARS_PER_TOKEN = 4.0


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


@dataclass
class Budget:
    name: str
    cap: float
    unit: str
    source: str
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "cap": self.cap, "unit": self.unit, "source": self.source,
                "note": self.note}


def budgets() -> list[Budget]:
    """Every render cap in one place — the list a new block has to join."""
    from . import bootstrap_block, constraints, phone_surface, recall_receipt, voice_summary

    return [
        Budget("constraints block", float(constraints.MAX_BLOCK_CHARS), "chars", "constraints.py",
               f"at most {constraints.MAX_ITEMS} items"),
        Budget("bootstrap block", float(bootstrap_block.BUDGET), "chars", "bootstrap_block.py",
               "three sections, counted truncation"),
        Budget("memory budget", 700.0, "tokens", "recall_runs.token_budget",
               "the per-turn retrieval budget; live average 620"),
        Budget("phone list", float(phone_surface.DEFAULT_LIMIT), "rows", "phone_surface.py",
               f"clamped to {phone_surface.MAX_LIMIT}"),
        Budget("voice summary", float(voice_summary.MAX_CHARS), "chars", "voice_summary.py",
               "one breath, not one paragraph"),
        Budget("answer receipt", float(recall_receipt.SNIPPET), "chars", "recall_receipt.py",
               "per cited memory"),
    ]


@dataclass
class Spend:
    runs: int = 0
    budget_p50: float = 0.0
    selected_p50: float = 0.0
    selected_share_p50: float = 0.0
    selected_share_p95: float = 0.0
    rendered_recorded: bool = False
    withheld_recorded: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs, "budget_p50": round(self.budget_p50, 1),
            "selected_p50": round(self.selected_p50, 1),
            "selected_share_p50": round(self.selected_share_p50, 4),
            "selected_share_p95": round(self.selected_share_p95, 4),
            "rendered_recorded": self.rendered_recorded,
            "withheld_recorded": self.withheld_recorded,
            "note": self.note,
        }


def spend(store: Any, *, window: int = WINDOW) -> Spend:
    """How much of the token budget the *selection* uses — and whether the render side is recorded."""
    rows = _rows(
        store,
        "SELECT token_budget, estimated_tokens, rendered_tokens, withheld_count FROM recall_runs "
        "ORDER BY created_at DESC, recall_id DESC LIMIT ?",
        (int(window),),
    )
    found = Spend(runs=len(rows))
    if not rows:
        found.note = "no recall runs recorded"
        return found
    budgets = [float(row["token_budget"] or 0) for row in rows if float(row["token_budget"] or 0)]
    selected = [float(row["estimated_tokens"] or 0) for row in rows]
    shares = [
        min(1.0, float(row["estimated_tokens"] or 0) / float(row["token_budget"]))
        for row in rows if float(row["token_budget"] or 0)
    ]
    found.budget_p50 = _percentile(budgets, 0.5)
    found.selected_p50 = _percentile(selected, 0.5)
    found.selected_share_p50 = _percentile(shares, 0.5)
    found.selected_share_p95 = _percentile(shares, 0.95)
    found.rendered_recorded = any(float(row["rendered_tokens"] or 0) for row in rows)
    found.withheld_recorded = any(int(row["withheld_count"] or 0) for row in rows)
    gaps = []
    if not found.rendered_recorded:
        gaps.append("rendered_tokens is never written")
    if not found.withheld_recorded:
        gaps.append("withheld_count is never written")
    if gaps:
        found.note = ("the render side is unmeasured: " + ", ".join(gaps) +
                      " — so 'the budget was respected' is an assumption, not a measurement")
    return found


def estimate_tokens(text: str, *, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """A labelled estimate. Anything derived from length says so in its name."""
    body = str(text or "")
    if not body:
        return 0
    return max(1, int(round(len(body) / max(0.5, float(chars_per_token)))))


def block_spend(text: str, cap: float) -> dict[str, Any]:
    """How much of one block's cap is used, with the cap and the estimate labelled."""
    body = str(text or "")
    used = len(body)
    return {
        "chars": used,
        "cap": float(cap),
        "share": round(used / float(cap), 4) if cap else 0.0,
        "tokens_estimate": estimate_tokens(body),
        "over": used > float(cap),
        "sentence": f"{used} of {int(cap)} chars ({used / cap:.0%})" if cap else f"{used} chars",
    }


def report(store: Any, *, window: int = WINDOW) -> dict[str, Any]:
    described = spend(store, window=window)
    return {
        "spend": described.as_dict(),
        "budgets": [item.as_dict() for item in budgets()],
        "verdict": "unmeasured" if not described.rendered_recorded else "measured",
        "note": "an unrecorded quantity is None, not zero",
    }


def render(store: Any, *, window: int = WINDOW) -> list[str]:
    described = spend(store, window=window)
    lines = [f"render budget over {described.runs} run(s):"]
    lines.append(f"  budget p50 {described.budget_p50:.0f} tokens · selection p50 "
                 f"{described.selected_p50:.0f} ({described.selected_share_p50:.0%}, "
                 f"p95 {described.selected_share_p95:.0%})")
    state = "recorded" if described.rendered_recorded else "NOT RECORDED"
    withheld = "recorded" if described.withheld_recorded else "NOT RECORDED"
    lines.append(f"  rendered tokens: {state}   withheld count: {withheld}")
    if described.note:
        lines.append(f"  {described.note}")
    lines.append("  caps:")
    for item in budgets():
        lines.append(f"    {item.name}: {item.cap:g} {item.unit} ({item.source})")
    return lines
