"""Spend the context budget where the evidence is densest (backlog item #12).

The preload budget is 4,000 tokens with a 20-item ceiling, and selection fills it by score
order until the count runs out. Those are two different stopping rules, and the count wins:
five long vault notes at a mediocre score can use the whole block while a dozen short,
sharp memories get nothing.

Density — score per token — is the honest exchange rate. This fills the budget greedily by
density instead of by rank, so the same 4,000 tokens carry more, better evidence.

Two guards keep it honest:

- a **score floor** relative to the best candidate, because a 12-token memory with almost
  no relevance is cheap and worthless;
- **deterministic ordering** on ties (score, then id), so the same pool always produces the
  same block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

DEFAULT_MIN_RATIO = 0.35
DEFAULT_SCORE_RATIO = 0.85


@dataclass(frozen=True)
class Choice:
    result: Any
    tokens: int
    density: float


def tokens_of(result: Any) -> int:
    """Estimated tokens for one result, never zero."""
    value = getattr(result, "estimated_tokens", 0) or 0
    try:
        tokens = int(value)
    except (TypeError, ValueError):
        tokens = 0
    if tokens <= 0:
        content = str((getattr(result, "memory", {}) or {}).get("content") or "")
        tokens = max(1, (len(content) + 3) // 4)
    return max(1, tokens)


def density(result: Any) -> float:
    return float(getattr(result, "score", 0.0) or 0.0) / tokens_of(result)


def fill_budget(
    results: Iterable[Any],
    token_budget: int,
    *,
    max_items: int | None = None,
    min_ratio: float = DEFAULT_MIN_RATIO,
) -> list[Choice]:
    """Choose the highest-density results that fit the budget.

    `min_ratio` is a floor against the best candidate's density, so cheap noise is skipped
    even when there is room for it.
    """
    pool = list(results)
    if not pool or token_budget <= 0:
        return []

    scored = [
        Choice(result=result, tokens=tokens_of(result), density=density(result))
        for result in pool
    ]
    best = max(choice.density for choice in scored)
    floor = best * max(0.0, float(min_ratio))

    ordered = sorted(
        (choice for choice in scored if choice.density >= floor and choice.density > 0),
        key=lambda choice: (-choice.density, -float(getattr(choice.result, "score", 0.0) or 0.0),
                            str((getattr(choice.result, "memory", {}) or {}).get("id") or "")),
    )

    chosen: list[Choice] = []
    spent = 0
    limit = max_items if max_items is not None else len(ordered)
    for choice in ordered:
        if len(chosen) >= limit:
            break
        if spent + choice.tokens > token_budget:
            continue  # a smaller item further down may still fit
        chosen.append(choice)
        spent += choice.tokens
    return chosen


def apply(
    results: Iterable[Any], token_budget: int, *, max_items: int | None = None,
    min_ratio: float = DEFAULT_MIN_RATIO,
) -> list[Any]:
    """The chosen results, in density order."""
    return [choice.result for choice in fill_budget(results, token_budget, max_items=max_items,
                                                    min_ratio=min_ratio)]


def by_confidence(
    results: Iterable[Any],
    token_budget: int,
    *,
    max_items: int | None = None,
    score_ratio: float = DEFAULT_SCORE_RATIO,
) -> list[Any]:
    """Keep the confident head, in score order, and stop when the budget is spent.

    Measured against density ordering: density fills the same 4,000 tokens with short,
    low-scoring memories (mean score 0.531 -> 0.423 on six probe queries) because a token
    of a long note is worth more than a token of a terse one here. Confidence gating keeps
    the quality and merely refuses to spend on the weak tail, which is what the budget was
    for in the first place.
    """
    pool = list(results)
    if not pool or token_budget <= 0:
        return []
    best = max(float(getattr(r, "score", 0.0) or 0.0) for r in pool)
    floor = best * max(0.0, float(score_ratio))
    head = [r for r in pool if float(getattr(r, "score", 0.0) or 0.0) >= floor and floor > 0]
    head.sort(key=lambda r: (-float(getattr(r, "score", 0.0) or 0.0),
                             str((getattr(r, "memory", {}) or {}).get("id") or "")))
    chosen: list[Any] = []
    spent = 0
    limit = max_items if max_items is not None else len(head)
    for result in head:
        if len(chosen) >= limit:
            break
        cost = tokens_of(result)
        if spent + cost > token_budget:
            continue
        chosen.append(result)
        spent += cost
    return chosen
