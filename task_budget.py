"""Allocate context by task type instead of one number for everything (#13).

Measured on the live store: **every** task type was given 612–680 tokens and the same
six-item ceiling — `general/lean` 618, `shell/focused` 612, `filesystem/deep` 641. The
budget is therefore not a decision, it is a constant, and it is not even the binding
constraint: the average turn selected **3.2** memories, so the *limit* stops selection long
before the token budget does.

What a task needs differs by task. A shell or filesystem turn wants several concrete facts
— file paths, service names, prior commands. A calendar or email turn wants one or two, and
paying for six is noise. This table makes that explicit, and it moves the limit as well as
the budget, because the measurement says the limit is what binds.
"""

from __future__ import annotations

from typing import Any

# (limit, token_budget) per task type. Ordered from most to least context-hungry.
TASK_BUDGETS: dict[str, tuple[int, int]] = {
    "shell": (8, 900),
    "filesystem": (8, 900),
    "github": (8, 900),
    "data_analysis": (7, 800),
    "web_research": (6, 700),
    "memory": (5, 600),
    "calendar": (4, 420),
    "email": (4, 420),
    "general": (6, 620),
}

DEFAULT = TASK_BUDGETS["general"]


def allocation(task_type: str) -> tuple[int, int]:
    """(limit, token_budget) for a task type; unknown types get the general allocation."""
    return TASK_BUDGETS.get(str(task_type or "").lower(), DEFAULT)


def plan_for(
    task_type: str,
    *,
    max_limit: int = 20,
    max_token_budget: int = 4000,
    floor_tokens: int = 120,
) -> dict[str, Any]:
    """Clamp the task-type allocation into the deployment's own ceilings."""
    limit, budget = allocation(task_type)
    limit = max(1, min(int(max_limit), int(limit)))
    budget = max(int(floor_tokens), min(int(max_token_budget), int(budget)))
    return {
        "task_type": str(task_type or "general"),
        "limit": limit,
        "token_budget": budget,
        "reason": f"{task_type or 'general'} allocation: {limit} memories / {budget} tokens",
    }


def explain(task_type: str) -> str:
    limit, budget = allocation(task_type)
    return f"{task_type or 'general'}: {limit} memories, {budget} tokens"
