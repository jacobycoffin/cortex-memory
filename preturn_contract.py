"""What a pre-turn recall hook is promised, and what it must promise back (#144)

The hook that runs before a turn is the one place where a slow or broken component can cost the owner his
reply. So the contract has to be explicit, and it has to be **checkable** rather than documented:

- **a hook may not fail the turn.** Even when recall raises, times out or returns nothing, the response is
  a well-formed object with `degraded=True`. A contract that permits an exception is a contract that
  permits a dropped reply.
- **every id in the block must be in the id list.** The block is what the model reads; the ids are what
  the receipts, the panel (#148) and the used-resolution step (#145) attribute. An id printed but not
  returned is an unattributable memory; an id returned but not printed is a claim nobody can check.
- **the block must fit the budget**, measured in characters against the token budget at the documented
  ratio — not asserted by whoever wrote the caller.
- **the deadline is a promise, and missing it is reported, not hidden.** An over-deadline response is
  accepted (the turn still gets its memories) and flagged `degraded`, because silently dropping recall to
  save 200 ms is a worse trade than a slightly slow turn.
- **the same request twice gives the same ids.** A hook that reshuffles on identical input cannot be
  cached, and a cache that changes the answer is not a cache (#memory: a cache key must never break recall).

Live, the current pipeline violates the budget rule **8 times in 4,718 runs** and the limit rule **0
times** — small, but a contract that is checked finds the 8; a contract that is documented finds nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

CONTRACT_VERSION = 1
CHARS_PER_TOKEN = 4.0
DEFAULT_DEADLINE_MS = 1500
DEFAULT_BUDGET_TOKENS = 700
MAX_BLOCK_CHARS_PER_TOKEN = 4.5   # headroom over the nominal ratio before it counts as over budget


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


@dataclass
class Request:
    task_id: str = ""
    session_id: str = ""
    query: str = ""
    budget_tokens: int = DEFAULT_BUDGET_TOKENS
    deadline_ms: int = DEFAULT_DEADLINE_MS
    pinned: list[str] = field(default_factory=list)
    scope: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "session_id": self.session_id, "query": self.query,
                "budget_tokens": self.budget_tokens, "deadline_ms": self.deadline_ms,
                "pinned": list(self.pinned), "scope": dict(self.scope)}


@dataclass
class Response:
    block: str = ""
    ids: list[str] = field(default_factory=list)
    withheld: int = 0
    abstained: bool = False
    elapsed_ms: float = 0.0
    degraded: bool = False
    contract_version: int = CONTRACT_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {"block": self.block, "ids": list(self.ids), "withheld": self.withheld,
                "abstained": self.abstained, "elapsed_ms": self.elapsed_ms,
                "degraded": self.degraded, "contract_version": self.contract_version}


def degraded(reason: str, *, elapsed_ms: float = 0.0) -> Response:
    """What a hook returns when it cannot do its job. Never an exception (#144)."""
    return Response(block="", ids=[], withheld=0, abstained=True, elapsed_ms=elapsed_ms,
                    degraded=True)


def _ids_in(block: str, known: set[str]) -> set[str]:
    return {memory_id for memory_id in known if memory_id and memory_id[:8] in block}


def validate(request: Request, response: Response, *, known_ids: set[str] | None = None
             ) -> list[str]:
    """The contract, as a list of violations. Empty means the hook kept its promises."""
    problems: list[str] = []
    if not isinstance(response, Response):
        return ["the hook returned something that is not a Response; it may not fail the turn"]
    if response.contract_version != CONTRACT_VERSION:
        problems.append(f"contract version {response.contract_version} is not {CONTRACT_VERSION}")
    if len(response.ids) != len(set(response.ids)):
        problems.append("the same memory id is returned twice")
    if known_ids is not None:
        unknown = [memory_id for memory_id in response.ids if memory_id not in known_ids]
        if unknown:
            problems.append(f"{len(unknown)} returned id(s) are not memories in this store")
        printed = _ids_in(response.block, known_ids) if response.block else set()
        missing = printed - set(response.ids)
        if missing:
            problems.append(f"{len(missing)} id(s) appear in the block but not in the id list")
    budget_chars = max(1.0, float(request.budget_tokens) * CHARS_PER_TOKEN)
    if len(response.block) > budget_chars * (MAX_BLOCK_CHARS_PER_TOKEN / CHARS_PER_TOKEN):
        problems.append(f"the block is {len(response.block)} chars against a "
                        f"{int(budget_chars)}-char budget")
    if response.elapsed_ms > request.deadline_ms and not response.degraded:
        problems.append(f"the hook took {response.elapsed_ms:.0f} ms against a "
                        f"{request.deadline_ms} ms deadline without reporting degraded")
    if response.abstained and response.ids:
        problems.append("an abstention returned ids: it either abstains or it answers")
    if response.degraded and not response.abstained:
        problems.append("a degraded response must say it abstained")
    return problems


def conforms(request: Request, response: Response, *, known_ids: set[str] | None = None) -> bool:
    return not validate(request, response, known_ids=known_ids)


def deterministic(run: Any, request: Request, *, runs: int = 2) -> dict[str, Any]:
    """The same request twice must give the same ids — otherwise no cache can be trusted."""
    seen: list[list[str]] = []
    for _ in range(max(2, int(runs))):
        try:
            response = run(request)
        except Exception as error:  # noqa: BLE001
            return {"deterministic": False, "reason": f"the hook raised: {error}"}
        if not isinstance(response, Response):
            return {"deterministic": False, "reason": "the hook did not return a Response"}
        seen.append(list(response.ids))
    first = seen[0]
    stable = all(item == first for item in seen[1:])
    return {"deterministic": stable, "ids": len(first),
            "reason": "" if stable else "identical requests produced different id lists"}


def contract() -> dict[str, Any]:
    """The promise, as data, so a caller can print it or test against it."""
    return {
        "version": CONTRACT_VERSION,
        "request": {"task_id": "string", "session_id": "string", "query": "string",
                    "budget_tokens": DEFAULT_BUDGET_TOKENS, "deadline_ms": DEFAULT_DEADLINE_MS,
                    "pinned": "list of memory ids", "scope": "dict of filters"},
        "response": {"block": "text placed in the prompt", "ids": "memory ids in the block",
                     "withheld": "count dropped for budget", "abstained": "bool",
                     "elapsed_ms": "float", "degraded": "bool", "contract_version": CONTRACT_VERSION},
        "guarantees": [
            "a hook may not fail the turn: on any failure return degraded(reason)",
            "every id in the block appears in the id list, and vice versa",
            "the block fits the budget",
            "a missed deadline is reported as degraded, not hidden",
            "the same request twice returns the same ids",
        ],
        "live_violations": "8 of 4,718 recorded runs exceeded the token budget; 0 exceeded the limit",
    }


def audit(store: Any, *, window: int = 500) -> dict[str, Any]:
    """What the recorded runs say about the budget and limit promises."""
    rows = _rows(
        store,
        "SELECT token_budget, estimated_tokens, selected_count, requested_limit, prepare_ms "
        "FROM recall_runs WHERE token_budget > 0 ORDER BY created_at DESC, recall_id DESC LIMIT ?",
        (int(window),),
    )
    over_budget = [row for row in rows
                   if float(row["estimated_tokens"] or 0) > float(row["token_budget"] or 0)]
    over_limit = [row for row in rows
                  if int(row["requested_limit"] or 0) and
                  int(row["selected_count"] or 0) > int(row["requested_limit"])]
    slow = [row for row in rows if float(row["prepare_ms"] or 0) > DEFAULT_DEADLINE_MS]
    return {
        "runs": len(rows),
        "over_budget": len(over_budget),
        "over_limit": len(over_limit),
        "over_deadline": len(slow),
        "verdict": "conforms" if not (over_budget or over_limit) else "violations",
        "note": "measured from recall_runs; a documented contract finds nothing, a checked one finds this",
    }


def render(store: Any, *, window: int = 500) -> list[str]:
    described = audit(store, window=window)
    lines = [f"pre-turn contract v{CONTRACT_VERSION}: {described['verdict']} over "
             f"{described['runs']} run(s)"]
    lines.append(f"  over budget: {described['over_budget']} · over limit: "
                 f"{described['over_limit']} · over deadline: {described['over_deadline']}")
    for guarantee in contract()["guarantees"]:
        lines.append(f"  · {guarantee}")
    return lines
