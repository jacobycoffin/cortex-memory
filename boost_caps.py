"""Popularity may break a tie; it may not become truth (#203).

Retrieval history raises scores in several independent places — times used, entity anchoring, session
working set, recorded associations. Each has its own cap, which is not the same as having a cap: a
memory that is popular **and** entity-anchored **and** part of today's working set collects all four,
and the total is what gets compared against a memory that is merely relevant.

So the rule is a ceiling on the **sum**, not on the parts:

| family | ceiling |
|---|---|
| usage (`use_boost`) | 0.09 |
| entity anchoring | 0.15 |
| session position | 0.10 |
| recorded association | 0.10 |
| **total history boost** | **0.25** |

`enforce()` reads the boost magnitudes each stage recorded in `components`, sums them, and where the
sum exceeds the ceiling it **removes the excess from the score** and notes `boost_capped` on the
result. Nothing is hidden: the components still say what each stage contributed, so a ranking remains
explainable after the fact.

The cap is deliberately above any single family's ceiling and below their sum. That way it never
changes a normal turn — a relevant memory with history behind it keeps its boost — and it only binds
for the memory that is winning on history alone, which is exactly the case this exists to stop.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

CAPS: dict[str, float] = {
    "usage": 0.09,
    "entity": 0.15,
    "session": 0.10,
    "association": 0.10,
}
TOTAL_CAP = 0.25

# The component keys a stage uses to record how much it added.
COMPONENT_KEYS: dict[str, str] = {
    "usage": "use_boost",
    "entity": "entity_boost",
    "session": "session_boost",
    "association": "association_boost",
}


def clamp(family: str, value: float) -> float:
    ceiling = CAPS.get(str(family))
    if ceiling is None:
        return max(0.0, float(value))
    return min(ceiling, max(0.0, float(value)))


def total_of(components: dict[str, Any] | None, *, memory_id: str = "") -> float:
    """The history boost this result is carrying, as recorded by the stages that added it."""
    payload = dict(components or {})
    if not payload:
        return 0.0
    total = 0.0
    try:
        for family, key in COMPONENT_KEYS.items():
            raw = payload.get(key)
            if raw is None:
                continue
            try:
                total += clamp(family, float(raw))
            except (TypeError, ValueError):
                continue
    except Exception:  # noqa: BLE001 - explainability must never raise
        return 0.0
    return total


def enforce(results: Iterable[Any]) -> tuple[list[Any], int]:
    """Trim any result whose combined history boost exceeds the ceiling.

    Returns (results, capped_count). A result that was capped carries `boost_capped` in its
    components, so the decision is visible in the trace rather than inferred from a number that
    looks slightly smaller than expected.
    """
    out: list[Any] = []
    capped = 0
    for result in results:
        components = dict(getattr(result, "components", {}) or {})
        total = total_of(components)
        if total <= TOTAL_CAP:
            out.append(result)
            continue
        excess = total - TOTAL_CAP
        components["boost_capped"] = round(excess, 6)
        components["boost_total"] = round(total, 6)
        out.append(replace(result, score=float(result.score) - excess, components=components))
        capped += 1
    return out, capped


def report(results: Iterable[Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "caps": dict(CAPS),
        "total_cap": TOTAL_CAP,
        "max_single_family": max(CAPS.values()),
        "sum_of_families": round(sum(CAPS.values()), 4),
    }
    if results is not None:
        observed = [total_of(getattr(item, "components", {}) or {}) for item in results]
        payload["observed"] = len(observed)
        payload["over_cap"] = sum(1 for value in observed if value > TOTAL_CAP)
        payload["largest"] = round(max(observed), 4) if observed else 0.0
    return payload
