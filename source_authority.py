"""Source authority: a document cannot quietly outrank what the user told you.

Cortex ingests from many places — vault documents, agent inferences, tool results and
the user's own words. Those are not equal evidence, and the ranking did not know that:
a vault page and a direct user statement competed purely on textual similarity.

This module caps the score of a lower-authority memory when a higher-authority memory
in the SAME fact slot (subject + predicate) is also retrieved. It never raises anything,
it never reorders across different facts, and it is applied before the final sort, so a
capped memory can still be selected when nothing better exists.

The cap is a ceiling, not a swap: a document that really is the only evidence still
wins, it just cannot claim to outrank a statement about the same thing.
"""

from __future__ import annotations

from typing import Any, Iterable

# Higher means more trustworthy as evidence about the world.
AUTHORITY: dict[str, float] = {
    "USER_EXPLICIT": 1.00,       # the user said it in so many words
    "OPERATOR_APPROVED": 0.95,   # a human reviewed and admitted it
    "TOOL_VERIFIED": 0.90,       # a tool actually observed it
    "AUTOMATIC_APPROVED": 0.70,  # judge admitted it without a human
    "REFLECTION": 0.60,          # the agent summarising itself
    "AGENT_INFERENCE": 0.55,     # the agent's own guess
    "DOCUMENT_EXTRACTED": 0.50,  # pages, vault notes, anything scraped
}
DEFAULT_AUTHORITY = 0.50

# A lower-authority memory may not exceed this fraction of the best higher-authority
# memory in the same fact slot.
CAP_RATIO = 0.90


def authority_for(memory: dict[str, Any]) -> float:
    category = str(memory.get("source_category") or "").upper()
    if category in AUTHORITY:
        return AUTHORITY[category]
    # Fall back to the record type when the category is absent or unrecognised.
    source_type = str(memory.get("source_type") or "").lower()
    if source_type in ("user_turn", "builtin_memory"):
        return AUTHORITY["USER_EXPLICIT"]
    if source_type in ("tool_execution", "explicit_tool"):
        return AUTHORITY["TOOL_VERIFIED"]
    if source_type in ("vault_markdown",):
        return AUTHORITY["DOCUMENT_EXTRACTED"]
    if source_type in ("assistant_turn", "agent_tool_proposal"):
        return AUTHORITY["AGENT_INFERENCE"]
    return DEFAULT_AUTHORITY


def _slot(memory: dict[str, Any]) -> tuple[str, str] | None:
    subject = str(memory.get("subject") or "").strip().lower()
    predicate = str(memory.get("predicate") or "").strip().lower()
    if not subject or not predicate:
        return None
    return subject, predicate


def cap_conflicts(results: Iterable[Any]) -> tuple[list[Any], int]:
    """Cap every lower-authority result that shares a fact slot with a stronger one.

    Returns (results, capped_count). Accepts anything with `.memory` and `.score`
    attributes.
    """
    results = list(results)
    slots: dict[tuple[str, str], list[Any]] = {}
    for result in results:
        slot = _slot(getattr(result, "memory", {}) or {})
        if slot:
            slots.setdefault(slot, []).append(result)

    capped = 0
    for group in slots.values():
        if len(group) < 2:
            continue
        top_authority = max(authority_for(r.memory) for r in group)
        top_score = max(
            (r.score for r in group if authority_for(r.memory) == top_authority), default=0.0
        )
        ceiling = top_score * CAP_RATIO
        for result in group:
            if authority_for(result.memory) >= top_authority:
                continue
            if result.score <= ceiling:
                continue
            components = dict(getattr(result, "components", {}) or {})
            components["authority_cap"] = round(ceiling, 6)
            components["authority_score_before_cap"] = round(float(result.score), 6)
            result.score = ceiling
            result.components = components
            capped += 1
    return results, capped
