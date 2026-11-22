"""A source's authority is a ceiling on the confidence it can carry (#41).

Somebody telling the store a fact is not the same as the store inferring it, and neither is the same
as a web page claiming it. Retrieval already weights sources when *ranking* (`_SOURCE_RELIABILITY`);
this makes the same ordering bind where it matters more — the confidence **stored** with the memory,
which later readers and reviewers take as the store's own opinion.

The rule is a **ceiling, not a multiplier**:

    effective = min(stated_confidence, authority_of_source)

A multiplier would let a very emphatic web claim reach 0.8 by being emphatic twice; a ceiling says
what is true — no amount of certainty in the phrasing makes a source more reliable than it is. And a
ceiling never *raises* confidence, so a diffident user statement stays diffident.

The ladder, highest first: a user correcting the store outranks a user stating a fact, which
outranks a user's implication, which outranks an agent's inference, which outranks an unvetted
agent proposal, which outranks an imported document, which outranks a web claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Categories, highest authority first. Values are ceilings on stored confidence.
AUTHORITY: dict[str, float] = {
    "USER_CORRECTION": 0.97,
    "TOOL_VERIFIED": 0.95,
    "USER_EXPLICIT": 0.92,
    "USER_STATED": 0.88,
    "USER_IMPLICIT": 0.74,
    "DOCUMENT_EXTRACTED": 0.80,
    "OPERATOR_APPROVED": 0.78,
    "AUTOMATIC_APPROVED": 0.70,
    "REFLECTION": 0.62,
    "AGENT_INFERENCE": 0.55,
    "AGENT_PROPOSED": 0.50,
    "DOCUMENT_IMPORT": 0.60,
    "WEB_CLAIM": 0.45,
    "UNKNOWN": 0.50,
}

# Source types that say more than a category does on their own.
SOURCE_TYPE_AUTHORITY: dict[str, float] = {
    "user_turn": 0.92,
    "agent_tool_proposal": 0.50,
    "harness_proposal": 0.50,
    "document_import": 0.60,
    "web_fetch": 0.45,
    "tool_output": 0.90,
}

USER_CATEGORIES = frozenset(
    {"USER_CORRECTION", "USER_EXPLICIT", "USER_STATED", "USER_IMPLICIT"}
)


def authority(source_category: str | None, source_type: str | None = None) -> float:
    """The ceiling for a source. Unknown sources get the middle, never the top."""
    category = str(source_category or "").strip().upper()
    if category in AUTHORITY:
        return AUTHORITY[category]
    kind = str(source_type or "").strip().lower()
    if kind in SOURCE_TYPE_AUTHORITY:
        return SOURCE_TYPE_AUTHORITY[kind]
    return AUTHORITY["UNKNOWN"]


def is_user(source_category: str | None) -> bool:
    return str(source_category or "").strip().upper() in USER_CATEGORIES


@dataclass
class Weighted:
    confidence: float
    stated: float
    ceiling: float
    source_category: str
    capped: bool

    def describe(self) -> str:
        if not self.capped:
            return f"{self.confidence:.3f} (below the {self.ceiling:.2f} ceiling for this source)"
        return (
            f"{self.confidence:.3f} (capped from {self.stated:.3f} by the "
            f"{self.ceiling:.2f} ceiling for {self.source_category or 'an unnamed source'})"
        )


def apply(
    confidence: float,
    source_category: str | None,
    *,
    source_type: str | None = None,
) -> Weighted:
    """Carry a source's authority into the stored confidence."""
    try:
        stated = float(confidence)
    except (TypeError, ValueError):
        stated = 0.0
    stated = min(1.0, max(0.0, stated))
    ceiling = authority(source_category, source_type)
    value = min(stated, ceiling)
    return Weighted(
        confidence=round(value, 4),
        stated=round(stated, 4),
        ceiling=ceiling,
        source_category=str(source_category or ""),
        capped=value < stated,
    )


def confidence(
    stated: float, source_category: str | None, *, source_type: str | None = None
) -> float:
    """The number to store."""
    return apply(stated, source_category, source_type=source_type).confidence


def ladder() -> list[tuple[str, float]]:
    """The ordering, highest first, for anyone who wants to read it."""
    return sorted(AUTHORITY.items(), key=lambda item: -item[1])


def report(store: Any) -> dict[str, Any]:
    """How many stored memories are above what their source can support."""
    # Every state except the gone ones: a quarantined agent proposal is exactly the memory whose
    # confidence most needs checking, so filtering to active/cold would hide the interesting rows.
    rows = store._conn.execute(
        """SELECT source_category, COUNT(*) AS n FROM memories
           WHERE state NOT IN ('tombstoned','deleted')
           GROUP BY source_category ORDER BY n DESC"""
    ).fetchall()
    counts = {str(row["source_category"] or "UNKNOWN"): int(row["n"]) for row in rows}
    total = sum(counts.values())
    return {
        "total": total,
        "by_source": counts,
        "ceilings": {name: authority(name) for name in counts},
    }
