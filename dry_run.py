"""Answer "what would you remember from this?" without writing anything (#46).

The write path is a chain: split → anchor → extract → refuse/redact → dedupe → score → throttle →
judge. When the store learns something wrong, or fails to learn something obvious, the only way to
find out where the chain dropped it is to run the chain and look — and until now that meant writing
real memories and then cleaning up.

So `project()` runs the same steps with **no writes at all** and reports, per candidate, what each
stage decided and what would happen next:

| projected | meaning |
|---|---|
| `remember` | would be stored (subject to review, depending on source) |
| `duplicate` | restates a memory the store already has |
| `refuse` | contains a credential — nothing would be stored |
| `redacted` | stored, with the secret replaced |
| `rejected_by_rule` | a learned rejection (#35) already covers it |
| `throttled` | the per-turn/hour/session cap would defer it |
| `rejected` | no durable signal, or rejected by the extraction filters |

Dry runs are deliberately **read-only in the strongest sense**: they open no transaction that
mutates, they do not bump counters, and they do not touch the throttle buckets. A dry run that has
side effects is a write with extra steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import (
    anchoring,
    importance_rubric,
    reject_learn,
    secret_refusal,
    source_weight,
    splitting,
    volatility,
    write_throttle,
)

MAX_PREVIEW = 120


@dataclass
class Projection:
    content: str
    projected: str
    kind: str = "unknown"
    importance: float = 0.0
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        detail = "; ".join(self.notes)
        return f"{self.projected:<16} {self.content[:70]}" + (f" [{detail}]" if detail else "")


def _similar(store: Any, content: str) -> str | None:
    """Is there already a memory saying this? Read-only, and cheap on failure."""
    try:
        from . import confirmation

        found = confirmation.find_confirmation(store, content)
        if found:
            return str(getattr(found, "memory_id", "") or "")
    except Exception:  # noqa: BLE001 - a dry run must never fail on an optional check
        return None
    return None


def project(
    store: Any,
    text: str,
    *,
    role: str = "user",
    source_category: str = "USER_EXPLICIT",
    session_id: str | None = None,
    check_exists: bool = True,
) -> dict[str, Any]:
    """Run the write pipeline in shadow mode. Returns what would happen and why."""
    from .extraction import extract_candidates

    raw = str(text or "")
    facts = splitting.split_facts(raw)
    anchored_text = anchoring.anchored(raw)
    candidates = extract_candidates(raw, role=role)
    projections: list[Projection] = []

    for candidate in candidates:
        verdict = secret_refusal.assess(candidate.content)
        scored = importance_rubric.score(
            candidate.content,
            candidate.kind,
            source_category=source_category,
        )
        ceiling = source_weight.authority(source_category)
        projection = Projection(
            content=candidate.content,
            projected="remember",
            kind=candidate.kind,
            importance=scored.importance,
            confidence=min(candidate.confidence, ceiling),
        )
        if verdict.refused:
            projection.projected = "refuse"
            projection.notes.append(verdict.describe())
            projections.append(projection)
            continue
        if verdict.action == secret_refusal.REDACT:
            projection.projected = "redacted"
            projection.notes.append(verdict.describe())
        rule = reject_learn.check(store, candidate.content, source_category=source_category)
        if rule is not None:
            projection.projected = "rejected_by_rule"
            projection.notes.append(rule.describe())
            projections.append(projection)
            continue
        duplicate = _similar(store, candidate.content) if check_exists else None
        if duplicate:
            projection.projected = "duplicate"
            projection.notes.append(f"restates memory {str(duplicate)[:8]}")
            projections.append(projection)
            continue
        if session_id and write_throttle.is_automatic(source_category):
            decision = write_throttle.check(store, session_id)
            if not decision.allowed:
                projection.projected = "throttled"
                projection.notes.append(decision.describe())
        projections.append(projection)

    # Candidates the filters dropped entirely are the interesting silences.
    kept = {projection.content for projection in projections}
    dropped = [
        fact for fact in facts
        if not any(fact.strip() and fact.strip() in content for content in kept)
    ]
    counts: dict[str, int] = {}
    for projection in projections:
        counts[projection.projected] = counts.get(projection.projected, 0) + 1

    return {
        "facts": len(facts),
        "candidates": len(candidates),
        "dropped": len(dropped),
        "counts": counts,
        "items": projections,
        "anchored": anchored_text != raw,
        "skipped": dropped[:10],
        "source_category": source_category,
    }


def summarise(result: dict[str, Any]) -> str:
    """One line for chat: what would be remembered, and what would not."""
    stored = sum(
        count for name, count in result["counts"].items() if name in {"remember", "redacted"}
    )
    parts = [f"would remember {stored} of {result['facts']} fact(s)"]
    for name in ("duplicate", "refuse", "rejected_by_rule", "throttled", "redacted"):
        if result["counts"].get(name):
            parts.append(f"{result['counts'][name]} {name.replace('_', ' ')}")
    if result["dropped"]:
        parts.append(f"{result['dropped']} below the extraction bar")
    return "; ".join(parts)


def render(result: dict[str, Any], *, limit: int = 12) -> str:
    lines = [summarise(result)]
    for projection in result["items"][:limit]:
        lines.append("  " + projection.describe())
    for skipped in result["skipped"][:5]:
        lines.append(f"  {'skipped':<16} {skipped[:70]}")
    return "\n".join(lines)
