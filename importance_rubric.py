"""Importance at write: a rubric, not a flat 0.5 (#32).

Importance is load-bearing — the core block ranks by it, the quota sweep spares it, triage skips
high-importance material — and until now it came from one of two places: a bare `0.5` default, or
whatever number the calling model happened to pass (`0.78` in this repo's own capture path). Both are
arbitrary, and neither varies with what the fact actually is.

So: a small rubric, anchored on the kind, adjusted by a short list of modifiers that are cheap to
compute and easy to argue about.

| kind | base | why |
|---|---|---|
| identity | 0.95 | who someone is outranks everything |
| commitment, decision | 0.88 | a promise has consequences if forgotten |
| correction | 0.86 | the store must not fall back to the wrong version |
| preference | 0.82 | shapes every future answer |
| prospective | 0.80 | will be needed at a specific time |
| procedure | 0.75 | needed again, but recoverable |
| semantic | 0.66 | ordinary fact |
| tool | 0.60 | machine detail |
| episode | 0.40 | working notes |

Modifiers move at most ±0.16 together: an explicit user statement and a credential raise it; a
hedged question and a throwaway fragment lower it. The band is deliberate — a rubric that can swing
importance from 0.1 to 0.9 stops being a rubric and becomes a second, worse classifier.

Importance is *not* volatility (#31): "my port is 8100" is both durable and modestly important,
while "I'm furious about this" is ephemeral and momentarily important. They are computed separately
and for different consumers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

BASE_BY_KIND: dict[str, float] = {
    "identity": 0.95,
    "correction": 0.86,
    "commitment": 0.88,
    "decision": 0.88,
    "preference": 0.82,
    "prospective": 0.80,
    "procedure": 0.75,
    "procedural": 0.75,
    "semantic": 0.66,
    "tool": 0.60,
    "episode": 0.40,
    "episodic": 0.40,
}
DEFAULT_BASE = 0.50

MIN_IMPORTANCE = 0.05
MAX_IMPORTANCE = 1.00
MODIFIER_BAND = 0.16

_EXPLICIT_SOURCES = frozenset({"USER_EXPLICIT", "USER_STATED", "USER_CORRECTION"})
_HEDGED = re.compile(r"\b(maybe|possibly|i think|not sure|probably|might be|i guess)\b", re.I)
_DURABLE_DETAIL = re.compile(
    # "port 8100" and "port is 8100" are the same fact; a regex that only matches the first
    # misses how people write it.
    r"\b(port\s*(?:is\s*)?\d+|password|api key|credential|address|path|~/|/root/|https?://)\b",
    re.I,
)
_EPHEMERAL = re.compile(
    r"\b(right now|today|tonight|this week|temporarily|for now|currently|at the moment)\b", re.I
)


def base(kind: str | None) -> float:
    return BASE_BY_KIND.get(str(kind or "").strip().lower(), DEFAULT_BASE)


@dataclass
class Scored:
    importance: float
    base: float
    kind: str
    modifiers: dict[str, float] = field(default_factory=dict)

    def describe(self) -> str:
        parts = ", ".join(f"{name} {value:+.2f}" for name, value in self.modifiers.items())
        return f"{self.importance:.3f} ({self.kind} base {self.base:.2f}{'; ' + parts if parts else ''})"


def score(
    content: str,
    kind: str | None = None,
    *,
    source_category: str | None = None,
    corroboration: int = 0,
) -> Scored:
    """Score one memory. Deterministic: same inputs, same number, always."""
    text = str(content or "")
    kind_name = str(kind or "").strip().lower() or "unknown"
    start = base(kind_name)
    modifiers: dict[str, float] = {}

    if str(source_category or "").upper() in _EXPLICIT_SOURCES:
        modifiers["user_stated"] = 0.05
    if corroboration >= 1:
        modifiers["corroborated"] = min(0.04, 0.02 * corroboration)
    if _DURABLE_DETAIL.search(text):
        modifiers["durable_detail"] = 0.06
    if _EPHEMERAL.search(text):
        modifiers["ephemeral"] = -0.05
    if _HEDGED.search(text):
        modifiers["hedged"] = -0.08
    if len(text.strip()) < 20:
        modifiers["fragment"] = -0.05

    total = sum(modifiers.values())
    if total > MODIFIER_BAND:
        scale = MODIFIER_BAND / total
        modifiers = {name: round(value * scale, 4) for name, value in modifiers.items()}
        total = MODIFIER_BAND
    elif total < -MODIFIER_BAND:
        scale = MODIFIER_BAND / abs(total)
        modifiers = {name: round(value * scale, 4) for name, value in modifiers.items()}
        total = -MODIFIER_BAND

    value = min(MAX_IMPORTANCE, max(MIN_IMPORTANCE, start + total))
    return Scored(round(value, 3), start, kind_name, modifiers)


def is_flat(value: Any) -> bool:
    """True for the placeholder importance a caller never actually chose."""
    try:
        return abs(float(value) - DEFAULT_BASE) < 1e-9
    except (TypeError, ValueError):
        return True


def fill(store: Any, *, limit: int = 0, at: Any = None) -> dict[str, int]:
    """Re-score memories still sitting at the flat default. Returns how many moved."""
    query = "SELECT id, content, kind, source_category FROM memories WHERE importance = ?"
    params: list[Any] = [DEFAULT_BASE]
    if limit:
        query += " LIMIT ?"
        params.append(int(limit))
    rows = store._conn.execute(query, tuple(params)).fetchall()
    moved = 0
    with store.transaction() as conn:
        for row in rows:
            scored = score(
                str(row["content"]), str(row["kind"]),
                source_category=row["source_category"],
            )
            if abs(scored.importance - DEFAULT_BASE) < 1e-9:
                continue
            conn.execute(
                "UPDATE memories SET importance = ? WHERE id = ?",
                (scored.importance, str(row["id"])),
            )
            moved += 1
    return {"candidates": len(rows), "updated": moved}


def distribution(store: Any) -> dict[str, Any]:
    rows = store._conn.execute(
        "SELECT kind, importance FROM memories WHERE state = 'active'"
    ).fetchall()
    if not rows:
        return {"active": 0, "mean": 0.0, "flat": 0, "by_kind": {}}
    values = [float(row["importance"]) for row in rows]
    flat = sum(1 for value in values if is_flat(value))
    by_kind: dict[str, list[float]] = {}
    for row in rows:
        by_kind.setdefault(str(row["kind"] or "unknown"), []).append(float(row["importance"]))
    return {
        "active": len(values),
        "mean": round(sum(values) / len(values), 3),
        "flat": flat,
        "by_kind": {name: round(sum(group) / len(group), 3)
                    for name, group in sorted(by_kind.items())},
    }
