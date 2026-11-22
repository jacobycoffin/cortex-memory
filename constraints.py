"""Constraints are respected regardless of similarity (#60).

Most memory is fetched because it is relevant. A constraint is not: *"no shellfish"*, *"$40 a month
on hosting"*, *"never touch the production database"* have to be in view whether or not the current
question resembles them, because their whole job is to stop something that would otherwise look
reasonable. Retrieval by similarity is exactly the mechanism that would miss them — a shellfish
allergy scores zero against "recommend a restaurant".

So constraints are a **separate, always-on channel**:

- they are detected once (a hard preference from #59, an allergy, a budget, an explicit "never",
  a hard no) and stored;
- they are rendered into every turn's context as a short block, with a hard cap, because a
  constraint list nobody reads is the same as no list;
- they are never scored, never ranked, never crowded out by ordinary recall.

Two asymmetries are deliberate:

- **detection leans inclusive.** A false positive costs one wasted line in the context; a false
  negative costs the accident the constraint existed to prevent;
- **the block is capped and ordered by importance**, so a store with sixty "never" rules still
  shows the ones that matter instead of truncating the top of an alphabetical list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

TABLE = "constraints"
MAX_BLOCK_CHARS = 320
MAX_ITEMS = 6

ALLERGY = "allergy"
BUDGET = "budget"
HARD_NO = "hard_no"
LIMIT = "limit"

_KINDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\ballerg\w*\b|\bintoleran\w*\b|\bcan'?t eat\b|\bcannot eat\b", re.I), ALLERGY),
    (re.compile(r"\bbudget\b|\bmax(?:imum)? (?:of|spend)\b|\bspend at most\b|"
                r"\bunder \$?\d+|\bno more than \$?\d+|\$\d+ ?(?:a|per) (?:month|year|week)\b",
                re.I), BUDGET),
    (re.compile(r"\bnever\b|\bon no account\b|\bdo not (?:ever|under any)\b|"
                r"\bmust not\b|\bno exceptions?\b|\babsolutely not\b", re.I), HARD_NO),
    (re.compile(r"\bonly\b|\bat most\b|\bno (?:more|fewer) than\b|\bcap(?:ped)? at\b", re.I), LIMIT),
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    memory_id TEXT PRIMARY KEY,
                    constraint_kind TEXT NOT NULL,
                    signal TEXT,
                    created_at TEXT NOT NULL
                )"""
        )


@dataclass
class Constraint:
    memory_id: str
    kind: str
    content: str
    signal: str = ""

    def line(self) -> str:
        # Short lines mean more constraints fit in the block: six 80-character rules carry more
        # than two 110-character ones, and a truncated rule is still the useful part.
        return f"- {self.content.strip()[:80]}"


def detect(text: str) -> str | None:
    """Which kind of constraint this is, or None. Inclusive on purpose."""
    value = str(text or "")
    for pattern, kind in _KINDS:
        match = pattern.search(value)
        if match:
            return kind
    return None


def promote(
    store: Any,
    memory_id: str,
    content: str,
    *,
    kind: str | None = None,
    at: datetime | None = None,
) -> str | None:
    """Record a memory as a constraint if it is one. Returns the kind, or None."""
    kind = kind or detect(content)
    if kind is None:
        return None
    _ensure(store)
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT INTO {TABLE} (memory_id, constraint_kind, signal, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET constraint_kind = excluded.constraint_kind""",
            (str(memory_id), kind, kind, (at or _now()).isoformat()),
        )
    return kind


def promote_from_strength(store: Any, memory_id: str, content: str) -> str | None:
    """A hard preference (#59) is a constraint even when no keyword fires."""
    kind = detect(content)
    if kind is not None:
        return promote(store, memory_id, content)
    try:
        from . import preference_strength

        if preference_strength.of(store, memory_id) == preference_strength.HARD:
            # The keyword list missed it, but the strength classifier did not: a hard rule is a
            # constraint whatever words it used.
            return promote(store, memory_id, content, kind=HARD_NO)
    except Exception:  # noqa: BLE001 - the strength table is optional here
        return None
    return None


def all(store: Any, *, limit: int = MAX_ITEMS * 4) -> list[Constraint]:
    """Every active constraint, most important first."""
    _ensure(store)
    rows = store._conn.execute(
        f"""SELECT c.memory_id, c.constraint_kind, c.signal, m.content
            FROM {TABLE} c JOIN memories m ON m.id = c.memory_id
            WHERE m.state IN ('active','cold')
            ORDER BY m.importance DESC, m.created_at DESC LIMIT ?""",
        (int(limit),),
    ).fetchall()
    return [
        Constraint(str(row["memory_id"]), str(row["constraint_kind"]), str(row["content"]),
                   str(row["signal"] or ""))
        for row in rows
    ]


def context_block(store: Any, *, max_items: int = MAX_ITEMS,
                  max_chars: int = MAX_BLOCK_CHARS) -> str:
    """The always-on block. Empty when there are no constraints — no header, no noise."""
    items = all(store, limit=max_items)
    if not items:
        return ""
    lines: list[str] = []
    used = 0
    for item in items:
        line = item.line()
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    if not lines:
        return ""
    return "Constraints (always apply, regardless of the question):\n" + "\n".join(lines)


def relevant(store: Any, text: str) -> list[Constraint]:
    """Constraints whose wording overlaps a proposed action — a soft warning, never a gate."""
    from .negative_facts import _terms  # shared, deliberately crude

    wanted = _terms(text)
    if not wanted:
        return []
    matches: list[Constraint] = []
    for item in all(store, limit=200):
        terms = _terms(item.content)
        # Jaccard is the wrong measure for a short constraint against a long action, so any
        # distinctive shared term is enough: "shellfish" appearing at all is the warning.
        if terms & wanted:
            matches.append(item)
    return matches


def remove(store: Any, memory_id: str) -> bool:
    _ensure(store)
    with store.transaction() as conn:
        cursor = conn.execute(f"DELETE FROM {TABLE} WHERE memory_id = ?", (str(memory_id),))
        return bool(cursor.rowcount)


def report(store: Any) -> dict[str, Any]:
    _ensure(store)
    rows = store._conn.execute(
        f"SELECT constraint_kind, COUNT(*) AS n FROM {TABLE} GROUP BY constraint_kind"
    ).fetchall()
    counts = {str(row["constraint_kind"]): int(row["n"]) for row in rows}
    return {
        "constraints": sum(counts.values()),
        "by_kind": counts,
        "block_chars": len(context_block(store)),
    }
