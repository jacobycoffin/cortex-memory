"""How hard is a preference? (#59)

"I prefer short answers" and "never touch the production database" are both preferences in the store
and they are not the same instruction. Flattening them costs in both directions: treat a soft
preference as a rule and the assistant refuses reasonable requests; treat a rule as a preference and
it deviates from something the user said never to do.

So a preference carries a **strength**:

| strength | examples | what the agent may do |
|---|---|---|
| `hard` | "never", "always", "must", an allergy, a budget, "no exceptions" | do not deviate; if it cannot be met, say so and ask |
| `firm` | "I prefer", "please don't", "I'd rather" | deviate only with a stated reason |
| `soft` | "usually", "if convenient", "I tend to", "mildly" | deviate freely, mention it |

Two orderings matter and both are tested:

- a **hard marker beats a soft one** in the same sentence ("always, when convenient" is still
  always) — the safe reading wins, because the cost of over-obeying is a question, and the cost of
  under-obeying is an accident;
- a **negated soft marker is not soft** ("I don't usually care, but this one matters") — the
  contrast clause is where the real instruction lives.

Strength is not importance (#32). A hard rule and a trivial preference can both matter a lot; one
constrains behaviour and the other shapes style.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

HARD = "hard"
FIRM = "firm"
SOFT = "soft"
STRENGTHS = (HARD, FIRM, SOFT)

# Ordered: hard wins, so these are checked first and a match short-circuits.
_HARD = re.compile(
    r"\b(?:always|never|must|must not|cannot|can't|required|mandatory|"
    r"no exceptions?|never ever|allerg\w*|intoleran\w*|"
    r"budget(?:ed)? (?:is|of|at)|max(?:imum)? (?:of|spend)|under \$?\d+|"
    r"do not (?:ever|under any)|on no account|absolutely not)\b",
    re.IGNORECASE,
)
_FIRM = re.compile(
    r"\b(?:prefers?|preferred|rather|please (?:don't|do not|avoid)|avoid|"
    r"don't (?:want|like)|do not (?:want|like)|hates?|annoy\w*|"
    r"strongly|firmly|important to me|make sure|matters?)\b",
    re.IGNORECASE,
)
_SOFT = re.compile(
    r"\b(?:usually|generally|typically|normally|tends? to|often|"
    r"if (?:possible|convenient|you can)|when convenient|whenever possible|"
    r"mildly|slightly|a bit|sometimes|prefer but|nice to have)\b",
    re.IGNORECASE,
)
# "I don't usually care, but…" — the contrast is the instruction.
_CONTRAST = re.compile(r"\b(?:but|except|however|unless)\b", re.IGNORECASE)

TABLE = "preference_strength"

GUIDANCE = {
    HARD: "do not deviate; if it cannot be met, say so and ask",
    FIRM: "deviate only with a stated reason",
    SOFT: "deviate freely, mention that you did",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Strength:
    strength: str
    signals: tuple[str, ...] = ()

    @property
    def may_deviate(self) -> bool:
        return self.strength != HARD

    def describe(self) -> str:
        detail = f" ({', '.join(self.signals[:3])})" if self.signals else ""
        return f"{self.strength}{detail}: {GUIDANCE[self.strength]}"


def classify(text: str) -> Strength:
    """Classify one preference-shaped statement. Deterministic and order-sensitive."""
    value = str(text or "")
    hard = [match.group(0) for match in _HARD.finditer(value)]
    if hard:
        return Strength(HARD, tuple(hard))

    # A soft marker after a contrast clause ("..., but this one matters") is a real instruction.
    contrast = _CONTRAST.search(value)
    tail = value[contrast.end():] if contrast else ""
    if tail.strip() and _FIRM.search(tail):
        return Strength(FIRM, tuple(match.group(0) for match in _FIRM.finditer(tail)))

    firm_matches = list(_FIRM.finditer(value))
    soft_matches = list(_SOFT.finditer(value))
    # A hedge that comes *before* the verb modifies it: "mildly prefers" is soft, while
    # "prefers X, usually" leaves the preference itself firm.
    if soft_matches and firm_matches and soft_matches[0].start() < firm_matches[0].start():
        return Strength(SOFT, tuple(match.group(0) for match in soft_matches))
    if firm_matches:
        return Strength(FIRM, tuple(match.group(0) for match in firm_matches))
    if soft_matches:
        return Strength(SOFT, tuple(match.group(0) for match in soft_matches))
    # An unmarked preference is firm: the user took the trouble to say it, and "prefers" is not
    # "doesn't care". Silence about intensity is not evidence of weakness.
    return Strength(FIRM, ())


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    memory_id TEXT PRIMARY KEY,
                    strength TEXT NOT NULL,
                    signals TEXT,
                    created_at TEXT NOT NULL
                )"""
        )


def record(
    store: Any, memory_id: str, content: str, *, at: datetime | None = None
) -> Strength | None:
    if not memory_id:
        return None
    result = classify(content)
    _ensure(store)
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT INTO {TABLE} (memory_id, strength, signals, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    strength = excluded.strength,
                    signals = excluded.signals""",
            (str(memory_id), result.strength, ", ".join(result.signals[:5]),
             (at or _now()).isoformat()),
        )
    return result


def of(store: Any, memory_id: str) -> str | None:
    try:
        row = store._conn.execute(
            f"SELECT strength FROM {TABLE} WHERE memory_id = ?", (str(memory_id),)
        ).fetchone()
    except Exception:  # noqa: BLE001 - a missing table means "not recorded yet"
        return None
    return str(row["strength"]) if row else None


def hard_rules(store: Any, *, limit: int = 50) -> list[dict[str, Any]]:
    """Every memory the agent may not deviate from — the list worth reading before a plan."""
    _ensure(store)
    rows = store._conn.execute(
        f"""SELECT p.memory_id, p.signals, m.content, m.kind, m.state
            FROM {TABLE} p JOIN memories m ON m.id = p.memory_id
            WHERE p.strength = 'hard' AND m.state IN ('active','cold')
            ORDER BY m.importance DESC LIMIT ?""",
        (int(limit),),
    ).fetchall()
    return [
        {
            "memory_id": str(row["memory_id"]),
            "content": str(row["content"]),
            "kind": str(row["kind"]),
            "signals": str(row["signals"] or ""),
        }
        for row in rows
    ]


def report(store: Any) -> dict[str, Any]:
    _ensure(store)
    rows = store._conn.execute(
        f"SELECT strength, COUNT(*) AS n FROM {TABLE} GROUP BY strength"
    ).fetchall()
    counts = {str(row["strength"]): int(row["n"]) for row in rows}
    return {
        "recorded": sum(counts.values()),
        **{name: counts.get(name, 0) for name in STRENGTHS},
    }
