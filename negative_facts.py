"""Store "X is not true" as a fact, not as an absence (#58).

A store that only records positives cannot tell three states apart:

| state | what the store knows |
|---|---|
| affirmative | *"the box runs Debian"* |
| **negative** | *"the box does not run Debian"* |
| unknown | nothing |

Without the middle row, "no" and "nobody ever said" look identical, and the assistant answers "I
don't have anything about that" when it has, in fact, been told the opposite. That is the failure
this fixes: a negative is a **recorded opinion with a date and a source**, not a hole.

Two properties keep it honest:

- **a negative is not a deletion.** The positive it contradicts stays in the store; the pair becomes
  a dispute (#48) if both are current, because the store should hedge rather than silently flip.
- **polarity is derived once, at write**, and stored, so a later reader does not have to re-parse
  English to decide whether a row said "not".

Absence still means lack of knowledge. Nothing here infers a negative from a missing positive —
that is the one inference a memory store must never make.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

POSITIVE = "affirmative"
NEGATIVE = "negative"
# A long document chunk is not a statement with a truth value: it *contains* the word "not"
# somewhere. Classifying those as negative facts inflated the live count to 53.6% — nearly all of
# it imported vault notes — so only statement-shaped text gets a polarity at all.
UNCLASSIFIED = "unclassified"
TABLE = "memory_polarity"

MAX_STATEMENT_CHARS = 400
MAX_SENTENCES = 3
_DOCUMENT_PREFIXES = ("vault note:", "section:", "path:")

# Ordered longest-first so "no longer" wins over "no".
_NEGATORS = (
    "no longer", "not anymore", "does not", "doesn't", "do not", "don't", "did not", "didn't",
    "is not", "isn't", "are not", "aren't", "was not", "wasn't", "were not", "weren't",
    "will not", "won't", "cannot", "can't", "never", "stopped", "disabled", "removed",
    "not",
)
_NEGATOR = re.compile(r"\b(" + "|".join(re.escape(word) for word in _NEGATORS) + r")\b", re.I)
# A question asking for confirmation is not itself a negative claim.
_QUESTION = re.compile(r"\?\s*$")

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")
_STOP = frozenset(
    """a an and are as at be been but by for from had has have i if in into is it its of on or the
    their them then there these they this to was we were what when where which who why will with
    you your""".split()
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def is_negative(text: str) -> bool:
    value = str(text or "")
    if not value.strip() or _QUESTION.search(value):
        return False
    return bool(_NEGATOR.search(value))


def polarity(text: str) -> str:
    return NEGATIVE if is_negative(text) else POSITIVE



def is_statement(text: str) -> bool:
    """Is this something with a truth value, or a passage that merely mentions "not"?"""
    value = str(text or "").strip()
    if not value or len(value) > MAX_STATEMENT_CHARS:
        return False
    lowered = value.lower()
    if lowered.startswith(_DOCUMENT_PREFIXES):
        return False
    if value.count(". ") + value.count("! ") + value.count("? ") >= MAX_SENTENCES:
        return False
    return True


def classify(text: str) -> str:
    """`affirmative`, `negative`, or `unclassified` for text that is not a statement."""
    if not is_statement(text):
        return UNCLASSIFIED
    return polarity(text)

def _stem(word: str) -> str:
    """A crude stem, because "runs" and "run" are the same claim and the overlap must see that."""
    value = str(word).lower().strip(".,;:!?\"'()[]")
    for suffix in ("ing", "ed", "es", "s"):
        if len(value) > len(suffix) + 2 and value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _terms(text: str) -> set[str]:
    return {
        _stem(match.group(0))
        for match in _WORD.finditer(str(text or ""))
        if match.group(0).lower() not in _STOP and len(match.group(0)) > 2
    }


def negated_terms(text: str) -> set[str]:
    """Distinctive terms of a negative statement, with the negator itself removed."""
    cleaned = _NEGATOR.sub(" ", str(text or ""))
    return _terms(cleaned)


def topic_overlap(left: str, right: str) -> float:
    """How much two statements are about the same thing, ignoring polarity."""
    left_terms = negated_terms(left) if polarity(left) == NEGATIVE else _terms(left)
    right_terms = negated_terms(right) if polarity(right) == NEGATIVE else _terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def same_claim(left: str, right: str) -> bool:
    """Are these two statements the same claim with opposite polarity?"""
    if polarity(left) == polarity(right):
        return False
    return topic_overlap(left, right) >= 0.6


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    memory_id TEXT PRIMARY KEY,
                    polarity TEXT NOT NULL,
                    negator TEXT,
                    created_at TEXT NOT NULL
                )"""
        )


@dataclass
class Recorded:
    memory_id: str
    polarity: str
    negator: str

    def describe(self) -> str:
        if self.polarity == POSITIVE:
            return "affirmative"
        if self.polarity == UNCLASSIFIED:
            return "unclassified (not a statement)"
        return f"negative (via '{self.negator}')" if self.negator else "negative"


def record(
    store: Any, memory_id: str, content: str, *, at: datetime | None = None
) -> Recorded | None:
    """Record the polarity of one memory. Idempotent per memory."""
    if not memory_id:
        return None
    value = str(content or "")
    verdict = classify(value)
    match = _NEGATOR.search(value)
    negator = match.group(0).lower() if (verdict == NEGATIVE and match) else ""
    _ensure(store)
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT INTO {TABLE} (memory_id, polarity, negator, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    polarity = excluded.polarity,
                    negator = excluded.negator""",
            (str(memory_id), verdict, negator, (at or _now()).isoformat()),
        )
    return Recorded(str(memory_id), verdict, negator)


def of(store: Any, memory_id: str) -> str | None:
    try:
        row = store._conn.execute(
            f"SELECT polarity FROM {TABLE} WHERE memory_id = ?", (str(memory_id),)
        ).fetchone()
    except Exception:  # noqa: BLE001 - a missing table means "not recorded yet"
        return None
    return str(row["polarity"]) if row else None


def negatives(store: Any, *, limit: int = 50) -> list[dict[str, Any]]:
    _ensure(store)
    rows = store._conn.execute(
        f"""SELECT p.memory_id, p.negator, m.content, m.kind, m.state
            FROM {TABLE} p JOIN memories m ON m.id = p.memory_id
            WHERE p.polarity = 'negative'
              AND m.state NOT IN ('tombstoned','deleted','suppressed')
            ORDER BY p.created_at DESC LIMIT ?""",
        (int(limit),),
    ).fetchall()
    return [
        {
            "memory_id": str(row["memory_id"]),
            "content": str(row["content"]),
            "kind": str(row["kind"]),
            "state": str(row["state"]),
            "negator": str(row["negator"] or ""),
        }
        for row in rows
    ]


def answer(store: Any, statement: str, *, limit: int = 50) -> dict[str, Any]:
    """Is this statement believed, denied, contradicted, or simply unknown?

    `both` is a first-class answer: if the store holds the affirmative and the negative it must
    say so rather than pick one (#48).
    """
    _ensure(store)
    rows = store._conn.execute(
        f"""SELECT p.memory_id, p.polarity, m.content, m.state
            FROM {TABLE} p JOIN memories m ON m.id = p.memory_id
            WHERE m.state NOT IN ('tombstoned','deleted','suppressed') LIMIT ?""",
        (int(limit),),
    ).fetchall()
    support: list[str] = []
    deny: list[str] = []
    for row in rows:
        content = str(row["content"])
        if str(row["polarity"]) == UNCLASSIFIED:
            # A passage with no truth value is not evidence either way.
            continue
        if topic_overlap(statement, content) < 0.6:
            continue
        # Same polarity is agreement; opposite polarity is denial. Both are evidence.
        if str(row["polarity"]) == polarity(statement):
            support.append(str(row["memory_id"]))
        else:
            deny.append(str(row["memory_id"]))
    if support and deny:
        verdict = "both"
    elif deny:
        verdict = "negative"
    elif support:
        verdict = "affirmative"
    else:
        verdict = "unknown"
    return {"statement": statement, "verdict": verdict, "support": support, "deny": deny}


def report(store: Any) -> dict[str, Any]:
    _ensure(store)
    rows = store._conn.execute(
        f"SELECT polarity, COUNT(*) AS n FROM {TABLE} GROUP BY polarity"
    ).fetchall()
    counts = {str(row["polarity"]): int(row["n"]) for row in rows}
    return {
        "recorded": sum(counts.values()),
        "affirmative": counts.get(POSITIVE, 0),
        "negative": counts.get(NEGATIVE, 0),
    }
