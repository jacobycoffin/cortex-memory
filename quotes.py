"""Keep the sentence a fact came from (#42).

A memory is a compression, and compression is lossy in the one direction that matters for audit:
when a stored fact is wrong, the only way to find out *why* is to see what was actually said. Without
that, a bad memory is unauditable — the assistant can say "the store says port 8100" forever and
nobody can tell whether it was told, inferred, or mistranscribed.

So the verbatim source sentence is kept beside the memory:

    memory:  "The dashboard runs on port 8100."
    quote:   "yeah the dashboard is on 8100, nginx in front of it" (session 41, turn 7)

Three properties:

- **the quote is not the memory** — it is never recalled, never scored, never injected; it exists
  only for a human asking "where did this come from";
- **proposals keep their quote too** — a fact often spends time as a staged proposal, and the audit
  trail must not begin at promotion;
- **`verify()` compares the two** — a quote that shares no terms with the memory is evidence of drift,
  and that is worth being able to measure instead of hoping.

Quotes carry the same privacy weight as the memory they belong to: they are user text, they stay in
the same store, and they are deleted with it. Keeping a quotation is not permission to keep it
longer, or in more places.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

TABLE = "memory_quotes"
MAX_QUOTE_CHARS = 2000
MIN_TERM_LEN = 3
DRIFT_THRESHOLD = 0.20

# Numbers count. "port 8100" is the fact, and a drift check that drops every digit cannot tell
# 8100 from 8200 — which is exactly the correction it exists to catch.
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]+")
_STOP = frozenset(
    """about after also and are because been before being both but can could did does doing for
    from had has have having here how into its itself just more most not now only other our out over
    same should some such than that the their them then there these they this those through too
    under until very was were what when where which while with would you your""".split()
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    subject_id TEXT NOT NULL,
                    subject_kind TEXT NOT NULL,
                    quote TEXT NOT NULL,
                    source_ref TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (subject_id, quote)
                )"""
        )
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_quotes_kind ON {TABLE}(subject_kind)")


def terms(text: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in _WORD.finditer(str(text or ""))
        if match.group(0).lower() not in _STOP and len(match.group(0)) >= MIN_TERM_LEN
    }


def record(
    store: Any,
    subject_id: str,
    quote: str,
    *,
    kind: str = "memory",
    source_ref: str | None = None,
    at: datetime | None = None,
) -> bool:
    """Keep one verbatim sentence. Idempotent per subject and quote."""
    text = " ".join(str(quote or "").split())
    if not subject_id or not text:
        return False
    _ensure(store)
    with store.transaction() as conn:
        cursor = conn.execute(
            f"""INSERT OR IGNORE INTO {TABLE}
                (subject_id, subject_kind, quote, source_ref, created_at)
                VALUES (?, ?, ?, ?, ?)""",
            (str(subject_id), str(kind), text[:MAX_QUOTE_CHARS], source_ref,
             (at or _now()).isoformat()),
        )
        return bool(cursor.rowcount)


@dataclass
class Quote:
    subject_id: str
    kind: str
    quote: str
    source_ref: str
    created_at: str

    def describe(self) -> str:
        origin = f" [{self.source_ref}]" if self.source_ref else ""
        return f'"{self.quote}"{origin}'


def quotes_for(store: Any, subject_id: str) -> list[Quote]:
    _ensure(store)
    rows = store._conn.execute(
        f"""SELECT subject_id, subject_kind, quote, source_ref, created_at
            FROM {TABLE} WHERE subject_id = ? ORDER BY created_at ASC""",
        (str(subject_id),),
    ).fetchall()
    return [
        Quote(str(row["subject_id"]), str(row["subject_kind"]), str(row["quote"]),
              str(row["source_ref"] or ""), str(row["created_at"]))
        for row in rows
    ]


def adopt(store: Any, proposal_id: str, memory_id: str) -> int:
    """Follow a promotion: the quote recorded against a proposal now belongs to its memory."""
    _ensure(store)
    if not proposal_id or not memory_id:
        return 0
    with store.transaction() as conn:
        rows = conn.execute(
            f"SELECT quote, source_ref, created_at FROM {TABLE} WHERE subject_id = ?",
            (str(proposal_id),),
        ).fetchall()
        moved = 0
        for row in rows:
            cursor = conn.execute(
                f"""INSERT OR IGNORE INTO {TABLE}
                    (subject_id, subject_kind, quote, source_ref, created_at)
                    VALUES (?, 'memory', ?, ?, ?)""",
                (str(memory_id), str(row["quote"]), row["source_ref"], row["created_at"]),
            )
            moved += int(cursor.rowcount or 0)
        return moved


def find(store: Any, phrase: str, *, limit: int = 20) -> list[Quote]:
    """Audit direction: which stored facts came from a sentence like this one."""
    text = str(phrase or "").strip()
    if not text:
        return []
    _ensure(store)
    rows = store._conn.execute(
        f"""SELECT subject_id, subject_kind, quote, source_ref, created_at
            FROM {TABLE} WHERE quote LIKE ? ORDER BY created_at DESC LIMIT ?""",
        (f"%{text}%", int(limit)),
    ).fetchall()
    return [
        Quote(str(row["subject_id"]), str(row["subject_kind"]), str(row["quote"]),
              str(row["source_ref"] or ""), str(row["created_at"]))
        for row in rows
    ]


@dataclass
class Verification:
    memory_id: str
    overlap: float
    quoted: bool

    @property
    def drifted(self) -> bool:
        return self.quoted and self.overlap < DRIFT_THRESHOLD

    def describe(self) -> str:
        if not self.quoted:
            return "no quote recorded"
        verdict = "drifted from its quote" if self.drifted else "consistent with its quote"
        return f"{verdict} (overlap {self.overlap:.2f})"


def verify(store: Any, memory_id: str) -> Verification:
    """Does the stored memory still share language with what was actually said?"""
    rows = quotes_for(store, memory_id)
    if not rows:
        return Verification(str(memory_id), 0.0, quoted=False)
    row = store._conn.execute(
        "SELECT content FROM memories WHERE id = ?", (str(memory_id),)
    ).fetchone()
    content_terms = terms(str(row["content"]) if row else "")
    quote_terms: set[str] = set()
    for quote in rows:
        quote_terms |= terms(quote.quote)
    if not content_terms or not quote_terms:
        return Verification(str(memory_id), 0.0, quoted=True)
    overlap = len(content_terms & quote_terms) / len(content_terms | quote_terms)
    return Verification(str(memory_id), round(overlap, 3), quoted=True)


def drifted(store: Any, *, limit: int = 200) -> list[dict[str, Any]]:
    """Memory quotes whose stored fact no longer resembles them — the audit shortlist."""
    _ensure(store)
    rows = store._conn.execute(
        f"""SELECT DISTINCT g.subject_id FROM {TABLE} g
            JOIN memories m ON m.id = g.subject_id
            WHERE g.subject_kind = 'memory' LIMIT ?""",
        (int(limit),),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        result = verify(store, str(row["subject_id"]))
        if result.drifted:
            out.append({"memory_id": result.memory_id, "overlap": result.overlap})
    return out


def report(store: Any) -> dict[str, Any]:
    _ensure(store)
    rows = store._conn.execute(
        f"SELECT subject_kind, COUNT(*) AS n FROM {TABLE} GROUP BY subject_kind"
    ).fetchall()
    counts = {str(row["subject_kind"]): int(row["n"]) for row in rows}
    total_memories = store._conn.execute(
        "SELECT COUNT(*) FROM memories WHERE state NOT IN ('tombstoned','deleted')"
    ).fetchone()[0]
    quoted = store._conn.execute(
        f"SELECT COUNT(DISTINCT subject_id) FROM {TABLE} WHERE subject_kind = 'memory'"
    ).fetchone()[0]
    return {
        "quotes": sum(counts.values()),
        "by_kind": counts,
        "memories": int(total_memories or 0),
        "memories_with_quotes": int(quoted or 0),
        "coverage": round(int(quoted or 0) / total_memories, 3) if total_memories else 0.0,
    }
