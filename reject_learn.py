"""Reject-then-learn: a rejection teaches the next similar write (#35).

The auto-judge rejects a proposal, the reasoning is thrown into a report nobody reads twice, and the
same kind of proposal is staged again next week. So a rejection is recorded as a **rule**: what was
rejected, why, and a signature of its distinctive terms. A later proposal that matches a rule is
rejected without spending a model call — and without a model having to notice it twice.

Two deliberate limits:

- **Only rejections are learned.** A rule that could auto-*accept* would be the same failure as
  letting a cheap model write (see #26); rejecting is always safe, so only that direction is cached.
- **A rule never touches user-stated material.** Rules are consulted by the auto-judge, which only
  ever sees agent proposals; `check()` also refuses to match a `USER_*` source category so a rule
  cannot quietly bury something the user said.

Matching is a keyword signature with a Jaccard threshold, not a model: it has to be cheap enough to
run on every proposal, and it has to be explainable when someone asks why their memory did not stick.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

TABLE = "rejection_rules"
MATCH_THRESHOLD = 0.60
MIN_SIGNATURE_TERMS = 3
MAX_SIGNATURE_TERMS = 12

REASONS = {
    "duplicate": "restates a memory the store already has",
    "ephemeral": "true only for a moment; not worth a record",
    "not_factual": "opinion, guess or speculation rather than a fact",
    "assistant_status": "the assistant reporting its own progress",
    "trivial": "too small to be worth recalling later",
    "low_confidence": "the judge could not tell what was being claimed",
    "unsafe": "contains a secret or something that must not be stored",
    "out_of_scope": "about another project, person or system entirely",
}

_STOPWORDS = frozenset(
    """a an and are as at be been but by for from has have i if in into is it its of on or
    our that the their them then there these they this to was we were what when where which who
    why will with you your""".split()
)
_USER_SOURCES = frozenset(
    {"USER_EXPLICIT", "USER_STATED", "USER_CORRECTION", "USER_IMPLICIT"}
)
_WORD = re.compile(r"[a-z0-9][a-z0-9._/-]*")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def signature(text: str) -> tuple[str, ...]:
    """Distinctive terms, order-free, deduplicated, capped."""
    terms = [
        match.group(0)
        for match in _WORD.finditer(str(text or "").lower())
        if match.group(0) not in _STOPWORDS and len(match.group(0)) > 1
    ]
    seen: list[str] = []
    for term in terms:
        if term not in seen:
            seen.append(term)
    return tuple(sorted(seen)[:MAX_SIGNATURE_TERMS])


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    rule_id TEXT PRIMARY KEY,
                    reason_code TEXT NOT NULL,
                    reason_text TEXT,
                    signature TEXT NOT NULL,
                    sample TEXT,
                    source TEXT,
                    hits INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )"""
        )


def _rule_id(code: str, terms: tuple[str, ...]) -> str:
    payload = code + "|" + " ".join(terms)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def record(
    store: Any,
    content: str,
    *,
    reason_code: str,
    reason_text: str = "",
    source: str = "autojudge",
    at: datetime | None = None,
) -> str | None:
    """Learn from one rejection. Returns the rule id, or None when the content is too thin."""
    terms = signature(content)
    if len(terms) < MIN_SIGNATURE_TERMS:
        return None
    code = str(reason_code or "").strip().lower() or "low_confidence"
    _ensure(store)
    rule_id = _rule_id(code, terms)
    stamp = (at or _now()).isoformat()
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT INTO {TABLE}
                (rule_id, reason_code, reason_text, signature, sample, source, hits, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                ON CONFLICT(rule_id) DO UPDATE SET
                    reason_text = excluded.reason_text,
                    sample = excluded.sample""",
            (rule_id, code, str(reason_text)[:400], " ".join(terms),
             str(content)[:400], source, stamp),
        )
    return rule_id


@dataclass
class Match:
    rule_id: str
    reason_code: str
    reason: str
    similarity: float

    def describe(self) -> str:
        return f"{self.reason_code} ({self.similarity:.2f}): {self.reason}"


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def check(
    store: Any,
    content: str,
    *,
    source_category: str | None = None,
    threshold: float = MATCH_THRESHOLD,
) -> Match | None:
    """Is this content covered by a learned rejection? Never for user-stated material."""
    if str(source_category or "").upper() in _USER_SOURCES:
        return None
    terms = set(signature(content))
    if len(terms) < MIN_SIGNATURE_TERMS:
        return None
    try:
        rows = store._conn.execute(
            f"SELECT rule_id, reason_code, reason_text, signature FROM {TABLE}"
        ).fetchall()
    except Exception:  # noqa: BLE001 - a missing table must not break the judge
        return None
    best: Match | None = None
    for row in rows:
        similarity = _jaccard(terms, set(str(row["signature"]).split()))
        if similarity < threshold:
            continue
        if best is None or similarity > best.similarity:
            best = Match(
                rule_id=str(row["rule_id"]),
                reason_code=str(row["reason_code"]),
                reason=str(row["reason_text"] or REASONS.get(str(row["reason_code"]), "")),
                similarity=round(similarity, 3),
            )
    return best


def note_hit(store: Any, rule_id: str) -> None:
    _ensure(store)
    with store.transaction() as conn:
        conn.execute(f"UPDATE {TABLE} SET hits = hits + 1 WHERE rule_id = ?", (str(rule_id),))


def report(store: Any, *, limit: int = 20) -> dict[str, Any]:
    _ensure(store)
    rows = store._conn.execute(
        f"""SELECT reason_code, COUNT(*) AS rules, SUM(hits) AS hits
            FROM {TABLE} GROUP BY reason_code ORDER BY hits DESC, rules DESC"""
    ).fetchall()
    by_reason = {
        str(row["reason_code"]): {"rules": int(row["rules"]), "hits": int(row["hits"] or 0)}
        for row in rows
    }
    top = store._conn.execute(
        f"SELECT sample, reason_code, hits FROM {TABLE} ORDER BY hits DESC LIMIT ?", (limit,)
    ).fetchall()
    return {
        "rules": sum(item["rules"] for item in by_reason.values()),
        "hits": sum(item["hits"] for item in by_reason.values()),
        "by_reason": by_reason,
        "top": [str(row["sample"] or "")[:80] for row in top],
    }


def forget_rule(store: Any, rule_id: str) -> bool:
    """A rule that misfires has to be removable, or nobody will trust the mechanism."""
    _ensure(store)
    with store.transaction() as conn:
        cursor = conn.execute(f"DELETE FROM {TABLE} WHERE rule_id = ?", (str(rule_id),))
        return bool(cursor.rowcount)
