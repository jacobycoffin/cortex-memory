"""When the user says two things are related, the edge learns (#79).

The graph is built from co-occurrence: two memories appeared in the same session often enough to
deserve a link. That signal is cheap and wrong in both directions. A person can look at it once and
settle the question — *"those two aren't related"* — and that judgement is worth more than a
thousand future co-occurrences, because it is the only signal in the system that is actually about
meaning rather than statistics.

So feedback goes two ways:

- **"related"** creates the edge if it is missing and reinforces it if it is not, capped at `1.0` and
  recorded as `user_feedback` evidence. A human link needs no co-occurrence threshold.
- **"not related"** is a **veto, not a weight of zero.** The edge is lowered to the floor, and the
  pair is recorded so `co_retrieval` never re-proposes it. A veto that only nudged a number would be
  undone by the next sweep, which is worse than not asking.

Neither direction deletes anything: the evidence trail stays, because *"I already told you these are
unrelated"* is exactly the history worth keeping when the link reappears.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

TABLE = "edge_feedback"
REINFORCE_TO = 0.80
REINFORCE_STEP = 0.10
VETO_WEIGHT = 0.02
MAX_WEIGHT = 1.0

_NEGATIVE = re.compile(
    r"\b(?:not\s+related|unrelated|no\s+relation|aren'?t\s+related|isn'?t\s+related|"
    r"don'?t\s+relate|disconnected|separate\s+things)\b",
    re.IGNORECASE,
)
_POSITIVE = re.compile(
    r"\b(?:related|connected|the\s+same\s+thing|same\s+topic|belongs?\s+with)\b", re.IGNORECASE
)
_SPLIT = re.compile(r"\s+(?:and|with|to|,)\s+", re.IGNORECASE)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    left_id TEXT NOT NULL,
                    right_id TEXT NOT NULL,
                    polarity TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (left_id, right_id)
                )"""
        )


def polarity(text: str) -> str | None:
    """`related`, `unrelated`, or None when the sentence does not say."""
    body = str(text or "")
    if _NEGATIVE.search(body):
        return "unrelated"
    if _POSITIVE.search(body):
        return "related"
    return None


def parse(text: str) -> tuple[str, str, str] | None:
    """Split "X and Y are not related" into (X, Y, polarity)."""
    body = str(text or "").strip()
    sense = polarity(body)
    if not body or sense is None:
        return None
    cleaned = _NEGATIVE.sub(" ", body)
    cleaned = _POSITIVE.sub(" ", cleaned)
    cleaned = re.sub(r"\b(?:are|is|were|was|these|those|they|the)\b", " ", cleaned,
                     flags=re.IGNORECASE)
    parts = [part.strip(" .;:") for part in _SPLIT.split(cleaned) if part.strip(" .;:")]
    if len(parts) < 2:
        return None
    return parts[0], parts[1], sense


@dataclass
class Applied:
    ok: bool
    left_id: str = ""
    right_id: str = ""
    polarity: str = ""
    weight: float = 0.0
    reason: str = ""

    def describe(self) -> str:
        if not self.ok:
            return f"not applied: {self.reason}"
        if self.polarity == "unrelated":
            return (f"noted: these are unrelated; the link is vetoed and will not be proposed again "
                    f"(weight {self.weight:.2f})")
        return f"noted: these are related (weight {self.weight:.2f})"


def _resolve(store: Any, text: str) -> str | None:
    from .path_query import resolve

    return resolve(store, text)


def _edge(store: Any, left: str, right: str) -> dict[str, Any] | None:
    try:
        row = store._conn.execute(
            """SELECT src_id, dst_id, relation, weight FROM edges
               WHERE (src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?)
               LIMIT 1""",
            (left, right, right, left),
        ).fetchone()
    except Exception:  # noqa: BLE001
        return None
    return dict(row) if row else None


def veto(store: Any, left_id: str, right_id: str) -> bool:
    """Has a person said these two are unrelated?"""
    try:
        _ensure(store)
        row = store._conn.execute(
            f"""SELECT polarity FROM {TABLE}
                WHERE (left_id = ? AND right_id = ?) OR (left_id = ? AND right_id = ?)""",
            (left_id, right_id, right_id, left_id),
        ).fetchone()
    except Exception:  # noqa: BLE001
        return False
    return bool(row and str(row["polarity"]) == "unrelated")


def apply(
    store: Any,
    left: str,
    right: str,
    sense: str,
    *,
    note: str = "",
    at: datetime | None = None,
) -> Applied:
    """Record and apply one human judgement about a pair."""
    left_id, right_id = _resolve(store, left), _resolve(store, right)
    if left_id is None or right_id is None:
        missing = left if left_id is None else right
        return Applied(False, reason=f"unknown memory: {missing}")
    if left_id == right_id:
        return Applied(False, reason="the same memory on both sides")
    if sense not in ("related", "unrelated"):
        return Applied(False, reason=f"unknown polarity: {sense}")

    _ensure(store)
    stamp = (at or _now()).isoformat()
    existing = _edge(store, left_id, right_id)
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT INTO {TABLE} (left_id, right_id, polarity, note, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(left_id, right_id) DO UPDATE SET
                    polarity = excluded.polarity, note = excluded.note,
                    created_at = excluded.created_at""",
            (str(left_id), str(right_id), sense, str(note)[:500], stamp),
        )

    if sense == "unrelated":
        target = VETO_WEIGHT
        if existing:
            with store.transaction() as conn:
                conn.execute(
                    """UPDATE edges SET weight = ?, last_reinforced_at = ?
                       WHERE src_id = ? AND dst_id = ? AND relation = ?""",
                    (target, stamp, existing["src_id"], existing["dst_id"],
                     existing["relation"]),
                )
        return Applied(True, left_id, right_id, sense, target)

    if existing:
        current = float(existing["weight"] or 0.0)
        target = min(MAX_WEIGHT, max(REINFORCE_TO, current + REINFORCE_STEP))
        with store.transaction() as conn:
            conn.execute(
                """UPDATE edges SET weight = ?, evidence_count = evidence_count + 1,
                       last_reinforced_at = ?
                   WHERE src_id = ? AND dst_id = ? AND relation = ?""",
                (target, stamp, existing["src_id"], existing["dst_id"], existing["relation"]),
            )
    else:
        target = REINFORCE_TO
        store.add_edge(
            left_id, right_id, "related", weight=target,
            evidence_type="user_feedback", evidence_key=f"user_feedback:{left_id}:{right_id}",
            explanation="a person said these two are related",
        )
    return Applied(True, left_id, right_id, sense, target)


def apply_statement(store: Any, statement: str, *, at: datetime | None = None) -> Applied:
    """"A and B are not related" -> an applied judgement, or a refusal saying why not."""
    parsed = parse(statement)
    if parsed is None:
        return Applied(False, reason="could not read two memories and a polarity")
    left, right, sense = parsed
    return apply(store, left, right, sense, note=statement, at=at)


def feedback(store: Any, *, limit: int = 50) -> list[dict[str, Any]]:
    try:
        _ensure(store)
        rows = store._conn.execute(
            f"SELECT * FROM {TABLE} ORDER BY created_at DESC LIMIT ?", (int(limit),)
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    return [dict(row) for row in rows]


def report(store: Any) -> dict[str, Any]:
    rows = feedback(store, limit=100000)
    return {
        "judgements": len(rows),
        "related": sum(1 for row in rows if str(row["polarity"]) == "related"),
        "unrelated": sum(1 for row in rows if str(row["polarity"]) == "unrelated"),
        "enforced_by_co_retrieval": True,
    }
