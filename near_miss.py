"""Near-miss pairs are similar on purpose, and wrong together (#80).

The duplicate ceiling in retrieval (similarity >= 0.86) catches restatements. It cannot catch the
harder case: *"the dashboard runs on port 8100"* and *"the dashboard runs on port 8100 in staging"*
score around 0.8, they are **not** duplicates, and surfacing both is worse than surfacing either —
the reader gets two nearly identical lines that disagree, with no way to tell which one the answer
depends on.

Similarity alone can never decide this: the pair differs by one discriminating token, and which one
matters is exactly what similarity cannot see. So the pair is recorded as a **negative association**
by one of two judges:

- a **person** — "those two aren't the same thing" — the reliable source;
- **detection**, which only *proposes*: a numeric, negational or qualifying difference inside a high
  similarity band. Proposals are reported, never applied, because a threshold that quietly suppresses
  recall is the failure mode #73 and #78 both warned about.

Once a pair is recorded, the weaker of the two is dropped whenever both would be selected, and the
drop is reported with its reason. Neither memory is touched, hidden from its own queries, or given a
negative score — the suppression is about the **pair**, not about either memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

TABLE = "near_miss_pairs"
BAND_LOW = 0.55
BAND_HIGH = 0.86
MIN_DISCRIMINATOR_CHARS = 2

_NUMBER = re.compile(r"\b\d[\d.,:]*\b")
_NEGATION = re.compile(r"\b(?:not|no|never|without|cannot|can't|don't|isn't|aren't)\b", re.I)
_QUALIFIER = re.compile(
    r"\b(?:staging|production|prod|dev|test|beta|old|new|legacy|backup|primary|secondary|"
    r"before|after|temporary|temporarily|only|except)\b",
    re.I,
)
_STOP = frozenset(
    """a an and are as at be been but by for from had has have in into is it its of on or that the
    their them then these they this to was were will with""".split()
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*", str(text or ""))
        if token.lower() not in _STOP
    }


def similarity(left: str, right: str) -> float:
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def discriminator(left: str, right: str) -> list[str]:
    """The tokens that make this pair *not* the same fact, strongest evidence first."""
    left_text, right_text = str(left or ""), str(right or "")
    found: list[str] = []
    left_numbers = set(_NUMBER.findall(left_text))
    right_numbers = set(_NUMBER.findall(right_text))
    found.extend(sorted(left_numbers ^ right_numbers))
    left_negation = bool(_NEGATION.search(left_text))
    right_negation = bool(_NEGATION.search(right_text))
    if left_negation != right_negation:
        found.append("negation")
    left_qualifiers = {match.lower() for match in _QUALIFIER.findall(left_text)}
    right_qualifiers = {match.lower() for match in _QUALIFIER.findall(right_text)}
    found.extend(sorted(left_qualifiers ^ right_qualifiers))
    if not found:
        only_left = _tokens(left_text) - _tokens(right_text)
        only_right = _tokens(right_text) - _tokens(left_text)
        for token in sorted(only_left | only_right):
            if len(token) >= MIN_DISCRIMINATOR_CHARS:
                found.append(token)
    return found


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    left_id TEXT NOT NULL,
                    right_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    recorded_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (left_id, right_id)
                )"""
        )


@dataclass
class Proposal:
    left_id: str
    right_id: str
    similarity: float
    discriminator: list[str]

    def describe(self) -> str:
        return (f"{self.left_id[:8]} ~ {self.right_id[:8]} ({self.similarity:.2f}) "
                f"differs by {','.join(self.discriminator[:3])}")


def record(store: Any, left_id: str, right_id: str, *, reason: str = "",
           recorded_by: str = "user", at: datetime | None = None) -> bool:
    """Record a negative association. Canonical order, so the pair is the same either way round."""
    left, right = sorted((str(left_id), str(right_id)))
    if not left or not right or left == right:
        return False
    _ensure(store)
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT INTO {TABLE} (left_id, right_id, reason, recorded_by, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(left_id, right_id) DO UPDATE SET
                    reason = excluded.reason, recorded_by = excluded.recorded_by""",
            (left, right, str(reason)[:300], str(recorded_by), (at or _now()).isoformat()),
        )
    return True


def release(store: Any, left_id: str, right_id: str) -> int:
    left, right = sorted((str(left_id), str(right_id)))
    try:
        with store.transaction() as conn:
            cursor = conn.execute(
                f"DELETE FROM {TABLE} WHERE left_id = ? AND right_id = ?", (left, right)
            )
            return int(cursor.rowcount or 0)
    except Exception:  # noqa: BLE001
        return 0


def pairs(store: Any, *, limit: int = 500) -> list[dict[str, Any]]:
    # Read-only: creating the schema here would turn every recall into a write, and a read path that
    # commits is a read path that can change what it was asked to observe.
    try:
        rows = store._conn.execute(
            f"SELECT * FROM {TABLE} ORDER BY created_at DESC LIMIT ?", (int(limit),)
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    return [dict(row) for row in rows]


def suppressed(store: Any, ids: Iterable[str]) -> set[tuple[str, str]]:
    """Which of these ids are in a suppressed pair together."""
    wanted = {str(item) for item in ids}
    if len(wanted) < 2:
        return set()
    found: set[tuple[str, str]] = set()
    for row in pairs(store, limit=100000):
        left, right = str(row["left_id"]), str(row["right_id"])
        if left in wanted and right in wanted:
            found.add((left, right))
    return found


def demote(store: Any, ranked: Sequence[Any], *, score_of=lambda item: 0.0,
           id_of=lambda item: str(item)) -> list[int]:
    """Indexes to drop: the weaker half of each suppressed pair currently selected together.

    The pair rule, not a threshold: if two suppressed memories are about to surface together, one of
    them is redundant by a judgement already made. The stronger one survives.
    """
    ids = [id_of(item) for item in ranked]
    blocked = suppressed(store, ids)
    if not blocked:
        return []
    positions = {memory_id: index for index, memory_id in enumerate(ids)}
    drop: list[int] = []
    for left, right in blocked:
        if left not in positions or right not in positions:
            continue
        left_index, right_index = positions[left], positions[right]
        weaker = left_index if score_of(ranked[left_index]) <= score_of(ranked[right_index]) \
            else right_index
        if weaker not in drop:
            drop.append(weaker)
    return sorted(drop)


def detect(store: Any, *, band_low: float = BAND_LOW, band_high: float = BAND_HIGH,
           scan: int = 400, limit: int = 20) -> list[Proposal]:
    """Propose pairs in the near-miss band that differ by a discriminating token. Read-only."""
    try:
        rows = store._conn.execute(
            """SELECT id, content FROM memories
               WHERE state NOT IN ('tombstoned','deleted') AND LENGTH(content) >= 24
               ORDER BY created_at DESC LIMIT ?""",
            (int(scan),),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    existing = {
        (str(row["left_id"]), str(row["right_id"])) for row in pairs(store, limit=100000)
    }
    items = [(str(row["id"]), str(row["content"])) for row in rows]
    # Candidate pruning by a shared-token inverted index. Comparing every pair of long notes is
    # quadratic and pointless: a near-miss pair always shares most of its words, so pairs sharing
    # fewer than two tokens cannot reach the band and are never scored.
    index: dict[str, list[int]] = {}
    token_sets: list[set[str]] = []
    for position, (_, text) in enumerate(items):
        found = _tokens(text)
        token_sets.append(found)
        for token in found:
            index.setdefault(token, []).append(position)

    proposals: list[Proposal] = []
    for position in range(len(items)):
        shared_with: dict[int, int] = {}
        for token in token_sets[position]:
            for other in index[token]:
                if other > position:
                    shared_with[other] = shared_with.get(other, 0) + 1
        for other, shared in shared_with.items():
            if shared < 2:
                continue
            left_id, left_text = items[position]
            right_id, right_text = items[other]
            if tuple(sorted((left_id, right_id))) in existing:
                continue
            score = similarity(left_text, right_text)
            if not band_low <= score < band_high:
                continue
            differing = discriminator(left_text, right_text)
            if not differing:
                continue
            proposals.append(Proposal(left_id, right_id, round(score, 3), differing))
    proposals.sort(key=lambda item: (-item.similarity, item.left_id, item.right_id))
    return proposals[: max(0, int(limit))]


def report(store: Any, *, scan: int = 400) -> dict[str, Any]:
    recorded = pairs(store, limit=100000)
    proposals = detect(store, scan=scan)
    return {
        "recorded": len(recorded),
        "by_user": sum(1 for row in recorded if str(row["recorded_by"]) == "user"),
        "proposed": len(proposals),
        "band": [BAND_LOW, BAND_HIGH],
        "examples": [item.describe() for item in proposals[:3]],
    }
