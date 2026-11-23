"""Decide before writing, against the neighbours the store already has (#28).

Dedupe after insert is cleanup: the duplicate is already in the store, already indexed, and already
capable of being retrieved alongside its twin, so every later pass has to undo a decision instead of
making one. The cheaper moment is **before** the row exists, with the fifty nearest memories in hand.

Three outcomes, and only one of them inserts:

| verdict | when | what happens |
|---|---|---|
| `reject` | the same fact verbatim, or a restatement that adds nothing | nothing is written; the existing memory is returned |
| `merge` | a restatement that adds detail, or the same fact from a better source | the existing row is updated, not duplicated |
| `insert` | genuinely new | a new memory is created |

The thresholds are the ones this store already uses elsewhere, because a second, slightly different
similarity rule is how two subsystems start disagreeing: **verbatim ≥ 0.85 Jaccard**, or a
**restatement ≥ 0.60 similarity with ≥ 0.80 containment** of the shorter text in the longer. Anything
below that is a neighbour, not a duplicate — and neighbours are worth showing, because the most
expensive dedupe mistake is merging two facts that merely share vocabulary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

VERBATIM_JACCARD = 0.85
RESTATEMENT_SIMILARITY = 0.60
RESTATEMENT_CONTAINMENT = 0.80
NEIGHBOURS = 50

_STOP = frozenset(
    """a an and are as at be been but by for from had has have in into is it its of on or that the
    their them then these they this to was were will with""".split()
)
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*")


_TRAILING = ".,;:!?\"'"


def tokens(text: str) -> set[str]:
    """Words without their punctuation: "nginx." and "nginx" are the same word.

    The first version kept the full stop, which cost a third of the overlap between two sentences
    that differ only by a trailing clause — enough to turn a duplicate into an insert.
    """
    found = (token.lower().strip(_TRAILING) for token in _TOKEN.findall(str(text or "")))
    return {token for token in found if token and token not in _STOP}


def jaccard(left: str, right: str) -> float:
    left_tokens, right_tokens = tokens(left), tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def containment(shorter: str, longer: str) -> float:
    """How much of the shorter text's vocabulary appears in the longer one."""
    short_tokens, long_tokens = tokens(shorter), tokens(longer)
    if not short_tokens:
        return 0.0
    return len(short_tokens & long_tokens) / len(short_tokens)


def neighbours(store: Any, content: str, *, limit: int = NEIGHBOURS,
               exclude: str = "") -> list[dict[str, Any]]:
    """The memories closest to this text, ranked by Jaccard. Read-only, no writes."""
    text = str(content or "").strip()
    if not text:
        return []
    query_tokens = sorted(tokens(text), key=lambda token: (-len(token), token))[:3]
    seen: dict[str, dict[str, Any]] = {}
    for token in query_tokens:
        try:
            rows = store._conn.execute(
                """SELECT id, content, state, source_category, confidence, created_at
                   FROM memories
                   WHERE state NOT IN ('tombstoned','deleted') AND content LIKE ?
                   ORDER BY created_at DESC LIMIT ?""",
                (f"%{token}%", int(limit) * 4),
            ).fetchall()
        except Exception:  # noqa: BLE001 - no memories table, no neighbours
            return []
        for row in rows:
            memory_id = str(row["id"])
            if memory_id == str(exclude):
                continue  # asking about an existing row must not find itself
            if memory_id not in seen:
                seen[memory_id] = dict(row)
    ranked = [
        {
            "memory_id": memory_id,
            "content": str(row["content"]),
            "state": str(row["state"]),
            "source_category": str(row["source_category"] or ""),
            "confidence": float(row["confidence"] or 0.0),
            "similarity": round(jaccard(text, str(row["content"])), 4),
        }
        for memory_id, row in seen.items()
    ]
    ranked.sort(key=lambda item: (-item["similarity"], item["memory_id"]))
    return ranked[: max(0, int(limit))]


@dataclass
class Verdict:
    action: str
    target_id: str = ""
    similarity: float = 0.0
    containment: float = 0.0
    reason: str = ""
    neighbours: list[dict[str, Any]] = field(default_factory=list)

    @property
    def writes(self) -> bool:
        return self.action in ("insert", "merge")

    def describe(self) -> str:
        if self.action == "insert":
            return "no duplicate found; write it"
        return (
            f"{self.action} with {self.target_id[:8]} ({self.similarity:.2f} similar, "
            f"{self.containment:.2f} containment): {self.reason}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "target_id": self.target_id,
            "similarity": round(self.similarity, 4),
            "containment": round(self.containment, 4),
            "reason": self.reason,
            "sentence": self.describe(),
            "neighbours": [{"id": item["memory_id"], "similarity": item["similarity"],
                            "content": item["content"][:120]} for item in self.neighbours[:5]],
        }


def decide(store: Any, content: str, *, limit: int = NEIGHBOURS,
           exclude: str = "") -> Verdict:
    """Insert, merge or reject — decided against the nearest memories, before any write."""
    text = str(content or "").strip()
    if not text:
        return Verdict("reject", reason="nothing to write")
    found = neighbours(store, text, limit=limit, exclude=exclude)
    verdict = Verdict("insert", neighbours=found)
    for candidate in found:
        other = str(candidate["content"])
        similarity = float(candidate["similarity"])
        # Containment is only meaningful on the shorter side: asking "how much of the *longer*
        # text appears in the shorter one" answers zero for every honest restatement that adds a
        # clause, and the first version of this test caught exactly that.
        short, long = (text, other) if len(tokens(text)) <= len(tokens(other)) else (other, text)
        contained = containment(short, long)
        if similarity >= VERBATIM_JACCARD:
            return Verdict(
                "reject", str(candidate["memory_id"]), similarity, contained,
                "the same fact verbatim", found,
            )
        if similarity >= RESTATEMENT_SIMILARITY and contained >= RESTATEMENT_CONTAINMENT:
            # Same claim, different words: keep one row and put the incoming text on it, so the
            # store never holds two memories that a future recall has to arbitrate between.
            return Verdict(
                "merge", str(candidate["memory_id"]), similarity, contained,
                "a restatement of the same fact", found,
            )
    return verdict


def apply(store: Any, content: str, verdict: Verdict) -> dict[str, Any]:
    """Carry out a verdict. Merge updates the existing row; it never creates a second one."""
    if verdict.action == "reject":
        return {"written": False, "memory_id": verdict.target_id, "action": "reject"}
    if verdict.action == "merge":
        try:
            with store.transaction() as conn:
                conn.execute(
                    """UPDATE memories SET content = ?, updated_at = ?
                       WHERE id = ?""",
                    (str(content), _now(), verdict.target_id),
                )
        except Exception as error:  # noqa: BLE001 - a failed merge must not lose the fact
            return {"written": False, "action": "merge_failed", "error": type(error).__name__}
        return {"written": True, "memory_id": verdict.target_id, "action": "merge"}
    return {"written": False, "action": "insert_required"}


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def report(store: Any, samples: Sequence[str] = ()) -> dict[str, Any]:
    described = {
        "thresholds": {
            "verbatim_jaccard": VERBATIM_JACCARD,
            "restatement_similarity": RESTATEMENT_SIMILARITY,
            "restatement_containment": RESTATEMENT_CONTAINMENT,
            "neighbours": NEIGHBOURS,
        },
        "samples": [],
    }
    for sample in samples:
        described["samples"].append(decide(store, sample).as_dict())
    return described
