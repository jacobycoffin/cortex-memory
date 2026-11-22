"""Core memory must never contain two contradictory lines (backlog item #130).

The always-on block is read on every turn, so a contradiction in it is not a bad answer once —
it is a bad answer forever, and the model resolves the conflict however it likes. A memory
store can tolerate tension (history is often the answer); the *profile* cannot.

Two signals, in order of certainty:

1. **Same claim, different value.** The store already carries `subject`, `predicate` and
   `object_value`. Two memories with the same subject and predicate but different objects are
   a contradiction by construction — "maintenance window / day / Sunday" against "… / Tuesday".
2. **Same words, opposite polarity.** Without structured triples, the tell is high lexical
   overlap plus a negation on exactly one side: "the channel is the V2 nightly" against "the
   channel is not the V2 nightly".

A near-duplicate with the *same* polarity is not a conflict — that is a restatement, and #216
already handles those by recording a confirmation.

The check blocks the write rather than hiding a line at render time. Silently dropping one of
two contradictory memories would mean the block quietly changing its mind depending on sort
order, which is worse than refusing the promotion and saying why.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .redundancy import jaccard, tokens

# Same words, opposite polarity: measured restatements sit at 0.60+ Jaccard, so a lower bar
# than that catches unrelated sentences that happen to share a negation.
POLARITY_SIMILARITY = 0.45
_NEGATIONS = ("not ", "never ", "no longer", "no ", "without", "stopped", "disabled",
              "removed", "instead")
# Words that carry no claim of their own: two sentences that differ only by these plus a
# negation are the same claim with opposite polarity.
_FILLER = {"not", "never", "no", "longer", "stopped", "disabled", "removed", "instead", "any",
           "all", "really", "ever", "now", "at", "anymore", "isn't", "aren't", "don't", "doesn't"}


def _polarity_claim(left: str, right: str) -> bool:
    """Same claim, one side negated: the *only* difference is negation words.

    A fixed similarity threshold could not separate these from restatements — "nightly" vs
    "not nightly at all" scored 0.50 Jaccard while "with no preamble" vs "without a preamble"
    scored higher still. Comparing the token difference is the discriminator: if everything
    one sentence adds is negation or filler, they are the same claim with opposite polarity.
    """
    left_tokens, right_tokens = tokens(left), tokens(right)
    if not left_tokens or not right_tokens:
        return False
    shared = left_tokens & right_tokens
    if len(shared) < 0.6 * min(len(left_tokens), len(right_tokens)):
        return False
    # Polarity is a property of the sentence, not of the token difference: "with no preamble"
    # and "without a preamble" mean the same negative thing and must not read as opposites.
    if _negated(left) == _negated(right):
        return False
    only_left = left_tokens - right_tokens
    only_right = right_tokens - left_tokens
    # Everything one sentence adds beyond the other must be negation or filler.
    return (not only_left or only_left <= _FILLER) and (
        not only_right or only_right <= _FILLER
    )


@dataclass(frozen=True)
class Conflict:
    memory_id: str
    reason: str

    def describe(self) -> str:
        return f"{self.memory_id[:8]}: {self.reason}"


def _negated(text: str) -> bool:
    lowered = f" {str(text or '').lower()} "
    return any(marker in lowered for marker in _NEGATIONS)


def _triple(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("subject") or "").strip().lower(),
        str(row.get("predicate") or "").strip().lower(),
        str(row.get("object_value") or "").strip().lower(),
    )


def conflicts_with(candidate: dict[str, Any], existing: dict[str, Any]) -> str | None:
    """Why these two cannot both be in core, or None."""
    if str(candidate.get("id")) == str(existing.get("id")):
        return None
    left_subject, left_predicate, left_object = _triple(candidate)
    right_subject, right_predicate, right_object = _triple(existing)
    if left_subject and left_subject == right_subject and left_predicate == right_predicate:
        if left_object and right_object and left_object != right_object:
            return (
                f"same claim ({left_subject} / {left_predicate}) with a different value: "
                f"'{left_object}' vs '{right_object}'"
            )
    left_text = str(candidate.get("content") or "")
    right_text = str(existing.get("content") or "")
    if _polarity_claim(left_text, right_text):
        similarity = jaccard(tokens(left_text), tokens(right_text))
        return f"opposite polarity on the same wording (similarity {similarity:.2f})"
    return None


def conflicts_for(
    candidate: dict[str, Any], existing: Iterable[dict[str, Any]]
) -> list[Conflict]:
    found = []
    for row in existing:
        reason = conflicts_with(candidate, row)
        if reason:
            found.append(Conflict(str(row.get("id") or ""), reason))
    return found


def core_rows(store: Any) -> list[dict[str, Any]]:
    """Everything currently rendered into core: pinned plus promoted members."""
    rows: list[dict[str, Any]] = []
    for query in (
        """SELECT id, content, subject, predicate, object_value FROM memories
           WHERE state = 'active' AND pinned = 1""",
        """SELECT m.id, m.content, m.subject, m.predicate, m.object_value
           FROM memories m JOIN core_members c ON c.memory_id = m.id
           WHERE m.state = 'active'""",
    ):
        try:
            rows.extend(dict(row) for row in store._conn.execute(query).fetchall())
        except Exception:  # noqa: BLE001 - core_members is optional
            continue
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        seen.setdefault(str(row.get("id")), row)
    return list(seen.values())


def check(store: Any, memory_id: str) -> list[Conflict]:
    """Conflicts that would be created by putting this memory into core."""
    row = store.get_memory(memory_id)
    if not row:
        return []
    candidates = {str(r.get("id")): r for r in core_rows(store)}
    candidates.pop(str(memory_id), None)
    return conflicts_for(dict(row), candidates.values())


def all_conflicts(store: Any) -> list[tuple[str, str, str]]:
    """Every contradictory pair already inside core: `(left, right, reason)`.

    A fresh install should have none; this exists so "should have none" can be checked rather
    than assumed.
    """
    rows = core_rows(store)
    found: list[tuple[str, str, str]] = []
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            reason = conflicts_with(left, right)
            if reason:
                found.append((str(left.get("id")), str(right.get("id")), reason))
    return found


def summarise(conflicts: Iterable[Conflict]) -> str:
    return "; ".join(conflict.describe() for conflict in conflicts)
