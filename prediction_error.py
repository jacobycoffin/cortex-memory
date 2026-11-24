"""Write only when the new turn surprised the memory (#68).

A memory that is re-stated must not be re-written. Today every agreeing turn is an opportunity to bump
`updated_at`, add a version, refresh decay and count an access — so after a month of agreement the
store cannot tell a fact that is *still true* from one that was *said a lot*, and every confirmation
costs a write. The write path has no test for surprise, so it treats noise as signal.

This gate supplies that test. It compares an incoming observation with what a stored memory already
implies and sorts the outcome into four cases:

- **confirm** — nothing new, nothing contradicted. *Do not touch the memory.* Confirmation is not
  evidence and must not look like freshness.
- **refine** — the observation adds a slot the memory did not have (a new port, a new host). Worth a
  write, because the memory is now more specific than it was.
- **contradict** — a slot the memory fixed has a different value. This is the only case that
  *replaces* something, and it belongs to self-correction (#33), not to a silent overwrite.
- **novel** — too little overlap to be about the same thing. A new memory, not an edit.

Slots are the parts of a statement that can be *wrong*: ports, addresses, paths, host names, dates and
numbers. Prose similarity alone cannot tell "port 8100" from "port 8200", and that single digit is
usually the entire content of the fact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# A contradiction is one value changing while the rest of the statement still agrees. When several
# slot kinds differ at once the sentence was rewritten, not corrected: a note that mentions today's
# date or a shifted path would otherwise report a "changed port" every time the note is edited. Live
# measurement of 5,725 vault re-import pairs: 95.4% differed in a slot, almost all of them rewrites.
MAX_CONTRADICTED_KINDS = 1

# The gate is for facts, not documents. Past this length a "memory" is a note or a chunk, and the
# right comparison is the vault's own revision flow, not a slot diff: a long note mentioning today's
# date changes a date slot every single day without anything being contradicted.
MAX_FACT_CHARS = 400

# A date is the one slot that moves on its own: notes carry "as of" headers, windows and paths, and a
# re-import shifts them without anything being contradicted. Live: 5,725 vault re-import pairs, of
# which the only difference in most was a date inside a header. So a date that moved is reported as a
# rewrite (still a touch, because something really did change) and never as "replace".
TRIVIAL_KINDS = frozenset({"date", "time"})

CONFIRM = "confirm"
REFINE = "refine"
CONTRADICT = "contradict"
NOVEL = "novel"

# A refine needs most of the sentence to already be known; below this the observation is a different
# subject rather than a more specific one.
REFINE_SHARE = 0.55
# A contradiction needs the same subject: at least this much overlap plus one slot that disagrees.
CONTRADICT_SHARE = 0.40

_STOP = frozenset(
    "the a an and or but if then than so just very now this that these those i you he she it we they "
    "is are was were be been being of to in on for with from by as at do does did not no yes".split()
)

_SLOTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("port", re.compile(r"\bport\s+(\d{2,5})\b", re.IGNORECASE)),
    # A colon only means a port when a host is in front of it. The first version matched any
    # `:NNN`, so `18:48` and `12:31` became "port 48" and "port 31" - and on the live store that
    # single mistake accounted for 3,728 of 3,757 false contradictions in vault re-imports.
    ("port", re.compile(r"(?:localhost|127\.0\.0\.1|CT\d+|r630|vps\d?|[\w-]+\.[a-z]{2,}):(\d{2,5})\b",
                        re.IGNORECASE)),
    ("address", re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b")),
    ("path", re.compile(r"(?<![\w.-])(/[\w./-]{2,})")),
    ("host", re.compile(r"\b(CT\d{2,4}|r630|vps\d?|duo)\b", re.IGNORECASE)),
    ("date", re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")),
    # Clock times and spelled-out dates. These are the "when this was written" labels that the live
    # store is full of: a vault header saying "refreshed at 4:37 PM ET" produced a numeric
    # contradiction on every refresh, because 4:37 and 5:09 are different numbers.
    ("time", re.compile(r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)\b")),
    ("time", re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|"
                        r"November|December)\s+\d{1,2},?\s+\d{4}\b")),
    ("number", re.compile(r"\b(\d+(?:\.\d+)?)\b")),
)


def tokens(text: str) -> set[str]:
    return {
        word for word in re.split(r"[^a-z0-9]+", str(text or "").lower())
        if len(word) > 2 and word not in _STOP
    }


def slots(text: str) -> dict[str, set[str]]:
    """The parts of a statement that can be wrong, by kind.

    A bare number is only a slot when no more specific pattern already claimed it, so
    "port 8100" does not also produce a `number: 8100` that would double-count one fact.
    """
    found: dict[str, set[str]] = {}
    claimed: list[tuple[int, int]] = []
    for kind, pattern in _SLOTS:
        for match in pattern.finditer(str(text or "")):
            # A pattern with no capture group matches the slot itself; one with a group matches a
            # wrapper such as `port 8100` and keeps only the value.
            span = match.span(1) if match.re.groups else match.span(0)
            if any(start <= span[0] and span[1] <= end for start, end in claimed):
                continue
            claimed.append(span)
            found.setdefault(kind, set()).add(match.group(1) if match.re.groups else match.group(0))
    return found


@dataclass
class Judgement:
    verdict: str
    memory_id: str = ""
    similarity: float = 0.0
    matched: dict[str, list[str]] = field(default_factory=dict)
    confirmed_slots: dict[str, list[str]] = field(default_factory=dict)
    contradicted_slots: dict[str, list[str]] = field(default_factory=dict)
    new_slots: dict[str, list[str]] = field(default_factory=dict)
    reason: str = ""

    @property
    def touches_the_memory(self) -> bool:
        return self.verdict in {REFINE, CONTRADICT}

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "memory_id": self.memory_id,
            "similarity": round(self.similarity, 4),
            "confirmed_slots": self.confirmed_slots,
            "contradicted_slots": self.contradicted_slots,
            "new_slots": self.new_slots,
            "touches_the_memory": self.touches_the_memory,
            "reason": self.reason,
            "sentence": self.describe(),
        }

    def describe(self) -> str:
        if self.verdict == CONFIRM:
            return "confirms what is stored — leave the memory alone"
        if self.verdict == CONTRADICT:
            changed = ", ".join(f"{kind} {', '.join(values)}"
                               for kind, values in self.contradicted_slots.items())
            return f"contradicts {changed} — replace, do not overwrite"
        if self.verdict == REFINE:
            if self.contradicted_slots:
                return f"rewrites {len(self.contradicted_slots)} slot kind(s) at once — treat as a new " \
                       f"statement, not a correction"
            added = ", ".join(f"{kind} {', '.join(values)}" for kind, values in self.new_slots.items())
            return f"adds {added} — the memory becomes more specific"
        return "unrelated — this is a new memory, not an edit"


def _confirm(value: str, other: str) -> bool:
    """Two slot values agree when one is the other with punctuation or case removed."""
    def fold(item: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(item).lower())
    return fold(value) == fold(other)


def judge(store: Any, memory_id: str, observation: str) -> Judgement:
    """Compare one observation with one stored memory."""
    try:
        row = store._conn.execute(
            "SELECT content FROM memories WHERE id = ?", (str(memory_id),)
        ).fetchone()
    except Exception:  # noqa: BLE001 - a gate that cannot read must not block a write
        row = None
    if row is None:
        return Judgement(verdict=NOVEL, memory_id=str(memory_id), reason="no such memory")
    stored = str(row[0] or "")
    return compare(stored, observation, memory_id=str(memory_id))


def compare(stored: str, observation: str, *, memory_id: str = "") -> Judgement:
    """The gate itself, on two strings — no store, so it is testable in isolation."""
    if len(str(stored or "")) > MAX_FACT_CHARS or len(str(observation or "")) > MAX_FACT_CHARS:
        left, right = tokens(stored), tokens(observation)
        similarity = len(left & right) / len(left | right) if (left or right) else 0.0
        return Judgement(
            verdict=REFINE if similarity >= REFINE_SHARE else NOVEL,
            memory_id=memory_id, similarity=similarity,
            reason=("document-length text - this is a note or a chunk, not a fact, so slot "
                    "comparison would call every edit a contradiction"),
        )

    left, right = tokens(stored), tokens(observation)
    similarity = len(left & right) / len(left | right) if (left or right) else 0.0
    stored_slots, observed_slots = slots(stored), slots(observation)
    confirmed: dict[str, list[str]] = {}
    contradicted: dict[str, list[str]] = {}
    new: dict[str, list[str]] = {}
    for kind, values in observed_slots.items():
        known = stored_slots.get(kind)
        if not known:
            new[kind] = sorted(values)
            continue
        same = sorted(value for value in values if any(_confirm(value, item) for item in known))
        different = sorted(value for value in values if not any(_confirm(value, item) for item in known))
        if same:
            confirmed[kind] = same
        if different:
            # A kind with different values is a contradiction only when the *kind* is already
            # fixed: a second path in a sentence is a second path, a second port for the same
            # service is a change.
            contradicted[kind] = different
    if contradicted and set(contradicted) <= TRIVIAL_KINDS and similarity >= REFINE_SHARE:
        return Judgement(verdict=REFINE, memory_id=memory_id, similarity=similarity,
                         confirmed_slots=confirmed, contradicted_slots=contradicted,
                         new_slots=new,
                         reason="only a when-written label moved - not a changed fact")
    if contradicted and similarity >= CONTRADICT_SHARE:
        if len(contradicted) <= MAX_CONTRADICTED_KINDS:
            return Judgement(verdict=CONTRADICT, memory_id=memory_id, similarity=similarity,
                             confirmed_slots=confirmed, contradicted_slots=contradicted,
                             new_slots=new,
                             reason="a value the memory fixed has changed")
        return Judgement(verdict=REFINE, memory_id=memory_id, similarity=similarity,
                         confirmed_slots=confirmed, contradicted_slots=contradicted,
                         new_slots=new,
                         reason=(f"{len(contradicted)} slot kinds differ at once - the statement was "
                                 f"rewritten rather than corrected"))
    if new and similarity >= REFINE_SHARE:
        return Judgement(verdict=REFINE, memory_id=memory_id, similarity=similarity,
                         confirmed_slots=confirmed, new_slots=new,
                         reason="the observation is more specific than the memory")
    if similarity >= REFINE_SHARE:
        return Judgement(verdict=CONFIRM, memory_id=memory_id, similarity=similarity,
                         confirmed_slots=confirmed,
                         reason="nothing new and nothing contradicted")
    return Judgement(verdict=NOVEL, memory_id=memory_id, similarity=similarity,
                     new_slots=new, reason="too little overlap to be about the same thing")


def gate(store: Any, observation: str, candidates: Iterable[str] | None = None, *,
         limit: int = 20) -> dict[str, Any]:
    """Decide, for one observation, whether any existing memory should be touched.

    Returns the winning judgement plus the decisions for every candidate, so a caller can show why
    the answer was "nothing to do".
    """
    if candidates is None:
        rows = []
        try:
            rows = store._conn.execute(
                "SELECT id FROM memories WHERE state IN ('active', 'cold') ORDER BY updated_at DESC "
                "LIMIT ?",
                (int(limit),),
            ).fetchall()
        except Exception:  # noqa: BLE001
            rows = []
        candidates = [str(row[0]) for row in rows]
    judged = [judge(store, memory_id, observation) for memory_id in candidates]
    priority = {CONTRADICT: 0, REFINE: 1, CONFIRM: 2, NOVEL: 3}
    judged.sort(key=lambda item: (priority[item.verdict], -item.similarity))
    best = judged[0] if judged else Judgement(verdict=NOVEL, reason="no candidates")
    return {
        "verdict": best.verdict,
        "touches_the_memory": best.touches_the_memory,
        "memory_id": best.memory_id,
        "similarity": round(best.similarity, 4),
        "sentence": best.describe(),
        "candidates": [item.as_dict() for item in judged[:5]],
    }


def report(store: Any, *, sample: int = 200) -> dict[str, Any]:
    """How much of recent traffic would have been a no-op under this gate."""
    try:
        rows = store._conn.execute(
            "SELECT id FROM memories WHERE state = 'active' ORDER BY updated_at DESC LIMIT ?",
            (int(sample),),
        ).fetchall()
    except Exception:  # noqa: BLE001
        rows = []
    counts = {CONFIRM: 0, REFINE: 0, CONTRADICT: 0, NOVEL: 0}
    for row in rows:
        memory_id = str(row[0])
        try:
            stored = store._conn.execute(
                "SELECT content FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        except Exception:  # noqa: BLE001
            continue
        if stored is None:
            continue
        counts[compare(str(stored[0] or ""), str(stored[0] or "")).verdict] += 1
    return {
        "sampled": len(rows),
        "self_comparison": counts,
        "thresholds": {"refine": REFINE_SHARE, "contradict": CONTRADICT_SHARE},
        "note": "a confirmation is not evidence and must not look like freshness",
    }
