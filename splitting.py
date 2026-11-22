"""One message carrying three facts becomes three records (#34).

A turn like *"my dashboard runs on 8100, the vault lives at ~/vault, and my timezone is Eastern"*
is three perfectly good memories and one bad one: stored whole, it matches a query about any of the
three only weakly, and it cannot be superseded, forgotten or pinned on its own.

Splitting sentence-wise is already done. The remaining cases are:

- **several sentences** — each stands alone;
- **a compound sentence** — "the dashboard runs on 8100 and the vault lives at ~/vault" — two facts
  joined by a conjunction, where each side has its own subject;
- **a comma list** — "ports 8100, 8101 and 8102 are open" — one fact, and splitting it produces
  fragments that mean nothing ("8101 and 8102 are open").

So the rule is not "split on punctuation", it is **split only where each side survives alone**.
A side survives when it has at least three words and a verb-ish token, is not a continuation
("which…", "it also…"), and the boundary is not inside parentheses, quotes, a URL, or a code span.

Over-splitting is worse than under-splitting: two half-facts cannot be recalled by the query that
would have found the whole one, and there is no repair path — the halves look like independent
memories.
"""

from __future__ import annotations

import re

MIN_SIDE_WORDS = 3
MAX_FACTS = 8

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
# Match the keyword alone, not the whitespace around it: a masked region is all spaces, so a
# pattern with `\s+` on both sides starts the match inside the masked text and the left-hand fact
# loses its tail ("runs at <url>" became "runs at").
_KEYWORD = re.compile(r"(?:and|but|plus|also|while|whereas)", re.I)
_CONTINUATION = re.compile(
    r"^(?:which|that|who|whose|it|this|these|those|they|he|she|so|then|therefore)\b", re.I
)
_VERBISH = re.compile(
    r"\b(?:is|are|was|were|be|been|runs?|hosts?|lives?|uses?|needs?|requires?|prefers?|wants?|"
    r"works?|goes?|starts?|owns?|has|have|had|does|do|did|can|will|should|must|never|always|"
    r"installed|configured|deployed|moved|renamed|keeps?|returns?|proxies|serves?|says?|means?|"
    r"contains?|holds?|covers?)\b",
    re.I,
)
_PROTECTED = (
    re.compile(r"https?://\S+"),
    re.compile(r"`[^`]*`"),
    re.compile(r"\([^)]*\)"),
    re.compile(r"\"[^\"]*\"|'[^']*'"),
    re.compile(r"~/[\w./-]+"),
)
_INERT = re.compile(r"\b[\w.]+@[\w.]+|\b\d+\.\d+\.\d+\.\d+\b|\b\d+:\d+\b")


def _mask(text: str) -> str:
    """Blank out regions where a conjunction is not a fact boundary, keeping offsets stable."""
    masked = text
    for pattern in _PROTECTED:
        masked = pattern.sub(lambda match: " " * len(match.group(0)), masked)
    return _INERT.sub(lambda match: " " * len(match.group(0)), masked)


def _survives(piece: str) -> bool:
    stripped = piece.strip()
    if len(stripped.split()) < MIN_SIDE_WORDS:
        return False
    if _CONTINUATION.match(stripped):
        return False
    return bool(_VERBISH.search(stripped))



def _boundaries(masked: str):
    """Yield (start, end) of conjunctions that stand alone as words."""
    for match in _KEYWORD.finditer(masked):
        before = masked[match.start() - 1] if match.start() else " "
        after = masked[match.end():match.end() + 1]
        if not before.isspace() or not (after.isspace() or after == ""):
            continue
        yield match.start(), match.end()


def split_sentences(text: str, splitter: re.Pattern[str] | None = None) -> list[str]:
    pattern = splitter or _SENTENCE
    return [piece.strip() for piece in pattern.split(str(text or "")) if piece.strip()]


def split_compound(sentence: str) -> list[str]:
    """Split a compound sentence where both sides stand alone. Never splits a comma list."""
    text = str(sentence or "").strip()
    if not text:
        return []
    masked = _mask(text)
    facts: list[str] = []
    cursor = 0
    for match in _boundaries(masked):
        left_side = text[cursor:match[0]].strip()
        right_side = text[match[1]:].strip()
        if not _survives(left_side) or not _survives(right_side):
            continue
        facts.append(left_side)
        cursor = match[1]
    if not facts:
        return [text]
    tail = text[cursor:].strip()
    if tail:
        facts.append(tail)
    return [fact for fact in facts if fact]


def split_facts(text: str, *, splitter: re.Pattern[str] | None = None) -> list[str]:
    """Sentence-split, then compound-split each sentence. Capped, order preserved."""
    facts: list[str] = []
    for sentence in split_sentences(text, splitter):
        facts.extend(split_compound(sentence))
        if len(facts) >= MAX_FACTS:
            break
    return facts[:MAX_FACTS]


def count(text: str, *, splitter: re.Pattern[str] | None = None) -> int:
    return len(split_facts(text, splitter=splitter))


def describe(text: str, *, splitter: re.Pattern[str] | None = None) -> dict[str, object]:
    facts = split_facts(text, splitter=splitter)
    return {
        "facts": facts,
        "count": len(facts),
        "multi": len(facts) > 1,
        "at_cap": len(facts) >= MAX_FACTS,
    }
