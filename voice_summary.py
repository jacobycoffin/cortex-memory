"""Recall, spoken (#147)

Text written for a screen is bad speech: ids, scores, brackets, bullets, and a sentence per memory
that reads like a database dump. On a voice channel none of that survives — a listener needs a
sentence they can hold in their head, and they need it in one breath.

So this module renders a recall result set as **speech**:

- **no ids, no scores, no brackets.** A memory id spoken aloud is noise; a score is not something a
  listener can act on.
- **one sentence, then the items.** "I know here's two things: …" sets the size before the content, so
  the listener knows when it ends.
- **bounded in characters, not items.** `MAX_CHARS` is a breath, and the summary says *"and N more"*
  rather than reading until the listener gives up.
- **hedged when the store is unsure.** A memory below `SURE` confidence is prefixed *"I think"*, because
  saying a guess in the same voice as a fact is the one thing speech cannot take back.
- **symbols become words.** `→` is not spoken, `&` is not spoken, `%` is not spoken. Numbers stay
  digits: "port 8100" reads correctly aloud, "eight thousand one hundred" does not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

MAX_ITEMS = 3
MAX_CHARS = 280
SURE = 0.8

_SPOKEN = (
    (re.compile(r"\s*[→➔➜]\s*"), " to "),
    (re.compile(r"\s*&\s*"), " and "),
    (re.compile(r"\s*[•·]\s*"), ", "),
    (re.compile(r"\s*/\s*"), " or "),
    (re.compile(r"\s*—\s*"), ", "),
    (re.compile(r"\s+"), " "),
)
_MARKUP = re.compile(r"[*_`#>|\[\]]")
# A memory id spoken aloud is noise. UUIDs and long hex runs are removed before anything else.
# Full UUIDs, truncated id fragments ("7b61d06a-fef8-40dc"), and bare hex runs. A fragment must
# contain a letter, so a date like 2026-11-24 is left alone.
_IDS = re.compile(
    r"\b(?=[0-9a-f-]*[a-f])[0-9a-f]{4,}(?:-[0-9a-f]{2,})+\b"
    r"|\b(?=[0-9a-f]*[a-f])[0-9a-f]{8,}\b",
    re.IGNORECASE,
)
_TRAILING = re.compile(r"[\s,;:.]+$")


def speakable(text: Any) -> str:
    """One clean sentence-shaped phrase, with the screen furniture removed."""
    body = str(text or "")
    body = _IDS.sub(" ", body)
    body = _MARKUP.sub(" ", body)
    body = re.sub(r"\([^)]*\)", " ", body)  # parenthetical detail is a screen habit
    for pattern, replacement in _SPOKEN:
        body = pattern.sub(replacement, body)
    body = body.replace("%", " percent")
    body = " ".join(body.split())
    return _TRAILING.sub("", body.strip())


def trim(text: str, limit: int) -> str:
    """Cut on a word boundary — a sentence that stops mid-word is not speech."""
    body = speakable(text)
    if len(body) <= limit:
        return body
    cut = body[: max(1, limit - 1)]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return _TRAILING.sub("", cut) + "…"


@dataclass
class Spoken:
    sentence: str = ""
    items: list[str] = field(default_factory=list)
    hedged: int = 0
    dropped: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "sentence": self.sentence,
            "items": self.items,
            "hedged": self.hedged,
            "dropped": self.dropped,
            "chars": len(self.sentence),
        }


def sure_threshold(values: Iterable[float], *, share: float = 0.2) -> float:
    """A hedging threshold set against the store's own confidence distribution.

    An absolute threshold does not survive contact with a real store: this one has a median of 0.78
    with a pile-up at exactly 0.78, so `SURE = 0.8` marks **half of everything** as a guess — and a
    hedge that fires on half the readouts stops carrying information. Ask instead for the confidence
    above which a stated `share` of memories sit, and hedge below that.
    """
    numbers = []
    for value in values or []:
        try:
            numbers.append(float(value))
        except (TypeError, ValueError):
            continue
    if not numbers:
        return SURE
    numbers.sort()
    target = max(0.0, min(1.0, float(share)))
    # Ties make an exact target impossible — this store has a pile-up at exactly 0.78 — so the rule is
    # "the highest threshold that hedges **no more than** `share`", which is answerable and honest.
    best = numbers[0]
    for value in numbers:
        below = sum(1 for other in numbers if other < value) / len(numbers)
        if below <= target:
            best = value
    return best


def _confidence(memory: Any) -> float:
    if isinstance(memory, dict):
        try:
            return float(memory.get("confidence") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(getattr(memory, "confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _content(memory: Any) -> str:
    if isinstance(memory, dict):
        return str(memory.get("content") or memory.get("text") or "")
    return str(getattr(memory, "content", "") or "")


def summarise(memories: Iterable[Any], *, max_items: int = MAX_ITEMS,
              max_chars: int = MAX_CHARS, sure: float = SURE) -> Spoken:
    """Render memories as one spoken sentence."""
    items = list(memories or [])
    found = Spoken()
    if not items:
        found.sentence = "I don't have anything on that."
        return found
    keep = items[: max(1, int(max_items))]
    found.dropped = max(0, len(items) - len(keep))
    phrases: list[str] = []
    budget = max(40, int(max_chars))
    for memory in keep:
        phrase = speakable(_content(memory))
        if not phrase:
            found.dropped += 1
            continue
        if _confidence(memory) and _confidence(memory) < sure:
            found.hedged += 1
            phrase = f"I think {phrase[0].lower() + phrase[1:] if phrase else phrase}"
        if len(phrase) > budget - 40:
            phrase = trim(phrase, max(24, budget - 40))
        phrases.append(phrase)
    if not phrases:
        found.sentence = "I don't have anything I can say about that."
        return found
    if len(phrases) == 1:
        opening = phrases[0]
    else:
        count = "two" if len(phrases) == 2 else "three" if len(phrases) == 3 else str(len(phrases))
        opening = f"Here's {count} things. {phrases[0]}, and {phrases[1]}"
        for phrase in phrases[2:]:
            opening += f", and {phrase}"
    opening += "."
    if found.dropped:
        opening += f" And {found.dropped} more I can go through if you want."
    found.items = phrases
    found.sentence = opening if len(opening) <= budget + 60 else trim(opening, budget) + "."
    return found


def spoken(memories: Iterable[Any], **kwargs: Any) -> str:
    """The single string a TTS engine should read."""
    return summarise(memories, **kwargs).sentence


def for_voice(rows: Iterable[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Convenience for the phone surface's list rows, which use `text` rather than `content`."""
    items = []
    for row in rows or []:
        items.append({
            "content": str((row or {}).get("text") or (row or {}).get("content") or ""),
            "confidence": float((row or {}).get("confidence") or 0.0),
        })
    return summarise(items, **kwargs).as_dict()
