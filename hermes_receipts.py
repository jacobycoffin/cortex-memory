"""Bounded receipt targeting and conversation-feedback recognition."""

from __future__ import annotations

import re
from typing import List


_POSITIVE_FEEDBACK = re.compile(
    r"\b(?:that worked|works now|perfect|exactly|great|thanks|thank you|solved|fixed it)\b", re.I
)


_STRONG_POSITIVE_FEEDBACK = re.compile(
    r"\b(?:that worked perfectly|exactly what i wanted|this is perfect|you nailed it|"
    r"nailed it|love this|couldn(?:'t| not) be better|best (?:answer|result|solution)|"
    r"completely solved|worked flawlessly)\b",
    re.I,
)


_CREATION_POSITIVE_FEEDBACK = re.compile(
    r"\b(?:that worked|works now|this helped|very helpful|perfect solution|solved it|fixed it)\b",
    re.I,
)


_NEGATIVE_FEEDBACK = re.compile(
    r"\b(?:that(?:'s| is) wrong|not right|outdated|incorrect|didn(?:'t| not) work|still broken|you forgot)\b", re.I
)


# The one token the recall receipt asks for. Deliberately not a bare word: "no" and "good" turn up inside ordinary
# sentences, and grading the wrong turn is worse than not grading one at all. ``cortex +`` means the answer was right,
# ``cortex -`` means it was wrong.
_EXPLICIT_GRADE = re.compile(r"(?:^|\s)cortex\s*([+-])(?=\s|$|[,.;:!?])", re.I)

GRADE_VALIDATED = "validated"
GRADE_CORRECTED = "corrected"


def parse_explicit_grade(text: str) -> str | None:
    """Return the task outcome the user just stated, or None.

    ``cortex +`` is a *validated* answer — the reasoning path and the memories behind it were checked and held. ``cortex -``
    is a *corrected* one, which is stronger evidence than "harmful": something was positively wrong, not merely unhelpful.
    """

    match = _EXPLICIT_GRADE.search(text or "")
    if not match:
        return None
    return GRADE_VALIDATED if match.group(1) == "+" else GRADE_CORRECTED


def format_grade_hint() -> str:
    """The single line that asks for a grade, phrased with a token the plugin actually reads."""

    return "Cortex grade: was this right? `cortex +` / `cortex -` changes how these memories rank next time."


_RECEIPT_IRRELEVANT_FEEDBACK = re.compile(
    r"\b(?:not relevant|irrelevant|wrong context|didn(?:'t| not) apply|doesn(?:'t| not) apply)\b",
    re.I,
)


_RECEIPT_WRONG_FEEDBACK = re.compile(
    r"\b(?:wrong|outdated|incorrect|not right|false)\b",
    re.I,
)


_AMBIGUOUS_FEEDBACK = re.compile(
    r"\?|\b(?:wish|at first|initially|temporarily|maybe|might|unsure|uncertain)\b|"
    r"\b(?:but|however|though|although)\b.{0,80}\b(?:not|didn(?:'t| not)|doesn(?:'t| not)|"
    r"fail(?:s|ed)?|broken|worse|problem|issue)\b",
    re.I,
)


_GREETING_ONLY = re.compile(r"^\s*(?:hi|hey|hello|thanks|thank you|good morning|good night)[.! ]*\s*$", re.I)


def _resolve_allowed_prefixes(prefixes: List[str], allowed_ids: List[str]) -> list[str]:
    """Resolve only unique prefixes from the bounded current/previous turn set."""

    resolved: list[str] = []
    for prefix in prefixes:
        matches = [
            memory_id
            for memory_id in allowed_ids
            if str(memory_id).casefold().startswith(str(prefix).casefold())
        ]
        if len(matches) == 1 and matches[0] not in resolved:
            resolved.append(matches[0])
    return resolved


def _receipt_feedback_outcome(
    text: str,
    referenced_ids: List[str],
) -> tuple[str, list[str]] | None:
    """Recognize an unambiguous one-memory negative receipt response."""

    if len(referenced_ids) != 1 or _AMBIGUOUS_FEEDBACK.search(text or ""):
        return None
    if _RECEIPT_IRRELEVANT_FEEDBACK.search(text or ""):
        return "irrelevant", referenced_ids
    if _RECEIPT_WRONG_FEEDBACK.search(text or ""):
        return "wrong", referenced_ids
    return None
