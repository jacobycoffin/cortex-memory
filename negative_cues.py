"""Suppress what the query explicitly ruled out (backlog item #7).

Queries carry exclusions — "not the maintenance window one", "anything but last week's
plan", "other than the truck". Recall reads the words and ignores the negation, then
confidently returns the exact thing that was excluded. That is worse than returning
nothing: it looks like the agent was not listening.

This extracts the excluded term and demotes matches. Demotion, not deletion: a parser that
occasionally misreads English must not be able to make a memory unreachable, so a matched
memory is scaled down hard (0.2x) and labelled, never dropped. If every candidate matches,
the least-demoted one still wins.

Deliberately conservative. Only patterns with an explicit exclusion word are parsed, and
"not" inside a word ("note", "nothing") never counts.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Iterable

PENALTY = 0.2
MAX_TERMS = 4

# "not the X", "other than X", "except X", "besides X", "instead of X", "anything but X",
# "without X", "don't include X", "excluding X"
_PATTERNS = [
    r"\bnot\s+(?:the\s+)?(?P<term>[a-z0-9][\w' -]{1,40}?)(?=[,.?!;:)\]]|\s+(?:one|ones|version|entry|memory|item)\b|$)",
    r"\bother\s+than\s+(?:the\s+)?(?P<term>[a-z0-9][\w' -]{1,40}?)(?=[,.?!;:)\]]|$)",
    r"\bexcept(?:\s+for)?\s+(?:the\s+)?(?P<term>[a-z0-9][\w' -]{1,40}?)(?=[,.?!;:)\]]|$)",
    r"\bbesides\s+(?:the\s+)?(?P<term>[a-z0-9][\w' -]{1,40}?)(?=[,.?!;:)\]]|$)",
    r"\binstead\s+of\s+(?:the\s+)?(?P<term>[a-z0-9][\w' -]{1,40}?)(?=[,.?!;:)\]]|$)",
    r"\banything\s+but\s+(?:the\s+)?(?P<term>[a-z0-9][\w' -]{1,40}?)(?=[,.?!;:)\]]|$)",
    r"\bwithout\s+(?:the\s+)?(?P<term>[a-z0-9][\w' -]{1,40}?)(?=[,.?!;:)\]]|$)",
    r"\bexclud\w*\s+(?:the\s+)?(?P<term>[a-z0-9][\w' -]{1,40}?)(?=[,.?!;:)\]]|$)",
    r"\bdon'?t\s+(?:include|mention|show|use)\s+(?:the\s+)?(?P<term>[a-z0-9][\w' -]{1,40}?)(?=[,.?!;:)\]]|$)",
]

_TRAILING_STOP = {"one", "ones", "version", "entry", "memory", "item", "thing", "please", "thanks"}
_WORD = re.compile(r"[a-z0-9][\w'-]*")


def excluded_terms(query: str) -> list[str]:
    """Terms the query says must not come back. Empty when there is no exclusion."""
    text = (query or "").lower()
    if not text:
        return []
    found: list[str] = []
    for pattern in _PATTERNS:
        for match in re.finditer(pattern, text):
            term = " ".join(match.group("term").split()).strip().strip("\"'()[]{}.,;:!?")
            # Trim trailing filler ("not the maintenance one", "not the truck please")
            words = term.split()
            while words and words[-1] in _TRAILING_STOP:
                words.pop()
            term = " ".join(words)
            if len(term) < 3 or term in found:
                continue
            found.append(term)
            if len(found) >= MAX_TERMS:
                return found
    return found


def _matches(content: str, term: str) -> bool:
    """Whole-word match, so 'not the truck' never suppresses 'trucking'."""
    words = set(_WORD.findall((content or "").lower()))
    return all(part in words for part in _WORD.findall(term))


def suppress(
    results: Iterable[Any], terms: list[str]
) -> tuple[list[Any], int]:
    """Demote results whose content matches an excluded term. Returns (results, count)."""
    results = list(results)
    if not terms:
        return results, 0

    out: list[Any] = []
    suppressed = 0
    for result in results:
        memory = getattr(result, "memory", {}) or {}
        content = str(memory.get("content") or "")
        hit = next((t for t in terms if _matches(content, t)), None)
        if hit is None:
            out.append(result)
            continue
        components = dict(getattr(result, "components", {}) or {})
        components["negative_cue"] = 1.0
        components["negative_cue_score_before"] = round(float(result.score), 6)
        out.append(
            replace(
                result,
                score=float(result.score) * PENALTY,
                components=components,
            )
        )
        suppressed += 1
    return out, suppressed
