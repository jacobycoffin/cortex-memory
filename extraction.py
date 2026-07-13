"""Lightweight, deterministic candidate extraction for Cortex v0.1.

The first release deliberately avoids an LLM call on the write path. Raw turns
are retained as episodes, while only sentences with durable-memory signals are
promoted into the active memory graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .security import normalize_text


_SPLIT = re.compile(r"(?<=[.!?])\s+|[\r\n]+|\s*;\s*")
_QUESTION = re.compile(r"\?\s*$")

_PATTERNS: tuple[tuple[str, re.Pattern[str], float, float], ...] = (
    (
        "prospective",
        re.compile(r"\b(?:remind me|need to|plan to|todo|to-do|follow up|don't forget|commit(?:ment)? to)\b", re.I),
        0.90,
        0.84,
    ),
    (
        "preference",
        re.compile(
            r"\b(?:i|we)\s+(?:prefer|like|want|dislike|hate|always|never)\b|\bplease\s+(?:always|never|remember)\b",
            re.I,
        ),
        0.88,
        0.85,
    ),
    (
        "identity",
        re.compile(r"\bmy\s+(?:name|timezone|role|job|location)\s+is\b|\bi\s+(?:am|live|work)\b", re.I),
        0.90,
        0.88,
    ),
    (
        "decision",
        re.compile(r"\b(?:we|i)\s+(?:decided|chose|picked|settled on|am going with)\b|\bthe decision is\b", re.I),
        0.84,
        0.82,
    ),
    (
        "procedure",
        re.compile(r"\b(?:workaround|root cause|the fix|fixed by|run this|command is|verified with)\b", re.I),
        0.76,
        0.72,
    ),
    (
        "operational",
        re.compile(r"\b(?:runs? on|hosted on|configured|version|path is|server|backend|database|provider)\b", re.I),
        0.68,
        0.68,
    ),
    (
        "semantic",
        re.compile(r"\b(?:project|uses|requires|depends on|belongs to|is located|is stored)\b", re.I),
        0.65,
        0.62,
    ),
)

_EXPLICIT = re.compile(r"^\s*(?:please\s+)?remember(?:\s+that)?\s*[:,.-]?\s*", re.I)
_ASSISTANT_SIGNALS = re.compile(r"\b(?:verified|resolved|root cause|the fix|successfully|decision|configured)\b", re.I)


@dataclass(frozen=True)
class Candidate:
    content: str
    kind: str
    importance: float
    confidence: float
    volatility: float


def extract_candidates(text: str, *, role: str = "user") -> list[Candidate]:
    """Extract conservative durable-memory candidates from a turn."""
    text = normalize_text(text)
    if not text:
        return []

    candidates: list[Candidate] = []
    seen: set[str] = set()
    for raw in _SPLIT.split(text):
        sentence = normalize_text(raw)
        if len(sentence) < 16 or len(sentence) > 600:
            continue
        explicit = bool(_EXPLICIT.search(sentence))
        if explicit:
            sentence = normalize_text(_EXPLICIT.sub("", sentence))
        if not sentence or (_QUESTION.search(sentence) and not explicit):
            continue
        if role == "assistant" and not _ASSISTANT_SIGNALS.search(sentence):
            continue

        match = None
        for kind, pattern, importance, confidence in _PATTERNS:
            if pattern.search(sentence):
                match = (kind, importance, confidence)
                break
        if explicit:
            match = match or ("semantic", 0.92, 0.92)
        if not match:
            continue

        kind, importance, confidence = match
        if role == "assistant":
            confidence = min(confidence, 0.58)
        volatility = {
            "identity": 0.05,
            "preference": 0.12,
            "decision": 0.25,
            "procedure": 0.25,
            "operational": 0.75,
            "prospective": 0.20,
            "semantic": 0.35,
        }.get(kind, 0.4)
        key = sentence.casefold()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(Candidate(sentence, kind, importance, confidence, volatility))
    return candidates
