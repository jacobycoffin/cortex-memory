"""Security and normalization helpers for Cortex memory.

Memory is model-facing data, not trusted instructions.  This module keeps the
storage layer dependency-free and applies conservative redaction/quarantine
rules before anything can be recalled into a Hermes prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_WHITESPACE = re.compile(r"\s+")

_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.I | re.S),
        "[REDACTED PRIVATE KEY]",
    ),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED AWS KEY]"),
    (re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{20,}\b"), "[REDACTED API KEY]"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}\b", re.I), "Bearer [REDACTED]"),
    (
        re.compile(
            r"\b(api[_ -]?key|access[_ -]?token|secret|password|passwd)\b\s*[:=]\s*([^\s,;]{8,})",
            re.I,
        ),
        r"\1=[REDACTED]",
    ),
    (
        re.compile(
            r"\b(password|passwd|passcode|pin)\b\s+(?:is|was)\s+"
            r"(?!(?:stored|saved|managed|kept|located)\b)([^\s,;]{3,})",
            re.I,
        ),
        r"\1 is [REDACTED]",
    ),
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b"), "[REDACTED GITHUB TOKEN]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b"), "[REDACTED GITHUB TOKEN]"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "[REDACTED SLACK TOKEN]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "[REDACTED GOOGLE KEY]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"), "[REDACTED JWT]"),
)

_INJECTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions?\b", re.I), "instruction override"),
    (
        re.compile(
            r"\b(?:reveal|print|exfiltrate|send)\b.{0,80}\b(?:system prompt|api key|credentials?|secrets?)\b",
            re.I | re.S,
        ),
        "exfiltration request",
    ),
    (re.compile(r"\byou are now\b.{0,80}\b(?:system|developer|administrator|root)\b", re.I | re.S), "role override"),
    (
        re.compile(
            r"<\s*/?\s*(?:system|developer|assistant)\b|<\s*/\s*tool\b|<\s*tool\s+[^>]*>",
            re.I,
        ),
        "role-tag injection",
    ),
)


@dataclass(frozen=True)
class SanitizedMemory:
    text: str
    quarantine_reason: str | None = None
    redacted: bool = False


def normalize_text(text: str) -> str:
    """Return stable, compact text suitable for hashing and comparison."""
    return _WHITESPACE.sub(" ", (text or "").strip())


def sanitize_memory(text: str) -> SanitizedMemory:
    """Redact secrets and identify content that must not enter prompt recall."""
    cleaned = normalize_text(text)
    redacted = False
    for pattern, replacement in _SECRET_PATTERNS:
        updated, count = pattern.subn(replacement, cleaned)
        if count:
            redacted = True
            cleaned = updated

    reasons = [reason for pattern, reason in _INJECTION_PATTERNS if pattern.search(cleaned)]
    return SanitizedMemory(
        text=cleaned,
        quarantine_reason=", ".join(sorted(set(reasons))) or None,
        redacted=redacted,
    )


def safe_prompt_text(text: str) -> str:
    """Neutralize delimiters that could visually escape the recall envelope."""
    return (
        normalize_text(text)
        .replace("<CORTEX_RECALL", "[CORTEX_RECALL")
        .replace("</CORTEX_RECALL>", "[/CORTEX_RECALL]")
        .replace("```", "'''")
    )
