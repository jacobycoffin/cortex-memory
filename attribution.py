"""Evidence-use attribution for Cortex feedback and utility learning."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .semantics import feature_similarity


_WORD = re.compile(r"[\w~./:@+-]{2,}", re.UNICODE)
_ANCHOR = re.compile(
    r"(?:https?://\S+|(?:~?/)?(?:[\w.-]+/){1,}[\w./-]+|\b\d{2,}(?:\.\d+)?\b|\b[\w]+-[\w-]{3,}\b)",
    re.I,
)
_MEMORY_RECEIPT_ITEM = (
    r"(?:M:[0-9a-f]{8}|"
    r"\[M:[0-9a-f]{8}\]\(https?://[^\s)]+\))"
)
_MEMORY_RECEIPT_LINE = re.compile(
    r"(?im)^[ \t]*Cortex memory:[ \t]*("
    + _MEMORY_RECEIPT_ITEM
    + r"(?:[ \t]*,[ \t]*"
    + _MEMORY_RECEIPT_ITEM
    + r"){0,2})[ \t]*$"
)
_MEMORY_REFERENCE = re.compile(r"\bM:([0-9a-f]{8})\b", re.I)
_STOP = {
    "about", "after", "again", "also", "because", "before", "could", "from", "have", "into", "memory",
    "completed", "done", "should", "task", "that", "the", "their", "there", "these", "they", "this",
    "through", "using", "what", "when", "where", "which", "with", "would", "your",
}


def attribution_score(memory: dict[str, Any], response: str) -> float:
    """Estimate whether an answer used a recalled memory.

    Exact structured values and distinctive anchors dominate.  Conceptual
    similarity can support attribution but cannot independently produce a
    high-confidence success signal.
    """

    content = str(memory.get("content") or "")
    answer = response or ""
    if not content or not answer:
        return 0.0
    answer_folded = answer.casefold()
    object_value = str(memory.get("object_value") or "").strip()
    if object_value and object_value.casefold() in answer_folded:
        return 1.0

    memory_tokens = _tokens(content)
    answer_tokens = _tokens(answer)
    shared_tokens = memory_tokens & answer_tokens
    overlap = len(shared_tokens) / max(1, min(len(memory_tokens), len(answer_tokens)))

    anchors = {anchor.casefold().rstrip(".,;:!?") for anchor in _ANCHOR.findall(content)}
    anchor_hits = sum(anchor in answer_folded for anchor in anchors)
    anchor_score = anchor_hits / len(anchors) if anchors else 0.0

    if len(shared_tokens) < 2 and not anchors:
        overlap *= 0.35

    conceptual = feature_similarity(content, answer)
    score = max(overlap, 0.88 * anchor_score, 0.68 * conceptual)
    if overlap < 0.12 and anchor_score == 0:
        score = min(score, 0.17)
    return max(0.0, min(1.0, score))


def memory_receipt_prefixes(response: str) -> list[str]:
    """Return the bounded IDs from one exact user-visible Cortex receipt."""

    matches = list(_MEMORY_RECEIPT_LINE.finditer(response or ""))
    if len(matches) != 1:
        return []
    return [item.casefold() for item in _MEMORY_REFERENCE.findall(matches[0].group(1))]


def format_memory_receipt(memory_ids: list[str], dashboard_url: str = "") -> str:
    """Return one bounded plain or dashboard-linked receipt line."""

    prefixes: list[str] = []
    for memory_id in memory_ids:
        match = re.match(r"^([0-9a-fA-F]{8})(?:[0-9a-fA-F-]*)$", str(memory_id).strip())
        if match:
            prefix = match.group(1).casefold()
            if prefix not in prefixes:
                prefixes.append(prefix)
        if len(prefixes) == 3:
            break
    base_url = _safe_dashboard_url(dashboard_url)
    items: list[str] = []
    for prefix in prefixes:
        label = f"M:{prefix}"
        if base_url:
            parsed = urlsplit(base_url)
            query = dict(parse_qsl(parsed.query, keep_blank_values=True))
            query["memory"] = prefix
            target = urlunsplit(
                (
                    parsed.scheme,
                    parsed.netloc,
                    parsed.path or "/",
                    urlencode(query),
                    parsed.fragment,
                )
            )
            items.append(f"[{label}]({target})")
        else:
            items.append(label)
    return f"Cortex memory: {', '.join(items)}" if items else ""


def strip_memory_receipt(response: str) -> str:
    """Remove the receipt before semantic attribution, capture, and replay."""

    return _MEMORY_RECEIPT_LINE.sub("", response or "").rstrip()


def referenced_memory_prefixes(text: str) -> list[str]:
    """Extract unique receipt-style memory references from operator feedback."""

    return list(dict.fromkeys(item.casefold() for item in _MEMORY_REFERENCE.findall(text or "")))


def _safe_dashboard_url(value: str) -> str:
    candidate = str(value or "").strip()
    if not candidate:
        return ""
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return ""
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
    ):
        return ""
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, parsed.fragment))


def _tokens(text: str) -> set[str]:
    return {
        token.casefold().strip(".,;:!?()[]{}\"'")
        for token in _WORD.findall(text)
        if len(token) >= 3 and token.casefold() not in _STOP
    }
