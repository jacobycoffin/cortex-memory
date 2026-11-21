"""Rewrite an underspecified query with terms from the session before retrieving (#16).

"do that", "and the second one", "what about the other box" — these carry no retrievable
content. Session-scoped recall (#17) reorders what is already found; it cannot find what the
query never asked for. The missing step is to *rewrite* the query with the topic the session
was already on, then retrieve with that.

Rewriting is deliberately narrow:

- only for elliptical or very short queries (a real question is left alone);
- only terms that recur in the recent session turns, not the whole transcript;
- the original words are always kept, so the rewrite can only add signal, never remove it;
- capped, so a long session cannot turn a two-word follow-up into a paragraph.

The original query is what gets logged; the rewrite is a retrieval aid, not a record.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

MAX_CONTENT_TOKENS = 2
MAX_ADDED_TERMS = 6
MIN_TERM_LEN = 4

_WORD = re.compile(r"[A-Za-z0-9][\w'’-]*")
_STOP = {
    "the", "and", "for", "with", "that", "this", "these", "those", "from", "into", "over",
    "under", "was", "were", "are", "you", "your", "our", "its", "it", "did", "does", "have",
    "has", "had", "will", "would", "could", "should", "can", "just", "then", "than", "there",
    "here", "they", "them", "about", "again", "same", "other", "another", "one", "ones",
    "yeah", "okay", "ok", "sure", "please", "like", "want", "need", "make", "made", "get",
    "got", "use", "using", "also", "next", "now", "what", "which", "when", "where", "why",
    "how", "does", "did", "doing", "done", "still", "more", "most", "very",
}


def content_terms(text: str) -> list[str]:
    return [
        word.lower()
        for word in _WORD.findall(str(text or ""))
        if len(word) >= MIN_TERM_LEN and word.lower() not in _STOP
    ]


def is_underspecified(query: str, *, max_content_tokens: int = MAX_CONTENT_TOKENS) -> bool:
    """True when the query has (almost) no retrievable content of its own."""
    return len(content_terms(query)) <= max_content_tokens


def recent_queries(store: Any, session_id: str | None, *, limit: int = 8) -> list[str]:
    """The last few queries this session asked. Never raises."""
    conn = getattr(store, "_conn", None)
    if conn is None or not session_id:
        return []
    try:
        rows = conn.execute(
            "SELECT query FROM access_log WHERE session_id = ? AND query IS NOT NULL "
            "AND query != '' ORDER BY created_at DESC LIMIT ?",
            (str(session_id), int(limit)),
        ).fetchall()
    except Exception:  # noqa: BLE001 - a retrieval aid must never break a turn
        return []
    return [str(row[0]) for row in rows]


def rewrite(
    query: str,
    recent_turns: Iterable[str],
    *,
    max_added: int = MAX_ADDED_TERMS,
) -> tuple[str, list[str]]:
    """Return (rewritten_query, added_terms). Unchanged when the query stands on its own."""
    original = " ".join(str(query or "").split())
    if not original or not is_underspecified(original):
        return original, []

    existing = {t for t in content_terms(original)}
    counts: dict[str, int] = {}
    order: list[str] = []
    for turn in recent_turns:
        for term in content_terms(turn):
            if term in existing:
                continue
            if term not in counts:
                counts[term] = 0
                order.append(term)
            counts[term] += 1

    # Recurring terms are what the session is actually about; a one-off mention is noise.
    recurring = [term for term in order if counts[term] >= 2]
    additions = (recurring or order)[: max(0, int(max_added))]
    if not additions:
        return original, []
    return f"{original} {' '.join(additions)}", additions
