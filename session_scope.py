"""Elliptical follow-ups should resolve against what this session was already about (#17).

"do that", "same for the other one", "and the second one?", "yeah do it" — a follow-up with
no content words of its own. Retrieval treats those as (almost) empty queries: there is
nothing to match, so it either returns noise or nothing, and the turn loses the thread it
was in the middle of.

This resolves an elliptical query against the current session instead. It finds the
memories this session has already touched, and gives them a small boost — small because it
is a tie-breaker, not a claim: if the follow-up genuinely names something, relevance wins,
and the boost only decides between candidates the query cannot separate.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Iterable, Sequence

BOOST = 0.06
RECENT_LIMIT = 20

# Function words that carry no topic of their own.
_FILLER = {
    "a", "an", "the", "and", "or", "but", "do", "does", "did", "can", "could", "would",
    "should", "will", "you", "your", "we", "our", "i", "me", "my", "it", "its", "that",
    "this", "these", "those", "them", "they", "there", "here", "same", "again", "also",
    "too", "now", "then", "next", "one", "ones", "other", "another", "yes", "yeah", "ok",
    "okay", "sure", "please", "go", "ahead", "just", "really", "actually", "instead",
    "what", "how", "why", "when", "where", "which", "about", "for", "with", "on", "in",
    "to", "of", "at", "by", "from", "up", "out", "off", "not", "no", "is", "are", "was",
    "were", "be", "been", "have", "has", "had", "get", "got", "make", "made", "use",
}
_WORD = re.compile(r"[a-z0-9][\w'-]*")
_MAX_CONTENT_TOKENS = 2


def content_tokens(query: str) -> list[str]:
    return [w for w in _WORD.findall(str(query or "").lower()) if w not in _FILLER]


_DEICTIC = {
    "that", "this", "those", "these", "it", "its", "one", "ones", "other", "another",
    "same", "again", "instead", "second", "third", "last", "previous", "former",
}


def is_elliptical(query: str, *, max_content_tokens: int = _MAX_CONTENT_TOKENS) -> bool:
    """True when the query points at something already discussed instead of naming it.

    No content tokens at all ("do that", "go ahead"), or a couple of content tokens that
    only make sense as a pointer ("the other one", "same for CT116"). A short but
    self-contained query ("cortex roadmap") is not elliptical: it names its own topic.
    """
    text = str(query or "").strip()
    if not text:
        return False
    words = set(_WORD.findall(text.lower()))
    tokens = content_tokens(text)
    if not tokens:
        return True
    return len(tokens) <= max_content_tokens and bool(words & _DEICTIC)


def recent_memory_ids(store: Any, session_id: str | None, *, limit: int = RECENT_LIMIT) -> list[str]:
    """Memory ids this session already touched, most recent first. Never raises."""
    if not session_id:
        return []
    conn = getattr(store, "_conn", None)
    if conn is None:
        return []
    try:
        rows = conn.execute(
            """
            SELECT memory_id, MAX(created_at) AS last
            FROM access_log
            WHERE session_id = ? AND memory_id IS NOT NULL
            GROUP BY memory_id
            ORDER BY last DESC
            LIMIT ?
            """,
            (str(session_id), int(limit)),
        ).fetchall()
    except Exception:  # noqa: BLE001 - a diagnostic must never break recall
        return []
    return [str(row[0]) for row in rows]


def apply_session_boost(
    results: Iterable[Any],
    recent_ids: Sequence[str],
    *,
    boost: float = BOOST,
) -> tuple[list[Any], int]:
    """Boost memories already touched in this session. Returns (results, count)."""
    results = list(results)
    if not recent_ids:
        return results, 0
    wanted = set(recent_ids)
    out: list[Any] = []
    boosted = 0
    for position, result in enumerate(results):
        memory_id = str((getattr(result, "memory", {}) or {}).get("id") or "")
        if memory_id not in wanted:
            out.append(result)
            continue
        # Earlier-in-session memories get a little more, so the freshest thread wins ties.
        recency_rank = recent_ids.index(memory_id)
        bonus = boost * (1.0 - recency_rank / max(1, len(recent_ids)))
        components = dict(getattr(result, "components", {}) or {})
        components["session_boost"] = round(bonus, 6)
        out.append(replace(result, score=float(result.score) + bonus, components=components))
        boosted += 1
    return out, boosted
