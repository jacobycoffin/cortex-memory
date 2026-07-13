"""Dependency-free semantic features for Cortex's transparent hybrid index.

This is not a neural embedding model.  It combines exact terms, lightweight
stems, concept aliases, adjacent pairs, and character fragments so FTS5 keeps
exact precision while the secondary index catches modest paraphrase and typo
variation without a network call or model dependency.
"""

from __future__ import annotations

import re
from collections import defaultdict


_TOKEN = re.compile(r"[\w~./:@+-]{2,}", re.UNICODE)
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "did", "do", "does", "for",
    "from", "had", "has", "have", "how", "i", "if", "in", "is", "it", "me", "my", "of", "on", "or",
    "our", "so", "that", "the", "their", "them", "there", "they", "this", "to", "us", "was", "we",
    "were", "what", "when", "where", "which", "who", "why", "will", "with", "would", "you", "your",
}
_CONCEPT_GROUPS: tuple[tuple[str, frozenset[str]], ...] = (
    ("host", frozenset({"host", "server", "vps", "machine", "computer", "backend", "node"})),
    ("location", frozenset({"where", "location", "path", "directory", "folder", "stored", "runs"})),
    ("preference", frozenset({"prefer", "preference", "favorite", "style", "usually", "default"})),
    ("decision", frozenset({"decision", "decide", "chosen", "chose", "choice", "approved", "picked"})),
    ("failure", frozenset({"error", "failed", "failure", "broken", "issue", "problem", "exception"})),
    ("repair", frozenset({"fix", "fixed", "repair", "resolved", "solution", "workaround", "recovery"})),
    ("tool", frozenset({"tool", "command", "action", "function", "call", "workflow", "procedure"})),
    ("research", frozenset({"search", "research", "source", "web", "online", "browse", "paper"})),
    ("deployment", frozenset({"deploy", "deployment", "release", "publish", "production", "install"})),
    ("current", frozenset({"current", "currently", "now", "latest", "today", "active", "present"})),
    ("historical", frozenset({"previous", "previously", "former", "formerly", "history", "historical", "old"})),
    ("identity", frozenset({"identity", "profile", "person", "name", "user"})),
    ("communication", frozenset({"response", "respond", "answer", "message", "email", "chat"})),
)


def semantic_features(text: str, *, max_features: int = 160) -> dict[str, float]:
    raw_tokens = [token.casefold().strip(".,;!?()[]{}\"'") for token in _TOKEN.findall(text or "")]
    tokens = [token for token in raw_tokens if token and token not in _STOP][:48]
    weighted: dict[str, float] = defaultdict(float)
    for token in tokens:
        weighted[f"tok:{token}"] = max(weighted[f"tok:{token}"], 1.0)
        stem = _stem(token)
        if stem != token and len(stem) >= 3:
            weighted[f"stem:{stem}"] = max(weighted[f"stem:{stem}"], 0.72)
        if len(token) >= 6 and token.isalpha():
            for trigram in _trigrams(token)[:10]:
                weighted[f"tri:{trigram}"] = max(weighted[f"tri:{trigram}"], 0.10)
    for left, right in zip(tokens, tokens[1:]):
        weighted[f"pair:{_stem(left)}:{_stem(right)}"] = 0.58
    token_set = set(tokens) | {_stem(token) for token in tokens}
    for concept, aliases in _CONCEPT_GROUPS:
        if token_set & aliases:
            weighted[f"concept:{concept}"] = 0.88
    ranked = sorted(weighted.items(), key=lambda item: (-item[1], item[0]))[:max_features]
    return dict(ranked)


def feature_similarity(left: str, right: str) -> float:
    a = semantic_features(left)
    b = semantic_features(right)
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    numerator = sum(min(a[key], b[key]) for key in shared)
    denominator = max(1.0, min(sum(a.values()), sum(b.values())))
    return min(1.0, numerator / denominator)


def _stem(token: str) -> str:
    if not token.isalpha() or len(token) < 5:
        return token
    for suffix in ("ingly", "edly", "ation", "ments", "ment", "ness", "ing", "ers", "ies", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            base = token[: -len(suffix)]
            return base + "y" if suffix == "ies" else base
    return token


def _trigrams(token: str) -> list[str]:
    padded = f"^{token}$"
    return list(dict.fromkeys(padded[index : index + 3] for index in range(len(padded) - 2)))
