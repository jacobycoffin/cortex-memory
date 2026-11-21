"""Infer which project a turn is about, so scoped memory can be applied (backlog item #19).

`RetrievalContext.active_project` decides whether a project-scoped memory is applicable.
Today it is whatever the caller configured, which in a single shared chat is usually nothing
— so scoped memories either never apply, or apply when they should not.

This detects the project from the evidence actually present: an explicit mention in the
query, the working directory, open file paths, or the last few messages. Evidence is scored,
and a tie returns nothing on purpose — guessing the wrong project is worse than returning no
project, because the wrong project silently admits the wrong memories.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

MENTION_WEIGHT = 2.0
CWD_WEIGHT = 3.0
FILE_WEIGHT = 1.5
MESSAGE_WEIGHT = 0.75

# Separators that differ between a repo name, a directory and prose.
_SEPARATOR = re.compile(r"[^a-z0-9]+")


def normalise(name: str) -> str:
    return _SEPARATOR.sub("-", str(name or "").strip().lower()).strip("-")


def alias(project: str) -> str:
    """The distinctive part of a repo name: `temple-ios` → `temple`, `website` → `website`.

    Repo names carry a qualifier ("-ios", "-memory") that file paths and prose usually drop
    — `Temple/Temple/Duo/...` never contains the string "temple-ios".
    """
    parts = [p for p in normalise(project).split("-") if len(p) >= 4]
    return max(parts, key=len) if parts else normalise(project)


def _contains(text: str, project: str) -> bool:
    """Whole-token match on the normalised forms, so 'road' never matches 'railroad'."""
    haystack = f"-{normalise(text)}-"
    if f"-{normalise(project)}-" in haystack:
        return True
    return f"-{alias(project)}-" in haystack


def detect(
    *,
    query: str = "",
    cwd: str | None = None,
    files: Iterable[str] = (),
    messages: Sequence[str] = (),
    candidates: Iterable[str] = (),
) -> tuple[str | None, dict[str, float]]:
    """Return (project, scores). No project when nothing matches or the top two tie."""
    names = sorted({normalise(c) for c in candidates if normalise(c)})
    if not names:
        return None, {}

    scores: dict[str, float] = {name: 0.0 for name in names}

    for name in names:
        if query and _contains(query, name):
            scores[name] += MENTION_WEIGHT
        if cwd and _contains(cwd, name):
            scores[name] += CWD_WEIGHT
        for path in files:
            if _contains(path, name):
                scores[name] += FILE_WEIGHT
                break
        for message in messages:
            if _contains(message, name):
                scores[name] += MESSAGE_WEIGHT

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    best_name, best_score = ranked[0]
    if best_score <= 0.0:
        return None, scores
    if len(ranked) > 1 and ranked[1][1] == best_score:
        # Ambiguous evidence: refuse rather than pick by alphabet.
        return None, scores
    return best_name, scores


def candidate_projects(store: Any, configured: Iterable[str] = (), *, limit: int = 200) -> list[str]:
    """Project names available to detection: configured ones plus any seen in scopes.

    The store is usually the weaker source — on the live database **none** of 3,829 memories
    carry `scope.active_project` — so a deployment that wants detection has to name its
    projects in config. Measured, not assumed.
    """
    names = {normalise(n) for n in configured if normalise(n)}
    names |= set(known_projects(store, limit=limit))
    return sorted(names)


def known_projects(store: Any, *, limit: int = 200) -> list[str]:
    """Project names the store has already seen in memory scopes. Never raises."""
    conn = getattr(store, "_conn", None)
    if conn is None:
        return []
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT json_extract(scope_json, '$.active_project') AS project
            FROM memories
            WHERE scope_json IS NOT NULL
              AND json_valid(scope_json)
              AND json_extract(scope_json, '$.active_project') IS NOT NULL
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    except Exception:  # noqa: BLE001 - detection must never break a recall
        return []
    return sorted({normalise(row[0]) for row in rows if normalise(row[0])})
