"""Deliberately surface a memory from another project when the match is strong (#78).

Namespacing is the right default and a bad absolute. Recall restricted to the active project stops
`duelist-road` answers leaking into `cortex` questions — and also hides the fact that the nginx
timeout was already diagnosed in a *different* project three months ago. The user then solves it
twice.

So a bridge is allowed, but it is **deliberate on every axis**:

- **a high bar** (`MIN_SIMILARITY = 0.55`) — near-miss vocabulary is not a bridge;
- **a hard cap** (`MAX_BRIDGES = 2` per query) — a bridge is a footnote, not a second result set;
- **always labelled** with the project it came from, so nobody mistakes it for the current context's
  answer;
- **never counted as a hit** for the active project — a bridge is extra, not a replacement.

The similarity here is token Jaccard with stopwords removed, computed on the fly. It is deliberately
dull: a bridge that needs a clever scorer to justify itself is a bridge nobody can audit.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

MIN_SIMILARITY = 0.60
MIN_OVERLAP = 3
MAX_BRIDGES = 2
_ENABLED_ENV = "CORTEX_CROSS_PROJECT"

_STOP = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in
    into is it its just like may me more most my no not of on or our out over so some than that the
    their them then there these they this to too up us was we were what when where which who why
    will with would you your""".split()
)

_TOKEN = re.compile(r"[a-z0-9][a-z0-9_./:-]{1,}")


def enabled() -> bool:
    """Off by default; the measurement decides, not the enthusiasm."""
    return os.environ.get(_ENABLED_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def tokens(text: str) -> set[str]:
    return {
        token for token in _TOKEN.findall(str(text or "").casefold())
        if token not in _STOP
    }


def similarity(left: str, right: str) -> float:
    """Containment on the shorter side, floored by an absolute overlap.

    Jaccard punishes a long question matched against a short memory, which is exactly the shape of a
    real bridge ("the opencode-go lane needs a User-Agent" against "opencode-go requires a real
    User-Agent"). Containment asks the question that matters — is the memory *about* the query —
    while `MIN_OVERLAP` stops three coincidental words from carrying a whole memory.
    """
    left_tokens, right_tokens = tokens(left), tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    shared = left_tokens & right_tokens
    if len(shared) < MIN_OVERLAP:
        return 0.0
    return len(shared) / min(len(left_tokens), len(right_tokens))


def project_of(row: dict[str, Any] | None) -> str:
    """The namespace a memory lives in, or an empty string when it was captured without one."""
    if not row:
        return ""
    raw = row.get("scope_json")
    if not raw:
        return ""
    try:
        scope = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        return ""
    value = scope.get("active_project") or scope.get("project") or ""
    return str(value).strip()


@dataclass
class Bridge:
    memory_id: str
    content: str
    project: str
    score: float

    def describe(self) -> str:
        return f"[{self.project or 'unscoped'}] {self.content[:100]} ({self.score:.2f})"

    def render(self) -> str:
        return f"Also relevant, from project {self.project or 'unscoped'}: {self.content[:200]}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "project": self.project,
            "score": round(self.score, 3),
        }


def _candidates(store: Any, query_tokens: set[str], scan: int) -> list[dict[str, Any]]:
    """A pool worth scoring: the longest query words, then recent memories as a floor.

    Picking one arbitrary token (say the alphabetically first) made bridges depend on which word it
    happened to be — the longest words are the informative ones, and a recent-memory floor keeps a
    bridge findable when every distinctive word is rare.
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    longest = sorted(query_tokens, key=lambda token: (-len(token), token))[:3]
    for token in longest:
        try:
            found = store._conn.execute(
                """SELECT id, content, scope_json FROM memories
                   WHERE state NOT IN ('tombstoned','deleted') AND content LIKE ?
                   ORDER BY created_at DESC LIMIT ?""",
                (f"%{token}%", scan),
            ).fetchall()
        except Exception:  # noqa: BLE001 - no table, no bridges
            return rows
        for row in found:
            memory_id = str(row["id"])
            if memory_id in seen:
                continue
            seen.add(memory_id)
            rows.append(dict(row))
        if len(rows) >= scan:
            break
    if len(rows) < scan // 8:
        try:
            for row in store._conn.execute(
                """SELECT id, content, scope_json FROM memories
                   WHERE state NOT IN ('tombstoned','deleted')
                   ORDER BY created_at DESC LIMIT ?""",
                (scan,),
            ).fetchall():
                memory_id = str(row["id"])
                if memory_id not in seen:
                    seen.add(memory_id)
                    rows.append(dict(row))
        except Exception:  # noqa: BLE001
            pass
    return rows


def find(
    store: Any,
    query: str,
    *,
    active_project: str | None = None,
    limit: int = MAX_BRIDGES,
    minimum: float = MIN_SIMILARITY,
    scan: int = 400,
) -> list[Bridge]:
    """Memories from *another* project that match this query strongly."""
    if not str(query or "").strip():
        return []
    home = str(active_project or "").strip()
    query_tokens = tokens(query)
    if not query_tokens:
        return []
    rows = _candidates(store, query_tokens, int(scan))

    found: list[Bridge] = []
    seen: set[str] = set()
    for row in rows:
        project = project_of(row)
        if project and home and project == home:
            continue
        if project and not home:
            pass  # an unscoped question may hear from any project, labelled
        score = similarity(query, str(row.get("content") or ""))
        if score < minimum:
            continue
        memory_id = str(row["id"])
        if memory_id in seen:
            continue
        seen.add(memory_id)
        found.append(Bridge(memory_id, str(row.get("content") or ""), project, score))
    found.sort(key=lambda item: (-item.score, item.memory_id))
    return found[: max(0, int(limit))]


def render(bridges: list[Bridge]) -> str:
    if not bridges:
        return ""
    return "\n".join(bridge.render() for bridge in bridges)


def report(store: Any, *, active_project: str | None = None) -> dict[str, Any]:
    """How many memories carry a project at all, and how the namespaces divide."""
    counts: dict[str, int] = {}
    unscoped = 0
    try:
        rows = store._conn.execute(
            """SELECT scope_json FROM memories
               WHERE state NOT IN ('tombstoned','deleted')"""
        ).fetchall()
    except Exception:  # noqa: BLE001
        rows = []
    for row in rows:
        project = project_of({"scope_json": row["scope_json"]})
        if not project:
            unscoped += 1
        else:
            counts[project] = counts.get(project, 0) + 1
    return {
        "memories": len(rows),
        "unscoped": unscoped,
        "scoped": sum(counts.values()),
        "projects": dict(sorted(counts.items(), key=lambda item: -item[1])[:10]),
        "min_similarity": MIN_SIMILARITY,
        "max_bridges": MAX_BRIDGES,
        "enabled": enabled(),
        "active_project": active_project or "",
    }
