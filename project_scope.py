"""Write the project into memory scope at capture time (backlog item #217).

`project_detect` (#19) can work out which project a *turn* is about, but it had nothing to
choose between: **0 of 3,829** active memories carried `scope.active_project`, so there were
no store-side candidates and no project-scoped memory could ever exist. Detection was
searching a set it was itself responsible for filling.

This fills it, at capture time, while the evidence is still at hand — the content being
stored, the paths it names, and the turn that produced it. Two rules, both learned from
measurement:

- **No project beats a wrong project.** A memory with the wrong project is actively harmful:
  it only becomes applicable in the wrong turns and never in the right ones.
- **Only what the content itself shows.** A capture is not the place to guess from vibes; the
  evidence is a path or a name written into the memory, not a feeling about the session.

The same inference runs as a backfill (`scripts/project_backfill.py`), because a capture-time
fix only helps memories written after it ships.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .project_detect import (
    candidate_projects,
    detect,
    known_projects,
    normalise,
)

# Paths inside stored content: `Path: plans/Temple.md`, `homelab/Docker.md`,
# `Temple/Temple/Duo/Tabletop/...swift`. The stem is what names a project.
_PATH = re.compile(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,6}")
_BARE_PATH = re.compile(r"(?:[A-Za-z0-9_.-]+/){1,}[A-Za-z0-9_.-]+")


def paths_in(content: str) -> list[str]:
    """Every path-looking token in a memory body, longest first.

    Vault memories carry their origin in the text (`Path: homelab/Docker.md`), which is the
    single strongest project signal available at capture time — stronger than the words
    around it.
    """
    text = str(content or "")
    found = {m.group(0) for m in _PATH.finditer(text)}
    if not found:
        found = {m.group(0) for m in _BARE_PATH.finditer(text)}
    return sorted(found, key=len, reverse=True)


@dataclass(frozen=True)
class ProjectEvidence:
    project: str | None
    scores: dict[str, float]
    paths: tuple[str, ...]

    @property
    def confident(self) -> bool:
        return self.project is not None


def infer_project(
    store: Any,
    content: str,
    *,
    query: str = "",
    cwd: str | None = None,
    messages: Sequence[str] = (),
    configured: Iterable[str] = (),
    extra_candidates: Iterable[str] = (),
) -> ProjectEvidence:
    """Best single project for this capture, or none when the evidence is split."""
    paths = paths_in(content)
    candidates = set(candidate_projects(store, configured))
    candidates |= {normalise(name) for name in extra_candidates if normalise(name)}
    # Deliberately NOT every path segment: the first version did that and "inferred" the
    # projects `references`, `research`, `plans` and `docker-compose` from vault folders.
    # A path is *evidence about* a candidate, never a candidate by itself — the project list
    # has to come from configuration or from scopes already written.
    project, scores = detect(
        query=query,
        cwd=cwd,
        files=paths,
        messages=messages,
        candidates=candidates,
    )
    return ProjectEvidence(project=project, scores=scores, paths=tuple(paths))


def _mentions(text: str, project: str) -> int:
    """How many times a project (or its distinctive alias) is named in the text."""
    from .project_detect import alias

    haystack = str(text or "").lower()
    total = 0
    for name in {normalise(project), alias(project)}:
        if not name:
            continue
        total += len(
            re.findall(rf"(?<![a-z0-9-]){re.escape(name)}(?![a-z0-9-])", haystack)
        )
    return total


def strict_project(
    store: Any,
    content: str,
    *,
    configured: Iterable[str] = (),
) -> str | None:
    """A backfill-strength judgement: only place a memory when the evidence is unambiguous.

    Capture has the turn, the cwd and the session; a backfill has only the body, so it must
    be stricter. The first pass counted any mention, and duly filed a note about a Minecraft
    bot under `temple-ios` because the bot is mentioned in a Temple plan. Two rules keep the
    false positives out:

    - a **path** naming the project is enough (it is where the note lives);
    - prose needs the project named at least twice **and** twice as often as the runner-up,
      which is the difference between "this note is about X" and "this note mentions X".
    """
    candidates = {
        normalise(name) for name in [*configured, *known_projects(store)] if normalise(name)
    }
    if not candidates:
        return None
    paths = [p.lower() for p in paths_in(content)]
    by_path = [name for name in candidates if any(_path_names(path, name) for path in paths)]
    if len(by_path) == 1:
        return by_path[0]
    # Prose alone is not enough for a backfill, and this is measured, not assumed: counting
    # mentions filed a Portfolio Website note under `cortex-memory` because the note discusses
    # cortex twice. At capture time the turn says what the turn is about; a stored body
    # mentions many things. So: a path, or nothing.
    return None


def _path_names(path: str, project: str) -> bool:
    from .project_detect import alias

    for name in {normalise(project), alias(project)}:
        if name and re.search(rf"(?<![a-z0-9-]){re.escape(name)}(?![a-z0-9-])", path):
            return True
    return False


def with_project(scope: dict[str, Any] | None, project: str | None) -> dict[str, Any]:
    """Scope with the project recorded, never overriding one the caller set."""
    merged = dict(scope or {})
    if project and not str(merged.get("active_project") or "").strip():
        merged["active_project"] = project
    return merged


def scope_for_capture(
    store: Any,
    content: str,
    *,
    scope: dict[str, Any] | None = None,
    query: str = "",
    cwd: str | None = None,
    messages: Sequence[str] = (),
    configured: Iterable[str] = (),
) -> tuple[dict[str, Any], ProjectEvidence]:
    """The scope to store a new memory with, plus the evidence that decided it."""
    evidence = infer_project(
        store,
        content,
        query=query,
        cwd=cwd,
        messages=messages,
        configured=configured,
    )
    return with_project(scope, evidence.project), evidence


def backfill_plan(
    store: Any,
    *,
    configured: Iterable[str] = (),
    limit: int = 0,
) -> list[tuple[str, str, str]]:
    """(memory_id, project, evidence) for memories that have no project yet.

    Read-only. Writing is a separate, explicit step — a wrong project on an old memory is
    not repairable by later recall.
    """
    rows = [
        (str(row[0]), str(row[1]))
        for row in store._conn.execute(
            "SELECT id, content FROM memories WHERE state = 'active' "
            "AND (scope_json IS NULL OR json_extract(scope_json, '$.active_project') IS NULL) "
            "ORDER BY id"
        ).fetchall()
    ]
    if limit:
        rows = rows[:limit]
    plan: list[tuple[str, str, str]] = []
    for memory_id, content in rows:
        project = strict_project(store, content, configured=configured)
        if project:
            paths = paths_in(content)
            why = paths[0] if paths else "prose"
            plan.append((memory_id, project, why))
    return plan


def apply_backfill(store: Any, plan: Iterable[tuple[str, str, str]]) -> int:
    """Write the planned projects. Returns how many memories changed."""
    changed = 0
    # The caller owns the transaction: `store.transaction()` is a context manager, not a
    # bare iterator, and a write path must never commit on its own (that burned twice).
    with store.transaction() as conn:
        for memory_id, project, _why in plan:
            cursor = conn.execute(
                """UPDATE memories
                   SET scope_json = json_set(
                         CASE WHEN scope_json IS NULL OR NOT json_valid(scope_json)
                              THEN '{}' ELSE scope_json END,
                         '$.active_project', ?)
                   WHERE id = ?
                     AND (scope_json IS NULL
                          OR json_extract(scope_json, '$.active_project') IS NULL)""",
                (project, memory_id),
            )
            changed += cursor.rowcount or 0
    return changed


def project_coverage(store: Any) -> dict[str, int]:
    """How many active memories carry a project — the number #217 exists to move."""
    row = store._conn.execute(
        """SELECT COUNT(*) AS total,
                  SUM(CASE WHEN scope_json IS NOT NULL
                            AND json_extract(scope_json, '$.active_project') IS NOT NULL
                           THEN 1 ELSE 0 END) AS scoped
           FROM memories WHERE state = 'active'"""
    ).fetchone()
    total = int(row["total"] or 0)
    scoped = int(row["scoped"] or 0)
    return {"total": total, "scoped": scoped, "missing": max(0, total - scoped)}


def known(store: Any) -> list[str]:
    """Projects the store now knows about (used by tests and reports)."""
    return known_projects(store)
