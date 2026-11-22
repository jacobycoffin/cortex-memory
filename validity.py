"""A superseded memory is history, not current truth (backlog item #4).

The store records that one memory replaces another (`supersedes_id`, plus a `supersedes`
edge), and then recall ignores it. An outdated fact and its replacement arrive with equal
confidence, so the agent can state the old value as easily as the new one — the exact
failure the validity window was supposed to prevent, on the case the store actually has.

`_currentness` already penalises an explicit `valid_to` in the past. This is the other
half: a memory that some other memory supersedes is demoted at read time.

Demotion, never deletion, for the same reason as negative cues: history is often the answer
("what did we change it from?"), so a superseded memory must remain reachable — just never
win a tie against its replacement.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from typing import Any, Iterable

SUPERSEDED_FACTOR = 0.35


def superseded_ids(store: Any) -> set[str]:
    """Ids that some other memory supersedes. One query, must never raise."""
    conn = getattr(store, "_conn", None)
    if conn is None:
        return set()
    try:
        rows = conn.execute(
            "SELECT DISTINCT supersedes_id FROM memories "
            "WHERE supersedes_id IS NOT NULL AND supersedes_id != ''"
        ).fetchall()
    except Exception:  # noqa: BLE001 - the docstring is a contract: this must never raise
        return set()
    # A row that cannot be read is skipped rather than fatal: this runs on every turn, and a
    # degraded penalty is always better than a failed recall. (CI on Python 3.13 once surfaced
    # an IndexError here from rows the local interpreter never produced.)
    superseded: set[str] = set()
    for row in rows:
        try:
            value = row[0]
        except (IndexError, KeyError, TypeError):
            continue
        if value:
            superseded.add(str(value))
    return superseded


def apply_validity_penalty(
    results: Iterable[Any], superseded: set[str], *, factor: float = SUPERSEDED_FACTOR
) -> tuple[list[Any], int]:
    """Demote results whose memory has been superseded. Returns (results, count)."""
    results = list(results)
    if not superseded:
        return results, 0
    out: list[Any] = []
    demoted = 0
    for result in results:
        memory_id = str((getattr(result, "memory", {}) or {}).get("id") or "")
        if memory_id not in superseded:
            out.append(result)
            continue
        components = dict(getattr(result, "components", {}) or {})
        components["superseded"] = 1.0
        components["superseded_score_before"] = round(float(result.score), 6)
        out.append(
            replace(result, score=float(result.score) * float(factor), components=components)
        )
        demoted += 1
    return out, demoted
