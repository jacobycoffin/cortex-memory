"""Summary regeneration (#109): a summary is stale when its facts changed, not when time passed.

`research.generate_summary_candidates` builds a candidate from an evidence bundle and records a signature over the two
memory ids, plus an excerpt of each source's text. That is enough to create a summary and not enough to maintain one:
if the source is edited afterwards the ids are unchanged, the signature is unchanged, `INSERT OR IGNORE` sees a
duplicate, and the summary keeps asserting the old text forever. Nothing in the store noticed.

This module is the noticing. It compares each candidate's recorded excerpts against the current text of the memories it
cites and reports the ones that drifted, with the reason:

- `source_changed` — the cited memory still exists but its text is different, so the summary quotes something the
  store no longer says.
- `source_retired` — the cited memory is archived, quarantined or tombstoned, so the summary rests on a fact that was
  withdrawn.

Five rules:

1. **Staleness is about facts, not age.** A summary of unchanged facts does not decay; a summary of an edited fact is
   wrong the moment the edit lands, however new it is.
2. **The reason travels with the finding**, and names the sources — "stale" alone tells a reviewer nothing about
   whether to re-summarise or to drop the claim.
3. **Supersede, never delete.** A retired candidate keeps its claims and sources; the record of what the store used to
   believe is evidence.
4. **Regeneration proposes, it never promotes.** A refreshed summary re-enters the same review path as any other
   candidate; nothing here creates recallable memory.
5. **Nothing is written unless asked** — planning writes nothing, and applying needs explicit consent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .security import normalize_text
from .store import utc_now

STALE = "stale"
CURRENT = "current"

CHANGED = "source_changed"
RETIRED = "source_retired"

ACTIVE_STATES = ("active", "cold")


@dataclass(frozen=True)
class Item:
    candidate_id: str
    title: str
    status: str
    reasons: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    retired: tuple[str, ...] = ()
    sources: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "title": self.title,
            "status": self.status,
            "reasons": list(self.reasons),
            "changed": list(self.changed),
            "retired": list(self.retired),
            "sources": self.sources,
        }

    def render(self) -> str:
        if self.status == CURRENT:
            return f"{self.candidate_id[:8]} current ({self.sources} source(s)) — {self.title[:70]}"
        parts = []
        if self.changed:
            parts.append(f"{len(self.changed)} source(s) changed")
        if self.retired:
            parts.append(f"{len(self.retired)} retired")
        return f"{self.candidate_id[:8]} {self.status} ({', '.join(parts) or 'no detail'}) — {self.title[:70]}"


@dataclass
class Plan:
    items: list[Item] = field(default_factory=list)
    considered: int = 0
    readable: bool = True
    reason: str = ""
    skips: dict[str, int] = field(default_factory=dict)

    @property
    def stale(self) -> list[Item]:
        return [item for item in self.items if item.status == STALE]

    @property
    def current(self) -> list[Item]:
        return [item for item in self.items if item.status == CURRENT]

    def render(self) -> list[str]:
        if not self.readable:
            return [f"summary candidates could not be read: {self.reason}"]
        lines = [f"{self.considered} candidate(s) reviewed: {len(self.stale)} stale, {len(self.current)} current"]
        for item in self.items[:10]:
            lines.append(f"  - {item.render()}")
        if len(self.items) > 10:
            lines.append(f"  … and {len(self.items) - 10} more")
        if self.skips:
            lines.append("  skipped: " + ", ".join(f"{k} {v}" for k, v in sorted(self.skips.items())))
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {
            "considered": self.considered,
            "readable": self.readable,
            "reason": self.reason,
            "stale": [item.as_dict() for item in self.stale],
            "current": [item.as_dict() for item in self.current],
            "skips": dict(self.skips),
        }


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]] | None:
    """Read without assuming the tables exist; None means the store could not answer."""
    try:
        cursor = store._conn.execute(sql, params)
    except Exception:
        return None
    try:
        return [dict(row) for row in cursor.fetchall()]
    except Exception:
        return None


def _excerpt(text: Any) -> str:
    return normalize_text(str(text or ""))[:320]


def plan(store: Any, *, limit: int = 50) -> Plan:
    """Compare each candidate's recorded excerpts with what the store says now. Writes nothing."""
    budget = max(1, min(int(limit), 500))
    candidates = _rows(
        store,
        "SELECT candidate_id,title,status,created_at FROM summary_candidates "
        "WHERE status='proposed' ORDER BY created_at DESC LIMIT ?",
        (budget,),
    )
    if candidates is None:
        return Plan(readable=False, reason="no summary_candidate table, or the store refused the query")
    out = Plan(considered=len(candidates))
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        sources = _rows(
            store,
            "SELECT s.memory_id,s.excerpt,m.content,m.state FROM summary_candidate_sources s "
            "LEFT JOIN memories m ON m.id=s.memory_id WHERE s.candidate_id=?",
            (candidate_id,),
        )
        if sources is None:
            out.skips["unreadable sources"] = out.skips.get("unreadable sources", 0) + 1
            continue
        if not sources:
            out.skips["no cited sources"] = out.skips.get("no cited sources", 0) + 1
            continue
        changed: list[str] = []
        retired: list[str] = []
        for source in sources:
            memory_id = str(source.get("memory_id"))
            state = str(source.get("state") or "")
            if state not in ACTIVE_STATES:
                retired.append(memory_id)
            elif _excerpt(source.get("content")) != str(source.get("excerpt") or ""):
                changed.append(memory_id)
        reasons: tuple[str, ...] = ()
        if changed:
            reasons += (CHANGED,)
        if retired:
            reasons += (RETIRED,)
        out.items.append(Item(
            candidate_id=candidate_id,
            title=str(candidate.get("title") or ""),
            status=STALE if reasons else CURRENT,
            reasons=reasons,
            changed=tuple(sorted(changed)),
            retired=tuple(sorted(retired)),
            sources=len(sources),
        ))
    return out


def apply(store: Any, plan_result: Plan, *, apply: bool = False, consented_by: str | None = None) -> dict[str, Any]:
    """Retire the stale candidates. Dry run unless `apply` and a named consenting person are both given."""
    if not plan_result.readable:
        return {"ok": False, "reason": plan_result.reason, "applied": 0, "dry_run": not apply}
    stale = plan_result.stale
    if not apply:
        return {"ok": True, "dry_run": True, "would_supersede": len(stale), "applied": 0,
                "next_step": "call research.generate_summary_candidates to propose replacements"}
    if not consented_by:
        return {"ok": False, "reason": "applying a summary regeneration needs a named consenting person",
                "applied": 0, "dry_run": False}
    actor = normalize_text(str(consented_by))[:80]
    now = utc_now()
    applied = 0
    failures: list[dict[str, Any]] = []
    for item in stale:
        try:
            with store.transaction() as conn:
                cursor = conn.execute(
                    "UPDATE summary_candidates SET status='superseded',reviewed_at=?,reviewed_by=? "
                    "WHERE candidate_id=? AND status='proposed'",
                    (now, actor, item.candidate_id),
                )
                if cursor.rowcount:
                    applied += 1
        except Exception as exc:  # a failed row must not stop the rest
            failures.append({"candidate_id": item.candidate_id, "error": type(exc).__name__})
    return {
        "ok": not failures,
        "dry_run": False,
        "applied": applied,
        "failed": len(failures),
        "failures": failures,
        "next_step": "call research.generate_summary_candidates to propose replacements",
    }


def render(result: dict[str, Any]) -> str:
    return json.dumps(result, sort_keys=True)
