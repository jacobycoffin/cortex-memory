"""Cross-domain transfer (#113): notice that a pattern from one project predicts another, and propose it.

`cross_project.py` (#78) bridges a *query* to another project's memories while answering. This is the other half:
at consolidation time, notice that two projects are shaped alike and that one of them holds a pattern the other
lacks, then **propose** the transfer.

What a pattern is here: a structured memory's `(predicate, object_value)` pair, counted per project. Free text has no
pattern, and pretending it does is how this becomes a similarity search with a story attached - so rows without a
predicate and an object are skipped and counted as skipped.

Six rules:

1. **A transfer is a proposal, never a write.** The pattern was established in another project's context; asserting it
   here is a decision a person makes, not something a nightly job does on its own.
2. **The likening is stated.** Every proposal names the shared patterns that justified it and the likeness score. A
   proposal whose evidence is invisible is a suggestion nobody can audit.
3. **A pattern the target already holds is not proposed** - that is duplication, not transfer.
4. **Evidence on both sides**: the source memories that support the pattern, and the shared patterns that made the two
   projects look alike.
5. **Thresholds are named in the output**, not buried in constants, so a quiet result can be read as "nothing crossed
   the bar" instead of "the feature is broken".
6. **Proposals are deduplicated per (source, target, pattern)**, so a nightly run does not pile the same suggestion up
   twenty times.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from .security import normalize_text
from .store import utc_now

# A pair of projects is worth comparing only when they share real structure. Three shared patterns and half of the
# union keeps "two projects that both use pytest" from counting as a transfer.
MIN_SHARED = 3
MIN_LIKENESS = 0.50

# How many memories to read per plan. Bounded because the pattern table is built in memory from the store's rows.
SCAN_ROWS = 5000


@dataclass(frozen=True)
class Proposal:
    source_project: str
    target_project: str
    predicate: str
    object_value: str
    likeness: float
    shared: tuple[str, ...]
    support_ids: tuple[str, ...]

    @property
    def pattern(self) -> str:
        return f"{self.predicate} -> {self.object_value}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_project": self.source_project,
            "target_project": self.target_project,
            "predicate": self.predicate,
            "object_value": self.object_value,
            "pattern": self.pattern,
            "likeness": round(self.likeness, 3),
            "shared": list(self.shared),
            "support_ids": list(self.support_ids),
        }

    def render(self) -> str:
        return (f"{self.source_project} -> {self.target_project}: {self.pattern} "
                f"(likeness {self.likeness:.2f} on {len(self.shared)} shared pattern(s))")


@dataclass
class Plan:
    proposals: list[Proposal] = field(default_factory=list)
    projects: int = 0
    compared: int = 0
    readable: bool = True
    reason: str = ""
    skips: dict[str, int] = field(default_factory=dict)

    def render(self) -> list[str]:
        if not self.readable:
            return [f"the store could not be read: {self.reason}"]
        lines = [f"{self.projects} project(s), {self.compared} pair(s) compared "
                 f"(min {MIN_SHARED} shared patterns, likeness >= {MIN_LIKENESS:.2f}): "
                 f"{len(self.proposals)} proposal(s)"]
        for proposal in self.proposals[:10]:
            lines.append(f"  - {proposal.render()}")
        if len(self.proposals) > 10:
            lines.append(f"  … and {len(self.proposals) - 10} more")
        if self.skips:
            lines.append("  skipped: " + ", ".join(f"{k} {v}" for k, v in sorted(self.skips.items())))
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {
            "projects": self.projects,
            "compared": self.compared,
            "readable": self.readable,
            "reason": self.reason,
            "proposals": [proposal.as_dict() for proposal in self.proposals],
            "skips": dict(self.skips),
            "thresholds": {"min_shared": MIN_SHARED, "min_likeness": MIN_LIKENESS},
        }


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]] | None:
    try:
        cursor = store._conn.execute(sql, params)
    except Exception:
        return None
    try:
        return [dict(row) for row in cursor.fetchall()]
    except Exception:
        return None


def _project_of(row: dict[str, Any]) -> str:
    """The project a memory belongs to, from its scope. Unscoped memories belong to no project and are skipped."""
    raw = row.get("scope_json")
    try:
        scope = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:
        return ""
    if not isinstance(scope, dict):
        return ""
    return str(scope.get("project") or "").strip()


def _pattern(row: dict[str, Any]) -> tuple[str, str] | None:
    predicate = normalize_text(str(row.get("predicate") or "")).strip()
    object_value = normalize_text(str(row.get("object_value") or "")).strip()
    if not predicate or not object_value:
        return None
    return predicate, object_value


def pattern_sets(store: Any, *, scan: int = SCAN_ROWS) -> tuple[dict[str, dict[tuple[str, str], list[str]]], dict[str, int]]:
    """`{project: {pattern: [memory ids]}}` from structured, active memories."""
    rows = _rows(
        store,
        "SELECT id,scope_json,predicate,object_value FROM memories "
        "WHERE state IN ('active','cold') ORDER BY used_count DESC LIMIT ?",
        (max(1, int(scan)),),
    )
    skips: dict[str, int] = {}
    if rows is None:
        return {}, {"unreadable": 1}
    patterns: dict[str, dict[tuple[str, str], list[str]]] = {}
    for row in rows:
        project = _project_of(row)
        if not project:
            skips["no project scope"] = skips.get("no project scope", 0) + 1
            continue
        pattern = _pattern(row)
        if pattern is None:
            skips["not structured (no predicate/object)"] = skips.get("not structured (no predicate/object)", 0) + 1
            continue
        patterns.setdefault(project, {}).setdefault(pattern, []).append(str(row["id"]))
    return patterns, skips


def likeness(left: set[tuple[str, str]], right: set[tuple[str, str]]) -> float:
    """Jaccard over pattern sets. Deliberately dull: a transfer that needs a clever scorer is not auditable."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def plan(store: Any, *, min_shared: int = MIN_SHARED, min_likeness: float = MIN_LIKENESS,
         limit: int = 20, scan: int = SCAN_ROWS) -> Plan:
    """Propose patterns one project holds that a look-alike project does not. Writes nothing."""
    patterns, skips = pattern_sets(store, scan=scan)
    if not patterns:
        return Plan(projects=0, readable="unreadable" not in skips, reason="no structured project memories",
                    skips=skips)
    projects = sorted(patterns)
    proposals: list[Proposal] = []
    compared = 0
    for index, source in enumerate(projects):
        for target in projects[index + 1:]:
            compared += 1
            source_patterns, target_patterns = patterns[source], patterns[target]
            shared = set(source_patterns) & set(target_patterns)
            if len(shared) < max(1, int(min_shared)):
                continue
            score = likeness(set(source_patterns), set(target_patterns))
            if score < min_likeness:
                continue
            # Each direction separately: what source has and target lacks, then the reverse. A pattern the target
            # already holds is not a transfer, it is duplication.
            for origin, destination in ((source, target), (target, source)):
                held, missing = patterns[origin], patterns[destination]
                for pattern, support in sorted(held.items()):
                    if pattern in missing:
                        continue
                    proposals.append(Proposal(
                        source_project=origin,
                        target_project=destination,
                        predicate=pattern[0],
                        object_value=pattern[1],
                        likeness=score,
                        shared=tuple(f"{p[0]} -> {p[1]}" for p in sorted(shared))[:6],
                        support_ids=tuple(sorted(support))[:5],
                    ))
    proposals.sort(key=lambda item: (-item.likeness, item.source_project, item.target_project, item.pattern))
    return Plan(projects=len(projects), compared=compared, proposals=proposals[:max(1, int(limit))],
                skips=skips)


def propose(store: Any, plan_result: Plan, *, run_id: str, apply: bool = False) -> dict[str, Any]:
    """Persist the proposals against a sleep run. Dry run unless `apply` is given."""
    if not plan_result.readable:
        return {"ok": False, "reason": plan_result.reason, "written": 0, "dry_run": not apply}
    if not apply:
        return {"ok": True, "dry_run": True, "would_write": len(plan_result.proposals), "written": 0}
    if not str(run_id or "").strip():
        return {"ok": False, "reason": "a proposal belongs to a sleep run; no run_id given", "written": 0,
                "dry_run": False}
    written = 0
    skipped = 0
    for proposal in plan_result.proposals:
        details = json.dumps(proposal.as_dict())
        try:
            with store.transaction() as conn:
                # Deduplicated on the same run: the same suggestion twenty times is noise, not twenty findings.
                existing = conn.execute(
                    "SELECT 1 FROM sleep_proposals WHERE run_id=? AND kind='cross_domain_transfer' "
                    "AND details_json=? LIMIT 1",
                    (str(run_id), details),
                ).fetchone()
                if existing:
                    skipped += 1
                    continue
                conn.execute(
                    "INSERT INTO sleep_proposals(proposal_id,run_id,kind,src_id,dst_id,status,score,"
                    "evidence_count,rationale,details_json,created_at) "
                    "VALUES(?,?,'cross_domain_transfer',NULL,NULL,'proposed',?,?,?,?,?)",
                    (
                        str(uuid.uuid4()), str(run_id), float(proposal.likeness),
                        len(proposal.support_ids),
                        (f"{proposal.source_project} and {proposal.target_project} share "
                         f"{len(proposal.shared)} pattern(s); {proposal.source_project} also holds "
                         f"{proposal.pattern}, which {proposal.target_project} does not"),
                        details, utc_now(),
                    ),
                )
        except Exception:
            return {"ok": False, "reason": "the store refused the proposal", "written": written, "dry_run": False}
        written += 1
    return {"ok": True, "dry_run": False, "written": written, "skipped_duplicates": skipped}
