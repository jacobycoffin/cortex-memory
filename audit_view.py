"""“Why did you say that?” — the audit view (#158)

The store already records the pieces: a trace per turn, every candidate with its rank, score,
components and the pipeline's rejection `reason`, plus the ids that were selected. What was missing is
the view that puts them in the order a person asks the question in:

1. **what was asked** and how retrieval decided to answer it;
2. **what came back**, with the score's *composition* rather than a bare number — a 0.61 that is 0.55
   relevance and 0.06 history is a different claim from a 0.61 that is mostly history;
3. **what was excluded, and why**, grouped by the reason the pipeline actually recorded;
4. **what was never considered** — the difference between "rejected" and "not a candidate" is the
   difference between a ranking problem and a recall problem, and the record can tell them apart.

Four rules, because an audit view is worthless if it flatters:

- **reasons are quoted, never inferred.** A candidate with no recorded reason is reported as
  *unrecorded*; inventing a plausible one is the failure mode this view exists to prevent.
- **a missing trace is a missing trace.** No row means "this turn was not recorded", not "nothing was
  retrieved" — the two are different bugs and are reported differently.
- **the excluded candidates are the interesting half.** They are counted and grouped, not dropped to
  keep the output short.
- **the sentence stays readable.** `render()` produces lines a person can read on a phone; the dicts
  carry the detail for anything that wants to plot it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

# Component names that are history rather than content. Naming them is the point: a score that is
# mostly history is a claim about popularity, not about relevance.
HISTORY_COMPONENTS = ("use_boost", "entity_boost", "session_boost", "association_boost",
                     "context_boost", "popularity", "recency")

# The recorder fills a placeholder when the pipeline gave no reason. A placeholder is not a reason,
# and treating it as one is how an audit view ends up reporting "why" for something nobody explained.
PLACEHOLDER_REASONS = frozenset({"", "unspecified decision", "unknown", "n/a"})


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001 - an audit view must never raise on a store it cannot read
        return []


def _json(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value) if value else []
    except Exception:  # noqa: BLE001
        return []


def _field(trace: dict[str, Any], name: str) -> Any:
    """Read a trace payload under either the decoded name or the stored `_json` column."""
    if name in trace and trace[name] is not None:
        return trace[name] if isinstance(trace[name], (list, dict)) else _json(trace[name])
    return _json(trace.get(f"{name}_json"))


# A gate marks yes/no; a graded component carries how much. They are not interchangeable: a score
# can be entirely explained by gates (every candidate matched, ranking by tie-break) or by graded
# weights, and mixing them in one "top components" list makes a boolean look like a contribution.
FLAG_VALUES = (0, 0.0, 1, -1, 1.0, -1.0)


def split_components(components: dict[str, Any]) -> tuple[list[tuple[str, float]],
                                                           list[tuple[str, float]]]:
    """(graded, flags) — what carried weight, and what merely passed a gate."""
    graded: list[tuple[str, float]] = []
    flags: list[tuple[str, float]] = []
    for key, value in (components or {}).items():
        try:
            amount = float(value)
        except (TypeError, ValueError):
            continue
        if amount in FLAG_VALUES:
            if amount:
                flags.append((str(key), amount))
            continue
        graded.append((str(key), round(amount, 4)))
    graded.sort(key=lambda item: (-abs(item[1]), item[0]))
    flags.sort(key=lambda item: item[0])
    return graded, flags


def _top_components(components: dict[str, Any], count: int = 3) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []
    for key, value in (components or {}).items():
        try:
            amount = float(value)
        except (TypeError, ValueError):
            continue
        if amount:
            scored.append((str(key), round(amount, 4)))
    scored.sort(key=lambda item: (-abs(item[1]), item[0]))
    return scored[:count]


def history_share(components: dict[str, Any]) -> float:
    """How much of a candidate's score came from history rather than content."""
    total = 0.0
    history = 0.0
    for key, value in (components or {}).items():
        try:
            amount = abs(float(value))
        except (TypeError, ValueError):
            continue
        total += amount
        if key in HISTORY_COMPONENTS:
            history += amount
    return round(history / total, 3) if total else 0.0


@dataclass
class Audit:
    task_id: str = ""
    query: str = ""
    retrieval_reason: str = ""
    recall_mode: str = ""
    recorded: bool = False
    selected: list[dict[str, Any]] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)
    by_reason: dict[str, int] = field(default_factory=dict)
    unrecorded_reasons: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "query": self.query,
            "retrieval_reason": self.retrieval_reason,
            "recall_mode": self.recall_mode,
            "recorded": self.recorded,
            "selected": self.selected,
            "excluded": self.excluded,
            "by_reason": self.by_reason,
            "unrecorded_reasons": self.unrecorded_reasons,
            "notes": self.notes,
        }

    def render(self, limit: int = 5) -> list[str]:
        lines: list[str] = []
        if not self.recorded:
            return [f"turn {self.task_id}: no trace recorded, so this turn cannot be audited"]
        if self.query:
            lines.append(f"asked: {self.query[:120]}")
        if self.retrieval_reason:
            lines.append(f"retrieval: {self.retrieval_reason}")
        lines.append(f"selected {len(self.selected)}, excluded {len(self.excluded)}")
        for row in self.selected[:limit]:
            if row["graded"]:
                parts = ", ".join(f"{key} {value:+.3f}" for key, value in row["graded"])
                detail = f" ({parts})"
            elif row["gates"]:
                detail = f" (gates only: {', '.join(row['gates'][:3])})"
            else:
                detail = ""
            history = row["history_share"]
            flag = f", {history:.0%} history" if history else ""
            lines.append(f"  ✓ {row['memory_id'][:8]} score {row['score']:.3f}{detail}{flag}")
        for row in self.excluded[:limit]:
            reason = row["reason"] if row.get("reason_recorded") else "reason not recorded"
            lines.append(f"  ✗ {row['memory_id'][:8]} rank {row['rank']} — {reason}")
        shown: dict[str, int] = {}
        for row in self.excluded[:limit]:
            shown[row["reason"]] = shown.get(row["reason"], 0) + 1
        for reason, count in sorted(self.by_reason.items(), key=lambda item: -item[1]):
            remaining = count - shown.get(reason, 0)
            if remaining <= 0:
                continue
            lines.append(f"  ✗ {remaining} more: {reason or 'reason not recorded'}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return lines


def audit_turn(store: Any, task_id: str) -> Audit:
    """Everything recorded about one turn's retrieval decision."""
    audit = Audit(task_id=str(task_id))
    # `memory_traces()` returns decoded payloads; reading the table directly gives the `_json`
    # suffixed columns instead, and mixing the two is how an audit view quietly reports nothing.
    try:
        rows = store.memory_traces(limit=1, task_id=str(task_id)) or []
    except Exception:  # noqa: BLE001
        rows = []
    if not rows:
        return audit
    trace = rows[0]
    audit.recorded = True
    queries = _field(trace, "queries")
    audit.query = str(queries[0]) if isinstance(queries, list) and queries else ""
    audit.retrieval_reason = str(trace.get("retrieval_reason") or "")
    audit.recall_mode = str(trace.get("recall_mode") or "")
    selected_ids = {str(item) for item in _field(trace, "selected_memory_ids") or []}
    candidates = _field(trace, "candidate_memories") or []
    for candidate in candidates:
        memory_id = str((candidate or {}).get("memory_id") or "")
        if not memory_id:
            continue
        components = dict((candidate or {}).get("components") or {})
        row = {
            "memory_id": memory_id,
            "rank": int((candidate or {}).get("rank") or 0),
            "score": round(float((candidate or {}).get("score") or 0.0), 6),
            "reason": str((candidate or {}).get("reason") or "").strip(),
            "reason_recorded": str((candidate or {}).get("reason") or "").strip().casefold()
            not in PLACEHOLDER_REASONS,
            "top_components": [list(pair) for pair in _top_components(components)],
            "graded": [list(pair) for pair in split_components(components)[0][:3]],
            "gates": [name for name, _ in split_components(components)[1][:6]],
            "explainable": bool(split_components(components)[0]),
            "history_share": history_share(components),
            "selected": bool((candidate or {}).get("selected")) or memory_id in selected_ids,
        }
        (audit.selected if row["selected"] else audit.excluded).append(row)
    audit.selected.sort(key=lambda row: row["rank"] or 10 ** 6)
    audit.excluded.sort(key=lambda row: row["rank"] or 10 ** 6)
    for row in audit.excluded:
        key = row["reason"] if row["reason_recorded"] else ""
        audit.by_reason[key] = audit.by_reason.get(key, 0) + 1
        if not key:
            audit.unrecorded_reasons += 1
    graded_share = (
        sum(1 for row in audit.selected + audit.excluded if row["explainable"])
        / max(1, len(audit.selected) + len(audit.excluded))
    )
    if audit.selected and graded_share < 0.5:
        audit.notes.append(
            f"{graded_share:.0%} of this turn's candidates carry a graded component; the rest record "
            f"gates only, so their scores cannot be explained from the trace"
        )
    if audit.excluded and audit.unrecorded_reasons:
        audit.notes.append(
            f"{audit.unrecorded_reasons} excluded candidate(s) carry no recorded reason"
        )
    if not candidates:
        audit.notes.append("the trace recorded no candidate list, so nothing was excluded")
    rejected_ids = {str(item) for item in _field(trace, "rejected_memory_ids") or []}
    known = {row["memory_id"] for row in audit.selected + audit.excluded}
    never_considered = sorted(rejected_ids - known)
    if never_considered:
        audit.notes.append(
            f"{len(never_considered)} rejected id(s) never appear as candidates: not a ranking "
            f"decision, a recall gap"
        )
    return audit


def why_memory(store: Any, memory_id: str, *, limit: int = 20) -> dict[str, Any]:
    """Every recorded turn that selected or rejected this memory, with the reason given."""
    chosen = 0
    rejected: dict[str, int] = {}
    turns: list[str] = []
    for trace in store.memory_traces(limit=int(limit)) or []:
        selected_ids = {str(item) for item in _field(trace, "selected_memory_ids") or []}
        for candidate in _field(trace, "candidate_memories") or []:
            if str((candidate or {}).get("memory_id") or "") != str(memory_id):
                continue
            turns.append(str(trace.get("task_id") or ""))
            if (candidate or {}).get("selected") or str(memory_id) in selected_ids:
                chosen += 1
            else:
                reason = str((candidate or {}).get("reason") or "") or "reason not recorded"
                rejected[reason] = rejected.get(reason, 0) + 1
    return {
        "memory_id": str(memory_id),
        "turns": len(set(turns)),
        "selected": chosen,
        "rejected": rejected,
        "sentence": (
            f"considered in {len(set(turns))} recorded turn(s): selected {chosen}, "
            f"rejected {sum(rejected.values())}"
            if turns else "this memory does not appear in any recorded trace"
        ),
    }


def render_lines(rows: Iterable[str]) -> str:
    return "\n".join(str(row) for row in rows)
