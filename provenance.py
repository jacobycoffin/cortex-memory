"""For each fact, the sequence of evidence that produced it (#51).

A memory is the end of an argument the store made with itself: a turn was read, something was
extracted, a source authority put a ceiling on the confidence, a person corrected it, another fact
corroborated it, an autojudge approved it. All of that is already recorded — in `memories`, in the
extraction columns, in `self_corrections`, `memory_merges`, `disputes`, `edge_evidence` — but scattered
across seven tables whose names only make sense to whoever wrote them.

`chain()` puts it in one ordered list, and `link()` records the events that have nowhere else to live
yet. Two properties:

- **the chain is assembled, not authored.** Every step points at a row that exists; a step with no
  evidence is not invented. Where the store genuinely does not know why a fact exists, the chain says
  `unknown origin` rather than a plausible story.
- **order is by time, and ties keep insertion order.** A chain is an argument, so its sequence is the
  content; sorting instants with equal timestamps alphabetically would scramble it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

TABLE = "belief_chain"

STEP_KINDS = (
    "captured",       # extracted from a turn
    "approved",       # autojudge or operator accepted it
    "corrected",      # a person replaced it
    "merged",         # another fact was folded into it
    "corroborated",   # independent evidence agreed
    "disputed",       # conflicting facts were recorded
    "superseded",     # a later fact replaced it
    "verified",       # a tool confirmed it
    "unknown",        # the store cannot say
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    step_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    source_id TEXT NOT NULL DEFAULT '',
                    at TEXT NOT NULL
                )"""
        )
        conn.execute(
            f"""CREATE INDEX IF NOT EXISTS {TABLE}_memory_idx
                ON {TABLE} (memory_id, at)"""
        )


def link(store: Any, memory_id: str, kind: str, *, detail: str = "", source_id: str = "",
         at: datetime | None = None) -> dict[str, Any] | None:
    """Record one step of a fact's evidence. Never raises into a write path."""
    identifier = str(memory_id or "").strip()
    resolved = str(kind or "").strip().lower()
    if not identifier:
        return None
    if resolved not in STEP_KINDS:
        resolved = "unknown"
    try:
        _ensure(store)
        step = {
            "step_id": uuid.uuid4().hex[:12],
            "memory_id": identifier,
            "kind": resolved,
            "detail": str(detail or "")[:400],
            "source_id": str(source_id or "")[:64],
            "at": (at or _now()).isoformat(),
        }
        with store.transaction() as conn:
            conn.execute(
                f"""INSERT INTO {TABLE} (step_id, memory_id, kind, detail, source_id, at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                (step["step_id"], step["memory_id"], step["kind"], step["detail"],
                 step["source_id"], step["at"]),
            )
        return step
    except Exception:  # noqa: BLE001 - provenance is evidence, never a reason a write fails
        return None


@dataclass
class Chain:
    memory_id: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    missing: bool = False

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [step for step in self.steps if step["kind"] == kind]

    @property
    def first(self) -> dict[str, Any] | None:
        return self.steps[0] if self.steps else None

    @property
    def last(self) -> dict[str, Any] | None:
        return self.steps[-1] if self.steps else None

    def describe(self) -> str:
        if self.missing:
            return "no such memory"
        if not self.steps:
            return "unknown origin: nothing in the store explains this fact"
        parts = [f"{step['kind']}" + (f" ({step['detail']})" if step["detail"] else "")
                 for step in self.steps]
        return " -> ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {"memory_id": self.memory_id, "steps": self.steps, "missing": self.missing,
                "sentence": self.describe()}


def _row(store: Any, sql: str, params: tuple) -> dict[str, Any] | None:
    try:
        row = store._conn.execute(sql, params).fetchone()
    except Exception:  # noqa: BLE001
        return None
    return dict(row) if row else None


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def _creation_step(store: Any, memory_id: str) -> dict[str, Any] | None:
    row = _row(
        store,
        """SELECT created_at, source_type, source_category, source_context, extraction_method,
                  supersedes_id, confidence
           FROM memories WHERE id = ?""",
        (memory_id,),
    )
    if row is None:
        return None
    how = "captured"
    category = str(row.get("source_category") or "").upper()
    if category == "TOOL_VERIFIED":
        how = "verified"
    elif category == "USER_CORRECTION":
        how = "corrected"
    detail_bits = [part for part in (
        str(row.get("source_type") or ""),
        category,
        str(row.get("extraction_method") or ""),
    ) if part]
    return {
        "step_id": f"origin-{memory_id}",
        "memory_id": memory_id,
        "kind": how,
        "detail": " ".join(detail_bits)[:400],
        "source_id": str(row.get("source_context") or "")[:64],
        "at": str(row.get("created_at") or ""),
        "supersedes_id": row.get("supersedes_id") or "",
    }


def chain(store: Any, memory_id: str) -> Chain:
    """Every step of evidence behind one fact, oldest first."""
    identifier = str(memory_id or "").strip()
    origin = _creation_step(store, identifier)
    if origin is None:
        return Chain(identifier, missing=True)
    _ensure(store)
    steps: list[dict[str, Any]] = [origin]
    extras: list[dict[str, Any]] = []

    for row in _rows(
        store,
        f"SELECT step_id, memory_id, kind, detail, source_id, at FROM {TABLE} WHERE memory_id = ?",
        (identifier,),
    ):
        extras.append(row)

    # a correction that produced this fact names the fact it replaced
    for row in _rows(
        store,
        "SELECT correction_id, superseded_id, marker, created_at FROM self_corrections WHERE memory_id = ?",
        (identifier,),
    ) if _table_exists(store, "self_corrections") else []:
        extras.append({
            "step_id": f"correction-{row['correction_id']}", "memory_id": identifier,
            "kind": "corrected",
            "detail": f"person corrected {str(row.get('superseded_id') or '')[:8]}"
                      + (f" ({row['marker']})" if row.get("marker") else ""),
            "source_id": str(row["correction_id"]), "at": str(row.get("created_at") or ""),
        })

    # a merge that produced this fact
    for row in _rows(
        store,
        "SELECT merge_id, source_id, consented_by, created_at FROM memory_merges WHERE target_id = ?",
        (identifier,),
    ) if _table_exists(store, "memory_merges") else []:
        extras.append({
            "step_id": f"merge-{row['merge_id']}", "memory_id": identifier, "kind": "merged",
            "detail": f"merged {str(row.get('source_id') or '')[:8]} by "
                      f"{row.get('consented_by') or 'someone'}",
            "source_id": str(row["merge_id"]), "at": str(row.get("created_at") or ""),
        })

    # a dispute it takes part in
    for row in _rows(
        store,
        "SELECT left_id, right_id, winner_id, created_at FROM disputes "
        "WHERE left_id = ? OR right_id = ?",
        (identifier, identifier),
    ) if _table_exists(store, "disputes") else []:
        other = row["right_id"] if str(row["left_id"]) == identifier else row["left_id"]
        # the pair is the key: `disputes` is one row per unordered pair, with no id of its own
        pair = f"{str(row['left_id'])[:8]}~{str(row['right_id'])[:8]}"
        verdict = " resolved" if row.get("winner_id") else ""
        extras.append({
            "step_id": f"dispute-{pair}", "memory_id": identifier, "kind": "disputed",
            "detail": f"conflicts with {str(other)[:8]}{verdict}",
            "source_id": pair, "at": str(row.get("created_at") or ""),
        })

    # independent evidence recorded against it
    evidence = _rows(
        store,
        "SELECT evidence_type, COUNT(*) AS count FROM edge_evidence "
        "WHERE src_id = ? OR dst_id = ? GROUP BY evidence_type",
        (identifier, identifier),
    ) if _table_exists(store, "edge_evidence") else []
    for row in evidence:
        extras.append({
            "step_id": f"evidence-{identifier}-{row['evidence_type']}", "memory_id": identifier,
            "kind": "corroborated",
            "detail": f"{row['evidence_type']} x{row['count']}",
            "source_id": str(row["evidence_type"]), "at": origin["at"],
        })

    if origin.get("supersedes_id"):
        steps.append({
            "step_id": f"supersedes-{identifier}", "memory_id": identifier, "kind": "superseded",
            "detail": f"replaces {str(origin['supersedes_id'])[:8]}",
            "source_id": str(origin["supersedes_id"]), "at": origin["at"],
        })

    # oldest first, and equal timestamps keep the order they were discovered in
    ordered = sorted(
        [step for step in steps if step["kind"] != origin["kind"]] + extras,
        key=lambda step: str(step.get("at") or ""),
    )
    return Chain(identifier, [origin] + ordered)


def _table_exists(store: Any, name: str) -> bool:
    try:
        row = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
    except Exception:  # noqa: BLE001
        return False
    return row is not None


def report(store: Any, *, limit: int = 5000) -> dict[str, Any]:
    _ensure(store)
    rows = _rows(
        store,
        f"SELECT kind, COUNT(*) AS count FROM {TABLE} GROUP BY kind LIMIT ?",
        (int(limit),),
    )
    by_kind = {str(row["kind"]): int(row["count"]) for row in rows}
    memories = _row(store, "SELECT COUNT(*) AS count FROM memories", ()) or {"count": 0}
    return {
        "steps": sum(by_kind.values()),
        "by_kind": by_kind,
        "memories": int(memories["count"]),
        "note": "every fact has an origin step assembled from its own row; link() adds events "
                "that have nowhere else to live",
    }
