"""Derived facts, and what to do when their sources move (#57).

An inference is a fact the store made up out of other facts: "the dashboard moved to 8200, so the
bookmark in the runbook is stale", "these two hosts are the same box, so their uptimes are one
number". It is genuinely useful and it is the only kind of memory that can become wrong **without
anything being written to it** — the sources change, and the inference silently keeps asserting what
it used to follow from.

So a derived fact carries three things:

- **its sources**, as `derived_from` rows in `memory_dependencies` — the table that already exists for
  exactly this relation;
- **a ceiling on its confidence** — never above the weakest source, because a conclusion cannot be
  more certain than the fact it rests on;
- **a dirty flag** computed from the sources, not stored: a source superseded, tombstoned, or edited
  after the derivation makes the inference dirty, and the store says so instead of hoping.

Dirty is not deleted. An inference is revalidated by whoever can check it, or retracted explicitly;
the one thing the store will not do is quietly keep asserting it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

VALIDATIONS = "derived_validations"
SNAPSHOTS = "derived_snapshots"
RELATION = "derived_from"
GONE_STATES = ("tombstoned", "deleted", "suppressed")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {VALIDATIONS} (
                    derived_id TEXT PRIMARY KEY,
                    validated_at TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT ''
                )"""
        )
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {SNAPSHOTS} (
                    derived_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    captured_at TEXT NOT NULL,
                    PRIMARY KEY (derived_id, source_id)
                )"""
        )


@dataclass
class Derived:
    memory_id: str
    content: str
    sources: list[str] = field(default_factory=list)
    dirty: bool = False
    reasons: list[str] = field(default_factory=list)
    validated_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "sources": self.sources,
            "dirty": self.dirty,
            "reasons": self.reasons,
            "validated_at": self.validated_at,
        }

    def describe(self) -> str:
        if not self.dirty:
            return f"clean: {self.content[:70]}"
        return (f"dirty ({', '.join(self.reasons)}): {self.content[:70]} "
                f"[{len(self.sources)} source(s)]")


def _source_rows(store: Any, sources: Iterable[str]) -> dict[str, dict[str, Any]]:
    ids = [str(source) for source in sources]
    if not ids:
        return {}
    clause = ",".join("?" for _ in ids)
    rows = _rows(
        store,
        f"""SELECT id, state, confidence, updated_at, created_at, supersedes_id,
                   COALESCE(content_hash, '') AS content_hash
            FROM memories WHERE id IN ({clause})""",
        tuple(ids),
    )
    detail = {str(row["id"]): row for row in rows}
    # `supersedes_id` lives on the WINNER's row, so a source is replaced when some *other* memory
    # names it. Reading the source's own column answers the opposite question — and the first
    # version of this check did exactly that, so it reported a replaced source as clean.
    for row in _rows(
        store,
        f"""SELECT supersedes_id FROM memories
            WHERE supersedes_id IN ({clause})""",
        tuple(ids),
    ):
        target = str(row["supersedes_id"])
        if target in detail:
            detail[target]["replaced"] = True
    return detail


def derive(store: Any, content: str, sources: Iterable[str], *, kind: str = "semantic",
           confidence: float | None = None, source_context: str = "",
           at: datetime | None = None) -> dict[str, Any]:
    """Write an inference and record what it was derived from.

    The confidence ceiling is `min(sources)`: a conclusion cannot be more certain than the weakest
    fact it rests on, and a caller asking for more gets the ceiling instead.
    """
    _ensure(store)  # the snapshots table is created here, before any dependency is written
    source_ids = [str(source) for source in sources if str(source).strip()]
    if not source_ids:
        return {"created": False, "reason": "a derived fact needs at least one source"}
    rows = _source_rows(store, source_ids)
    missing = [source for source in source_ids if source not in rows]
    if missing:
        return {"created": False, "reason": f"unknown source(s): {', '.join(missing[:3])}"}
    ceiling = min(float(row.get("confidence") or 0.0) for row in rows.values())
    stated = ceiling if confidence is None else min(float(confidence), ceiling)
    moment = at or _now()
    memory_id = store.add_memory(
        content=str(content),
        kind=kind,
        source_type="derivation",
        source_category="AGENT_INFERENCE",
        confidence=stated,
        source_context=str(source_context) or
        f"derived from {len(source_ids)} memory(ies)",
    )[0]
    with store.transaction() as conn:
        conn.execute("UPDATE memories SET record_role = 'derived' WHERE id = ?", (memory_id,))
        for source in source_ids:
            conn.execute(
                """INSERT OR IGNORE INTO memory_dependencies
                   (memory_id, evidence_id, relation, weight, active, created_at)
                   VALUES (?, ?, ?, 1.0, 1, ?)""",
                (memory_id, source, RELATION, moment.isoformat()),
            )
            # The source's content at the moment of derivation. Comparing timestamps instead was
            # wrong: this store stamps `updated_at` in bulk during migrations, so every inference
            # read as dirty forever (78 of 78 on the live store).
            conn.execute(
                f"""INSERT INTO {SNAPSHOTS} (derived_id, source_id, content_hash, captured_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(derived_id, source_id) DO UPDATE SET
                        content_hash = excluded.content_hash,
                        captured_at = excluded.captured_at""",
                (memory_id, source, str(rows[source].get("content_hash") or ""),
                 moment.isoformat()),
            )
    return {
        "created": True,
        "memory_id": memory_id,
        "sources": source_ids,
        "confidence": stated,
        "ceiling": ceiling,
        "capped": stated < float(confidence) if confidence is not None else False,
    }


def sources_of(store: Any, memory_id: str) -> list[str]:
    rows = _rows(
        store,
        """SELECT evidence_id FROM memory_dependencies
           WHERE memory_id = ? AND relation = ? AND active = 1""",
        (str(memory_id), RELATION),
    )
    return sorted({str(row["evidence_id"]) for row in rows})


def dependents(store: Any, source_id: str) -> list[str]:
    rows = _rows(
        store,
        """SELECT memory_id FROM memory_dependencies
           WHERE evidence_id = ? AND relation = ? AND active = 1""",
        (str(source_id), RELATION),
    )
    return sorted({str(row["memory_id"]) for row in rows})


def _snapshots(store: Any, memory_id: str) -> dict[str, str]:
    _ensure(store)
    return {
        str(row["source_id"]): str(row["content_hash"])
        for row in _rows(
            store, f"SELECT source_id, content_hash FROM {SNAPSHOTS} WHERE derived_id = ?",
            (str(memory_id),),
        )
    }


def _derived_at(store: Any, memory_id: str) -> str:
    """When the derivation was recorded — the baseline for "the sources moved since"."""
    rows = _rows(
        store,
        """SELECT MAX(created_at) AS at FROM memory_dependencies
           WHERE memory_id = ? AND relation = ?""",
        (str(memory_id), RELATION),
    )
    return str(rows[0]["at"]) if rows and rows[0].get("at") else ""


def _validated(store: Any) -> dict[str, str]:
    _ensure(store)
    return {
        str(row["derived_id"]): str(row["validated_at"])
        for row in _rows(store, f"SELECT derived_id, validated_at FROM {VALIDATIONS}")
    }


def check(store: Any, memory_id: str) -> Derived:
    """Whether one inference still follows from what it was derived from."""
    sources = sources_of(store, memory_id)
    row = _rows(store, "SELECT id, content FROM memories WHERE id = ?", (str(memory_id),))
    content = str(row[0]["content"]) if row else ""
    validated = _validated(store).get(str(memory_id), "")
    snapshots = _snapshots(store, memory_id)
    reasons: list[str] = []
    detail = _source_rows(store, sources)
    unverifiable = bool(sources) and not snapshots
    for source in sources:
        source_row = detail.get(source)
        if source_row is None:
            reasons.append("source missing")
            continue
        if str(source_row.get("state") or "") in GONE_STATES:
            reasons.append(f"source {source[:8]} {source_row.get('state')}")
        if source_row.get("replaced"):
            reasons.append(f"source {source[:8]} was replaced")
        recorded = snapshots.get(source)
        current = str(source_row.get("content_hash") or "")
        if recorded is not None and recorded != current:
            reasons.append(f"source {source[:8]} changed since the derivation")
    unique = sorted(set(reasons))
    if unverifiable:
        # An inference written before snapshots existed has nothing to compare against. Saying
        # "dirty" would be a claim the store cannot support, so it says what it knows.
        unique = sorted(set(unique + ["no snapshot of the sources (predates this check)"]))
    return Derived(memory_id=str(memory_id), content=content, sources=sources,
                   dirty=bool(unique), reasons=unique, validated_at=validated)


def dirty(store: Any, *, limit: int = 200) -> list[Derived]:
    """Every inference whose sources have moved since it was last checked."""
    rows = _rows(
        store,
        """SELECT DISTINCT m.id FROM memories m
           JOIN memory_dependencies d ON d.memory_id = m.id AND d.relation = ? AND d.active = 1
           WHERE m.state NOT IN ('tombstoned','deleted') LIMIT ?""",
        (RELATION, int(limit)),
    )
    found = [check(store, str(row["id"])) for row in rows]
    return [item for item in found if item.dirty]


def all_derived(store: Any, *, limit: int = 200) -> list[Derived]:
    rows = _rows(
        store,
        """SELECT DISTINCT m.id FROM memories m
           JOIN memory_dependencies d ON d.memory_id = m.id AND d.relation = ? AND d.active = 1
           WHERE m.state NOT IN ('tombstoned','deleted') LIMIT ?""",
        (RELATION, int(limit)),
    )
    return [check(store, str(row["id"])) for row in rows]


def revalidate(store: Any, memory_id: str, *, note: str = "",
               at: datetime | None = None) -> dict[str, Any]:
    """A person or a pass has re-checked the inference: it is clean until its sources move again."""
    if not sources_of(store, memory_id):
        return {"revalidated": False, "reason": "not a derived fact"}
    _ensure(store)
    moment = (at or _now()).isoformat()
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT INTO {VALIDATIONS} (derived_id, validated_at, note)
                VALUES (?, ?, ?)
                ON CONFLICT(derived_id) DO UPDATE SET
                    validated_at = excluded.validated_at,
                    note = excluded.note""",
            (str(memory_id), moment, str(note)[:200]),
        )
        snapshot = _source_rows(store, sources_of(store, memory_id))
        for source, row in snapshot.items():
            conn.execute(
                f"""INSERT INTO {SNAPSHOTS} (derived_id, source_id, content_hash, captured_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(derived_id, source_id) DO UPDATE SET
                        content_hash = excluded.content_hash,
                        captured_at = excluded.captured_at""",
                (str(memory_id), source, str(row.get("content_hash") or ""), moment),
            )
    return {"revalidated": True, "memory_id": str(memory_id), "validated_at": moment}


def retract(store: Any, memory_id: str, *, reason: str = "",
            at: datetime | None = None) -> dict[str, Any]:
    """Stop asserting an inference. The row stays; the dependency rows stay as history."""
    if not sources_of(store, memory_id):
        return {"retracted": False, "reason": "not a derived fact"}
    moment = at or _now()
    with store.transaction() as conn:
        conn.execute(
            "UPDATE memories SET state = 'tombstoned', updated_at = ? WHERE id = ?",
            (moment.isoformat(), str(memory_id)),
        )
        conn.execute(
            """UPDATE memory_dependencies SET active = 0
               WHERE memory_id = ? AND relation = ?""",
            (str(memory_id), RELATION),
        )
    return {"retracted": True, "memory_id": str(memory_id), "reason": str(reason)[:200]}


def report(store: Any, *, limit: int = 500) -> dict[str, Any]:
    items = all_derived(store, limit=limit)
    reasons: dict[str, int] = {}
    for item in items:
        for reason in item.reasons:
            key = reason.split("(")[0].strip()
            if reason.startswith("source "):
                key = "source " + " ".join(reason.split(" ")[2:]).split("(")[0].strip()
            reasons[key[:48]] = reasons.get(key[:48], 0) + 1
    return {
        "derived": len(items),
        "dirty": sum(1 for item in items if item.dirty),
        "clean": sum(1 for item in items if not item.dirty),
        "reasons": reasons,
        "note": "a derived fact is dirty, never silently stale: it is revalidated or retracted",
    }
