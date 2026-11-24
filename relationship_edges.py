"""Typed, evidenced relations between two entities (#62).

`entity_nodes` gives each thing a page. What it cannot answer is how two things stand to each other:
*"who owns the r630"*, *"what does Jarvis manage"*, *"who does Jacoby work with"*. Memory edges record
that two **facts** were seen together; this records that two **entities** have a named relationship,
which is a different claim and needs its own table.

Four rules, each of which exists because the alternative is a graph of confident nonsense:

- **a relation is typed or it is not recorded.** `owns`, `manages`, `works_with`, `depends_on`,
  `runs_on`, `member_of`, `reports_to`. "Related to" is what an untyped edge says, and it says nothing.
- **an edge is evidence or it is a guess.** Every row carries the memory it came from, or the person
  who asserted it; a relation with no evidence is reported as a proposal, never stored as fact.
- **direction is part of the meaning.** `manages` runs one way; `works_with` does not, and is stored
  in canonical order so the same pair cannot appear twice with the roles swapped.
- **nothing is deleted.** A relation that stops being true is closed (`valid_to`), because "who used
  to own this" is a real question and the answer is in the same table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

TABLE = "entity_relations"

# The vocabulary. Typed on purpose: an open-ended relation string is a comment, not a fact.
RELATIONS = ("owns", "manages", "works_with", "depends_on", "runs_on", "member_of", "reports_to")
SYMMETRIC = frozenset({"works_with"})

VERBS = {
    "owns": ("owns", "own", "is the owner of", "belongs to"),
    "manages": ("manages", "manage", "administers", "maintains"),
    "works_with": ("works with", "collaborates with", "works alongside"),
    "depends_on": ("depends on", "relies on", "requires"),
    "runs_on": ("runs on", "is hosted on", "lives on", "is deployed on"),
    "member_of": ("is a member of", "belongs to the", "is part of"),
    "reports_to": ("reports to", "answers to"),
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    relation_id TEXT PRIMARY KEY,
                    source_entity TEXT NOT NULL,
                    target_entity TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    evidence TEXT NOT NULL DEFAULT '',
                    weight REAL NOT NULL DEFAULT 1.0,
                    asserted_by TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    valid_to TEXT,
                    UNIQUE(source_entity, target_entity, relation)
                )"""
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {TABLE}_source_idx ON {TABLE} (source_entity)"
        )


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def _normalise(relation: str) -> str:
    value = str(relation or "").strip().lower().replace(" ", "_").replace("-", "_")
    return value if value in RELATIONS else ""


def _pair(source: str, target: str, relation: str) -> tuple[str, str]:
    """Canonical order for symmetric relations, so the same pair cannot appear twice swapped."""
    left, right = str(source), str(target)
    if relation in SYMMETRIC and right < left:
        return right, left
    return left, right


@dataclass
class Relation:
    source: str
    target: str
    relation: str
    evidence: str = ""
    weight: float = 1.0
    asserted_by: str = ""
    created_at: str = ""
    valid_to: str = ""

    @property
    def open(self) -> bool:
        return not self.valid_to

    def triple(self) -> str:
        arrow = "<->" if self.relation in SYMMETRIC else "->"
        return f"{self.source} {arrow}[{self.relation}] {self.target}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "evidence": self.evidence,
            "weight": self.weight,
            "asserted_by": self.asserted_by,
            "created_at": self.created_at,
            "valid_to": self.valid_to or None,
            "open": self.open,
            "triple": self.triple(),
        }

    def describe(self) -> str:
        state = "open" if self.open else f"closed {self.valid_to[:10]}"
        proof = f" — {self.evidence[:60]}" if self.evidence else " — no evidence"
        return f"{self.triple()} ({state}){proof}"


def link(store: Any, source: str, target: str, relation: str, *, evidence: str = "",
         asserted_by: str = "", weight: float = 1.0,
         at: datetime | None = None) -> dict[str, Any]:
    """Record a typed relation. Refuses without a type, and refuses without evidence."""
    resolved = _normalise(relation)
    if not resolved:
        return {"linked": False, "reason": f"relation must be one of {', '.join(RELATIONS)}"}
    left, right = _pair(source, target, resolved)
    if not left or not right or left == right:
        return {"linked": False, "reason": "a relation needs two different entities"}
    proof = str(evidence or "").strip()
    who = str(asserted_by or "").strip()
    if not proof and not who:
        return {"linked": False, "reason": "a relation needs evidence or a person asserting it"}
    moment = (at or _now()).isoformat()
    _ensure(store)
    with store.transaction() as conn:
        existing = conn.execute(
            f"""SELECT relation_id, weight FROM {TABLE}
                WHERE source_entity = ? AND target_entity = ? AND relation = ?""",
            (left, right, resolved),
        ).fetchone()
        if existing is not None:
            # Re-asserting a relation reinforces it rather than duplicating it.
            conn.execute(
                f"""UPDATE {TABLE} SET weight = ?, valid_to = NULL,
                        evidence = CASE WHEN ? <> '' THEN ? ELSE evidence END,
                        asserted_by = CASE WHEN ? <> '' THEN ? ELSE asserted_by END
                    WHERE relation_id = ?""",
                (min(1.0, float(existing["weight"]) + 0.1), proof, proof, who, who,
                 str(existing["relation_id"])),
            )
            return {"linked": True, "reinforced": True, "relation_id": str(existing["relation_id"])}
        relation_id = f"{left[:24]}|{resolved}|{right[:24]}"
        conn.execute(
            f"""INSERT INTO {TABLE}
                (relation_id, source_entity, target_entity, relation, evidence, weight,
                 asserted_by, created_at, valid_to)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
            (relation_id, left, right, resolved, proof, float(weight), who, moment),
        )
    return {"linked": True, "reinforced": False, "relation_id": relation_id}


def close(store: Any, source: str, target: str, relation: str, *,
          at: datetime | None = None) -> dict[str, Any]:
    """A relation that stopped being true. Closed, never deleted."""
    resolved = _normalise(relation)
    if not resolved:
        return {"closed": False, "reason": "unknown relation"}
    left, right = _pair(source, target, resolved)
    moment = (at or _now()).isoformat()
    _ensure(store)
    with store.transaction() as conn:
        cursor = conn.execute(
            f"""UPDATE {TABLE} SET valid_to = ?
                WHERE source_entity = ? AND target_entity = ? AND relation = ?
                  AND valid_to IS NULL""",
            (moment, left, right, resolved),
        )
        changed = int(cursor.rowcount or 0)
    return {"closed": bool(changed), "rows": changed}


def relations_of(store: Any, entity: str, *, include_closed: bool = False,
                 limit: int = 50) -> list[Relation]:
    """Everything this entity owns, manages, runs on, or works with — in either direction."""
    _ensure(store)
    clause = "" if include_closed else "AND valid_to IS NULL"
    rows = _rows(
        store,
        f"""SELECT * FROM {TABLE}
            WHERE (source_entity = ? OR target_entity = ?) {clause}
            ORDER BY relation, created_at DESC LIMIT ?""",
        (str(entity), str(entity), int(limit)),
    )
    return [
        Relation(source=str(row["source_entity"]), target=str(row["target_entity"]),
                 relation=str(row["relation"]), evidence=str(row["evidence"] or ""),
                 weight=float(row["weight"] or 0.0), asserted_by=str(row["asserted_by"] or ""),
                 created_at=str(row["created_at"] or ""), valid_to=str(row["valid_to"] or ""))
        for row in rows
    ]


def between(store: Any, left: str, right: str, *, include_closed: bool = False) -> list[Relation]:
    """Direct relations between two specific entities, either way round."""
    return [
        relation for relation in relations_of(store, left, include_closed=include_closed, limit=500)
        if {relation.source, relation.target} == {str(left), str(right)}
    ]


def _known_names(store: Any, limit: int = 2000) -> list[str]:
    try:
        from . import entity_nodes

        return sorted({name for name in entity_nodes.known_names(store, limit=limit)}, key=len,
                      reverse=True)
    except Exception:  # noqa: BLE001
        return []


def propose(store: Any, text: str, *, limit: int = 5) -> list[dict[str, Any]]:
    """Relations a sentence asserts between two known entities. Proposals only, nothing written.

    Detection is deliberately dull: find two known names in the sentence, find a verb phrase between
    them, and report the typed triple. A sentence with no known name, or with no relation phrase, is
    silence — inventing a relation from proximity is how a knowledge graph becomes fiction.
    """
    body = str(text or "")
    lowered = body.lower()
    names = [name for name in _known_names(store) if name and name.lower() in lowered]
    if len(names) < 2:
        return []
    found: list[dict[str, Any]] = []
    for relation, verbs in VERBS.items():
        for verb in verbs:
            position = lowered.find(verb)
            if position < 0:
                continue
            before = [name for name in names if lowered.find(name.lower(), 0, position) >= 0]
            after = [name for name in names if lowered.find(name.lower(), position + len(verb)) >= 0]
            if not before or not after:
                continue
            source = max(before, key=lambda name: lowered.rfind(name.lower(), 0, position))
            target = min(after, key=lambda name: lowered.find(name.lower(), position + len(verb)))
            if source.lower() == target.lower():
                continue
            entry = {
                "source": source,
                "target": target,
                "relation": relation,
                "evidence": body.strip()[:200],
                "sentence": f"{source} [{relation}] {target}",
            }
            if entry not in found:
                found.append(entry)
            break
    return found[: max(0, int(limit))]


def record_proposals(store: Any, text: str, *, asserted_by: str = "",
                     at: datetime | None = None) -> list[dict[str, Any]]:
    """Turn a sentence into relations, keeping only what has a named asserter or evidence."""
    return [
        link(store, proposal["source"], proposal["target"], proposal["relation"],
             evidence=proposal["evidence"], asserted_by=asserted_by, at=at)
        for proposal in propose(store, text)
    ]


def report(store: Any) -> dict[str, Any]:
    _ensure(store)
    rows = _rows(
        store,
        f"""SELECT relation, COUNT(*) AS count,
                   SUM(CASE WHEN valid_to IS NULL THEN 1 ELSE 0 END) AS open
            FROM {TABLE} GROUP BY relation""",
    )
    by_relation = {str(row["relation"]): int(row["count"]) for row in rows}
    return {
        "relations": sum(by_relation.values()),
        "open": sum(int(row["open"] or 0) for row in rows),
        "by_relation": by_relation,
        "vocabulary": list(RELATIONS),
        "note": "an untyped edge says 'related to', which says nothing",
    }
