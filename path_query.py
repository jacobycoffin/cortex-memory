"""How are these two related? The hops, with the evidence attached (#75).

A path is only worth showing if the person asking can judge it. "Memory A is connected to memory B"
is unfalsifiable; *A —co_observed→ C —supersedes→ B*, where A and C were retrieved together in three
sessions and B replaced C, is checkable. So every hop carries its relation, its current weight and
the `edge_evidence` rows behind it, and a hop with no evidence says so rather than hiding it.

Two details that decide whether this is useful or annoying:

- **traversal is undirected.** A relation like `supersedes` is directional, but the question "how are
  these related" is not: a path that exists only one way answers the question just as well.
- **no path is an answer too.** When nothing connects them within the depth limit, the reply lists
  **shared neighbours and shared entities** instead of an empty result, because the honest answer is
  usually "not directly, but both sit near the same thing".

Determinism matters here as in ranking (#220): equal-length paths are ordered by (relation, id) so
the same question gets the same answer twice.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any

DEFAULT_DEPTH = 4
MAX_FRONTIER = 20_000


def _ensure_schema_loaded() -> None:  # pragma: no cover - documentation hook
    """Nothing to do; the query only reads existing tables."""


@dataclass
class Hop:
    src_id: str
    dst_id: str
    relation: str
    weight: float
    evidence: list[str] = field(default_factory=list)
    content_src: str = ""
    content_dst: str = ""

    def describe(self) -> str:
        text = f"{self.relation} ({self.weight:.2f})"
        if self.evidence:
            return f"{text}: " + "; ".join(self.evidence[:2])
        return f"{text}: no recorded evidence"

    def as_dict(self) -> dict[str, Any]:
        return {
            "src": self.src_id,
            "dst": self.dst_id,
            "relation": self.relation,
            "weight": round(self.weight, 4),
            "evidence": list(self.evidence),
            "src_content": self.content_src[:120],
            "dst_content": self.content_dst[:120],
        }


@dataclass
class Path:
    left: str
    right: str
    hops: list[Hop] = field(default_factory=list)
    reason: str = ""
    shared_entities: list[str] = field(default_factory=list)
    shared_neighbours: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.hops)

    def explain(self) -> str:
        if not self.found:
            if self.reason in ("", "no path"):
                headline = "no path within the depth limit"
            elif self.reason == "the same memory":
                headline = "the same memory on both sides"
            else:
                headline = self.reason
            parts = [headline]
            if self.shared_neighbours:
                parts.append(f"{len(self.shared_neighbours)} shared neighbours")
            if self.shared_entities:
                parts.append("shared entities: " + ", ".join(self.shared_entities[:3]))
            return "; ".join(parts)
        return " -> ".join([self.hops[0].content_src or self.hops[0].src_id] +
                           [hop.content_dst or hop.dst_id for hop in self.hops])


def _content(store: Any, memory_id: str) -> str:
    try:
        row = store._conn.execute(
            "SELECT content FROM memories WHERE id = ?", (str(memory_id),)
        ).fetchone()
    except Exception:  # noqa: BLE001
        return ""
    return str(row["content"]) if row else ""


def resolve(store: Any, text: str) -> str | None:
    """A memory id, or the best memory matching a phrase. Ids pass through untouched."""
    candidate = str(text or "").strip()
    if not candidate:
        return None
    try:
        row = store._conn.execute("SELECT id FROM memories WHERE id = ?", (candidate,)).fetchone()
    except Exception:  # noqa: BLE001
        row = None
    if row:
        return str(row["id"])
    try:
        rows = store._conn.execute(
            """SELECT id FROM memories
               WHERE state NOT IN ('tombstoned','deleted') AND content LIKE ?
               ORDER BY LENGTH(content) ASC LIMIT 5""",
            (f"%{candidate}%",),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return None
    return str(rows[0]["id"]) if rows else None


def _neighbours(store: Any, memory_id: str) -> list[tuple[str, str, float]]:
    try:
        rows = store._conn.execute(
            """SELECT src_id, dst_id, relation, weight FROM edges
               WHERE src_id = ? OR dst_id = ?""",
            (str(memory_id), str(memory_id)),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    out: list[tuple[str, str, float]] = []
    for row in rows:
        src, dst, relation, weight = (
            str(row["src_id"]), str(row["dst_id"]), str(row["relation"]), float(row["weight"] or 0.0)
        )
        other = dst if src == str(memory_id) else src
        out.append((other, relation, weight))
    # Deterministic frontier: strongest first, then by id/relation.
    out.sort(key=lambda item: (-item[2], item[0], item[1]))
    return out


def _evidence(store: Any, left: str, right: str, relation: str) -> list[str]:
    try:
        rows = store._conn.execute(
            """SELECT summary FROM edge_evidence
               WHERE relation = ?
                 AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
               ORDER BY evidence_id LIMIT 3""",
            (relation, left, right, right, left),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    return [str(row["summary"]) for row in rows if row["summary"]]


def _shared_entities(store: Any, left: str, right: str) -> list[str]:
    found: list[str] = []
    for memory_id in (left, right):
        try:
            row = store._conn.execute(
                "SELECT entities_json FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        except Exception:  # noqa: BLE001
            return []
        if not row or not row["entities_json"]:
            found.append("")
            continue
        try:
            names = json.loads(row["entities_json"])
        except (TypeError, ValueError):
            names = []
        found.append("\x00".join(sorted(str(name).lower() for name in names)))
    if len(found) != 2 or not found[0] or not found[1]:
        return []
    left_names = set(found[0].split("\x00"))
    right_names = set(found[1].split("\x00"))
    return sorted(left_names & right_names)


def between(
    store: Any,
    left: str,
    right: str,
    *,
    depth: int = DEFAULT_DEPTH,
    explain_missing: bool = True,
    frontier: int = MAX_FRONTIER,
) -> Path:
    """Breadth-first shortest path, undirected, with evidence on every hop."""
    left_id = resolve(store, left)
    right_id = resolve(store, right)
    if left_id is None or right_id is None:
        missing = left if left_id is None else right
        return Path(str(left or ""), str(right or ""), reason=f"unknown memory: {missing}")
    if left_id == right_id:
        return Path(left_id, right_id, reason="the same memory")

    parents: dict[str, tuple[str, str, float]] = {}
    seen = {left_id}
    queue: deque[str] = deque([left_id])
    while queue and len(seen) < frontier:
        node = queue.popleft()
        path_length = 0
        walker = node
        while walker in parents:
            walker = parents[walker][0]
            path_length += 1
        if path_length >= depth:
            continue
        for other, relation, weight in _neighbours(store, node):
            if other in seen:
                continue
            seen.add(other)
            parents[other] = (node, relation, weight)
            if other == right_id:
                queue.clear()
                break
            queue.append(other)

    if right_id not in parents:
        result = Path(left_id, right_id, reason="no path")
        if explain_missing:
            result.shared_entities = _shared_entities(store, left_id, right_id)
            left_neighbours = {item[0] for item in _neighbours(store, left_id)}
            right_neighbours = {item[0] for item in _neighbours(store, right_id)}
            result.shared_neighbours = sorted(left_neighbours & right_neighbours)
        return result

    chain: list[tuple[str, str, str, float]] = []
    node = right_id
    while node != left_id:
        parent, relation, weight = parents[node]
        chain.append((parent, node, relation, weight))
        node = parent
    chain.reverse()

    hops = [
        Hop(
            src_id=src, dst_id=dst, relation=relation, weight=weight,
            evidence=_evidence(store, src, dst, relation),
            content_src=_content(store, src), content_dst=_content(store, dst),
        )
        for src, dst, relation, weight in chain
    ]
    return Path(left_id, right_id, hops=hops)


def describe(store: Any, left: str, right: str, *, depth: int = DEFAULT_DEPTH) -> dict[str, Any]:
    path = between(store, left, right, depth=depth)
    return {
        "found": path.found,
        "hops": [hop.as_dict() for hop in path.hops],
        "explanation": path.explain(),
        "shared_entities": path.shared_entities,
        "shared_neighbours": path.shared_neighbours[:5],
    }
