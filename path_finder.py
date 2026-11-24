"""How are these two memories connected? (#165)

The graph answers one question a list cannot: *why are these two things related?* The honest answer is
a **path**, and the honest caveats are that a path is a chain of recorded co-use, not proof of
causation, and that a short path through a hub is mostly about the hub.

Rules:

- **breadth-first, shortest path.** The shortest chain is the least assumptive one; a longer route
  through five memories asserts five relationships where one would do.
- **deterministic.** Neighbours are visited in id order, so the path between two memories does not
  change between runs. A pathfinder that answers differently each time cannot be reasoned about.
- **bounded.** `MAX_HOPS` and a visited cap, and running out of either is reported as *no path within
  N hops*, not as "unrelated" — those are different claims.
- **the weakest link is stated.** A path of three strong edges and one 0.05 edge is not a strong
  connection; `weakest` names the edge the conclusion depends on.
- **hubs are named as hubs.** A path that routes through the 5,362-memory component is a path through
  everything; saying so is more useful than a hop count.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

MAX_HOPS = 6
MAX_VISITED = 20000
HUB_DEGREE = 24


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def neighbours(store: Any, memory_id: str) -> list[dict[str, Any]]:
    """Every edge touching a memory, in a deterministic order."""
    rows = _rows(
        store,
        "SELECT src_id, dst_id, weight, relation FROM edges "
        "WHERE src_id = ? OR dst_id = ? ORDER BY src_id, dst_id",
        (str(memory_id), str(memory_id)),
    )
    out = []
    for row in rows:
        left, right = str(row["src_id"]), str(row["dst_id"])
        other = right if left == str(memory_id) else left
        out.append({"id": other, "weight": round(float(row.get("weight") or 0.0), 4),
                    "relation": str(row.get("relation") or "")})
    out.sort(key=lambda item: item["id"])
    return out


def degree(store: Any, memory_id: str) -> int:
    rows = _rows(
        store,
        "SELECT COUNT(*) AS count FROM edges WHERE src_id = ? OR dst_id = ?",
        (str(memory_id), str(memory_id)),
    )
    return int(rows[0]["count"] or 0) if rows else 0


@dataclass
class Path:
    found: bool = False
    nodes: list[str] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    hops: int = 0
    weakest: float | None = None
    through_hubs: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "nodes": self.nodes,
            "edges": self.edges,
            "hops": self.hops,
            "weakest": self.weakest,
            "through_hubs": self.through_hubs,
            "reason": self.reason,
            "sentence": self.describe(),
        }

    def describe(self) -> str:
        if not self.found:
            return self.reason or "no path"
        if self.hops == 0:
            return "the same memory"
        weakest = f", weakest link {self.weakest:.3f}" if self.weakest is not None else ""
        hubs = (f", through {len(self.through_hubs)} hub(s)" if self.through_hubs else "")
        return f"{self.hops} hop(s){weakest}{hubs}"


def find(store: Any, source: str, target: str, *, max_hops: int = MAX_HOPS,
         max_visited: int = MAX_VISITED) -> Path:
    """The shortest recorded chain between two memories."""
    start, goal = str(source), str(target)
    found = Path()
    if start == goal:
        found.found, found.nodes = True, [start]
        return found
    if not start or not goal:
        found.reason = "two memory ids are needed"
        return found
    queue: deque[tuple[str, list[str], list[dict[str, Any]]]] = deque([(start, [start], [])])
    seen = {start}
    while queue:
        node, trail, edges = queue.popleft()
        if len(trail) - 1 >= int(max_hops):
            continue
        for edge in neighbours(store, node):
            step = edge["id"]
            if step in seen:
                continue
            seen.add(step)
            path_edges = edges + [{"from": node, "to": step, "weight": edge["weight"],
                                   "relation": edge["relation"]}]
            if step == goal:
                found.found = True
                found.nodes = trail + [step]
                found.edges = path_edges
                found.hops = len(path_edges)
                weights = [item["weight"] for item in path_edges if item["weight"]]
                found.weakest = round(min(weights), 4) if weights else None
                found.through_hubs = [
                    item for item in found.nodes[1:-1]
                    if degree(store, item) >= HUB_DEGREE
                ]
                return found
            if len(seen) >= int(max_visited):
                found.reason = (f"stopped after visiting {max_visited} memories without reaching "
                                f"the target; that is not the same as unrelated")
                return found
            queue.append((step, trail + [step], path_edges))
    found.reason = (f"no path within {int(max_hops)} hop(s); that is not the same as unrelated")
    return found


def between(store: Any, memory_ids: list[str], *, max_hops: int = MAX_HOPS) -> list[dict[str, Any]]:
    """Paths between consecutive memories in a list — what makes a result set hang together."""
    pairs = []
    for left, right in zip(memory_ids, memory_ids[1:]):
        found = find(store, left, right, max_hops=max_hops)
        pairs.append({"from": str(left), "to": str(right), **found.as_dict()})
    return pairs


def render(path: Path, *, limit: int = 8) -> list[str]:
    """Terminal lines: the chain, one arrow per hop, with the weight on the arrow."""
    if not path.found:
        return [path.reason or "no path"]
    if path.hops == 0:
        return ["the same memory"]
    lines = [f"{path.nodes[0][:8]}"]
    for edge in path.edges[: int(limit)]:
        lines.append(f"  → {edge['to'][:8]} ({edge['weight']:.3f})")
    if len(path.edges) > limit:
        lines.append(f"  … {len(path.edges) - limit} more hop(s)")
    lines.append(path.describe())
    if path.through_hubs:
        lines.append("  note: this route passes through a hub, so it is mostly about the hub")
    return lines
