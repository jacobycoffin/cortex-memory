"""Positions computed once, on the server, and read by everything (#162)

A force-directed layout is a simulation: the client runs it, it takes seconds, it gives a different
picture each time, and it costs battery on the device least able to spare it. For a graph whose job is
"where is this memory relative to its neighbours", none of that is necessary.

So the layout is computed **once, deterministically, and stored**:

- **connected components** are found with a union-find over `edges`;
- **components are laid out in a deterministic grid** — the largest first, ties broken by the smallest
  member id — so the same store always produces the same picture;
- **inside a component**, nodes sit on concentric rings by degree: the most connected memory at the
  centre, the next at radius 1, and so on. That is not a measurement of meaning; it is a stable,
  readable arrangement, and it says so;
- **isolated memories get no position.** A memory with no edges has nothing to be near, and inventing
  coordinates for it would fill the picture with dots that mean nothing.

Two rules keep this honest:

- **a position is a convenience, not a claim.** Nothing about a memory is encoded in its x/y; the
  layout answers "where does this sit in the picture", never "how important is it". Degree is shown
  because it is a fact, not because it is a rank.
- **the cache states its age.** `positions()` returns `computed_at` and `stale` counts, because a
  layout from three weeks ago describes a graph that no longer exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

TABLE = "graph_layout"
RING_SPACING = 120.0
COMPONENT_GAP = 600.0
COMPONENTS_PER_ROW = 4
STALE_AFTER_HOURS = 24


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001 - a layout must never break a reader
        return []


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    memory_id TEXT PRIMARY KEY,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    component INTEGER NOT NULL DEFAULT 0,
                    degree INTEGER NOT NULL DEFAULT 0,
                    component_size INTEGER NOT NULL DEFAULT 0,
                    computed_at TEXT NOT NULL
                )"""
        )


def _edges(store: Any) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for row in _rows(store, "SELECT src_id, dst_id FROM edges"):
        left, right = str(row["src_id"]), str(row["dst_id"])
        if left and right and left != right:
            pairs.append((left, right))
    return pairs


def components(store: Any) -> list[list[str]]:
    """Connected components, largest first — a union-find, so it is one pass over the edges."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            # The smaller id wins as the root, so the result does not depend on edge order.
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

    for left, right in _edges(store):
        union(left, right)
    groups: dict[str, list[str]] = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    found = [sorted(members) for members in groups.values()]
    found.sort(key=lambda members: (-len(members), members[0]))
    return found


def compute(store: Any, *, at: datetime | None = None) -> dict[str, Any]:
    """Compute and store positions. Deterministic: the same graph gives the same coordinates."""
    moment = at or _now()
    _ensure(store)
    groups = components(store)
    degree: dict[str, int] = {}
    for left, right in _edges(store):
        degree[left] = degree.get(left, 0) + 1
        degree[right] = degree.get(right, 0) + 1
    rows: list[tuple] = []
    for index, members in enumerate(groups):
        column, row = index % COMPONENTS_PER_ROW, index // COMPONENTS_PER_ROW
        centre_x = column * COMPONENT_GAP
        centre_y = row * COMPONENT_GAP
        ordered = sorted(members, key=lambda node: (-degree.get(node, 0), node))
        for position, node in enumerate(ordered):
            ring = int(math.isqrt(position)) if position else 0
            radius = RING_SPACING * ring
            if ring == 0:
                x = y = 0.0
            else:
                # Distribute around the ring by the position inside it, which is stable because the
                # order above is stable: no randomness anywhere in this function.
                within = position - ring * ring
                per_ring = max(1, (ring + 1) * (ring + 1) - ring * ring)
                angle = 2.0 * math.pi * within / per_ring
                x, y = radius * math.cos(angle), radius * math.sin(angle)
            rows.append((node, round(centre_x + x, 2), round(centre_y + y, 2), index,
                         degree.get(node, 0), len(members), moment.isoformat()))
    with store.transaction() as conn:
        conn.execute(f"DELETE FROM {TABLE}")
        conn.executemany(
            f"""INSERT INTO {TABLE}
                (memory_id, x, y, component, degree, component_size, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
    return {
        "placed": len(rows),
        "components": len(groups),
        "largest": len(groups[0]) if groups else 0,
        "isolated_excluded": True,
        "computed_at": moment.isoformat(),
        "note": "a position is a convenience, not a claim about meaning",
    }


def positions(store: Any, *, stale_after_hours: int = STALE_AFTER_HOURS,
              now: datetime | None = None) -> dict[str, Any]:
    """The stored layout, with its age stated rather than assumed current."""
    _ensure(store)
    rows = _rows(store, f"SELECT * FROM {TABLE}")
    if not rows:
        return {"placed": 0, "computed_at": "", "age_hours": None, "stale": True,
                "points": [], "note": "nothing has been placed yet"}
    moment = now or _now()
    computed_at = str(rows[0]["computed_at"] or "")
    age_hours = None
    try:
        age_hours = round((moment - datetime.fromisoformat(computed_at)).total_seconds() / 3600, 2)
    except Exception:  # noqa: BLE001
        age_hours = None
    return {
        "placed": len(rows),
        "components": len({int(row["component"]) for row in rows}),
        "computed_at": computed_at,
        "age_hours": age_hours,
        "stale": age_hours is None or age_hours > float(stale_after_hours),
        "points": [
            {"id": str(row["memory_id"]), "x": float(row["x"]), "y": float(row["y"]),
             "degree": int(row["degree"])}
            for row in sorted(rows, key=lambda item: str(item["memory_id"]))
        ],
        "note": "a position is a convenience, not a claim about meaning",
    }


def refresh_if_stale(store: Any, *, stale_after_hours: int = STALE_AFTER_HOURS,
                     now: datetime | None = None) -> dict[str, Any]:
    """Recompute only when the stored layout has aged out."""
    current = positions(store, stale_after_hours=stale_after_hours, now=now)
    if current["placed"] and not current["stale"]:
        return {**current, "recomputed": False}
    return {**compute(store, at=now), "recomputed": True}


def neighbours(store: Any, memory_id: str, *, limit: int = 8) -> list[dict[str, Any]]:
    """The placed neighbours of one memory, which is what a drill-down actually needs."""
    _ensure(store)
    rows = _rows(
        store,
        f"SELECT src_id, dst_id FROM edges WHERE src_id = ? OR dst_id = ? LIMIT ?",
        (str(memory_id), str(memory_id), int(limit)),
    )
    other_ids = [str(row["dst_id"]) if str(row["src_id"]) == str(memory_id) else str(row["src_id"])
                 for row in rows]
    if not other_ids:
        return []
    marks = ",".join("?" for _ in other_ids)
    placed = _rows(store, f"SELECT * FROM {TABLE} WHERE memory_id IN ({marks})", tuple(other_ids))
    by_id = {str(row["memory_id"]): row for row in placed}
    return [
        {"id": other, "x": float(by_id[other]["x"]), "y": float(by_id[other]["y"])}
        for other in other_ids if other in by_id
    ]
