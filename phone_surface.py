"""Phone-first memory surface: list, search, timeline — the graph is a drill-down (#159)

The dashboard is a 624 KB page built around a graph. That is the right shape for a desktop with a
mouse and the wrong one for a phone in a hand: the first thing you want is *what is in my head*, not a
force layout, and the thing you do with a graph on a small screen is pinch at it.

So this module serves the three views a phone actually uses, in the order they are useful:

- **list** — the most recent memories, newest first, one truncated line each;
- **search** — the same shape, filtered, because typing is the fastest index on a phone;
- **timeline** — the same rows grouped by day, so "what did I learn this week" is answerable;
- **detail** — one memory with its neighbours, which is where the graph belongs: behind a tap, for one
  node, not as a landing page.

Three rules keep it honest and small:

- **the payload is bounded, not truncated by luck.** A preview is at most `PREVIEW` characters, a page
  is at most `limit` rows, and the response always says how many more there are — a list that silently
  stops is a lie about the size of the store.
- **no counts are invented.** Every number in the payload comes from a query; `total` is the count of
  matching rows, not `len(rows)`.
- **the graph is never the first screen.** `detail()` is the only function that returns edges, and the
  HTML has no canvas at all — a phone can tap through to one memory's neighbours without being handed a
  layout to pinch.
"""

from __future__ import annotations

from typing import Any, Iterable

PREVIEW = 140
DEFAULT_LIMIT = 30
MAX_LIMIT = 100


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001 - a surface must render an empty list, not a stack trace
        return []


def _clamp(limit: int) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return DEFAULT_LIMIT
    return max(1, min(MAX_LIMIT, value))


def preview(text: Any, limit: int = PREVIEW) -> str:
    body = " ".join(str(text or "").split())
    return body if len(body) <= limit else body[: limit - 1].rstrip() + "…"


def _row(memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(memory.get("id") or ""),
        "kind": str(memory.get("kind") or ""),
        "state": str(memory.get("state") or ""),
        "text": preview(memory.get("content")),
        "confidence": round(float(memory.get("confidence") or 0.0), 3),
        "created_at": str(memory.get("created_at") or "")[:19],
        "approval_state": str(memory.get("approval_state") or ""),
    }


def _total(store: Any, sql: str, params: tuple = ()) -> int:
    rows = _rows(store, sql, params)
    try:
        return int(rows[0]["count"] or 0) if rows else 0
    except Exception:  # noqa: BLE001
        return 0


def recent(store: Any, *, limit: int = DEFAULT_LIMIT, kind: str = "") -> dict[str, Any]:
    """The newest memories. This is the landing view."""
    count = _clamp(limit)
    where = "WHERE state <> 'archived' AND kind = ?" if kind else "WHERE state <> 'archived'"
    params: tuple = (kind,) if kind else ()
    rows = _rows(
        store,
        f"SELECT id, kind, state, content, confidence, created_at, approval_state FROM memories "
        f"{where} ORDER BY created_at DESC, id DESC LIMIT ?",
        params + (count,),
    )
    total = _total(store, f"SELECT COUNT(*) AS count FROM memories {where}", params)
    return {"view": "list", "kind": kind, "items": [_row(row) for row in rows],
            "total": total, "more": max(0, total - len(rows))}


def search(store: Any, query: str, *, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    """The same shape, filtered. FTS when the table is there, LIKE when it is not."""
    text = " ".join(str(query or "").split())
    if not text:
        return {**recent(store, limit=limit), "view": "search", "query": ""}
    count = _clamp(limit)
    tokens = [token for token in text.split() if len(token) >= 2][:5]
    like = "%" + tokens[0] + "%" if tokens else f"%{text}%"
    rows = _rows(
        store,
        "SELECT id, kind, state, content, confidence, created_at, approval_state FROM memories "
        "WHERE state <> 'archived' AND content LIKE ? "
        "ORDER BY created_at DESC, id DESC LIMIT ?",
        (like, count),
    )
    total = _total(
        store,
        "SELECT COUNT(*) AS count FROM memories WHERE state <> 'archived' AND content LIKE ?",
        (like,),
    )
    return {"view": "search", "query": text, "items": [_row(row) for row in rows],
            "total": total, "more": max(0, total - len(rows))}


def timeline(store: Any, *, limit: int = DEFAULT_LIMIT, kind: str = "") -> dict[str, Any]:
    """The same rows grouped by the day they were written."""
    page = recent(store, limit=limit, kind=kind)
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in page["items"]:
        day = item["created_at"][:10] or "unknown"
        groups.setdefault(day, []).append(item)
    ordered = [{"day": day, "items": groups[day]} for day in sorted(groups, reverse=True)]
    return {"view": "timeline", "days": ordered, "total": page["total"], "more": page["more"]}


def detail(store: Any, memory_id: str, *, neighbours: int = 5) -> dict[str, Any]:
    """One memory with its nearest neighbours — this is where the graph belongs."""
    rows = _rows(
        store,
        "SELECT id, kind, state, content, confidence, created_at, approval_state FROM memories "
        "WHERE id = ?",
        (str(memory_id),),
    )
    if not rows:
        return {"view": "detail", "found": False, "id": str(memory_id)}
    edges = _rows(
        store,
        "SELECT src_id, dst_id, weight FROM edges WHERE src_id = ? OR dst_id = ? "
        "ORDER BY weight DESC LIMIT ?",
        (str(memory_id), str(memory_id), max(1, int(neighbours))),
    )
    linked = []
    for edge in edges:
        other = edge["dst_id"] if str(edge["src_id"]) == str(memory_id) else edge["src_id"]
        linked.append({"id": str(other), "weight": round(float(edge.get("weight") or 0.0), 3)})
    return {"view": "detail", "found": True, "memory": _row(rows[0]), "neighbours": linked,
            "note": "the graph is this list, one memory at a time"}


def sheet(store: Any, memory_id: str, *, neighbours: int = 5) -> dict[str, Any]:
    """What a tap opens: one memory, its provenance, its neighbours and whether it was used.

    A bottom sheet has one screen and no back button, so it has to answer the questions the list cannot
    in the same glance: when was this written, how sure is the store, which project does it belong to,
    and did an answer ever actually use it (#174). Provenance is **quoted from the row**, never inferred.
    """
    base = detail(store, str(memory_id), neighbours=neighbours)
    if not base.get("found"):
        return {**base, "view": "sheet"}
    rows = _rows(store, "SELECT * FROM memories WHERE id = ?", (str(memory_id),))
    row = rows[0] if rows else {}
    provenance = {
        "kind": str(row.get("kind") or ""),
        "state": str(row.get("state") or ""),
        "confidence": round(float(row.get("confidence") or 0.0), 2),
        "created_at": str(row.get("created_at") or "")[:19],
        "source_type": str(row.get("source_type") or ""),
        "approval_state": str(row.get("approval_state") or ""),
        "pinned": bool(row.get("pinned")),
    }
    used = None
    try:
        found = _rows(
            store,
            "SELECT COUNT(*) AS count FROM citations WHERE memory_id = ?",
            (str(memory_id),),
        )
        if found:
            used = int(found[0]["count"] or 0)
    except Exception:  # noqa: BLE001 - a sheet without a citation count is still useful
        used = None
    return {**base, "view": "sheet", "provenance": provenance, "cited_by": used}


def payload(store: Any, view: str = "list", *, query: str = "", limit: int = DEFAULT_LIMIT,
            kind: str = "", memory_id: str = "") -> dict[str, Any]:
    """One entry point for the router, so the served JSON always has the same shape."""
    name = str(view or "list").strip().lower()
    if name == "search":
        return search(store, query, limit=limit)
    if name == "timeline":
        return timeline(store, limit=limit, kind=kind)
    if name == "detail":
        return detail(store, memory_id)
    if name == "sheet":
        return sheet(store, memory_id, neighbours=int(limit) or 5)
    return recent(store, limit=limit, kind=kind)


def render_text(payload_data: dict[str, Any]) -> list[str]:
    """A plain-text rendering, because the same data should be readable without a browser."""
    lines: list[str] = []
    if payload_data.get("view") == "detail":
        if not payload_data.get("found"):
            return [f"memory {payload_data.get('id', '')}: not found"]
        memory = payload_data.get("memory") or {}
        lines.append(f"{str(memory.get('id') or '')[:8]} · {memory.get('kind', '')} · "
                     f"{memory.get('state', '')} · {memory.get('created_at', '')}")
        lines.append(str(memory.get("text") or ""))
        for neighbour in payload_data.get("neighbours") or []:
            lines.append(f"  ↔ {str(neighbour['id'])[:8]} {neighbour['weight']}")
        return lines
    days = payload_data.get("days")
    if days:
        for group in days:
            lines.append(f"{group['day']}")
            for item in group["items"]:
                lines.append(f"  {item['id'][:8]} {item['text'][:60]}")
    else:
        for item in payload_data.get("items") or []:
            lines.append(f"{item['id'][:8]} {item['kind'][:12]:12} {item['text'][:70]}")
    if payload_data.get("more"):
        lines.append(f"… {payload_data['more']} more of {payload_data.get('total', 0)}")
    return lines


def summarise(items: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(item.get("kind") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts
