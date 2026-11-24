"""Saved views: a filter with a name, never a script (#163)

On a phone, typing a question is the slow path. What people actually do is open the same four or five
lists over and over — what is new, what was never used, what is waiting for review, what belongs to the
current project.

So a saved view is **a named filter**, and the whole design is about keeping it that:

- **an allowlist, not a query language.** A spec may name a kind, a project, a state, a search term, a
  sort and a limit. It may not contain SQL, a column name, or an operator. A saved view that could run
  arbitrary SQL would turn "remember this list" into "run this code", and the difference matters the
  first time a view arrives from anywhere but your own hands.
- **unknown fields are refused, not ignored.** A typo that silently disappears produces a view that is
  quietly wrong — the worst outcome, because it looks like it works.
- **a view is testable against the store it will run on.** `explain()` returns the SQL fragment it built
  and the number of rows it matches, so a view can be checked before it is trusted.
- **use is counted.** `use_count` and `last_used_at` exist so unused views are visible; a menu nobody
  opens should be deleted rather than maintained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

TABLE = "saved_views"

# The complete vocabulary of a saved view. Anything outside this is refused.
FIELDS = ("kind", "project", "state", "query", "sort", "limit", "pinned_only")
SORTS = {
    "recent": "created_at DESC, id DESC",
    "oldest": "created_at ASC, id ASC",
    "confidence": "confidence DESC, id ASC",
    "kind": "kind ASC, created_at DESC",
}
STATES = ("active", "cold", "archived", "pending", "superseded")
MAX_LIMIT = 200


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
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    name TEXT PRIMARY KEY,
                    spec_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT NOT NULL DEFAULT '',
                    use_count INTEGER NOT NULL DEFAULT 0
                )"""
        )


class SpecError(ValueError):
    """A saved view is a filter; this error says the spec tried to be more than one."""


def validate(spec: dict[str, Any]) -> dict[str, Any]:
    """Normalise a spec, refusing anything outside the vocabulary."""
    import json as _json

    if not isinstance(spec, dict):
        raise SpecError("a saved view is a mapping of the allowed fields")
    unknown = sorted(set(spec) - set(FIELDS))
    if unknown:
        raise SpecError(f"unknown field(s) {unknown}; allowed: {list(FIELDS)}")
    clean: dict[str, Any] = {}
    for key, value in spec.items():
        if value is None or value == "":
            continue
        if key == "limit":
            try:
                clean["limit"] = max(1, min(MAX_LIMIT, int(value)))
            except (TypeError, ValueError) as error:
                raise SpecError(f"limit must be a number, got {value!r}") from error
        elif key == "sort":
            if value not in SORTS:
                raise SpecError(f"sort must be one of {sorted(SORTS)}")
            clean["sort"] = value
        elif key == "state":
            if value not in STATES:
                raise SpecError(f"state must be one of {list(STATES)}")
            clean["state"] = value
        elif key == "pinned_only":
            clean["pinned_only"] = bool(value)
        else:
            text = str(value)
            if ";" in text or "--" in text:
                # The value is going into a parameterised query, but a fragment like this in a spec is
                # a sign the author expected SQL to work. Say so instead of storing a trap.
                raise SpecError(f"{key} must be a plain value, not SQL: {text!r}")
            clean[key] = text
    _ = _json  # imported for symmetry with the reader below
    return clean


def save(store: Any, name: str, spec: dict[str, Any], *, at: datetime | None = None) -> dict[str, Any]:
    """Store a named view. The name is the identity: saving twice replaces the spec, not the history."""
    import json

    label = str(name or "").strip()
    if not label:
        raise SpecError("a view needs a name")
    clean = validate(spec)
    moment = (at or _now()).isoformat()
    _ensure(store)
    with store.transaction() as conn:
        existing = conn.execute(f"SELECT created_at FROM {TABLE} WHERE name = ?", (label,)).fetchone()
        created = str(existing["created_at"]) if existing else moment
        conn.execute(
            f"""INSERT INTO {TABLE} (name, spec_json, created_at, last_used_at, use_count)
                VALUES (?, ?, ?, '', 0)
                ON CONFLICT(name) DO UPDATE SET spec_json = excluded.spec_json""",
            (label, json.dumps(clean, sort_keys=True), created),
        )
    return {"name": label, "spec": clean, "created_at": created, "replaced": bool(existing)}


def build_sql(spec: dict[str, Any]) -> tuple[str, tuple]:
    """The parameterised WHERE clause for a spec. Every value is a bound parameter."""
    clean = validate(spec)
    clauses: list[str] = []
    params: list[Any] = []
    if "state" in clean:
        clauses.append("state = ?")
        params.append(clean["state"])
    else:
        clauses.append("state <> 'archived'")
    if clean.get("kind"):
        clauses.append("kind = ?")
        params.append(clean["kind"])
    if clean.get("pinned_only"):
        clauses.append("pinned = 1")
    if clean.get("project"):
        clauses.append("scope_json LIKE ?")
        params.append(f'%"{clean["project"]}"%')
    if clean.get("query"):
        clauses.append("content LIKE ?")
        params.append(f'%{clean["query"]}%')
    where = " AND ".join(clauses)
    order = SORTS.get(clean.get("sort") or "recent", SORTS["recent"])
    return f"WHERE {where} ORDER BY {order}", tuple(params)


def run(store: Any, name: str, *, at: datetime | None = None) -> dict[str, Any]:
    """Run a saved view: rows in the phone surface's shape, plus the spec that produced them."""
    import json

    _ensure(store)
    rows = _rows(store, f"SELECT * FROM {TABLE} WHERE name = ?", (str(name),))
    if not rows:
        return {"name": str(name), "found": False, "items": [], "total": 0}
    spec = json.loads(rows[0]["spec_json"] or "{}")
    where, params = build_sql(spec)
    limit = int(spec.get("limit") or 30)
    items = _rows(
        store,
        f"SELECT id, kind, state, content, confidence, created_at, pinned FROM memories "
        f"{where} LIMIT ?",
        params + (limit,),
    )
    counted = _rows(store, f"SELECT COUNT(*) AS count FROM memories {where}", params)
    total = int(counted[0]["count"] or 0) if counted else 0
    moment = (at or _now()).isoformat()
    with store.transaction() as conn:
        conn.execute(
            f"UPDATE {TABLE} SET use_count = use_count + 1, last_used_at = ? WHERE name = ?",
            (moment, str(name)),
        )
    return {
        "name": str(name),
        "found": True,
        "spec": spec,
        "items": [
            {"id": str(row["id"]), "kind": str(row["kind"] or ""),
             "state": str(row["state"] or ""),
             "text": " ".join(str(row["content"] or "").split())[:140],
             "confidence": round(float(row["confidence"] or 0.0), 2),
             "created_at": str(row["created_at"] or "")[:19],
             "pinned": bool(row["pinned"])}
            for row in items
        ],
        "total": total,
        "more": max(0, total - len(items)),
    }


def explain(store: Any, name: str) -> dict[str, Any]:
    """What a view would run, and how many rows it matches — before anyone trusts it."""
    import json

    _ensure(store)
    rows = _rows(store, f"SELECT spec_json FROM {TABLE} WHERE name = ?", (str(name),))
    if not rows:
        return {"name": str(name), "found": False}
    spec = json.loads(rows[0]["spec_json"] or "{}")
    where, params = build_sql(spec)
    counted = _rows(store, f"SELECT COUNT(*) AS count FROM memories {where}", params)
    return {
        "name": str(name),
        "found": True,
        "spec": spec,
        "sql": where,
        "params": list(params),
        "matches": int(counted[0]["count"] or 0) if counted else 0,
    }


def views(store: Any) -> list[dict[str, Any]]:
    """Every saved view, least recently used first — the ones worth deleting lead the list."""
    import json

    _ensure(store)
    rows = _rows(
        store,
        f"SELECT name, spec_json, created_at, last_used_at, use_count FROM {TABLE} "
        f"ORDER BY use_count ASC, name ASC",
    )
    return [
        {"name": str(row["name"]), "spec": json.loads(row["spec_json"] or "{}"),
         "created_at": str(row["created_at"] or ""),
         "last_used_at": str(row["last_used_at"] or ""),
         "use_count": int(row["use_count"] or 0)}
        for row in rows
    ]


def delete(store: Any, name: str) -> bool:
    _ensure(store)
    with store.transaction() as conn:
        cursor = conn.execute(f"DELETE FROM {TABLE} WHERE name = ?", (str(name),))
        return bool(cursor.rowcount)


@dataclass
class Preset:
    name: str
    spec: dict[str, Any] = field(default_factory=dict)


def presets() -> list[Preset]:
    """The four views a phone actually uses, offered as a starting point rather than a menu."""
    return [
        Preset("new", {"sort": "recent", "limit": 30}),
        Preset("pinned", {"pinned_only": True, "limit": 20}),
        Preset("unsure", {"sort": "confidence", "limit": 20}),
        Preset("cold", {"state": "cold", "sort": "recent", "limit": 30}),
    ]


def install_presets(store: Any, *, at: datetime | None = None) -> list[str]:
    """Save the presets that are not already there; never overwrite a view somebody tuned."""
    existing = {row["name"] for row in views(store)}
    added: list[str] = []
    for preset in presets():
        if preset.name in existing:
            continue
        save(store, preset.name, preset.spec, at=at)
        added.append(preset.name)
    return added


def unused_views(store: Any) -> list[str]:
    """Views nobody has opened, with their age in days — the denominator of a menu that grew."""
    out = []
    moment = _now()
    for row in views(store):
        if row["use_count"]:
            continue
        age = None
        try:
            age = round((moment - datetime.fromisoformat(row["created_at"])).total_seconds() / 86400, 1)
        except Exception:  # noqa: BLE001
            age = None
        out.append({"name": row["name"], "age_days": age})
    return out


def _days_ago(days: int, *, now: datetime | None = None) -> str:
    return ((now or _now()) - timedelta(days=days)).isoformat()
