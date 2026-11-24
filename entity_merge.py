"""Two ids for one thing, collapsed, with the alias trail kept (#63).

Entity resolution guesses, and sometimes it guesses twice: "CT116", "t3code" and "the t3 box" end up
as three rows, three alias sets and three separate histories of what is really one machine. Fixing
that by hand in SQL is exactly the kind of repair that quietly loses a name, so the store does it —
and the shape of the fix is the whole argument:

- **shown before it happens.** `propose()` returns both sides' aliases, both nodes and every memory
  the merge would touch, so the reader can see the damage before consenting;
- **consent is a name.** `apply()` refuses without a non-empty `by`; a merge that happened because a
  caller forgot a flag is indistinguishable from a bug;
- **nothing is deleted.** Alias rows move, memories keep every other name they carried, and the
  `entity_merges` row stores the pre-merge alias rows and entity lists, so `undo()` can put them
  back. The alias trail is the product, not a side effect.

What this deliberately does not do: merge nodes. A node is a summary somebody wrote, and combining
two summaries automatically invents a third. `propose()` reports both so a human can rewrite the
kept one; the losing node stays as it was, still reachable, which is why undo does not need to
restore it.

`undo()` is honest about its limits. Alias rows move back exactly as recorded. A memory that was
edited after the merge gets its source name back, but a name the person removed by hand in the
meantime is not re-added beyond the source id, and alias hits earned on the target after the merge
stay with the target.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

TABLE = "entity_merges"
ALIAS_TABLE = "entity_aliases"

_ARTICLES = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_NOISE = re.compile(r"[\s_-]+")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aliases() -> Any:
    """The alias module, when this branch has it.

    #39 owns `entity_aliases`; the merge must call *its* `merge()` rather than re-implement the
    rules. On a branch where that module has not landed the merge still has to work, so the few
    functions it needs are mirrored below and used only until the import succeeds.
    """
    try:
        from . import aliases as module

        return module
    except Exception:  # noqa: BLE001 - a missing sibling must not break the merge tool
        return None


def _nodes() -> Any:
    try:
        from . import entity_nodes as module

        return module
    except Exception:  # noqa: BLE001
        return None


def _key(text: Any) -> str:
    module = _aliases()
    if module is not None:
        return str(module.key(text))
    value = str(text or "").strip().strip(".,;:!?\"'()[]")
    value = _ARTICLES.sub("", value)
    value = value.casefold()
    return _NOISE.sub(" ", value).strip()


def _alias_table() -> str:
    module = _aliases()
    return str(getattr(module, "TABLE", ALIAS_TABLE)) if module is not None else ALIAS_TABLE


def ensure(store: Any) -> None:
    """Create the merge log, and the alias table only when its owner is not on this branch."""
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    merge_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    aliases_moved INTEGER NOT NULL DEFAULT 0,
                    aliases_dropped INTEGER NOT NULL DEFAULT 0,
                    memories_touched INTEGER NOT NULL DEFAULT 0,
                    alias_rows_json TEXT NOT NULL DEFAULT '{{}}',
                    memory_rows_json TEXT NOT NULL DEFAULT '[]',
                    by TEXT NOT NULL,
                    at TEXT NOT NULL,
                    undone_at TEXT
                )"""
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_at ON {TABLE}(at)"
        )
    if _aliases() is None:
        with store.transaction() as conn:
            conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {ALIAS_TABLE} (
                        entity_id TEXT NOT NULL,
                        alias_key TEXT NOT NULL,
                        alias TEXT NOT NULL,
                        source TEXT,
                        hits INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (entity_id, alias_key)
                    )"""
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{ALIAS_TABLE}_key ON {ALIAS_TABLE}(alias_key)"
            )


def _alias_rows(store: Any, entity_id: str) -> list[dict[str, Any]]:
    """Every alias row for one entity, as plain dicts, ordered by key for a stable diff."""
    try:
        rows = store._conn.execute(
            f"""SELECT entity_id, alias_key, alias, source, hits, created_at, updated_at
                FROM {_alias_table()} WHERE entity_id = ? ORDER BY alias_key""",
            (str(entity_id),),
        ).fetchall()
    except Exception:  # noqa: BLE001 - a read that cannot fail the caller
        return []
    return [dict(row) for row in rows]


def _node(store: Any, entity_id: str) -> dict[str, Any] | None:
    module = _nodes()
    if module is not None:
        try:
            node = module.get(store, entity_id)
        except Exception:  # noqa: BLE001
            return None
        return node.as_dict() if node is not None else None
    try:
        row = store._conn.execute(
            "SELECT * FROM entity_nodes WHERE entity_id = ?", (str(entity_id),)
        ).fetchone()
    except Exception:  # noqa: BLE001
        return None
    return dict(row) if row else None


def _load_names(raw: Any) -> list[str]:
    try:
        loaded = json.loads(str(raw))
    except (TypeError, ValueError):
        return []
    if not isinstance(loaded, list):
        return []
    return [str(name) for name in loaded if str(name).strip()]


def _memory_lists(store: Any) -> list[dict[str, Any]]:
    """(id, names) for every memory with a readable entity list."""
    try:
        rows = store._conn.execute(
            "SELECT id, entities_json FROM memories WHERE json_valid(entities_json)"
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append({"id": str(row["id"]), "before": _load_names(row["entities_json"])})
    return out


def _source_keys(store: Any, source_id: str, alias_rows: Iterable[dict[str, Any]]) -> set[str]:
    keys = {_key(source_id)}
    for row in alias_rows:
        keys.add(_key(row["alias"]))
        keys.add(str(row["alias_key"]))
    keys.discard("")
    return keys


def _touches(names: Iterable[str], keys: set[str]) -> bool:
    return any(_key(name) in keys for name in names)


@dataclass
class Plan:
    source_id: str
    target_id: str
    source_aliases: list[str] = field(default_factory=list)
    target_aliases: list[str] = field(default_factory=list)
    source_node: dict[str, Any] | None = None
    target_node: dict[str, Any] | None = None
    memories_touched: int = 0
    conflicts: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """One line a person can paste into a question before saying yes."""
        return (
            f"merge {self.source_id} into {self.target_id}: "
            f"{len(self.source_aliases)} alias(es) move, "
            f"{self.memories_touched} memory(ies) re-point, "
            f"{len(self.conflicts)} conflict(s) to review"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "aliases_before": {
                "source": list(self.source_aliases),
                "target": list(self.target_aliases),
            },
            "nodes": {"source": self.source_node, "target": self.target_node},
            "memories_touched": int(self.memories_touched),
            "conflicts": list(self.conflicts),
            "sentence": self.describe(),
        }


def _conflicts(source_node: dict[str, Any] | None, target_node: dict[str, Any] | None) -> list[str]:
    """The attributes a person has to settle before the merge reads as one thing."""
    found: list[str] = []
    if source_node and target_node:
        source_kind = str(source_node.get("kind") or "").strip().lower()
        target_kind = str(target_node.get("kind") or "").strip().lower()
        if source_kind and target_kind and source_kind != target_kind:
            found.append(f"kind: source is '{source_kind}', target is '{target_kind}'")
        source_summary = str(source_node.get("summary") or "").strip()
        target_summary = str(target_node.get("summary") or "").strip()
        if source_summary and target_summary and source_summary != target_summary:
            found.append("summary: both sides carry one; the target's is kept")
    return found


def propose(store: Any, source_id: str, target_id: str) -> dict[str, Any]:
    """The reviewable diff: both sides' names, both nodes, and what the merge would touch."""
    source = str(source_id or "").strip()
    target = str(target_id or "").strip()
    try:
        ensure(store)
        source_rows = _alias_rows(store, source)
        target_rows = _alias_rows(store, target)
        keys = _source_keys(store, source, source_rows)
        touched = sum(
            1 for item in _memory_lists(store) if _touches(item["before"], keys)
        )
        source_node = _node(store, source)
        target_node = _node(store, target)
        plan = Plan(
            source_id=source,
            target_id=target,
            source_aliases=[str(row["alias"]) for row in source_rows],
            target_aliases=[str(row["alias"]) for row in target_rows],
            source_node=source_node,
            target_node=target_node,
            memories_touched=touched,
            conflicts=_conflicts(source_node, target_node),
        )
    except Exception as exc:  # noqa: BLE001 - a preview must never take the caller down
        return {
            "source_id": source,
            "target_id": target,
            "aliases_before": {"source": [], "target": []},
            "nodes": {"source": None, "target": None},
            "memories_touched": 0,
            "conflicts": [],
            "sentence": f"cannot read the store for this merge: {exc}",
        }
    if _key(source) == _key(target):
        plan.conflicts.append("same entity: source and target fold to one name")
    return plan.as_dict()


def _merge_aliases(store: Any, source: str, target: str) -> dict[str, int]:
    """Re-point the alias rows, through #39's `merge()` when it is importable."""
    module = _aliases()
    if module is not None:
        return module.merge(store, source, target)
    # Bridge for a branch without the alias module: the same rule as aliases.merge — move each
    # row, add its hits where the target already carries the name, then drop the emptied side.
    moved = 0
    dropped = 0
    with store.transaction() as conn:
        rows = conn.execute(
            f"SELECT alias_key, hits FROM {ALIAS_TABLE} WHERE entity_id = ?", (source,)
        ).fetchall()
        for row in rows:
            existing = conn.execute(
                f"SELECT hits FROM {ALIAS_TABLE} WHERE entity_id = ? AND alias_key = ?",
                (target, str(row["alias_key"])),
            ).fetchone()
            if existing is not None:
                conn.execute(
                    f"UPDATE {ALIAS_TABLE} SET hits = hits + ? WHERE entity_id = ? AND alias_key = ?",
                    (int(row["hits"]), target, str(row["alias_key"])),
                )
                dropped += 1
                continue
            conn.execute(
                f"UPDATE {ALIAS_TABLE} SET entity_id = ? WHERE entity_id = ? AND alias_key = ?",
                (target, source, str(row["alias_key"])),
            )
            moved += 1
        conn.execute(f"DELETE FROM {ALIAS_TABLE} WHERE entity_id = ?", (source,))
    return {"moved": moved, "dropped": dropped}


def _repoint_memories(
    store: Any, source_id: str, target_id: str, keys: set[str]
) -> list[dict[str, Any]]:
    """Replace the source's names with the target id, deduplicated, everything else untouched."""
    touched: list[dict[str, Any]] = []
    with store.transaction() as conn:
        for item in _memory_lists(store):
            names = item["before"]
            if not _touches(names, keys):
                continue
            out: list[str] = []
            for name in names:
                value = target_id if _key(name) in keys else str(name)
                if value and value not in out:
                    out.append(value)
            if out == names:
                continue
            conn.execute(
                "UPDATE memories SET entities_json = ? WHERE id = ?",
                (json.dumps(out, ensure_ascii=False), item["id"]),
            )
            touched.append({"id": item["id"], "before": names, "after": out})
    return touched


def apply(
    store: Any,
    source_id: str,
    target_id: str,
    *,
    by: str = "",
    at: datetime | None = None,
) -> dict[str, Any]:
    """Collapse the source into the target, but only when a person has signed for it."""
    source = str(source_id or "").strip()
    target = str(target_id or "").strip()
    consent = str(by or "").strip()
    if not source or not target:
        return {"merged": False, "reason": "both entity ids are required"}
    if _key(source) == _key(target):
        return {"merged": False, "reason": "an entity cannot be merged into itself"}
    if not consent:
        return {
            "merged": False,
            "reason": "consent required: apply() needs the name of the person merging: nothing changed",
        }
    try:
        ensure(store)
        source_rows = _alias_rows(store, source)
        target_rows = _alias_rows(store, target)
        keys = _source_keys(store, source, source_rows)
        merge_id = uuid.uuid4().hex[:12]
        stamp = (at or _now()).isoformat()
        with store.transaction() as conn:
            result = _merge_aliases(store, source, target)
            touched = _repoint_memories(store, source, target, keys)
            conn.execute(
                f"""INSERT INTO {TABLE}
                    (merge_id, source_id, target_id, aliases_moved, aliases_dropped,
                     memories_touched, alias_rows_json, memory_rows_json, by, at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    merge_id,
                    source,
                    target,
                    int(result.get("moved", 0)),
                    int(result.get("dropped", 0)),
                    len(touched),
                    json.dumps({"source": source_rows, "target": target_rows}, ensure_ascii=False),
                    json.dumps(touched, ensure_ascii=False),
                    consent,
                    stamp,
                ),
            )
    except Exception as exc:  # noqa: BLE001 - a failed merge rolls back and says so
        return {"merged": False, "reason": f"merge failed and was rolled back: {exc}"}
    return {
        "merged": True,
        "merge_id": merge_id,
        "source_id": source,
        "target_id": target,
        "aliases_moved": int(result.get("moved", 0)),
        "aliases_dropped": int(result.get("dropped", 0)),
        "memories_touched": len(touched),
        "sentence": (
            f"merged {source} into {target}: {int(result.get('moved', 0))} alias(es) moved, "
            f"{len(touched)} memory(ies) re-pointed; undo with {merge_id}"
        ),
    }


def history(store: Any, limit: int = 10) -> list[dict[str, Any]]:
    """The merge log, newest first, including the rows that have been undone."""
    try:
        ensure(store)
        rows = store._conn.execute(
            f'SELECT * FROM {TABLE} ORDER BY "at" DESC, rowid DESC LIMIT ?', (int(limit),)
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    return [dict(row) for row in rows]


def _snapshot_keys(source_id: str, snapshot: dict[str, Any]) -> set[str]:
    keys = {_key(source_id)}
    for row in snapshot.get("source") or []:
        keys.add(_key(row.get("alias")))
    keys.discard("")
    return keys


def _restored_names(current: list[str], before: list[str], source_keys: set[str]) -> list[str]:
    """Put back every recorded name that the merge collapsed, without inventing new ones.

    Used only when the list has changed since the merge (the exact case restores the recorded
    list verbatim). A name is restored at the end, because the position it held before the merge
    is no longer recoverable once the person has edited the list.
    """
    out = list(current)
    present = {_key(name) for name in out}
    for name in before:
        alias_key = _key(name)
        if alias_key in source_keys and alias_key not in present:
            out.append(name)
            present.add(alias_key)
    return out


def _restore_aliases(store: Any, snapshot: dict[str, Any], target_id: str) -> int:
    source_rows = list(snapshot.get("source") or [])
    target_rows = {str(row["alias_key"]): row for row in (snapshot.get("target") or [])}
    table = _alias_table()
    restored = 0
    with store.transaction() as conn:
        for row in source_rows:
            alias_key = str(row["alias_key"])
            entity_id = str(row["entity_id"])
            if alias_key in target_rows:
                # The source row was folded into an existing target row; give its hits back.
                conn.execute(
                    f"""UPDATE {table} SET hits = MAX(0, hits - ?)
                        WHERE entity_id = ? AND alias_key = ?""",
                    (int(row["hits"]), str(target_rows[alias_key]["entity_id"]), alias_key),
                )
                conn.execute(
                    f"""INSERT OR REPLACE INTO {table}
                        (entity_id, alias_key, alias, source, hits, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entity_id,
                        alias_key,
                        str(row["alias"]),
                        row.get("source"),
                        int(row["hits"]),
                        str(row["created_at"]),
                        str(row["updated_at"]),
                    ),
                )
                restored += 1
                continue
            existing = conn.execute(
                f"SELECT entity_id FROM {table} WHERE alias_key = ? AND entity_id = ?",
                (alias_key, target_id),
            ).fetchone()
            if existing is not None:
                # `apply()` only moves rows, so the row can go home with its hits intact.
                conn.execute(
                    f"UPDATE {table} SET entity_id = ? WHERE alias_key = ? AND entity_id = ?",
                    (entity_id, alias_key, str(existing["entity_id"])),
                )
                restored += 1
                continue
            conn.execute(
                f"""INSERT OR REPLACE INTO {table}
                    (entity_id, alias_key, alias, source, hits, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    entity_id,
                    alias_key,
                    str(row["alias"]),
                    row.get("source"),
                    int(row["hits"]),
                    str(row["created_at"]),
                    str(row["updated_at"]),
                ),
            )
            restored += 1
    return restored


def undo(store: Any, merge_id: str | None = None, *, at: datetime | None = None) -> dict[str, Any]:
    """Put the names and lists back from the recorded row; a second undo refuses."""
    rows = history(store, limit=1000)
    if merge_id is not None:
        matches = [row for row in rows if str(row["merge_id"]) == str(merge_id)]
        if not matches:
            return {"undone": False, "reason": f"no merge {merge_id!r} in the log"}
        entry = matches[0]
        if entry.get("undone_at"):
            return {
                "undone": False,
                "merge_id": str(entry["merge_id"]),
                "reason": "that merge was already undone; undo restores once",
            }
    else:
        pending = [row for row in rows if not row.get("undone_at")]
        if not pending:
            if rows:
                return {
                    "undone": False,
                    "reason": "the most recent merge was already undone; undo restores once",
                }
            return {"undone": False, "reason": "no merge to undo"}
        entry = pending[0]

    try:
        snapshot = json.loads(str(entry.get("alias_rows_json") or "{}"))
        touched = json.loads(str(entry.get("memory_rows_json") or "[]"))
    except (TypeError, ValueError):
        return {"undone": False, "reason": "the recorded merge row is unreadable"}
    if not isinstance(snapshot, dict):
        snapshot = {}
    source_id = str(entry["source_id"])
    target_id = str(entry["target_id"])
    try:
        ensure(store)
        memories_restored = 0
        # One outer transaction: aliases and entity lists go back together or not at all.
        with store.transaction():
            aliases_restored = _restore_aliases(store, snapshot, target_id)
            with store.transaction() as conn:
                source_keys = _snapshot_keys(source_id, snapshot)
                for item in touched if isinstance(touched, list) else []:
                    memory_id = str(item.get("id") or "")
                    if not memory_id:
                        continue
                    row = conn.execute(
                        "SELECT entities_json FROM memories WHERE id = ?", (memory_id,)
                    ).fetchone()
                    if row is None:
                        continue
                    current = _load_names(row["entities_json"])
                    before = [str(name) for name in (item.get("before") or [])]
                    after = item.get("after")
                    if isinstance(after, list) and [str(name) for name in after] == current:
                        # Untouched since the merge: the recorded list is the exact prior state.
                        out = before
                    else:
                        out = _restored_names(current, before, source_keys)
                    if out == current:
                        continue
                    conn.execute(
                        "UPDATE memories SET entities_json = ? WHERE id = ?",
                        (json.dumps(out, ensure_ascii=False), memory_id),
                    )
                    memories_restored += 1
                conn.execute(
                    f"UPDATE {TABLE} SET undone_at = ? WHERE merge_id = ?",
                    ((at or _now()).isoformat(), str(entry["merge_id"])),
                )
    except Exception as exc:  # noqa: BLE001
        return {"undone": False, "reason": f"undo failed and was rolled back: {exc}"}
    return {
        "undone": True,
        "merge_id": str(entry["merge_id"]),
        "aliases_restored": aliases_restored,
        "memories_restored": memories_restored,
        "sentence": (
            f"restored merge {entry['merge_id']}: {aliases_restored} alias row(s) and "
            f"{memories_restored} entity list(s) back to {source_id}"
        ),
    }


def report(store: Any) -> dict[str, Any]:
    """How much merging this store has, and how much of it can still be taken back."""
    rows = history(store, limit=5000)
    return {
        "merges": len(rows),
        "undoable": sum(1 for row in rows if not row.get("undone_at")),
        "undone": sum(1 for row in rows if row.get("undone_at")),
        "aliases_moved": sum(int(row.get("aliases_moved") or 0) for row in rows),
        "memories_touched": sum(int(row.get("memories_touched") or 0) for row in rows),
        "note": "alias rows and entity lists are recorded before the merge, so undo restores them",
    }
