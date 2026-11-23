"""Per-entity aliases, learned from usage and editable (#39).

The same thing has many names. "CT116", "the t3 box", "the Proxmox box", "t3code" — a person says
whichever comes to mind, and a store that treats them as four entities retrieves a quarter of what it
knows. Entity resolution at write (#38) guesses from the text; this is the part that **remembers the
guess** and lets a human fix it.

Three properties matter more than the matching itself:

- **learned from usage, not from a model** — an alias is recorded when a name is used to mean the
  entity, and `hits` counts how many times, so a one-off mistake fades and a habit sticks;
- **editable in both directions** — `add`, `remove`, `merge`, `rename`. An alias table nobody can
  correct becomes a source of confident wrong answers, which is worse than no aliases;
- **normalised but not squashed** — case, punctuation, a leading "the", and spacing are folded, and
  nothing else. "CT116" and "CT-116" are the same; "ct1" and "ct11" are not.

`expand()` is the one function retrieval needs: given the words of a query, return the canonical
names to match. It never invents an entity — an alias with no canonical form stays as itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

TABLE = "entity_aliases"
MAX_ALIASES_PER_ENTITY = 24
MIN_KEY_LEN = 3
# An entity is a thing you can name out loud. A 60-character title is a document heading that
# some importer put in `entities_json`, and teaching it as an alias makes the table noise.
MAX_ALIAS_LEN = 48

_ARTICLES = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_NOISE = re.compile(r"[\s_-]+")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def key(text: str) -> str:
    """Fold case, articles, spacing and surrounding punctuation. Nothing more."""
    value = str(text or "").strip().strip(".,;:!?\"'()[]")
    value = _ARTICLES.sub("", value)
    value = value.casefold()
    return _NOISE.sub(" ", value).strip()


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
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
            f"CREATE INDEX IF NOT EXISTS idx_entity_aliases_key ON {TABLE}(alias_key)"
        )


@dataclass
class Alias:
    entity_id: str
    alias: str
    hits: int
    source: str

    def describe(self) -> str:
        return f"{self.entity_id} <- {self.alias} ({self.hits} use{'s' if self.hits != 1 else ''})"


def add(
    store: Any,
    entity_id: str,
    alias: str,
    *,
    source: str = "manual",
    at: datetime | None = None,
) -> str | None:
    """Record one alias. Idempotent; a repeat use increments the hit count."""
    canonical = str(entity_id or "").strip()
    alias_key = key(alias)
    if not canonical or len(alias_key) < MIN_KEY_LEN or len(alias_key) > MAX_ALIAS_LEN:
        return None
    _ensure(store)
    stamp = (at or _now()).isoformat()
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT INTO {TABLE} (entity_id, alias_key, alias, source, hits, created_at, updated_at)
                VALUES (?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(entity_id, alias_key) DO UPDATE SET
                    hits = hits + 1,
                    updated_at = excluded.updated_at,
                    source = excluded.source""",
            (canonical, alias_key, str(alias).strip(), source, stamp, stamp),
        )
    return alias_key


def learn(
    store: Any, entity_id: str, names: Iterable[str], *, source: str = "usage"
) -> int:
    """Learn several names for one entity, as a usage pass would."""
    learned = 0
    for name in names:
        if add(store, entity_id, name, source=source) is not None:
            learned += 1
    return learned


def resolve(store: Any, name: str) -> str | None:
    """The canonical entity for a name, or None if this name is unknown."""
    alias_key = key(name)
    if len(alias_key) < MIN_KEY_LEN:
        return None
    try:
        rows = store._conn.execute(
            f"SELECT entity_id FROM {TABLE} WHERE alias_key = ? ORDER BY hits DESC LIMIT 2",
            (alias_key,),
        ).fetchall()
    except Exception:  # noqa: BLE001 - alias lookup must never break a query
        return None
    if not rows:
        return None
    # An ambiguous name resolves to the habit, not to an arbitrary row.
    return str(rows[0]["entity_id"])


def aliases_for(store: Any, entity_id: str) -> list[Alias]:
    _ensure(store)
    rows = store._conn.execute(
        f"""SELECT entity_id, alias, hits, source FROM {TABLE}
            WHERE entity_id = ? ORDER BY hits DESC, alias ASC""",
        (str(entity_id),),
    ).fetchall()
    return [
        Alias(str(row["entity_id"]), str(row["alias"]), int(row["hits"]),
              str(row["source"] or ""))
        for row in rows
    ]


def entities(store: Any, *, limit: int = 100) -> list[dict[str, Any]]:
    _ensure(store)
    rows = store._conn.execute(
        f"""SELECT entity_id, COUNT(*) AS aliases, SUM(hits) AS hits
            FROM {TABLE} GROUP BY entity_id
            ORDER BY hits DESC, aliases DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [
        {"entity_id": str(row["entity_id"]), "aliases": int(row["aliases"]),
         "hits": int(row["hits"] or 0)}
        for row in rows
    ]


def expand(store: Any, names: Iterable[str]) -> list[str]:
    """Canonical names for a set of mentioned names, deduplicated, order preserved."""
    out: list[str] = []
    for name in names:
        canonical = resolve(store, name) or str(name).strip()
        if canonical and canonical not in out:
            out.append(canonical)
    return out


def remove(store: Any, name: str, *, entity_id: str | None = None) -> int:
    """Forget an alias. With an entity, only that entity's row; without, every match."""
    alias_key = key(name)
    if not alias_key:
        return 0
    _ensure(store)
    with store.transaction() as conn:
        if entity_id:
            cursor = conn.execute(
                f"DELETE FROM {TABLE} WHERE alias_key = ? AND entity_id = ?",
                (alias_key, str(entity_id)),
            )
        else:
            cursor = conn.execute(f"DELETE FROM {TABLE} WHERE alias_key = ?", (alias_key,))
        return int(cursor.rowcount or 0)


def merge(store: Any, source_entity: str, target_entity: str) -> dict[str, int]:
    """Fold one entity's aliases into another — the correction for a wrong resolution."""
    src = str(source_entity).strip()
    dst = str(target_entity).strip()
    if not src or not dst or key(src) == key(dst):
        return {"moved": 0, "dropped": 0}
    _ensure(store)
    moved = 0
    dropped = 0
    with store.transaction() as conn:
        rows = conn.execute(
            f"SELECT alias_key, alias, hits, source, created_at FROM {TABLE} WHERE entity_id = ?",
            (src,),
        ).fetchall()
        for row in rows:
            existing = conn.execute(
                f"SELECT hits FROM {TABLE} WHERE entity_id = ? AND alias_key = ?",
                (dst, str(row["alias_key"])),
            ).fetchone()
            if existing is not None:
                conn.execute(
                    f"""UPDATE {TABLE} SET hits = hits + ?
                        WHERE entity_id = ? AND alias_key = ?""",
                    (int(row["hits"]), dst, str(row["alias_key"])),
                )
                dropped += 1
                continue
            conn.execute(
                f"""UPDATE {TABLE} SET entity_id = ? WHERE entity_id = ? AND alias_key = ?""",
                (dst, src, str(row["alias_key"])),
            )
            moved += 1
        conn.execute(f"DELETE FROM {TABLE} WHERE entity_id = ?", (src,))
    return {"moved": moved, "dropped": dropped}


_NOT_A_NAME = re.compile(
    r"^(?:/|[A-Za-z]:\\|\d{4}-\d{2}-\d{2})|\s›\s|@\w+\.\w+$"
)


def looks_like_a_name(text: str) -> bool:
    """Reject the values that reach `entities_json` without being entities.

    The live store's entity lists carry vault paths ("/tank"), dates ("2026-07-15") and section
    headings ("2026-07-17 › HomeBox asset ID structure"). Seeding the alias table from those teaches
    resolution 811 "entities" that are not things, and every later write would attach a date to a
    memory as if it were a machine.
    """
    value = str(text or "").strip()
    if not value or len(value) > 60:
        return False
    return _NOT_A_NAME.search(value) is None


def from_memories(store: Any, *, limit: int = 500) -> dict[str, int]:
    """Usage pass: every entity already attached to a memory becomes known to itself.

    This is the honest starting point — it teaches the table the names the store already uses, so
    `resolve()` answers for them without anyone typing an alias by hand.
    """
    _ensure(store)
    rows = store._conn.execute(
        """SELECT m.entities_json FROM memories m
           WHERE m.state IN ('active','cold') AND json_valid(m.entities_json)
             AND json_array_length(m.entities_json) > 0
           LIMIT ?""",
        (int(limit),),
    ).fetchall()
    learned = 0
    for row in rows:
        try:
            import json

            names = json.loads(str(row["entities_json"]))
        except (ValueError, TypeError):
            continue
        for name in names if isinstance(names, list) else []:
            text = str(name).strip()
            if not (MIN_KEY_LEN <= len(key(text)) <= MAX_ALIAS_LEN):
                continue
            if not looks_like_a_name(text):
                continue
            if add(store, text, text, source="memory_entity") is not None:
                learned += 1
    return {"memories": len(rows), "aliases": learned}


def report(store: Any) -> dict[str, Any]:
    described = entities(store)
    total_aliases = sum(item["aliases"] for item in described)
    ambiguous = store._conn.execute(
        f"""SELECT alias_key FROM {TABLE} GROUP BY alias_key HAVING COUNT(*) > 1 LIMIT 10"""
    ).fetchall()
    return {
        "entities": len(described),
        "aliases": total_aliases,
        "ambiguous": [str(row["alias_key"]) for row in ambiguous],
        "top": described[:10],
    }
