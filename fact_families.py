"""Group variant phrasings of one fact so they age together (#56).

"Backups run at 2am", "the backup window is 02:00" and "backups start at two in the morning" are one
fact written three times. Retrieval treats them as three, which is mostly harmless; the forget ladder
does not — it ages each on its own use, so the phrasing that happens to get retrieved goes warm and
survives while its twins, equally true, quietly walk down the ladder and get archived.

A family fixes the *aging*, not the storage. The members stay separate rows (a variant can be a better
match for a particular question), and the family's **anchor** is the freshest use among its members:
if one phrasing was useful last week, the fact was useful last week, whichever words were in front of
it at the time.

Ships **off** (`CORTEX_FAMILY_AGING=1`): it changes what the ladder forgets, and a change like that
should be measured on a store before it becomes the default. The families themselves are cheap to
build and useful for review either way.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

TABLE = "fact_families"
ENV = "CORTEX_FAMILY_AGING"


def enabled(environ: Any = None) -> bool:
    env = environ if environ is not None else os.environ
    return str(env.get(ENV, "")).strip().lower() in {"1", "true", "yes", "on"}


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
                    family_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'member',
                    built_at TEXT NOT NULL,
                    PRIMARY KEY (memory_id)
                )"""
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {TABLE}_family_idx ON {TABLE} (family_id)"
        )


def _same_fact(write_dedup: Any, left: str, right: str) -> bool:
    """The store's duplicate rule, applied to two texts rather than to a text and the store."""
    if write_dedup.jaccard(left, right) >= write_dedup.VERBATIM_JACCARD:
        return True
    shorter, longer = sorted((left, right), key=len)
    return (
        write_dedup.jaccard(left, right) >= write_dedup.RESTATEMENT_SIMILARITY
        and write_dedup.containment(shorter, longer) >= write_dedup.RESTATEMENT_CONTAINMENT
    )


@dataclass
class Family:
    family_id: str
    members: list[str] = field(default_factory=list)
    representative: str = ""
    anchor: str = ""
    sizes: dict[str, int] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.members)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "members": self.members,
            "representative": self.representative,
            "anchor": self.anchor,
            "size": self.size,
        }

    def describe(self) -> str:
        return (f"{self.size} phrasing(s) of one fact, freshest use {self.anchor[:10] or 'never'}"
                f"; kept as {self.representative[:8]}")


def build(store: Any, *, scan: int = 400, at: datetime | None = None,
          persist: bool = True) -> list[Family]:
    """Group near-identical memories into families, using the store's existing duplicate rule.

    The rule is `write_dedup`'s: verbatim at 0.85 Jaccard, or a restatement at 0.60 similarity with
    0.80 containment of the shorter text. A second similarity rule for families would be a second
    definition of "the same fact", and the two would drift.
    """
    from . import write_dedup

    rows = _rows(
        store,
        """SELECT id, content, confidence, importance, last_used_at, created_at,
                  COALESCE(used_count, 0) AS uses, state
           FROM memories WHERE state IN ('active','cold','archived')
           ORDER BY created_at DESC LIMIT ?""",
        (int(scan),),
    )
    parent: dict[str, str] = {str(row["id"]): str(row["id"]) for row in rows}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            # deterministic: the lexicographically smaller id becomes the root
            first, second = sorted((left_root, right_root))
            parent[second] = first

    contents = {str(row["id"]): str(row["content"] or "") for row in rows}
    token_sets = {memory_id: write_dedup.tokens(text) for memory_id, text in contents.items()}
    ids = list(contents)
    for position, left in enumerate(ids):
        for right in ids[position + 1:]:
            # A pair that shares fewer than three words cannot reach the duplicate rule, and
            # scoring it anyway is what makes this quadratic pass slow on a real store.
            if len(token_sets[left] & token_sets[right]) < 3:
                continue
            if _same_fact(write_dedup, contents[left], contents[right]):
                union(left, right)

    groups: dict[str, list[str]] = {}
    for memory_id in ids:
        groups.setdefault(find(memory_id), []).append(memory_id)

    families: list[Family] = []
    detail = {str(row["id"]): row for row in rows}
    for root, members in sorted(groups.items()):
        members.sort()
        anchor = max(
            (str(detail[member].get("last_used_at") or detail[member].get("last_helpful_at")
                 or detail[member].get("created_at") or "") for member in members),
            default="",
        )
        representative = max(
            members,
            key=lambda member: (
                float(detail[member].get("confidence") or 0.0),
                int(detail[member].get("uses") or 0),
                str(detail[member].get("created_at") or ""),
            ),
        )
        families.append(Family(family_id=root, members=members, representative=representative,
                               anchor=anchor))
    if persist:
        _persist(store, families, at=at)
    return families


def _persist(store: Any, families: Iterable[Family], *, at: datetime | None = None) -> int:
    _ensure(store)
    moment = (at or _now()).isoformat()
    written = 0
    with store.transaction() as conn:
        for family in families:
            for member in family.members:
                conn.execute(
                    f"""INSERT INTO {TABLE} (family_id, memory_id, role, built_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(memory_id) DO UPDATE SET
                            family_id = excluded.family_id,
                            role = excluded.role,
                            built_at = excluded.built_at""",
                    (family.family_id, member,
                     "representative" if member == family.representative else "member", moment),
                )
                written += 1
    return written


def family_of(store: Any, memory_id: str) -> list[str]:
    """Every phrasing of the same fact, including this one. Never empty for a known memory."""
    _ensure(store)
    rows = _rows(
        store,
        f"""SELECT memory_id FROM {TABLE}
            WHERE family_id = (SELECT family_id FROM {TABLE} WHERE memory_id = ?)""",
        (str(memory_id),),
    )
    members = sorted({str(row["memory_id"]) for row in rows})
    return members or [str(memory_id)]


def anchor(store: Any, memory_id: str) -> dict[str, Any]:
    """The freshest use across the family — the date its retention should be judged on."""
    members = family_of(store, memory_id)
    # The same coalesce the ladder uses (`last_use`): a fact that has never been retrieved is still
    # dated by when the store learned it, and an anchor that ignored that would read "never" for a
    # whole family and quietly do nothing.
    _FRESH = "COALESCE(last_used_at, last_helpful_at, created_at)"
    if len(members) == 1:
        row = _rows(
            store,
            f"""SELECT {_FRESH} AS last_used, COALESCE(used_count,0) AS uses
                FROM memories WHERE id = ?""",
            (str(memory_id),),
        )
        detail = row[0] if row else {}
        return {"members": members, "size": 1,
                "last_used": str(detail.get("last_used") or ""),
                "uses": int(detail.get("uses") or 0)}
    rows = _rows(
        store,
        f"""SELECT m.id, COALESCE(m.last_used_at, m.last_helpful_at, m.created_at) AS last_used,
                   COALESCE(m.used_count,0) AS uses
            FROM memories m JOIN {TABLE} f ON f.memory_id = m.id
            WHERE f.family_id = (SELECT family_id FROM {TABLE} WHERE memory_id = ?)""",
        (str(memory_id),),
    )
    freshest = max((str(row.get("last_used") or "") for row in rows), default="")
    return {
        "members": members,
        "size": len(members),
        "last_used": freshest,
        "uses": sum(int(row.get("uses") or 0) for row in rows),
    }


def apply_to(store: Any, memory_id: str, used: datetime | None,
             *, environ: Any = None) -> datetime | None:
    """Replace a member's own last use with the family's, when the change is switched on.

    Off by default, and a failure returns the original value: the ladder must keep working even if the
    family table has never been built.
    """
    if not enabled(environ) or used is None:
        return used
    try:
        found = anchor(store, memory_id)
    except Exception:  # noqa: BLE001
        return used
    if int(found.get("size") or 1) < 2:
        return used  # no twin, nothing to share
    candidate = str(found.get("last_used") or "")
    if not candidate:
        return used
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError:
        return used
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(used, parsed)


def report(store: Any, *, scan: int = 400) -> dict[str, Any]:
    families = build(store, scan=scan, persist=False)
    grouped = [family for family in families if family.size > 1]
    return {
        "families": len(grouped),
        "memories": sum(family.size for family in grouped),
        "largest": max((family.size for family in grouped), default=0),
        "singletons": len(families) - len(grouped),
        "aging_enabled": enabled(),
        "note": "a family changes the ageing of its members, never their storage",
    }
