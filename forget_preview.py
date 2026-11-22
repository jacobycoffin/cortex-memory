"""Preview a semantic forget, then apply it by exact id (#89).

"Forget everything about the old NAS" is a request the store should be able to serve, and it is
also the single easiest way to destroy a memory system: a similarity query that returns eleven
memories is a *proposal*, and by the time a human has read the eleventh the tenth may no longer
be the same memory it was when the query ran.

So the two halves are separated by an **envelope**: the preview returns its matches and a token
that names exactly those ids, and the apply is only accepted if every id it names was in the
envelope and the envelope has not expired. There is no code path that forgets "whatever the
query returns" — an apply can only ever act on what was actually shown.

Defaults are deliberately conservative:

- **preview expires** after 30 minutes (`CORTEX_FORGET_PREVIEW_MINUTES`); a stale preview is a
  different decision made on different evidence;
- **tombstone, not delete** — the ladder's `restore()` still works;
- **a partial apply is refused rather than trimmed**: if one id is wrong, the whole request is
  wrong, and silently forgetting nine of eleven would report success on a lie.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

PREVIEW_MINUTES = 30
_TABLE = "forget_previews"
_SNIPPET = 160


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _minutes() -> int:
    raw = os.environ.get("CORTEX_FORGET_PREVIEW_MINUTES", "").strip()
    try:
        value = int(raw)
    except ValueError:
        return PREVIEW_MINUTES
    return value if value > 0 else PREVIEW_MINUTES


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {_TABLE} (
                    envelope TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    ids_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    applied_at TEXT,
                    applied_ids_json TEXT
                )"""
        )


def envelope_for(ids: list[str], query: str, created_at: str) -> str:
    payload = json.dumps({"ids": sorted(ids), "query": query, "at": created_at},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class Preview:
    envelope: str
    query: str
    created_at: str
    expires_at: str
    matches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ids(self) -> list[str]:
        return [str(match["memory_id"]) for match in self.matches]

    def describe(self, width: int = 8, snippet: int = _SNIPPET) -> str:
        lines = [f"envelope {self.envelope[:12]} expires {self.expires_at[:16]}"]
        for match in self.matches:
            content = " ".join(str(match.get("content") or "").split())
            lines.append(
                f"  {match['memory_id'][:width]} [{match.get('kind')}/{match.get('state')}] "
                f"{content[:snippet]}"
            )
        return "\n".join(lines)


@dataclass
class Applied:
    envelope: str
    forgotten: list[str] = field(default_factory=list)
    refused: str = ""

    @property
    def ok(self) -> bool:
        return not self.refused

    @property
    def summary(self) -> str:
        if self.refused:
            return f"refused: {self.refused}"
        return f"tombstoned {len(self.forgotten)} memories from envelope {self.envelope[:12]}"


def preview(
    store: Any,
    query: str,
    *,
    limit: int = 10,
    at: datetime | None = None,
) -> Preview:
    """What *would* be forgotten. Read-only: nothing is created except the envelope record."""
    from .retrieval import MemoryRetriever

    retriever = MemoryRetriever(store)
    results, _diagnostics = retriever.search_detailed(query, limit=limit)
    matches = []
    for result in results:
        memory = result.memory
        matches.append(
            {
                "memory_id": str(memory["id"]),
                "kind": memory.get("kind"),
                "state": memory.get("state"),
                "pinned": bool(memory.get("pinned")),
                "content": memory.get("content"),
                "score": round(float(getattr(result, "score", 0.0) or 0.0), 4),
            }
        )
    stamp = at or _utc_now()
    created = stamp.isoformat()
    expires = (stamp + timedelta(minutes=_minutes())).isoformat()
    envelope = envelope_for([m["memory_id"] for m in matches], query, created)
    _ensure(store)
    with store.transaction() as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO {_TABLE}(envelope, query, ids_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (envelope, query, json.dumps([m["memory_id"] for m in matches]), created),
        )
    return Preview(
        envelope=envelope, query=query, created_at=created, expires_at=expires, matches=matches
    )


def _expiry(created_at: str) -> datetime | None:
    try:
        stamp = datetime.fromisoformat(created_at)
    except ValueError:
        return None
    if not stamp.tzinfo:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp + timedelta(minutes=_minutes())


def apply(
    store: Any,
    envelope: str,
    ids: list[str],
    *,
    at: datetime | None = None,
    allow_protected: bool = False,
) -> Applied:
    """Tombstone exactly the previewed ids. Refuses anything the preview did not show."""
    _ensure(store)
    row = store._conn.execute(
        f"SELECT envelope, query, ids_json, created_at, applied_at FROM {_TABLE} WHERE envelope = ?",
        (envelope,),
    ).fetchone()
    if row is None:
        return Applied(envelope=envelope, refused="unknown envelope")
    if row["applied_at"]:
        return Applied(envelope=envelope, refused="envelope already applied")
    now = at or _utc_now()
    if _is_expired(str(row["created_at"]), at=now):
        return Applied(envelope=envelope, refused="envelope expired; re-run the preview")

    allowed = [str(item) for item in json.loads(row["ids_json"])]
    requested = [str(item) for item in ids]
    outside = [item for item in requested if item not in allowed]
    if outside:
        return Applied(
            envelope=envelope,
            refused=(
                "ids not in the previewed set: "
                + ", ".join(item[:8] for item in outside)
            ),
        )
    if not requested:
        return Applied(envelope=envelope, refused="no ids requested")

    if not allow_protected:
        from .forget_ladder import protections

        held = protections(store)
        blocked = [item for item in requested if item in held]
        if blocked:
            return Applied(
                envelope=envelope,
                refused=(
                    "protected memories: "
                    + ", ".join(f"{item[:8]} ({held[item]})" for item in blocked)
                ),
            )

    forgotten = []
    for memory_id in requested:
        try:
            if store.set_state(memory_id, "tombstoned",
                               reason=f"semantic forget (envelope {envelope[:12]})"):
                forgotten.append(memory_id)
        except Exception:  # noqa: BLE001 - one failure must not abort a reviewed batch
            continue
    with store.transaction() as conn:
        conn.execute(
            f"UPDATE {_TABLE} SET applied_at = ?, applied_ids_json = ? WHERE envelope = ?",
            (now.isoformat(), json.dumps(forgotten), envelope),
        )
    if forgotten:
        from . import prune_receipts

        prune_receipts.record(
            store, "semantic_forget", "tombstone", forgotten,
            summary=f"envelope {envelope[:12]} for {str(row['query'])[:60]}",
        )
    return Applied(envelope=envelope, forgotten=forgotten)


def envelopes(store: Any, limit: int = 20) -> list[dict[str, Any]]:
    """The audit trail: what was previewed, and what was applied."""
    _ensure(store)
    rows = store._conn.execute(
        f"""SELECT envelope, query, ids_json, created_at, applied_at, applied_ids_json
            FROM {_TABLE} ORDER BY created_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    out = []
    for row in rows:
        out.append(
            {
                "envelope": str(row["envelope"]),
                "query": str(row["query"]),
                "previewed": len(json.loads(row["ids_json"])),
                "applied": len(json.loads(row["applied_ids_json"] or "[]")),
                "created_at": str(row["created_at"]),
                "applied_at": row["applied_at"],
                "expired": _is_expired(str(row["created_at"])),
            }
        )
    return out


def _is_expired(created_at: str, *, at: datetime | None = None) -> bool:
    expiry = _expiry(created_at)
    if expiry is None:
        return True
    return (at or _utc_now()) > expiry
