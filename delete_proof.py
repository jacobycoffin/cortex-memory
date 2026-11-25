"""A receipt for anything that destroys a memory, computed before it happens (#180).

The store's rule is *demote, never delete*, and this module is the exception request: it is for the cases
where an operator has to remove something — a leaked secret, a wrong person's data, a table so broken it
should not exist. The failure it prevents is the one nobody can audit afterwards: a row gone, no record of
what went with it, and no way to answer "did this actually happen".

Five rules, each a way a destructive path goes wrong:

- **the proof is computed before the change and shown to a person.** What changes (table by table, with
  counts), what is lost, what survives, and whether it can be undone. A destructive action with no stated
  reason is refused outright — "because it was called" is not a reason.
- **the receipt is written in the same transaction as the change.** A receipt written first can describe
  a change that never happened; one written afterwards can be lost by the crash it is meant to record.
- **a receipt names tables and counts, never content.** It is a record that can be read aloud, pasted in a
  ticket, or kept for years — quoting the row would make it a second copy of the thing being removed.
- **reversibility is stated per action, and a purge says `not reversible`.** Archiving is a demotion you
  can undo; a purge is not, and a panel that says "deleted" for both teaches people to click through it.
- **a receipt can be checked later.** `verify()` answers whether the change described is the change that
  exists now — a receipt nobody can verify is a diary entry.

The reference list is **discovered, not hardcoded**: this store has grown past thirty tables with a
`memory_id` column, and a literal list would go stale the day a thirty-eighth arrives, silently leaving
rows behind. That is the failure this module exists to prevent, so it does not repeat it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

MEMORY_TABLE = "memories"
RECEIPT_TABLE = "deletion_receipts"
EDGE_TABLE = "edges"

ARCHIVE = "archive"
TOMBSTONE = "tombstone"
PURGE = "purge"

ACTIONS: dict[str, dict[str, Any]] = {
    ARCHIVE: {
        "label": "Archive",
        "effect": "state -> 'archived'",
        "lost": "nothing - the row, its content and its edges survive",
        "reversible": True,
        "destroys": False,
    },
    TOMBSTONE: {
        "label": "Tombstone",
        "effect": "state -> 'tombstoned'",
        "lost": "nothing, but the row is excluded from recall for good unless restored explicitly",
        "reversible": True,
        "destroys": False,
    },
    PURGE: {
        "label": "Purge",
        "effect": "content replaced by a marker, chunks and embeddings removed",
        "lost": "the content itself, its document chunks, and its embeddings",
        "reversible": False,
        "destroys": True,
    },
}

# What a purge removes, and therefore what the proof must count. Everything else that points at the
# memory is left in place: a receipt that also deleted the audit trail would be a receipt nobody kept.
PURGE_REMOVES = ("document_chunks", "memory_embeddings", "memory_fts")
PURGE_MARKER = "[purged]"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]] | None:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return None


def _tables(store: Any) -> set[str] | None:
    rows = _rows(store, "SELECT name FROM sqlite_master WHERE type = 'table'")
    return {str(row["name"]) for row in rows} if rows is not None else None


def references(store: Any) -> list[str]:
    """Every table with a `memory_id` column, discovered rather than listed.

    A hardcoded reference list is the bug that leaves rows behind when a new table is added, so this asks
    the schema instead: `PRAGMA table_info` for each table, filtered on the column name.
    """
    tables = _tables(store) or set()
    found: list[str] = []
    for table in sorted(tables):
        if table in {RECEIPT_TABLE, "sqlite_sequence"}:
            continue
        columns = _rows(store, f"PRAGMA table_info({table})")
        if not columns:
            continue
        if any(str(column.get("name")) == "memory_id" for column in columns):
            found.append(table)
    return found


def _count(store: Any, table: str, memory_id: str) -> int | None:
    rows = _rows(store, f"SELECT COUNT(*) AS n FROM {table} WHERE memory_id = ?", (str(memory_id),))
    if not rows:
        return None
    return int(rows[0].get("n") or 0)


@dataclass
class Proof:
    """What a destructive action would do, before it does it."""

    memory_id: str = ""
    action: str = ""
    reason: str = ""
    ok: bool = False
    refusal: str = ""
    exists: bool = False
    affects: dict[str, int | None] = field(default_factory=dict)
    lost: str = ""
    survives: str = ""
    reversible: bool = True
    keeps: dict[str, int] = field(default_factory=dict)
    receipt_id: str = ""

    @property
    def destructive(self) -> bool:
        return bool(ACTIONS.get(self.action, {}).get("destroys"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "action": self.action,
            "reason": self.reason,
            "ok": self.ok,
            "refusal": self.refusal,
            "affects": self.affects,
            "lost": self.lost,
            "survives": self.survives,
            "reversible": self.reversible,
            "keeps": self.keeps,
            "receipt_id": self.receipt_id,
        }

    def render(self) -> list[str]:
        if not self.ok:
            return [f"refused: {self.refusal}"]
        name = ACTIONS[self.action]["label"]
        lines = [f"{name} {self.memory_id[:8]} - receipt {self.receipt_id}"]
        lines.append(f"  effect: {ACTIONS[self.action]['effect']}")
        lines.append(f"  reason: {self.reason}")
        for table, count in sorted(self.affects.items()):
            lines.append(f"  {table}: {count if count is not None else 'unknown'} row(s) affected")
        lines.append(f"  lost: {self.lost}")
        lines.append(f"  survives: {self.survives}")
        lines.append(f"  reversible: {'yes' if self.reversible else 'NOT reversible'}")
        return lines


def plan(store: Any, memory_id: str, *, action: str = ARCHIVE, reason: str = "") -> Proof:
    """Compute the proof. Never raises, and writes nothing."""
    proof = Proof(memory_id=str(memory_id or ""), action=str(action), reason=str(reason or "").strip())
    spec = ACTIONS.get(proof.action)
    if spec is None:
        proof.refusal = f"unknown action {action!r}; the choices are {', '.join(sorted(ACTIONS))}"
        return proof
    if not proof.reason:
        proof.refusal = "a destructive action needs a stated reason; 'because it was called' is not one"
        return proof
    if _tables(store) is None:
        proof.refusal = "the store could not be read, so nothing can be proven about it"
        return proof
    row = _rows(store, f"SELECT id, state FROM {MEMORY_TABLE} WHERE id = ?", (proof.memory_id,))
    if row is None:
        proof.refusal = "the store could not be read while looking for the memory"
        return proof
    if not row:
        proof.refusal = f"unknown memory: no row in {MEMORY_TABLE} with that id"
        return proof

    proof.exists = True
    proof.reversible = bool(spec["reversible"])
    proof.lost = str(spec["lost"])

    targets = references(store)
    affects: dict[str, int | None] = {MEMORY_TABLE: 1}
    for table in targets:
        if table in PURGE_REMOVES and not proof.destructive:
            # Archiving and tombstoning leave the derived rows alone; counting them as "affected" would
            # overstate what an archive does and make the number useless for judging a purge.
            affects[table] = 0
            continue
        affects[table] = _count(store, table, proof.memory_id)
    proof.affects = affects

    edges = _rows(store, f"SELECT COUNT(*) AS n FROM {EDGE_TABLE} WHERE src_id = ? OR dst_id = ?",
                  (proof.memory_id, proof.memory_id))
    proof.keeps = {
        "edges": int(edges[0].get("n") or 0) if edges else 0,
        "tables_with_memory_id": len(targets),
    }
    proof.survives = (
        f"the row itself, marked {ACTIONS[proof.action]['effect'].split('-> ')[-1]}, and all "
        f"{proof.keeps['edges']} edge(s)" if not proof.destructive else
        f"the row id, its state, and all {proof.keeps['edges']} edge(s) - but not its content"
    )
    proof.receipt_id = hashlib.sha256(
        json.dumps([proof.memory_id, proof.action, proof.reason, sorted(affects.items())],
                   default=str).encode("utf-8")
    ).hexdigest()[:16]
    proof.ok = True
    return proof


def _ensure_receipts(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {RECEIPT_TABLE} (
                    receipt_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    affects_json TEXT NOT NULL,
                    reversible INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    state_after TEXT
                )"""
        )


class _NothingToChange(Exception):
    """Raised inside the transaction when the row a proof describes is not there any more."""


def apply(store: Any, proof: Proof, *, consent: bool = False) -> dict[str, Any]:
    """Carry out a proof: receipt and change in one transaction, or neither."""
    if not proof.ok:
        return {"ok": False, "reason": proof.refusal}
    if not consent:
        return {"ok": False, "reason": f"{proof.action} asks first; pass consent to run it"}
    _ensure_receipts(store)
    moment = _now().isoformat()
    try:
        with store.transaction() as conn:
            state_after = {ARCHIVE: "archived", TOMBSTONE: "tombstoned", PURGE: "tombstoned"}[proof.action]
            if proof.action == PURGE:
                cursor = conn.execute(
                    f"UPDATE {MEMORY_TABLE} SET content = ?, state = ?, updated_at = ? WHERE id = ?",
                    (PURGE_MARKER, state_after, moment, proof.memory_id),
                )
                if cursor.rowcount == 0:
                    raise _NothingToChange(proof.memory_id)
                for table in PURGE_REMOVES:
                    columns = _rows(store, f"PRAGMA table_info({table})") or []
                    if not columns:
                        continue
                    if any(str(column.get("name")) == "content" for column in columns):
                        conn.execute(
                            f"UPDATE {table} SET content = ? WHERE memory_id = ?",
                            (PURGE_MARKER, proof.memory_id),
                        )
                    else:
                        conn.execute(f"DELETE FROM {table} WHERE memory_id = ?", (proof.memory_id,))
            else:
                cursor = conn.execute(
                    f"UPDATE {MEMORY_TABLE} SET state = ?, updated_at = ? WHERE id = ?",
                    (state_after, moment, proof.memory_id))
                if cursor.rowcount == 0:
                    # The proof described a row that has since gone. Writing a receipt here would
                    # record a change that never happened, which is the failure this module exists for.
                    raise _NothingToChange(proof.memory_id)
            conn.execute(
                f"INSERT OR REPLACE INTO {RECEIPT_TABLE} (receipt_id, memory_id, action, reason, "
                "affects_json, reversible, created_at, state_after) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (proof.receipt_id, proof.memory_id, proof.action, proof.reason,
                 json.dumps(proof.affects, sort_keys=True, default=str),
                 int(proof.reversible), moment, state_after),
            )
    except _NothingToChange:
        return {"ok": False, "reason": "the memory was not there when the change ran; no receipt was written"}
    except Exception as error:  # noqa: BLE001 - a failed destructive action must not escape as an exception
        return {"ok": False, "reason": f"the store refused the change: {type(error).__name__}"}
    return {"ok": True, "receipt_id": proof.receipt_id, "action": proof.action,
            "state_after": state_after}


def verify(store: Any, receipt_id: str) -> dict[str, Any]:
    """Does the change described by a receipt match the store now?"""
    _ensure_receipts(store)
    rows = _rows(store, f"SELECT * FROM {RECEIPT_TABLE} WHERE receipt_id = ?", (str(receipt_id),))
    if rows is None:
        return {"ok": False, "reason": "the receipt table could not be read"}
    if not rows:
        return {"ok": False, "reason": f"no receipt {receipt_id}"}
    receipt = rows[0]
    memory = _rows(store, f"SELECT state, content FROM {MEMORY_TABLE} WHERE id = ?",
                   (str(receipt.get("memory_id")),))
    if memory is None:
        return {"ok": False, "reason": "the memories table could not be read"}
    if not memory:
        return {"ok": False, "reason": "the memory the receipt describes no longer exists"}
    state = str(memory[0].get("state") or "")
    matches = state == str(receipt.get("state_after") or "")
    if receipt.get("action") == PURGE:
        matches = matches and str(memory[0].get("content") or "") == PURGE_MARKER
    return {
        "ok": matches,
        "reason": ("the store matches the receipt" if matches
                   else f"the store does not match the receipt: state is {state!r}, expected "
                        f"{receipt.get('state_after')!r}"),
        "action": receipt.get("action"),
        "memory_id": receipt.get("memory_id"),
        "reversible": bool(receipt.get("reversible")),
        "reason_given": receipt.get("reason"),
        "affects": json.loads(str(receipt.get("affects_json") or "{}")),
    }


def receipts(store: Any, *, limit: int = 20) -> list[dict[str, Any]]:
    _ensure_receipts(store)
    rows = _rows(store, f"SELECT * FROM {RECEIPT_TABLE} ORDER BY created_at DESC LIMIT ?", (int(limit),))
    return rows or []
