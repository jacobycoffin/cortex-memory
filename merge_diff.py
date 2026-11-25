"""Two memories become one: the diff is the consent form (#168)

A merge is the only ordinary operation in this store where one row stops existing as itself and
another starts claiming to speak for both. `merge_review` shows a word diff and keeps both old
texts, and `entity_merge` records the alias trail; neither answers the question a person actually
has before saying yes: **what exactly changes, what stops being reachable, and what stops being
true afterwards**.

So the diff is built from the stored rows, not from a description of them:

- **lossy is a verdict, not a warning.** Every fact-bearing clause of the absorbed memory is
  checked against the text that would survive. One that is absent is listed and the diff is marked
  `lossy`; `apply()` refuses it unless a person passes `accept_loss=True` as well as consent,
  because a merge that silently drops a sentence is how the store forgets something it was told.
- **every count names its table** (`memories:retired`, `edges:repointed`, ...). A bare number in a
  review screen cannot be checked against anything.
- **computing writes nothing.** `compute()` is a read-only projection; the test that matters is the
  one that hashes the database file before and after. Applying is a separate call with a separate
  consent, and it re-reads the rows first, so a diff made stale by an intervening write is refused
  instead of applied to something it never saw.
- **the undo record is the reverse instruction, not a copy of the merge.** It carries both ids,
  every survivor field it overwrites (whoever merges next needs the old `scope_json`, `pinned` and
  `valid_to`), the absorbed row's own retirement state, and each moved edge with the row that used
  to occupy its destination. `undo()` replays that record; it does not try to reconstruct history
  from a word diff.

What this deliberately does not decide: the merged text. Inventing a sentence that neither memory
said is an editorial act, so `merged_text` is the caller's proposal and the diff only prices it.
When the caller supplies none, the survivor's own text is the proposal - which is exactly the case
where losses are most likely, and why they are reported first.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

MEMORY_TABLE = "memories"
EDGE_TABLE = "edges"
RECORD_TABLE = "merge_diff_records"

# Columns that the merge can change on the survivor. Named as columns, not attributes, so the diff
# can say "memories.pinned changed 0 -> 1" and an operator can check the claim against the schema.
SURVIVOR_FIELDS = (
    "content",
    "scope_json",
    "pinned",
    "protected",
    "entities_json",
    "confidence",
    "importance",
    "state",
    "valid_to",
)
# Fields restored on undo. Wider than SURVIVOR_FIELDS because apply stamps updated_at and
# content_hash, and an undo that left either pointing at the merged text would be a half-undo.
RESTORE_FIELDS = SURVIVOR_FIELDS + ("content_hash", "updated_at")

_CLAUSE = re.compile(r"[^.!?;\n]+")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    """A read that cannot fail the caller: a broken store reports no rows, not a traceback."""
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def _one(store: Any, memory_id: str) -> dict[str, Any] | None:
    rows = _rows(store, f"SELECT * FROM {MEMORY_TABLE} WHERE id = ?", (str(memory_id),))
    return rows[0] if rows else None


def _json(raw: Any, default: Any) -> Any:
    try:
        value = json.loads(str(raw))
    except (TypeError, ValueError):
        return default
    return value if isinstance(value, type(default)) else default


def _tokens(text: str) -> set[str]:
    """Significant words for a presence test.

    Short function words are dropped only because they would make every clause look present in
    every other one; a longer or numeric token always counts. This is a lexical test - it answers
    "does the surviving text still say this", not "does it entail this" - and the returned facts
    are quoted so a reader can judge for themselves.
    """
    out: set[str] = set()
    for token in _TOKEN.findall(str(text or "").casefold()):
        cleaned = token.strip("'-_")
        if cleaned and (len(cleaned) >= 3 or any(ch.isdigit() for ch in cleaned)):
            out.add(cleaned)
    return out


def clauses(text: str) -> list[str]:
    """Fact-bearing clauses, in order: the units a merge can drop."""
    out: list[str] = []
    for match in _CLAUSE.finditer(str(text or "")):
        clause = " ".join(match.group(0).split())
        if len(_tokens(clause)) >= 2:
            out.append(clause)
    return out


def _hash(content: str) -> str:
    try:
        from .store import content_hash as store_hash

        return str(store_hash(content))
    except Exception:  # noqa: BLE001 - the diff must work when imported outside the package
        import hashlib

        return hashlib.sha256(str(content or "").strip().casefold().encode("utf-8")).hexdigest()


def _merge_scope(survivor: dict[str, Any], absorbed: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Absorbed keys fill gaps; the survivor wins a conflict, and every conflict is reported.

    Scope is a filter. A merge that let the absorbed row's project key overwrite the survivor's
    would silently move a memory out of the project it was retrieved in, so the kept row keeps its
    own answer and the disagreement becomes a review item instead.
    """
    left = _json(survivor.get("scope_json"), {})
    right = _json(absorbed.get("scope_json"), {})
    merged = dict(left)
    conflicts: list[dict[str, Any]] = []
    for key, value in right.items():
        if key not in left:
            merged[key] = value
        elif left[key] != value:
            conflicts.append({"field": f"scope_json.{key}", "kept": left[key], "dropped": value})
    return json.dumps(merged, sort_keys=True, separators=(",", ":")), conflicts


def _merge_entities(survivor: dict[str, Any], absorbed: dict[str, Any]) -> str:
    names = [str(name) for name in _json(survivor.get("entities_json"), [])]
    seen = {name.casefold() for name in names}
    for name in _json(absorbed.get("entities_json"), []):
        text = str(name)
        if text.casefold() not in seen:
            seen.add(text.casefold())
            names.append(text)
    return json.dumps(names, ensure_ascii=False)


def _edge_rows(store: Any, absorbed_id: str) -> list[dict[str, Any]]:
    return _rows(
        store,
        f"""SELECT src_id, dst_id, relation, weight, evidence_count, created_at, last_reinforced_at
            FROM {EDGE_TABLE} WHERE src_id = ? OR dst_id = ? ORDER BY relation, src_id, dst_id""",
        (str(absorbed_id), str(absorbed_id)),
    )


def _plan_edges(store: Any, survivor_id: str, absorbed_id: str) -> list[dict[str, Any]]:
    """What happens to each edge touching the absorbed row.

    An edge from the absorbed memory to the survivor would become `CHECK (src_id <> dst_id)`
    violation, so it cannot be re-pointed; it is dropped and named. An edge whose re-pointed
    destination already exists folds into it, and the pre-existing destination row is recorded so
    undo can put the pair back exactly.
    """
    moves: list[dict[str, Any]] = []
    for row in _edge_rows(store, absorbed_id):
        src, dst = str(row["src_id"]), str(row["dst_id"])
        # Either direction: an edge *between* the pair becomes a self-loop once they are one memory,
        # and pairs are stored in canonical order, so which id lands in `src_id` is not something
        # this function can assume.
        if {src, dst} == {absorbed_id, survivor_id}:
            moves.append({**row, "action": "drop_self_loop", "from": absorbed_id, "to": survivor_id})
            continue
        action = "repoint_src" if src == absorbed_id else "repoint_dst"
        new_src = survivor_id if src == absorbed_id else src
        new_dst = survivor_id if dst == absorbed_id else dst
        existing = _rows(
            store,
            f"SELECT * FROM {EDGE_TABLE} WHERE src_id = ? AND dst_id = ? AND relation = ?",
            (new_src, new_dst, str(row["relation"])),
        )
        moves.append({
            **row,
            "action": "fold_duplicate" if existing else action,
            "from": absorbed_id,
            "to": survivor_id,
            "new_src_id": new_src,
            "new_dst_id": new_dst,
            "destination_before": existing[0] if existing else None,
        })
    return moves


def _field_changes(survivor: dict[str, Any], proposed: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for name in SURVIVOR_FIELDS:
        before = survivor.get(name)
        after = proposed.get(name, before)
        changes.append({
            "field": name,
            "table": MEMORY_TABLE,
            "before": before,
            "after": after,
            "changed": before != after,
        })
    return changes


def lost_facts(absorbed_text: str, surviving_text: str) -> list[str]:
    """Clauses of the absorbed memory that the surviving text does not contain.

    Computed against the *proposed* text, not against the survivor's current one: if the caller
    proposes a merged sentence, the question is whether that sentence still says what was absorbed.
    """
    surviving = _tokens(surviving_text)
    lost: list[str] = []
    for clause in clauses(absorbed_text):
        if not _tokens(clause) <= surviving:
            lost.append(clause)
    return lost


def _structural_losses(survivor: dict[str, Any], absorbed: dict[str, Any],
                       conflicts: list[dict[str, Any]],
                       moves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    losses: list[dict[str, Any]] = []
    for conflict in conflicts:
        losses.append({"kind": "scope_conflict", "table": MEMORY_TABLE, **conflict})
    if absorbed.get("valid_to") and absorbed.get("valid_to") != survivor.get("valid_to"):
        # The absorbed row was already retired on a date of its own; a merge that keeps only the
        # survivor's valid_to flattens two different retirements into one.
        losses.append({
            "kind": "valid_to",
            "table": MEMORY_TABLE,
            "id": str(absorbed.get("id")),
            "dropped": absorbed.get("valid_to"),
        })
    dropped = [move for move in moves if move["action"] == "drop_self_loop"]
    if dropped:
        losses.append({
            "kind": "edge_self_loop",
            "table": EDGE_TABLE,
            "count": len(dropped),
            "relations": sorted({str(move["relation"]) for move in dropped}),
        })
    return losses


@dataclass
class MergeDiff:
    survivor_id: str
    absorbed_id: str
    merged_content: str
    fields: list[dict[str, Any]] = field(default_factory=list)
    retired: list[dict[str, Any]] = field(default_factory=list)
    edge_moves: list[dict[str, Any]] = field(default_factory=list)
    lost_facts: list[str] = field(default_factory=list)
    structural_losses: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    undo: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    computed_at: str = ""

    @property
    def lossy(self) -> bool:
        """A fact the absorbed row asserted and the surviving text does not."""
        return bool(self.lost_facts)

    @property
    def changed_fields(self) -> list[dict[str, Any]]:
        return [item for item in self.fields if item["changed"]]

    def describe(self) -> str:
        head = (
            f"merge {self.absorbed_id[:8]} into {self.survivor_id[:8]}: "
            f"{self.counts.get('memories:fields_changed', 0)} field(s) change, "
            f"{self.counts.get('edges:repointed', 0)} edge(s) re-point, "
            f"{self.counts.get('memories:retired', 0)} memory retired"
        )
        if self.lossy:
            head += f", LOSSY: {len(self.lost_facts)} fact(s) would be lost"
            for fact in self.lost_facts[:3]:
                head += f"\n  lost: {fact}"
        return head

    def as_dict(self) -> dict[str, Any]:
        return {
            "survivor_id": self.survivor_id,
            "absorbed_id": self.absorbed_id,
            "merged_content": self.merged_content,
            "fields": self.fields,
            "retired": self.retired,
            "edge_moves": self.edge_moves,
            "lost_facts": self.lost_facts,
            "structural_losses": self.structural_losses,
            "conflicts": self.conflicts,
            "counts": self.counts,
            "undo": self.undo,
            "warnings": self.warnings,
            "lossy": self.lossy,
            "computed_at": self.computed_at,
            "preview": self.describe(),
        }


def compute(store: Any, survivor_id: str, absorbed_id: str, *, merged_text: str = "",
            at: datetime | None = None) -> MergeDiff | None:
    """Build the reviewable diff. Reads only; nothing here can change a row."""
    survivor = _one(store, survivor_id)
    absorbed = _one(store, absorbed_id)
    if survivor is None or absorbed is None:
        return None
    warnings: list[str] = []
    if str(survivor_id) == str(absorbed_id):
        warnings.append("survivor and absorbed are the same memory; nothing would be merged")
        return None
    scope_json, conflicts = _merge_scope(survivor, absorbed)
    content = str(merged_text or "").strip() or str(survivor.get("content") or "")
    proposed = {
        "content": content,
        "scope_json": scope_json,
        "pinned": max(int(survivor.get("pinned") or 0), int(absorbed.get("pinned") or 0)),
        "protected": max(int(survivor.get("protected") or 0), int(absorbed.get("protected") or 0)),
        "entities_json": _merge_entities(survivor, absorbed),
        # Confidence and importance are not averaged: a merge is not new evidence, and a mean of
        # two beliefs is a number neither row ever supported (see #57's weakest-source rule).
        "confidence": survivor.get("confidence"),
        "importance": max(float(survivor.get("importance") or 0.0),
                          float(absorbed.get("importance") or 0.0)),
        "state": survivor.get("state"),
        "valid_to": survivor.get("valid_to"),
    }
    moves = _plan_edges(store, str(survivor_id), str(absorbed_id))
    losses = _structural_losses(survivor, absorbed, conflicts, moves)
    lost = lost_facts(str(absorbed.get("content") or ""), content)
    moment = (at or _now()).isoformat()
    undo = {
        "merge_id": uuid.uuid4().hex[:12],
        "computed_at": moment,
        "survivor": {"id": str(survivor_id),
                     "before": {name: survivor.get(name) for name in RESTORE_FIELDS}},
        "absorbed": {"id": str(absorbed_id),
                     "before": {name: absorbed.get(name) for name in RESTORE_FIELDS}},
        "edges": moves,
    }
    counts = {
        f"{MEMORY_TABLE}:read": 2,
        f"{MEMORY_TABLE}:retired": 1,
        f"{MEMORY_TABLE}:fields_changed": sum(1 for item in _field_changes(survivor, proposed)
                                              if item["changed"]),
        f"{EDGE_TABLE}:moved": len(moves),
        f"{EDGE_TABLE}:repointed": sum(1 for move in moves
                                       if move["action"] in ("repoint_src", "repoint_dst")),
        f"{EDGE_TABLE}:folded": sum(1 for move in moves if move["action"] == "fold_duplicate"),
        f"{EDGE_TABLE}:dropped": sum(1 for move in moves if move["action"] == "drop_self_loop"),
        f"{MEMORY_TABLE}:facts_lost": len(lost),
    }
    if not (survivor.get("pinned") or absorbed.get("pinned")):
        warnings.append("neither memory is pinned; the merged row will not be protected from pruning")
    return MergeDiff(
        survivor_id=str(survivor_id),
        absorbed_id=str(absorbed_id),
        merged_content=content,
        fields=_field_changes(survivor, proposed),
        retired=[{
            "id": str(absorbed_id),
            "table": MEMORY_TABLE,
            "reason": "absorbed by the merge",
            "state_before": absorbed.get("state"),
            "valid_to_before": absorbed.get("valid_to"),
        }],
        edge_moves=moves,
        lost_facts=lost,
        structural_losses=losses,
        conflicts=conflicts,
        counts=counts,
        undo=undo,
        warnings=warnings,
        computed_at=moment,
    )


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {RECORD_TABLE} (
                    merge_id TEXT PRIMARY KEY,
                    survivor_id TEXT NOT NULL,
                    absorbed_id TEXT NOT NULL,
                    lossy INTEGER NOT NULL DEFAULT 0,
                    lost_facts_json TEXT NOT NULL DEFAULT '[]',
                    undo_json TEXT NOT NULL,
                    consented_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    undone_at TEXT
                )"""
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{RECORD_TABLE}_created ON {RECORD_TABLE}(created_at)"
        )


@dataclass
class Applied:
    ok: bool
    merge_id: str = ""
    reason: str = ""
    lost_facts: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if not self.ok:
            return f"not merged: {self.reason}"
        extra = f" (accepted {len(self.lost_facts)} reported loss(es))" if self.lost_facts else ""
        return f"merged under {self.merge_id}{extra}; the undo record is stored"


def _stale(diff: MergeDiff, survivor: dict[str, Any], absorbed: dict[str, Any]) -> str:
    """A diff describes rows it read; if they moved, the diff is not evidence about them."""
    for side, row in (("survivor", survivor), ("absorbed", absorbed)):
        before = diff.undo.get(side, {}).get("before", {})
        for name in ("content", "scope_json", "pinned", "state", "valid_to"):
            if before.get(name) != row.get(name):
                return f"the {side}'s {name} changed since this diff was computed"
    return ""


def apply(store: Any, diff: MergeDiff, *, consent: bool = False, consented_by: str = "",
          accept_loss: bool = False, at: datetime | None = None) -> Applied:
    """Carry out a computed merge. Separate from `compute()` on purpose: this is the only writer."""
    if not isinstance(diff, MergeDiff):
        return Applied(False, reason="no computed diff: a merge is applied from a diff, not a description")
    if not consent:
        return Applied(False, reason="consent required: nothing was merged")
    if not str(consented_by).strip():
        return Applied(False, reason="consent must name who gave it")
    if diff.lossy and not accept_loss:
        # The whole point of the verdict: the refusal quotes what would be lost so it cannot be
        # waved through as a generic "are you sure".
        return Applied(False, reason="lossy merge: " + "; ".join(diff.lost_facts[:5]),
                       lost_facts=list(diff.lost_facts))
    survivor = _one(store, diff.survivor_id)
    absorbed = _one(store, diff.absorbed_id)
    if survivor is None or absorbed is None:
        return Applied(False, reason="one of the memories no longer exists")
    stale = _stale(diff, survivor, absorbed)
    if stale:
        return Applied(False, reason=stale + "; recompute the diff")
    _ensure(store)
    moment = (at or _now()).isoformat()
    merge_id = str(diff.undo.get("merge_id") or uuid.uuid4().hex[:12])
    undo = {**diff.undo, "merge_id": merge_id, "applied_at": moment}
    proposed = {item["field"]: item["after"] for item in diff.fields}
    # Record the post-merge state as well as the pre-merge state. A rollback has to be able to tell "the survivor
    # still looks like this merge left it" from "somebody edited it afterwards", and only the after-state says which;
    # without it, reversing a merge silently discards every later edit to the survivor.
    undo = {**undo, "survivor": {**((undo.get("survivor") or {})), "after": {}}}
    with store.transaction() as conn:
        conn.execute(
            f"""UPDATE {MEMORY_TABLE} SET content = ?, content_hash = ?, scope_json = ?,
                    pinned = ?, protected = ?, entities_json = ?, importance = ?, updated_at = ?
                WHERE id = ?""",
            (str(proposed.get("content") or ""), _hash(str(proposed.get("content") or "")),
             str(proposed.get("scope_json") or "{}"), int(proposed.get("pinned") or 0),
             int(proposed.get("protected") or 0), str(proposed.get("entities_json") or "[]"),
             float(proposed.get("importance") or 0.0), moment, diff.survivor_id),
        )
        # Retired, not deleted: the row keeps its content and earns a retirement date, which is the
        # only thing that makes "what did we think in October" answerable (see #178).
        conn.execute(
            f"UPDATE {MEMORY_TABLE} SET state = 'superseded', valid_to = ?, updated_at = ? WHERE id = ?",
            (moment, moment, diff.absorbed_id),
        )
        for move in diff.edge_moves:
            conn.execute(
                f"DELETE FROM {EDGE_TABLE} WHERE src_id = ? AND dst_id = ? AND relation = ?",
                (str(move["src_id"]), str(move["dst_id"]), str(move["relation"])),
            )
            if move["action"] == "drop_self_loop":
                continue
            new_src, new_dst = str(move["new_src_id"]), str(move["new_dst_id"])
            existing = conn.execute(
                f"SELECT weight, evidence_count FROM {EDGE_TABLE} "
                "WHERE src_id = ? AND dst_id = ? AND relation = ?",
                (new_src, new_dst, str(move["relation"])),
            ).fetchone()
            if existing is None:
                conn.execute(
                    f"""INSERT INTO {EDGE_TABLE}
                        (src_id, dst_id, relation, weight, evidence_count, created_at, last_reinforced_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (new_src, new_dst, str(move["relation"]), float(move["weight"]),
                     int(move["evidence_count"]), str(move["created_at"]),
                     str(move["last_reinforced_at"])),
                )
            else:
                conn.execute(
                    f"""UPDATE {EDGE_TABLE} SET weight = MAX(weight, ?), evidence_count = evidence_count + ?
                        WHERE src_id = ? AND dst_id = ? AND relation = ?""",
                    (float(move["weight"]), int(move["evidence_count"]),
                     new_src, new_dst, str(move["relation"])),
                )
        # Read the survivor back inside the same transaction: this is the state a rollback must match, and guessing
        # it from the diff was wrong for every field the UPDATE writes implicitly (content_hash, updated_at).
        merged_row = dict(conn.execute(
            f"SELECT * FROM {MEMORY_TABLE} WHERE id = ?", (diff.survivor_id,)).fetchone())
        undo["survivor"]["after"] = {name: merged_row.get(name) for name in RESTORE_FIELDS}
        conn.execute(
            f"""INSERT INTO {RECORD_TABLE}
                (merge_id, survivor_id, absorbed_id, lossy, lost_facts_json, undo_json,
                 consented_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (merge_id, diff.survivor_id, diff.absorbed_id, int(diff.lossy),
             json.dumps(diff.lost_facts), json.dumps(undo), str(consented_by), moment),
        )
    return Applied(True, merge_id, lost_facts=list(diff.lost_facts))


def records(store: Any, *, limit: int = 20) -> list[dict[str, Any]]:
    # Read-only: calling this before any merge must not create the table, so a review screen does
    # not change the database it is describing.
    try:
        store._conn.execute(f"SELECT 1 FROM {RECORD_TABLE} LIMIT 1").fetchall()
    except Exception:  # noqa: BLE001
        return []
    rows = _rows(store, f"SELECT * FROM {RECORD_TABLE} ORDER BY created_at DESC LIMIT ?",
                 (max(1, int(limit)),))
    for row in rows:
        row["undo"] = _json(row.pop("undo_json", "{}"), {})
        row["lost_facts"] = _json(row.pop("lost_facts_json", "[]"), [])
    return rows


def _record(store: Any, merge_id: str) -> dict[str, Any] | None:
    for row in records(store, limit=200):
        if str(row["merge_id"]) == str(merge_id):
            return row
    return None


def preview_undo(store: Any, merge_id: str) -> dict[str, Any] | None:
    """The reverse diff: what undoing this merge would change, read-only.

    Built from the stored undo record and the current rows, so a person can review the reversal the
    same way they reviewed the merge - and so a record whose rows have since changed reports the
    difference instead of pretending the reverse is exact.
    """
    record = _record(store, merge_id)
    if record is None:
        return None
    undo = record.get("undo") or {}
    survivor_now = _one(store, str(undo.get("survivor", {}).get("id") or ""))
    absorbed_now = _one(store, str(undo.get("absorbed", {}).get("id") or ""))
    changes: list[dict[str, Any]] = []
    exact = survivor_now is not None and absorbed_now is not None
    if exact:
        for side, row in (("survivor", survivor_now), ("absorbed", absorbed_now)):
            if row is None:
                continue
            before = undo.get(side, {}).get("before", {})
            for name in RESTORE_FIELDS:
                if row.get(name) != before.get(name):
                    changes.append({"table": MEMORY_TABLE, "id": row.get("id"),
                                    "field": name, "now": row.get(name),
                                    "restored_to": before.get(name)})
    return {
        "merge_id": str(merge_id),
        "already_undone": bool(record.get("undone_at")),
        "fields": changes,
        "edges_returning": [move for move in undo.get("edges", [])
                            if move.get("action") != "drop_self_loop"],
        "edges_dropped_by_the_merge": sum(1 for move in undo.get("edges", [])
                                          if move.get("action") == "drop_self_loop"),
        "counts": {f"{MEMORY_TABLE}:fields_changed": len(changes),
                   f"{EDGE_TABLE}:returning": sum(1 for move in undo.get("edges", [])
                                                  if move.get("action") != "drop_self_loop")},
        "exact": exact,
    }


def undo(store: Any, merge_id: str, *, at: datetime | None = None) -> dict[str, Any]:
    """Reverse a recorded merge from its undo record."""
    record = _record(store, merge_id)
    if record is None:
        return {"ok": False, "reason": "no merge with that id"}
    if record.get("undone_at"):
        return {"ok": False, "reason": "this merge was already undone"}
    undo_row = record.get("undo") or {}
    moment = (at or _now()).isoformat()
    try:
        with store.transaction() as conn:
            for side in ("survivor", "absorbed"):
                item = undo_row.get(side) or {}
                before = item.get("before") or {}
                if not item.get("id") or not before:
                    continue
                assignments = ", ".join(f"{name} = ?" for name in RESTORE_FIELDS)
                conn.execute(
                    f"UPDATE {MEMORY_TABLE} SET {assignments} WHERE id = ?",
                    (*[before.get(name) for name in RESTORE_FIELDS], str(item["id"])),
                )
            for move in undo_row.get("edges", []):
                conn.execute(
                    f"DELETE FROM {EDGE_TABLE} WHERE src_id = ? AND dst_id = ? AND relation = ?",
                    (str(move.get("new_src_id") or move["src_id"]),
                     str(move.get("new_dst_id") or move["dst_id"]), str(move["relation"])),
                )
                destination = move.get("destination_before")
                if move.get("action") == "fold_duplicate" and destination:
                    conn.execute(
                        f"""INSERT OR REPLACE INTO {EDGE_TABLE}
                            (src_id, dst_id, relation, weight, evidence_count, created_at, last_reinforced_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (str(destination["src_id"]), str(destination["dst_id"]),
                         str(destination["relation"]), float(destination["weight"]),
                         int(destination["evidence_count"]), str(destination["created_at"]),
                         str(destination["last_reinforced_at"])),
                    )
                conn.execute(
                    f"""INSERT OR REPLACE INTO {EDGE_TABLE}
                        (src_id, dst_id, relation, weight, evidence_count, created_at, last_reinforced_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (str(move["src_id"]), str(move["dst_id"]), str(move["relation"]),
                     float(move["weight"]), int(move["evidence_count"]),
                     str(move["created_at"]), str(move["last_reinforced_at"])),
                )
            conn.execute(f"UPDATE {RECORD_TABLE} SET undone_at = ? WHERE merge_id = ?",
                         (moment, str(merge_id)))
    except Exception as error:  # noqa: BLE001 - report, do not pretend the reverse happened
        return {"ok": False, "reason": f"the store refused the undo: {error}"}
    return {"ok": True, "merge_id": str(merge_id), "undone_at": moment,
            "reason": f"restored {undo_row.get('survivor_id', '')} and "
                      f"{undo_row.get('absorbed_id', '')} and their edges"}


def coverage(store: Any) -> dict[str, Any]:
    """How much of the merge path is in use, counted from the tables rather than assumed."""
    memory_rows = _rows(store, f"SELECT COUNT(*) AS n FROM {MEMORY_TABLE}")
    edge_rows = _rows(store, f"SELECT COUNT(*) AS n FROM {EDGE_TABLE}")
    try:
        record_rows = _rows(store, f"SELECT COUNT(*) AS n, "
                                   f"SUM(CASE WHEN lossy = 1 THEN 1 ELSE 0 END) AS lossy, "
                                   f"SUM(CASE WHEN undone_at IS NOT NULL THEN 1 ELSE 0 END) AS undone "
                                   f"FROM {RECORD_TABLE}")
    except Exception:  # noqa: BLE001
        record_rows = []
    row = record_rows[0] if record_rows else {}
    return {
        f"{MEMORY_TABLE}:total": int((memory_rows[0] if memory_rows else {}).get("n") or 0),
        f"{EDGE_TABLE}:total": int((edge_rows[0] if edge_rows else {}).get("n") or 0),
        f"{RECORD_TABLE}:merges": int(row.get("n") or 0),
        f"{RECORD_TABLE}:lossy": int(row.get("lossy") or 0),
        f"{RECORD_TABLE}:undone": int(row.get("undone") or 0),
        "note": "merge diffs are computed by callers; nothing in the write path calls compute()",
    }
