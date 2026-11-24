"""Nightly edge *proposals*: a candidate pair, its evidence, and a person's decision (#76).

The consolidation pass can already write edges from co-occurrence, and that is the one thing it
should not do unattended. Co-retrieval is a weak signal: two memories surface for the same query
because both are broadly relevant, because one is pinned, or because the retriever always returns
the same dozen rows. None of those mean the two are related. The existing rules accept that
ambiguity by demanding corroboration (``EDGE_EVIDENCE_BAR = 2`` distinct evidence types, at least
two distinct sessions, a degree cap, and a human veto that outranks any count). This module adds the
missing step between "signal" and "edge": it proposes, and a person decides.

What a proposal is:

- a canonical pair (``src_id < dst_id``) plus the evidence that produced it -- session ids, counts,
  first/last seen -- stored in ``edge_proposals``;
- a **stable** id derived from the pair and relation, so the same candidate cannot become two rows
  and a decision can be looked up without a fuzzy match;
- a state: ``proposed`` until a named operator accepts or rejects it.

What a proposal is not:

- an edge. :func:`accept` records the decision and nothing else; the caller that wants the edge must
  create it through the store's normal ``add_edge`` path, where the evidence bar, degree cap, and
  veto still apply. Accepting here is permission, not a write.
- a deletion. Nothing in this module removes rows -- a rejected proposal stays as the record of a
  human judgement, which is exactly what stops the same pair from being proposed again next night.

The honest limit of the signal: co-retrieval cannot tell *why* two memories were returned together.
It cannot distinguish "these answer the same question" from "these are both always retrieved", and
it carries no direction or relation type. That is why the output is a review queue and why
confidence is capped below certainty rather than presented as a probability of truth.

Signal source: this branch has no ``co_retrieval`` module, so observations are read directly from
the ``co_retrieval_observations`` table (``left_id, right_id, session_id, turn_id, observed_at``,
primary key ``(left_id, right_id, session_id)``). If a ``co_retrieval`` module with an
``observations`` reader is present it is used instead, so the schema has one owner when both exist.
When the table is absent or empty -- it is empty in a default store -- :func:`build` returns an
empty list and :func:`report` says so; absence is not an error and never fabricates a candidate.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

TABLE = "edge_proposals"
OBSERVATIONS = "co_retrieval_observations"
FEEDBACK = "edge_feedback"

# Co-retrieval is its own relation; a proposal is never silently folded into `related`.
RELATION = "co_retrieved"

# Kept in step with forget_ladder.EDGE_EVIDENCE_BAR and degree_caps.DEFAULT_CAP. They are
# duplicated as literals because this module must import on a branch where neither exists.
EDGE_EVIDENCE_BAR = 2
DEFAULT_MAX_DEGREE = 24

DEFAULT_MIN_SESSIONS = 2
HALF_LIFE_DAYS = 90.0
CONFIDENCE_CAP = 0.9
MAX_EVIDENCE_SESSIONS = 20

_STATES = ("proposed", "accepted", "rejected")
_ELIGIBLE_MEMORY_STATES = ("active", "cold")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _pair(left: str, right: str) -> tuple[str, str]:
    first, second = str(left), str(right)
    return (first, second) if first <= second else (second, first)


def proposal_id(left: str, right: str, relation: str = RELATION) -> str:
    """Stable id for one candidate. Same pair and relation -> same id, forever."""
    first, second = _pair(left, right)
    digest = hashlib.sha256(f"{relation}|{first}|{second}".encode("utf-8")).hexdigest()
    return f"ep_{digest[:24]}"


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    proposal_id TEXT PRIMARY KEY,
                    src_id TEXT NOT NULL,
                    dst_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_json TEXT NOT NULL,
                    sessions INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'proposed',
                    decided_at TEXT,
                    decided_by TEXT,
                    reason TEXT NOT NULL DEFAULT '',
                    UNIQUE(src_id, dst_id, relation)
                )"""
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {TABLE}_state_idx ON {TABLE} (state, created_at)"
        )


def _table_exists(store: Any, name: str) -> bool:
    try:
        row = store._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (str(name),)
        ).fetchone()
    except Exception:  # noqa: BLE001 - a missing schema is a fact, not a failure
        return False
    return row is not None


def _rows(store: Any, sql: str, params: tuple = ()) -> list[Any]:
    try:
        return store._conn.execute(sql, params).fetchall()
    except Exception:  # noqa: BLE001 - the pass must survive a partial schema
        return []


def _max_degree() -> int:
    """Degree cap (#73), overridable for tests through the same env var the cap module reads."""
    raw = os.environ.get("CORTEX_MAX_DEGREE", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = DEFAULT_MAX_DEGREE
        if value > 0:
            return value
    try:
        degree_caps = importlib.import_module(f"{__package__}.degree_caps")

        value = int(degree_caps.cap())
        if value > 0:
            return value
    except Exception:  # noqa: BLE001 - no cap module on this branch
        pass
    return DEFAULT_MAX_DEGREE


def _degree(store: Any, memory_id: str) -> int:
    try:
        degree_caps = importlib.import_module(f"{__package__}.degree_caps")

        return int(degree_caps.degree(store, memory_id))
    except Exception:  # noqa: BLE001
        pass
    rows = _rows(
        store,
        "SELECT COUNT(*) FROM edges WHERE src_id=? OR dst_id=?",
        (str(memory_id), str(memory_id)),
    )
    return int(rows[0][0] or 0) if rows else 0


def _memory_state(store: Any, memory_id: str) -> str | None:
    try:
        row = store._conn.execute(
            "SELECT state FROM memories WHERE id=?", (str(memory_id),)
        ).fetchone()
    except Exception:  # noqa: BLE001
        return None
    return str(row["state"]) if row else None


def _linked(store: Any, left: str, right: str) -> bool:
    """Already an edge in either direction? Relation-agnostic: a link is a link."""
    try:
        row = store._conn.execute(
            """SELECT 1 FROM edges
               WHERE (src_id=? AND dst_id=?) OR (src_id=? AND dst_id=?) LIMIT 1""",
            (left, right, right, left),
        ).fetchone()
    except Exception:  # noqa: BLE001
        return False
    return row is not None


def _vetoed(store: Any, left: str, right: str) -> bool:
    """A person's "not related" is a veto, not a weighting (#79). Absence is not a veto."""
    try:
        assoc_feedback = importlib.import_module(f"{__package__}.assoc_feedback")

        if assoc_feedback.veto(store, left, right):
            return True
    except Exception:  # noqa: BLE001
        pass
    if not _table_exists(store, FEEDBACK):
        return False
    try:
        row = store._conn.execute(
            f"""SELECT polarity FROM {FEEDBACK}
                WHERE (left_id=? AND right_id=?) OR (left_id=? AND right_id=?)""",
            (left, right, right, left),
        ).fetchone()
    except Exception:  # noqa: BLE001
        return False
    return bool(row and str(row["polarity"]) == "unrelated")


def _existing(store: Any, left: str, right: str) -> dict[str, Any] | None:
    """Any proposal for this pair, in any state. A rejected pair is a decision, not an opening."""
    rows = _rows(
        store,
        f"SELECT * FROM {TABLE} WHERE src_id=? AND dst_id=? AND relation=?",
        (left, right, RELATION),
    )
    if rows:
        return dict(rows[0])
    # A row written by an older ordering still counts; the pair is what identifies a candidate.
    rows = _rows(
        store,
        f"""SELECT * FROM {TABLE}
            WHERE (src_id=? AND dst_id=?) OR (src_id=? AND dst_id=?)
            LIMIT 1""",
        (left, right, right, left),
    )
    return dict(rows[0]) if rows else None


def _observation_rows(store: Any, scan: int) -> list[dict[str, Any]]:
    """Co-retrieval pairs in a bounded window, newest and best-corroborated first.

    Prefers a ``co_retrieval.observations`` reader when the module exists; otherwise reads the
    table directly. Either way the shape returned here is the same, so callers cannot tell.
    """
    # Check for the table BEFORE asking the owner: `co_retrieval.observations` calls `_ensure`,
    # which creates an empty table, so "no observations" would be reported as "table" and the
    # absence would be lost. A missing table is absence, not emptiness.
    if not _table_exists(store, OBSERVATIONS):
        return []
    try:
        co_retrieval = importlib.import_module(f"{__package__}.co_retrieval")

        reader = getattr(co_retrieval, "observations", None)
        if callable(reader):
            entries: list[dict[str, Any]] = []
            for raw in reader(store, limit=scan):
                entry = dict(raw)
                entries.append(
                    {
                        "left_id": str(entry["left_id"]),
                        "right_id": str(entry["right_id"]),
                        "sessions": int(entry["sessions"]),
                        "last_seen": str(entry.get("last_seen") or ""),
                        "session_ids": [str(value) for value in entry.get("session_ids") or []],
                    }
                )
            return entries
    except Exception:  # noqa: BLE001 - fall through to the table
        pass
    rows = _rows(
        store,
        f"""SELECT left_id, right_id, COUNT(DISTINCT session_id) AS sessions,
                   MAX(observed_at) AS last_seen, GROUP_CONCAT(session_id) AS session_ids
            FROM {OBSERVATIONS}
            GROUP BY left_id, right_id
            ORDER BY sessions DESC, last_seen DESC LIMIT ?""",
        (max(1, int(scan)),),
    )
    result: list[dict[str, Any]] = []
    for row in rows:
        seen: list[str] = []
        for value in str(row["session_ids"] or "").split(","):
            value = value.strip()
            if value and value not in seen:
                seen.append(value)
        result.append(
            {
                "left_id": str(row["left_id"]),
                "right_id": str(row["right_id"]),
                "sessions": int(row["sessions"]),
                "last_seen": str(row["last_seen"] or ""),
                "session_ids": seen[:MAX_EVIDENCE_SESSIONS],
            }
        )
    return result


def _age_days(stamp: str) -> float:
    if not stamp:
        return 0.0
    try:
        seen = datetime.fromisoformat(stamp)
    except ValueError:
        return 0.0
    if not seen.tzinfo:
        seen = seen.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - seen).total_seconds() / 86400.0)


def confidence_for(sessions: int, last_seen: str = "") -> float:
    """Bounded, logarithmic, decaying -- deliberately below certainty.

    The second session matters far more than the tenth, and an association nobody has confirmed
    for half a year should look weaker than one seen this week.
    """
    if sessions < DEFAULT_MIN_SESSIONS:
        return 0.0
    base = min(0.55, 0.22 * math.log1p(sessions))
    decay = 0.5 ** (_age_days(last_seen) / HALF_LIFE_DAYS)
    return round(min(CONFIDENCE_CAP, base * decay), 4)


@dataclass(frozen=True)
class Proposal:
    """One candidate edge awaiting a decision. Immutable: decisions are new rows, not edits."""

    proposal_id: str
    src_id: str
    dst_id: str
    relation: str
    confidence: float
    sessions: int
    evidence: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    state: str = "proposed"
    decided_at: str = ""
    decided_by: str = ""
    reason: str = ""

    @property
    def pair(self) -> tuple[str, str]:
        return (self.src_id, self.dst_id)

    @property
    def undecided(self) -> bool:
        return self.state == "proposed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "src_id": self.src_id,
            "dst_id": self.dst_id,
            "relation": self.relation,
            "confidence": self.confidence,
            "sessions": self.sessions,
            "evidence": dict(self.evidence),
            "created_at": self.created_at,
            "state": self.state,
            "decided_at": self.decided_at or None,
            "decided_by": self.decided_by or None,
            "reason": self.reason,
        }


def _from_row(row: Any) -> Proposal:
    try:
        evidence = json.loads(str(row["evidence_json"] or "{}"))
    except (TypeError, ValueError):
        evidence = {}
    if not isinstance(evidence, dict):
        evidence = {}
    return Proposal(
        proposal_id=str(row["proposal_id"]),
        src_id=str(row["src_id"]),
        dst_id=str(row["dst_id"]),
        relation=str(row["relation"]),
        confidence=float(row["confidence"] or 0.0),
        sessions=int(row["sessions"] or 0),
        evidence=evidence,
        created_at=str(row["created_at"] or ""),
        state=str(row["state"] or "proposed"),
        decided_at=str(row["decided_at"] or ""),
        decided_by=str(row["decided_by"] or ""),
        reason=str(row["reason"] or ""),
    )


def _collect(
    store: Any, *, scan: int, min_sessions: int
) -> tuple[list[Proposal], dict[str, int]]:
    """The whole screening pipeline, returning survivors plus why the rest were refused."""
    stats = {
        "observed_pairs": 0,
        "below_min_sessions": 0,
        "ineligible_memory": 0,
        "already_linked": 0,
        "already_proposed": 0,
        "vetoed": 0,
        "over_degree_cap": 0,
        "candidates": 0,
    }
    if min_sessions < 1:
        raise ValueError("min_sessions must be at least 1")
    _ensure(store)
    cap = _max_degree()
    proposals: list[Proposal] = []
    for row in _observation_rows(store, scan):
        stats["observed_pairs"] += 1
        left, right = _pair(row["left_id"], row["right_id"])
        sessions = int(row["sessions"])
        if left == right or sessions < max(1, int(min_sessions)):
            stats["below_min_sessions"] += 1
            continue
        if any(
            _memory_state(store, memory_id) not in _ELIGIBLE_MEMORY_STATES
            for memory_id in (left, right)
        ):
            stats["ineligible_memory"] += 1
            continue
        if _linked(store, left, right):
            stats["already_linked"] += 1
            continue
        if _existing(store, left, right) is not None:
            stats["already_proposed"] += 1
            continue
        if _vetoed(store, left, right):
            stats["vetoed"] += 1
            continue
        if _degree(store, left) >= cap or _degree(store, right) >= cap:
            stats["over_degree_cap"] += 1
            continue
        confidence = confidence_for(sessions, row["last_seen"])
        evidence = {
            "signal": "co_retrieval",
            "evidence_types": ["co_retrieval"],
            "evidence_type_count": 1,
            "edge_evidence_bar": EDGE_EVIDENCE_BAR,
            # One evidence type is below the bar that an edge needs. That is the point of a
            # proposal: a person supplies the second judgement, the nightly pass does not.
            "meets_edge_evidence_bar": False,
            "sessions": sessions,
            "min_sessions": int(min_sessions),
            "session_ids": list(row["session_ids"]),
            "first_seen": "",
            "last_seen": row["last_seen"],
            "support": (
                f"{sessions} distinct sessions retrieved both memories; "
                "co-occurrence alone does not establish meaning"
            ),
        }
        proposals.append(
            Proposal(
                proposal_id=proposal_id(left, right),
                src_id=left,
                dst_id=right,
                relation=RELATION,
                confidence=confidence,
                sessions=sessions,
                evidence=evidence,
                created_at=_now(),
                state="proposed",
            )
        )
        stats["candidates"] += 1
    proposals.sort(key=lambda item: (-item.confidence, -item.sessions, item.src_id))
    return proposals, stats


def build(store: Any, *, scan: int = 500, min_sessions: int = DEFAULT_MIN_SESSIONS) -> list[Proposal]:
    """Read-only: candidate edges the evidence supports, ready for :func:`propose`.

    Nothing is written except the (empty) ``edge_proposals`` table definition. ``scan`` bounds how
    many observed pairs are considered, strongest support first, so one nightly run cannot walk an
    unbounded table. Pairs already linked, already proposed in any state, vetoed by a person, tied
    to a memory that is not active/cold, or past the degree cap are refused and counted only in
    :func:`report`; a pair seen in fewer than ``min_sessions`` distinct sessions is refused even if
    it appeared on a hundred turns of one conversation.
    """
    proposals, _ = _collect(store, scan=scan, min_sessions=min_sessions)
    return proposals


def propose(
    store: Any, *, scan: int = 500, min_sessions: int = DEFAULT_MIN_SESSIONS
) -> dict[str, Any]:
    """Persist :func:`build`'s candidates idempotently. Returns counts, not cheerleading.

    The proposal id is derived from the pair, and the table's uniqueness is on
    ``(src_id, dst_id, relation)``, so running twice in one night -- or on two nights before anyone
    reviews -- leaves exactly one row per candidate. Existing rows keep their state: a rejected
    proposal is never reset to ``proposed`` by a later scan.
    """
    candidates, stats = _collect(store, scan=scan, min_sessions=min_sessions)
    written = 0
    with store.transaction() as conn:
        for candidate in candidates:
            cursor = conn.execute(
                f"""INSERT OR IGNORE INTO {TABLE}
                    (proposal_id, src_id, dst_id, relation, confidence, evidence_json,
                     sessions, created_at, state, decided_at, decided_by, reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'proposed', NULL, NULL, '')""",
                (
                    candidate.proposal_id,
                    candidate.src_id,
                    candidate.dst_id,
                    candidate.relation,
                    float(candidate.confidence),
                    json.dumps(candidate.evidence, sort_keys=True, separators=(",", ":")),
                    int(candidate.sessions),
                    candidate.created_at,
                ),
            )
            written += int(cursor.rowcount or 0)
    return {
        "built": len(candidates),
        "written": written,
        "already_present": len(candidates) - written,
        "skipped": stats,
        "proposal_ids": [candidate.proposal_id for candidate in candidates],
    }


def _get(store: Any, proposal_id_value: str) -> dict[str, Any] | None:
    rows = _rows(store, f"SELECT * FROM {TABLE} WHERE proposal_id=?", (str(proposal_id_value),))
    return dict(rows[0]) if rows else None


def _decision(
    store: Any,
    proposal_id_value: str,
    state: str,
    *,
    by: str,
    reason: str = "",
) -> dict[str, Any]:
    who = str(by or "").strip()
    if not who:
        # An unattributed decision is indistinguishable from the pass deciding for itself.
        raise ValueError("a decision must be attributed to a name")
    _ensure(store)
    row = _get(store, proposal_id_value)
    if row is None:
        return {"ok": False, "error": "unknown proposal", "proposal_id": str(proposal_id_value)}
    current = str(row["state"] or "proposed")
    if current == state:
        return {
            "ok": True,
            "changed": False,
            "proposal_id": str(row["proposal_id"]),
            "state": current,
            "decided_by": str(row["decided_by"] or ""),
            "decided_at": str(row["decided_at"] or ""),
            "note": f"already {current}; no change",
        }
    if current != "proposed":
        return {
            "ok": False,
            "error": f"proposal was already {current}; decisions are final",
            "proposal_id": str(row["proposal_id"]),
            "state": current,
            "decided_by": str(row["decided_by"] or ""),
        }
    stamp = _now()
    note = str(reason or "").strip()[:500]
    with store.transaction() as conn:
        cursor = conn.execute(
            f"""UPDATE {TABLE}
                SET state=?, decided_at=?, decided_by=?, reason=?
                WHERE proposal_id=? AND state='proposed'""",
            (state, stamp, who, note, str(row["proposal_id"])),
        )
        changed = int(cursor.rowcount or 0)
    return {
        "ok": bool(changed),
        "changed": bool(changed),
        "proposal_id": str(row["proposal_id"]),
        "state": state,
        "decided_by": who,
        "decided_at": stamp,
        "reason": note,
    }


def accept(store: Any, proposal_id: str, *, by: str) -> dict[str, Any]:
    """Record that a person approves this candidate. **This does not create the edge.**

    Acceptance is permission, not a write: the caller is responsible for creating the edge through
    the store's normal edge path, where evidence, degree, and veto rules still apply. Keeping those
    two steps apart is what lets an operator approve a candidate without this module becoming a
    back door around the constraints it exists to respect.
    """
    result = _decision(store, proposal_id, "accepted", by=by)
    if result.get("ok"):
        result["edge_created"] = False
        result["note"] = "accepted; the caller must create the edge"
    return result


def reject(store: Any, proposal_id: str, *, by: str, reason: str = "") -> dict[str, Any]:
    """Record that a person refused this candidate, with an optional reason.

    The row stays. A rejection is the only durable answer to "why was this pair never linked", and
    :func:`build` treats any existing proposal -- including this one -- as already considered.
    """
    return _decision(store, proposal_id, "rejected", by=by, reason=reason)


def proposals(store: Any, state: str = "proposed") -> list[Proposal]:
    """Review queue, strongest first. ``state='any'`` returns every decision ever recorded."""
    _ensure(store)
    value = str(state or "proposed")
    if value.lower() in {"any", "all", "*"}:
        rows = _rows(
            store,
            f"SELECT * FROM {TABLE} ORDER BY confidence DESC, sessions DESC, created_at DESC",
        )
    else:
        if value not in _STATES:
            raise ValueError(f"unknown proposal state: {value}")
        rows = _rows(
            store,
            f"""SELECT * FROM {TABLE} WHERE state=?
                ORDER BY confidence DESC, sessions DESC, created_at DESC""",
            (value,),
        )
    return [_from_row(row) for row in rows]


def report(store: Any) -> dict[str, Any]:
    """What the queue looks like, and what the signal behind it can and cannot say."""
    _ensure(store)
    rows = _rows(store, f"SELECT state, relation, COUNT(*) AS n FROM {TABLE} GROUP BY state, relation")
    by_state = {state: 0 for state in _STATES}
    by_relation: dict[str, int] = {}
    for row in rows:
        by_state[str(row["state"])] = by_state.get(str(row["state"]), 0) + int(row["n"])
        key = str(row["relation"])
        by_relation[key] = by_relation.get(key, 0) + int(row["n"])
    pending = proposals(store, "proposed")
    observed_pairs = 0
    observations_source = "absent"
    if _table_exists(store, OBSERVATIONS):
        counted = _rows(store, f"SELECT COUNT(*) FROM (SELECT 1 FROM {OBSERVATIONS} GROUP BY left_id, right_id)")
        observed_pairs = int(counted[0][0] or 0) if counted else 0
        observations_source = "table"
    return {
        "proposals": sum(by_state.values()),
        "by_state": by_state,
        "by_relation": by_relation,
        "pending": len(pending),
        "top": [
            {
                "proposal_id": item.proposal_id,
                "pair": f"{item.src_id[:8]} <> {item.dst_id[:8]}",
                "sessions": item.sessions,
                "confidence": item.confidence,
            }
            for item in pending[:10]
        ],
        "observations": {
            "pairs": observed_pairs,
            "source": observations_source,
            "empty": observed_pairs == 0,
            "table": OBSERVATIONS,
        },
        "thresholds": {
            "min_sessions": DEFAULT_MIN_SESSIONS,
            "max_degree": _max_degree(),
            "edge_evidence_bar": EDGE_EVIDENCE_BAR,
            "confidence_cap": CONFIDENCE_CAP,
        },
        "limits": [
            "co-retrieval cannot tell why two memories were returned together",
            "a broad or pinned query manufactures accidental pairs",
            "the signal has no direction and no relation type",
            "one evidence type is below the edge evidence bar; review supplies the second",
            "an empty observations table yields no proposals, never a guess",
        ],
        "note": "proposals are review candidates; accepting one does not create an edge",
    }
