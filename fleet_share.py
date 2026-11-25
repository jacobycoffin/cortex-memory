"""Promoting general knowledge to a shared layer, one review at a time (#155).

A fleet shares what is *general* — how a tool behaves, what a version changed, which approach failed — and
never what is particular to one person or one machine. The failure this module prevents is the one that has no
undo button once it happens: a memory leaves the box and is quoted back later, having carried a secret, a
device name, or somebody's medical detail with it.

Five rules, each a way a sharing path leaks:

- **nothing is shared without an approval row.** A proposal is evidence, not permission: `propose()` only
  records what *would* go and what would not, and `approve()` is the single door to a bundle. The default for
  every memory is "not shared" and there is no batch approve-anything call.
- **the eligibility test fails closed.** A memory is ineligible unless it is proven general: sensitive domains
  (credential, finance, health, legal, third-party personal) refuse, a project-scoped or device-scoped memory
  refuses, and so does a memory nobody can classify — "we could not tell" is not permission to publish.
- **the bundle says what was withheld, and why.** A shared layer that silently drops half a proposal teaches
  the operator to distrust the count; the payload carries `withheld` with the reason per item.
- **redaction happens at export, and a credential-only memory is refused rather than redacted.** A redacted
  credential still says where it lives.
- **approval is bound to the text that was reviewed.** The proposal records a hash of the prepared text; if
  the memory changed between proposal and approval, approval refuses as stale, because the reviewer approved
  something that is no longer there.

The shared layer itself is out of scope here: this module produces a bundle a sync job can carry, and refuses
to be the thing that decides what a different machine may see.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from . import share_export

PROPOSAL_TABLE = "fleet_shares"
MEMORY_TABLE = "memories"

# Domains that may never be shared, whatever else is true of them. The names match `sensitive_policy` so a
# reader can compare the two lists and see they are the same list.
SENSITIVE_DOMAINS = frozenset({"credential", "finance", "health", "legal", "third_party_personal"})

PROPOSED = "proposed"
APPROVED = "approved"
REJECTED = "rejected"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]] | None:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return None


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {PROPOSAL_TABLE} (
                    proposal_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    eligibility TEXT NOT NULL,
                    refusal TEXT NOT NULL DEFAULT '',
                    text_hash TEXT NOT NULL DEFAULT '',
                    shared_text TEXT NOT NULL DEFAULT '',
                    labels_json TEXT NOT NULL DEFAULT '[]',
                    withheld_json TEXT NOT NULL DEFAULT '[]',
                    proposed_at TEXT NOT NULL,
                    reviewed_at TEXT,
                    reviewer TEXT,
                    review_reason TEXT
                )"""
        )


def _classify(text: str) -> str:
    """The domain of a memory, using the policy module's detector when it is importable."""
    try:
        from . import sensitive_policy

        return str(sensitive_policy.detect_domain(str(text or "")))
    except Exception:  # noqa: BLE001
        return "unrecognised"


@dataclass
class Proposal:
    memory_id: str = ""
    proposal_id: str = ""
    eligible: bool = False
    reason: str = ""
    shared_text: str = ""
    labels: list[str] = field(default_factory=list)
    withheld: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    domain: str = ""
    state: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "proposal_id": self.proposal_id,
            "eligible": self.eligible,
            "reason": self.reason,
            "shared_text": self.shared_text,
            "labels": self.labels,
            "withheld": self.withheld,
            "notes": self.notes,
            "domain": self.domain,
            "state": self.state,
        }

    def render(self) -> list[str]:
        head = (f"proposal {self.proposal_id} for {self.memory_id[:8]}: "
                f"{'eligible' if self.eligible else 'NOT eligible'}")
        lines = [head]
        if self.reason:
            lines.append(f"  reason: {self.reason}")
        if self.domain:
            lines.append(f"  domain: {self.domain}")
        if self.labels:
            lines.append(f"  redacted: {', '.join(self.labels)}")
        for item in self.withheld:
            lines.append(f"  withheld {item['field']}: {item['why']}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return lines


def _scope_of(row: dict[str, Any]) -> dict[str, Any]:
    try:
        scope = json.loads(str(row.get("scope_json") or "{}"))
    except Exception:  # noqa: BLE001
        return {}
    return scope if isinstance(scope, dict) else {}


def propose(store: Any, memory_id: str, *, reason: str = "") -> Proposal:
    """Record what sharing this memory would move, without moving it."""
    proposal = Proposal(memory_id=str(memory_id or ""))
    if not str(reason).strip():
        proposal.reason = "a proposal needs a stated reason; sharing has no default"
        return proposal
    try:
        _ensure(store)
    except Exception:  # noqa: BLE001 - an unusable store is a refusal, not a traceback in a view
        proposal.reason = "the store could not be read"
        return proposal
    rows = _rows(store, f"SELECT id, content, scope_json, state, source_category FROM {MEMORY_TABLE} "
                        "WHERE id = ?", (proposal.memory_id,))
    if rows is None:
        proposal.reason = "the store could not be read"
        return proposal
    if not rows:
        proposal.reason = f"unknown memory: no row in {MEMORY_TABLE} with that id"
        return proposal
    row = rows[0]

    content = str(row.get("content") or "")
    proposal.domain = _classify(content)
    withheld: list[dict[str, str]] = []

    scope = _scope_of(row)
    if scope.get("project"):
        proposal.reason = "memory is scoped to a project, so it is not general knowledge"
        proposal.withheld.append({"field": "content", "why": "project scope"})
        return proposal
    if scope.get("device"):
        proposal.reason = "memory is scoped to a device, so sharing it would name that device"
        proposal.withheld.append({"field": "content", "why": "device scope"})
        return proposal
    if proposal.domain in SENSITIVE_DOMAINS:
        proposal.reason = f"domain {proposal.domain!r} is never shared"
        proposal.withheld.append({"field": "content", "why": f"{proposal.domain} domain"})
        return proposal
    # `unrecognised` is NOT a refusal: measured against the live store, ordinary technical statements
    # ("proxmox 8.4 renamed the pveproxy service unit") classify as unrecognised, so refusing on it would
    # block every piece of general knowledge. It is carried to the reviewer as a flag instead - the human
    # approval is the gate, and pretending the classifier is one would be a lie in the other direction.
    if proposal.domain == "unrecognised":
        # A note, not a withheld item: nothing has been taken away from this proposal, the reviewer is
        # simply told the classifier had nothing to say about it.
        proposal.notes.append("classifier could not place this memory; review closely")

    prepared = share_export.prepare(content)
    if prepared.refused:
        proposal.reason = prepared.refused
        proposal.withheld.append({"field": "content", "why": "credential-only"})
        return proposal
    proposal.labels = list(prepared.labels) + list(prepared.genericised)
    proposal.shared_text = prepared.text
    proposal.eligible = True
    proposal.reason = reason.strip()
    proposal.state = PROPOSED
    proposal.proposal_id = hashlib.sha256(
        json.dumps([proposal.memory_id, proposal.shared_text, proposal.reason],
                   sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    with store.transaction() as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO {PROPOSAL_TABLE} (proposal_id, memory_id, state, eligibility, refusal, "
            "text_hash, shared_text, labels_json, withheld_json, proposed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (proposal.proposal_id, proposal.memory_id, PROPOSED, "eligible", "",
             _hash(proposal.shared_text), proposal.shared_text, json.dumps(proposal.labels),
             json.dumps(proposal.withheld), _now().isoformat()),
        )
    return proposal


def _hash(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:16]


@dataclass
class Review:
    ok: bool = False
    status: str = ""  # approved | rejected | already | stale | unknown | refused
    reason: str = ""
    proposal_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "status": self.status, "reason": self.reason,
                "proposal_id": self.proposal_id}

    def render(self) -> list[str]:
        return [f"{self.status}: {self.reason}"]


def review(store: Any, proposal_id: str, *, decision: str, reviewer: str = "",
           reason: str = "") -> Review:
    """Approve or reject one proposal. Approval is the only door to a shared bundle."""
    _ensure(store)
    out = Review(proposal_id=str(proposal_id))
    if not str(reviewer).strip():
        out.status, out.reason = "refused", "a review needs a reviewer; an anonymous approval is not a review"
        return out
    rows = _rows(store, f"SELECT * FROM {PROPOSAL_TABLE} WHERE proposal_id = ?", (out.proposal_id,))
    if rows is None:
        out.status, out.reason = "refused", "the proposal table could not be read"
        return out
    if not rows:
        out.status, out.reason = "unknown", f"no proposal {proposal_id}"
        return out
    row = rows[0]
    if str(row.get("state")) == APPROVED and decision == "approve":
        out.status, out.reason = "already", "already approved"
        return out
    if str(row.get("state")) == REJECTED and decision == "reject":
        out.status, out.reason = "already", "already rejected"
        return out
    if decision not in {"approve", "reject"}:
        out.status, out.reason = "refused", f"unknown decision {decision!r}; approve or reject"
        return out

    memory = _rows(store, f"SELECT content FROM {MEMORY_TABLE} WHERE id = ?", (str(row.get("memory_id")),))
    if memory is None:
        out.status, out.reason = "refused", "the memory could not be read"
        return out
    if not memory:
        out.status, out.reason = "stale", "the memory is gone; there is nothing left to share"
        return out
    current = share_export.prepare(str(memory[0].get("content") or ""))
    if _hash(current.text) != str(row.get("text_hash")):
        out.status = "stale"
        out.reason = ("the memory changed since this proposal; the reviewer approved text that is no longer "
                      "there, so propose it again")
        return out

    state = APPROVED if decision == "approve" else REJECTED
    with store.transaction() as conn:
        conn.execute(
            f"UPDATE {PROPOSAL_TABLE} SET state = ?, reviewed_at = ?, reviewer = ?, review_reason = ? "
            "WHERE proposal_id = ?",
            (state, _now().isoformat(), reviewer.strip(), reason.strip(), out.proposal_id),
        )
    out.ok = True
    out.status = state
    out.reason = f"{decision}d by {reviewer.strip()}"
    return out


def bundle(store: Any) -> dict[str, Any]:
    """What an approved set would move to the shared layer - including what was withheld and why."""
    _ensure(store)
    rows = _rows(store, f"SELECT * FROM {PROPOSAL_TABLE} ORDER BY proposed_at")
    if rows is None:
        return {"readable": False, "items": [], "withheld": [], "reason": "the proposal table could not be read"}
    items: list[dict[str, Any]] = []
    withheld: list[dict[str, Any]] = []
    for row in rows or []:
        state = str(row.get("state"))
        if state != APPROVED:
            if state == REJECTED:
                withheld.append({"memory_id": row.get("memory_id"), "why": "rejected in review"})
            else:
                withheld.extend({
                    "memory_id": row.get("memory_id"),
                    "why": item.get("why", "ineligible"),
                    "field": item.get("field", "content"),
                } for item in json.loads(str(row.get("withheld_json") or "[]")))
            continue
        items.append({
            "memory_id": row.get("memory_id"),
            "text": row.get("shared_text"),
            "labels": json.loads(str(row.get("labels_json") or "[]")),
            "approved_by": row.get("reviewer"),
        })
    return {
        "readable": True,
        "items": items,
        "withheld": withheld,
        "counts": {"approved": len(items), "withheld": len(withheld)},
    }


def proposals(store: Any, *, state: str = "", limit: int = 50) -> list[dict[str, Any]]:
    _ensure(store)
    if state:
        rows = _rows(store, f"SELECT * FROM {PROPOSAL_TABLE} WHERE state = ? ORDER BY proposed_at DESC LIMIT ?",
                     (state, int(limit)))
    else:
        rows = _rows(store, f"SELECT * FROM {PROPOSAL_TABLE} ORDER BY proposed_at DESC LIMIT ?", (int(limit),))
    return rows or []
