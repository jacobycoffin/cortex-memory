"""The forget ladder: cool -> archive -> tombstone, keyed on last use and memory kind (#86).

Existing pruning in this repo is **model-driven**: `relevance_pruning` asks a provider to judge
each low-relevance memory and returns cool/archive/quarantine/keep. That is the right tool for
"is this still true or useful", and the wrong tool for "nobody has touched this in eight
months". This module is the deterministic half: a ladder that walks without a model, so it
works on a host with no provider, in a cron job, and reproducibly.

Three rules make it safe to run unattended:

1. **Kind sets the clock.** An episode is a working note about Tuesday; a procedure is how the
   release process works. Treating them the same is how a memory system forgets the thing it
   most needed to keep, so each kind carries its own dwell times and unresolved commitments and
   identity-shaped kinds never walk at all.
2. **One rung per sweep.** A memory that has been quiet for a year moves to `cold` this run and
   to `archived` the next. Skipping rungs would make a single bad sweep catastrophically
   destructive instead of merely wrong for one cycle, and every rung is checked against
   protection *again* on the way down.
3. **Nothing is destroyed.** `tombstoned` means unreachable, not shredded: the content stays,
   the id stays so edges do not dangle, and `restore()` walks it back up. Deletion already
   exists in the store's lifecycle trash and is a human action.

`last use` is `last_used_at`, falling back to `last_helpful_at` and then `created_at` — a fact
that is still informing answers must never be demoted for want of a new restatement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from . import dormancy

LADDER = ("active", "cold", "archived", "tombstoned")
_NEXT = {state: LADDER[index + 1] for index, state in enumerate(LADDER[:-1])}

POLICY_VERSION = "forget-ladder-v2"

# (days quiet before cool, before archive, before tombstone). Measured from last use.
#
# The windows are the declared policy, not a guess: episodes are working notes and go first
# (30/90/180d), semantic facts follow (90/180/365d), and the kinds that describe how things are
# done or who the user is are never walked automatically at all — a stale procedure is still
# the procedure, and losing it is the failure this whole ladder is supposed to avoid.
DWELL_DAYS: dict[str, tuple[int, int, int]] = {
    "episode": (30, 90, 180),
    "episodic": (30, 90, 180),
    "semantic": (90, 180, 365),
}
# Never walked automatically: preferences and procedures (declared policy), plus the kinds whose
# purpose is to be recalled later or that report a state of the world. Those end by being
# resolved or corrected, never by going quiet.
NEVER_WALK = frozenset(
    {
        "preference",
        "procedural",
        "procedure",
        "identity",
        "prospective",
        "commitment",
        "tool",
        "correction",
    }
)
DEFAULT_DWELL = (180, 365, 730)
# environment overrides, e.g. CORTEX_FORGET_EPISODE_DAYS="45,120,240"
_ENV_PREFIX = "CORTEX_FORGET_"


def _regret_policy() -> dict[str, Any]:
    from . import regret as _regret

    return {
        "enabled": _regret.enabled(),
        "step": _regret.STEP,
        "cap": _regret.CAP,
        "window_days": _regret.REGRET_WINDOW_DAYS,
        "note": "a kind that was forgotten and came back widens its own windows",
    }


def policy() -> dict[str, Any]:
    """The effective windows, so a run can say which policy it used."""
    return {
        "version": POLICY_VERSION,
        "regret_guard": _regret_policy(),
        "dwell_days": {kind: list(dwell) for kind, dwell in DWELL_DAYS.items()},
        "default_dwell": list(DEFAULT_DWELL),
        "never_walk": sorted(NEVER_WALK),
    }


def _env_dwell(kind: str) -> tuple[int, int, int] | None:
    raw = os.environ.get(f"{_ENV_PREFIX}{kind.upper()}_DAYS", "").strip()
    if not raw:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    try:
        values = tuple(int(float(part)) for part in parts)
    except ValueError:
        return None
    if len(values) != 3 or any(value < 0 for value in values):
        return None
    return values  # type: ignore[return-value]


def _utc(text: Any) -> datetime | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def last_use(row: dict[str, Any]) -> datetime | None:
    for key in ("last_used_at", "last_helpful_at", "last_confirmed_at", "created_at"):
        stamp = _utc(row.get(key))
        if stamp:
            return stamp
    return None


@dataclass(frozen=True)
class Step:
    memory_id: str
    kind: str
    from_state: str
    to_state: str
    quiet_days: float
    reason: str

    def describe(self, prefix: int = 8) -> str:
        return (
            f"{self.memory_id[:prefix]} {self.kind} {self.from_state}->{self.to_state} "
            f"({self.quiet_days:.0f}d quiet)"
        )


@dataclass
class Sweep:
    steps: list[Step] = field(default_factory=list)
    protected: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    protection_failures: dict[str, str] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for step in self.steps:
            out[f"{step.from_state}->{step.to_state}"] = out.get(
                f"{step.from_state}->{step.to_state}", 0
            ) + 1
        return out

    @property
    def summary(self) -> str:
        moves = ", ".join(f"{key} {value}" for key, value in sorted(self.counts().items()))
        return moves or "no memory qualified to move"


def _declared_dwell(kind: str) -> tuple[int, int, int]:
    name = str(kind or "").strip().lower()
    override = _env_dwell(name)
    if override is not None:
        return override
    return DWELL_DAYS.get(name, DEFAULT_DWELL)


def _dwell_for(
    kind: str, store: Any = None, *, regrets: dict[str, int] | None = None
) -> tuple[int, int, int]:
    """Declared windows, widened by measured regret (#90).

    `store` is optional so the pure policy stays testable without one; when it is passed, a kind
    that has been walked down and come back gets wider windows and the guard is visible in
    `policy()`/`explain()` rather than being an invisible adjustment.
    """
    declared = _declared_dwell(kind)
    if store is None:
        return declared
    from . import regret as _regret

    if not _regret.enabled():
        return declared
    counts = regrets if regrets is not None else _regret.revivals(store)
    multiplier = _regret.factor(store, kind, revivals_by_kind=counts)
    if multiplier <= 1.0:
        return declared
    return _regret.windows(declared, multiplier, floors=declared)



def _volatility_apply(
    store: Any, memory_id: str, dwell: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Scale the window by how fast this fact goes stale (#31).

    Durable facts get a longer window, ephemeral ones a shorter one. A missing class or a broken
    lookup leaves the declared policy alone - tuning must never be able to disable forgetting.
    """
    try:
        from . import volatility

        assigned = volatility.lookup(store, memory_id)
        if not assigned:
            return dwell
        return volatility.apply_to(dwell, assigned)
    except Exception:  # noqa: BLE001 - the ladder must run without the volatility table
        return dwell

def walks(kind: str) -> bool:
    """Whether this kind may ever be moved automatically."""
    return str(kind or "").strip().lower() not in NEVER_WALK


def _threshold(state: str, dwell: tuple[int, int, int]) -> int | None:
    index = LADDER.index(state) if state in LADDER else None
    if index is None or index >= len(LADDER) - 1:
        return None
    return dwell[index]


def _has_table(store: Any, name: str) -> bool:
    try:
        row = store._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
    except Exception:  # noqa: BLE001
        return False
    return row is not None


# Named protections (#88). Each is one query for the whole store, not one per memory, and each
# carries a name that appears in reports so "why did this survive" has an answer.
#
# An edge protects only when it is corroborated. The store holds ~14,000 edges, most of them
# `legacy_unattributed` similarity links; treating any edge as evidence would protect nearly
# every memory and turn the ladder into a no-op. `evidence_count >= 2` is the bar that made
# "a bare edge is not evidence" operational.
EDGE_EVIDENCE_BAR = 2
USE_EVIDENCE_DAYS = 90


def _edge_bar() -> int:
    raw = os.environ.get("CORTEX_FORGET_EDGE_EVIDENCE", "").strip()
    try:
        value = int(raw)
    except ValueError:
        return EDGE_EVIDENCE_BAR
    return value if value >= 1 else EDGE_EVIDENCE_BAR


def protection_policy() -> dict[str, Any]:
    return {
        "pinned": "a human decision outranks a timer",
        "open_commitment": "an open or due commitment still has to happen",
        "useful_feedback": "a resolved helpful/validated outcome is evidence the memory worked",
        "use_evidence": f"recalled and used within {USE_EVIDENCE_DAYS} days",
        "edge_bearing": f"connected by an edge with evidence_count >= {_edge_bar()}",
        "core_member": "currently rendered into the always-on profile",
    }


def protections(
    store: Any, *, failures: dict[str, str] | None = None
) -> dict[str, str]:
    """Memory id -> why it must not walk. One query per protection, not one per memory."""
    held: dict[str, str] = {}
    queries = (
        # (name, required table, sql). A table that does not exist yet is not a failure - the
        # store creates tables lazily - but a *present* table that errors is.
        # Legal hold comes first: it is the strongest statement, and `setdefault` would otherwise
        # report the "pinned" label that placing a hold also adds.
        (
            "legal_hold",
            "legal_holds",
            """SELECT memory_id AS id FROM legal_holds WHERE released_at IS NULL""",
        ),
        ("pinned", "memories",
         "SELECT id FROM memories WHERE pinned = 1 AND state IN ('active','cold','archived')"),
        (
            "open_commitment",
            "prospective_items",
            """SELECT DISTINCT p.memory_id AS id FROM prospective_items p
               JOIN memories m ON m.id = p.memory_id
               WHERE p.status IN ('open','due')""",
        ),
        (
            "core_member",
            "core_members",
            """SELECT DISTINCT c.memory_id AS id FROM core_members c
               JOIN memories m ON m.id = c.memory_id""",
        ),
        (
            "useful_feedback",
            "usage_records",
            """SELECT DISTINCT memory_id AS id FROM usage_records
               WHERE outcome IN ('helpful','validated')""",
        ),
        (
            "use_evidence",
            "access_log",
            """SELECT DISTINCT memory_id AS id FROM access_log
               WHERE event IN ('used','helpful','confirmed','validated')
                 AND created_at >= ?""",
        ),
        (
            "edge_bearing",
            "edges",
            f"""SELECT DISTINCT src_id AS id FROM edges WHERE evidence_count >= {_edge_bar()}
                UNION SELECT DISTINCT dst_id FROM edges
                WHERE evidence_count >= {_edge_bar()}""",
        ),
    )
    recent = (datetime.now(timezone.utc) - timedelta(days=USE_EVIDENCE_DAYS)).isoformat()
    for label, table, query in queries:
        if not _has_table(store, table):
            continue
        try:
            rows = store._conn.execute(
                query, (recent,) if "?" in query else ()
            ).fetchall()
        except Exception as error:  # noqa: BLE001 - a missing table must not block the sweep
            # Silent here once hid a typo'd column for an entire protection (src_id vs
            # source_id), so the failure is recorded and reported instead of swallowed.
            if failures is not None:
                failures[label] = str(error)
            continue
        for row in rows:
            held.setdefault(str(row["id"]), label)
    return held


def plan(
    store: Any,
    *,
    at: datetime | None = None,
    limit: int = 500,
    hold: dict[str, str] | None = None,
    max_per_transition: int = 100,
) -> Sweep:
    """Which memories move this sweep, at most one rung each.

    `max_per_transition` exists because a sweep is not a verdict: on a store where thousands of
    memories went quiet at once (an import, a years-old backlog), a single run would tombstone
    four hundred of them. Bounded, the same decision is made over several cycles, and anything
    that becomes used in between is spared.
    """
    now = at or datetime.now(timezone.utc)
    has_core = _has_table(store, "core_members")
    from . import regret as _regret

    regrets = _regret.revivals(store) if _regret.enabled() else {}
    failures: dict[str, str] = {}
    held = protections(store, failures=failures) if hold is None else hold
    sweep = Sweep(protection_failures=failures)
    skipped: dict[str, int] = {}

    rows = store._conn.execute(
        # `last_confirmed_at` lives on core_members, not memories: joining it keeps "still
        # confirmed but unused" from reading as "quiet". The table only exists once something
        # has been promoted, so the join is conditional rather than assumed.
        (
            """SELECT m.id, m.kind, m.state, m.pinned, m.importance, m.last_used_at,
                      m.last_helpful_at, c.last_confirmed_at, m.created_at,
                      m.retrieved_count, m.injected_count, m.used_count
               FROM memories m
               LEFT JOIN core_members c ON c.memory_id = m.id
               WHERE m.state IN ('active','cold','archived')
               ORDER BY COALESCE(m.last_used_at, m.last_helpful_at, m.created_at) ASC
               LIMIT ?"""
            if has_core
            else """SELECT m.id, m.kind, m.state, m.pinned, m.importance, m.last_used_at,
                           m.last_helpful_at, NULL AS last_confirmed_at, m.created_at,
                           m.retrieved_count, m.injected_count, m.used_count
                    FROM memories m
                    WHERE m.state IN ('active','cold','archived')
                    ORDER BY COALESCE(m.last_used_at, m.last_helpful_at, m.created_at) ASC
                    LIMIT ?"""
        ),
        (max(limit * 4, 2000),),
    ).fetchall()

    per_transition: dict[str, int] = {}
    for raw in rows:
        if len(sweep.steps) >= limit:
            break
        row = dict(raw)
        memory_id = str(row["id"])
        kind = str(row.get("kind") or "").strip().lower()
        if kind in NEVER_WALK:
            skipped["kind_never_walks"] = skipped.get("kind_never_walks", 0) + 1
            continue
        if memory_id in held:
            label = held[memory_id]
            sweep.protected[label] = sweep.protected.get(label, 0) + 1
            continue
        state = str(row.get("state") or "active")
        next_state = _NEXT.get(state)
        if next_state is None:
            continue
        # Dormancy narrows the window for memories that were offered many times and never used
        # (#94). It runs after regret, so a kind that has been forgotten wrongly widens first.
        assessment = dormancy.assess(row)
        dwell = _volatility_apply(store, memory_id, _dwell_for(kind, store, regrets=regrets))
        dwell = dormancy.apply_to(dwell, row)
        threshold = _threshold(state, dwell)
        used = last_use(row)
        # Variant phrasings of one fact age together (#56), when switched on: if a twin was useful
        # last week the fact was useful last week, whichever words were in front of it. Off by
        # default, and a failure returns the member's own date.
        try:
            from . import fact_families

            used = fact_families.apply_to(store, memory_id, used)
        except Exception:  # noqa: BLE001 - family ageing must never stop a sweep
            pass
        if threshold is None or used is None:
            skipped["no_usable_timestamp"] = skipped.get("no_usable_timestamp", 0) + 1
            continue
        quiet_days = (now - used).total_seconds() / 86400.0
        if quiet_days < threshold:
            skipped["still_warm"] = skipped.get("still_warm", 0) + 1
            continue
        transition = f"{state}->{next_state}"
        if per_transition.get(transition, 0) >= max_per_transition:
            skipped[f"capped:{transition}"] = skipped.get(f"capped:{transition}", 0) + 1
            continue
        per_transition[transition] = per_transition.get(transition, 0) + 1
        sweep.steps.append(
            Step(
                memory_id=memory_id,
                kind=kind or "unknown",
                from_state=state,
                to_state=next_state,
                quiet_days=quiet_days,
                reason=(
                    f"{kind or 'unknown'} quiet {quiet_days:.0f}d at or past the "
                    f"{threshold}d {state}->{next_state} threshold"
                    + (
                        f" (window halved by dormancy: {assessment.retrieved} offers, "
                        f"{assessment.used} uses)"
                        if dormancy.scale(row) != 1.0
                        else ""
                    )
                ),
            )
        )
    sweep.skipped = skipped
    return sweep


def apply(store: Any, sweep: Sweep) -> dict[str, int]:
    """Move one rung per planned memory. Returns per-transition counts."""
    applied: dict[str, int] = {}
    moved_ids: list[str] = []
    for step in sweep.steps:
        try:
            store.set_state(
                step.memory_id,
                step.to_state,
                reason=f"forget ladder: {step.reason}",
            )
        except Exception:  # noqa: BLE001 - one failed move must not stop the sweep
            continue
        key = f"{step.from_state}->{step.to_state}"
        applied[key] = applied.get(key, 0) + 1
        moved_ids.append(step.memory_id)
    if moved_ids:
        from . import prune_receipts

        prune_receipts.record(
            store, "forget_ladder", "walk", moved_ids,
            summary=", ".join(f"{key} {value}" for key, value in sorted(applied.items())),
        )
    return applied


def sweep(store: Any, *, at: datetime | None = None, apply_steps: bool = False,
          limit: int = 500) -> tuple[Sweep, dict[str, int]]:
    planned = plan(store, at=at, limit=limit)
    applied = apply(store, planned) if apply_steps else {}
    return planned, applied


def restore(store: Any, memory_id: str, *, to: str = "active") -> bool:
    """Walk a memory back up the ladder. Tombstone is not a shredder.

    Delegates to `restore_ladder`, which counts restores and pins a memory that has come back
    twice — one implementation, so a caller cannot restore without the counting (#99).
    """
    from .restore_ladder import restore as restore_and_count

    return restore_and_count(store, memory_id, to=to).restored


def explain(store: Any, memory_id: str, *, at: datetime | None = None) -> dict[str, Any]:
    """Why this memory is or is not eligible to move: every protection checked, named."""
    row = store.get_memory(memory_id)
    if not row:
        return {"memory_id": memory_id, "found": False}
    row = dict(row)
    failures: dict[str, str] = {}
    held = protections(store, failures=failures)
    kind = str(row.get("kind") or "").strip().lower()
    state = str(row.get("state") or "active")
    dwell = _dwell_for(kind, store)
    threshold = _threshold(state, dwell)
    used = last_use(row)
    now = at or datetime.now(timezone.utc)
    quiet = (now - used).total_seconds() / 86400.0 if used else None
    if not walks(kind):
        verdict = "held: this kind never walks automatically"
    elif memory_id in held:
        verdict = f"held: {held[memory_id]}"
    elif threshold is None:
        verdict = "held: already at the end of the ladder"
    elif quiet is None:
        verdict = "held: no usable timestamp"
    elif quiet < threshold:
        verdict = f"warm: {quiet:.0f}d quiet, needs {threshold}d"
    else:
        verdict = f"eligible: {quiet:.0f}d quiet at or past the {threshold}d threshold"
    return {
        "memory_id": memory_id,
        "found": True,
        "kind": kind,
        "state": state,
        "next_state": _NEXT.get(state),
        "dwell_days": list(dwell),
        "threshold_days": threshold,
        "quiet_days": round(quiet, 1) if quiet is not None else None,
        "protections": held.get(memory_id),
        "verdict": verdict,
        "protection_failures": failures,
    }


def state_counts(store: Any) -> dict[str, int]:
    rows = store._conn.execute(
        "SELECT state, COUNT(*) AS n FROM memories GROUP BY state ORDER BY state"
    ).fetchall()
    return {str(row["state"]): int(row["n"]) for row in rows}


def kind_counts(store: Any) -> dict[tuple[str, str], int]:
    rows = store._conn.execute(
        """SELECT COALESCE(kind,'') AS kind, state, COUNT(*) AS n FROM memories
           GROUP BY COALESCE(kind,''), state"""
    ).fetchall()
    return {(str(row["kind"]), str(row["state"])): int(row["n"]) for row in rows}
