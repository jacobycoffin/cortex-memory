"""A memory is current, superseded, or disputed (#48).

Three states, because two is not enough. A store with only current/superseded has to pick a
winner whenever two sources disagree, and picking is a coin flip dressed as a fact: "the
maintenance window is Sunday" and "the maintenance window is Tuesday" cannot both be current, but
choosing one because it was written a day later is worse than saying both are held and letting the
answer hedge.

So a **dispute** is a first-class record:

- it names the pair, so neither is silently preferred;
- it does **not** demote either side — a dispute is not evidence that one is wrong, just that the
  store has not resolved which is right;
- `resolve(winner=...)` turns it into an ordinary supersession, which *is* a claim about truth and
  therefore has to be made deliberately.

Supersession already exists (`validity.superseded_ids` reads `memories.supersedes_id`); this module
does not duplicate it — it reports the three states together so a caller never has to guess which
mechanism applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

_TABLE = "disputes"
DISPUTED = "disputed"
SUPERSEDED = "superseded"
CURRENT = "current"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {_TABLE} (
                    left_id TEXT NOT NULL,
                    right_id TEXT NOT NULL,
                    reason TEXT,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT,
                    winner_id TEXT,
                    PRIMARY KEY (left_id, right_id)
                )"""
        )


def _pair(left: str, right: str) -> tuple[str, str]:
    """One row per unordered pair, so marking the same dispute twice cannot double-count."""
    first, second = str(left), str(right)
    return (first, second) if first <= second else (second, first)


@dataclass
class Dispute:
    left_id: str
    right_id: str
    reason: str
    created_at: str

    def describe(self, width: int = 8) -> str:
        return f"{self.left_id[:width]} <> {self.right_id[:width]}: {self.reason}"


def mark(store: Any, left: str, right: str, *, reason: str = "", at: Any = None) -> Dispute | None:
    """Record that these two memories disagree. Returns None for a self-dispute or unknown id."""
    _ensure(store)
    first, second = _pair(store.resolve_id(str(left)) or str(left),
                          store.resolve_id(str(right)) or str(right))
    if first == second:
        return None
    if not store.get_memory(first) or not store.get_memory(second):
        return None
    stamp = (at or _utc_now()).isoformat()
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT INTO {_TABLE}(left_id, right_id, reason, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(left_id, right_id) DO UPDATE SET
                    reason = excluded.reason,
                    created_at = excluded.created_at,
                    resolved_at = NULL,
                    winner_id = NULL""",
            (first, second, reason, stamp),
        )
    return Dispute(left_id=first, right_id=second, reason=reason, created_at=stamp)


def resolve(
    store: Any,
    left: str,
    right: str,
    *,
    winner: str | None = None,
    reason: str = "resolved",
    at: Any = None,
) -> bool:
    """Close a dispute. With a winner, the loser is superseded — an explicit claim about truth."""
    _ensure(store)
    first, second = _pair(store.resolve_id(str(left)) or str(left),
                          store.resolve_id(str(right)) or str(right))
    row = store._conn.execute(
        f"SELECT 1 FROM {_TABLE} WHERE left_id = ? AND right_id = ? AND resolved_at IS NULL",
        (first, second),
    ).fetchone()
    if row is None:
        return False
    stamp = (at or _utc_now()).isoformat()
    with store.transaction() as conn:
        conn.execute(
            f"UPDATE {_TABLE} SET resolved_at = ?, winner_id = ? WHERE left_id = ? AND right_id = ?",
            (stamp, winner, first, second),
        )
    if winner:
        loser = second if str(winner) == first else first
        try:
            with store.transaction() as conn:
                # `supersedes_id` reads "this memory supersedes that one", so the WINNER row
                # points at the loser (#48). Written the other way round it marks the loser as
                # the superseding memory, which is a silent inversion of the fact.
                conn.execute(
                    "UPDATE memories SET supersedes_id = ? WHERE id = ?",
                    (loser, str(winner)),
                )
        except Exception:  # noqa: BLE001 - supersession is an enhancement, not a requirement
            pass
    return True


def pairs(store: Any, *, include_resolved: bool = False, limit: int = 200) -> list[Dispute]:
    _ensure(store)
    clause = "" if include_resolved else "WHERE resolved_at IS NULL"
    rows = store._conn.execute(
        f"""SELECT left_id, right_id, reason, created_at FROM {_TABLE} {clause}
            ORDER BY created_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [
        Dispute(str(row["left_id"]), str(row["right_id"]), str(row["reason"] or ""),
                str(row["created_at"]))
        for row in rows
    ]


def disputed_ids(store: Any) -> set[str]:
    """Every memory currently in an open dispute."""
    _ensure(store)
    try:
        rows = store._conn.execute(
            f"""SELECT left_id, right_id FROM {_TABLE} WHERE resolved_at IS NULL"""
        ).fetchall()
    except Exception:  # noqa: BLE001 - an unreadable table means "nothing disputed"
        return set()
    found: set[str] = set()
    for row in rows:
        found.add(str(row["left_id"]))
        found.add(str(row["right_id"]))
    return found


def status(store: Any, memory_id: str) -> str:
    """`disputed`, `superseded`, or `current` — in that order of precedence."""
    from .validity import superseded_ids

    resolved = store.resolve_id(memory_id) or memory_id
    if resolved in disputed_ids(store):
        return DISPUTED
    if resolved in superseded_ids(store):
        return SUPERSEDED
    return CURRENT


def statuses(store: Any, memory_ids: list[str]) -> dict[str, str]:
    """Status for a batch, computed with one dispute query rather than one per memory."""
    from .validity import superseded_ids

    disputed = disputed_ids(store)
    superseded = superseded_ids(store)
    out: dict[str, str] = {}
    for memory_id in memory_ids:
        resolved = str(memory_id)
        if resolved in disputed:
            out[resolved] = DISPUTED
        elif resolved in superseded:
            out[resolved] = SUPERSEDED
        else:
            out[resolved] = CURRENT
    return out


def hedge(store: Any, memory_ids: list[str]) -> str:
    """A one-line note for the answer, when the recalled set contains a dispute.

    Empty when nothing is disputed — a hedge that appears unprompted trains people to ignore it.
    """
    state = statuses(store, memory_ids)
    disputed = [memory_id for memory_id, value in state.items() if value == DISPUTED]
    if not disputed:
        return ""
    return (
        f"{len(disputed)} of the recalled memories are disputed: "
        f"{', '.join(memory_id[:8] for memory_id in disputed[:5])} — sources disagree, "
        "so treat the specifics as uncertain rather than settled"
    )


def report(store: Any, *, limit: int = 20) -> dict[str, Any]:
    open_pairs = pairs(store, limit=limit)
    return {
        "open": len(pairs(store, limit=10_000)),
        "pairs": [dispute.describe() for dispute in open_pairs],
        "memories": len(disputed_ids(store)),
    }
