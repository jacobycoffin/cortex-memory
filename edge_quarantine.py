"""A new edge must earn two independent witnesses before it scores (#84).

`add_edge` is generous on purpose: one co-retrieval, one shared answer, one wikilink is enough to
record a relationship, and recording it is cheap. Letting it *score* immediately is where the trouble
starts — a single lucky co-occurrence becomes a permanent graph signal.

So the rule is a quarantine with a grandfather clause:

- an edge created **before** the adoption epoch keeps whatever standing it already had;
- an edge created **after** it is recorded but **non-scoring** until it has **two independent
  witnesses** — different evidence *types*, because two rows of the same type from the same source are
  one witness wearing two hats;
- a person can release a link by hand (`release()`), and the release is recorded.

The grandfather clause is not laziness, it is the measurement: on the live store **415 of 13,970
edges carry two distinct evidence types**. Applying the bar retroactively would silence 97% of the
graph in one commit, which is not a safety improvement, it is an outage.

Because the epoch is set explicitly and everything older is grandfathered, enabling this changes
nothing today. What it changes is the *next* thousand edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

TABLE = "edge_quarantine"
WITNESS_BAR = 2

# `add_edge` writes this row itself, and its own summary says it preserves no supporting evidence.
# Counting it as a witness would mean every new edge arrives half-admitted, which is the opposite of
# the point.
NON_WITNESS_TYPES = frozenset({"legacy_unattributed"})


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    key TEXT PRIMARY KEY,
                    adopted_at TEXT NOT NULL,
                    released_at TEXT,
                    released_by TEXT
                )"""
        )


def _key(src: str, dst: str, relation: str) -> str:
    return f"{src}|{dst}|{relation}"


def adopt(store: Any, *, at: datetime | None = None) -> str:
    """Set the epoch. Edges older than this are grandfathered; everything newer is quarantined."""
    _ensure(store)
    stamp = (at or _now()).isoformat()
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT INTO {TABLE} (key, adopted_at) VALUES ('__epoch__', ?)
                ON CONFLICT(key) DO UPDATE SET adopted_at = excluded.adopted_at""",
            (stamp,),
        )
    return stamp


def epoch(store: Any) -> datetime | None:
    try:
        row = store._conn.execute(
            f"SELECT adopted_at FROM {TABLE} WHERE key = '__epoch__'"
        ).fetchone()
    except Exception:  # noqa: BLE001 - not adopted yet
        return None
    if not row:
        return None
    raw = str(row["adopted_at"])
    try:
        stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def witnesses(store: Any, src_id: str, dst_id: str, relation: str) -> list[str]:
    """Independent evidence types behind an edge. Two rows of one type are one witness."""
    try:
        rows = store._conn.execute(
            """SELECT DISTINCT evidence_type FROM edge_evidence
               WHERE relation = ? AND evidence_type NOT IN ('legacy_unattributed')
                 AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
               ORDER BY evidence_type""",
            (relation, src_id, dst_id, dst_id, src_id),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    return [str(row["evidence_type"]) for row in rows]


def released(store: Any, src_id: str, dst_id: str, relation: str) -> bool:
    try:
        row = store._conn.execute(
            f"SELECT released_at FROM {TABLE} WHERE key = ?", (_key(src_id, dst_id, relation),)
        ).fetchone()
    except Exception:  # noqa: BLE001
        return False
    return bool(row and row["released_at"])


def release(store: Any, src_id: str, dst_id: str, relation: str, *, by: str = "user",
            at: datetime | None = None) -> bool:
    """A person vouches for a link. Recorded, so the quarantine is auditable."""
    _ensure(store)
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT INTO {TABLE} (key, adopted_at, released_at, released_by)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    released_at = excluded.released_at, released_by = excluded.released_by""",
            (_key(src_id, dst_id, relation), (at or _now()).isoformat(),
             (at or _now()).isoformat(), str(by)),
        )
    return True


@dataclass
class Status:
    trusted: bool
    reason: str
    witnesses: int

    def describe(self) -> str:
        return f"{'trusted' if self.trusted else 'quarantined'}: {self.reason}"


def status(store: Any, src_id: str, dst_id: str, relation: str,
           *, created_at: str | None = None) -> Status:
    """Does this edge contribute to scoring right now?"""
    stamp = epoch(store)
    if stamp is None:
        return Status(True, "no adoption epoch: everything is grandfathered", 0)
    created = _parse(created_at)
    if created is None:
        row = None
        try:
            # Either direction: add_edge stores the pair in canonical order, so looking up only
            # (src, dst) finds nothing and silently re-quarantines a grandfathered edge.
            row = store._conn.execute(
                """SELECT created_at FROM edges
                   WHERE relation = ? AND ((src_id = ? AND dst_id = ?)
                                            OR (src_id = ? AND dst_id = ?))""",
                (relation, src_id, dst_id, dst_id, src_id),
            ).fetchone()
        except Exception:  # noqa: BLE001
            row = None
        if row:
            created = _parse(str(row["created_at"]))
    if created is not None and created < stamp:
        return Status(True, "created before the adoption epoch", 0)
    if released(store, src_id, dst_id, relation):
        return Status(True, "released by a person", 0)
    found = witnesses(store, src_id, dst_id, relation)
    if len(found) >= WITNESS_BAR:
        return Status(True, f"{len(found)} independent witnesses", len(found))
    return Status(False, f"{len(found)} of {WITNESS_BAR} witnesses", len(found))


def _parse(text: Any) -> datetime | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def scoring_edges(store: Any, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter an edge list down to the ones allowed to contribute to a graph signal."""
    kept: list[dict[str, Any]] = []
    for row in rows:
        verdict = status(
            store, str(row.get("src_id")), str(row.get("dst_id")), str(row.get("relation")),
            created_at=row.get("created_at"),
        )
        if verdict.trusted:
            kept.append(row)
    return kept


def report(store: Any) -> dict[str, Any]:
    stamp = epoch(store)
    total = 0
    after = 0
    try:
        total = int(store._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] or 0)
        if stamp is not None:
            after = int(
                store._conn.execute(
                    "SELECT COUNT(*) FROM edges WHERE created_at >= ?", (stamp.isoformat(),)
                ).fetchone()[0] or 0
            )
    except Exception:  # noqa: BLE001
        pass
    return {
        "adopted_at": stamp.isoformat() if stamp else "",
        "edges": total,
        "grandfathered": total - after,
        "quarantined_candidates": after,
        "witness_bar": WITNESS_BAR,
    }
