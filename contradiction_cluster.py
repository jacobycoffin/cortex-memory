"""Group conflicting facts into one review item instead of N pairwise ones (#52).

`dispute.py` records a conflict per pair, which is right for the record and wrong for the reader: five
statements about the same thing produce ten pairs, and a person is asked to resolve ten questions that
are really one. The pairs are facts about the store; the **cluster** is the question.

So the pairs are collapsed with a union-find over the dispute graph, one item per connected component:

- the **topic** is the shared subject/predicate when the memories carry one, and otherwise the terms
  they all share — a heading the person can recognise, not a list of ids;
- the **recommendation** names the strongest member by confidence and says why, but nothing is ever
  auto-resolved: a store that picks a winner without being asked is a store that can be wrong quietly;
- **resolving an item resolves every pair in it**, in one recording, with the actor's name attached.

A dispute with no partner stays a singleton and is reported as one — a single conflicting pair is
still a review item, it is just not a cluster.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

STOPWORDS = frozenset(
    "the a an is are was were be been being it its of to in on for and or but not with from by as at "
    "if then than so just very now this that these those there here".split()
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def pairs(store: Any) -> list[dict[str, Any]]:
    """Every unresolved dispute, as recorded."""
    return _rows(
        store,
        "SELECT left_id, right_id, reason, created_at, winner_id, resolved_at "
        "FROM disputes ORDER BY created_at",
    )


def _tokens(text: str) -> set[str]:
    return {
        word for word in re.split(r"[^a-z0-9]+", str(text or "").lower())
        if len(word) > 2 and word not in STOPWORDS
    }


def _memories(store: Any, ids: list[str]) -> dict[str, dict[str, Any]]:
    if not ids:
        return {}
    clause = ",".join("?" for _ in ids)
    rows = _rows(
        store,
        f"""SELECT id, content, kind, confidence, importance, subject, predicate, state, created_at
            FROM memories WHERE id IN ({clause})""",
        tuple(ids),
    )
    return {str(row["id"]): row for row in rows}


# A pair distinguished only by a timestamp, a date, a bare number or a filename is the same claim
# written twice at different times — not a contradiction. Inspecting the live store's near-miss
# proposals at a wide scan, every one of them had exactly this shape ("18:48" vs "21:03",
# "10 + NN%" vs "10 ticks + NN%", two daily-note paths), so without this filter, enabling recording
# would have filed forty review items of noise.
# A bare number is deliberately NOT here: "8100" vs "8200" is a changed port and exactly the kind of
# conflict worth reviewing. Only labels that say *when* or *where* a statement was written are
# dropped, because those describe the note rather than the world.
TRIVIAL = (
    re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$"),                     # a time of day
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),                          # an ISO date
    re.compile(r"\d{1,2}:\d{2}"),                                  # a time inside a longer token
    re.compile(r"\.(md|py|json|db|png|jpg|log|txt)$", re.IGNORECASE),  # a filename
)


def trivial_difference(discriminators: list[str]) -> bool:
    """True when nothing in the difference says anything about the world."""
    if not discriminators:
        return True
    for item in discriminators:
        text = str(item).strip()
        if not text:
            continue
        if not any(pattern.search(text) for pattern in TRIVIAL):
            return False
    return True


@dataclass
class Cluster:
    members: list[str] = field(default_factory=list)
    pairs: list[tuple[str, str]] = field(default_factory=list)
    topic: str = ""
    resolved: bool = False
    detail: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.members)

    def strongest(self) -> dict[str, Any] | None:
        if not self.detail:
            return None
        return max(
            self.detail.values(),
            key=lambda row: (float(row.get("confidence") or 0.0),
                             float(row.get("importance") or 0.0),
                             str(row.get("created_at") or "")),
        )

    def recommendation(self) -> str:
        best = self.strongest()
        if best is None:
            return "no recommendation: none of the members is readable"
        confidence = float(best.get("confidence") or 0.0)
        return (f"keep {str(best['id'])[:8]} (confidence {confidence:.2f}) — "
                f"{len(self.members) - 1} other statement(s) about {self.topic or 'this'}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "members": self.members,
            "pairs": [list(pair) for pair in self.pairs],
            "size": self.size,
            "resolved": self.resolved,
            "recommendation": self.recommendation(),
            "members_detail": self.detail,
        }

    def describe(self) -> str:
        state = "resolved" if self.resolved else "needs a decision"
        return (f"{self.size} conflicting statement(s) about {self.topic or 'an unnamed subject'} "
                f"({len(self.pairs)} pair(s), {state}): {self.recommendation()}")


def _topic(rows: dict[str, dict[str, Any]], members: list[str]) -> str:
    subjects = {str(rows[mid].get("subject") or "") for mid in members if mid in rows}
    predicates = {str(rows[mid].get("predicate") or "") for mid in members if mid in rows}
    subjects.discard("")
    predicates.discard("")
    if len(subjects) == 1:
        subject = next(iter(subjects))
        if len(predicates) == 1:
            return f"{subject}·{next(iter(predicates))}"
        return subject
    shared: set[str] | None = None
    for mid in members:
        words = _tokens(str((rows.get(mid) or {}).get("content") or ""))
        shared = words if shared is None else shared & words
        if not shared:
            break
    if shared:
        return " ".join(sorted(shared)[:3])
    return ""


def clusters(store: Any, *, include_resolved: bool = False) -> list[Cluster]:
    """One item per connected component of the dispute graph."""
    recorded = pairs(store)
    if not include_resolved:
        recorded = [row for row in recorded if not row.get("winner_id")]
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    edges: dict[frozenset, dict[str, Any]] = {}
    for row in recorded:
        left, right = str(row["left_id"]), str(row["right_id"])
        union(left, right)
        edges[frozenset((left, right))] = row

    grouped: dict[str, Cluster] = {}
    for row in recorded:
        left, right = str(row["left_id"]), str(row["right_id"])
        root = find(left)
        cluster = grouped.setdefault(root, Cluster())
        for member in (left, right):
            if member not in cluster.members:
                cluster.members.append(member)
        if (left, right) not in cluster.pairs and (right, left) not in cluster.pairs:
            cluster.pairs.append((left, right))
        if row.get("winner_id"):
            cluster.resolved = True
    for cluster in grouped.values():
        cluster.members.sort()
        cluster.detail = _memories(store, cluster.members)
        cluster.topic = _topic(cluster.detail, cluster.members)
    return sorted(grouped.values(), key=lambda item: (-item.size, item.topic))


def pending(store: Any) -> list[dict[str, Any]]:
    return [cluster.as_dict() for cluster in clusters(store) if not cluster.resolved]


def candidates(store: Any, *, scan: int = 400, limit: int = 40) -> list[dict[str, Any]]:
    """Conflicts the store can see but has not recorded.

    Two existing detectors already answer "are these two claims incompatible": the triple check in
    `contradiction.py`, and the near-miss band in `near_miss.py` where two similar statements differ
    by a number, a negation or a qualifier. Neither records anything, which is why the live `disputes`
    table is empty and this module has nothing to cluster. Bridging them is what makes a cluster
    possible; recording is opt-in, because a store that starts filing disputes on its own is a store
    that can file forty of them from one bad scan.
    """
    found: list[dict[str, Any]] = []
    try:
        from . import contradiction

        for left, right, reason in contradiction.all_conflicts(store)[:limit]:
            found.append({"left_id": left, "right_id": right, "reason": reason,
                          "source": "triple_check"})
    except Exception:  # noqa: BLE001
        pass
    try:
        from . import near_miss

        room = max(0, int(limit) - len(found))
        for proposal in near_miss.detect(store, scan=int(scan), limit=room) if room else []:
            if trivial_difference(list(proposal.discriminator)):
                continue
            found.append({
                "left_id": str(proposal.left_id),
                "right_id": str(proposal.right_id),
                "reason": "near-miss: " + ", ".join(proposal.discriminator[:3]),
                "source": "near_miss",
            })
    except Exception:  # noqa: BLE001
        pass
    return found[: max(0, int(limit))]


def record_conflicts(store: Any, *, scan: int = 400, limit: int = 40, apply: bool = False,
                     at: datetime | None = None) -> dict[str, Any]:
    """Record the conflicts the detectors already see, then cluster them.

    `apply=False` reports what would be filed and changes nothing — the default, because this is the
    step that creates review work for a person.
    """
    proposals = candidates(store, scan=scan, limit=limit)
    recorded = 0
    if apply:
        from . import dispute

        for proposal in proposals:
            result = dispute.mark(store, proposal["left_id"], proposal["right_id"],
                                  reason=proposal["reason"], at=at)
            if result is not None:
                recorded += 1
    return {
        "found": len(proposals),
        "recorded": recorded,
        "applied": bool(apply),
        "proposals": proposals,
        "stats": stats(store),
    }


def stats(store: Any) -> dict[str, Any]:
    recorded = pairs(store)
    open_pair_records = [row for row in recorded if not row.get("winner_id")]
    grouped = clusters(store)
    return {
        "pairs_recorded": len(recorded),
        "pairs_open": len(open_pair_records),
        "clusters": len(grouped),
        "largest": max((cluster.size for cluster in grouped), default=0),
        "collapsed": (len(open_pair_records) - len(grouped)) if grouped else 0,
        "note": "each cluster is one question; the pairs inside it are the store's own bookkeeping",
    }


def resolve(store: Any, cluster: Cluster, winner_id: str, *, by: str = "",
            at: datetime | None = None) -> dict[str, Any]:
    """Resolve every pair in a cluster at once, naming who decided."""
    actor = str(by or "").strip()
    if not actor:
        return {"resolved": False, "reason": "resolution must name who decided"}
    if str(winner_id) not in cluster.members:
        return {"resolved": False, "reason": "the winner must be one of the cluster's members"}
    moment = at or _now()
    resolved_pairs = 0
    try:
        from . import dispute

        dispute._ensure(store)  # a cluster cannot be resolved before any dispute exists
    except Exception:  # noqa: BLE001
        pass
    try:
        with store.transaction() as conn:
            for left, right in cluster.pairs:
                cursor = conn.execute(
                    """UPDATE disputes SET winner_id = ?, resolved_at = ?
                       WHERE ((left_id = ? AND right_id = ?) OR (left_id = ? AND right_id = ?))
                         AND winner_id IS NULL""",
                    (str(winner_id), moment.isoformat(), left, right, right, left),
                )
                resolved_pairs += int(cursor.rowcount or 0)
    except Exception as error:  # noqa: BLE001
        return {"resolved": False, "reason": f"could not record: {str(error)[:80]}"}
    return {
        "resolved": True,
        "winner_id": str(winner_id),
        "topic": cluster.topic,
        "pairs": resolved_pairs,
        "by": actor,
        "sentence": (f"resolved {resolved_pairs} pair(s) about {cluster.topic or 'an unnamed subject'} "
                     f"in favour of {str(winner_id)[:8]} ({actor})"),
    }
