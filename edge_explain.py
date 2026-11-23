"""Every relation renders as a sentence a person can check (#85).

An edge table is a graph. A person asking "why are these two connected?" needs a sentence:

    Route 73 job → Vertex account because retrieved together 4× in different sessions

The difference is not decoration. `relation=co_used weight=0.42` cannot be argued with; "retrieved
together four times in different sessions" can be — and the reader immediately knows what would
change their mind, which is the point of an explanation.

The sentence is built from what the store actually recorded:

- **who says so** — evidence types, each with its own phrase, and a count;
- **how strong** it is — the weight, in words rather than decimals;
- **whether it is allowed to count at all** — a quarantined edge (#84), a vetoed pair (#79) and a
  near-miss (#80) all say so in the same sentence, because "why are these connected" and "why did
  this not surface" are the same question from different sides.

Nothing here quotes private values: it names evidence *types* and counts, and shortens content to a
readable prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import edge_provenance  # (#206)

CONTENT_CAP = 60

# What each recorded evidence type actually means, in the reader's terms.
PHRASES: dict[str, str] = {
    "co_retrieval": "retrieved together",
    "co_used": "used in the same session",
    "shared_answer_attribution": "attributed to the same answer",
    "explicit_wikilink": "one note links to the other",
    "migrated_explicit_wikilink": "an imported note links to the other",
    "migrated_shared_task_use": "imported records used them for the same task",
    "migrated_structured_claim_conflict": "imported records flag a claim conflict",
    "auto_judge": "an automatic review judged them related",
    "user_feedback": "a person said these are related",
    "legacy_unattributed": "an older similarity link with no evidence preserved",
    "similarity": "the text is similar",
}

RELATION_PHRASES: dict[str, str] = {
    "related": "related to",
    "co_used": "used alongside",
    "co_observed": "observed alongside",
    "supports": "supports",
    "extends": "extends",
    "example_of": "an example of",
    "contradicts": "contradicts",
    "supersedes": "replaces",
    "vault_link": "links to",
    "near_miss": "similar to but not the same as",
}

STRENGTH: tuple[tuple[float, str], ...] = (
    (0.80, "strongly"),
    (0.50, "moderately"),
    (0.25, "weakly"),
    (0.0, "very weakly"),
)


def _short(text: str, cap: int = CONTENT_CAP) -> str:
    body = " ".join(str(text or "").split())
    if len(body) <= cap:
        return body
    return body[: cap - 1].rstrip() + "…"


def strength_word(weight: float) -> str:
    for floor, word in STRENGTH:
        if float(weight) >= floor:
            return word
    return "very weakly"


def _counts(store: Any, src_id: str, dst_id: str, relation: str | None) -> dict[str, int]:
    try:
        rows = store._conn.execute(
            """SELECT evidence_type, COUNT(*) AS n FROM edge_evidence
               WHERE (src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?)
               GROUP BY evidence_type""",
            (src_id, dst_id, dst_id, src_id),
        ).fetchall()
    except Exception:  # noqa: BLE001 - no evidence table, no witnesses
        return {}
    return {str(row["evidence_type"]): int(row["n"]) for row in rows}


def _content(store: Any, memory_id: str) -> str:
    try:
        row = store._conn.execute(
            "SELECT content FROM memories WHERE id = ?", (str(memory_id),)
        ).fetchone()
    except Exception:  # noqa: BLE001
        return ""
    return str(row["content"]) if row else ""


def _edge(store: Any, left: str, right: str, relation: str | None) -> dict[str, Any] | None:
    try:
        if relation:
            row = store._conn.execute(
                """SELECT * FROM edges WHERE relation = ?
                   AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))""",
                (relation, left, right, right, left),
            ).fetchone()
        else:
            row = store._conn.execute(
                """SELECT * FROM edges
                   WHERE (src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?)
                   ORDER BY weight DESC LIMIT 1""",
                (left, right, right, left),
            ).fetchone()
    except Exception:  # noqa: BLE001
        return None
    return dict(row) if row else None


@dataclass
class Explanation:
    left_id: str
    right_id: str
    sentence: str
    relation: str = ""
    weight: float = 0.0
    witnesses: dict[str, int] = field(default_factory=dict)
    status: str = "counting"

    def as_dict(self) -> dict[str, Any]:
        return {
            "left": self.left_id,
            "right": self.right_id,
            "relation": self.relation,
            "weight": round(self.weight, 4),
            "witnesses": dict(self.witnesses),
            "status": self.status,
            "sentence": self.sentence,
        }


def explain(store: Any, left: str, right: str, relation: str | None = None) -> Explanation:
    """One sentence: what these two are, why, how strongly, and whether it counts."""
    from . import assoc_feedback, edge_quarantine, near_miss, path_query

    left_id = path_query.resolve(store, left) or str(left)
    right_id = path_query.resolve(store, right) or str(right)
    left_text = _short(_content(store, left_id)) or left_id[:8]
    right_text = _short(_content(store, right_id)) or right_id[:8]
    edge = _edge(store, left_id, right_id, relation)

    if edge is None and not relation:
        verdict = ["no link is recorded between them"]
        if assoc_feedback.veto(store, left_id, right_id):
            verdict.append("a person has said they are unrelated, so none will be proposed")
        pair = tuple(sorted((left_id, right_id)))
        if any(
            (str(row["left_id"]), str(row["right_id"])) == pair
            for row in near_miss.pairs(store, limit=10000)
        ):
            verdict.append("they are recorded as a near-miss pair")
        return Explanation(left_id, right_id, " — ".join(verdict) + ".", status="absent")

    if edge is None:
        return Explanation(
            left_id, right_id,
            f"No {RELATION_PHRASES.get(str(relation), relation)} link is recorded between them.",
            relation=str(relation or ""), status="absent",
        )

    edge_relation = str(edge.get("relation") or "")
    weight = float(edge.get("weight") or 0.0)
    counts = _counts(store, left_id, right_id, edge_relation)

    parts: list[str] = []
    for evidence_type, count in sorted(counts.items()):
        phrase = PHRASES.get(evidence_type, evidence_type.replace("_", " "))
        if evidence_type == "co_retrieval" or evidence_type == "co_used":
            parts.append(f"{phrase} {count}×")
        elif count > 1:
            parts.append(f"{phrase} ({count}×)")
        else:
            parts.append(phrase)
    supporting = [part for part in parts
                  if "no evidence preserved" not in part]

    relation_phrase = RELATION_PHRASES.get(edge_relation, edge_relation)
    if supporting:
        because = "; ".join(supporting)
    else:
        because = f"a {relation_phrase} link was recorded with no preserved evidence"
    sentence = f"{left_text} → {right_text} because {because}"

    status = "counting"
    quarantine = edge_quarantine.status(store, left_id, right_id, edge_relation)
    if not quarantine.trusted:
        status = "quarantined"
        sentence += f" — recorded but not scoring yet ({quarantine.reason})"
    if assoc_feedback.veto(store, left_id, right_id):
        status = "vetoed"
        sentence += " — a person has since said they are unrelated, so it does not surface"
    pair = tuple(sorted((left_id, right_id)))
    if any(
        (str(row["left_id"]), str(row["right_id"])) == pair
        for row in near_miss.pairs(store, limit=10000)
    ):
        status = "near-miss"
        sentence += " — recorded as a near-miss, so only the stronger of the two surfaces"

    # Provenance (#206): a link the retriever inferred is never presented as something stated.
    origin = edge_provenance.provenance(store, left_id, right_id, edge_relation)
    sentence += f" — {strength_word(weight)} {relation_phrase}, "
    sentence += f"{edge_provenance.CAVEAT.get(origin.origin, 'inferred')}."
    return Explanation(left_id, right_id, sentence, relation=edge_relation, weight=weight,
                       witnesses=counts, status=status)


def for_memory(store: Any, memory_id: str, *, limit: int = 10) -> list[Explanation]:
    """Every recorded link for one memory, strongest first."""
    try:
        rows = store._conn.execute(
            """SELECT src_id, dst_id, relation FROM edges
               WHERE src_id = ? OR dst_id = ?
               ORDER BY weight DESC, src_id, dst_id LIMIT ?""",
            (str(memory_id), str(memory_id), int(limit)),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    out: list[Explanation] = []
    for row in rows:
        src, dst = str(row["src_id"]), str(row["dst_id"])
        other = dst if src == str(memory_id) else src
        out.append(explain(store, str(memory_id), other, str(row["relation"])))
    return out


def report(store: Any) -> dict[str, Any]:
    try:
        total = int(store._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] or 0)
    except Exception:  # noqa: BLE001
        total = 0
    return {"edges": total, "phrases": len(PHRASES), "relations_named": len(RELATION_PHRASES)}
