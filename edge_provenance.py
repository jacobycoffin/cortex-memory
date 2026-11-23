"""An edge the retriever made is not something the user said (#206).

Every edge in the store looks the same in the table, and they are not the same kind of thing:

| origin | how it came to exist | may be quoted as a fact |
|---|---|---|
| `human` | a person said these are related | yes — but as *their* claim |
| `document` | a note links to another note, or an import preserved a link | yes, with the note as the source |
| `machine` | co-retrieval, shared answers, an automatic review | **no** — it is a guess about relevance |

The distinction matters because machine edges are produced *by recall itself*: two memories that got
retrieved together are linked because the retriever retrieved them, which is a loop, not evidence. Used
as a hint to find more context that is fine. Quoted back as "these are related" it invents a
relationship nobody stated.

So an edge carries its origin, and anything that renders an edge for a person says which it is.
`citable()` is the narrow gate: only `human` and `document` origins pass. There is no third path, and
`mixed` (a machine edge that later earned a human witness) is judged by its **strongest** witness —
being told once outranks a thousand co-occurrences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

HUMAN = "human"
DOCUMENT = "document"
MACHINE = "machine"
MIXED = "mixed"

ORIGIN_OF: dict[str, str] = {
    "user_feedback": HUMAN,
    "explicit_wikilink": DOCUMENT,
    "migrated_explicit_wikilink": DOCUMENT,
    "migrated_structured_claim_conflict": DOCUMENT,
    "co_retrieval": MACHINE,
    "co_used": MACHINE,
    "co_observed": MACHINE,
    "shared_answer_attribution": MACHINE,
    "auto_judge": MACHINE,
    "similarity": MACHINE,
    "legacy_unattributed": MACHINE,
    "migrated_shared_task_use": MACHINE,
}

CITABLE = frozenset({HUMAN, DOCUMENT})
# Strongest first: a single human word outranks any amount of machine agreement.
RANK = {HUMAN: 3, DOCUMENT: 2, MIXED: 1, MACHINE: 0}

CAVEAT = {
    MACHINE: "inferred from recall history, not stated by anyone",
    MIXED: "inferred, with one human confirmation",
    DOCUMENT: "supported by a note",
    HUMAN: "stated by a person",
}


def origin_of(evidence_type: str) -> str:
    return ORIGIN_OF.get(str(evidence_type), MACHINE)


def evidence_types(store: Any, src_id: str, dst_id: str, relation: str | None = None) -> list[str]:
    try:
        if relation:
            rows = store._conn.execute(
                """SELECT DISTINCT evidence_type FROM edge_evidence
                   WHERE relation = ?
                     AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))""",
                (str(relation), str(src_id), str(dst_id), str(dst_id), str(src_id)),
            ).fetchall()
        else:
            rows = store._conn.execute(
                """SELECT DISTINCT evidence_type FROM edge_evidence
                   WHERE (src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?)""",
                (str(src_id), str(dst_id), str(dst_id), str(src_id)),
            ).fetchall()
    except Exception:  # noqa: BLE001 - no evidence table, no provenance
        return []
    return sorted(str(row["evidence_type"]) for row in rows)


@dataclass
class Provenance:
    origin: str
    witnesses: list[str]
    citable: bool
    mixed: bool = False

    def describe(self) -> str:
        state = "citable" if self.citable else "not citable"
        suffix = " (mixed witnesses)" if self.mixed else ""
        return f"{self.origin} ({state}){suffix}: {CAVEAT.get(self.origin, '')}"

    def as_dict(self) -> dict[str, Any]:
        return {"origin": self.origin, "witnesses": list(self.witnesses),
                "citable": self.citable, "mixed": self.mixed,
                "caveat": CAVEAT.get(self.origin, "")}


def provenance(store: Any, src_id: str, dst_id: str, relation: str | None = None) -> Provenance:
    types = evidence_types(store, src_id, dst_id, relation)
    if not types:
        return Provenance(MACHINE, [], False)
    origins = {origin_of(evidence_type) for evidence_type in types}
    # The strongest witness decides: being told once outranks a thousand co-occurrences. Anything
    # still machine-only stays machine, however many rows it has.
    strongest = max(origins, key=lambda name: RANK.get(name, 0))
    return Provenance(
        origin=strongest,
        witnesses=types,
        citable=strongest in CITABLE,
        mixed=len(origins) > 1,
    )


def citable(store: Any, src_id: str, dst_id: str, relation: str | None = None) -> bool:
    return provenance(store, src_id, dst_id, relation).citable


def caveat_for(store: Any, src_id: str, dst_id: str, relation: str | None = None) -> str:
    """A clause to append when rendering the edge to a person."""
    return CAVEAT.get(provenance(store, src_id, dst_id, relation).origin, "")


def report(store: Any) -> dict[str, Any]:
    try:
        rows = store._conn.execute(
            """SELECT evidence_type, COUNT(*) AS n FROM edge_evidence GROUP BY evidence_type"""
        ).fetchall()
    except Exception:  # noqa: BLE001
        rows = []
    counts: dict[str, int] = {}
    for row in rows:
        origin = origin_of(str(row["evidence_type"]))
        counts[origin] = counts.get(origin, 0) + int(row["n"])
    return {
        "evidence_rows": sum(counts.values()),
        "by_origin": counts,
        "citable_origins": sorted(CITABLE),
        "machine_types": sorted(name for name, origin in ORIGIN_OF.items() if origin == MACHINE),
    }
