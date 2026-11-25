"""Cluster-then-expand: grow a result set along the one direction that hangs together (#160).

A search returns neighbours of the query, not neighbours of each other. Given five memories about
the same homelab box, the sixth thing the store knows about that box is usually one hop from one of
the five - and adding it is the difference between an answer and a list. But "one hop from a seed"
is also how a retrieval set turns into the whole graph: the live store has a 5,362-memory component
(#74), so every seed is within one hop of almost everything, and an expander that follows all of them
is a `SELECT *`.

So expansion is a decision about a *cluster*, not about a memory:

- **expansion is bounded.** One cluster may add at most `cap` memories, and the cap is printed in
  the answer. A budget that lives only in the caller's head is not a budget.
- **only a cluster that is measurably denser than the rest expands.** The bar is computed from the
  other clusters' densities (average plus one robust spread step, never below twice the average),
  never a constant, because the definition of "dense" depends on this query's pool. Matching the
  average is not being denser.
- **the seed set is a floor.** No added id is already in the seed set; the seed is what the query
  asked for and appending a copy of it wastes the budget and lies about the result size.
- **the answer names what it did.** Which cluster, how big, how many of its members were already
  seeds, how many candidates were left on the floor by the cap.
- **no standout means no expansion.** A store that cannot distinguish one region does not get to
  invent one; the seed set comes back unchanged with the numbers that justified the abstention.

A cluster is a connected region of candidates that share distinctive terms (at least two shared, and
at least a quarter of their combined vocabulary). Terms come from `entities_json` and from content
with the vault importer's scaffolding stripped - the same reason #74 needed it: every chunk begins
"Vault note: <title> Path: <file> Section: <heading>", and those words would cluster the entire
store into one region.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable

# The hard cap on how many memories one cluster may add. Stated in every answer, because the caller
# has to be able to budget the injected tokens against the number the module actually used.
CAP_PER_CLUSTER = 6

# Candidate pool bounds. The expander reads one hop from the seeds, which on this store is thousands
# of edges; the pool is capped so a query costs a bounded number of reads regardless of the graph.
MAX_NEIGHBOURS_PER_SEED = 12
MAX_CANDIDATES = 240

# Two memories belong to the same cluster when they share at least two terms and those terms are at
# least a quarter of their combined vocabulary. Two shared words out of a handful is the smallest
# evidence of "same subject"; one shared word is how "note" or "port" joins everything to everything.
LINK_SHARED = 2
LINK_JACCARD = 0.25

# States whose memories may be *added*. An archived memory is history, not a neighbour worth spending
# the cap on, and it can still seed a cluster (it was in the result set for a reason).
ADDABLE_STATES = ("active", "cold")

_WORD = re.compile(r"[a-z][a-z0-9_.-]{2,}", re.IGNORECASE)
# Vault scaffolding, not content: the importer's header appears on 79% of active memories (#223, #74).
# Each label is removed on its own, never "the rest of the line": on a one-line note the heading is
# the only description there is, and eating it leaves a memory with no subject at all.
_SCAFFOLD = re.compile(
    r"(?im)\bvault note\s*:"
    r"|\bpath\s*:\s*[^\n|]*?\.md\b"
    r"|\bsection\s*:|^#{1,6}\s*"
)
_STOP = {
    "about", "after", "again", "against", "all", "also", "and", "any", "are", "because", "been",
    "before", "being", "both", "but", "can", "could", "did", "does", "done", "each", "every",
    "for", "from", "had", "has", "have", "here", "how", "into", "its", "just", "like", "made",
    "make", "many", "may", "memory", "more", "most", "much", "must", "need", "needs", "nor",
    "not", "now", "off", "one", "only", "other", "our", "out", "over", "per", "same", "see",
    "set", "should", "some", "still", "such", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "through", "too", "under", "until", "use", "used",
    "using", "very", "via", "want", "was", "way", "were", "what", "when", "where", "which",
    "while", "will", "with", "would", "yet", "you", "your",
    # Scaffolding words: they appear on every imported note, so they would cluster the store into one
    # region named after the template rather than the subject.
    "note", "notes", "vault", "path", "section", "tags", "tag",
}


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    """Read rows, treating an unreadable store as an empty one.

    A retrieval aid that raises on a locked database takes the turn down with it; an unreadable
    store means "nothing to expand", which is a true and usable answer.
    """
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def terms_of(text: str, entities: Iterable[str] = ()) -> set[str]:
    """The words worth clustering on: content minus scaffolding, plus every recorded entity name."""
    body = _SCAFFOLD.sub(" ", str(text or ""))
    found = {
        token.casefold().strip("._-")
        for token in _WORD.findall(body)
        if len(token) >= 3 and token.casefold() not in _STOP
    }
    for name in entities:
        for token in _WORD.findall(str(name or "")):
            lowered = token.casefold()
            if len(lowered) >= 3 and lowered not in _STOP:
                found.add(lowered)
    return found


def _facts(store: Any, ids: Iterable[str], *, chunk: int = 400) -> dict[str, dict[str, Any]]:
    """Content, entities and state for a set of ids, in chunks so sqlite's variable cap is moot."""
    wanted = [str(item) for item in ids]
    found: dict[str, dict[str, Any]] = {}
    for start in range(0, len(wanted), chunk):
        part = wanted[start:start + chunk]
        marks = ",".join("?" * len(part))
        for row in _rows(
            store,
            f"SELECT id, content, entities_json, state FROM memories WHERE id IN ({marks})",
            tuple(part),
        ):
            entities: list[str] = []
            try:
                loaded = json.loads(str(row.get("entities_json") or "[]"))
                entities = [str(item) for item in loaded] if isinstance(loaded, list) else []
            except (TypeError, ValueError):
                entities = []
            found[str(row["id"])] = {
                "content": str(row.get("content") or ""),
                "state": str(row.get("state") or "active"),
                "terms": terms_of(str(row.get("content") or ""), entities),
            }
    return found


def _neighbours(store: Any, memory_id: str, *, limit: int = MAX_NEIGHBOURS_PER_SEED) -> list[str]:
    """One hop from a seed, strongest edges first, id order breaking ties.

    Deterministic ordering is what makes two identical calls return the same set; a cap applied to
    an unordered list is a different expansion on every run.
    """
    rows = _rows(
        store,
        "SELECT src_id, dst_id, weight FROM edges WHERE src_id = ? OR dst_id = ? "
        "ORDER BY weight DESC, src_id, dst_id",
        (str(memory_id), str(memory_id)),
    )
    out: list[str] = []
    for row in rows:
        other = str(row["dst_id"]) if str(row["src_id"]) == str(memory_id) else str(row["src_id"])
        if other not in out:
            out.append(other)
        if len(out) >= int(limit):
            break
    return out


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    shared = len(left & right)
    if not shared:
        return 0.0
    return shared / len(left | right)


@dataclass
class Cluster:
    """A region of the candidate pool: who is in it, what they agree on, how tightly."""

    members: tuple[str, ...]
    terms: tuple[str, ...]
    density: float

    @property
    def size(self) -> int:
        return len(self.members)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": " ".join(self.terms),
            "terms": list(self.terms),
            "members": list(self.members),
            "size": self.size,
            "density": self.density,
        }


def _components(ids: list[str], vocabularies: dict[str, set[str]]) -> list[list[str]]:
    """Union-find over candidates that share enough distinctive terms; largest cluster first."""
    parent = {item: item for item in ids}

    def find(node: str) -> str:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for index, left in enumerate(ids):
        left_terms = vocabularies[left]
        for right in ids[index + 1:]:
            right_terms = vocabularies[right]
            if len(left_terms & right_terms) < LINK_SHARED:
                continue
            if _jaccard(left_terms, right_terms) < LINK_JACCARD:
                continue
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

    grouped: dict[str, list[str]] = {}
    for node in ids:
        grouped.setdefault(find(node), []).append(node)
    found = [sorted(members) for members in grouped.values()]
    found.sort(key=lambda members: (-len(members), members))
    return found


def _density(members: Iterable[str], vocabularies: dict[str, set[str]]) -> float:
    """Mean pairwise vocabulary overlap inside a cluster: how much its members have in common.

    A single memory has nothing to be dense *with*, so its density is 0.0 rather than a perfect 1.0 -
    otherwise every seed that arrived alone would look like the strongest cluster in the pool.
    """
    ids = list(members)
    if len(ids) < 2:
        return 0.0
    pairs = 0
    total = 0.0
    for index, left in enumerate(ids):
        for right in ids[index + 1:]:
            total += _jaccard(vocabularies[left], vocabularies[right])
            pairs += 1
    return round(total / pairs, 6) if pairs else 0.0


def _label(members: Iterable[str], vocabularies: dict[str, set[str]], *, words: int = 3) -> tuple[str, ...]:
    """The terms this region repeats: the shortest honest name for a cluster."""
    ids = list(members)
    counts: dict[str, int] = {}
    for memory_id in ids:
        for term in vocabularies[memory_id]:
            counts[term] = counts.get(term, 0) + 1
    if not counts:
        return ()
    # A term present in one memory of ten is that memory's word, not the region's topic.
    repeat = 2 if len(ids) < 4 else max(2, round(len(ids) * 0.4))
    ranked = sorted(
        ((count, term) for term, count in counts.items() if count >= repeat),
        key=lambda item: (-item[0], item[1]),
    )
    if not ranked:  # a cluster of size two held together by two terms still has a name
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ranked = [(count, term) for term, count in ranked]
    return tuple(term for _count, term in ranked[: int(words)])


def _threshold(others: list[float]) -> tuple[float, float, float]:
    """The bar the best cluster must clear, read from the clusters it is competing with.

    `average + one robust spread step`, and never below twice the average. Both parts are computed
    from this query's pool: the same density means "dense" in a homogeneous pool and "unremarkable"
    in a pool of tight little clusters, and a hard-coded bar would be wrong in one of them. The
    "twice the average" floor exists because a distribution with no spread (every other cluster
    exactly as tight as the mean) would otherwise let a cluster expand for merely matching it.
    """
    if not others:
        return 0.0, 0.0, 0.0
    baseline = round(statistics.fmean(others), 6)
    spread = round(statistics.median([abs(value - baseline) for value in others]), 6)
    return round(baseline + max(spread, baseline), 6), baseline, spread


@dataclass
class Expansion:
    """What the expander did, and the numbers that made it decide."""

    seed: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    expanded: bool = False
    cluster: dict[str, Any] | None = None
    clusters: list[dict[str, Any]] = field(default_factory=list)
    cap: int = CAP_PER_CLUSTER
    threshold: float | None = None
    baseline: float | None = None
    spread: float | None = None
    dropped: int = 0
    candidates: int = 0
    reason: str = ""

    @property
    def ids(self) -> list[str]:
        """The seed set with the expansion appended, seed order preserved."""
        return list(self.seed) + list(self.added)

    def describe(self) -> str:
        if not self.seed:
            return "nothing to expand: the seed set was empty"
        if not self.expanded:
            return f"seed kept as-is ({len(self.seed)}): {self.reason}"
        cluster = self.cluster or {}
        name = cluster.get("name") or "unnamed"
        already = int(cluster.get("already_seeded") or 0)
        floor = (f", {self.dropped} left by the cap" if self.dropped else "")
        return (
            f"expanded `{name}` ({cluster.get('size', 0)} memories, {already} already in the seed) "
            f"with {len(self.added)} of at most {self.cap}{floor}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": list(self.seed),
            "ids": self.ids,
            "added": list(self.added),
            "expanded": self.expanded,
            "cluster": self.cluster,
            "clusters": list(self.clusters),
            "cap": self.cap,
            "threshold": self.threshold,
            "baseline": self.baseline,
            "spread": self.spread,
            "dropped": self.dropped,
            "candidates": self.candidates,
            "reason": self.reason,
            "sentence": self.describe(),
        }


def expand(
    store: Any,
    seed_ids: Iterable[str],
    *,
    cap: int = CAP_PER_CLUSTER,
    neighbours_per_seed: int = MAX_NEIGHBOURS_PER_SEED,
    max_candidates: int = MAX_CANDIDATES,
) -> Expansion:
    """Grow a seed result set along its densest cluster, or decline and say why."""
    seed: list[str] = []
    for item in seed_ids or []:
        text = str(item or "").strip()
        if text and text not in seed:  # a seed set with a repeat is still one memory
            seed.append(text)
    budget = max(0, int(cap))
    result = Expansion(seed=seed, cap=budget)
    if not seed:
        result.reason = "the seed set was empty"
        return result

    pool: list[str] = list(seed)
    for memory_id in seed:
        if len(pool) >= int(max_candidates):
            break
        for other in _neighbours(store, memory_id, limit=neighbours_per_seed):
            if other not in pool and len(pool) < int(max_candidates):
                pool.append(other)
    result.candidates = len(pool)

    facts = _facts(store, pool)
    # A candidate the store cannot describe is not evidence of anything. Dropping it keeps the pool
    # equal to what was actually readable, which is what the reported densities are computed over.
    described = [item for item in pool if item in facts]
    vocabularies = {item: facts[item]["terms"] for item in described}
    if len(described) < 2:
        result.reason = "fewer than two readable candidates, so there is nothing to cluster"
        return result

    groups = _components(described, vocabularies)
    clusters = [
        Cluster(
            members=tuple(members),
            terms=_label(members, vocabularies),
            density=_density(members, vocabularies),
        )
        for members in groups
    ]
    clusters.sort(key=lambda item: (-item.density, -item.size, item.members))
    result.clusters = [cluster.as_dict() for cluster in clusters]

    if len(clusters) < 2:
        result.reason = (
            "all candidates fall in one cluster, so there is nothing for it to be denser than"
        )
        return result

    best, runner_up = clusters[0], clusters[1]
    threshold, baseline, spread = _threshold([cluster.density for cluster in clusters[1:]])
    result.threshold, result.baseline, result.spread = threshold, baseline, spread

    if best.density <= 0.0:
        result.reason = "no cluster shares enough terms to be a cluster"
        return result
    if abs(best.density - runner_up.density) < 1e-9:
        # Two equally tight regions means the store does not know which one the query was about;
        # picking the larger one is how an expander turns a coin flip into a confident answer.
        result.reason = (
            f"two clusters are equally dense ({best.density:.3f}), so neither is the better expansion"
        )
        return result
    if best.density <= threshold:
        result.reason = (
            f"no cluster is meaningfully denser than the rest "
            f"(best {best.density:.3f}, threshold {threshold:.3f} from {len(clusters) - 1} others)"
        )
        return result

    members = [item for item in best.members if item in facts]
    inside_seed = [item for item in members if item in set(seed)]
    outside = [
        item for item in members
        if item not in set(seed) and facts[item]["state"] in ADDABLE_STATES
    ]
    result.cluster = {
        **best.as_dict(),
        "already_seeded": len(inside_seed),
        "addable": len(outside),
    }
    if budget <= 0:
        result.reason = f"the cap is {budget}, so no memory may be added"
        return result
    if not outside:
        # The densest region is either all seed or all archived. Saying "expanded" here would be a
        # claim about new evidence that does not exist; the cluster is still named so the caller can
        # see what lost.
        result.reason = "the densest cluster has no addable memory outside the seed"
        return result

    def affinity(memory_id: str) -> tuple[float, str]:
        overlap = [_jaccard(vocabularies[memory_id], vocabularies[item]) for item in inside_seed]
        score = round(statistics.fmean(overlap), 6) if overlap else 0.0
        return (-score, memory_id)  # strongest link to what was asked for first, then id order

    outside.sort(key=affinity)
    chosen = outside[:budget]
    result.added = chosen
    result.dropped = max(0, len(outside) - len(chosen))
    result.expanded = True
    result.reason = (
        f"cluster `{' '.join(best.terms) or 'unnamed'}` is densest at {best.density:.3f} "
        f"against a threshold of {threshold:.3f}"
    )
    return result


def render(expansion: Expansion, *, limit: int = 10) -> list[str]:
    """Terminal lines: what was expanded, what it added, and the numbers behind the decision."""
    lines = [expansion.describe()]
    for memory_id in expansion.added[: int(limit)]:
        lines.append(f"  + {memory_id[:8]}")
    if len(expansion.added) > limit:
        lines.append(f"  … {len(expansion.added) - limit} more")
    return lines
