"""A dense region of the graph deserves a name, not another thousand edges (#74).

The live graph has 270 connected components: one giant of 5,362 memories and 238 islands under five.
That shape is the problem. Every walk over the giant starts from a single memory and has no landmark
to aim at, so "what do we know about the homelab" is answered by whatever happened to be closest to
the seed rather than by the region itself.

A topic node is a **label over rows that already exist**, not a new kind of fact:

- it is a name for a dense region (a connected component worth naming), so the graph has landmarks;
- it never deletes, rewrites or shortens a memory or an edge — the members are the members;
- it never invents a member. A memory id that is not in the store is dropped, not stored;
- its name comes from the terms that repeat *inside* the region and are *rarer outside* it, which is
  the only evidence the store actually has for "what this region is about".

The name is deterministic by construction — frequency inside, frequency outside, alphabetical
tie-break — because a label that changes between two identical calls is a label nobody can cite.

`degree_caps` refuses edges past the cap and parks them in `edge_overflow` grouped by hub and
relation; those groups are what a topic node would eventually adopt. This module does not adopt them
yet. It only gives the region a name and a membership row, so the adoption has something to hang on.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

TOPIC_TABLE = "topic_nodes"
MEMBER_TABLE = "topic_members"

# How many memories one region may cover before the label stops meaning anything useful. Not a
# hard limit — a warning sign, reported rather than enforced.
GIANT_WARNING = 1000

_WORD = re.compile(r"[a-z][a-z0-9_-]{2,}", re.IGNORECASE)
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b", re.IGNORECASE)
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
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TOPIC_TABLE} (
                    topic_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
        )
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {MEMBER_TABLE} (
                    topic_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    PRIMARY KEY (topic_id, memory_id)
                )"""
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {MEMBER_TABLE}_memory_idx ON {MEMBER_TABLE} (memory_id)"
        )


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001 - a store that cannot be read is an empty store to a caller
        return []


# Vault scaffolding, not content. Every note the vault importer produced carries the same header
# ("Vault note: <title> Path: <file> Section: <heading>"), so those words appear in *every* memory and
# score enormous lift exactly because they are meaningless -- the first live run labelled the giant
# component "vault path note section". Stripping the shape before counting is what lets a real subject
# outrank the template.
_SCAFFOLD = re.compile(
    r"(?im)^\s*vault note\s*:.*$"
    r"|\bpath\s*:\s*[^\n|]*?\.md\b"
    # Bounded: a header value is short. An unbounded `.*` would eat the rest of a single-line note,
    # which is the one way this pattern could remove a subject instead of a label.
    r"|\bsection\s*:\s*[^\n]{0,80}"
    r"|\btags?\s*:\s*[^\n]{0,80}"
    r"|^\s*#{1,6}\s*"
)


def _strip_scaffolding(text: str) -> str:
    """Remove the note-template shape, keeping whatever the note was actually about."""
    stripped = _SCAFFOLD.sub(" ", str(text or ""))
    # Dates and clock times are metadata about when a note was written, never its subject.
    stripped = _DATE.sub(" ", stripped)
    return _TIME.sub(" ", stripped)


def _terms(text: str, *, strip: bool = True) -> set[str]:
    """The words worth scoring: long enough to mean something, minus the ones every sentence has."""
    body = _strip_scaffolding(text) if strip else str(text or "")
    return {
        token.casefold().strip("_")
        for token in _WORD.findall(body)
        if len(token) >= 3 and token.casefold() not in _STOP
    }


def _contents(store: Any, ids: Iterable[str], *, chunk: int = 400) -> dict[str, str]:
    """Content for a set of memory ids, in chunks so sqlite's variable limit is never the issue."""
    wanted = [str(item) for item in ids]
    found: dict[str, str] = {}
    for start in range(0, len(wanted), chunk):
        part = wanted[start:start + chunk]
        marks = ",".join("?" * len(part))
        for row in _rows(store, f"SELECT id, content FROM memories WHERE id IN ({marks})", tuple(part)):
            found[str(row["id"])] = str(row["content"] or "")
    return found


def _existing(store: Any, ids: Iterable[str], *, chunk: int = 400) -> set[str]:
    wanted = [str(item) for item in ids]
    known: set[str] = set()
    for start in range(0, len(wanted), chunk):
        part = wanted[start:start + chunk]
        marks = ",".join("?" * len(part))
        for row in _rows(store, f"SELECT id FROM memories WHERE id IN ({marks})", tuple(part)):
            known.add(str(row["id"]))
    return known


def _global_terms(store: Any, *, limit: int = 20000) -> tuple[dict[str, int], int]:
    """Document frequency for every term in the store: the denominator of distinctiveness."""
    counts: dict[str, int] = {}
    total = 0
    for row in _rows(store, "SELECT content FROM memories LIMIT ?", (int(limit),)):
        total += 1
        for term in _terms(row["content"]):
            counts[term] = counts.get(term, 0) + 1
    return counts, total


def _region_terms(store: Any, members: Iterable[str], *, strip: bool = True) -> dict[str, int]:
    """How many memories inside the region mention each term."""
    counts: dict[str, int] = {}
    contents = _contents(store, members)
    for text in contents.values():
        for term in _terms(text, strip=strip):
            counts[term] = counts.get(term, 0) + 1
    return counts


def _label_from(
    terms: dict[str, int],
    size: int,
    global_df: dict[str, int],
    total: int,
    *,
    words: int = 6,
) -> str:
    """Turn term counts into a title. Deterministic: no clock, no randomness, no insertion order."""
    if not terms or size <= 0:
        return ""
    repeat = max(2, round(size * 0.2))
    candidates = {term: n for term, n in terms.items() if n >= repeat}
    if not candidates:
        # A region with no term shared by two memories has no topic; the best it has is its words.
        candidates = dict(terms)
    scored: list[tuple[float, int, str]] = []
    for term, count in candidates.items():
        inside = (count + 0.5) / size
        outside = (global_df.get(term, 0) + 0.5) / max(total, 1)
        # Rounded before sorting so two runs on the same data cannot order themselves differently.
        scored.append((round(inside / outside, 6), count, term))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    distinctive = [term for lift, _count, term in scored if lift > 1.5][:words]
    if distinctive:
        return " ".join(distinctive)
    return " ".join(term for _lift, _count, term in scored[: min(2, words)])


def label(store: Any, members: Iterable[str]) -> str:
    """A short title for a region: the terms that repeat here more than they repeat everywhere."""
    ids = sorted({str(item) for item in members if str(item).strip()})
    if not ids:
        return ""
    try:
        terms = _region_terms(store, ids)
        global_df, total = _global_terms(store)
    except Exception:  # noqa: BLE001 - an unnamed region is better than a broken caller
        return ""
    name = _label_from(terms, len(ids), global_df, total)
    if name:
        return name
    # Stripping the scaffolding can leave a region with nothing to say -- a vault chunk whose entire
    # text was a header. An empty title is worse than a templatey one, so fall back to the raw words
    # rather than claiming the region has no subject.
    try:
        raw = _region_terms(store, ids, strip=False)
    except Exception:  # noqa: BLE001
        raw = {}
    return _label_from(raw, len(ids), global_df, total)


@dataclass(frozen=True)
class Region:
    """A component worth naming: who is in it, what repeats inside it, what it would be called."""

    size: int
    members: tuple[str, ...]
    terms: tuple[str, ...]
    name: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "members": list(self.members),
            "terms": list(self.terms),
            "name": self.name,
        }

    def describe(self) -> str:
        return f"{self.name or '(unnamed)'} — {self.size} memories"


@dataclass
class Topic:
    """A persisted label. The members are the region; nothing here changes a memory."""

    topic_id: str
    name: str
    size: int
    members: tuple[str, ...] = ()
    created_at: str = ""
    updated_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "name": self.name,
            "size": self.size,
            "members": list(self.members),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _components(edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    """Union-find over an undirected edge list; components sorted by size, then by their members."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # path compression, so a long chain stays cheap
            parent[node], node = root, parent[node]
        return root

    for left, right in edges:
        left_root, right_root = find(str(left)), find(str(right))
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    grouped: dict[str, list[str]] = {}
    for node in parent:
        grouped.setdefault(find(node), []).append(node)
    found = [sorted(members) for members in grouped.values()]
    found.sort(key=lambda members: (-len(members), members))
    return found


def components(store: Any, *, scan: int = 20000) -> list[list[str]]:
    """Connected components of `edges`, largest first.

    Undirected on purpose: `add_edge` stores pairs in canonical order, so an edge may be written
    either way round, and a member that is only ever a `dst_id` is still in the component.

    `scan` bounds how many edge rows are read, because the whole table is 100k+ rows on the live
    store and a report must not become a full-table walk. The bound is a prefix of the table, so a
    component that only appears in the unscanned tail is simply absent — not partly invented.
    """
    if int(scan) <= 0:
        return []
    try:
        rows = _rows(
            store,
            "SELECT src_id, dst_id FROM edges ORDER BY rowid LIMIT ?",
            (int(scan),),
        )
    except Exception:  # noqa: BLE001
        return []
    return _components((row["src_id"], row["dst_id"]) for row in rows)


def dense_regions(store: Any, min_size: int = 8, scan: int = 20000) -> list[Region]:
    """Components worth naming, with the terms that repeat inside each one."""
    try:
        groups = components(store, scan=scan)
        if not groups:
            return []
        global_df, total = _global_terms(store)
        regions: list[Region] = []
        for members in groups:
            if len(members) < int(min_size):
                continue
            terms = _region_terms(store, members)
            repeat = max(2, round(len(members) * 0.2))
            repeating = {term: n for term, n in terms.items() if n >= repeat}
            ranked = sorted(repeating.items(), key=lambda item: (-item[1], item[0]))
            regions.append(
                Region(
                    size=len(members),
                    members=tuple(members),
                    terms=tuple(term for term, _count in ranked[:8]),
                    name=_label_from(terms, len(members), global_df, total),
                )
            )
        return regions
    except Exception:  # noqa: BLE001
        return []


def _topic_id(members: Iterable[str]) -> str:
    """Same member set, same id — this is what makes `create` idempotent without a lookup table."""
    joined = "|".join(sorted({str(item) for item in members}))
    return "topic:" + hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]


def create(store: Any, members: Iterable[str], *, name: str = "",
           at: datetime | None = None) -> Topic | None:
    """Persist a topic node over an existing member set. Never invents a member, never renames."""
    ids = sorted({str(item) for item in members if str(item).strip()})
    if not ids:
        return None
    try:
        known = _existing(store, ids)
    except Exception:  # noqa: BLE001
        return None
    ids = [item for item in ids if item in known]
    if not ids:
        return None
    title = str(name or "").strip()
    if not title:
        title = label(store, ids) or "untitled region"
    topic_id = _topic_id(ids)
    stamp = (at or _now()).isoformat()
    try:
        _ensure(store)
        with store.transaction() as conn:
            conn.execute(
                f"""INSERT OR IGNORE INTO {TOPIC_TABLE}
                    (topic_id, name, size, created_at, updated_at) VALUES (?, ?, ?, ?, ?)""",
                (topic_id, title, len(ids), stamp, stamp),
            )
            # Re-asserting a member set re-counts it and bumps the clock; an explicitly supplied
            # name wins, a computed one never overwrites a name that is already there.
            conn.execute(
                f"""UPDATE {TOPIC_TABLE}
                    SET size = ?, updated_at = ?,
                        name = CASE WHEN ? <> '' THEN ? ELSE name END
                    WHERE topic_id = ?""",
                (len(ids), stamp, str(name or "").strip(), str(name or "").strip(), topic_id),
            )
            conn.executemany(
                f"INSERT OR IGNORE INTO {MEMBER_TABLE} (topic_id, memory_id) VALUES (?, ?)",
                [(topic_id, memory_id) for memory_id in ids],
            )
    except Exception:  # noqa: BLE001
        return None
    return get(store, topic_id)


def get(store: Any, topic_id: str) -> Topic | None:
    rows = _rows(store, f"SELECT * FROM {TOPIC_TABLE} WHERE topic_id = ?", (str(topic_id),))
    if not rows:
        return None
    return _topic(store, rows[0])


def _topic(store: Any, row: dict[str, Any]) -> Topic:
    topic_id = str(row["topic_id"])
    members = [
        str(member["memory_id"])
        for member in _rows(
            store,
            f"SELECT memory_id FROM {MEMBER_TABLE} WHERE topic_id = ? ORDER BY memory_id",
            (topic_id,),
        )
    ]
    return Topic(
        topic_id=topic_id,
        name=str(row["name"] or ""),
        size=int(row["size"] or 0),
        members=tuple(members),
        created_at=str(row["created_at"] or ""),
        updated_at=str(row["updated_at"] or ""),
    )


def topics(store: Any, limit: int = 50) -> list[Topic]:
    """Every named region, biggest first — the landmarks the graph was missing."""
    rows = _rows(
        store,
        f"SELECT * FROM {TOPIC_TABLE} ORDER BY size DESC, name, topic_id LIMIT ?",
        (max(0, int(limit)),),
    )
    return [_topic(store, row) for row in rows]


def topic_of(store: Any, memory_id: str) -> Topic | None:
    """Which landmark this memory belongs to, if any (the largest, when it belongs to several)."""
    rows = _rows(
        store,
        f"""SELECT t.* FROM {TOPIC_TABLE} t
            JOIN {MEMBER_TABLE} m ON m.topic_id = t.topic_id
            WHERE m.memory_id = ?
            ORDER BY t.size DESC, t.name, t.topic_id LIMIT 1""",
        (str(memory_id),),
    )
    if not rows:
        return None
    return _topic(store, rows[0])


def report(store: Any) -> dict[str, Any]:
    """How much of the graph has a landmark, and how much is still an anonymous island."""
    empty = {"topics": 0, "members": 0, "largest": None, "uncovered": 0, "giant": None}
    try:
        rows = _rows(
            store,
            f"""SELECT t.topic_id, t.name, t.size,
                       (SELECT COUNT(*) FROM {MEMBER_TABLE} m WHERE m.topic_id = t.topic_id) AS n
                FROM {TOPIC_TABLE} t ORDER BY t.size DESC, t.name, t.topic_id""",
        )
        covered = _rows(
            store,
            f"SELECT COUNT(DISTINCT memory_id) AS n FROM {MEMBER_TABLE}",
        )
        connected = _rows(
            store,
            """SELECT COUNT(DISTINCT id) AS n FROM (
                   SELECT src_id AS id FROM edges UNION SELECT dst_id FROM edges
               )""",
        )
        unmatched = _rows(
            store,
            f"""SELECT COUNT(*) AS n FROM (
                    SELECT src_id AS id FROM edges UNION SELECT dst_id FROM edges
                ) e
                WHERE NOT EXISTS (SELECT 1 FROM {MEMBER_TABLE} m WHERE m.memory_id = e.id)""",
        )
    except Exception:  # noqa: BLE001
        return empty
    largest = None
    if rows:
        largest = {
            "topic_id": str(rows[0]["topic_id"]),
            "name": str(rows[0]["name"]),
            "size": int(rows[0]["n"] or 0),
        }
    giant = None
    for row in rows:
        if int(row["n"] or 0) >= GIANT_WARNING:
            giant = {"topic_id": str(row["topic_id"]), "name": str(row["name"]),
                     "size": int(row["n"] or 0)}
            break
    return {
        "topics": len(rows),
        "members": int(covered[0]["n"] or 0) if covered else 0,
        "largest": largest,
        "uncovered": int(unmatched[0]["n"] or 0) if unmatched else 0,
        "connected": int(connected[0]["n"] or 0) if connected else 0,
        "giant": giant,
    }
