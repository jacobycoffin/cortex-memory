"""One page per entity: its memories, links, changes, disputes and retirements (#166).

A memory page (`explain_memory`) answers "what is this one belief"; a recall page
(`recall_explain`) answers "why did this come back". Neither answers the question a person
actually asks about a *thing*: what do you know about the Proxmox box, how sure are you, what
changed, and where do you disagree with yourself.

So this is the **entity-centric** view — one page, assembled from the modules that already own
each answer:

- `entity_timeline` — which memories mention it, in the order the store learned them, and which
  supersession pairs changed a belief;
- `edge_explain` — why two of those memories are linked, as a sentence a person can argue with;
- `entity_nodes` — the canonical name, the guessed kind, and the aliases it answers to.

Rules, and why each one is a rule rather than a preference:

- **every section states its source table.** An assembled page is exactly the place where a
  plausible story can replace a record; a reader who cannot see which table a line came from
  cannot check it, and a section that cannot name its source does not belong on the page.
- **a section with no data is omitted, not printed empty.** A heading with nothing under it
  reads as a bug, and the page is read at a glance; "no contradictions recorded" is a claim the
  page makes in its header (by absence), not a row it spends a line on.
- **a retired belief is shown and marked, never hidden.** The point of an entity page is
  history: "the port was 8100 until March" is the answer to "why did you say 8100", and a page
  that silently drops retired rows destroys the only evidence that the store changed its mind.
  `entity_timeline.DEFAULT_STATES` already includes `archived`; this page adds the `valid_to`
  mark on top, so a superseded row is visibly *old* rather than merely present.
- **the page is bounded and says what it truncated.** An entity that appears on four hundred
  memories produces a page nobody reads; every section carries `total`, `shown` and the
  difference, so "showing 12 of 340 (328 not shown)" is printed rather than implied.
- **an unknown entity is an answer, not an empty page.** A blank page is indistinguishable from
  a broken query, and the two need different fixes — so a name with no node, no alias and no
  memory returns `found: False` with the reason, naming the three tables that were asked.

Read-only by construction: every statement here is a SELECT. `entity_nodes.get()` and
`aliases.aliases_for()` create their own tables on first use, so they are only called when the
table already exists — a page about a live store may not write the schema it is describing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

from . import aliases, entity_nodes, entity_timeline

# Section caps. Small on purpose: the page is read, not scrolled.
MAX_MEMORIES = 12
MAX_EDGES = 10
MAX_RELATIONS = 6
MAX_CHANGES = 8
MAX_CONTRADICTIONS = 5
MAX_RETIREMENTS = 5

# How many rows a helper may pull before this module slices them. The helpers cap their own
# queries, so this is the page's stated ceiling rather than a silent one.
READ_CAP = 200
# Pairwise conflict detection is quadratic; the scan is bounded so a hub entity cannot make
# the page slow (a 200-memory entity is 19,900 comparisons, which is a second, not a glance).
CONFLICT_SCAN_CAP = 25

CONTENT_CAP = 110


def _rows(store: Any, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    """One SELECT, no transaction — `store.transaction()` is a *write* transaction."""
    try:
        return [dict(row) for row in store._conn.execute(sql, tuple(params)).fetchall()]
    except Exception:  # noqa: BLE001 - a page reports a broken store, it never raises into a caller
        return []


def _tables_raw(store: Any) -> set[str] | None:
    """The tables that exist, or ``None`` when the store cannot answer.

    ``None`` and ``set()`` are not the same answer: an empty set means "this database has no
    tables", which is a claim about the store, while ``None`` means "I could not look", and a page
    built on that must say so instead of reporting the entity as absent.
    """
    try:
        return {str(row["name"]) for row in
                store._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()}
    except Exception:  # noqa: BLE001
        return None


def _tables(store: Any) -> set[str]:
    """The callable most of this module wants: a set, with "could not look" collapsed to empty."""
    return _tables_raw(store) or set()


def _as_datetime(value: Any) -> Any:
    """A stored timestamp as a datetime, or None - the caller formats, this only parses."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except Exception:  # noqa: BLE001
        return None


def _short(text: Any, cap: int = CONTENT_CAP) -> str:
    body = " ".join(str(text or "").split())
    return body if len(body) <= cap else body[: cap - 1].rstrip() + "…"


def _stamp(value: Any) -> str:
    text = str(value or "")
    return text[:10] if text else "unknown"


def _placeholders(values: Iterable[Any]) -> str:
    return ",".join("?" for _ in values)


@dataclass
class Limits:
    """Per-section caps, as data so a caller can widen one section without a new function."""

    memories: int = MAX_MEMORIES
    edges: int = MAX_EDGES
    relations: int = MAX_RELATIONS
    changes: int = MAX_CHANGES
    contradictions: int = MAX_CONTRADICTIONS
    retirements: int = MAX_RETIREMENTS


DEFAULT_LIMITS = Limits()


@dataclass
class Section:
    """One block of the page. `total` is what the store holds, `shown` what the page prints."""

    key: str
    source: str
    total: int
    shown: int
    items: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    @property
    def truncated(self) -> int:
        return max(0, int(self.total) - int(self.shown))

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "source": self.source,
            "total": self.total,
            "shown": self.shown,
            "truncated": self.truncated,
            "note": self.note,
            "items": self.items,
        }


@dataclass
class Page:
    entity: str
    found: bool = False
    entity_id: str = ""
    canonical_name: str = ""
    kind: str = "unknown"
    aliases: list[str] = field(default_factory=list)
    summary: str = ""
    resolved_by: str = ""
    sections: list[Section] = field(default_factory=list)
    reason: str = ""

    def section(self, key: str) -> Section | None:
        for candidate in self.sections:
            if candidate.key == key:
                return candidate
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "found": self.found,
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "kind": self.kind,
            "aliases": self.aliases,
            "summary": self.summary,
            "resolved_by": self.resolved_by,
            "reason": self.reason,
            "sections": [section.as_dict() for section in self.sections],
        }


# --------------------------------------------------------------------------- resolution

@dataclass
class _Resolved:
    entity_id: str
    name: str
    kind: str
    aliases: list[str]
    summary: str
    source: str


def _node_row(store: Any, key: str, tables: set[str]) -> dict[str, Any] | None:
    """An entity_nodes row, read directly: `entity_nodes.get()` would CREATE the table."""
    if entity_nodes.TABLE not in tables:
        return None
    rows = _rows(
        store,
        f"SELECT * FROM {entity_nodes.TABLE} "
        "WHERE entity_id = ? OR canonical_name = ? COLLATE NOCASE LIMIT 1",
        (key, key),
    )
    return rows[0] if rows else None


def _alias_row(store: Any, key: str, tables: set[str]) -> dict[str, Any] | None:
    if aliases.TABLE not in tables:
        return None
    rows = _rows(
        store,
        f"SELECT entity_id FROM {aliases.TABLE} WHERE alias_key = ? ORDER BY hits DESC LIMIT 1",
        (str(key).casefold(),),
    )
    return rows[0] if rows else None


def _aliases_for(store: Any, entity_id: str, tables: set[str]) -> list[str]:
    if aliases.TABLE not in tables:
        return []
    return sorted({str(row["alias"]) for row in _rows(
        store, f"SELECT alias FROM {aliases.TABLE} WHERE entity_id = ?", (str(entity_id),)) if row.get("alias")})


def _mentioned_name(store: Any, key: str) -> str:
    """The exact spelling a memory uses for this name, so alias lookups compare like with like."""
    rows = _rows(
        store,
        """SELECT value AS name FROM memories, json_each(memories.entities_json)
           WHERE json_valid(memories.entities_json) AND value = ? COLLATE NOCASE LIMIT 1""",
        (key,),
    )
    return str(rows[0]["name"]) if rows else ""


def resolve(store: Any, entity: str) -> _Resolved | None:
    """Find the entity by node id, canonical name, alias, or the spelling used in memories."""
    key = str(entity or "").strip()
    if not key:
        return None
    tables = _tables(store)
    node = _node_row(store, key, tables)
    if node is not None:
        return _Resolved(
            entity_id=str(node.get("entity_id") or key),
            name=str(node.get("canonical_name") or key),
            kind=str(node.get("kind") or "unknown"),
            aliases=_aliases_for(store, str(node.get("entity_id") or key), tables),
            summary=str(node.get("summary") or ""),
            source=entity_nodes.TABLE,
        )
    alias = _alias_row(store, key, tables)
    if alias is not None:
        named = _node_row(store, str(alias["entity_id"]), tables)
        name = str(named.get("canonical_name")) if named else str(alias["entity_id"])
        return _Resolved(
            entity_id=str(alias["entity_id"]),
            name=name,
            kind=str((named or {}).get("kind") or entity_nodes.classify(name)),
            aliases=_aliases_for(store, str(alias["entity_id"]), tables),
            summary=str((named or {}).get("summary") or ""),
            source="aliases",
        )
    mentioned = _mentioned_name(store, key)
    if mentioned:
        # No node yet: the store knows the name but has never given it a row. The page still
        # works, because the memories are the evidence and the node was only a summary of them.
        return _Resolved(
            entity_id=mentioned,
            name=mentioned,
            kind=entity_nodes.classify(mentioned),
            aliases=[],
            summary="",
            source="memories",
        )
    return None


# --------------------------------------------------------------------------- sections

def _memories(store: Any, name: str, limit: int,
              tables: set[str] | None = None) -> tuple[Section, list[dict[str, Any]]]:
    # `entity_timeline.memories()` resolves aliases, and `aliases.aliases_for()` creates its table on
    # first use. A page that is read-only cannot inherit that, so when the alias table does not exist
    # yet the page reads the memories table directly instead (same rows for an exact name; the alias
    # expansion is what is missing, and there are no aliases to expand from).
    if tables is not None and aliases.TABLE not in tables:
        rows = _rows(
            store,
            "SELECT id, kind, state, content, created_at, observed_at FROM memories "
            "WHERE (content LIKE ? OR entities_json LIKE ?) ORDER BY created_at DESC LIMIT ?",
            (f"%{name}%", f"%{name}%", int(READ_CAP)),
        )
    else:
        rows = entity_timeline.memories(store, name, limit=READ_CAP)
    # Newest first: the first line of a page should be what is believed *now*, and the history
    # below it is what makes the page worth reading.
    rows.sort(key=lambda row: str(row.get("observed_at") or row.get("created_at") or ""),
              reverse=True)
    ids = [str(row.get("id")) for row in rows]
    retired = _retired_at(store, ids)
    items = [{
        "id": str(row.get("id")),
        "kind": str(row.get("kind") or "unknown"),
        "state": str(row.get("state") or "active"),
        "when": _stamp(row.get("observed_at") or row.get("created_at")),
        "content": _short(row.get("content")),
        "retired_at": retired.get(str(row.get("id")), ""),
    } for row in rows[:limit]]
    note = ""
    if len(rows) >= READ_CAP:
        note = f"the query cap ({READ_CAP}) may hide older memories"
    return Section("memories", "memories.entities_json", len(rows), len(items), items, note), rows


def _retired_at(store: Any, ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    rows = _rows(
        store,
        f"SELECT id, valid_to FROM memories WHERE id IN ({_placeholders(ids)}) AND valid_to IS NOT NULL",
        ids,
    )
    return {str(row["id"]): str(row.get("valid_to") or "") for row in rows}


def _contents(store: Any, ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    rows = _rows(store, f"SELECT id, content FROM memories WHERE id IN ({_placeholders(ids)})", ids)
    return {str(row["id"]): str(row.get("content") or "") for row in rows}


def _edge_rows(store: Any, ids: list[str]) -> list[dict[str, Any]]:
    """Both sides of every edge touching the entity's memories.

    `add_edge()` stores pairs in canonical order, so filtering on `src_id` alone reports a
    connected memory as isolated — the one-sided-query bug `explain_memory` already documents.
    """
    if not ids:
        return []
    marks = _placeholders(ids)
    return _rows(
        store,
        f"""SELECT src_id, dst_id, relation, weight, evidence_count FROM edges
            WHERE src_id IN ({marks}) OR dst_id IN ({marks})
            ORDER BY weight DESC, src_id, dst_id""",
        (*ids, *ids),
    )


def _edge_sentence(store: Any, tables: set[str], left: str, right: str, relation: str) -> str:
    """Why these two are linked, in words — from `edge_explain`, which owns that sentence.

    `edge_explain.explain()` lazily creates its quarantine / veto / near-miss tables, so it is
    only asked when those tables already exist; otherwise the page falls back to the same
    relation phrase and strength word the explainer uses, with no write and no invented reason.
    """
    if {"assoc_feedback", "edge_quarantine", "near_miss"} <= tables:
        try:
            from . import edge_explain  # noqa: PLC0415 - optional, see docstring

            return edge_explain.explain(store, left, right, relation).sentence
        except Exception:  # noqa: BLE001 - a missing sentence is not a missing edge
            pass
    from . import edge_explain  # noqa: PLC0415

    phrase = edge_explain.RELATION_PHRASES.get(relation, relation)
    return f"{phrase} — recorded with no readable evidence trail"


def _edges(store: Any, ids: list[str], limit: int, relation_limit: int,
           tables: set[str]) -> Section:
    rows = _edge_rows(store, ids)
    if not rows:
        return Section("edges", "edges", 0, 0, [])
    mine = set(ids)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        src, dst = str(row.get("src_id")), str(row.get("dst_id"))
        # An edge between two of the entity's own memories has no "other" side; anchor it on
        # whichever end is this entity's first-mentioned memory so the sentence still names it.
        anchor = src if src in mine else dst
        other = dst if anchor == src else src
        groups.setdefault(str(row.get("relation") or "related"), []).append({
            "anchor": anchor,
            "other": other,
            "weight": round(float(row.get("weight") or 0.0), 3),
            "evidence": int(row.get("evidence_count") or 0),
        })
    ordered = sorted(groups, key=lambda rel: (-len(groups[rel]), rel))
    others = sorted({item["other"] for rel in ordered[:relation_limit] for item in groups[rel]})
    contents = _contents(store, others)

    relations: list[dict[str, Any]] = []
    budget = int(limit)
    explained = False  # one sentence per page: the rest would be prose nobody reads
    for relation in ordered[:relation_limit]:
        members = groups[relation]
        take = members[: max(1, min(3, budget))] if budget > 0 else []
        budget -= len(take)
        items = []
        for member in take:
            item = {key: value for key, value in member.items() if key != "anchor"}
            item["content"] = _short(contents.get(member["other"]), 70)
            if not explained:
                item["sentence"] = _edge_sentence(store, tables, member["anchor"],
                                                  member["other"], relation)
                explained = True
            items.append(item)
        weights = [member["weight"] for member in members]
        relations.append({
            "relation": relation,
            "count": len(members),
            "mean_weight": round(sum(weights) / len(weights), 3) if weights else 0.0,
            "max_weight": max(weights) if weights else 0.0,
            "shown": len(items),
            "items": items,
        })
        if budget <= 0:
            break
    shown = sum(relation["shown"] for relation in relations)
    return Section("edges", "edges", len(rows), shown, relations)


def _changes(store: Any, name: str, limit: int, tables: set[str] | None = None) -> Section:
    if tables is not None and aliases.TABLE not in tables:
        pairs = _rows(
            store,
            "SELECT newer.id AS new_id, older.id AS old_id, newer.created_at AS \"when\", "
            "older.content AS was, newer.content AS now FROM memories newer "
            "JOIN memories older ON newer.supersedes_id = older.id "
            "WHERE (newer.content LIKE ? OR older.content LIKE ?) ORDER BY newer.created_at DESC LIMIT ?",
            (f"%{name}%", f"%{name}%", int(READ_CAP)),
        )
        pairs = [{"new_id": r["new_id"], "old_id": r["old_id"],
                  "when": _as_datetime(r.get("when")), "was": r.get("was"), "now": r.get("now")}
                 for r in pairs]
    else:
        pairs = entity_timeline.changes(store, name, limit=READ_CAP)
    if not pairs or limit <= 0:
        return Section("changes", "memories.supersedes_id", 0, 0, [])
    # Most recent first: "what changed lately" is the question; the full history is the memory
    # section's job.
    items = [{
        "old_id": pair["old_id"],
        "new_id": pair["new_id"],
        "when": _stamp(pair["when"].isoformat() if pair.get("when") else ""),
        "was": _short(pair["was"], 70),
        "now": _short(pair["now"], 70),
    } for pair in reversed(pairs[-limit:])]
    return Section("changes", "memories.supersedes_id", len(pairs), len(items), items)


def _contradictions(store: Any, memory_ids: list[str], limit: int) -> Section:
    """Active beliefs about the same entity that disagree — from `contradiction`, which owns it."""
    if len(memory_ids) < 2:
        return Section("contradictions", "memories", 0, 0, [])
    rows = _rows(
        store,
        f"""SELECT id, content, subject, predicate, object_value, state FROM memories
            WHERE id IN ({_placeholders(memory_ids)}) AND state = 'active'""",
        memory_ids,
    )
    rows = rows[:CONFLICT_SCAN_CAP]
    if len(rows) < 2:
        return Section("contradictions", "memories", 0, 0, [])
    try:
        from . import contradiction  # noqa: PLC0415 - pure comparison, no writes

        conflicts_with = contradiction.conflicts_with
    except Exception:  # noqa: BLE001
        return Section("contradictions", "memories", 0, 0, [])

    found: list[dict[str, Any]] = []
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            reason = conflicts_with(left, right)
            if reason:
                found.append({
                    "left_id": str(left["id"]),
                    "right_id": str(right["id"]),
                    "reason": str(reason),
                    "was": _short(left.get("content"), 70),
                    "now": _short(right.get("content"), 70),
                })
    if not found:
        return Section("contradictions", "memories", 0, 0, [])
    return Section("contradictions", "memories", len(found), min(limit, len(found)),
                   found[:limit])


def _retirements(store: Any, ids: list[str], limit: int) -> Section:
    """Beliefs the store has retired — shown with their date, because the page is history."""
    if not ids:
        return Section("retirements", "memories.valid_to", 0, 0, [])
    rows = _rows(
        store,
        f"""SELECT id, content, valid_to FROM memories
            WHERE id IN ({_placeholders(ids)}) AND valid_to IS NOT NULL
            ORDER BY valid_to DESC""",
        ids,
    )
    if not rows:
        return Section("retirements", "memories.valid_to", 0, 0, [])
    shown = rows[:limit]
    successors = _successors(store, [str(row["id"]) for row in shown])
    items = [{
        "id": str(row["id"]),
        "retired_at": _stamp(row.get("valid_to")),
        "content": _short(row.get("content"), 70),
        "replaced_by": successors.get(str(row["id"]), ""),
    } for row in shown]
    return Section("retirements", "memories.valid_to", len(rows), len(items), items)


def _successors(store: Any, ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    rows = _rows(
        store,
        f"""SELECT id, supersedes_id FROM memories
            WHERE supersedes_id IN ({_placeholders(ids)})""",
        ids,
    )
    return {str(row["supersedes_id"]): str(row["id"]) for row in rows}


# --------------------------------------------------------------------------- assembly

def build(store: Any, entity: str, *, limits: Limits | None = None) -> Page:
    """Assemble one entity's page. Never raises: a broken store is a page that says so."""
    caps = limits or DEFAULT_LIMITS
    page = Page(entity=str(entity or ""))
    if _tables_raw(store) is None:
        # Rule (c): undecidable is reported, never turned into "no such entity".
        page.reason = "the store could not be read: sqlite_master is unqueryable"
        return page
    try:
        resolved = resolve(store, page.entity)
        if resolved is None:
            page.reason = (f"unknown entity: no row in {entity_nodes.TABLE}, aliases or memories "
                           f"names {page.entity!r}")
            return page
        page.found = True
        page.entity_id = resolved.entity_id
        page.canonical_name = resolved.name
        page.kind = resolved.kind
        page.aliases = list(resolved.aliases)
        page.summary = resolved.summary
        page.resolved_by = resolved.source
        tables = _tables(store)

        memories_section, rows = _memories(store, resolved.name, caps.memories, tables)
        ids = [str(row.get("id")) for row in rows]
        sections = [
            memories_section,
            _edges(store, ids, caps.edges, caps.relations, tables),
            _changes(store, resolved.name, caps.changes, tables),
            _contradictions(store, ids, caps.contradictions),
            _retirements(store, ids, caps.retirements),
        ]
        # Rule (b): an empty section is dropped rather than printed as a heading with nothing
        # under it. `Section` still exists for the caller; `page.section(key)` returning None
        # means "the store had nothing", which `as_dict()` makes explicit.
        page.sections = [section for section in sections if section.total > 0]
        return page
    except Exception as exc:  # noqa: BLE001 - a page must not take a caller down with it
        # The exception's message can carry a machine path; the type is enough to act on.
        page.found = False
        page.sections = []
        page.reason = f"the store could not be read: {type(exc).__name__}"
        return page


def render(store: Any, entity: str, *, limits: Limits | None = None) -> list[str]:
    """The page as printable lines, one section per block, truncation stated."""
    page = build(store, entity, limits=limits)
    if not page.found:
        return [f"entity {page.entity!r}: {page.reason}"]
    header = f"entity {page.canonical_name} [{page.kind}]"
    if page.entity_id and page.entity_id != page.canonical_name:
        header += f" id={page.entity_id}"
    lines = [header]
    if page.aliases:
        lines.append("  also known as: " + ", ".join(page.aliases[:8]))
    if page.summary:
        lines.append("  " + _short(page.summary, 140))
    for section in page.sections:
        count = f"{section.shown} of {section.total}"
        if section.truncated:
            count += f" ({section.truncated} not shown)"
        lines.append(f"  {section.key}: {count}  [source: {section.source}]")
        for item in section.items:
            lines.append("    " + _item_line(section.key, item))
        if section.note:
            lines.append(f"    note: {section.note}")
    return lines


def _item_line(key: str, item: dict[str, Any]) -> str:
    if key == "memories":
        mark = f"  !retired {item['retired_at']}" if item.get("retired_at") else ""
        state = "" if item["state"] == "active" else f"  ({item['state']})"
        return f"{item['when']}  {item['content']}{state}{mark}  {item['id'][:8]}"
    if key == "edges":
        return (f"{item['relation']} ×{item['count']} mean {item['mean_weight']}"
                f"{' (showing ' + str(item['shown']) + ')' if item['shown'] < item['count'] else ''}")
    if key == "changes":
        return f"{item['when']}  {item['was']!r} → {item['now']!r}"
    if key == "contradictions":
        return f"{item['left_id'][:8]} vs {item['right_id'][:8]}: {item['reason']}"
    if key == "retirements":
        tail = f" → replaced by {item['replaced_by'][:8]}" if item.get("replaced_by") else ""
        return f"{item['retired_at']}  {item['content']}{tail}"
    return str(item)


def report(store: Any) -> dict[str, Any]:
    """What the page can say about a store without naming an entity."""
    tables = _tables(store)
    return {
        "entity_nodes": entity_nodes.TABLE in tables,
        "aliases": "aliases" in tables,
        "edges": "edges" in tables,
        "cap": {"memories": MAX_MEMORIES, "edges": MAX_EDGES, "changes": MAX_CHANGES},
    }
