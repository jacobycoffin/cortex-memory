"""A one-page profile of a person, assembled from the facts that mention them (#61).

"What do I know about Jacoby?" is the question the store gets asked most and answers worst: today it
is a search for a name, and the reader still has to sift the results themselves. This assembles the
answer instead — how many facts mention the person, what kinds they are, when they were first and
last seen, what is still outstanding, and the facts themselves, newest first.

Three limits are deliberate:

- **the profile is generated, never maintained.** It reads `entities_json` on the rows that already
  hold the facts, so it cannot drift from the store or become a second copy of the truth. There is
  nothing to keep up to date, and nothing that can be right in the profile and wrong in the store.
- **a person is only a person when the entity layer says so.** `people()` is not a name detector; it
  takes `entity_nodes.classify()`'s opinion, honours a kind a human corrected on the node, and
  otherwise stays quiet. A guessed wrong person is a wrong answer to every question built on it.
- **a failure reads as "nothing known", not as an exception.** The profile is a view; a broken view
  must never break the caller, so a store error yields an empty profile rather than a traceback.

`entity_nodes` and `aliases` are imported lazily: they own the node and alias tables, and this module
reads those tables directly (read-only) when the modules are not importable, so a page still renders
between the items landing.
"""

from __future__ import annotations

from collections import Counter
from importlib import import_module
from typing import Any

#: Names that classify as people even when the entity layer cannot be imported. It mirrors
#: `entity_nodes.PEOPLE`; the real lexicon wins whenever that module is available.
_FALLBACK_PEOPLE = ("jacoby", "hermes", "ledger", "teknium", "kaya", "jarvis", "rowan")

MAX_LINES = 25
DEFAULT_FACTS = 50
MAX_OPEN_QUESTIONS = 5

#: The columns a profile shows. `entities_json` is left out: the caller already knows who this is.
FACT_COLUMNS = (
    "m.id", "m.content", "m.kind", "m.confidence", "m.importance", "m.state",
    "m.created_at", "m.updated_at", "m.last_used_at", "m.used_count", "m.subject", "m.predicate",
)


def _module(name: str) -> Any:
    """The sibling module if it can be imported, else None. Never raises."""
    try:
        return import_module(f"{__package__}.{name}" if __package__ else name)
    except Exception:  # noqa: BLE001 - the page must render before the entity layer lands
        return None


# --------------------------------------------------------------------------- people


def classify(name: str) -> str:
    """The kind a name most likely is, using the entity layer's lexicon when it is importable."""
    layer = _module("entity_nodes")
    if layer is not None:
        try:
            return str(layer.classify(name))
        except Exception:  # noqa: BLE001
            pass
    return "person" if str(name or "").strip().casefold() in _FALLBACK_PEOPLE else "unknown"


def _node(store: Any, name: str) -> dict[str, Any]:
    """The entity row for a name, read directly so a page works before `entity_nodes` is wired."""
    try:
        row = store._conn.execute(
            """SELECT canonical_name, kind FROM entity_nodes
               WHERE LOWER(canonical_name) = LOWER(?) OR LOWER(entity_id) = LOWER(?) LIMIT 1""",
            (str(name), str(name)),
        ).fetchone()
    except Exception:  # noqa: BLE001 - no node table yet is not an error
        return {}
    return dict(row) if row else {}


def _is_person(store: Any, name: str) -> bool:
    """A kind a human corrected outranks the lexicon; an `unknown` node is no opinion at all."""
    node = _node(store, name)
    kind = str(node.get("kind") or "").strip().lower()
    if kind and kind != "unknown":
        return kind == "person"
    return classify(name) == "person"


# --------------------------------------------------------------------------- reads


def _names_for(store: Any, name: str) -> list[str]:
    """The name plus whatever the alias table says means the same person, first spelling first."""
    label = str(name or "").strip()
    if not label:
        return []
    names = [label]
    for alias in _alias_names(store, label):
        text = str(alias or "").strip()
        if text and text.casefold() not in {item.casefold() for item in names}:
            names.append(text)
    return names


def _alias_names(store: Any, name: str) -> list[str]:
    layer = _module("aliases")
    if layer is not None:
        try:
            target = layer.resolve(store, name) or name
            return [alias.alias for alias in layer.aliases_for(store, target)]
        except Exception:  # noqa: BLE001
            return []
    try:
        key = str(name).strip().casefold()
        row = store._conn.execute(
            "SELECT entity_id FROM entity_aliases WHERE LOWER(alias) = ? ORDER BY hits DESC LIMIT 1",
            (key,),
        ).fetchone()
        target = str(row["entity_id"]) if row else name
        rows = store._conn.execute(
            "SELECT alias FROM entity_aliases WHERE LOWER(entity_id) = LOWER(?)", (target,)
        ).fetchall()
    except Exception:  # noqa: BLE001 - aliases are an enhancement, never a requirement
        return []
    return [str(item["alias"]) for item in rows]


def _mentions(store: Any, *, limit: int) -> list[dict[str, Any]]:
    """One row per entity name attached to at least one fact: how many, and when first/last seen."""
    try:
        rows = store._conn.execute(
            """SELECT j.value AS name, COUNT(DISTINCT m.id) AS fact_count,
                      MIN(m.created_at) AS first_seen, MAX(m.created_at) AS last_seen
               FROM memories m JOIN json_each(m.entities_json) AS j
               WHERE m.state IN ('active','cold') AND json_valid(m.entities_json)
                 AND json_type(m.entities_json) = 'array'
               GROUP BY LOWER(j.value)
               ORDER BY fact_count DESC, name ASC
               LIMIT ?""",
            (int(limit),),
        ).fetchall()
    except Exception:  # noqa: BLE001 - a view never raises into its caller
        return []
    described: list[dict[str, Any]] = []
    for row in rows:
        name = str(row["name"] or "").strip()
        if not name:
            continue
        described.append({
            "name": name,
            "fact_count": int(row["fact_count"] or 0),
            "first_seen": str(row["first_seen"] or ""),
            "last_seen": str(row["last_seen"] or ""),
        })
    return described


# How many *names* to consider before filtering to people. This is deliberately independent of how
# many profiles the caller asks for: the first version widened the scan to `limit * 5`, so
# `index(limit=8)` looked at 40 names, found no person among them, and reported an empty list while
# `people()` happily returned three. "How many to return" and "how deep to look" are different
# questions and must not share one number.
DEFAULT_SCAN = 4000


def _scan(store: Any, *, limit: int = DEFAULT_SCAN) -> list[dict[str, Any]]:
    """Every mention grouped, over a window wide enough that filtering by person still finds them."""
    return _mentions(store, limit=max(int(limit), DEFAULT_SCAN))


def _people_rows(store: Any, *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in _scan(store):
        folded = row["name"].casefold()
        if not row["fact_count"] or folded in seen:
            continue
        if not _is_person(store, row["name"]):
            continue
        seen.add(folded)
        rows.append(row)
    return rows


def people(store: Any, limit: int = 200) -> list[str]:
    """Names the store knows as people and has at least one fact about, busiest first."""
    return [row["name"] for row in _people_rows(store, limit=int(limit))][: int(limit)]


def _fact_rows(store: Any, names: list[str], *, limit: int, kind: str = "") -> list[dict[str, Any]]:
    """Facts naming any of `names`, newest first. Raises on a broken store; callers decide."""
    lowered = [str(name).strip().casefold() for name in names if str(name or "").strip()]
    if not lowered:
        return []
    marks = ",".join("?" for _ in lowered)
    clause = " AND LOWER(m.kind) = ?" if kind else ""
    params: list[Any] = list(lowered) + ([str(kind).lower()] if kind else []) + [int(limit)]
    rows = store._conn.execute(
        f"""SELECT {', '.join(FACT_COLUMNS)}
            FROM memories m JOIN json_each(m.entities_json) AS j
            WHERE LOWER(j.value) IN ({marks}) AND m.state IN ('active','cold')
              AND json_valid(m.entities_json) AND json_type(m.entities_json) = 'array'{clause}
            ORDER BY m.created_at DESC, m.id DESC LIMIT ?""",
        tuple(params),
    ).fetchall()
    return [dict(row) for row in rows]


def facts_about(store: Any, name: str, limit: int = DEFAULT_FACTS) -> list[dict[str, Any]]:
    """Facts that name this person, newest first. Never raises; an unreadable store reads as empty."""
    try:
        return _fact_rows(store, _names_for(store, name), limit=int(limit))
    except Exception:  # noqa: BLE001
        return []


def _totals(store: Any, names: list[str]) -> dict[str, Any]:
    """Counts over *every* fact about the person, not just the ones a page happens to show."""
    lowered = [str(name).strip().casefold() for name in names if str(name or "").strip()]
    if not lowered:
        return {"fact_count": 0, "first_seen": "", "last_seen": "", "by_kind": {}}
    marks = ",".join("?" for _ in lowered)
    rows = store._conn.execute(
        f"""SELECT m.kind AS kind, COUNT(DISTINCT m.id) AS count,
                   MIN(m.created_at) AS first_seen, MAX(m.created_at) AS last_created,
                   MAX(COALESCE(m.last_used_at, '')) AS last_used
            FROM memories m JOIN json_each(m.entities_json) AS j
            WHERE LOWER(j.value) IN ({marks}) AND m.state IN ('active','cold')
              AND json_valid(m.entities_json) AND json_type(m.entities_json) = 'array'
            GROUP BY m.kind""",
        tuple(lowered),
    ).fetchall()
    by_kind: Counter[str] = Counter()
    created: list[str] = []
    lasts: list[str] = []
    for row in rows:
        by_kind[str(row["kind"] or "unknown")] += int(row["count"] or 0)
        created.append(str(row["first_seen"] or ""))
        # "seen" is either when the fact was recorded or when it was last put to use: a fact about
        # someone that was recalled yesterday is evidence the person came up yesterday.
        for value in (row["last_created"], row["last_used"]):
            if str(value or ""):
                lasts.append(str(value))
    return {
        "fact_count": sum(by_kind.values()),
        "first_seen": min((value for value in created if value), default=""),
        "last_seen": max(lasts, default=""),
        "by_kind": by_kind,
    }


# --------------------------------------------------------------------------- profile


def _day(stamp: str) -> str:
    """The date part of an ISO timestamp, so a sentence does not carry microseconds."""
    value = str(stamp or "").strip()
    return value[:10] if len(value) >= 10 and value[4] == "-" else value


def _sentence(name: str, total: int, by_kind: Counter[str], first: str, last: str,
              open_count: int = 0) -> str:
    parts = [f"{name}: {total} fact{'' if total == 1 else 's'}"]
    if by_kind:
        described = ", ".join(f"{kind} {count}" for kind, count in by_kind.most_common(3))
        parts.append(f"({described})")
    first_day, last_day = _day(first), _day(last)
    if first_day and last_day and first_day != last_day:
        parts.append(f"seen {first_day} to {last_day}")
    elif last_day:
        parts.append(f"seen {last_day}")
    if open_count:
        parts.append(f"{open_count} open question{'' if open_count == 1 else 's'}")
    return ", ".join(parts) + "."


def _empty(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "person",
        "fact_count": 0,
        "first_seen": "",
        "last_seen": "",
        "facts": [],
        "by_kind": Counter(),
        "open_questions": [],
        "sentence": _sentence(name, 0, Counter(), "", ""),
    }


def _display_name(store: Any, name: str) -> str:
    canonical = str(_node(store, name).get("canonical_name") or "").strip()
    return canonical or str(name).strip()


def profile(store: Any, name: str, limit: int = DEFAULT_FACTS) -> dict[str, Any] | None:
    """Everything the store can say about one person, or None if it has never mentioned them."""
    label = str(name or "").strip()
    if not label:
        return None
    try:
        names = _names_for(store, label)
        totals = _totals(store, names)
        if not totals["fact_count"]:
            return None
        facts = _fact_rows(store, names, limit=int(limit))
        open_questions = _fact_rows(store, names, limit=MAX_OPEN_QUESTIONS, kind="prospective")
    except Exception:  # noqa: BLE001 - a broken view is empty, not an exception
        return _empty(label)
    display = _display_name(store, label)
    by_kind: Counter[str] = totals["by_kind"]
    return {
        "name": display,
        "kind": "person",
        "fact_count": totals["fact_count"],
        "first_seen": totals["first_seen"],
        "last_seen": totals["last_seen"],
        "facts": facts,
        "by_kind": by_kind,
        "open_questions": open_questions,
        "sentence": _sentence(display, totals["fact_count"], by_kind, totals["first_seen"],
                              totals["last_seen"], len(open_questions)),
    }


def render(store: Any, name: str, style: str = "markdown", *, limit: int = MAX_LINES) -> str:
    """The profile as a small block: a heading, when they were seen, the facts, what is open."""
    described = profile(store, name)
    if not described or not described["fact_count"]:
        return ""
    markdown = str(style or "markdown").strip().lower() != "text"
    heading = f"## {described['name']}" if markdown else described["name"]
    bullet = "-" if markdown else "*"
    kinds = ", ".join(f"{kind} {count}" for kind, count in described["by_kind"].most_common(4))
    plural = "" if described["fact_count"] == 1 else "s"
    seen = f"seen first {_day(described['first_seen']) or 'never'}, last " \
           f"{_day(described['last_seen']) or 'never'}; {described['fact_count']} fact{plural}"
    lines = [heading, seen + (f" ({kinds})" if kinds else "")]
    for fact in described["facts"]:
        text = " ".join(str(fact.get("content") or "").split())
        if text:
            lines.append(f"{bullet} {text[:200]} ({fact.get('kind') or 'fact'}, {_day(fact.get('created_at'))})")
    if described["open_questions"]:
        lines.append("open questions:" if markdown else "open questions")
        for question in described["open_questions"]:
            text = " ".join(str(question.get("content") or "").split())
            if text:
                lines.append(f"{bullet} {text[:200]} ({_day(question.get('created_at'))})")
    if len(lines) > int(limit):
        kept = lines[: max(int(limit) - 1, 1)]
        kept.append(f"{bullet} ... and {len(lines) - len(kept)} more")
        lines = kept
    return "\n".join(lines)


def index(store: Any, limit: int = 50) -> list[dict[str, Any]]:
    """Everyone the store knows, busiest first: the list to pick a profile from."""
    rows = [{"name": row["name"], "fact_count": row["fact_count"], "last_seen": row["last_seen"]}
            for row in _people_rows(store, limit=int(limit))]
    rows.sort(key=lambda row: (-row["fact_count"], row["name"].casefold()))
    return rows[: int(limit)]
