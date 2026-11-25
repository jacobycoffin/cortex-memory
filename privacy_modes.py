"""How much of memory a caller is allowed to see, decided by the caller (#183)

Every other surface here answers a question *about* memory: what was recalled, why a belief is held,
what was written. This one answers the question that comes before all of them - **should this caller be
shown the store at all** - and it exists because the answer changes by context: a shared screen, a
projected terminal, a phone handed to somebody, an agent asked to draft something quotable.

The design starts from the failure it prevents. A filter placed *inside* retrieval is a filter that can be
argued with: a better query, a wider limit, a fallback path that skips the check. So the mode is resolved
once, at the edge, and everything downstream is handed a **narrower store**, not a promise to be careful.

Six rules, each of which is a way this could leak instead of protect:

- **modes only ever narrow.** The visible set is `normal AND mode`, never `mode OR something`. There is no
  mode that shows more than `normal`, and `describe()` reports the widening list as empty rather than
  leaving it to inspection - a "debug" or "everything" mode is the one an operator enables at 2am and
  forgets, and it is exactly the mode that ends up in a screenshot.
- **the mode comes from the caller's context and is never inferred from content.** "This memory looks
  financial, so they probably wanted private" is an inference that is wrong in the direction that leaks:
  the one note that does not match the pattern is the one that gets shown. A caller states its mode; a
  caller that states nothing gets `normal`, and a caller that states nonsense gets `off_record`.
- **the mode in force is stated in every output.** A blank page is indistinguishable from a broken store,
  so every view carries `privacy mode: private; excluded 412 of 3,843 memories` - the number is what makes
  the emptiness explainable, and its absence is what makes it a bug report.
- **`off_record` refuses writes with a reason.** Silently dropping a write means the caller believes the
  memory is kept and behaves as if it is; the refusal names the mode so the caller can say "I did not keep
  that" instead of discovering it a week later.
- **an unknown mode falls back to the most restrictive known mode**, not to `normal`. A typo, a version
  skew, an old client sending `secret` - every one of those is a request for *less* visibility, and
  defaulting to `normal` turns a failed lookup into a disclosure.
- **reading a mode never writes.** Tag and pin tables are created only by the code that tags and pins; a
  privacy check that creates schema has a side effect on the store it was asked to protect.

The one place content is consulted is **what a memory is tagged as**, not whether it looks sensitive to
this call: `private` excludes memories carrying a `finance`/`health`/`credential` tag, and only tags the
store already holds are authoritative. `consent_gate.classify()` runs as a **backstop for untagged rows**,
because a store with no tags yet would otherwise show everything in `private` mode, which is the leak this
module is for. That is an upper bound on what is hidden, not a claim that the hidden set is exactly right.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

NORMAL = "normal"
PRIVATE = "private"
SEALED = "sealed"
OFF_RECORD = "off_record"

# Ordered widest to narrowest. Everything downstream uses this order, so "most restrictive" has one answer.
MODES: tuple[str, ...] = (NORMAL, PRIVATE, SEALED, OFF_RECORD)
RESTRICTIVENESS: dict[str, int] = {name: index for index, name in enumerate(MODES)}

# The classes `private` hides. Exactly the three named by the item: finance, health, credential. A fourth
# class added here later is a behaviour change, so the list is explicit rather than "everything consent
# treats as protected" - `work_systems` is refused at write and does not need hiding at read.
PRIVATE_CLASSES: tuple[str, ...] = ("credential", "finance", "health")

# Context keys a caller may state the mode under. The first one present wins; no other key is consulted,
# and nothing in the text of a request is parsed for intent.
CONTEXT_KEYS: tuple[str, ...] = ("privacy_mode", "mode")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]] | None:
    """Rows, or `None` when the store cannot be read. `None` is not `[]`: an unreadable store is not empty."""
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return None


@dataclass(frozen=True)
class Mode:
    name: str
    reads: bool
    writes: bool
    why: str

    @property
    def rank(self) -> int:
        return RESTRICTIVENESS.get(self.name, len(MODES) - 1)

    def as_dict(self) -> dict[str, Any]:
        return {"mode": self.name, "reads": self.reads, "writes": self.writes, "why": self.why}


_MODE_TABLE: dict[str, Mode] = {
    NORMAL: Mode(NORMAL, True, True, "the store as it is; the widest view that exists"),
    PRIVATE: Mode(PRIVATE, True, True,
                  "reads, minus credential/finance/health memories; writes are unaffected"),
    SEALED: Mode(SEALED, True, False,
                 "reads only what was explicitly pinned as safe; refuses new memories, because nothing "
                 "new is safe until somebody says so"),
    OFF_RECORD: Mode(OFF_RECORD, False, False, "retrieves nothing and keeps nothing"),
}


@dataclass
class Resolution:
    """Which mode is in force and why - the `why` is printed, so an unexpected mode is never silent."""

    mode: Mode
    requested: str = ""
    fell_back: bool = False
    reason: str = ""

    @property
    def name(self) -> str:
        return self.mode.name

    def as_dict(self) -> dict[str, Any]:
        return {"mode": self.name, "requested": self.requested, "fell_back": self.fell_back,
                "reason": self.reason}


def mode_for(name: str) -> Mode:
    """A known mode by name; anything else is `off_record`, not `normal` (rule e)."""
    return _MODE_TABLE.get(str(name or "").strip().casefold(), _MODE_TABLE[OFF_RECORD])


def _stated(context: Any) -> str:
    """Read the caller's stated mode. Only the explicit key, never the content of the request."""
    if context is None:
        return ""
    if isinstance(context, str):
        return context
    if isinstance(context, Mapping):
        for key in CONTEXT_KEYS:
            if key in context and context[key] is not None:
                return str(context[key])
        return ""
    for key in CONTEXT_KEYS:
        value = getattr(context, key, None)
        if value:
            return str(value)
    return ""


def resolve(context: Any = None) -> Resolution:
    """The mode in force for this caller.

    A missing mode is `normal`: silence is the default surface, not a claim about intent. A mode that is
    stated but unrecognised falls back to the most restrictive *known* mode, because a typo is a request
    for protection that failed to parse, never a request for everything.
    """
    stated = _stated(context).strip().casefold()
    if not stated:
        return Resolution(_MODE_TABLE[NORMAL], "", False, "no mode stated; the default surface is normal")
    if stated in _MODE_TABLE:
        return Resolution(_MODE_TABLE[stated], stated, False, f"caller stated {stated!r}")
    return Resolution(
        _MODE_TABLE[OFF_RECORD], stated, True,
        f"{stated!r} is not a known mode; fell back to {OFF_RECORD!r}, the most restrictive known mode",
    )


# --------------------------------------------------------------------------------------------------
# Tags and safe pins: the only content judgements this module makes, and they are recorded, not guessed.
# --------------------------------------------------------------------------------------------------

def ensure_schema(store: Any) -> bool:
    """Create the tag and pin tables. Called by `tag()`/`pin_safe()`, never by a read path."""
    try:
        with store.transaction() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS privacy_tags ("
                "memory_id TEXT NOT NULL, tag TEXT NOT NULL, tagged_at TEXT NOT NULL, "
                "PRIMARY KEY (memory_id, tag))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS privacy_safe_pins ("
                "memory_id TEXT PRIMARY KEY, pinned_at TEXT NOT NULL, why TEXT NOT NULL)"
            )
        return True
    except Exception:  # noqa: BLE001
        return False


def tag(store: Any, memory_id: str, tag_name: str, *, at: datetime | None = None) -> bool:
    """Record that a memory belongs to a sensitivity class. Explicit, reviewable, reversible."""
    ensure_schema(store)
    try:
        with store.transaction() as conn:
            conn.execute(
                "INSERT INTO privacy_tags(memory_id, tag, tagged_at) VALUES(?,?,?) "
                "ON CONFLICT(memory_id, tag) DO UPDATE SET tagged_at = excluded.tagged_at",
                (str(memory_id), str(tag_name).strip().casefold(), (at or _now()).isoformat()),
            )
        return True
    except Exception:  # noqa: BLE001
        return False


def untag(store: Any, memory_id: str, tag_name: str) -> bool:
    try:
        with store.transaction() as conn:
            conn.execute("DELETE FROM privacy_tags WHERE memory_id = ? AND tag = ?",
                         (str(memory_id), str(tag_name).strip().casefold()))
        return True
    except Exception:  # noqa: BLE001
        return False


def pin_safe(store: Any, memory_id: str, *, why: str, at: datetime | None = None) -> bool:
    """Explicitly pin a memory as safe to show in `sealed` mode.

    This is deliberately **not** the `memories.pinned` column: pinning is already used as a scratchpad
    (#151, 6 pinned rows consuming a whole prompt budget), so a store-wide `pinned` flag would make
    `sealed` show whatever happened to be pinned for other reasons. "Safe to show on a projector" is a
    different sentence from "keep this in mind", and it gets its own record with its reason attached.
    """
    ensure_schema(store)
    try:
        with store.transaction() as conn:
            conn.execute(
                "INSERT INTO privacy_safe_pins(memory_id, pinned_at, why) VALUES(?,?,?) "
                "ON CONFLICT(memory_id) DO UPDATE SET pinned_at = excluded.pinned_at, why = excluded.why",
                (str(memory_id), (at or _now()).isoformat(), str(why or "")),
            )
        return True
    except Exception:  # noqa: BLE001
        return False


def unpin_safe(store: Any, memory_id: str) -> bool:
    try:
        with store.transaction() as conn:
            conn.execute("DELETE FROM privacy_safe_pins WHERE memory_id = ?", (str(memory_id),))
        return True
    except Exception:  # noqa: BLE001
        return False


def _tags_by_id(store: Any) -> dict[str, set[str]]:
    rows = _rows(store, "SELECT memory_id, tag FROM privacy_tags")
    found: dict[str, set[str]] = {}
    for row in rows or []:
        found.setdefault(str(row["memory_id"]), set()).add(str(row["tag"]).casefold())
    return found


def _pins(store: Any) -> set[str]:
    rows = _rows(store, "SELECT memory_id FROM privacy_safe_pins")
    return {str(row["memory_id"]) for row in rows or []}


def _classify(content: str) -> str:
    try:
        from .consent_gate import classify  # noqa: PLC0415 - optional helper

        return classify(content)
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------------------------------
# The view
# --------------------------------------------------------------------------------------------------

@dataclass
class View:
    """What one caller gets, with the reason it is not more.

    `statement` is always present, including when nothing was excluded: a mode that is in force but
    unstated is indistinguishable from no mode at all.
    """

    resolution: Resolution
    memories: list[dict[str, Any]] = field(default_factory=list)
    total: int = 0
    excluded: int = 0
    excluded_by_reason: dict[str, int] = field(default_factory=dict)
    readable: bool = True

    @property
    def mode(self) -> str:
        return self.resolution.name

    @property
    def statement(self) -> str:
        if not self.readable:
            return (f"privacy mode: {self.mode}; the store could not be read, so 0 memories are shown "
                    f"(this is not an empty store)")
        head = f"privacy mode: {self.mode}"
        if not self.resolution.requested:
            head += " (default)"
        elif self.resolution.fell_back:
            head += f" (requested {self.resolution.requested!r})"
        if self.excluded == 0:
            return f"{head}; nothing excluded of {self.total} memories"
        detail = ", ".join(f"{name} {count}" for name, count in sorted(self.excluded_by_reason.items()))
        return f"{head}; excluded {self.excluded} of {self.total} memories ({detail})"

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "statement": self.statement,
            "requested": self.resolution.requested,
            "fell_back": self.resolution.fell_back,
            "readable": self.readable,
            "memories": self.memories,
            "total": self.total,
            "excluded": self.excluded,
            "excluded_by_reason": dict(sorted(self.excluded_by_reason.items())),
        }


def _exclusion(mode: str, row: dict[str, Any], tags: set[str], pins: set[str]) -> str:
    """Why this row is not visible in this mode - `""` when it is.

    Each branch can only remove a row. `normal` is the baseline the others subtract from, so there is no
    code path here that adds a memory a wider mode would have hidden.
    """
    if mode == NORMAL:
        return ""
    if mode == OFF_RECORD:
        return "off_record"
    if mode == SEALED:
        return "" if str(row.get("id")) in pins else "not pinned safe"
    # private
    explicit = tags & set(PRIVATE_CLASSES)
    if explicit:
        return sorted(explicit)[0]
    if not tags:
        guessed = _classify(str(row.get("content") or ""))
        if guessed in PRIVATE_CLASSES:
            return guessed
    return ""


def _visible_rows(store: Any, *, with_content: bool) -> list[dict[str, Any]] | None:
    columns = "id, kind, content, confidence" if with_content else "id, kind"
    return _rows(store, f"SELECT {columns} FROM memories WHERE state <> 'archived'")


def view(store: Any, context: Any = None, *, rows: Iterable[dict[str, Any]] | None = None,
         with_content: bool = True) -> View:
    """Filter the store for this caller. `rows` lets a caller pass an already-fetched candidate set."""
    resolution = resolve(context)
    mode = resolution.mode
    if not mode.reads:
        # off_record does not even load content: the only thing counted is how much was not looked at.
        total = len(list(rows)) if rows is not None else _store_count(store)
        if total is None:
            return View(resolution, [], 0, 0, {}, readable=False)
        return View(resolution, [], total, total, {OFF_RECORD: total})
    loaded = list(rows) if rows is not None else _visible_rows(store, with_content=with_content)
    if loaded is None:
        return View(resolution, [], 0, 0, {}, readable=False)
    tags = _tags_by_id(store)
    pins = _pins(store)
    kept: list[dict[str, Any]] = []
    excluded: dict[str, int] = {}
    for row in loaded:
        memory_id = str(row.get("id") or "")
        reason = _exclusion(mode.name, row, tags.get(memory_id, set()), pins)
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        kept.append(row)
    return View(resolution, kept, len(loaded), len(loaded) - len(kept), excluded)


def _store_count(store: Any) -> int | None:
    rows = _rows(store, "SELECT COUNT(*) AS count FROM memories WHERE state <> 'archived'")
    if not rows:
        return None
    return int(rows[0]["count"] or 0)


def census(store: Any) -> dict[str, Any]:
    """How much of the store each mode would hide - counts only, no content, no ids.

    This is the operator-facing signal: "private would exclude 412" is a number that can be checked
    against what the store actually holds, whereas a claim that filtering works is not.
    """
    rows = _visible_rows(store, with_content=True)
    if rows is None:
        return {"readable": False, "total": None,
                "modes": {}, "note": "the store could not be read; this is not a store with 0 memories"}
    tags = _tags_by_id(store)
    pins = _pins(store)
    described: dict[str, Any] = {}
    for name in MODES:
        if name == OFF_RECORD:
            described[name] = {"excluded": len(rows), "reasons": {OFF_RECORD: len(rows)}}
            continue
        excluded: dict[str, int] = {}
        for row in rows:
            memory_id = str(row.get("id") or "")
            reason = _exclusion(name, row, tags.get(memory_id, set()), pins)
            if reason:
                excluded[reason] = excluded.get(reason, 0) + 1
        described[name] = {"excluded": sum(excluded.values()), "reasons": dict(sorted(excluded.items()))}
    return {
        "readable": True,
        "total": len(rows),
        "modes": described,
        "note": ("exclusion is counted after the normal view, so a mode can only remove memories; "
                 "private counts explicit tags first and falls back to the consent keyword screen"),
    }


def describe() -> dict[str, Any]:
    """The modes as data, including the empty widening list a test can assert on."""
    return {
        "modes": [mode.as_dict() for mode in _MODE_TABLE.values()],
        "order": list(MODES),
        "widening_modes": [],
        "guarantees": [
            "a mode can only remove memories; the visible set is normal AND mode",
            "the mode is read from the caller's context, never inferred from content",
            "every output states the mode in force and how many memories it excluded",
            "off_record refuses a write with a reason instead of dropping it",
            "an unknown mode falls back to off_record, the most restrictive known mode",
            "reading a mode never creates schema or writes to the store",
        ],
        "limits": {"private_classes": list(PRIVATE_CLASSES)},
    }


# --------------------------------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------------------------------

@dataclass
class WriteDecision:
    allowed: bool
    mode: str
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "mode": self.mode, "reason": self.reason}


def screen_write(context: Any = None, *, content: str = "") -> WriteDecision:
    """Whether this caller may keep a memory. Refusals name the mode; nothing is stored and nothing is lost
    silently."""
    resolution = resolve(context)
    mode = resolution.mode
    if mode.writes:
        return WriteDecision(True, mode.name, f"{mode.name} mode stores memories")
    if mode.name == OFF_RECORD:
        return WriteDecision(
            False, mode.name,
            "off_record retains nothing: the write was refused, not dropped, so the caller can say the "
            "memory was not kept",
        )
    return WriteDecision(
        False, mode.name,
        f"{mode.name} mode accepts no new memories: nothing is safe to show until it is explicitly "
        f"pinned, and a write that would be invisible is a silent loss",
    )


# --------------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------------

def render(view: View, *, limit: int = 10) -> list[str]:
    """One screen: the mode and its count first, then a few memories. Never content-free and never
    count-free."""
    lines = [view.statement]
    for row in view.memories[: max(0, int(limit))]:
        text = " ".join(str(row.get("content") or "").split())[:80]
        lines.append(f"  · {row.get('kind', '?')} {str(row.get('id') or '')[:8]} {text}")
    if len(view.memories) > limit:
        lines.append(f"  … {len(view.memories) - limit} more not shown (limit {limit})")
    if not view.memories and view.readable:
        lines.append("  nothing to show in this mode")
    return lines


def render_census(store: Any) -> list[str]:
    described = census(store)
    if not described["readable"]:
        return ["privacy modes: " + str(described["note"])]
    total = described["total"]
    lines = [f"privacy modes over {total} memories:"]
    for name in MODES:
        entry = described["modes"][name]
        detail = ", ".join(f"{reason} {count}" for reason, count in entry["reasons"].items())
        lines.append(f"  · {name}: excludes {entry['excluded']} of {total}"
                     + (f" ({detail})" if detail else ""))
    return lines
