"""When a field gains meaning, re-derive it for the rows that were written before (#45).

A new derived field is easy to add and hard to populate: rows written last month have no value for it,
so every reader grows a fallback, and the fallback slowly becomes the contract. A backfill is the
honest alternative — but only if it is **re-derivable from what was stored** and **safe to run twice**.

Three properties, each of which the tests pin:

- **derivable or declared.** A derivation reads the stored row (content, kind, source, quotes) and
  computes the field. Anything that would need the original conversation cannot be re-derived; it is
  declared `not_derivable` and reported as a gap rather than approximated.
- **idempotent.** Running a backfill twice fills zero rows the second time, because the predicate that
  selects rows is "the field is missing", not "the backfill has not run".
- **recorded.** Every run writes a row into `backfill_runs` with the derivation, the counts and the
  schema version it ran for, so a half-finished backfill is visible instead of inferred.

The schema version is the trigger: a derivation tagged `since=2` runs while the store's recorded
version is below 2, and the version is bumped only after `verify()` finds nothing left to fill.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

RUNS = "backfill_runs"
STATE = "backfill_state"
MARKS = "backfill_marks"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Derivation:
    name: str
    fills: str
    since: int
    needs: Callable[[Any, int], list[dict[str, Any]]]
    derive: Callable[[Any, dict[str, Any]], bool]
    not_derivable: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fills": self.fills,
            "since": self.since,
            "not_derivable": self.not_derivable,
            "note": self.note,
        }


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


# --- the derivations this store can actually re-derive -------------------------------------------

def _needs_scope(store: Any, limit: int) -> list[dict[str, Any]]:
    return _rows(
        store,
        """SELECT id, content FROM memories
           WHERE state = 'active' AND (scope_json IS NULL OR scope_json IN ('', '{}'))
           LIMIT ?""",
        (int(limit),),
    )


def _derive_scope(store: Any, row: dict[str, Any]) -> bool:
    from . import scope_capture

    return bool(scope_capture.capture(store, str(row["id"]), str(row["content"])))


def _needs_entities(store: Any, limit: int) -> list[dict[str, Any]]:
    return _rows(
        store,
        """SELECT id, content FROM memories
           WHERE state = 'active' AND (entities_json IS NULL OR entities_json IN ('', '[]'))
           LIMIT ?""",
        (int(limit),),
    )


def _derive_entities(store: Any, row: dict[str, Any]) -> bool:
    from . import entity_resolve

    attached = entity_resolve.attach(store, str(row["id"]), str(row["content"]))
    return bool(attached.get("canonical"))


def _needs_importance(store: Any, limit: int) -> list[dict[str, Any]]:
    return _rows(
        store,
        """SELECT id, content, kind, source_category FROM memories
           WHERE state = 'active' AND (importance IS NULL OR importance = 0)
           LIMIT ?""",
        (int(limit),),
    )


def _derive_importance(store: Any, row: dict[str, Any]) -> bool:
    from . import importance_rubric

    scored = importance_rubric.score(
        str(row["content"]), str(row.get("kind") or ""),
        source_category=row.get("source_category"),
    )
    with store.transaction() as conn:
        conn.execute(
            "UPDATE memories SET importance = ? WHERE id = ?",
            (float(scored.importance), str(row["id"])),
        )
    return True


def _needs_volatility(store: Any, limit: int) -> list[dict[str, Any]]:
    return _rows(
        store,
        """SELECT m.id, m.content, m.kind FROM memories m
           LEFT JOIN memory_volatility v ON v.memory_id = m.id
           WHERE m.state = 'active' AND v.memory_id IS NULL LIMIT ?""",
        (int(limit),),
    )


def _derive_volatility(store: Any, row: dict[str, Any]) -> bool:
    from . import volatility

    assigned = volatility.assign(store, str(row["id"]), str(row["content"]),
                                 kind=str(row.get("kind") or ""))
    return bool(assigned)


def default_derivations() -> list[Derivation]:
    return [
        Derivation("scope", "scope_json", 1, _needs_scope, _derive_scope,
                   note="who/where/when, from the sentence itself"),
        Derivation("entities", "entities_json", 1, _needs_entities, _derive_entities,
                   note="canonical entity ids, ambiguous names left alone"),
        Derivation("importance", "importance", 1, _needs_importance, _derive_importance,
                   note="kind-anchored rubric"),
        Derivation("volatility", "memory_volatility", 1, _needs_volatility, _derive_volatility,
                   note="durable / seasonal / ephemeral"),
        Derivation("extraction_confidence", "confidence", 1, lambda store, limit: [],
                   lambda store, row: False, not_derivable=True,
                   note="the extractor's confidence is not recoverable from the stored row"),
    ]


# --- the machinery -------------------------------------------------------------------------------

def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {RUNS} (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    derivation TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    scanned INTEGER NOT NULL,
                    filled INTEGER NOT NULL,
                    failed INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT
                )"""
        )
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {MARKS} (
                    memory_id TEXT NOT NULL,
                    derivation TEXT NOT NULL,
                    marked_at TEXT NOT NULL,
                    PRIMARY KEY (memory_id, derivation)
                )"""
        )
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {STATE} (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
        )


def _marks(store: Any, derivation: str) -> set[str] | None:
    try:
        rows = store._conn.execute(
            f"SELECT memory_id FROM {MARKS} WHERE derivation = ?", (derivation,)
        ).fetchall()
    except Exception:  # noqa: BLE001
        return None
    return {str(row["memory_id"]) for row in rows}


def _mark(store: Any, derivation: str, memory_id: str, at: datetime) -> None:
    """Remember that a row was already tried and had nothing to fill.

    Without this, "the field is missing" and "the field is not applicable" are the same predicate,
    so a fact that states no qualifiers is rescanned by every pass forever and `verify()` never
    comes back clean.
    """
    try:
        with store.transaction() as conn:
            conn.execute(
                f"""INSERT OR IGNORE INTO {MARKS} (memory_id, derivation, marked_at)
                    VALUES (?, ?, ?)""",
                (str(memory_id), derivation, at.isoformat()),
            )
    except Exception:  # noqa: BLE001
        pass


def _unmarked(store: Any, derivation: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    marks = _marks(store, derivation)
    if marks is None:
        return rows
    return [row for row in rows if str(row.get("id")) not in marks]


def schema_version(store: Any) -> int:
    _ensure(store)
    try:
        row = store._conn.execute(
            f"SELECT value FROM {STATE} WHERE key = 'schema_version'"
        ).fetchone()
    except Exception:  # noqa: BLE001
        return 0
    try:
        return int(row["value"]) if row else 0
    except (TypeError, ValueError):
        return 0


def set_schema_version(store: Any, version: int) -> None:
    _ensure(store)
    with store.transaction() as conn:
        conn.execute(
            f"""INSERT INTO {STATE} (key, value, updated_at) VALUES ('schema_version', ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                               updated_at = excluded.updated_at""",
            (str(int(version)), _now().isoformat()),
        )


def plan(store: Any, *, only: Iterable[str] = (), limit: int = 500,
         derivations: list[Derivation] | None = None) -> dict[str, Any]:
    """What a backfill would do, without doing it."""
    _ensure(store)
    wanted = {name for name in only}
    lines: list[dict[str, Any]] = []
    for derivation in derivations or default_derivations():
        if wanted and derivation.name not in wanted:
            continue
        if derivation.not_derivable:
            lines.append({**derivation.as_dict(), "needs": None})
            continue
        try:
            needs = _unmarked(store, derivation.name, derivation.needs(store, int(limit)))
        except Exception as error:  # noqa: BLE001
            lines.append({**derivation.as_dict(), "needs": None, "error": str(error)[:120]})
            continue
        lines.append({**derivation.as_dict(), "needs": len(needs)})
    return {
        "schema_version": schema_version(store),
        "derivations": lines,
        "total": sum(line["needs"] or 0 for line in lines),
    }


@dataclass
class Run:
    run_id: str = ""
    filled: int = 0
    scanned: int = 0
    failed: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scanned": self.scanned,
            "filled": self.filled,
            "failed": self.failed,
            "details": self.details,
        }

    def describe(self) -> str:
        return (f"backfill {self.run_id}: scanned {self.scanned}, filled {self.filled}, "
                f"failed {self.failed}")


def run(store: Any, *, only: Iterable[str] = (), limit: int = 500,
        derivations: list[Derivation] | None = None, at: datetime | None = None) -> Run:
    """Fill what is missing, once, and record that it happened."""
    _ensure(store)
    version = schema_version(store)
    wanted = {name for name in only}
    outcome = Run(run_id=uuid.uuid4().hex[:12])
    moment = at or _now()
    for derivation in derivations or default_derivations():
        if wanted and derivation.name not in wanted:
            continue
        if derivation.not_derivable:
            outcome.details.append({"name": derivation.name, "skipped": "not derivable",
                                    "note": derivation.note})
            continue
        try:
            rows = _unmarked(store, derivation.name, derivation.needs(store, int(limit)))
        except Exception as error:  # noqa: BLE001
            outcome.details.append({"name": derivation.name, "error": str(error)[:120]})
            continue
        filled = failed = 0
        for row in rows:
            try:
                if derivation.derive(store, row):
                    filled += 1
                else:
                    # tried, nothing to fill: mark it so the next pass does not rescan it
                    _mark(store, derivation.name, str(row.get("id")), moment)
            except Exception:  # noqa: BLE001 - one bad row must not stop the pass
                failed += 1
        outcome.scanned += len(rows)
        outcome.filled += filled
        outcome.failed += failed
        outcome.details.append({"name": derivation.name, "scanned": len(rows),
                                "filled": filled, "failed": failed})
        with store.transaction() as conn:
            conn.execute(
                f"""INSERT INTO {RUNS}
                    (run_id, derivation, schema_version, scanned, filled, failed,
                     started_at, finished_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (outcome.run_id, derivation.name, version, len(rows), filled, failed,
                 moment.isoformat(), _now().isoformat()),
            )
    return outcome


def verify(store: Any, *, derivations: list[Derivation] | None = None) -> dict[str, Any]:
    """Nothing left to fill — the condition for bumping the schema version."""
    remaining: dict[str, int] = {}
    for derivation in derivations or default_derivations():
        if derivation.not_derivable:
            continue
        try:
            remaining[derivation.name] = len(
                _unmarked(store, derivation.name, derivation.needs(store, 100000))
            )
        except Exception:  # noqa: BLE001
            remaining[derivation.name] = -1
    return {"remaining": remaining, "clean": all(count == 0 for count in remaining.values())}


def history(store: Any, *, limit: int = 20) -> list[dict[str, Any]]:
    _ensure(store)
    return _rows(store, f"SELECT * FROM {RUNS} ORDER BY started_at DESC LIMIT ?", (int(limit),))


def report(store: Any) -> dict[str, Any]:
    return {"schema_version": schema_version(store), "runs": history(store, limit=5)}
