"""See exactly how the always-on block changed, and when (backlog item #124).

Core memory is the one part of the prompt that is present on *every* turn, so a silent change
to it changes every future answer — and until now the only way to notice was to remember what
it looked like yesterday. `core_history` (#123) records membership events; this records the
**rendered block**, which is the thing the model actually reads.

Two deliberate choices:

- **Snapshots are content-addressed.** A row is written only when the block's SHA-256 changes,
  so a quiet store writes nothing and a busy one writes one row per genuine change rather than
  one per turn.
- **The diff is a real diff.** Added and removed *lines*, not just ids: a memory whose content
  was edited shows up as a changed line, which an id-level comparison would report as
  unchanged.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass
from typing import Any

TABLE = "core_snapshots"
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    block_hash TEXT NOT NULL,
    chars INTEGER NOT NULL,
    memory_ids_json TEXT NOT NULL DEFAULT '[]',
    text TEXT NOT NULL
)
"""
INDEX = f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_at ON {TABLE}(at)"

# A year of daily changes is still tiny (one block is ~2 KB), but unbounded growth in a table
# nobody reads is how stores bloat. Keep the most recent N and drop the tail.
MAX_SNAPSHOTS = 500


@dataclass(frozen=True)
class Snapshot:
    at: str
    block_hash: str
    chars: int
    memory_ids: tuple[str, ...]
    text: str


def _ensure(store: Any) -> None:
    store._conn.execute(SCHEMA)
    store._conn.execute(INDEX)


def hash_block(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def latest(store: Any) -> Snapshot | None:
    _ensure(store)
    row = store._conn.execute(
        f"SELECT at, block_hash, chars, memory_ids_json, text FROM {TABLE} ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return _row_to_snapshot(row) if row else None


def _row_to_snapshot(row: Any) -> Snapshot:
    return Snapshot(
        at=str(row["at"]),
        block_hash=str(row["block_hash"]),
        chars=int(row["chars"]),
        memory_ids=tuple(json.loads(row["memory_ids_json"] or "[]")),
        text=str(row["text"]),
    )


def record(store: Any, *, at: str | None = None, block: Any = None) -> Snapshot | None:
    """Snapshot the current block if it differs from the last one. Returns the new row or None."""
    from .core_block import build
    from .store import utc_now

    rendered = block if block is not None else build(store)
    digest = hash_block(rendered.text)
    previous = latest(store)
    if previous is not None and previous.block_hash == digest:
        return None
    stamp = at or utc_now()
    _ensure(store)
    with store.transaction() as conn:
        conn.execute(
            f"INSERT INTO {TABLE}(at, block_hash, chars, memory_ids_json, text) VALUES(?,?,?,?,?)",
            (stamp, digest, rendered.chars, json.dumps(list(rendered.memory_ids)), rendered.text),
        )
        conn.execute(
            f"DELETE FROM {TABLE} WHERE id NOT IN "
            f"(SELECT id FROM {TABLE} ORDER BY id DESC LIMIT ?)",
            (MAX_SNAPSHOTS,),
        )
    return Snapshot(stamp, digest, rendered.chars, tuple(rendered.memory_ids), rendered.text)


def history(store: Any, *, limit: int = 20) -> list[Snapshot]:
    """Snapshots, newest first."""
    _ensure(store)
    rows = store._conn.execute(
        f"SELECT at, block_hash, chars, memory_ids_json, text FROM {TABLE} ORDER BY id DESC LIMIT ?",
        (max(1, int(limit)),),
    ).fetchall()
    return [_row_to_snapshot(row) for row in rows]


@dataclass(frozen=True)
class CoreDiff:
    before: Snapshot | None
    after: Snapshot | None
    added_lines: tuple[str, ...]
    removed_lines: tuple[str, ...]
    added_ids: tuple[str, ...]
    removed_ids: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return bool(self.added_lines or self.removed_lines)

    def summary(self) -> str:
        before, after = self.before, self.after
        if before is None and after is None:
            return "no snapshots recorded yet"
        if before is None and after is not None:
            return f"first snapshot at {after.at[:19]}: {len(after.memory_ids)} memories"
        if after is None and before is not None:
            return f"unchanged since {before.at[:19]}"
        assert before is not None and after is not None  # both present past this point
        if not self.changed:
            return f"unchanged since {after.at[:19]}"
        return (
            f"changed at {after.at[:19]}: +{len(self.added_lines)} lines, "
            f"-{len(self.removed_lines)} lines, {after.chars - before.chars:+d} chars"
        )


def diff(store: Any, *, older_index: int = 1) -> CoreDiff:
    """Compare the newest snapshot with the one before it (`older_index=2` for the one prior)."""
    snapshots = history(store, limit=max(2, older_index + 1))
    after = snapshots[0] if snapshots else None
    before = snapshots[older_index] if len(snapshots) > older_index else None
    if before is None or after is None:
        return CoreDiff(before, after, (), (), (), ())
    before_lines = before.text.splitlines()
    after_lines = after.text.splitlines()
    added: list[str] = []
    removed: list[str] = []
    for line in difflib.unified_diff(before_lines, after_lines, lineterm="", n=0):
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            added.append(line[1:])
        elif line.startswith("-"):
            removed.append(line[1:])
    before_ids = set(before.memory_ids)
    after_ids = set(after.memory_ids)
    return CoreDiff(
        before=before,
        after=after,
        added_lines=tuple(added),
        removed_lines=tuple(removed),
        added_ids=tuple(sorted(after_ids - before_ids)),
        removed_ids=tuple(sorted(before_ids - after_ids)),
    )
