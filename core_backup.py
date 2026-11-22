"""Export, inspect and restore core memory (backlog item #131).

The always-on block is the part of the system whose loss is hardest to notice: everything
still runs, the model just stops knowing who it is talking to. So it gets a backup that is
readable without Cortex — a Markdown file a person can read, diff and keep in a vault — with
a machine-readable block at the end so it can be restored.

Two decisions:

- **Human first.** The body is the rendered block, grouped by section, with ids and verification
  dates. If Cortex is gone entirely, that file is still the profile.
- **Restore matches content, not ids.** A restore into a different database cannot expect the
  same UUIDs. Memories are matched by normalised content (and by id when it exists), so a
  backup taken from one store can rebuild the profile in another — which is the point of a
  backup, and something an id-only restore cannot do.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

FORMAT_VERSION = 1
FRONT = "# Core memory backup"
_RECORD = re.compile(r"^- \[(?P<id>[^\]]+)\] (?P<content>.*?)(?: \[(?P<tags>[^\]]*)\])?$")
_JSON_BLOCK = re.compile(r"```json\n(?P<body>.*?)\n```", re.S)


def _normalise(text: str) -> str:
    return " ".join(str(text or "").lower().split())


@dataclass
class Backup:
    text: str
    entries: list[dict[str, Any]] = field(default_factory=list)


def collect(store: Any) -> list[dict[str, Any]]:
    """Everything in the profile: pinned memories and promoted members, with their state."""
    rows: dict[str, dict[str, Any]] = {}
    queries = (
        """SELECT m.id, m.content, m.kind, m.pinned, m.importance,
                  COALESCE(m.last_used_at, m.last_helpful_at) AS verified_at,
                  'pin' AS source
           FROM memories m WHERE m.state = 'active' AND m.pinned = 1""",
        """SELECT m.id, m.content, m.kind, m.pinned, m.importance,
                  COALESCE(c.last_confirmed_at, m.last_used_at, m.last_helpful_at) AS verified_at,
                  COALESCE(c.promoted_by, 'use') AS source
           FROM memories m JOIN core_members c ON c.memory_id = m.id
           WHERE m.state = 'active'""",
    )
    for query in queries:
        try:
            for row in store._conn.execute(query).fetchall():
                rows.setdefault(str(row["id"]), dict(row))
        except Exception:  # noqa: BLE001 - core_members is optional
            continue
    return sorted(rows.values(), key=lambda r: (-float(r.get("importance") or 0), str(r["id"])))


def export(store: Any, *, max_chars: int = 20000) -> Backup:
    """Render the profile as Markdown with a restorable JSON block."""
    entries = collect(store)
    lines = [
        FRONT,
        "",
        "Human-readable, and restorable: the JSON block at the end is what `load` reads.",
        f"format_version: {FORMAT_VERSION}",
        "",
    ]
    size = 0
    written: list[dict[str, Any]] = []
    for row in entries:
        content = " ".join(str(row.get("content") or "").split())
        verified = str(row.get("verified_at") or "")[:10]
        tags = f"{row.get('source') or 'core'}{', verified ' + verified if verified else ''}"
        line = f"- [{str(row['id'])[:8]}] {content} [{tags}]"
        if max_chars > 0 and size + len(line) > max_chars:
            break
        lines.append(line)
        written.append(row)
        size += len(line) + 1
    # A backup that truncates silently is not a backup: report what was written and what was
    # left out, and keep the JSON consistent with the body.
    lines.insert(3, f"entries: {len(written)} of {len(entries)}")
    if len(written) < len(entries):
        lines.append(
            f"_truncated: {len(entries) - len(written)} entries omitted at "
            f"max_chars={max_chars}; re-run with a larger cap_"
        )
        lines.append("")
    payload = {
        "format_version": FORMAT_VERSION,
        "entries": [
            {
                "id": str(row["id"]),
                "content": " ".join(str(row.get("content") or "").split()),
                "kind": str(row.get("kind") or "semantic"),
                "pinned": bool(row.get("pinned")),
                "source": str(row.get("source") or "core"),
                "verified_at": str(row.get("verified_at") or ""),
            }
            for row in written
        ],
    }
    lines.extend(["", "```json", json.dumps(payload, indent=2, sort_keys=True), "```", ""])
    return Backup(text="\n".join(lines), entries=written)


@dataclass
class RestoreResult:
    restored: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return (
            f"restored {len(self.restored)}, missing {len(self.missing)}, "
            f"skipped {len(self.skipped)}"
        )


def parse(text: str) -> list[dict[str, Any]]:
    """Read a backup: the JSON block when present, otherwise the Markdown lines."""
    match = _JSON_BLOCK.search(text or "")
    if match:
        try:
            payload = json.loads(match.group("body"))
            entries = payload.get("entries")
            if isinstance(entries, list):
                return [entry for entry in entries if isinstance(entry, dict)]
        except json.JSONDecodeError:
            pass
    entries = []
    for line in (text or "").splitlines():
        found = _RECORD.match(line.strip())
        if found:
            entries.append(
                {"id": found.group("id"), "content": found.group("content"),
                 "tags": found.group("tags") or ""}
            )
    return entries


def restore(store: Any, text: str) -> RestoreResult:
    """Rebuild the profile from a backup, matching memories by id first and content second."""
    from .promotion import promote

    result = RestoreResult()
    by_content: dict[str, str] = {}
    for row in store._conn.execute(
        "SELECT id, content FROM memories WHERE state = 'active'"
    ).fetchall():
        by_content.setdefault(_normalise(row["content"]), str(row["id"]))

    for entry in parse(text):
        content = _normalise(entry.get("content") or "")
        memory_id = str(entry.get("id") or "")
        existing = store.get_memory(memory_id) if memory_id and len(memory_id) > 8 else None
        if existing is None:
            memory_id = by_content.get(content, "")
        if not memory_id:
            result.missing.append(str(entry.get("content") or "")[:60])
            continue
        if entry.get("pinned"):
            store.set_pinned(memory_id, True)
        try:
            promote(store, memory_id, by=str(entry.get("source") or "restore"),
                    reason="restored from backup", check_conflicts=False)
            result.restored.append(memory_id)
        except Exception:  # noqa: BLE001 - a single bad entry must not abort a restore
            result.skipped.append(memory_id)
    return result
