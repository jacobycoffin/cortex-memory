"""Resolve an entity's names to one identity before the fact is stored (#38).

The same machine is "the box", "CT116", "the Proxmox box" and "ct116.lan" depending on the sentence,
and a store that keeps all four as separate entities can never answer a question about any of them:
each name holds a fragment of the history, and every query sees one fragment.

Resolution happens **at write**, in three steps, and it fails closed:

1. **mention** — proper nouns, host-shaped tokens (`CT116`, `r630`) and descriptor phrases
   ("the Proxmox box");
2. **canonicalise** — the mention is looked up in the alias table. A descriptor resolves through its
   qualifier ("proxmox box" → `proxmox`), never through the head noun alone ("box" means nothing);
3. **learn** — a mention that resolved confidently is recorded as an alias, so the next sentence is
   free.

A mention with **two or more candidate entities is left unresolved** and reported. Guessing would
merge two machines into one identity, and an identity error is the one kind of error a memory store
cannot notice later: every future fact about either host would land on the survivor.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# Host-shaped: contains a digit and is not a bare number ("CT116", "r630", "ct116.lan").
HOST = re.compile(r"\b(?=[A-Za-z0-9._-]*\d)[A-Za-z][A-Za-z0-9._-]{1,30}\b")
# A capitalised word that is not the first word of the sentence.
PROPER = re.compile(r"(?<=[a-z0-9,;:]\s)([A-Z][A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*)")
# "the Proxmox box", "the r630 host", "the Unraid server".
DESCRIPTOR = re.compile(
    r"\bthe\s+([A-Za-z0-9][A-Za-z0-9._-]*)\s+(?:box|host|machine|server|nas|vps|container|lxc)\b",
    re.IGNORECASE,
)
# Words that look like names but are not entities in this store.
STOP_NAMES = frozenset(
    "the a an and or but if then this that these those i you he she it we they is are was were "
    "note root cause fix error update status ok yes no monday tuesday wednesday thursday friday "
    "saturday sunday january february march april may june july august september october november "
    "december".split()
)
MIN_LEN = 3


@dataclass
class Mention:
    text: str
    kind: str
    entity_id: str = ""
    learned: bool = False
    ambiguous: tuple[str, ...] = ()

    @property
    def resolved(self) -> bool:
        return bool(self.entity_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.text,
            "kind": self.kind,
            "entity_id": self.entity_id or None,
            "learned": self.learned,
            "ambiguous": list(self.ambiguous),
        }


@dataclass
class Resolution:
    mentions: list[Mention] = field(default_factory=list)

    @property
    def canonical(self) -> list[str]:
        seen: list[str] = []
        for mention in self.mentions:
            if mention.entity_id and mention.entity_id not in seen:
                seen.append(mention.entity_id)
        return seen

    @property
    def unresolved(self) -> list[str]:
        return [m.text for m in self.mentions if not m.resolved]

    def as_dict(self) -> dict[str, Any]:
        return {
            "canonical": self.canonical,
            "unresolved": self.unresolved,
            "mentions": [m.as_dict() for m in self.mentions],
        }


def mentions(text: str) -> list[tuple[str, str]]:
    """Names worth resolving, in the order they appear, with how each was found."""
    body = str(text or "")
    found: list[tuple[str, str]] = []

    def add(name: str, kind: str) -> None:
        cleaned = name.strip(" .,;:()[]{}\"'")
        if len(cleaned) < MIN_LEN or cleaned.lower() in STOP_NAMES:
            return
        if cleaned.lower() not in [existing.lower() for existing, _ in found]:
            found.append((cleaned, kind))

    for match in DESCRIPTOR.finditer(body):
        add(match.group(1), "descriptor")
    for match in HOST.finditer(body):
        if re.fullmatch(r"\d+", match.group(0)):
            continue
        add(match.group(0), "host")
    for match in PROPER.finditer(body):
        add(match.group(1), "proper")
    return found


def _candidates(store: Any, name: str) -> list[str]:
    """Every canonical entity this name could mean."""
    try:
        from . import aliases

        alias_key = aliases.key(name)
        rows = store._conn.execute(
            f"SELECT DISTINCT entity_id FROM {aliases.TABLE} WHERE alias_key = ?", (alias_key,)
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    return [str(row["entity_id"]) for row in rows]


def canonicalise(store: Any, name: str, *, kind: str = "proper") -> Mention:
    """One name to one entity, or to nothing with the reason attached."""
    candidates = _candidates(store, name)
    if not candidates and "." in name:
        # A hostname's first label is the host: "ct116.lan" is ct116 with a domain attached.
        candidates = _candidates(store, name.split(".", 1)[0])
    if len(candidates) == 1:
        return Mention(name, kind, entity_id=candidates[0])
    if len(candidates) > 1:
        return Mention(name, kind, ambiguous=tuple(sorted(candidates)))
    return Mention(name, kind)


def resolve(store: Any, text: str, *, learn: bool = True) -> Resolution:
    """Resolve every mention in the text, learning the ones that resolved confidently."""
    outcome = Resolution()
    try:
        from . import aliases
    except Exception:  # noqa: BLE001
        aliases = None  # type: ignore[assignment]
    for name, kind in mentions(text):
        mention = canonicalise(store, name, kind=kind)
        if mention.resolved and learn and aliases is not None:
            try:
                aliases.learn(store, mention.entity_id, [name], source="write")
                mention.learned = True
            except Exception:  # noqa: BLE001
                mention.learned = False
        outcome.mentions.append(mention)
    return outcome


def attach(store: Any, memory_id: str, text: str | None = None, *,
           learn: bool = True) -> dict[str, Any]:
    """Write the resolved entities onto a memory, as a union with what it already has."""
    content = text
    if content is None:
        try:
            row = store._conn.execute(
                "SELECT content, entities_json FROM memories WHERE id = ?", (str(memory_id),)
            ).fetchone()
        except Exception:  # noqa: BLE001
            return {"attached": [], "unresolved": []}
        content = str(row["content"]) if row else ""
    else:
        row = None
    resolution = resolve(store, str(content or ""), learn=learn)
    if not resolution.canonical:
        return resolution.as_dict()
    existing: list[str] = []
    try:
        raw = store._conn.execute(
            "SELECT entities_json FROM memories WHERE id = ?", (str(memory_id),)
        ).fetchone()
        if raw and raw["entities_json"]:
            loaded = json.loads(raw["entities_json"])
            if isinstance(loaded, list):
                existing = [str(item) for item in loaded]
    except Exception:  # noqa: BLE001
        existing = []
    merged = existing + [item for item in resolution.canonical if item not in existing]
    try:
        with store.transaction() as conn:
            conn.execute(
                "UPDATE memories SET entities_json = ? WHERE id = ?",
                (json.dumps(merged), str(memory_id)),
            )
    except Exception:  # noqa: BLE001 - resolution is an improvement, never a reason to fail a write
        return resolution.as_dict()
    return resolution.as_dict()


def coverage(store: Any, *, limit: int = 2000) -> dict[str, Any]:
    """How much of the store carries a resolved entity."""
    try:
        rows = store._conn.execute(
            """SELECT entities_json FROM memories
               WHERE state NOT IN ('tombstoned','deleted') LIMIT ?""",
            (int(limit),),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return {"memories": 0, "with_entities": 0, "empty": 0}
    with_entities = 0
    for row in rows:
        try:
            loaded = json.loads(row["entities_json"] or "[]")
        except (TypeError, ValueError):
            loaded = []
        if isinstance(loaded, list) and loaded:
            with_entities += 1
    return {
        "memories": len(rows),
        "with_entities": with_entities,
        "empty": len(rows) - with_entities,
    }


def report(store: Any) -> dict[str, Any]:
    try:
        from . import aliases

        known = aliases.report(store)
    except Exception:  # noqa: BLE001
        known = {}
    return {**coverage(store), "aliases": known}


def audit(store: Any, samples: Iterable[str]) -> dict[str, Any]:
    """Run resolution over a corpus and say how it did — the measurement that gates the feature."""
    totals: dict[str, Any] = {"texts": 0, "mentions": 0, "resolved": 0, "ambiguous": 0,
                              "unknown": 0}
    for text in samples:
        totals["texts"] += 1
        resolution = resolve(store, str(text), learn=False)
        for mention in resolution.mentions:
            totals["mentions"] += 1
            if mention.resolved:
                totals["resolved"] += 1
            elif mention.ambiguous:
                totals["ambiguous"] += 1
            else:
                totals["unknown"] += 1
    if totals["mentions"]:
        totals["resolved_share"] = round(totals["resolved"] / totals["mentions"], 4)
    return totals
