"""Take the vault header out of the content it was pasted into (#223)

Every vault-imported memory in this store begins the same way:

```
Vault note: AdGuard Home Path: homelab/AdGuard Home.md Section: AdGuard Home › Notes - Router DHCP DNS points clients to…
```

Measured 2026-11-24: **3,016 of 3,817 active memories (79%)** carry that prefix, so what retrieval hands the
model is mostly a template with a line of substance behind it. It poisons everything that counts words:
labels (#156), citation phrases (#152), receipts (#174) and any future embedding of the text.

The fix has to be cautious, because the header is not uniformly shaped. Measured on the same store:

```
rows with the prefix          3016
rows with a " - " delimiter   1415   (section ends here, body follows)
rows without one              1601   (section runs straight into the body)
```

So there are two mechanical modes, and they are **not** equally trustworthy:

- **`delimited`** — `Section: <heading> - <body>`. The boundary is in the text, so `body` can be extracted
  exactly. This is the default and the only mode that rewrites content.
- **`prefix`** — only the deterministic prefix (`Vault note:`, `Path:`, the `Section:` label) is removed,
  and whatever followed it stays. Safe for every row, but it leaves the heading in place, because
  guessing where a heading ends inside free text is how a tool mangles a note.

Rules:

- **the original is kept**, verbatim, so a bad strip is reversible without a backup;
- **nothing is ever blanked.** A note whose body would be empty keeps its title as the content;
- **a dry run is the default.** `strip()` reports what it *would* do until told otherwise;
- **it is idempotent**, because a re-run must be a no-op rather than a second strip of a strip.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

TABLE = "vault_headers"
DELIMITED = "delimited"
PREFIX = "prefix"

_HEADER = re.compile(
    r"^\s*Vault note:\s*(?P<title>.*?)\s+Path:\s*(?P<path>.*?)\s+Section:\s*(?P<rest>.*)$",
    re.DOTALL,
)
_SECTION_LABEL = re.compile(r"^\s*Section:\s*", re.IGNORECASE)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    memory_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '',
                    path TEXT NOT NULL DEFAULT '',
                    section TEXT NOT NULL DEFAULT '',
                    mode TEXT NOT NULL DEFAULT '',
                    stripped_at TEXT NOT NULL,
                    original TEXT NOT NULL
                )"""
        )


@dataclass
class Parsed:
    title: str = ""
    path: str = ""
    section: str = ""
    body: str = ""
    mode: str = ""
    matched: bool = False

    @property
    def changed(self) -> bool:
        return self.matched and bool(self.body)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title, "path": self.path, "section": self.section,
            "body": self.body, "mode": self.mode, "matched": self.matched,
        }


def parse(content: Any, *, mode: str = DELIMITED) -> Parsed:
    """Split a vault header out of `content`, or report that it does not match."""
    text = str(content or "")
    match = _HEADER.match(text)
    if not match:
        return Parsed()
    title = " ".join(match.group("title").split())
    path = match.group("path").strip()
    rest = match.group("rest").strip()
    found = Parsed(title=title, path=path, matched=True)
    if mode == PREFIX:
        found.section, found.body = rest, rest
        found.mode = PREFIX
        return found
    if " - " in rest:
        section, body = rest.split(" - ", 1)
        found.section, found.body = section.strip(), body.strip()
        found.mode = DELIMITED
        return found
    if rest.rstrip().endswith("-"):
        # A note with a heading and no body: the delimiter is there, the text is not. Delimited with an
        # empty body, which callers may resolve to the title rather than leaving a note blank.
        found.section, found.body = rest.rstrip()[:-1].strip(), ""
        found.mode = DELIMITED
        return found
    # No delimiter: the section runs into the body. Do not guess where it ends — leave the row for the
    # caller to decide, and say so by returning an empty body with the raw text in `section`.
    found.section, found.body, found.mode = rest, "", ""
    return found


def candidates(store: Any, *, scan: int = 5000) -> list[dict[str, Any]]:
    return _rows(
        store,
        "SELECT id, content FROM memories WHERE state = 'active' "
        "AND content LIKE 'Vault note:%' ORDER BY created_at DESC LIMIT ?",
        (int(scan),),
    )


@dataclass
class Sweep:
    examined: int = 0
    delimited: int = 0
    ambiguous: int = 0
    rewritten: int = 0
    already: int = 0
    title_only: int = 0
    saved_chars: int = 0
    applied: bool = False
    samples: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "examined": self.examined,
            "delimited": self.delimited,
            "ambiguous": self.ambiguous,
            "rewritten": self.rewritten,
            "already": self.already,
            "title_only": self.title_only,
            "saved_chars": self.saved_chars,
            "applied": self.applied,
            "samples": self.samples,
            "sentence": self.describe(),
        }

    def describe(self) -> str:
        verb = "rewrote" if self.applied else "would rewrite"
        tail = (f"; {self.already} already stripped" if self.already else "")
        tail += (f"; {self.ambiguous} ambiguous and left alone" if self.ambiguous else "")
        return (f"{self.examined} vault memory(ies): {verb} {self.rewritten} "
                f"(saving {self.saved_chars} char(s)){tail}")


def strip(store: Any, *, mode: str = DELIMITED, apply: bool = False,
          scan: int = 5000, at: datetime | None = None) -> Sweep:
    """Remove the vault header from content, keeping the original. Dry run by default."""
    found = Sweep(applied=bool(apply))
    if apply:
        _ensure(store)
    moment = (at or _now()).isoformat()
    for row in candidates(store, scan=scan):
        found.examined += 1
        memory_id = str(row["id"])
        original = str(row["content"] or "")
        parsed = parse(original, mode=mode)
        if not parsed.matched:
            continue
        if parsed.mode == DELIMITED:
            found.delimited += 1
        elif not parsed.body:
            found.ambiguous += 1
            continue
        body = parsed.body or parsed.title
        if not body or body == original:
            found.already += 1
            continue
        if not parsed.body:
            found.title_only += 1
        found.rewritten += 1
        found.saved_chars += max(0, len(original) - len(body))
        if len(found.samples) < 3:
            found.samples.append({
                "memory_id": memory_id,
                "before": original[:90],
                "after": body[:90],
                "mode": parsed.mode,
            })
        if not apply:
            continue
        with store.transaction() as conn:
            conn.execute(
                "UPDATE memories SET content = ?, updated_at = ? WHERE id = ?",
                (body, moment, memory_id),
            )
            conn.execute(
                f"""INSERT OR REPLACE INTO {TABLE}
                    (memory_id, title, path, section, mode, stripped_at, original)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (memory_id, parsed.title, parsed.path, parsed.section, parsed.mode, moment,
                 original),
            )
    return found


def header_of(store: Any, memory_id: str) -> dict[str, Any] | None:
    _ensure(store)
    rows = _rows(store, f"SELECT * FROM {TABLE} WHERE memory_id = ?", (str(memory_id),))
    return rows[0] if rows else None


def restore(store: Any, memory_id: str, *, at: datetime | None = None) -> bool:
    """Put the original text back — a strip has to be reversible without a backup."""
    stored = header_of(store, str(memory_id))
    if not stored:
        return False
    moment = (at or _now()).isoformat()
    with store.transaction() as conn:
        conn.execute("UPDATE memories SET content = ?, updated_at = ? WHERE id = ?",
                     (str(stored["original"] or ""), moment, str(memory_id)))
    return True


def report(store: Any, *, scan: int = 5000) -> dict[str, Any]:
    rows = candidates(store, scan=scan)
    total = _rows(store, "SELECT COUNT(*) AS count FROM memories WHERE state = 'active'")
    active = int(total[0]["count"] or 0) if total else 0
    delimited = ambiguous = 0
    for row in rows:
        parsed = parse(row["content"])
        if not parsed.matched:
            continue
        if parsed.mode == DELIMITED:
            delimited += 1
        elif not parsed.body:
            ambiguous += 1
    return {
        "active": active,
        "with_header": len(rows),
        "share": round(len(rows) / active, 4) if active else 0.0,
        "delimited": delimited,
        "ambiguous": ambiguous,
        "note": "an ambiguous header is left alone: guessing where a heading ends mangles a note",
    }
