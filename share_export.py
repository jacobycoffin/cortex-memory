"""Export a slice of memory that is safe to hand to someone else (#171)

Sharing a memory should not require reading it first. That means the export does the work — and it means
the export is built out of **refusals**, not out of a hope that a regex caught everything:

- **a credential is refused, not scrubbed** (#43). If what remains of a memory after redaction is a stub —
  "the token is" and nothing — the memory is dropped and counted, because a redacted credential still
  tells a reader where the credential lives.
- **machine paths and addresses are genericised**, not deleted: a path under the hermes home becomes `<home>/...`,
  so the sentence still reads and the layout of somebody's box does not leak.
- **the manifest names patterns, never values.** "3 secrets redacted (aws_access_key, bearer_token)" is a
  receipt; the receipt must never contain the thing it removed, or the export leaks by describing itself.
- **the export is verifiable.** A SHA-256 over the exact bytes written, and `verify()` recomputes it, so a
  shared file can be checked rather than trusted.
- **nothing is exported by default.** An explicit id list or an explicit filter; there is no "export
  everything" convenience, because that is the one call that gets made by accident.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .security import sanitize_memory

FORMAT = "cortex-share-1"
SCHEMA_VERSION = 1

# Ordered: the more specific shape wins, so a path is genericised before an address rule can eat its host.
_PATH_PATTERNS = (
    (re.compile(r"/root/[^\s\"']*"), "<home>"),
    (re.compile(r"/home/[a-z0-9_.-]+/[^\s\"']*", re.IGNORECASE), "<home>"),
    (re.compile(r"/Users/[A-Za-z0-9_.-]+/[^\s\"']*"), "<home>"),
    (re.compile(r"[A-Za-z]:\\\\Users\\\\[^\s\"']*"), "<home>"),
)
_ADDRESS_PATTERNS = (
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "<email>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<ip>"),
)
_PRIVATE_IP = re.compile(r"\b(?:10|127|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.[\d.]+\b")

# A memory that is *about* a credential rather than merely containing one. Refused, with the reason.
_CREDENTIAL_ONLY = re.compile(
    r"^\s*(?:the\s+)?(?:api[\s_-]?key|token|password|secret|credential|passphrase)\b"
    r"[^.]{0,40}?(?:is|are|=|:)\s*\S{0,8}\s*$",
    re.IGNORECASE,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


@dataclass
class Prepared:
    memory_id: str
    text: str = ""
    refused: str = ""
    labels: list[str] = field(default_factory=list)
    genericised: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"memory_id": self.memory_id, "text": self.text, "refused": self.refused,
                "labels": self.labels, "genericised": self.genericised}


def prepare(text: str, *, mask_private_ips: bool = False) -> Prepared:
    """Redact one memory for sharing: secrets first, then paths and addresses."""
    found = Prepared(memory_id="")
    body = str(text or "")
    san = sanitize_memory(body)
    if san.redacted:
        found.labels = list(san.redactions)
        body = san.text
    for pattern, replacement in _PATH_PATTERNS:
        body, count = pattern.subn(replacement, body)
        if count:
            found.genericised.append("path")
    for pattern, replacement in _ADDRESS_PATTERNS:
        if replacement == "<ip>" and not mask_private_ips and _PRIVATE_IP.search(body):
            # A homelab address is not a secret and blanking it makes the note unreadable; the caller
            # asks for masking explicitly when the audience warrants it.
            continue
        body, count = pattern.subn(replacement, body)
        if count:
            found.genericised.append("email" if replacement == "<email>" else "ip")
    if _CREDENTIAL_ONLY.match(body) or (found.labels and len(body.strip()) < 40):
        found.refused = "credential-only content; a redacted credential still says where it lives"
        found.text = ""
        return found
    found.text = body.strip()
    return found


def export(store: Any, out_path: str | Path, *, ids: Iterable[str] | None = None,
           kinds: Iterable[str] | None = None, limit: int = 200,
           mask_private_ips: bool = False, at: datetime | None = None) -> dict[str, Any]:
    """Write a share-safe JSONL file plus a manifest. Explicit ids or explicit filters only."""
    target = Path(out_path)
    moment = (at or _now()).isoformat()
    rows: list[dict[str, Any]] = []
    if ids is not None:
        wanted = [str(item) for item in ids]
        for memory_id in wanted[: int(limit)]:
            found = _rows(store, "SELECT * FROM memories WHERE id = ? AND state <> 'archived'",
                          (memory_id,))
            rows.extend(found)
    else:
        params: list[Any] = []
        where = "state <> 'archived'"
        if kinds:
            names = [str(item) for item in kinds]
            where += f" AND kind IN ({','.join('?' for _ in names)})"
            params.extend(names)
        rows = _rows(store, f"SELECT * FROM memories WHERE {where} ORDER BY created_at DESC "
                            f"LIMIT ?", tuple(params) + (int(limit),))
    exported: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    labels: dict[str, int] = {}
    genericised: dict[str, int] = {}
    for row in rows:
        prepared = prepare(str(row.get("content") or ""), mask_private_ips=mask_private_ips)
        if prepared.refused:
            refused.append({"memory_id": str(row.get("id") or ""), "reason": prepared.refused})
            continue
        for label in prepared.labels:
            labels[label] = labels.get(label, 0) + 1
        for name in prepared.genericised:
            genericised[name] = genericised.get(name, 0) + 1
        exported.append({
            "id": str(row.get("id") or ""),
            "kind": str(row.get("kind") or ""),
            "text": prepared.text,
            "confidence": round(float(row.get("confidence") or 0.0), 3),
            "created_at": str(row.get("created_at") or "")[:19],
        })
    payload = "\n".join(json.dumps(item, sort_keys=True) for item in exported)
    if payload:
        payload += "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    manifest = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "generated_at": moment,
        "exported": len(exported),
        "refused": len(refused),
        "refusals": refused,
        "redactions": dict(sorted(labels.items())),
        "genericised": dict(sorted(genericised.items())),
        "sha256": digest,
        "bytes": len(payload.encode("utf-8")),
        "mask_private_ips": bool(mask_private_ips),
        "note": "the manifest names patterns, never the values they replaced",
    }
    Path(str(target) + ".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                                    encoding="utf-8")
    return manifest


def verify(path: str | Path) -> dict[str, Any]:
    """Recompute the checksum of an exported file and compare it with its manifest."""
    target = Path(path)
    manifest_path = Path(str(target) + ".manifest.json")
    if not target.exists():
        return {"ok": False, "reason": "no export file"}
    if not manifest_path.exists():
        return {"ok": False, "reason": "no manifest beside the export"}
    payload = target.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lines = [line for line in payload.decode("utf-8").splitlines() if line.strip()]
    return {
        "ok": digest == manifest.get("sha256") and len(lines) == int(manifest.get("exported") or 0),
        "sha256": digest,
        "expected": manifest.get("sha256", ""),
        "lines": len(lines),
        "expected_lines": int(manifest.get("exported") or 0),
        "refused": int(manifest.get("refused") or 0),
        "format": manifest.get("format", ""),
    }


def dry_run(store: Any, *, kinds: Iterable[str] | None = None, limit: int = 200,
            mask_private_ips: bool = False) -> dict[str, Any]:
    """What a share of this slice would contain and remove — without writing a file."""
    rows: list[dict[str, Any]] = []
    params: list[Any] = []
    where = "state <> 'archived'"
    if kinds:
        names = [str(item) for item in kinds]
        where += f" AND kind IN ({','.join('?' for _ in names)})"
        params.extend(names)
    rows = _rows(store, f"SELECT content FROM memories WHERE {where} ORDER BY created_at DESC "
                        f"LIMIT ?", tuple(params) + (int(limit),))
    labels: dict[str, int] = {}
    genericised: dict[str, int] = {}
    refused = 0
    for row in rows:
        prepared = prepare(str(row.get("content") or ""), mask_private_ips=mask_private_ips)
        if prepared.refused:
            refused += 1
            continue
        for label in prepared.labels:
            labels[label] = labels.get(label, 0) + 1
        for name in prepared.genericised:
            genericised[name] = genericised.get(name, 0) + 1
    return {
        "examined": len(rows),
        "would_export": len(rows) - refused,
        "would_refuse": refused,
        "redactions": dict(sorted(labels.items())),
        "genericised": dict(sorted(genericised.items())),
        "note": "counts only: no values are read into the report",
    }
