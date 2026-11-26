"""Typed environment facts, extracted from prose and dated (backlog #135).

Half of what the store knows about Jacoby's machines is buried in sentences: "CT116
(192.168.1.116) runs t3 serve on 127.0.0.1:3773", "the key is id_ed25519_t3_theo",
"T3 is on 0.1.4-nightly.20261028.3824". Each of those is answerable only if recall happens to
surface the right paragraph, and none of them carries a lookup key.

This module turns them into rows: **kind, subject, value, confirmed_at**. Five kinds — host,
port, path, credential pointer, version pin. Nothing is invented: every row points at the memory
it came from, and its date is the date *the store learned it*, never a date guessed from the text.

Three rules keep it honest:

- **A registry entry is evidence, not truth.** Two memories that disagree about the same
  subject+kind are both kept and reported as a conflict; the registry never picks a winner.
  Silent arbitration is how a store ends up confidently wrong about a port number.
- **No secrets, only pointers.** A value that looks like a credential (`ghp_…`, 40-char hex,
  `password: …`) is dropped, not stored. A *path* to a secret file is a fact worth keeping.
- **Undated is a failure.** Every row carries a confirmed date; `report()["undated"]` must be 0.

Extraction is deliberately cheap and deterministic (rule 3: cheap model reads) and conservative:
a port with no host in its sentence is not recorded, because "port 8100" with no subject collides
with every other port in the store.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

KINDS = ("host", "port", "path", "credential", "version")
TABLE = "environment_facts"

MAX_SNIPPET = 180
MAX_TEXT = 20_000

# A value matching any of these is a secret, not a fact. Dropped at extraction time.
_SECRET = (
    re.compile(r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\b[A-Fa-f0-9]{40,}\b"),
    re.compile(r"\b[A-Za-z0-9+/]{43,}={0,2}\b"),
    re.compile(r"(?i)\b(?:password|passwd|secret|token|api[_ -]?key)\s*[:=]\s*\S+"),
)

_IPV4 = re.compile(r"(?<![\d.])((?:\d{1,3}\.){3}\d{1,3})(?!\d|\.\d)")
_CT = re.compile(r"\bCT\s?-?(\d{2,3})\b", re.IGNORECASE)
_LABELLED_IP = re.compile(
    r"\b([A-Za-z][\w.-]{1,30})\s*(?:\(\s*|=\s*|:\s*|@\s*|→\s*|->\s*|\bis at\s+|\bis on\s+|"
    r"\bruns on\s+|\bsits at\s+|\blives at\s+|\bat\s+)"
    r"((?:\d{1,3}\.){3}\d{1,3})(?!\d|\.\d)"
)
_HOST_PORT = re.compile(r"(?<![\w.])((?:[A-Za-z][\w.-]{1,40})|(?:\d{1,3}\.){3}\d{1,3}):(\d{2,5})\b(?![/\d])")
_PORT_WORD = re.compile(r"\bports?\s+(\d{2,5})\b", re.IGNORECASE)
_PATH = re.compile(
    r"(?<![\w/.-])(/(?:root|etc|opt|srv|var|home|usr|mnt|data|run)/[A-Za-z0-9._@%+-]*(?:/[A-Za-z0-9._@%+-]+)*)"
)
_VERSION = re.compile(
    r"(?<![\w.-])v?(\d+\.\d+\.\d+(?:-[A-Za-z0-9][A-Za-z0-9.+-]*[A-Za-z0-9])?)(?!\w|\.\w)"
)

# Words that never name the thing a version belongs to.
_STOP = {
    "a", "an", "and", "the", "at", "on", "in", "is", "are", "was", "were", "to", "for", "of",
    "build", "builds", "version", "versions", "pin", "pinned", "release", "releases", "running",
    "runs", "run", "installed", "upgraded", "updated", "new", "old", "beta", "nightly", "v",
    "from", "by", "with", "as", "or", "if", "not", "no", "it", "its", "this", "that", "always",
    "never", "must", "should", "can", "will", "only", "then", "than", "so", "but", "also",
}

_CRED_PATH = re.compile(
    r"(?i)(?:/(?:secrets?|\.ssh|\.gnupg)(?:/|$)|/(?:id_[a-z0-9_]+)|\b[./\w-]*\.(?:env|pem|p12|key)\b|\btoken\b|\bcredential\b)"
)
_TOKEN = re.compile(r"[A-Za-z][\w.+-]{0,40}")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _has_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SECRET)


@dataclass
class Fact:
    kind: str
    value: str
    subject: str = ""
    memory_id: str = ""
    confirmed_at: datetime | None = None
    snippet: str = ""
    confirmations: int = 1

    @property
    def fact_id(self) -> str:
        key = f"{self.kind}|{self.subject.lower()}|{self.value.lower()}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "kind": self.kind,
            "subject": self.subject,
            "value": self.value,
            "memory_id": self.memory_id,
            "confirmed_at": self.confirmed_at.isoformat() if self.confirmed_at else "",
            "snippet": self.snippet,
        }

    def describe(self) -> str:
        stamp = self.confirmed_at.date().isoformat() if self.confirmed_at else "undated"
        subject = f"{self.subject} = " if self.subject else ""
        return f"{self.kind:<10} {subject}{self.value}  ({stamp})"


@dataclass
class Conflict:
    kind: str
    subject: str
    values: list[tuple[str, datetime | None]] = field(default_factory=list)

    def describe(self) -> str:
        parts = [f"{value} ({when.date().isoformat() if when else 'undated'})" for value, when in self.values]
        return f"{self.kind} {self.subject}: " + " vs ".join(parts)


_GENERIC = {
    "file", "files", "path", "paths", "dir", "dirs", "folder", "cat", "ls", "ssh", "scp", "cp",
    "mv", "rm", "grep", "curl", "sudo", "note", "notes", "see", "use", "uses", "used", "run",
    "runs", "ran", "there", "this", "that", "it", "its", "they", "them", "we", "you", "he", "she",
    "here", "where", "what", "which", "who", "when", "how", "why", "all", "any", "some", "more",
    "less", "one", "two", "first", "last", "next", "same", "other", "only", "just", "still",
    "also", "but", "and", "or", "if", "so", "then", "now", "never", "always", "must", "should",
    "can", "will", "would", "could", "may", "might", "does", "did", "has", "have", "had", "is",
    "are", "was", "were", "be", "been", "being", "list", "check", "verify", "important",
    "hard", "fact", "facts", "current", "correct", "wrong", "update", "updated", "changed", "new",
    "old", "good", "bad", "work", "works", "working", "lives", "located", "stored", "saved",
    # technical nouns that name no particular system
    "api", "cli", "server", "config", "configs", "source", "compose", "full", "fix", "fixes",
    "install", "installed", "script", "scripts", "root", "image", "images", "container",
    "containers", "volume", "volumes", "service", "services", "daemon", "port", "ports", "db",
    "sql", "http", "https", "url", "urls", "json", "yaml", "yml", "bash", "sh", "et", "eg",
    "ip", "ips", "a-record", "record", "records", "cname", "ptr", "mx", "dns", "host", "hosts",
    "monthly", "weekly", "daily", "upload", "uploads", "download", "downloads", "cron", "job",
    "confirmed", "store", "size", "count", "line", "page", "id", "value", "values", "piece",
    "the", "an", "a", "of", "in", "at", "on", "to", "for", "with", "by", "from", "as",
}


def _is_name(token: str) -> bool:
    """A token that plausibly names one machine or one piece of software."""
    low = token.lower().rstrip(".")
    if len(low) < 2 or low in _STOP or low in _GENERIC or not any(c.isalnum() for c in low):
        return False
    if re.fullmatch(r"[0-9a-f]{7,}", low):  # a hash or container id, not a name
        return False
    if any(character.isdigit() for character in low):  # r630, t3, pve01, ct116
        return True
    return token[0].isupper()  # Proxmox, Cortex, VPS — proper nouns only


_TOOLS = {
    "python", "python3", "python2", "pip", "pip3", "node", "npm", "npx", "docker", "podman",
    "systemctl", "bash", "sh", "zsh", "uv", "curl", "wget", "git", "apt", "make", "cargo",
    "go", "java", "ruby", "perl", "php", "nmap", "ssh-keygen", "journalctl", "kubectl",
}

# A token that looks like an address but is really four numbers: "FTB Teams 26.1.2.4".
_NETWORK = re.compile(
    r"(?i)\b(?:ip|ips|address|addr|host|hosts|ssh|scp|http|https|ping|curl|dns|tunnel|lan|"
    r"subnet|vlan|gateway|router|proxmox|container|node|vps|server|port|url|tailscale|twingate)\b"
)


def _private(ip: str) -> bool:
    octets = [int(part) for part in ip.split(".")]
    if octets[0] in (10, 127):
        return True
    if octets[0] == 192 and octets[1] == 168:
        return True
    if octets[0] == 172 and 16 <= octets[1] <= 31:
        return True
    if octets[0] == 100 and 64 <= octets[1] <= 127:  # CGNAT / Tailscale
        return True
    return False


def _name_token(raw: str) -> str:
    """A token that could name a system: not generic, not a hash, not a filename fragment."""
    token = raw.rstrip(".")
    low = token.lower()
    if len(token) > 30 or not re.search(r"[A-Za-z]", token):
        return ""
    if re.search(r"\.\d", token):  # "beta.3", "mc1.21", "neoforge-2.0.0"
        return ""
    if re.fullmatch(r"[0-9a-f]{5,}", low):  # a hash or container id
        return ""
    return low


def _host_subject(sentence: str, pos: int) -> str:
    """The machine a fact belongs to, when the sentence names one. ``""`` otherwise.

    Only boxes count here: a labelled address ("r630 (192.168.1.208)"), a CT number, or a
    token with digits in it. Filing a path under "python3" or "fix" is filing it nowhere —
    it collides with every other path in the store and makes the conflict report noise.
    """
    head = sentence[:pos]
    named = ""
    for match in _LABELLED_IP.finditer(head):
        if _is_name(match.group(1)):
            named = match.group(1).lower()
    if named:
        return named
    for match in _CT.finditer(head):
        named = f"ct{match.group(1)}"
    if named:
        return named
    for match in reversed(list(_TOKEN.finditer(head))):
        raw = match.group(0)
        if raw.lower().rstrip(".") in _STOP or raw.lower().rstrip(".") in _GENERIC:
            continue
        token = _name_token(raw)
        if token in _TOOLS:  # "python3 /root/.hermes/webui.env" names no machine
            continue
        if token and any(character.isdigit() for character in token):
            return token[:40]
    return ""


def _owner_subject(sentence: str, pos: int, *, exclude: str = "") -> str:
    """The thing a version belongs to: usually the nearest name, else the box in the sentence."""
    head = sentence[:pos]
    for match in reversed(list(_TOKEN.finditer(head))):
        raw = match.group(0)
        if raw.lower().rstrip(".") == exclude.lower():
            continue
        token = _name_token(raw)
        if token and _is_name(raw):
            return token[:40]
    return _host_subject(sentence, pos)


def _looks_like_date(value: str) -> bool:
    """``2026.9.16`` is a date that matched the version shape, not a release."""
    parts = value.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        return False
    year = int(parts[0])
    return 1990 <= year <= 2100 and int(parts[1]) <= 12 and int(parts[2]) <= 31


def _clip(text: str) -> str:
    return text if len(text) <= MAX_SNIPPET else text[:MAX_SNIPPET] + "…"


def detect(text: str) -> list[Fact]:
    """Extract environment facts from one memory's text. Never returns a secret."""
    body = str(text or "")[:MAX_TEXT]
    if not body.strip():
        return []
    facts: list[Fact] = []
    masked = list(body)

    def take(start: int, end: int) -> None:
        for index in range(start, min(end, len(masked))):
            masked[index] = " "

    # ports first, masking only their digits: `host:port` is a fact about both.
    for match in _HOST_PORT.finditer(body):
        owner, port = match.group(1), match.group(2)
        if not port.isdigit() or not (100 <= int(port) <= 65535):
            continue  # `postgres:14` is an image tag, not a listening port
        if owner.lower() in {"http", "https", "ftp", "ssh", "tcp", "udp", "ws", "wss"}:
            continue
        if owner.lower() in _GENERIC:
            continue  # `root:1001` is a docker user mapping, not a port
        if re.search(r"(?i)\.(?:py|js|ts|tsx|jsx|json|md|sh|ya?ml|txt|log|sql|html|css|toml|ini|conf|db|env)$", owner):
            continue  # `store.py:13177` is a stack-trace line number
        facts.append(Fact("port", port, subject=owner.lower(), snippet=_clip(match.group(0))))
        take(match.start(2), match.end(2))

    for match in _PORT_WORD.finditer("".join(masked)):
        port = match.group(1)
        if not (100 <= int(port) <= 65535):
            continue
        sentence = _sentence_at(body, match.start())
        offset = match.start() - _sentence_start(body, match.start())
        subject = _host_subject(sentence, offset)
        if not subject:
            continue  # an unattributed port is a number, not a fact
        facts.append(Fact("port", port, subject=subject, snippet=_clip(sentence)))
        take(match.start(), match.end())

    # hosts: "r630 (192.168.1.208)" names the box; a bare IP is its own subject.
    for match in _LABELLED_IP.finditer("".join(masked)):
        ip, label = match.group(2), match.group(1).lower()
        if _has_secret(ip) or not _is_name(match.group(1)):
            continue
        facts.append(Fact("host", ip, subject=label, snippet=_clip(match.group(0))))
        take(match.start(), match.end())
    for match in _IPV4.finditer("".join(masked)):
        ip = match.group(1)
        octets = [int(part) for part in ip.split(".")]
        if any(part > 255 for part in octets) or ip.startswith("0."):
            continue
        sentence = _sentence_at(body, match.start())
        if sentence.count("|") >= 2:
            continue  # a markdown table row; cells are not facts in a sentence
        if not _private(ip) and not _NETWORK.search(sentence):
            continue  # four numbers with no network context is a version, not a host
        facts.append(Fact("host", ip, subject=ip, snippet=_clip(sentence)))
        take(match.start(), match.end())

    joined = "".join(masked)
    for match in _PATH.finditer(joined):
        path = match.group(1).rstrip(".,;:)")
        if len(path) < 4 or _has_secret(path):
            continue
        sentence = _sentence_at(body, match.start())
        offset = match.start() - _sentence_start(body, match.start())
        subject = _host_subject(sentence, offset)
        kind = "credential" if _CRED_PATH.search(path) else "path"
        facts.append(Fact(kind, path, subject=subject, snippet=_clip(sentence)))
        take(match.start(), match.end())

    joined = "".join(masked)
    for match in _VERSION.finditer(joined):
        if _has_secret(match.group(0)) or _looks_like_date(match.group(1)):
            continue
        sentence = _sentence_at(body, match.start())
        offset = match.start() - _sentence_start(body, match.start())
        if sentence.count("|") >= 2:
            continue  # table row: "| store | 8.160.3 |" is not a pin on something called store
        subject = _owner_subject(sentence, offset)
        if not subject:
            continue  # a version with no owner tells nobody anything
        facts.append(Fact("version", match.group(1), subject=subject, snippet=_clip(sentence)))

    # drop anything that still smells like a credential value
    return [fact for fact in facts if fact.value and not _has_secret(fact.value)]


def _sentence_start(text: str, pos: int) -> int:
    start = text.rfind("\n", 0, pos)
    boundary = max(start, text.rfind(". ", 0, pos), text.rfind("! ", 0, pos), text.rfind("? ", 0, pos))
    return boundary + 1


def _sentence_at(text: str, pos: int) -> str:
    start = _sentence_start(text, pos)
    end_candidates = [index for index in (text.find("\n", pos), text.find(". ", pos)) if index != -1]
    end = min(end_candidates) + 1 if end_candidates else len(text)
    return text[start:end].strip()


def _ensure(store: Any) -> None:
    with store.transaction() as conn:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    fact_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    subject TEXT NOT NULL DEFAULT '',
                    value TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    confirmed_at TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    confirmations INTEGER NOT NULL DEFAULT 1,
                    snippet TEXT NOT NULL DEFAULT '',
                    built_at TEXT NOT NULL
                )"""
        )
        conn.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_lookup ON {TABLE} (kind, subject)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_value ON {TABLE} (value)")


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001 - an absent table is an empty registry
        return []


def build(store: Any, *, scan: int = 0, persist: bool = True) -> list[Fact]:
    """Scan active memories and register every environment fact found in them."""
    sql = "SELECT id, content, created_at, observed_at FROM memories WHERE state = 'active' ORDER BY id"
    params: tuple = ()
    if scan:
        sql = ("SELECT id, content, created_at, observed_at FROM memories WHERE state = 'active' "
               "ORDER BY id DESC LIMIT ?")
        params = (int(scan),)
    rows = _rows(store, sql, params)
    found: dict[str, Fact] = {}
    for row in rows:
        when = _utc(row.get("observed_at")) or _utc(row.get("created_at"))
        if when is None:
            continue
        for fact in detect(row.get("content") or ""):
            fact.memory_id = str(row.get("id") or "")
            fact.confirmed_at = when
            current = found.get(fact.fact_id)
            if current is None:
                found[fact.fact_id] = fact
            elif fact.confirmed_at and current.confirmed_at and fact.confirmed_at > current.confirmed_at:
                current.confirmed_at = fact.confirmed_at
                current.memory_id = fact.memory_id
                current.snippet = fact.snippet
                current.confirmations += 1
            else:
                current.confirmations += 1
    facts = sorted(found.values(), key=lambda f: (f.kind, f.subject, f.value))
    if persist and facts:
        _persist(store, facts)
    return facts


def _persist(store: Any, facts: Iterable[Fact]) -> int:
    _ensure(store)
    now = _now().isoformat()
    written = 0
    with store.transaction() as conn:
        for fact in facts:
            existing = conn.execute(
                f"SELECT first_seen_at, confirmations FROM {TABLE} WHERE fact_id = ?",
                (fact.fact_id,),
            ).fetchone()
            first_seen = existing["first_seen_at"] if existing else (
                fact.confirmed_at.isoformat() if fact.confirmed_at else now
            )
            confirmations = int(existing["confirmations"]) if existing else 0
            conn.execute(
                f"""INSERT INTO {TABLE}
                        (fact_id, kind, subject, value, memory_id, confirmed_at,
                         first_seen_at, confirmations, snippet, built_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(fact_id) DO UPDATE SET
                        memory_id = excluded.memory_id,
                        confirmed_at = excluded.confirmed_at,
                        confirmations = excluded.confirmations,
                        snippet = excluded.snippet,
                        built_at = excluded.built_at""",
                (
                    fact.fact_id, fact.kind, fact.subject, fact.value, fact.memory_id,
                    fact.confirmed_at.isoformat() if fact.confirmed_at else "",
                    first_seen, max(confirmations, fact.confirmations),
                    fact.snippet, now,
                ),
            )
            written += 1
    return written


def lookup(store: Any, query: str = "", *, kind: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    """Facts matching ``query`` in subject or value, newest confirmation first."""
    text = str(query or "").strip().lower()
    where, params = [], []
    if kind:
        where.append("kind = ?")
        params.append(kind)
    if text:
        where.append("(lower(subject) LIKE ? OR lower(value) LIKE ?)")
        params.extend([f"%{text}%", f"%{text}%"])
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    rows = _rows(
        store,
        f"SELECT * FROM {TABLE} {clause} ORDER BY confirmed_at DESC, confirmations DESC LIMIT ?",
        tuple(params) + (int(limit),),
    )
    return rows


MONOVALUED = ("host",)


def conflicts(store: Any, *, limit: int = 50) -> list[Conflict]:
    """One name, two addresses. Both sides kept and dated; a human decides which is live.

    Only ``host`` is monovalued. A box legitimately holds many ports, paths and credential
    pointers, so calling those "conflicts" would bury the one case that matters — the store
    believing a machine is at two different addresses.
    """
    rows = _rows(
        store,
        f"SELECT kind, subject, value, confirmed_at FROM {TABLE} "
        "WHERE subject != '' AND kind IN (%s) ORDER BY kind, subject, confirmed_at"
        % ",".join("?" for _ in MONOVALUED),
        MONOVALUED,
    )
    grouped: dict[tuple[str, str], Conflict] = {}
    for row in rows:
        key = (row["kind"], str(row["subject"]).lower())
        entry = grouped.setdefault(key, Conflict(kind=key[0], subject=key[1]))
        entry.values.append((row["value"], _utc(row["confirmed_at"])))
    out = [conflict for conflict in grouped.values() if len({value for value, _ in conflict.values}) > 1]
    return sorted(out, key=lambda c: (c.kind, c.subject))[:limit]


def drift(store: Any, *, limit: int = 50) -> list[Conflict]:
    """Pins for the same software that disagree, newest first. History, not necessarily error."""
    rows = _rows(
        store,
        f"SELECT kind, subject, value, confirmed_at FROM {TABLE} "
        "WHERE kind = 'version' AND subject != '' ORDER BY subject, confirmed_at DESC",
    )
    grouped: dict[tuple[str, str], Conflict] = {}
    for row in rows:
        key = (row["kind"], str(row["subject"]).lower())
        entry = grouped.setdefault(key, Conflict(kind=key[0], subject=key[1]))
        entry.values.append((row["value"], _utc(row["confirmed_at"])))
    out = [entry for entry in grouped.values() if len({value for value, _ in entry.values}) > 1]
    return sorted(out, key=lambda c: c.subject)[:limit]


RENDER_KINDS = (("host", 40), ("port", 40), ("version", 30), ("path", 15), ("credential", 10))


def render(store: Any, *, budget: int = 2000, kinds: Iterable[tuple[str, int]] = RENDER_KINDS) -> str:
    """A compact environment block: one line per fact, conflicts marked ``!!``.

    Ordered by what a turn actually needs — which box, what runs where, which version — with a
    per-kind cap so a store full of `.env` paths cannot crowd the pins out of the budget.
    """
    lines = ["## Environment (registry, confirmed dates)"]
    flagged = {(c.kind, c.subject.lower()) for c in conflicts(store)}
    rows: list[dict[str, Any]] = []
    for kind, cap in kinds:
        rows.extend(
            _rows(
                store,
                f"SELECT * FROM {TABLE} WHERE kind = ? "
                "ORDER BY (subject = value) ASC, subject, value LIMIT ?",
                (kind, int(cap)),
            )
        )
    if not rows:
        return ""
    for row in rows:
        stamp = (_utc(row["confirmed_at"]).date().isoformat() if _utc(row["confirmed_at"]) else "undated")
        mark = "!!" if (row["kind"], str(row["subject"]).lower()) in flagged else "  "
        subject = f"{row['subject']} = " if row["subject"] and row["subject"] != row["value"] else ""
        line = f"{mark} {row['kind']}: {subject}{row['value']} ({stamp})"
        if sum(len(existing) + 1 for existing in lines) + len(line) > budget:
            break
        lines.append(line)
    return "\n".join(lines)


def report(store: Any, *, scan: int = 0, rebuild: bool = False) -> dict[str, Any]:
    """Counts for the weekly digest: how much of the environment is actually registered."""
    if rebuild:
        build(store, scan=scan)
    rows = _rows(store, f"SELECT kind, value, subject, confirmed_at FROM {TABLE}")
    by_kind: dict[str, int] = {kind: 0 for kind in KINDS}
    undated = 0
    for row in rows:
        by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1
        if not _utc(row["confirmed_at"]):
            undated += 1
    found = conflicts(store, limit=10_000)
    pinned = drift(store, limit=10_000)
    return {
        "total": len(rows),
        "by_kind": by_kind,
        "undated": undated,
        "conflicts": len(found),
        "conflict_examples": [conflict.describe() for conflict in found[:5]],
        "drift": len(pinned),
        "drift_examples": [entry.describe() for entry in pinned[:5]],
        "subjects": len({(row["kind"], row["subject"].lower()) for row in rows if row["subject"]}),
    }


def main(argv: list[str] | None = None) -> int:
    """``python -m cortex.env_registry --db X [--report] [--build] [--lookup Q]``."""
    import argparse
    import sqlite3

    from .store import CortexStore

    parser = argparse.ArgumentParser(description="Environment facts registry (#135)")
    parser.add_argument("--db", required=True)
    parser.add_argument("--build", action="store_true", help="scan memories and persist the registry")
    parser.add_argument("--lookup", default="", help="find facts matching this text")
    parser.add_argument("--kind", default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)

    store = CortexStore(args.db)
    try:
        if args.build:
            facts = build(store)
            print(f"registered {len(facts)} facts")
        if args.lookup:
            for row in lookup(store, args.lookup, kind=args.kind):
                print(f"{row['kind']:<10} {row['subject'] or '-':<22} {row['value']}  ({row['confirmed_at'][:10]})")
            return 0
        if args.render:
            print(render(store))
            return 0
        print(json.dumps(report(store), indent=2))
    finally:
        store.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
