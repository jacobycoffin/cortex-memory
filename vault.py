"""Read-only Obsidian vault indexing for Cortex.

The importer never edits source notes. It creates provenance-rich document
memories, updates them incrementally, archives chunks whose source disappears,
and rebuilds Obsidian wikilink associations after each applied run.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .security import sanitize_memory
from .store import CortexStore


IMPORTER_VERSION = "obsidian_vault_importer_v1"
DEFAULT_MAX_CHARS = 1600
DEFAULT_MAX_FILE_BYTES = 1_000_000
_SKIP_PARTS = {".git", ".obsidian", ".trash", "node_modules", "attachments", "assets"}
# Build-stamp notes are rewritten with a fresh timestamp on every site rebuild,
# so they always look "changed" to the importer and would spawn a new archived
# memory each cycle. They carry no knowledge — exclude them from indexing.
# Paths are matched casefolded against the vault-relative POSIX path. Add future
# stamp notes here (keep the set small and knowledge-free).
_SKIP_NOTES = {"_meta/build info.md"}
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_WIKILINK = re.compile(r"!?\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_MARKDOWN_DECORATION = re.compile(r"[`*_~]+")
_RELATED_HEADING = re.compile(r"^(?:related|links?|backlinks?)$", re.I)
_NAVIGATION_HEADING = re.compile(r"^(?:quick links?|index|navigation|table of contents)$", re.I)
_PLACEHOLDER_BODY = re.compile(
    r"\b(?:add detail here|tbd|todo|placeholder|fill this in|not yet documented)\b",
    re.I,
)
_IMPORTER_ARCHIVE_REASONS = {"vault chunk superseded", "vault source removed"}


@dataclass(frozen=True)
class VaultChunk:
    key: str
    heading: str
    ordinal: int
    content: str
    digest: str
    kind: str
    confidence: float
    currentness: float
    importance: float
    volatility: float
    valid_from: str | None
    redacted: bool
    quarantine_reason: str | None


@dataclass(frozen=True)
class VaultNote:
    relative_path: str
    title: str
    digest: str
    modified_at: str
    size_bytes: int
    chunks: tuple[VaultChunk, ...]
    links: tuple[str, ...]
    link_reasons: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class VaultScan:
    root: Path
    notes: tuple[VaultNote, ...]
    skipped_files: int


class VaultIndexer:
    def __init__(
        self,
        store: CortexStore,
        vault_path: str | Path,
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ):
        self.store = store
        self.vault_path = Path(vault_path).expanduser().resolve()
        self.max_chars = max(400, min(int(max_chars), 8000))
        self.max_file_bytes = max(10_000, int(max_file_bytes))

    def scan(self) -> VaultScan:
        if not self.vault_path.is_dir():
            raise ValueError(f"vault directory does not exist: {self.vault_path}")
        notes: list[VaultNote] = []
        skipped = 0
        for path in sorted(self.vault_path.rglob("*.md")):
            relative = path.relative_to(self.vault_path)
            if path.is_symlink() or any(part.startswith(".") or part in _SKIP_PARTS for part in relative.parts):
                skipped += 1
                continue
            if relative.as_posix().casefold() in _SKIP_NOTES:
                skipped += 1
                continue
            # Guard against a symlinked *directory* inside the vault leading rglob
            # to files whose real location is outside the vault root.
            if not path.resolve().is_relative_to(self.vault_path):
                skipped += 1
                continue
            stat = path.stat()
            if stat.st_size > self.max_file_bytes:
                skipped += 1
                continue
            raw = path.read_bytes()
            text = raw.decode("utf-8", errors="replace")
            notes.append(
                _parse_note(
                    relative,
                    text,
                    digest=hashlib.sha256(raw).hexdigest(),
                    modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                    size_bytes=stat.st_size,
                    max_chars=self.max_chars,
                )
            )
        return VaultScan(self.vault_path, tuple(notes), skipped)

    def plan(self) -> tuple[VaultScan, dict[str, Any]]:
        scan = self.scan()
        manifest = self.store.document_manifest()
        scanned_paths = {note.relative_path for note in scan.notes}
        report: dict[str, Any] = {
            "vault_path": str(scan.root),
            "files_scanned": len(scan.notes),
            "files_skipped": scan.skipped_files,
            "new_files": 0,
            "changed_files": 0,
            "unchanged_files": 0,
            "removed_files": 0,
            "chunks_total": sum(len(note.chunks) for note in scan.notes),
            "chunks_add": 0,
            "chunks_update": 0,
            "chunks_reactivate": 0,
            "chunks_archive": 0,
            "wikilinks": sum(len(note.links) for note in scan.notes),
            "redacted_chunks": sum(chunk.redacted for note in scan.notes for chunk in note.chunks),
            "quarantined_chunks": sum(bool(chunk.quarantine_reason) for note in scan.notes for chunk in note.chunks),
        }
        for note in scan.notes:
            previous = manifest.get(note.relative_path)
            existing = {
                row["chunk_key"]: row
                for row in self.store.document_chunks(note.relative_path, active_only=False)
            }
            incoming = {chunk.key: chunk for chunk in note.chunks}
            if not previous:
                report["new_files"] += 1
                report["chunks_add"] += len(incoming)
            elif previous["content_hash"] == note.digest and previous["status"] == "active":
                report["unchanged_files"] += 1
                report["chunks_reactivate"] += sum(
                    self._should_reactivate_importer_archive(existing[key])
                    for key in incoming
                    if key in existing
                )
            else:
                report["changed_files"] += 1
                report["chunks_add"] += sum(key not in existing for key in incoming)
                report["chunks_reactivate"] += sum(
                    key in existing and not bool(existing[key]["active"]) for key in incoming
                )
                report["chunks_update"] += sum(
                    key in existing and existing[key]["chunk_hash"] != chunk.digest
                    for key, chunk in incoming.items()
                )
                report["chunks_archive"] += sum(
                    key not in incoming and bool(row["active"]) for key, row in existing.items()
                )
        removed = [path for path, source in manifest.items() if source["status"] == "active" and path not in scanned_paths]
        report["removed_files"] = len(removed)
        report["chunks_archive"] += sum(len(self.store.document_chunks(path, active_only=True)) for path in removed)
        return scan, report

    def apply(self) -> dict[str, Any]:
        scan, report = self.plan()
        manifest = self.store.document_manifest()
        scanned_paths = {note.relative_path for note in scan.notes}
        counters = {
            "memories_created": 0,
            "memories_updated": 0,
            "memories_reactivated": 0,
            "memories_superseded": 0,
            "memories_archived": 0,
            "vault_links_created": 0,
        }

        for note in scan.notes:
            self.store.upsert_document_source(
                source_path=note.relative_path,
                title=note.title,
                content_hash=note.digest,
                modified_at=note.modified_at,
                size_bytes=note.size_bytes,
            )
            existing = {
                row["chunk_key"]: row
                for row in self.store.document_chunks(note.relative_path, active_only=False)
            }
            incoming_keys: set[str] = set()
            for chunk in note.chunks:
                incoming_keys.add(chunk.key)
                previous = existing.get(chunk.key)
                memory_id: str
                if previous and bool(previous["active"]) and previous["chunk_hash"] == chunk.digest:
                    if not self._should_reactivate_importer_archive(previous):
                        continue
                if previous:
                    memory_id = str(previous["memory_id"])
                    current = self.store.get_memory(memory_id)
                    if current:
                        preserve_manual_archive = (
                            bool(previous["active"])
                            and str(current["state"]) == "archived"
                            and not self._should_reactivate_importer_archive(previous)
                        )
                        next_state = (
                            "quarantine"
                            if chunk.quarantine_reason
                            else "archived"
                            if preserve_manual_archive
                            else "active"
                        )
                        changed = self.store.update_document_memory(
                            memory_id,
                            chunk.content,
                            kind=chunk.kind,
                            source_ref=f"vault:{note.relative_path}#{chunk.heading}",
                            entities=[note.title, chunk.heading],
                            source_context=(
                                f"Vault document {note.relative_path} under heading {chunk.heading}."
                            ),
                            observed_at=note.modified_at,
                            confidence=chunk.confidence,
                            currentness_confidence=chunk.currentness,
                            importance=chunk.importance,
                            uniqueness=0.9,
                            volatility=chunk.volatility,
                            trust=0.78,
                            state=next_state,
                            quarantine_reason=chunk.quarantine_reason,
                            valid_from=chunk.valid_from,
                            subject=f"vault:{note.relative_path}#{chunk.key}",
                            predicate="documents",
                            object_value=chunk.heading,
                            extraction_method=IMPORTER_VERSION,
                            reason=(
                                "vault section restored"
                                if not bool(previous["active"])
                                else "vault chunk revised in place"
                            ),
                        )
                        counters["memories_updated"] += int(changed["updated"])
                        counters["memories_reactivated"] += int(changed["reactivated"])
                    else:
                        previous = None
                if not previous:
                    memory_id, created = self.store.add_memory(
                        chunk.content,
                        kind=chunk.kind,
                        source_type="vault_markdown",
                        source_category="DOCUMENT_EXTRACTED",
                        record_role="reference",
                        source_ref=f"vault:{note.relative_path}#{chunk.heading}",
                        entities=[note.title, chunk.heading],
                        source_context=f"Vault document {note.relative_path} under heading {chunk.heading}.",
                        observed_at=note.modified_at,
                        confidence=chunk.confidence,
                        currentness_confidence=chunk.currentness,
                        importance=chunk.importance,
                        uniqueness=0.9,
                        volatility=chunk.volatility,
                        trust=0.78,
                        state="quarantine" if chunk.quarantine_reason else "active",
                        quarantine_reason=chunk.quarantine_reason,
                        valid_from=chunk.valid_from,
                        subject=f"vault:{note.relative_path}#{chunk.key}",
                        predicate="documents",
                        object_value=chunk.heading,
                        extraction_method=IMPORTER_VERSION,
                    )
                    counters["memories_created"] += int(created)
                self.store.upsert_document_chunk(
                    source_path=note.relative_path,
                    chunk_key=chunk.key,
                    memory_id=memory_id,
                    chunk_hash=chunk.digest,
                    heading=chunk.heading,
                    ordinal=chunk.ordinal,
                )
            for key, previous in existing.items():
                if key in incoming_keys or not bool(previous["active"]):
                    continue
                if self.store.set_state(previous["memory_id"], "archived", reason="vault section removed"):
                    counters["memories_archived"] += 1
                self.store.deactivate_document_chunk(note.relative_path, key)

        for source_path, source in manifest.items():
            if source["status"] == "active" and source_path not in scanned_paths:
                counters["memories_archived"] += len(self.store.mark_document_missing(source_path))

        self.store.clear_edges("vault_link")
        aliases, first_chunks = self._note_aliases(scan)
        linked_pairs: set[tuple[str, str]] = set()
        for note in scan.notes:
            source_memory = first_chunks.get(note.relative_path)
            if not source_memory:
                continue
            for link in note.links:
                target_path = aliases.get(_normalize_link(link))
                target_memory = first_chunks.get(target_path or "")
                if not target_memory or target_memory == source_memory:
                    continue
                pair = tuple(sorted((source_memory, target_memory)))
                if pair in linked_pairs:
                    continue
                linked_pairs.add(pair)
                link_reason = dict(note.link_reasons).get(link, "")
                if self.store.add_edge(
                    source_memory,
                    target_memory,
                    "vault_link",
                    weight=0.35,
                    evidence_type="explicit_wikilink",
                    evidence_key=f"{note.relative_path}:{link}:{target_path}",
                    explanation=(
                        f"The vault note {note.title} explicitly links to {link}. "
                        + (
                            f"Source context: {link_reason}"
                            if link_reason
                            else "This is a documented relationship, not a similarity guess."
                        )
                    ),
                    source_ref=f"vault:{note.relative_path}",
                    metadata={
                        "source_path": note.relative_path,
                        "target_path": target_path,
                        "wikilink": link,
                        "link_context": link_reason,
                    },
                ):
                    counters["vault_links_created"] += 1

        return {**report, **counters, "applied": True, "audit": self.store.audit(), "stats": self.store.stats()}

    def _should_reactivate_importer_archive(self, chunk: dict[str, Any]) -> bool:
        if not bool(chunk.get("active")):
            return False
        memory = self.store.get_memory(str(chunk.get("memory_id") or ""))
        if not memory or str(memory.get("state") or "") != "archived":
            return False
        event = self.store.latest_lifecycle_event(str(memory["id"]))
        reason = str(event.get("reason") if event else "").strip().casefold()
        return reason in _IMPORTER_ARCHIVE_REASONS

    def _note_aliases(self, scan: VaultScan) -> tuple[dict[str, str], dict[str, str]]:
        aliases: dict[str, str] = {}
        first_chunks: dict[str, str] = {}
        for note in scan.notes:
            chunks = self.store.document_chunks(note.relative_path, active_only=True)
            if chunks:
                first_chunks[note.relative_path] = chunks[0]["memory_id"]
            relative_no_suffix = str(Path(note.relative_path).with_suffix(""))
            aliases.setdefault(_normalize_link(relative_no_suffix), note.relative_path)
            aliases.setdefault(_normalize_link(Path(note.relative_path).stem), note.relative_path)
            aliases.setdefault(_normalize_link(note.title), note.relative_path)
        return aliases, first_chunks


def _parse_note(
    relative: Path,
    text: str,
    *,
    digest: str,
    modified_at: str,
    size_bytes: int,
    max_chars: int,
) -> VaultNote:
    text = _HTML_COMMENT.sub("", _strip_frontmatter(text)).replace("\x00", "")
    title = _note_title(relative, text)
    links = tuple(dict.fromkeys(match.strip() for match in _WIKILINK.findall(text) if match.strip()))
    link_contexts = _wikilink_contexts(text)
    sections = _sections(text, title)
    chunks: list[VaultChunk] = []
    heading_occurrences: dict[str, int] = {}
    ordinal = 0
    for heading, body in sections:
        if _skip_low_value_section(heading, body) and len(sections) > 1:
            continue
        base = _slug(heading) or "note"
        heading_occurrences[base] = heading_occurrences.get(base, 0) + 1
        occurrence = heading_occurrences[base]
        for part_index, part in enumerate(_split_text(body, max_chars=max_chars), start=1):
            cleaned = part.strip()
            if not cleaned or _skip_low_value_fragment(cleaned, heading=heading):
                continue
            prefix = f"Vault note: {title}\nPath: {relative.as_posix()}\nSection: {heading}\n"
            sanitized = sanitize_memory(prefix + cleaned)
            if len(sanitized.text) < 35:
                continue
            key = f"{base}-{occurrence}-p{part_index}"
            kind, confidence, currentness, importance, volatility, valid_from = _memory_profile(relative, heading)
            chunks.append(
                VaultChunk(
                    key=key,
                    heading=heading,
                    ordinal=ordinal,
                    content=sanitized.text,
                    digest=hashlib.sha256(sanitized.text.encode()).hexdigest(),
                    kind=kind,
                    confidence=confidence,
                    currentness=currentness,
                    importance=importance,
                    volatility=volatility,
                    valid_from=valid_from,
                    redacted=sanitized.redacted,
                    quarantine_reason=sanitized.quarantine_reason,
                )
            )
            ordinal += 1
    if not chunks and not _skip_low_value_fragment(text):
        fallback = sanitize_memory(f"Vault note: {title}\nPath: {relative.as_posix()}\n{text.strip()}")
        if fallback.text:
            kind, confidence, currentness, importance, volatility, valid_from = _memory_profile(relative, title)
            chunks.append(
                VaultChunk(
                    key="note-1-p1",
                    heading=title,
                    ordinal=0,
                    content=fallback.text,
                    digest=hashlib.sha256(fallback.text.encode()).hexdigest(),
                    kind=kind,
                    confidence=confidence,
                    currentness=currentness,
                    importance=importance,
                    volatility=volatility,
                    valid_from=valid_from,
                    redacted=fallback.redacted,
                    quarantine_reason=fallback.quarantine_reason,
                )
            )
    return VaultNote(
        relative.as_posix(),
        title,
        digest,
        modified_at,
        size_bytes,
        tuple(chunks),
        links,
        tuple((link, link_contexts.get(link, "")) for link in links),
    )


def _strip_frontmatter(text: str) -> str:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for index in range(1, min(len(lines), 200)):
        if lines[index].strip() == "---":
            return "\n".join(lines[index + 1 :])
    return text


def _note_title(relative: Path, text: str) -> str:
    for line in text.splitlines():
        match = _HEADING.match(line)
        if match and len(match.group(1)) == 1:
            return _clean_heading(match.group(2))
    return relative.stem


def _sections(text: str, title: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    stack: list[str] = []
    current_heading = title
    body: list[str] = []
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _HEADING.match(line)
        if match:
            if body and "\n".join(body).strip():
                sections.append((current_heading, "\n".join(body).strip()))
            level = len(match.group(1))
            heading = _clean_heading(match.group(2))
            stack = stack[: level - 1]
            stack.append(heading)
            current_heading = " › ".join(stack)
            body = []
        else:
            body.append(line)
    if body and "\n".join(body).strip():
        sections.append((current_heading, "\n".join(body).strip()))
    return sections or [(title, text.strip())]


def _split_text(text: str, *, max_chars: int) -> Iterable[str]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    buffer = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > max_chars:
            if buffer:
                yield buffer
                buffer = ""
            for start in range(0, len(paragraph), max_chars):
                yield paragraph[start : start + max_chars]
            continue
        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            yield buffer
            buffer = paragraph
    if buffer:
        yield buffer


def _skip_low_value_section(heading: str, body: str) -> bool:
    leaf = heading.split(" › ")[-1].strip()
    without_links = _WIKILINK.sub("", body)
    without_markup = _MARKDOWN_DECORATION.sub("", without_links)
    remaining_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]+", without_markup)
    if _RELATED_HEADING.search(leaf) and len(remaining_tokens) < 8:
        # Explicit wikilinks already become evidence-backed vault edges.  A
        # second memory containing only the link list adds noise, not knowledge.
        return True
    if _NAVIGATION_HEADING.search(leaf) and len(remaining_tokens) < 20:
        return True
    return _skip_low_value_fragment(body, heading=heading)


def _skip_low_value_fragment(body: str, *, heading: str = "") -> bool:
    undecorated = _WIKILINK.sub(lambda match: match.group(1), body)
    raw_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/-]+", undecorated)
    plain = _MARKDOWN_DECORATION.sub("", undecorated)
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/-]+", plain)
    if _PLACEHOLDER_BODY.search(plain) and len(tokens) < 24:
        return True
    meaningful = [
        token
        for token in tokens
        if token.casefold()
        not in {"the", "a", "an", "and", "or", "to", "of", "in", "for", "with", "here", "notes"}
    ]
    leaf = heading.split(" › ")[-1].strip().casefold()
    if re.search(r"https?://\S+", plain, re.I) and any(label in leaf for label in ("url", "endpoint", "address")):
        return False
    technical_identifiers = [token for token in raw_tokens if "_" in token]
    if len(technical_identifiers) >= 2:
        return False
    return len(meaningful) < 3


def _wikilink_contexts(text: str) -> dict[str, str]:
    contexts: dict[str, str] = {}
    for raw_line in text.splitlines():
        links = [match.strip() for match in _WIKILINK.findall(raw_line) if match.strip()]
        if not links:
            continue
        rendered = _WIKILINK.sub(lambda match: match.group(1).strip(), raw_line)
        rendered = " ".join(_MARKDOWN_DECORATION.sub("", rendered).strip(" -–—\t").split())[:300]
        for link in links:
            if rendered and rendered.casefold() != link.casefold():
                contexts.setdefault(link, rendered)
    return contexts


def _memory_profile(relative: Path, heading: str) -> tuple[str, float, float, float, float, str | None]:
    top = relative.parts[0].casefold() if len(relative.parts) > 1 else ""
    heading_lower = heading.casefold()
    valid_from = None
    if top in {"daily", "lifelog"}:
        kind = "episode"
        confidence, currentness, importance, volatility = 0.82, 0.62, 0.48, 0.68
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", relative.stem)
        valid_from = f"{date_match.group(0)}T00:00:00+00:00" if date_match else None
    elif top in {"manual", "tools"} or re.search(r"runbook|procedure|setup|recovery|troubleshoot|how to", heading_lower):
        kind = "procedure"
        confidence, currentness, importance, volatility = 0.80, 0.78, 0.70, 0.30
    elif top == "homelab":
        kind = "semantic"
        confidence, currentness, importance, volatility = 0.80, 0.70, 0.64, 0.48
    elif top in {"projects", "plans"}:
        kind = "semantic"
        confidence, currentness, importance, volatility = 0.78, 0.72, 0.62, 0.42
    else:
        kind = "semantic"
        confidence, currentness, importance, volatility = 0.78, 0.76, 0.56, 0.34
    return kind, confidence, currentness, importance, volatility, valid_from


def _clean_heading(value: str) -> str:
    value = re.sub(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", lambda m: m.group(2) or m.group(1), value)
    return _MARKDOWN_DECORATION.sub("", value).strip().strip("#") or "Untitled"


def _slug(value: str) -> str:
    value = _clean_heading(value).casefold()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value[:80]


def _normalize_link(value: str) -> str:
    value = value.strip().replace("\\", "/")
    if value.casefold().endswith(".md"):
        value = value[:-3]
    return value.casefold().strip("/")
