"""Deterministic Memory Refinery classification and presentation for Cortex.

The refinery separates readable memories from raw reference evidence without
mutating stored content. Everything in this module is deterministic and
inspectable: no model call, no randomness, no network. The classifier proposes
a record role plus readability flags; the presentation generator produces a
derived display layer that is never treated as additional factual evidence.

Stored memory content is whitespace-normalized (one line), so every structural
signal here must work on collapsed text as well as raw multiline drafts.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from .security import normalize_text


ROLE_CLASSIFIER_METHOD = "role_classifier"
ROLE_CLASSIFIER_VERSION = "role_classifier_v1"
from .temporal import classify_temporal  # noqa: E402  (temporal imports nothing from this package)

PRESENTATION_METHOD = "deterministic_presentation"
# v2 adds temporal-validity flags + an as-of stamp to the display layer. Bumping
# the version makes the refinery re-derive existing presentations.
PRESENTATION_VERSION = "presentation_v2"
OPERATOR_ROLE_METHOD = "operator_review"
LEGACY_ROLE_METHOD = "legacy_default"

RECORD_ROLES = ("canonical", "reference", "event", "claim")

# Flags that route a record into the Needs clarity queue. Structural flags
# such as contains_code mark reference material and are intentionally absent:
# code that is filed as reference evidence is already presented honestly.
CLARITY_FLAGS = (
    "unresolved_reference",
    "missing_source_context",
    "placeholder_content",
    "transient_status",
    "raw_output",
    "long_multi_claim",
    "ambiguous_temporal",
    "unconfirmed_claim",
)

_DOCUMENT_SOURCE_TYPES = {"vault_markdown", "document", "document_import"}
_DOCUMENT_SOURCE_CATEGORIES = {"DOCUMENT_EXTRACTED"}
_EXPLICIT_CANONICAL_KINDS = {"identity", "preference", "decision", "prospective", "procedure"}

_FENCED_CODE = re.compile(r"```|~~~~")
_TABLE_SEPARATOR = re.compile(r"\|[\s:]*-{3,}[\s:|-]*\|")
_TREE_DIAGRAM = re.compile(r"[├└│┌┐┤┬┴┼]|(?:^|\s)[|+][-+]{2,}")
_CONFIG_PAIR = re.compile(r"(?:^|\s)[A-Za-z0-9_.\-]{2,}\s?[:=]\s?[^\s=:]{1,80}(?=\s|$)")
_COMMAND_TOKEN = re.compile(
    r"(?:^|\s)(?:\$ \S|sudo \S|curl -|docker(?: compose)? \w|git \w+|systemctl \w|journalctl|"
    r"npm (?:run|install|ci)\b|pip3? install\b|apt(?:-get)? \w|ssh \S+@|python3? -m \S|kubectl \w|chmod \d)",
)
_STACK_TRACE = re.compile(
    r"Traceback \(most recent call last\)|(?:^|\s)at [\w.$<>]+\([^)]*:\d+\)|File \"[^\"]+\", line \d+",
)
_LOG_TOKEN = re.compile(
    r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}|\[(?:INFO|WARN(?:ING)?|ERROR|DEBUG|TRACE)\]|"
    r"(?:^|\s)(?:INFO|WARNING|ERROR|DEBUG):\s",
)
_UNRESOLVED_REFERENCE = re.compile(
    r"^\s*(?:this|that|it|they|he|she|those|these)\b|"
    r"\b(?:the above|the previous one|same as before|as discussed|over there|this one|that one)\b",
    re.I,
)
_PLACEHOLDER_CONTENT = re.compile(
    r"\b(?:add detail here|tbd|todo|placeholder|fill this in|not yet documented)\b",
    re.I,
)
_TRANSIENT_STATUS = re.compile(
    r"^tool execution observation:|^reinforced [a-z0-9_-]+ workflow:|"
    r"\b(?:task|command|tool|step|job|run|build|deploy(?:ment)?) (?:completed|succeeded|failed|finished)"
    r"(?: successfully)?[.!]?$|"
    r"\b(?:exit code|status code)\s*[=:]?\s*\d+\b|\bis (?:now )?(?:done|complete|finished)[.!]?$",
    re.I,
)
_AMBIGUOUS_TEMPORAL = re.compile(
    r"\b(?:currently|right now|at the moment|as of (?:now|today)|for now|temporarily)\b",
    re.I,
)
_UNCERTAIN_LANGUAGE = re.compile(
    r"\b(?:probably|likely|maybe|might|appears? to|seems? to|possibly|unconfirmed|unsure|I think)\b",
    re.I,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[-•*(\"']?\s?[A-Z0-9\"'`])")
_BULLET_SPLIT = re.compile(r"\s+[-•*]\s+(?=[A-Z0-9`\"'])|\s+\d+[.)]\s+(?=[A-Z0-9`\"'])")
_LEADING_BULLET = re.compile(r"^[-•*]\s+|^\d+[.)]\s+")
_CODE_PUNCTUATION = frozenset("{}[]();<>$=&|\\`#")

_EVIDENCE_LABELS = {
    "code": "Code reference",
    "table": "Operational table",
    "config": "Configuration reference",
    "diagram": "Diagram or tree reference",
    "commands": "Command reference",
    "log_output": "Log or output capture",
    "raw_json": "Raw structured capture",
    "document": "Document section",
    "raw": "Raw technical capture",
}


def _flag(flags: list[str], value: str) -> None:
    if value not in flags:
        flags.append(value)


def _json_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    try:
        loaded = json.loads(str(value or "[]"))
    except (json.JSONDecodeError, ValueError):
        return []
    return [str(item) for item in loaded if str(item).strip()] if isinstance(loaded, list) else []


def record_parts(memory: dict[str, Any]) -> tuple[str, dict[str, str] | None]:
    """Return (body, vault_meta) for a stored record.

    Vault imports carry a provenance prefix inside their normalized content
    ("Vault note: … Path: … Section: …"). The structured columns — entities
    and the section heading in object_value — are the reliable way to strip
    it, because whitespace normalization removed the original line breaks.
    """

    content = " ".join(str(memory.get("content") or "").split())
    if not content.startswith("Vault note:"):
        return content, None
    entities = _json_list(memory.get("entities") if memory.get("entities") is not None else memory.get("entities_json"))
    title = entities[0] if entities else ""
    section = str(memory.get("object_value") or "") or (entities[1] if len(entities) > 1 else "")
    marker = f" Section: {section} " if section else " Section: "
    index = content.find(marker)
    if index >= 0:
        body = content[index + len(marker):].strip()
    else:
        fallback = content.find(" Section: ")
        body = content[fallback + len(" Section: "):].strip() if fallback >= 0 else content
    if not title:
        match = re.match(r"^Vault note:\s*(.*?)\s+Path:\s", content)
        title = match.group(1).strip() if match else ""
    return body, {"title": title, "section": section.strip()}


def _code_punctuation_density(text: str) -> float:
    if not text:
        return 0.0
    hits = sum(1 for char in text if char in _CODE_PUNCTUATION)
    return hits / max(1, len(text))


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 24 or stripped[:1] not in "{[":
        return False
    try:
        json.loads(stripped)
        return True
    except (json.JSONDecodeError, ValueError):
        pairs = len(re.findall(r"\"[^\"]+\"\s*:", stripped))
        return pairs >= 3


def _sentences(text: str) -> list[str]:
    collapsed = " ".join(text.split())
    if not collapsed:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT.split(collapsed) if part.strip()]


def detect_structures(body: str) -> list[str]:
    """Return deterministic structural signals found in (collapsed) raw content."""

    signals: list[str] = []
    if _FENCED_CODE.search(body):
        signals.append("code")
    pipe_count = body.count("|")
    if _TABLE_SEPARATOR.search(body) or pipe_count >= 8:
        signals.append("table")
    if _TREE_DIAGRAM.search(body):
        signals.append("diagram")
    if len(_CONFIG_PAIR.findall(body)) >= 3 and "code" not in signals:
        signals.append("config")
    if len(_COMMAND_TOKEN.findall(body)) >= 2:
        signals.append("commands")
    if _STACK_TRACE.search(body) or len(_LOG_TOKEN.findall(body)) >= 3:
        signals.append("log_output")
    if _looks_like_json(body):
        signals.append("raw_json")
    if "code" not in signals and _code_punctuation_density(body) >= 0.055 and len(body) >= 120:
        signals.append("code_like_density")
    return signals


def classify_record_role(
    memory: dict[str, Any],
    *,
    has_active_dependencies: bool = False,
) -> dict[str, Any]:
    """Propose a record role, readability flags, and plain reasons.

    Pure function over the stored record: it never mutates, never calls a
    model, and always returns an explainable result.
    """

    kind = str(memory.get("kind") or "semantic").casefold()
    source_type = str(memory.get("source_type") or "").casefold()
    source_category = str(memory.get("source_category") or "").upper()
    body, vault_meta = record_parts(memory)
    structures = detect_structures(body)
    sentences = _sentences(body)
    flags: list[str] = []
    reasons: list[str] = []

    for signal in structures:
        if signal in {"code", "code_like_density"}:
            _flag(flags, "contains_code")
        elif signal == "table":
            _flag(flags, "contains_table")
        elif signal == "config":
            _flag(flags, "contains_config")
        elif signal == "diagram":
            _flag(flags, "contains_diagram")
        elif signal == "commands":
            _flag(flags, "contains_commands")
        elif signal in {"log_output", "raw_json"}:
            _flag(flags, "raw_output")

    if _PLACEHOLDER_CONTENT.search(body):
        _flag(flags, "placeholder_content")
    if _TRANSIENT_STATUS.search(" ".join(body.split())):
        _flag(flags, "transient_status")
    if _UNRESOLVED_REFERENCE.search(body):
        _flag(flags, "unresolved_reference")
    if (
        source_category in {"AGENT_INFERENCE", "REFLECTION"}
        and not normalize_text(str(memory.get("source_context") or ""))
    ):
        _flag(flags, "missing_source_context")
    if (
        _AMBIGUOUS_TEMPORAL.search(body)
        and not memory.get("valid_from")
        and not memory.get("valid_to")
    ):
        _flag(flags, "ambiguous_temporal")
    if len(sentences) >= 6 and len(body) >= 900:
        _flag(flags, "long_multi_claim")

    is_document = (
        source_type in _DOCUMENT_SOURCE_TYPES
        or source_category in _DOCUMENT_SOURCE_CATEGORIES
        or bool(vault_meta)
    )
    structural_reference = bool(
        {"code", "table", "diagram", "config", "commands", "log_output", "raw_json", "code_like_density"}
        & set(structures)
    )

    if is_document:
        role = "reference"
        reasons.append(
            "The record was extracted from a document source, so it is kept as raw reference evidence "
            "rather than presented as a normal memory statement."
        )
        if structural_reference:
            reasons.append(f"Structural signals support the reference role: {', '.join(structures)}.")
    elif structural_reference:
        role = "reference"
        reasons.append(
            "The raw content is structured technical material "
            f"({', '.join(structures)}), which is searchable evidence, not a readable memory statement."
        )
    elif kind == "episode":
        role = "event"
        reasons.append("Episodic observations have bounded temporal value and are kept as events.")
    elif source_category in {"AGENT_INFERENCE", "REFLECTION"} and not has_active_dependencies:
        role = "claim"
        _flag(flags, "unconfirmed_claim")
        reasons.append(
            "The statement was inferred without active supporting evidence, so it stays an unconfirmed "
            "claim until reviewed or supported."
        )
    else:
        role = "canonical"
        if source_category == "USER_EXPLICIT" and kind in _EXPLICIT_CANONICAL_KINDS:
            reasons.append("You explicitly stored this durable statement, so it remains a readable memory.")
        else:
            reasons.append("The statement reads as one durable, independently understandable memory.")

    if role == "canonical" and any(flag in CLARITY_FLAGS for flag in flags):
        reasons.append(
            "Readability flags were preserved for review instead of silently discarding the record: "
            + ", ".join(flag for flag in flags if flag in CLARITY_FLAGS)
            + "."
        )

    return {
        "record_role": role,
        "readability_flags": flags,
        "reasons": reasons,
        "structures": structures,
        "role_method": ROLE_CLASSIFIER_METHOD,
        "role_version": ROLE_CLASSIFIER_VERSION,
    }


def needs_clarity(record_role: str, readability_flags: Sequence[str]) -> bool:
    """A record needs clarity review when flagged and not raw reference material."""

    if record_role == "reference":
        return False
    return any(flag in CLARITY_FLAGS for flag in readability_flags)


def _evidence_label(structures: Sequence[str], *, is_document: bool) -> str:
    order = ("code", "code_like_density", "commands", "config", "table", "diagram", "log_output", "raw_json")
    for key in order:
        if key in structures:
            if key == "code_like_density":
                return _EVIDENCE_LABELS["code"]
            return _EVIDENCE_LABELS[key]
    return _EVIDENCE_LABELS["document"] if is_document else _EVIDENCE_LABELS["raw"]


def _title_from_text(text: str, *, limit: int = 84) -> str:
    sentences = _sentences(text)
    first = sentences[0] if sentences else " ".join(text.split())
    first = _LEADING_BULLET.sub("", first)
    if len(first) <= limit:
        return first
    cut = first[:limit].rsplit(" ", 1)[0].rstrip(",;:·- ")
    return f"{cut}…"


def _summary_from_text(text: str, *, limit: int = 360) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    sentences = _sentences(collapsed)
    summary = ""
    for sentence in sentences:
        candidate = f"{summary} {sentence}".strip()
        if len(candidate) > limit:
            break
        summary = candidate
    if summary:
        return summary
    cut = collapsed[:limit].rsplit(" ", 1)[0].rstrip(",;:·- ")
    return f"{cut}…"


def _applies_when(memory: dict[str, Any], role: str) -> str:
    parts: list[str] = []
    scope = memory.get("scope")
    if isinstance(scope, str):
        try:
            scope = json.loads(scope)
        except (json.JSONDecodeError, ValueError):
            scope = {}
    if not isinstance(scope, dict) or not scope:
        try:
            scope = json.loads(str(memory.get("scope_json") or "{}"))
        except (json.JSONDecodeError, ValueError):
            scope = {}
        if not isinstance(scope, dict):
            scope = {}
    for key, value in sorted(scope.items()):
        parts.append(f"{key}: {value}")
    systems = _json_list(
        memory.get("applicable_systems")
        if memory.get("applicable_systems") is not None
        else memory.get("applicable_systems_json")
    )
    versions = _json_list(
        memory.get("applicable_versions")
        if memory.get("applicable_versions") is not None
        else memory.get("applicable_versions_json")
    )
    if systems:
        parts.append(f"systems: {', '.join(systems)}")
    if versions:
        parts.append(f"versions: {', '.join(versions)}")
    valid_from = str(memory.get("valid_from") or "").strip()
    valid_to = str(memory.get("valid_to") or "").strip()
    if valid_from or valid_to:
        parts.append(f"valid {valid_from or 'open'} → {valid_to or 'open'}")
    if role == "event":
        observed = str(memory.get("observed_at") or "").strip()
        if observed:
            parts.insert(0, f"observed {observed[:10]}")
    if parts:
        return "Applies when " + " · ".join(parts)
    if str(memory.get("context_mode") or "standalone") == "context_dependent":
        return "Context-dependent, but the stored scope is incomplete"
    return "Applies everywhere"


def _retention_reason(memory: dict[str, Any], role: str, has_active_dependencies: bool) -> str:
    source_category = str(memory.get("source_category") or "").upper()
    if role == "reference":
        if source_category in _DOCUMENT_SOURCE_CATEGORIES:
            return "Imported from your documents and retained as searchable raw reference evidence."
        return "Retained as raw reference evidence that can be searched when directly relevant."
    if role == "event":
        return "Retained as an episodic observation with bounded temporal value."
    if role == "claim":
        return (
            "Inferred without confirmed supporting evidence. It stays labeled as an unconfirmed claim "
            "until you review it or evidence links are added."
        )
    if source_category == "USER_EXPLICIT":
        return "You explicitly asked Kaya to remember this."
    if source_category == "TOOL_VERIFIED":
        return "Verified by an observed tool outcome."
    if source_category in {"AGENT_INFERENCE", "REFLECTION"} and has_active_dependencies:
        return "Inferred with preserved supporting evidence links."
    return "Retained as durable knowledge that passed the storage quality checks."


def build_presentation(
    memory: dict[str, Any],
    classification: dict[str, Any],
    *,
    has_active_dependencies: bool = False,
) -> dict[str, Any]:
    """Produce the derived display layer for one record.

    The output never replaces stored content, never raises confidence, and
    never invents a summary: when a faithful summary is impossible it uses a
    transparent evidence-type description instead.
    """

    role = str(classification.get("record_role") or "canonical")
    flags = list(classification.get("readability_flags") or [])
    body, vault_meta = record_parts(memory)
    structures = list(classification.get("structures") or detect_structures(body))
    is_document = bool(vault_meta) or (
        str(memory.get("source_category") or "").upper() in _DOCUMENT_SOURCE_CATEGORIES
    )

    if role == "reference":
        label = _evidence_label(structures, is_document=is_document)
        if vault_meta and vault_meta.get("title"):
            title = vault_meta["title"]
            section = vault_meta.get("section") or ""
            # Vault section headings are already breadcrumbs that can start
            # with the note title; avoid repeating it in the display title.
            if not section or section == title:
                display_title = title
            elif section.startswith(f"{title} › "):
                display_title = section
            else:
                display_title = f"{title} › {section}"
            section_leaf = section.split(" › ")[-1].strip()
            display_summary = (
                f"{label} from the vault note “{title}”"
                + (f", section “{section_leaf}”." if section_leaf and section_leaf != title else ".")
                + " Stored as raw evidence, not as a memory statement."
            )
        else:
            heading = str(memory.get("object_value") or "").strip()
            display_title = heading or _title_from_text(body)
            display_summary = f"{label} kept as searchable raw evidence, not as a memory statement."
    else:
        display_title = _title_from_text(body)
        display_summary = _summary_from_text(body)
        if _UNCERTAIN_LANGUAGE.search(body) and "uncertain_language" not in flags:
            flags.append("uncertain_language")

    # Temporal validity. A record can be relevant and still contain values that
    # were only true at capture ("Uptime 36.5 days", "as of 2026-08-07"). This
    # only ADDS flags and an as-of note to the display layer: it never rewrites
    # content, never changes the role, and never raises confidence. What to do
    # with the verdict is the caller's decision (see temporal.ACTION_*).
    temporal = classify_temporal(body)
    if temporal.volatile_score > 0:
        for flag in temporal.flags():
            if flag not in flags:
                flags.append(flag)

    applies_when = _applies_when(memory, role)
    if temporal.as_of and temporal.needs_stamp:
        applies_when = f"{applies_when.rstrip()} Values shown are as of {temporal.as_of}."

    return {
        "display_title": display_title or "Untitled record",
        "display_summary": display_summary or "No readable statement could be derived faithfully.",
        "applies_when": applies_when,
        "retention_reason": _retention_reason(memory, role, has_active_dependencies),
        "readability_flags": flags,
        "presentation_method": PRESENTATION_METHOD,
        "presentation_version": PRESENTATION_VERSION,
    }


def deterministic_split_preview(
    content: str,
    *,
    max_parts: int = 6,
    memory: dict[str, Any] | None = None,
) -> list[str]:
    """Split collapsed content into candidate standalone statements.

    Splits on bullet markers first, then sentence boundaries. The preview is a
    starting point for operator editing; it is never applied without
    confirmation.
    """

    body, _meta = record_parts({**(memory or {}), "content": content})
    collapsed = " ".join(body.split())
    if not collapsed:
        return []
    bullet_parts = [
        _LEADING_BULLET.sub("", part).strip()
        for part in _BULLET_SPLIT.split(collapsed)
        if part and part.strip()
    ]
    parts = [part for part in bullet_parts if len(part) >= 12]
    if len(parts) < 2:
        sentence_parts = [
            _LEADING_BULLET.sub("", sentence).strip() for sentence in _sentences(collapsed)
        ]
        parts = [part for part in sentence_parts if len(part) >= 12]
    if len(parts) < 2:
        return [collapsed]
    return parts[:max_parts]


def deterministic_rewrite_preview(content: str, *, memory: dict[str, Any] | None = None) -> str:
    """Propose a normalized starting point for an operator rewrite.

    The vault provenance prefix is removed and whitespace collapsed; the words
    themselves are preserved so negation, anchors, and uncertainty stay
    exactly as stored. The operator edits and confirms the final text.
    """

    body, _meta = record_parts({**(memory or {}), "content": content})
    return " ".join(body.split())
