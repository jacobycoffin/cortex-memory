"""Jev (TypeSafe System One) bridge for Cortex's judgment passes.

Jev answers *typed questions* (Noul / Choice / Score) over a supplied state and
returns probabilities, not prose. Cortex keeps every threshold, permission, and
side effect in code; the model only supplies calibrated semantic judgments.
This module is the single integration point, used by:

- ``autojudge.AutoJudge`` admission decisions when ``CORTEX_AUTO_JUDGE_ENGINE=jev``
- link suggestions for admitted candidates (``CORTEX_AUTO_JUDGE_JEV_LINKS=1``)
- the orphan linker when ``CORTEX_AUTO_JUDGE_LINK_ENGINE=jev``

Harness contract (see the Jev integration proposal / HARNESS-CONTRACT.md):

- Closed sets with escalations: every question set has a no-match outcome and
  code maps unknown/low-confidence answers to ``defer`` (admission) or *skip*
  (links). The model never triggers an action by probability alone.
- Fail direction per use: admission fails closed (no new memory), links fail
  toward creating nothing.
- Input hygiene: bounded content, no session ids or source refs, per-item
  isolation so one bad answer never sinks a batch.
- Batching: one state per call by default (the validated shape); larger batches
  switch to labeled per-candidate questions. Adaptive halving on oversize.
- Versioning: question sets are versioned constants; thresholds are calibrated
  on the recorded corpus (see ``ADMISSION_PATHS``), never invented defaults.
- Outcome logging: one JSONL line per judgment (question version, sanitized
  input id, answers + probabilities, chosen policy branch). No raw content.

The HTTP client is stdlib-only (urllib) so the plugin keeps zero dependencies;
the wire format matches ``typesafe-sdk`` (POST ``/v1/systemone`` with
``{"state", "model", "questions"}`` and an ``Authorization: Bearer`` header).

This module never writes to the Cortex database. Policy, permissions, and side
effects stay in the caller.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
DEFAULT_API_KEY_ENV = "TYPESAFE_API_KEY"
MODEL_PIN_HINT = "jev-1.13.0"

QUESTION_SET_VERSION = "cortex_admission.2026-09-17-v3"
LINK_QUESTION_SET_VERSION = "cortex_links.2026-09-18-v1"
POLICY_VERSION = "cortex_admission_policy.2026-09-18.1"

_MAX_STATE_BYTES = 262_144
_MAX_RESPONSE_BYTES = 1_000_000
_MAX_BATCH_SIZE = 12
_MIN_BATCH_SIZE = 1
_MAX_CONCURRENCY = 16
# Bounded content per provider record; mirrors autojudge._MAX_CANDIDATE_CONTENT_CHARS.
_MAX_CONTENT_CHARS = 1600
_MAX_LINK_CONTENT_CHARS = 300
_MAX_RETRY_SLEEP_SECONDS = 4.0

# Feedback-boost defaults; kept identical to AutoJudgeConfig so the engine's
# threshold arithmetic matches what the ledger reason text has always said.
DEFAULT_STRONG_FEEDBACK_BOOST = 0.08
DEFAULT_POSITIVE_FEEDBACK_BOOST = 0.02

ProviderCall = Callable[[str, str, dict[str, Any], float], dict[str, Any]]


class JevError(RuntimeError):
    """Raised when configuration or provider output cannot be trusted."""


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JevSettings:
    """Connection + policy knobs for every Jev use.

    ``from_env`` is the production constructor; direct construction (with
    ``decision_log=None``) is the test-friendly path and never writes logs.
    """

    endpoint: str = DEFAULT_ENDPOINT
    model: str = DEFAULT_MODEL
    api_key_env: str = DEFAULT_API_KEY_ENV
    credential_file: Path | None = None
    timeout_seconds: float = 30.0
    batch_size: int = _MIN_BATCH_SIZE
    concurrency: int = 8
    max_attempts: int = 3
    # Calibrated 2026-09-18 on 400 sampled pairs: at 0.65 Jev re-creates 91% of
    # operator edges and 84% of auto-judge edges, with 3% of random pairs above
    # the gate. See the link-quality replay in the Jev integration report.
    link_threshold: float = 0.65
    max_links_per_item: int = 3
    decision_log: Path | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "JevSettings":
        env = os.environ if environ is None else environ

        def _text(name: str, default: str) -> str:
            return str(env.get(name, default) or default).strip()

        def _int(name: str, default: int) -> int:
            raw = str(env.get(name, "") or "").strip()
            if not raw:
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        def _float(name: str, default: float) -> float:
            raw = str(env.get(name, "") or "").strip()
            if not raw:
                return default
            try:
                return float(raw)
            except ValueError:
                return default

        credential_file = _text(
            "CORTEX_JEV_CREDENTIAL_FILE",
            _text("CORTEX_AUTO_JUDGE_CREDENTIAL_FILE", ""),
        )
        log_raw = env.get("CORTEX_JEV_DECISION_LOG")
        if log_raw is None:
            decision_log: Path | None = Path.home() / ".hermes" / "cortex" / "jev-decisions.jsonl"
        else:
            log_raw = str(log_raw).strip()
            decision_log = Path(log_raw).expanduser() if log_raw else None
        return cls(
            endpoint=_text("CORTEX_JEV_ENDPOINT", DEFAULT_ENDPOINT),
            model=_text("CORTEX_JEV_MODEL", DEFAULT_MODEL),
            api_key_env=_text("CORTEX_JEV_API_KEY_ENV", DEFAULT_API_KEY_ENV),
            credential_file=Path(credential_file).expanduser() if credential_file else None,
            timeout_seconds=_float("CORTEX_JEV_TIMEOUT_SECONDS", 30.0),
            batch_size=_int("CORTEX_JEV_BATCH_SIZE", _MIN_BATCH_SIZE),
            concurrency=_int("CORTEX_JEV_CONCURRENCY", 8),
            max_attempts=_int("CORTEX_JEV_MAX_ATTEMPTS", 3),
            link_threshold=_float("CORTEX_JEV_LINK_THRESHOLD", 0.65),
            max_links_per_item=_int("CORTEX_JEV_MAX_LINKS_PER_ITEM", 3),
            decision_log=decision_log,
        )

    def validate(self) -> None:
        if not isinstance(self.endpoint, str) or not self.endpoint:
            raise JevError("jev endpoint must be a non-empty string")
        parsed = urllib.parse.urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise JevError("jev endpoint must be an absolute HTTP(S) URL")
        if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise JevError("plain HTTP jev endpoints are limited to loopback")
        if not isinstance(self.model, str) or not self.model or len(self.model) > 200:
            raise JevError("jev model must be a bounded non-empty string")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)):
            raise JevError("jev timeout must be numeric")
        if not math.isfinite(self.timeout_seconds) or not 1.0 <= float(self.timeout_seconds) <= 120.0:
            raise JevError("jev timeout must be between 1 and 120 seconds")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise JevError("jev batch size must be an integer")
        if not _MIN_BATCH_SIZE <= self.batch_size <= _MAX_BATCH_SIZE:
            raise JevError(f"jev batch size must be between {_MIN_BATCH_SIZE} and {_MAX_BATCH_SIZE}")
        if isinstance(self.concurrency, bool) or not isinstance(self.concurrency, int):
            raise JevError("jev concurrency must be an integer")
        if not 1 <= self.concurrency <= _MAX_CONCURRENCY:
            raise JevError(f"jev concurrency must be between 1 and {_MAX_CONCURRENCY}")
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int):
            raise JevError("jev max attempts must be an integer")
        if not 1 <= self.max_attempts <= 5:
            raise JevError("jev max attempts must be between 1 and 5")
        if isinstance(self.link_threshold, bool) or not isinstance(self.link_threshold, (int, float)):
            raise JevError("jev link threshold must be numeric")
        if not math.isfinite(float(self.link_threshold)) or not 0.0 <= float(self.link_threshold) <= 1.0:
            raise JevError("jev link threshold must be between 0.0 and 1.0")
        if isinstance(self.max_links_per_item, bool) or not isinstance(self.max_links_per_item, int):
            raise JevError("jev max links per item must be an integer")
        if not 1 <= self.max_links_per_item <= 20:
            raise JevError("jev max links per item must be between 1 and 20")

    def api_key(self) -> str:
        if self.api_key_env:
            direct = os.environ.get(self.api_key_env, "").strip()
            if direct:
                return direct
            if self.credential_file is not None:
                return _read_named_credential(self.credential_file, self.api_key_env)
        return ""


def _read_named_credential(path: Path, key: str) -> str:
    """Read ``KEY=value`` from a dotenv-style file without importing the plugin."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        name, _, value = line.partition("=")
        if name.strip() != key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        return value.strip()
    return ""


# ---------------------------------------------------------------------------
# Question sets (versioned; bump the version string when instructions change)
# ---------------------------------------------------------------------------


def admission_questions(ref: str = "candidate") -> dict[str, dict[str, Any]]:
    """The v3 admission set, validated 2026-09-17/18 at 93.6% binary agreement.

    ``ref`` names the candidate inside the state (``"candidate"`` for the
    single-candidate shape; ``"candidate c2"`` for labeled batches).
    """
    return {
        "worth_saving": {
            "type": "noul",
            "instructions": (
                f"Judge the staged candidate for long-term memory admission "
                f"(`{ref}.content` with its metadata and pipeline pre-assessment). "
                "Should it be admitted as a durable, reusable memory?"
            ),
            "criteria": {
                "true": "Reusable, well-scoped, likely to help again in future sessions",
                "false": "Transient, noisy, redundant, or not safe for broad recall",
            },
        },
        "durable": {
            "type": "noul",
            "instructions": (
                f"Will `{ref}.content` still be relevant months from now, "
                "rather than being a one-off exchange?"
            ),
        },
        "standalone": {
            "type": "noul",
            "instructions": (
                f"Is the {ref} understandable on its own, without the surrounding conversation?"
            ),
        },
        "useful_again": {
            "type": "noul",
            "instructions": f"Is this {ref} likely to help again in future sessions?",
        },
        "scope_clear": {
            "type": "noul",
            "instructions": (
                f"Is it clear from the {ref} alone what or whom it applies to, "
                "so it can be safely recalled later without the original conversation?"
            ),
        },
        "curated": {
            "type": "noul",
            "instructions": (
                f"Is this a curated, self-contained fact, decision, or preference "
                "rather than a raw log, digest, or harvest artifact?"
            ),
        },
        "duplicate_of_related": {
            "type": "noul",
            "instructions": (
                f"Is the {ref} already covered by one of the related memories, "
                "such that it adds no new durable information?"
            ),
        },
    }


_LINK_RELATION_CRITERIA: dict[str, str] = {
    "supports": "reinforces or is consistent with the related memory",
    "extends": "adds detail, scope, or depth to the related memory",
    "refines": "corrects or narrows the related memory",
    "example_of": "concrete instance of the broader concept in the related memory",
    "generalizes": "broader rule or pattern that covers the related memory",
    "prerequisite": "should be understood before the related memory",
    "contradicts": "conflicts with or supersedes the related memory",
    "none": "no meaningful connection, or only surface/coincidental similarity",
}


def link_questions(related_count: int, ref: str = "candidate") -> dict[str, dict[str, Any]]:
    """Pairwise link questions: one gate + one relation per related memory."""
    questions: dict[str, dict[str, Any]] = {}
    for index in range(1, related_count + 1):
        label = f"related_{index}"
        questions[f"l{index}"] = {
            "type": "noul",
            "instructions": (
                f"Should the {ref} be linked to `{label}` as a meaningful connection "
                "in the knowledge graph? Only answer true for durable, useful "
                "connections; false for coincidental similarity."
            ),
            "criteria": {
                "true": "a meaningful, durable connection worth recording as a graph edge",
                "false": "no meaningful connection, or only surface/coincidental similarity",
            },
        }
        questions[f"r{index}"] = {
            "type": "choice",
            "instructions": (
                f"If the {ref} is linked to `{label}`, which relation best describes "
                f"the connection from the {ref} to `{label}`?"
            ),
            "criteria": dict(_LINK_RELATION_CRITERIA),
        }
    return questions


# ---------------------------------------------------------------------------
# Policy: thresholds per category, calibrated on the September corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdmitPath:
    """One calibrated policy path. ``None`` thresholds never auto-decide."""

    label: str
    admit: float | None
    reject: float | None
    audit: bool = False


DEFAULT_ADMIT_PATH = AdmitPath(label="default_p1", admit=0.60, reject=0.40)
# Path selection order: source_type, then kind, then the default.
# - ``builtin_memory`` measured 72.7% agreement and is not separable by
#   threshold (disagreements interleave at every confidence): during the canary
#   every builtin_memory candidate defers to review, flagged for audit.
# - ``semantic`` is the weakest kind (88.3%); the stricter knee (90.8% @ 35%
#   defer) defers more and is flagged for audit sampling.
ADMISSION_PATHS: dict[str, AdmitPath] = {
    "builtin_memory": AdmitPath(label="builtin_memory_review", admit=None, reject=None, audit=True),
    "semantic": AdmitPath(label="semantic_strict", admit=0.65, reject=0.35, audit=True),
}


def admission_path_for(kind: str, source_type: str) -> AdmitPath:
    key_source = str(source_type or "").casefold()
    if key_source in ADMISSION_PATHS:
        return ADMISSION_PATHS[key_source]
    key_kind = str(kind or "").casefold()
    if key_kind in ADMISSION_PATHS:
        return ADMISSION_PATHS[key_kind]
    return DEFAULT_ADMIT_PATH


def map_admission(
    *,
    worth_saving: float | None,
    boost: float,
    path: AdmitPath,
) -> str:
    """Map the calibrated probability to remember / reject / defer.

    The feedback boost shifts the decision toward keeping, exactly like the
    chat engine's confidence adjustment: it raises the effective value used for
    both gates, making rejection harder and admission easier. Ambiguous values
    defer (the harness contract's escalation).
    """
    if worth_saving is None or not math.isfinite(worth_saving):
        return "defer"
    effective = min(1.0, max(0.0, float(worth_saving) + max(0.0, float(boost))))
    if path.admit is not None and effective >= path.admit:
        return "remember"
    if path.reject is not None and effective <= path.reject:
        return "reject"
    return "defer"


# ---------------------------------------------------------------------------
# HTTP client (stdio-only; mirrors typesafe-sdk's wire format)
# ---------------------------------------------------------------------------


def systemone_call(
    settings: JevSettings,
    state: Any,
    questions: dict[str, Any],
    *,
    call: ProviderCall | None = None,
) -> dict[str, Any]:
    """POST one system-one request with bounded retries on transient errors.

    ``call`` is an injection point for tests; when provided it replaces the
    HTTP call entirely and must return the parsed response dict.
    """
    api_key = settings.api_key()
    payload: dict[str, Any] = {"state": state, "model": settings.model, "questions": questions}
    body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    if len(body) > _MAX_STATE_BYTES:
        raise JevError("jev request exceeded the size limit")
    if call is not None:
        return call(settings.endpoint, api_key, payload, settings.timeout_seconds)
    parsed = urllib.parse.urlparse(settings.endpoint)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"} and not api_key:
        raise JevError(f"jev credential {settings.api_key_env or '<unset>'} is unavailable")

    attempts = max(1, int(settings.max_attempts))
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            settings.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "cortex-jev/1.0",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(settings.timeout_seconds)) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            if exc.code in {429, 500, 502, 503, 504, 529}:
                last_error = JevError(f"jev provider returned HTTP {exc.code}")
                _sleep_backoff(attempt)
                continue
            raise JevError(f"jev provider returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = JevError(f"jev provider request failed: {type(exc).__name__}")
            _sleep_backoff(attempt)
            continue
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise JevError("jev provider response exceeded the size limit")
        try:
            result = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise JevError("jev provider returned a non-JSON response") from exc
        if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
            raise JevError("jev provider response is missing the answers object")
        return result
    raise last_error or JevError("jev provider request failed")


def _sleep_backoff(attempt: int) -> None:
    time.sleep(min(_MAX_RETRY_SLEEP_SECONDS, 0.75 * (2**attempt)))


def _answers_as_floats(response: dict[str, Any]) -> dict[str, float | None]:
    """Flatten noul answers to floats; non-noul or malformed entries become None."""
    out: dict[str, float | None] = {}
    answers = response.get("answers") or {}
    if not isinstance(answers, dict):
        return out
    for qid, answer in answers.items():
        value: float | None = None
        if isinstance(answer, dict):
            raw = answer.get("noul")
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                candidate = float(raw)
                if math.isfinite(candidate) and 0.0 <= candidate <= 1.0:
                    value = candidate
        out[str(qid)] = value
    return out


def _choice_for(response: dict[str, Any], qid: str) -> tuple[str | None, float | None]:
    answers = response.get("answers") or {}
    if not isinstance(answers, dict):
        return None, None
    answer = answers.get(qid)
    if not isinstance(answer, dict):
        return None, None
    choice = answer.get("choice")
    confidence = answer.get("confidence")
    choice_value = str(choice).strip().casefold() if isinstance(choice, str) else None
    confidence_value = (
        float(confidence)
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
        else None
    )
    return choice_value, confidence_value


# ---------------------------------------------------------------------------
# Admission engine
# ---------------------------------------------------------------------------


def _bounded_record(record: Mapping[str, Any], ref: str) -> dict[str, Any]:
    """Project a provider candidate record into a bounded state document."""
    projected: dict[str, Any] = {
        "proposal_id": str(record.get("proposal_id") or ""),
        "content": str(record.get("content") or "")[:_MAX_CONTENT_CHARS],
        "kind": str(record.get("kind") or "semantic"),
        "source_type": str(record.get("source_type") or "unknown"),
        "source_category": str(record.get("source_category") or "AGENT_PROPOSED"),
        "context_mode": str(record.get("context_mode") or "standalone"),
        "first_seen_at": str(record.get("first_seen_at") or ""),
        "quarantined": bool(record.get("quarantined")),
        "redacted": bool(record.get("redacted")),
        "recurrence_count": int(record.get("recurrence_count") or 1),
    }
    for key in ("scope", "preconditions", "entities", "applicable_systems", "applicable_versions"):
        value = record.get(key)
        if value:
            projected[key] = value
    assessment = record.get("assessment")
    if isinstance(assessment, dict) and assessment:
        projected["assessment"] = assessment
    related = record.get("related_memories")
    if isinstance(related, list) and related:
        projected["related_memories"] = related
    projected["ref"] = ref
    return projected


def _state_for_batch(batch: list[Mapping[str, Any]]) -> tuple[Any, list[str], bool]:
    """Project one call's batch: single-candidate shape for size 1, labeled otherwise."""
    if len(batch) == 1:
        return {"candidate": _bounded_record(batch[0], "candidate")}, ["candidate"], False
    docs: dict[str, Any] = {}
    refs: list[str] = []
    for index, record in enumerate(batch, start=1):
        ref = f"candidate c{index}"
        refs.append(ref)
        docs[f"c{index}"] = _bounded_record(record, ref)
    return {"candidates": docs}, refs, True


def _batch_questions(refs: list[str], batched: bool) -> dict[str, dict[str, Any]]:
    questions: dict[str, dict[str, Any]] = {}
    for index, ref in enumerate(refs, start=1):
        prefix = f"c{index}_" if batched else ""
        for qid, question in admission_questions(ref).items():
            questions[f"{prefix}{qid}"] = question
    return questions


def judge_admission_batch(
    settings: JevSettings,
    records: Sequence[Mapping[str, Any]],
    *,
    strong_boost: float = DEFAULT_STRONG_FEEDBACK_BOOST,
    positive_boost: float = DEFAULT_POSITIVE_FEEDBACK_BOOST,
    call: ProviderCall | None = None,
    run_ref: str | None = None,
    log_decisions: bool = True,
) -> list[dict[str, Any]]:
    """Judge up to ``len(records)`` candidates; returns one decision per record.

    Batches run concurrently; each batch is isolated (a failed call degrades to
    per-record ``defer`` only when ``on_error`` is not raised — callers choose).
    Provider errors raise :class:`JevError` after retries; the caller decides
    between fallback and defer-all.
    """
    settings.validate()
    records = list(records)
    if not records:
        return []
    batch_size = max(_MIN_BATCH_SIZE, min(settings.batch_size, _MAX_BATCH_SIZE))
    batches: list[list[Mapping[str, Any]]] = [
        list(records[start : start + batch_size]) for start in range(0, len(records), batch_size)
    ]
    run_ref = run_ref or uuid.uuid4().hex[:12]
    results: list[dict[str, Any] | None] = [None] * len(records)
    log_lines: list[dict[str, Any]] = []

    def _one_batch(batch_start: int, batch: list[Mapping[str, Any]]) -> None:
        state, refs, batched = _state_for_batch(batch)
        questions = _batch_questions(refs, batched)
        started = time.perf_counter()
        response = systemone_call(settings, state, questions, call=call)
        latency_ms = int((time.perf_counter() - started) * 1000)
        model = str(response.get("model") or settings.model)
        floats = _answers_as_floats(response)
        for offset, record in enumerate(batch):
            index = batch_start + offset
            prefix = f"c{offset + 1}_" if batched else ""
            answers = {
                qid: floats.get(f"{prefix}{qid}")
                for qid in (
                    "worth_saving",
                    "durable",
                    "standalone",
                    "useful_again",
                    "scope_clear",
                    "curated",
                    "duplicate_of_related",
                )
            }
            result = decide_one(
                record,
                answers,
                settings=settings,
                strong_boost=strong_boost,
                positive_boost=positive_boost,
                model_name=model,
            )
            usage = {
                str(key): float(value)
                for key, value in (response.get("usage") or {}).items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            result.update(
                {"model": model, "latency_ms": latency_ms, "run_ref": run_ref, "usage": usage}
            )
            results[index] = result
            if log_decisions:
                log_lines.append(_decision_log_line(result, record, run_ref, model, latency_ms))

    workers = max(1, min(settings.concurrency, len(batches)))
    if workers == 1:
        for batch_index, batch in enumerate(batches):
            _one_batch(batch_index * batch_size, batch)
    else:
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_one_batch, batch_index * batch_size, batch)
                for batch_index, batch in enumerate(batches)
            ]
            for future in futures:
                future.result()  # raise the first provider error, fail closed
    if log_decisions and settings.decision_log is not None and log_lines:
        _write_log_lines(settings.decision_log, log_lines)
    return [item for item in results if item is not None]


def decide_one(
    record: Mapping[str, Any],
    answers: dict[str, float | None],
    *,
    settings: JevSettings,
    strong_boost: float,
    positive_boost: float,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Map one candidate's answers to a decision via the calibrated policy."""
    worth_saving = answers.get("worth_saving")
    strong_count = int(record.get("strong_feedback_count") or 0)
    positive_count = int(record.get("positive_feedback_count") or 0)
    ordinary_count = max(0, positive_count - strong_count)
    boost = strong_count * float(strong_boost) + ordinary_count * float(positive_boost)
    path = admission_path_for(str(record.get("kind") or ""), str(record.get("source_type") or ""))
    action = map_admission(worth_saving=worth_saving, boost=boost, path=path)
    effective = (
        None
        if worth_saving is None
        else min(1.0, max(0.0, float(worth_saving) + max(0.0, boost)))
    )
    annotations = []
    duplicate = answers.get("duplicate_of_related")
    scope_clear = answers.get("scope_clear")
    if duplicate is not None and duplicate >= 0.60:
        annotations.append(f"duplicate_of_related {duplicate:.2f}")
    if scope_clear is not None and scope_clear <= 0.40:
        annotations.append(f"scope_clear {scope_clear:.2f}")
    reason_bits = []
    if worth_saving is None:
        reason_bits.append("worth_saving unavailable")
    else:
        reason_bits.append(f"worth_saving {worth_saving:.2f}")
        reason_bits.append(f"feedback-adjusted {effective:.2f}")
    reason_bits.append(f"positive feedback {positive_count}, strong feedback {strong_count}")
    reason_bits.append(f"path {path.label}")
    if annotations:
        reason_bits.append("notes " + "; ".join(annotations))
    model = str(model_name or settings.model)
    reason = f"Automatic LLM judgment ({model}); " + ", ".join(reason_bits) + "."
    return {
        "proposal_id": str(record.get("proposal_id") or ""),
        "action": action,
        "confidence": float(worth_saving) if worth_saving is not None else 0.0,
        "worth_adjusted": effective,
        "reason": reason,
        "gate_applied": True,
        "audit": bool(path.audit),
        "path": path.label,
        "answers": answers,
    }


def _decision_log_line(
    result: dict[str, Any],
    record: Mapping[str, Any],
    run_ref: str,
    model: str,
    latency_ms: int,
) -> dict[str, Any]:
    answers = result.get("answers") or {}
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "admission",
        "question_set": QUESTION_SET_VERSION,
        "policy": POLICY_VERSION,
        "run_ref": run_ref,
        "proposal_id": result.get("proposal_id"),
        "kind": str(record.get("kind") or ""),
        "source_type": str(record.get("source_type") or ""),
        "path": result.get("path"),
        "worth_saving": answers.get("worth_saving"),
        "durable": answers.get("durable"),
        "standalone": answers.get("standalone"),
        "useful_again": answers.get("useful_again"),
        "scope_clear": answers.get("scope_clear"),
        "curated": answers.get("curated"),
        "duplicate_of_related": answers.get("duplicate_of_related"),
        "worth_adjusted": result.get("worth_adjusted"),
        "action": result.get("action"),
        "audit": bool(result.get("audit")),
        "gate": "policy" if result.get("gate_applied") else "confidence",
        "model": model,
        "latency_ms": latency_ms,
    }


def _write_log_lines(path: Path, lines: list[dict[str, Any]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n")
    except OSError as exc:  # logging must never break the pass
        logger.warning("jev decision log write failed: %s", exc)


# ---------------------------------------------------------------------------
# Link judgments
# ---------------------------------------------------------------------------


def judge_links(
    settings: JevSettings,
    *,
    candidate: Mapping[str, Any],
    related: Sequence[Mapping[str, Any]],
    threshold: float | None = None,
    max_links: int | None = None,
    call: ProviderCall | None = None,
    log_context: str = "",
) -> list[dict[str, Any]]:
    """Judge pairwise links from *candidate* to each of *related*.

    Returns bounded link suggestions ``{memory_id, relation, rationale,
    link_probability}`` for pairs whose gate clears ``threshold`` and whose
    relation is not ``none``. The caller still owns edge creation.
    """
    settings.validate()
    bounded_related = list(related)[:20]
    if not bounded_related:
        return []
    gate = float(settings.link_threshold if threshold is None else threshold)
    cap = int(settings.max_links_per_item if max_links is None else max_links)
    state = {
        "candidate": {
            "content": str(candidate.get("content") or "")[:_MAX_LINK_CONTENT_CHARS],
            "kind": str(candidate.get("kind") or "semantic"),
        },
        "related": [
            {
                "label": f"related_{index}",
                "content": str(item.get("content") or "")[:_MAX_LINK_CONTENT_CHARS],
                "kind": str(item.get("kind") or "semantic"),
            }
            for index, item in enumerate(bounded_related, start=1)
        ],
    }
    started = time.perf_counter()
    response = systemone_call(settings, state, link_questions(len(bounded_related)), call=call)
    latency_ms = int((time.perf_counter() - started) * 1000)
    model = str(response.get("model") or settings.model)
    floats = _answers_as_floats(response)
    suggestions: list[dict[str, Any]] = []
    for index, item in enumerate(bounded_related, start=1):
        target_id = str(item.get("memory_id") or "").strip()
        if not target_id:
            continue
        probability = floats.get(f"l{index}")
        relation, relation_confidence = _choice_for(response, f"r{index}")
        if probability is None or probability < gate:
            continue
        if relation is None or relation == "none" or relation not in _LINK_RELATION_CRITERIA:
            continue
        suggestions.append(
            {
                "memory_id": target_id,
                "relation": relation,
                "link_probability": round(float(probability), 4),
                "relation_confidence": relation_confidence,
                "rationale": (
                    f"jev link {probability:.2f} "
                    f"({relation} {relation_confidence if relation_confidence is not None else 0:.2f})"
                ),
            }
        )
    suggestions.sort(key=lambda item: item["link_probability"], reverse=True)
    suggestions = suggestions[: max(1, cap)]
    if log_context and settings.decision_log is not None:
        _write_log_lines(
            settings.decision_log,
            [
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": "links",
                    "question_set": LINK_QUESTION_SET_VERSION,
                    "context": log_context,
                    "candidate_id": str(candidate.get("memory_id") or ""),
                    "related_ids": [str(item.get("memory_id") or "") for item in bounded_related],
                    "links": [
                        {"memory_id": s["memory_id"], "relation": s["relation"], "p": s["link_probability"]}
                        for s in suggestions
                    ],
                    "model": model,
                    "latency_ms": latency_ms,
                }
            ],
        )
    return suggestions
