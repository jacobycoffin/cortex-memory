"""SQLite persistence for Cortex adaptive memory."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from .refinery import (
    CLARITY_FLAGS,
    LEGACY_ROLE_METHOD,
    OPERATOR_ROLE_METHOD,
    PRESENTATION_METHOD,
    PRESENTATION_VERSION,
    RECORD_ROLES,
    ROLE_CLASSIFIER_VERSION,
    build_presentation,
    classify_record_role,
    deterministic_rewrite_preview,
    deterministic_split_preview,
    needs_clarity,
)
from .security import normalize_text, sanitize_memory
from .semantics import feature_similarity, semantic_features


SCHEMA_VERSION = 32


logger = logging.getLogger(__name__)

REFINERY_BACKFILL_KEY = "refinery_backfill_version"
REFINERY_BACKFILL_VERSION = f"{ROLE_CLASSIFIER_VERSION}:{PRESENTATION_VERSION}"
POLICY_MIN_SUPPORT = 5
POLICY_MIN_CONSISTENCY = 0.80
POLICY_SHADOW_MIN_OBSERVATIONS = 3
POLICY_CORE_MIN_SUPPORT = 15
POLICY_CORE_MIN_CONTEXTS = 3
DYNAMIC_NEIGHBORHOOD_MIN_MEMORIES = 5
DYNAMIC_NEIGHBORHOOD_MIN_CONTEXTS = 2
_INACTIVE_PROJECT_STATUSES = {"abandoned", "cancelled", "canceled", "inactive", "paused", "stopped"}
_TOKEN = re.compile(r"[\w'-]{2,}", re.UNICODE)
_UNRESOLVED_REFERENCE = re.compile(
    r"^\s*(?:this|that|it|they|he|she|those|these)\b|"
    r"\b(?:the above|the previous one|same as before|as discussed|over there|this one|that one)\b",
    re.I,
)
_TOOL_TELEMETRY_SOURCE_TYPES = {
    "tool_execution",
    "tool_outcome_aggregation",
    "tool_workflow_aggregation",
}
_AUTOMATION_NOISE = re.compile(
    r"^tool execution observation:|"
    r"\bexpected argument keys\b|"
    r"^reinforced [a-z0-9_-]+ workflow:|"
    r"\b(?:task|command|tool|step) (?:completed|succeeded|failed)(?: successfully)?[.!]?$|"
    r"\b(?:exit code|status code)\s*[=:]?\s*0\b",
    re.I,
)
_PLACEHOLDER_CONTENT = re.compile(
    r"\b(?:add detail here|tbd|todo|placeholder|fill this in|not yet documented)\b",
    re.I,
)
_STOP = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def utc_now_dt() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso8601(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def content_hash(content: str) -> str:
    return hashlib.sha256(normalize_text(content).casefold().encode("utf-8")).hexdigest()


def query_tokens(text: str) -> list[str]:
    tokens = [t.casefold().strip("'-") for t in _TOKEN.findall(text or "")]
    return list(dict.fromkeys(t for t in tokens if t and t not in _STOP))[:24]


class StaleCreationProposalError(ValueError):
    """Raised when a creation proposal changed after a reviewer observed it."""


class CortexStore:
    """Thread-safe, auditable SQLite store with explicit lifecycle states."""

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._local_retrieval_revision = 0
        self._active_policy_cache: list[dict[str, Any]] | None = None
        self._conn.create_function("cortex_bump_revision", 0, self._bump_local_retrieval_revision)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._create_schema()
            self._create_connection_revision_triggers()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            DROP TRIGGER IF EXISTS cortex_revision_memory_insert;
            DROP TRIGGER IF EXISTS cortex_revision_memory_delete;
            DROP TRIGGER IF EXISTS cortex_revision_memory_material_update;
            DROP TRIGGER IF EXISTS cortex_revision_edge_insert;
            DROP TRIGGER IF EXISTS cortex_revision_edge_update;
            DROP TRIGGER IF EXISTS cortex_revision_edge_delete;
            DROP TRIGGER IF EXISTS cortex_revision_tool_stats_insert;
            DROP TRIGGER IF EXISTS cortex_revision_tool_stats_update;
            DROP TRIGGER IF EXISTS cortex_revision_tool_stats_delete;
            DROP TRIGGER IF EXISTS cortex_revision_workflow_stats_insert;
            DROP TRIGGER IF EXISTS cortex_revision_workflow_stats_update;
            DROP TRIGGER IF EXISTS cortex_revision_workflow_stats_delete;

            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL DEFAULT 'semantic',
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'conversation',
                source_category TEXT NOT NULL DEFAULT 'AGENT_INFERENCE',
                origin_source_category TEXT NOT NULL DEFAULT 'AGENT_INFERENCE',
                approval_state TEXT NOT NULL DEFAULT 'unreviewed',
                source_ref TEXT,
                session_id TEXT,
                context_mode TEXT NOT NULL DEFAULT 'standalone',
                scope_json TEXT NOT NULL DEFAULT '{}',
                entities_json TEXT NOT NULL DEFAULT '[]',
                preconditions_json TEXT NOT NULL DEFAULT '{}',
                source_context TEXT,
                applicable_systems_json TEXT NOT NULL DEFAULT '[]',
                applicable_versions_json TEXT NOT NULL DEFAULT '[]',
                metadata_completeness REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                subject TEXT,
                predicate TEXT,
                object_value TEXT,
                extraction_method TEXT NOT NULL DEFAULT 'unknown',
                confidence REAL NOT NULL DEFAULT 0.6,
                currentness_confidence REAL NOT NULL DEFAULT 0.7,
                importance REAL NOT NULL DEFAULT 0.5,
                uniqueness REAL NOT NULL DEFAULT 1.0,
                volatility REAL NOT NULL DEFAULT 0.4,
                trust REAL NOT NULL DEFAULT 0.7,
                record_role TEXT NOT NULL DEFAULT 'canonical',
                role_method TEXT NOT NULL DEFAULT 'legacy_default',
                role_version TEXT,
                role_reviewed_at TEXT,
                state TEXT NOT NULL DEFAULT 'active',
                stranded INTEGER NOT NULL DEFAULT 0,
                stranded_reason TEXT,
                stranded_at TEXT,
                pinned INTEGER NOT NULL DEFAULT 0,
                protected INTEGER NOT NULL DEFAULT 0,
                dirty INTEGER NOT NULL DEFAULT 0,
                dirty_reason TEXT,
                supersedes_id TEXT REFERENCES memories(id),
                quarantine_reason TEXT,
                retrieved_count INTEGER NOT NULL DEFAULT 0,
                selected_count INTEGER NOT NULL DEFAULT 0,
                injected_count INTEGER NOT NULL DEFAULT 0,
                used_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                confirmed_count INTEGER NOT NULL DEFAULT 0,
                validated_count INTEGER NOT NULL DEFAULT 0,
                helpful_count INTEGER NOT NULL DEFAULT 0,
                harmful_count INTEGER NOT NULL DEFAULT 0,
                correction_count INTEGER NOT NULL DEFAULT 0,
                false_positive_count INTEGER NOT NULL DEFAULT 0,
                duplicate_count INTEGER NOT NULL DEFAULT 0,
                last_retrieved_at TEXT,
                last_injected_at TEXT,
                last_used_at TEXT,
                last_helpful_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash);
            CREATE INDEX IF NOT EXISTS idx_memories_state ON memories(state, pinned, updated_at);
            CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);

            CREATE TABLE IF NOT EXISTS memory_versions (
                version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                confidence REAL NOT NULL,
                state TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                system_from TEXT NOT NULL,
                system_to TEXT,
                reason TEXT,
                source_ref TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_versions_memory ON memory_versions(memory_id, system_from);

            CREATE TABLE IF NOT EXISTS edges (
                src_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                dst_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                relation TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 0.25,
                evidence_count INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                last_reinforced_at TEXT NOT NULL,
                PRIMARY KEY (src_id, dst_id, relation),
                CHECK (src_id <> dst_id)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id, relation);

            CREATE TABLE IF NOT EXISTS edge_evidence (
                evidence_id TEXT PRIMARY KEY,
                src_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                dst_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                relation TEXT NOT NULL,
                evidence_type TEXT NOT NULL,
                evidence_key TEXT NOT NULL,
                summary TEXT NOT NULL,
                source_ref TEXT,
                task_id TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                UNIQUE(src_id,dst_id,relation,evidence_type,evidence_key),
                CHECK(src_id <> dst_id)
            );
            CREATE INDEX IF NOT EXISTS idx_edge_evidence_edge
              ON edge_evidence(src_id,dst_id,relation,created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_edge_evidence_task
              ON edge_evidence(task_id,created_at DESC);

            CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                event TEXT NOT NULL,
                query TEXT,
                session_id TEXT,
                score REAL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_access_memory ON access_log(memory_id, created_at);

            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_content TEXT NOT NULL,
                assistant_content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                content_hash TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS usage_records (
                usage_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                session_id TEXT,
                query TEXT,
                score REAL,
                selected INTEGER NOT NULL DEFAULT 1,
                used INTEGER NOT NULL DEFAULT 0,
                attribution REAL NOT NULL DEFAULT 0.0,
                outcome TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                resolved_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_usage_task ON usage_records(task_id, outcome);
            CREATE INDEX IF NOT EXISTS idx_usage_memory ON usage_records(memory_id, created_at);

            CREATE TABLE IF NOT EXISTS memory_dependencies (
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                evidence_id TEXT NOT NULL REFERENCES memories(id) ON DELETE RESTRICT,
                relation TEXT NOT NULL DEFAULT 'derived_from',
                weight REAL NOT NULL DEFAULT 1.0,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                PRIMARY KEY(memory_id, evidence_id, relation),
                CHECK(memory_id <> evidence_id)
            );
            CREATE INDEX IF NOT EXISTS idx_dependencies_evidence ON memory_dependencies(evidence_id, active);

            CREATE TABLE IF NOT EXISTS tool_executions (
                execution_id TEXT PRIMARY KEY,
                session_id TEXT,
                task_type TEXT NOT NULL,
                task_context TEXT,
                tool_name TEXT NOT NULL,
                argument_keys TEXT NOT NULL,
                success INTEGER NOT NULL,
                error_type TEXT,
                result_summary TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tool_exec_lookup ON tool_executions(task_type, tool_name, success, created_at);

            CREATE TABLE IF NOT EXISTS tool_stats (
                task_type TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                helpful_count INTEGER NOT NULL DEFAULT 0,
                harmful_count INTEGER NOT NULL DEFAULT 0,
                argument_keys TEXT NOT NULL DEFAULT '[]',
                last_success_at TEXT,
                last_failure_at TEXT,
                last_error_type TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(task_type, tool_name)
            );

            CREATE TABLE IF NOT EXISTS tool_workflows (
                workflow_id TEXT PRIMARY KEY,
                session_id TEXT,
                task_type TEXT NOT NULL,
                task_fingerprint TEXT NOT NULL,
                task_context TEXT,
                steps_json TEXT NOT NULL,
                workflow_key TEXT NOT NULL,
                success INTEGER NOT NULL,
                error_type TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tool_workflows_lookup
              ON tool_workflows(task_type,task_fingerprint,success,created_at);

            CREATE TABLE IF NOT EXISTS tool_workflow_stats (
                task_type TEXT NOT NULL,
                task_fingerprint TEXT NOT NULL,
                workflow_key TEXT NOT NULL,
                steps_json TEXT NOT NULL,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                distinct_tasks INTEGER NOT NULL DEFAULT 0,
                last_error_type TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(task_type,task_fingerprint,workflow_key)
            );

            CREATE TABLE IF NOT EXISTS document_sources (
                source_path TEXT PRIMARY KEY,
                source_type TEXT NOT NULL DEFAULT 'obsidian_markdown',
                title TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                modified_at TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                imported_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_document_sources_status
              ON document_sources(status,last_seen_at);

            CREATE TABLE IF NOT EXISTS document_chunks (
                source_path TEXT NOT NULL REFERENCES document_sources(source_path) ON DELETE CASCADE,
                chunk_key TEXT NOT NULL,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE RESTRICT,
                chunk_hash TEXT NOT NULL,
                heading TEXT,
                ordinal INTEGER NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                imported_at TEXT NOT NULL,
                PRIMARY KEY(source_path,chunk_key)
            );
            CREATE INDEX IF NOT EXISTS idx_document_chunks_memory
              ON document_chunks(memory_id,active);

            CREATE TABLE IF NOT EXISTS maintenance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                details TEXT NOT NULL,
                dry_run INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS recall_runs (
                recall_id TEXT PRIMARY KEY,
                task_id TEXT,
                session_id TEXT,
                query TEXT,
                mode TEXT NOT NULL,
                reason TEXT,
                requested_limit INTEGER NOT NULL,
                token_budget INTEGER NOT NULL,
                candidate_count INTEGER NOT NULL DEFAULT 0,
                selected_count INTEGER NOT NULL DEFAULT 0,
                estimated_tokens INTEGER NOT NULL DEFAULT 0,
                prepare_ms REAL NOT NULL DEFAULT 0,
                abstained INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_recall_runs_created ON recall_runs(created_at);

            CREATE TABLE IF NOT EXISTS memory_traces (
                trace_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL UNIQUE,
                session_id TEXT,
                goal TEXT NOT NULL,
                context_summary TEXT NOT NULL,
                retrieval_context_json TEXT NOT NULL DEFAULT '{}',
                task_type TEXT NOT NULL,
                recall_mode TEXT NOT NULL,
                retrieval_used INTEGER NOT NULL DEFAULT 0,
                retrieval_reason TEXT NOT NULL,
                queries_json TEXT NOT NULL DEFAULT '[]',
                candidate_memories_json TEXT NOT NULL DEFAULT '[]',
                selected_memory_ids_json TEXT NOT NULL DEFAULT '[]',
                rejected_memory_ids_json TEXT NOT NULL DEFAULT '[]',
                influence_json TEXT NOT NULL DEFAULT '[]',
                evaluations_json TEXT NOT NULL DEFAULT '[]',
                memory_actions_json TEXT NOT NULL DEFAULT '[]',
                outcome TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memory_traces_created
              ON memory_traces(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_traces_outcome
              ON memory_traces(outcome,created_at DESC);

            CREATE TABLE IF NOT EXISTS memory_trace_events (
                sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                task_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_trace_events_task
              ON memory_trace_events(task_id,sequence_id);

            CREATE TABLE IF NOT EXISTS memory_write_decisions (
                decision_id TEXT PRIMARY KEY,
                session_id TEXT,
                candidate_hash TEXT NOT NULL,
                kind TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'conversation',
                context_mode TEXT NOT NULL,
                reusable_score REAL NOT NULL,
                durability TEXT NOT NULL,
                quality_flags_json TEXT NOT NULL DEFAULT '[]',
                duplicate_memory_id TEXT,
                contradiction_ids_json TEXT NOT NULL DEFAULT '[]',
                independently_understandable INTEGER NOT NULL,
                decision TEXT NOT NULL,
                reason TEXT NOT NULL,
                memory_id TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_write_decisions_created
              ON memory_write_decisions(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_write_decisions_memory
              ON memory_write_decisions(memory_id,created_at DESC);

            CREATE TABLE IF NOT EXISTS memory_creation_proposals (
                proposal_id TEXT PRIMARY KEY,
                proposal_key TEXT NOT NULL,
                candidate_hash TEXT NOT NULL,
                content TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'semantic',
                source_type TEXT NOT NULL DEFAULT 'conversation',
                source_category TEXT NOT NULL DEFAULT 'AGENT_INFERENCE',
                source_ref TEXT,
                session_id TEXT,
                context_mode TEXT NOT NULL DEFAULT 'standalone',
                candidate_json TEXT NOT NULL DEFAULT '{}',
                assessment_json TEXT NOT NULL DEFAULT '{}',
                storage_policy TEXT NOT NULL DEFAULT 'automatic',
                quarantine_reason TEXT,
                redacted INTEGER NOT NULL DEFAULT 0,
                recurrence_count INTEGER NOT NULL DEFAULT 1,
                positive_feedback_count INTEGER NOT NULL DEFAULT 0,
                strong_feedback_count INTEGER NOT NULL DEFAULT 0,
                last_feedback_at TEXT,
                status TEXT NOT NULL DEFAULT 'pending'
                  CHECK(status IN ('pending','remembered','evidence_only','rejected','needs_context')),
                decision_action TEXT,
                decision_note TEXT,
                result_memory_id TEXT REFERENCES memories(id) ON DELETE SET NULL,
                review_id TEXT,
                actor TEXT,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                decided_at TEXT,
                review_started_at TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_creation_pending_key
              ON memory_creation_proposals(proposal_key)
              WHERE status IN ('pending','needs_context');
            CREATE INDEX IF NOT EXISTS idx_memory_creation_status
              ON memory_creation_proposals(status,last_seen_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_creation_candidate
              ON memory_creation_proposals(candidate_hash,last_seen_at DESC);

            CREATE TABLE IF NOT EXISTS memory_creation_feedback (
                feedback_id TEXT PRIMARY KEY,
                proposal_id TEXT NOT NULL REFERENCES memory_creation_proposals(proposal_id) ON DELETE CASCADE,
                session_id TEXT,
                strength TEXT NOT NULL CHECK(strength IN ('positive','strong')),
                source_type TEXT NOT NULL DEFAULT 'implicit_user_feedback',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_creation_feedback_proposal
              ON memory_creation_feedback(proposal_id,created_at DESC);

            CREATE TABLE IF NOT EXISTS memory_experience_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL CHECK(event_type IN (
                  'candidate_observed','memory_selected','memory_used','outcome_labeled'
                )),
                proposal_id TEXT REFERENCES memory_creation_proposals(proposal_id) ON DELETE SET NULL,
                memory_id TEXT REFERENCES memories(id) ON DELETE CASCADE,
                task_id TEXT,
                session_id TEXT,
                source_type TEXT NOT NULL DEFAULT '',
                source_category TEXT NOT NULL DEFAULT '',
                outcome TEXT NOT NULL DEFAULT '',
                evidence_key TEXT NOT NULL UNIQUE,
                event_day TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_experience_memory
              ON memory_experience_events(memory_id,event_type,event_day,created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_experience_proposal
              ON memory_experience_events(proposal_id,event_type,event_day,created_at DESC);

            CREATE TABLE IF NOT EXISTS memory_recall_sets (
                recall_set_id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'trained'
                  CHECK(kind IN ('legacy','trained')),
                status TEXT NOT NULL DEFAULT 'inactive'
                  CHECK(status IN ('active','inactive')),
                parent_recall_set_id TEXT REFERENCES memory_recall_sets(recall_set_id),
                actor TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                activated_at TEXT,
                deactivated_at TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_recall_sets_one_active
              ON memory_recall_sets(status) WHERE status='active';
            CREATE INDEX IF NOT EXISTS idx_memory_recall_sets_created
              ON memory_recall_sets(created_at DESC);

            CREATE TABLE IF NOT EXISTS memory_recall_memberships (
                recall_set_id TEXT NOT NULL REFERENCES memory_recall_sets(recall_set_id) ON DELETE CASCADE,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                eligibility TEXT NOT NULL DEFAULT 'primary'
                  CHECK(eligibility IN ('primary','evidence_only')),
                origin TEXT NOT NULL DEFAULT 'trusted_write',
                review_id TEXT,
                actor TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                revoked_at TEXT,
                PRIMARY KEY(recall_set_id,memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_memory_recall_memberships_active
              ON memory_recall_memberships(recall_set_id,eligibility,memory_id)
              WHERE revoked_at IS NULL;
            CREATE INDEX IF NOT EXISTS idx_memory_recall_memberships_memory
              ON memory_recall_memberships(memory_id,recall_set_id);

            CREATE TABLE IF NOT EXISTS memory_recall_set_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                recall_set_id TEXT NOT NULL REFERENCES memory_recall_sets(recall_set_id) ON DELETE CASCADE,
                prior_recall_set_id TEXT,
                memory_id TEXT,
                eligibility TEXT,
                actor TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                details_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_recall_set_events_created
              ON memory_recall_set_events(created_at DESC);

            CREATE TABLE IF NOT EXISTS memory_neighborhoods (
                neighborhood_id TEXT PRIMARY KEY,
                slug TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'topic',
                parent_neighborhood_id TEXT REFERENCES memory_neighborhoods(neighborhood_id),
                description TEXT NOT NULL DEFAULT '',
                safety_class TEXT NOT NULL DEFAULT 'normal'
                  CHECK(safety_class IN ('normal','secret_reference_only')),
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_neighborhood_memberships (
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                neighborhood_id TEXT NOT NULL REFERENCES memory_neighborhoods(neighborhood_id) ON DELETE CASCADE,
                confidence REAL NOT NULL DEFAULT 1.0,
                origin TEXT NOT NULL DEFAULT 'deterministic',
                explanation TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                reviewed_at TEXT,
                PRIMARY KEY(memory_id,neighborhood_id)
            );
            CREATE INDEX IF NOT EXISTS idx_memory_neighborhood_memberships_neighborhood
              ON memory_neighborhood_memberships(neighborhood_id,memory_id);

            CREATE TABLE IF NOT EXISTS memory_neighborhood_evaluations (
                category TEXT NOT NULL CHECK(category IN ('project','service')),
                name_key TEXT NOT NULL,
                display_name TEXT NOT NULL,
                memory_count INTEGER NOT NULL DEFAULT 0,
                context_count INTEGER NOT NULL DEFAULT 0,
                decision TEXT NOT NULL CHECK(decision IN ('admitted','held','inactive')),
                reason TEXT NOT NULL,
                evaluated_at TEXT NOT NULL,
                PRIMARY KEY(category,name_key)
            );
            CREATE TABLE IF NOT EXISTS memory_neighborhood_decision_events (
                event_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                name_key TEXT NOT NULL,
                display_name TEXT NOT NULL,
                prior_decision TEXT,
                decision TEXT NOT NULL,
                memory_count INTEGER NOT NULL DEFAULT 0,
                context_count INTEGER NOT NULL DEFAULT 0,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_neighborhood_events_created
              ON memory_neighborhood_decision_events(created_at DESC);

            CREATE TRIGGER IF NOT EXISTS cortex_recall_membership_on_memory_insert
            AFTER INSERT ON memories BEGIN
              INSERT OR IGNORE INTO memory_recall_memberships(
                recall_set_id,memory_id,eligibility,origin,actor,reason,created_at
              )
              SELECT recall_set_id,NEW.id,
                     CASE WHEN kind='trained' AND (
                       NEW.source_type='vault_markdown' OR NEW.record_role='reference'
                     ) THEN 'evidence_only' ELSE 'primary' END,
                     'trusted_write','system','memory created while this recall set was active',NEW.created_at
              FROM memory_recall_sets WHERE status='active' LIMIT 1;
            END;

            CREATE TABLE IF NOT EXISTS memory_context_outcomes (
                task_id TEXT NOT NULL,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                context_key TEXT NOT NULL,
                context_json TEXT NOT NULL,
                selected INTEGER NOT NULL DEFAULT 1,
                used INTEGER NOT NULL DEFAULT 0,
                outcome TEXT NOT NULL DEFAULT 'pending',
                updated_at TEXT NOT NULL,
                PRIMARY KEY(task_id,memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_memory_context_outcomes_lookup
              ON memory_context_outcomes(memory_id,context_key,outcome);

            CREATE TABLE IF NOT EXISTS recall_budget_observations (
                task_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                mode TEXT NOT NULL,
                requested_budget INTEGER NOT NULL,
                estimated_tokens INTEGER NOT NULL,
                selected_count INTEGER NOT NULL,
                used_count INTEGER NOT NULL DEFAULT 0,
                outcome TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                resolved_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_budget_observations_lookup
              ON recall_budget_observations(task_type,mode,outcome,created_at);

            CREATE TABLE IF NOT EXISTS attention_observations (
                task_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                topics_json TEXT NOT NULL DEFAULT '[]',
                live_mode TEXT NOT NULL,
                live_budget INTEGER NOT NULL,
                shadow_mode TEXT NOT NULL,
                shadow_budget INTEGER NOT NULL,
                policy_version TEXT NOT NULL,
                selected_count INTEGER NOT NULL DEFAULT 0,
                usage_outcome TEXT NOT NULL DEFAULT 'pending'
                  CHECK(usage_outcome IN ('pending','used','ignored','no_context','control')),
                final_outcome TEXT NOT NULL DEFAULT 'pending'
                  CHECK(final_outcome IN ('pending','helpful','harmful','validated','corrected')),
                created_at TEXT NOT NULL,
                resolved_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_attention_observations_lookup
              ON attention_observations(task_type,live_mode,usage_outcome,created_at);

            CREATE TABLE IF NOT EXISTS attention_weights (
                task_type TEXT NOT NULL,
                topic_key TEXT NOT NULL,
                mode TEXT NOT NULL,
                used_count INTEGER NOT NULL DEFAULT 0,
                helpful_count INTEGER NOT NULL DEFAULT 0,
                ignored_count INTEGER NOT NULL DEFAULT 0,
                harmful_count INTEGER NOT NULL DEFAULT 0,
                weight_delta REAL NOT NULL DEFAULT 0.0,
                last_observed_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(task_type,topic_key,mode)
            );
            CREATE INDEX IF NOT EXISTS idx_attention_weights_topic
              ON attention_weights(topic_key,task_type,updated_at DESC);

            CREATE TABLE IF NOT EXISTS metacognitive_predictions (
                prediction_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                task_type TEXT NOT NULL,
                recall_mode TEXT NOT NULL,
                monitor_mode TEXT NOT NULL DEFAULT 'shadow',
                source_category TEXT NOT NULL,
                raw_probability REAL NOT NULL,
                calibrated_probability REAL NOT NULL,
                decision TEXT NOT NULL,
                applied INTEGER NOT NULL DEFAULT 0,
                reason TEXT NOT NULL,
                calibration_scope TEXT NOT NULL DEFAULT 'prior',
                calibration_samples INTEGER NOT NULL DEFAULT 0,
                features_json TEXT NOT NULL DEFAULT '{}',
                outcome TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                UNIQUE(task_id,memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_metacognitive_outcomes
              ON metacognitive_predictions(task_type,source_category,outcome,created_at);
            CREATE INDEX IF NOT EXISTS idx_metacognitive_created
              ON metacognitive_predictions(created_at DESC);

            CREATE TABLE IF NOT EXISTS task_outcome_labels (
                label_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                outcome TEXT NOT NULL,
                actor TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'dashboard',
                prior_outcome TEXT NOT NULL DEFAULT 'used',
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                reversed_at TEXT,
                reversed_by TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_task_outcome_active
              ON task_outcome_labels(task_id) WHERE active=1;
            CREATE INDEX IF NOT EXISTS idx_task_outcome_created
              ON task_outcome_labels(created_at DESC);

            CREATE TABLE IF NOT EXISTS evaluation_cases (
                case_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL UNIQUE,
                query TEXT NOT NULL,
                relevant_memory_ids TEXT NOT NULL,
                task_type TEXT NOT NULL DEFAULT 'general',
                source_label_id TEXT NOT NULL REFERENCES task_outcome_labels(label_id),
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_evaluation_cases_active
              ON evaluation_cases(active,updated_at DESC);

            CREATE TABLE IF NOT EXISTS evaluation_runs (
                run_id TEXT PRIMARY KEY,
                suite TEXT NOT NULL,
                suite_version TEXT NOT NULL,
                status TEXT NOT NULL,
                phase TEXT NOT NULL DEFAULT 'queued',
                progress INTEGER NOT NULL DEFAULT 0,
                message TEXT NOT NULL DEFAULT '',
                case_count INTEGER NOT NULL DEFAULT 0,
                adaptive_hit_at_k REAL,
                adaptive_recall_at_k REAL,
                adaptive_mrr REAL,
                fixed_hit_at_k REAL,
                fixed_recall_at_k REAL,
                fixed_mrr REAL,
                context_delta_p50 REAL,
                latency_delta_p95 REAL,
                result_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_evaluation_runs_started
              ON evaluation_runs(started_at DESC);

            CREATE TABLE IF NOT EXISTS tool_guidance_exposures (
                exposure_id TEXT PRIMARY KEY,
                session_id TEXT,
                task_id TEXT,
                task_type TEXT NOT NULL,
                guidance_type TEXT NOT NULL,
                guidance_key TEXT NOT NULL,
                recommended_tool TEXT,
                predicted_reliability REAL NOT NULL,
                followed INTEGER,
                success INTEGER,
                task_outcome TEXT,
                created_at TEXT NOT NULL,
                resolved_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_guidance_exposure_session
              ON tool_guidance_exposures(session_id,resolved_at,created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_guidance_exposure_type
              ON tool_guidance_exposures(guidance_type,created_at DESC);

            CREATE TABLE IF NOT EXISTS benchmark_runs (
                run_id TEXT PRIMARY KEY,
                suite TEXT NOT NULL,
                suite_version TEXT NOT NULL,
                status TEXT NOT NULL,
                phase TEXT NOT NULL DEFAULT 'queued',
                progress INTEGER NOT NULL DEFAULT 0,
                message TEXT NOT NULL DEFAULT '',
                score REAL,
                quality_score REAL,
                speed_score REAL,
                recall_at_k REAL,
                mrr REAL,
                precision_at_k REAL,
                p50_ms REAL,
                p95_ms REAL,
                context_tokens_p50 REAL,
                default_coverage REAL,
                corpus_memories INTEGER NOT NULL,
                queries INTEGER NOT NULL,
                result_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_benchmark_runs_started
              ON benchmark_runs(started_at DESC);

            CREATE TABLE IF NOT EXISTS lifecycle_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                from_state TEXT NOT NULL,
                to_state TEXT NOT NULL,
                reason TEXT,
                retention_score REAL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_lifecycle_memory ON lifecycle_events(memory_id,created_at);

            CREATE TABLE IF NOT EXISTS pruning_regret (
                regret_id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                query TEXT,
                score REAL NOT NULL,
                restored INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pruning_regret_created ON pruning_regret(created_at);

            CREATE TABLE IF NOT EXISTS adaptive_pruning_runs (
                run_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL CHECK(mode IN ('shadow','apply')),
                status TEXT NOT NULL,
                model TEXT NOT NULL,
                relevance_threshold REAL NOT NULL,
                candidate_count INTEGER NOT NULL DEFAULT 0,
                judgment_count INTEGER NOT NULL DEFAULT 0,
                applied_count INTEGER NOT NULL DEFAULT 0,
                usage_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_adaptive_pruning_runs_started
              ON adaptive_pruning_runs(started_at DESC);

            CREATE TABLE IF NOT EXISTS adaptive_pruning_decisions (
                decision_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES adaptive_pruning_runs(run_id) ON DELETE CASCADE,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE RESTRICT,
                memory_hash TEXT NOT NULL,
                relevance_score REAL NOT NULL,
                score_evidence_json TEXT NOT NULL DEFAULT '{}',
                action TEXT NOT NULL
                  CHECK(action IN ('cool','archive','quarantine','keep','orphan_strand')),
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL
                  CHECK(status IN ('proposed','applied','kept','skipped','reversed')),
                prior_state TEXT NOT NULL,
                result_state TEXT,
                edge_snapshot_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                applied_at TEXT,
                reversed_at TEXT,
                UNIQUE(run_id,memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_adaptive_pruning_decisions_status
              ON adaptive_pruning_decisions(status,created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_adaptive_pruning_decisions_memory
              ON adaptive_pruning_decisions(memory_id,created_at DESC);

            CREATE TABLE IF NOT EXISTS memory_strands (
                strand_id TEXT PRIMARY KEY,
                decision_id TEXT NOT NULL UNIQUE
                  REFERENCES adaptive_pruning_decisions(decision_id) ON DELETE CASCADE,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE RESTRICT,
                prior_state TEXT NOT NULL,
                reason TEXT NOT NULL,
                edge_snapshot_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('active','restored','reversed')),
                created_at TEXT NOT NULL,
                restored_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memory_strands_active
              ON memory_strands(memory_id,status,created_at DESC);

            CREATE TABLE IF NOT EXISTS scoring_weight_runs (
                run_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL DEFAULT 'shadow' CHECK(mode='shadow'),
                status TEXT NOT NULL,
                model TEXT NOT NULL,
                lookback_days INTEGER NOT NULL DEFAULT 7,
                task_type_count INTEGER NOT NULL DEFAULT 0,
                proposal_count INTEGER NOT NULL DEFAULT 0,
                usage_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_scoring_weight_runs_started
              ON scoring_weight_runs(started_at DESC);

            CREATE TABLE IF NOT EXISTS scoring_weight_proposals (
                proposal_id TEXT PRIMARY KEY,
                run_id TEXT REFERENCES scoring_weight_runs(run_id) ON DELETE SET NULL,
                task_type TEXT NOT NULL,
                baseline_version TEXT NOT NULL,
                baseline_weights_json TEXT NOT NULL,
                proposed_weights_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL DEFAULT '{}',
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                max_deviation REAL NOT NULL,
                explicit_confirmation_required INTEGER NOT NULL DEFAULT 0,
                baseline_precision REAL,
                status TEXT NOT NULL CHECK(status IN (
                  'proposed','approved','rejected','rolled_back','auto_reverted'
                )),
                created_at TEXT NOT NULL,
                decided_at TEXT,
                decided_by TEXT,
                decision_note TEXT,
                rolled_back_at TEXT,
                rollback_reason TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_scoring_weight_proposals_status
              ON scoring_weight_proposals(status,created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_scoring_weight_proposals_task
              ON scoring_weight_proposals(task_type,created_at DESC);

            CREATE TABLE IF NOT EXISTS scoring_weight_history (
                history_id TEXT PRIMARY KEY,
                proposal_id TEXT REFERENCES scoring_weight_proposals(proposal_id) ON DELETE SET NULL,
                task_type TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                weights_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('active','superseded','rolled_back')),
                precision_at_activation REAL,
                activation_sample_count INTEGER NOT NULL DEFAULT 0,
                activated_at TEXT NOT NULL,
                deactivated_at TEXT,
                actor TEXT NOT NULL,
                note TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_scoring_weight_active_task
              ON scoring_weight_history(task_type) WHERE status='active';

            CREATE TABLE IF NOT EXISTS adaptive_reconsolidation_runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                model TEXT NOT NULL,
                task_id TEXT,
                lability_minutes INTEGER NOT NULL DEFAULT 30,
                candidate_count INTEGER NOT NULL DEFAULT 0,
                proposal_count INTEGER NOT NULL DEFAULT 0,
                usage_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_adaptive_reconsolidation_runs_started
              ON adaptive_reconsolidation_runs(started_at DESC);

            CREATE TABLE IF NOT EXISTS adaptive_reconsolidation_proposals (
                proposal_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL
                  REFERENCES adaptive_reconsolidation_runs(run_id) ON DELETE CASCADE,
                task_id TEXT NOT NULL,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE RESTRICT,
                evidence_memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE RESTRICT,
                memory_hash TEXT NOT NULL,
                evidence_hash TEXT NOT NULL,
                action TEXT NOT NULL CHECK(action IN ('supersede','extend','conflict')),
                replacement_content TEXT,
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                protected_confirmation_required INTEGER NOT NULL DEFAULT 0,
                source_snapshot_json TEXT NOT NULL DEFAULT '{}',
                edge_snapshot_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL CHECK(status IN (
                  'proposed','applied','rejected','reversed','skipped'
                )),
                created_at TEXT NOT NULL,
                applied_at TEXT,
                applied_by TEXT,
                reversed_at TEXT,
                feedback_label TEXT CHECK(feedback_label IN ('correct','wrong')),
                feedback_reason TEXT,
                feedback_at TEXT,
                UNIQUE(run_id,task_id,memory_id,evidence_memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_adaptive_reconsolidation_status
              ON adaptive_reconsolidation_proposals(status,created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_adaptive_reconsolidation_memory
              ON adaptive_reconsolidation_proposals(memory_id,created_at DESC);

            CREATE TABLE IF NOT EXISTS schema_formation_runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                model TEXT NOT NULL,
                minimum_cluster INTEGER NOT NULL DEFAULT 3,
                candidate_count INTEGER NOT NULL DEFAULT 0,
                proposal_count INTEGER NOT NULL DEFAULT 0,
                usage_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_schema_formation_runs_started
              ON schema_formation_runs(started_at DESC);

            CREATE TABLE IF NOT EXISTS schema_formation_proposals (
                proposal_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES schema_formation_runs(run_id) ON DELETE CASCADE,
                cluster_signature TEXT NOT NULL,
                action TEXT NOT NULL CHECK(action IN ('abstract','no_schema','partial')),
                abstract_content TEXT,
                source_ids_json TEXT NOT NULL,
                included_source_ids_json TEXT NOT NULL,
                source_hashes_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL DEFAULT '{}',
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN (
                  'proposed','applied','dismissed','rejected','reversed','skipped'
                )),
                result_memory_id TEXT REFERENCES memories(id) ON DELETE SET NULL,
                created_at TEXT NOT NULL,
                applied_at TEXT,
                applied_by TEXT,
                reversed_at TEXT,
                feedback_label TEXT CHECK(feedback_label IN ('correct','wrong')),
                feedback_reason TEXT,
                feedback_at TEXT,
                UNIQUE(run_id,cluster_signature)
            );
            CREATE INDEX IF NOT EXISTS idx_schema_formation_status
              ON schema_formation_proposals(status,created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_schema_formation_signature
              ON schema_formation_proposals(cluster_signature,created_at DESC);

            CREATE TABLE IF NOT EXISTS consolidation_runs (
                run_id TEXT PRIMARY KEY,
                dry_run INTEGER NOT NULL,
                cluster_count INTEGER NOT NULL,
                member_count INTEGER NOT NULL,
                details TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS consolidation_members (
                run_id TEXT NOT NULL REFERENCES consolidation_runs(run_id) ON DELETE CASCADE,
                canonical_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                member_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                similarity REAL NOT NULL,
                prior_state TEXT NOT NULL,
                PRIMARY KEY(run_id,member_id)
            );

            CREATE TABLE IF NOT EXISTS semantic_consolidation_runs (
                run_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL CHECK(mode IN ('shadow','apply')),
                status TEXT NOT NULL,
                model TEXT NOT NULL,
                candidate_count INTEGER NOT NULL DEFAULT 0,
                judgment_count INTEGER NOT NULL DEFAULT 0,
                merge_count INTEGER NOT NULL DEFAULT 0,
                keep_count INTEGER NOT NULL DEFAULT 0,
                link_count INTEGER NOT NULL DEFAULT 0,
                applied_count INTEGER NOT NULL DEFAULT 0,
                usage_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_semantic_consolidation_runs_started
              ON semantic_consolidation_runs(started_at DESC);

            CREATE TABLE IF NOT EXISTS semantic_consolidation_decisions (
                decision_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES semantic_consolidation_runs(run_id) ON DELETE CASCADE,
                left_id TEXT NOT NULL REFERENCES memories(id) ON DELETE RESTRICT,
                right_id TEXT NOT NULL REFERENCES memories(id) ON DELETE RESTRICT,
                left_hash TEXT NOT NULL,
                right_hash TEXT NOT NULL,
                action TEXT NOT NULL CHECK(action IN ('merge','keep_separate','link_as_related')),
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                merged_content TEXT,
                candidate_evidence_json TEXT NOT NULL DEFAULT '[]',
                source_snapshot_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL CHECK(
                    status IN ('proposed','applied','kept_separate','linked','skipped','reversed')
                ),
                result_memory_id TEXT REFERENCES memories(id) ON DELETE SET NULL,
                created_at TEXT NOT NULL,
                applied_at TEXT,
                reversed_at TEXT,
                UNIQUE(run_id,left_id,right_id)
            );
            CREATE INDEX IF NOT EXISTS idx_semantic_consolidation_decisions_status
              ON semantic_consolidation_decisions(status,created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_semantic_consolidation_decisions_pair
              ON semantic_consolidation_decisions(left_id,right_id,created_at DESC);

            CREATE TABLE IF NOT EXISTS semantic_consolidation_feedback (
                feedback_id TEXT PRIMARY KEY,
                decision_id TEXT NOT NULL
                  REFERENCES semantic_consolidation_decisions(decision_id) ON DELETE CASCADE,
                label TEXT NOT NULL CHECK(label IN ('correct','wrong')),
                reason TEXT NOT NULL DEFAULT '',
                actor TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_semantic_consolidation_feedback_decision
              ON semantic_consolidation_feedback(decision_id,created_at DESC);

            CREATE TABLE IF NOT EXISTS sleep_runs (
                run_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                cutoff_at TEXT NOT NULL,
                episodes_scanned INTEGER NOT NULL DEFAULT 0,
                episodes_replayed INTEGER NOT NULL DEFAULT 0,
                association_proposals INTEGER NOT NULL DEFAULT 0,
                interference_proposals INTEGER NOT NULL DEFAULT 0,
                edge_decay_proposals INTEGER NOT NULL DEFAULT 0,
                lifecycle_candidates INTEGER NOT NULL DEFAULT 0,
                consolidation_candidates INTEGER NOT NULL DEFAULT 0,
                dependency_candidates INTEGER NOT NULL DEFAULT 0,
                applied_changes INTEGER NOT NULL DEFAULT 0,
                reflection_token_budget INTEGER NOT NULL DEFAULT 0,
                reflection_estimated_tokens INTEGER NOT NULL DEFAULT 0,
                reflection_billed_tokens INTEGER,
                reflection_status TEXT NOT NULL DEFAULT 'disabled',
                report_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_sleep_runs_started
              ON sleep_runs(started_at DESC);

            CREATE TABLE IF NOT EXISTS sleep_episode_state (
                episode_id INTEGER PRIMARY KEY REFERENCES episodes(id) ON DELETE CASCADE,
                first_replayed_at TEXT NOT NULL,
                last_replayed_at TEXT NOT NULL,
                replay_count INTEGER NOT NULL DEFAULT 1,
                last_run_id TEXT NOT NULL REFERENCES sleep_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS sleep_usage_state (
                task_id TEXT PRIMARY KEY,
                processed_at TEXT NOT NULL,
                last_run_id TEXT NOT NULL REFERENCES sleep_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS sleep_association_evidence (
                src_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                dst_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                witness_key TEXT NOT NULL,
                evidence_kind TEXT NOT NULL,
                score REAL NOT NULL,
                episode_id INTEGER REFERENCES episodes(id) ON DELETE SET NULL,
                run_id TEXT NOT NULL REFERENCES sleep_runs(run_id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                PRIMARY KEY(src_id,dst_id,witness_key,evidence_kind),
                CHECK(src_id < dst_id)
            );
            CREATE INDEX IF NOT EXISTS idx_sleep_evidence_pair
              ON sleep_association_evidence(src_id,dst_id,evidence_kind);

            CREATE TABLE IF NOT EXISTS sleep_proposals (
                proposal_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES sleep_runs(run_id) ON DELETE CASCADE,
                kind TEXT NOT NULL,
                src_id TEXT REFERENCES memories(id) ON DELETE CASCADE,
                dst_id TEXT REFERENCES memories(id) ON DELETE CASCADE,
                status TEXT NOT NULL DEFAULT 'proposed',
                score REAL NOT NULL DEFAULT 0.0,
                evidence_count INTEGER NOT NULL DEFAULT 0,
                rationale TEXT NOT NULL,
                details_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sleep_proposals_run
              ON sleep_proposals(run_id,kind,status);

            CREATE TABLE IF NOT EXISTS operator_review_decisions (
                review_id TEXT PRIMARY KEY,
                item_type TEXT NOT NULL,
                item_key TEXT NOT NULL,
                proposal_id TEXT,
                src_id TEXT,
                dst_id TEXT,
                action TEXT NOT NULL,
                reason_code TEXT NOT NULL,
                reason_text TEXT,
                prior_json TEXT NOT NULL DEFAULT '{}',
                effect_json TEXT NOT NULL DEFAULT '{}',
                learning_signal_json TEXT NOT NULL DEFAULT '{}',
                decision_scope TEXT NOT NULL DEFAULT 'item_only'
                  CHECK(decision_scope IN ('item_only','exact_duplicates','policy_evidence')),
                actor TEXT NOT NULL,
                created_at TEXT NOT NULL,
                reversed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_operator_review_created
              ON operator_review_decisions(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_operator_review_item
              ON operator_review_decisions(item_type,item_key,created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_operator_review_proposal
              ON operator_review_decisions(proposal_id, reversed_at);

            CREATE TABLE IF NOT EXISTS review_copilot_interpretations (
                interpretation_id TEXT PRIMARY KEY,
                proposal_id TEXT REFERENCES sleep_proposals(proposal_id) ON DELETE SET NULL,
                item_key TEXT NOT NULL,
                operator_text TEXT NOT NULL,
                conversation_json TEXT NOT NULL DEFAULT '[]',
                response_mode TEXT NOT NULL CHECK(response_mode IN ('clarify','recommendation')),
                response_json TEXT NOT NULL DEFAULT '{}',
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                created_at TEXT NOT NULL,
                confirmed_review_id TEXT REFERENCES operator_review_decisions(review_id) ON DELETE SET NULL,
                confirmed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_review_copilot_item
              ON review_copilot_interpretations(item_key,created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_review_copilot_confirmed
              ON review_copilot_interpretations(confirmed_review_id,confirmed_at DESC);

            CREATE TABLE IF NOT EXISTS policy_candidates (
                candidate_id TEXT PRIMARY KEY,
                signal_key TEXT NOT NULL UNIQUE,
                domain TEXT NOT NULL,
                title TEXT NOT NULL,
                explanation TEXT NOT NULL,
                selector_json TEXT NOT NULL DEFAULT '{}',
                change_json TEXT NOT NULL DEFAULT '{}',
                direction TEXT NOT NULL,
                support_count INTEGER NOT NULL DEFAULT 0,
                oppose_count INTEGER NOT NULL DEFAULT 0,
                context_count INTEGER NOT NULL DEFAULT 0,
                consistency REAL NOT NULL DEFAULT 0.0,
                stage TEXT NOT NULL DEFAULT 'collecting',
                evidence_json TEXT NOT NULL DEFAULT '{}',
                replay_json TEXT NOT NULL DEFAULT '{}',
                shadow_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                decided_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_policy_candidates_stage
              ON policy_candidates(stage,updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_policy_candidates_domain
              ON policy_candidates(domain,updated_at DESC);

            CREATE TABLE IF NOT EXISTS policy_versions (
                version_id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL REFERENCES policy_candidates(candidate_id) ON DELETE RESTRICT,
                domain TEXT NOT NULL,
                activation_scope TEXT NOT NULL,
                selector_json TEXT NOT NULL DEFAULT '{}',
                change_json TEXT NOT NULL DEFAULT '{}',
                evidence_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'active',
                activated_by TEXT NOT NULL,
                activated_at TEXT NOT NULL,
                deactivated_at TEXT,
                deactivated_by TEXT,
                rollback_reason TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_policy_versions_active
              ON policy_versions(status,domain,activated_at DESC);

            CREATE TABLE IF NOT EXISTS policy_events (
                event_id TEXT PRIMARY KEY,
                candidate_id TEXT REFERENCES policy_candidates(candidate_id) ON DELETE SET NULL,
                version_id TEXT REFERENCES policy_versions(version_id) ON DELETE SET NULL,
                event_type TEXT NOT NULL,
                details_json TEXT NOT NULL DEFAULT '{}',
                actor TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_policy_events_created
              ON policy_events(created_at DESC);

            CREATE TABLE IF NOT EXISTS policy_proposal_effects (
                version_id TEXT NOT NULL REFERENCES policy_versions(version_id) ON DELETE CASCADE,
                proposal_id TEXT NOT NULL REFERENCES sleep_proposals(proposal_id) ON DELETE CASCADE,
                prior_status TEXT NOT NULL,
                next_status TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                reversed_at TEXT,
                PRIMARY KEY(version_id,proposal_id)
            );
            CREATE INDEX IF NOT EXISTS idx_policy_proposal_effects_active
              ON policy_proposal_effects(version_id,next_status,reversed_at);

            CREATE TABLE IF NOT EXISTS sleep_edge_changes (
                change_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES sleep_runs(run_id) ON DELETE CASCADE,
                src_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                dst_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                relation TEXT NOT NULL,
                prior_exists INTEGER NOT NULL,
                prior_weight REAL,
                prior_evidence_count INTEGER,
                prior_last_reinforced_at TEXT,
                next_weight REAL NOT NULL,
                next_evidence_count INTEGER NOT NULL,
                next_last_reinforced_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                reversed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS sleep_state_changes (
                change_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES sleep_runs(run_id) ON DELETE CASCADE,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                prior_state TEXT NOT NULL,
                next_state TEXT NOT NULL,
                created_at TEXT NOT NULL,
                reversed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS sleep_edge_downscale_state (
                src_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                dst_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                relation TEXT NOT NULL,
                source_last_reinforced_at TEXT NOT NULL,
                last_run_id TEXT NOT NULL REFERENCES sleep_runs(run_id) ON DELETE CASCADE,
                last_downscaled_at TEXT NOT NULL,
                PRIMARY KEY(src_id,dst_id,relation)
            );

            CREATE TABLE IF NOT EXISTS controlled_experiments (
                experiment_id TEXT PRIMARY KEY,
                experiment_key TEXT NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'stopped',
                conditions_json TEXT NOT NULL,
                assignment_rule TEXT NOT NULL,
                primary_metric TEXT NOT NULL,
                minimum_labeled_per_condition INTEGER NOT NULL DEFAULT 8,
                started_at TEXT,
                stopped_at TEXT,
                created_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_controlled_experiment_active
              ON controlled_experiments(experiment_key) WHERE status='active';

            CREATE TABLE IF NOT EXISTS recall_experiment_assignments (
                assignment_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL REFERENCES controlled_experiments(experiment_id) ON DELETE CASCADE,
                task_id TEXT UNIQUE,
                session_id TEXT,
                query_hash TEXT NOT NULL,
                task_type TEXT NOT NULL,
                condition TEXT NOT NULL,
                random_bucket INTEGER NOT NULL,
                outcome TEXT NOT NULL DEFAULT 'pending',
                outcome_source TEXT,
                created_at TEXT NOT NULL,
                resolved_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_recall_experiment_results
              ON recall_experiment_assignments(experiment_id,condition,outcome_source,created_at);

            CREATE TABLE IF NOT EXISTS agent_task_observations (
                task_id TEXT PRIMARY KEY,
                session_id TEXT,
                task_type TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                query_preview TEXT,
                recall_condition TEXT NOT NULL,
                recall_mode TEXT NOT NULL,
                memory_count INTEGER NOT NULL DEFAULT 0,
                context_tokens INTEGER NOT NULL DEFAULT 0,
                prepare_ms REAL NOT NULL DEFAULT 0,
                response_ms REAL,
                tool_calls INTEGER NOT NULL DEFAULT 0,
                tool_successes INTEGER NOT NULL DEFAULT 0,
                correction_detected INTEGER NOT NULL DEFAULT 0,
                outcome TEXT NOT NULL DEFAULT 'pending',
                outcome_source TEXT,
                experiment_assignment_id TEXT REFERENCES recall_experiment_assignments(assignment_id),
                started_at TEXT NOT NULL,
                completed_at TEXT,
                resolved_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_agent_tasks_day
              ON agent_task_observations(started_at,task_type,outcome);

            CREATE TABLE IF NOT EXISTS sleep_trials (
                trial_id TEXT PRIMARY KEY,
                sleep_run_id TEXT REFERENCES sleep_runs(run_id) ON DELETE SET NULL,
                status TEXT NOT NULL,
                assignment_rule TEXT NOT NULL,
                primary_metric TEXT NOT NULL,
                pair_count INTEGER NOT NULL DEFAULT 0,
                minimum_labeled_per_arm INTEGER NOT NULL DEFAULT 8,
                started_at TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS sleep_trial_items (
                item_id TEXT PRIMARY KEY,
                trial_id TEXT NOT NULL REFERENCES sleep_trials(trial_id) ON DELETE CASCADE,
                pair_key TEXT NOT NULL,
                proposal_id TEXT NOT NULL REFERENCES sleep_proposals(proposal_id) ON DELETE CASCADE,
                assignment TEXT NOT NULL,
                exposure TEXT NOT NULL,
                src_id TEXT REFERENCES memories(id) ON DELETE CASCADE,
                dst_id TEXT REFERENCES memories(id) ON DELETE CASCADE,
                score REAL NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(trial_id,proposal_id)
            );
            CREATE INDEX IF NOT EXISTS idx_sleep_trial_items
              ON sleep_trial_items(trial_id,assignment,pair_key);

            CREATE TABLE IF NOT EXISTS summary_candidates (
                candidate_id TEXT PRIMARY KEY,
                source_signature TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'proposed',
                approved_memory_id TEXT REFERENCES memories(id) ON DELETE SET NULL,
                created_at TEXT NOT NULL,
                reviewed_at TEXT,
                reviewed_by TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_summary_candidates_status
              ON summary_candidates(status,created_at DESC);

            CREATE TABLE IF NOT EXISTS summary_candidate_claims (
                candidate_id TEXT NOT NULL REFERENCES summary_candidates(candidate_id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                claim_text TEXT NOT NULL,
                PRIMARY KEY(candidate_id,ordinal)
            );

            CREATE TABLE IF NOT EXISTS summary_candidate_sources (
                candidate_id TEXT NOT NULL,
                claim_ordinal INTEGER NOT NULL,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE RESTRICT,
                excerpt TEXT NOT NULL,
                PRIMARY KEY(candidate_id,claim_ordinal,memory_id),
                FOREIGN KEY(candidate_id,claim_ordinal)
                  REFERENCES summary_candidate_claims(candidate_id,ordinal) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS prospective_items (
                memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                status TEXT NOT NULL DEFAULT 'open',
                due_at TEXT,
                completed_at TEXT,
                abandoned_at TEXT,
                note TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_prospective_due
              ON prospective_items(status,due_at);

            CREATE TABLE IF NOT EXISTS reconsolidation_events (
                event_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                prior_version_id INTEGER REFERENCES memory_versions(version_id) ON DELETE SET NULL,
                new_version_id INTEGER REFERENCES memory_versions(version_id) ON DELETE SET NULL,
                trigger_access_at TEXT,
                corrected_at TEXT NOT NULL,
                first_reused_at TEXT,
                stabilized_at TEXT,
                failed_at TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                outcome TEXT,
                source_ref TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_reconsolidation_status
              ON reconsolidation_events(status,corrected_at DESC);

            CREATE TABLE IF NOT EXISTS memory_presentations (
                memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                display_title TEXT NOT NULL,
                display_summary TEXT NOT NULL,
                applies_when TEXT NOT NULL,
                retention_reason TEXT NOT NULL,
                readability_flags_json TEXT NOT NULL DEFAULT '[]',
                presentation_method TEXT NOT NULL,
                presentation_version TEXT NOT NULL,
                source_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_refinery_proposals (
                proposal_id TEXT PRIMARY KEY,
                proposal_kind TEXT NOT NULL
                  CHECK(proposal_kind IN ('promote','rewrite','split','role_change','merge','archive','trash')),
                source_memory_ids_json TEXT NOT NULL DEFAULT '[]',
                proposed_records_json TEXT NOT NULL DEFAULT '[]',
                dependencies_json TEXT NOT NULL DEFAULT '[]',
                rationale TEXT NOT NULL DEFAULT '',
                readability_evidence_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'proposed'
                  CHECK(status IN ('proposed','applied','rejected','undone')),
                actor TEXT NOT NULL DEFAULT '',
                review_id TEXT,
                before_state_json TEXT NOT NULL DEFAULT '{}',
                undo_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                decided_at TEXT,
                undone_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_refinery_proposals_status
              ON memory_refinery_proposals(status,created_at DESC);

            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                memory_id UNINDEXED,
                content,
                tokenize='porter unicode61'
            );

            CREATE TABLE IF NOT EXISTS memory_features (
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                feature TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY(memory_id,feature)
            );
            CREATE INDEX IF NOT EXISTS idx_memory_features_feature ON memory_features(feature,memory_id);

            CREATE TABLE IF NOT EXISTS memory_context_terms (
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                term_type TEXT NOT NULL,
                term_key TEXT NOT NULL DEFAULT '',
                term_value TEXT NOT NULL,
                PRIMARY KEY(memory_id,term_type,term_key,term_value)
            );
            CREATE INDEX IF NOT EXISTS idx_memory_context_terms_lookup
              ON memory_context_terms(term_type,term_key,term_value,memory_id);

            CREATE TABLE IF NOT EXISTS feature_stats (
                feature TEXT PRIMARY KEY,
                document_frequency INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        self._migrate_columns()
        self._backfill_explainable_edge_evidence()
        self._backfill_memory_features()
        self._backfill_memory_context_terms()
        self._rebuild_feature_stats()
        self._backfill_refinery()
        self._backfill_recall_sets()
        self._backfill_memory_neighborhoods()
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_claim ON memories(subject, predicate, state, valid_from, valid_to)"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_dirty ON memories(dirty, state)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_role ON memories(record_role, state)")
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )
        self._compile_policy_candidates_tx(self._conn)
        self._conn.commit()

    def _bump_local_retrieval_revision(self) -> int:
        """Advance the in-process material revision from a TEMP trigger."""

        self._local_retrieval_revision += 1
        return self._local_retrieval_revision

    def _create_connection_revision_triggers(self) -> None:
        """Track local material writes without adding a write to every transaction.

        TEMP triggers exist only on this connection. Other connections are
        detected through SQLite's ``data_version`` value in
        :meth:`retrieval_revision`, so raw SQLite clients remain compatible.
        """

        self._conn.executescript(
            """
            CREATE TEMP TRIGGER cortex_local_revision_memory_insert
            AFTER INSERT ON main.memories BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_revision_memory_delete
            AFTER DELETE ON main.memories BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_revision_memory_material_update
            AFTER UPDATE OF kind,content,source_category,context_mode,scope_json,entities_json,
              preconditions_json,source_context,applicable_systems_json,applicable_versions_json,
              metadata_completeness,valid_from,valid_to,subject,predicate,object_value,
              confidence,currentness_confidence,importance,uniqueness,volatility,trust,state,
              pinned,protected,supersedes_id,quarantine_reason,used_count,success_count,
              confirmed_count,validated_count,helpful_count,harmful_count,correction_count,
              false_positive_count ON main.memories BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_revision_edge_insert
            AFTER INSERT ON main.edges BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_revision_edge_update
            AFTER UPDATE ON main.edges BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_revision_edge_delete
            AFTER DELETE ON main.edges BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_revision_tool_stats_insert
            AFTER INSERT ON main.tool_stats BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_revision_tool_stats_update
            AFTER UPDATE ON main.tool_stats BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_revision_tool_stats_delete
            AFTER DELETE ON main.tool_stats BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_revision_workflow_stats_insert
            AFTER INSERT ON main.tool_workflow_stats BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_revision_workflow_stats_update
            AFTER UPDATE ON main.tool_workflow_stats BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_revision_workflow_stats_delete
            AFTER DELETE ON main.tool_workflow_stats BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_context_terms_memory_insert
            AFTER INSERT ON main.memories BEGIN
              INSERT OR IGNORE INTO memory_context_terms(memory_id,term_type,term_key,term_value)
                VALUES(NEW.id,'mode','',lower(NEW.context_mode));
              INSERT OR IGNORE INTO memory_context_terms(memory_id,term_type,term_key,term_value)
                SELECT NEW.id,'scope',lower(j.key),lower(trim(CAST(j.value AS TEXT)))
                FROM json_each(CASE WHEN json_valid(NEW.scope_json) THEN NEW.scope_json ELSE '{}' END) j;
              INSERT OR IGNORE INTO memory_context_terms(memory_id,term_type,term_key,term_value)
                SELECT NEW.id,'entity','',lower(trim(CAST(j.value AS TEXT)))
                FROM json_each(CASE WHEN json_valid(NEW.entities_json) THEN NEW.entities_json ELSE '[]' END) j;
              INSERT OR IGNORE INTO memory_context_terms(memory_id,term_type,term_key,term_value)
                SELECT NEW.id,'precondition',lower(j.key),lower(trim(CAST(j.value AS TEXT)))
                FROM json_each(CASE WHEN json_valid(NEW.preconditions_json) THEN NEW.preconditions_json ELSE '{}' END) j;
              INSERT OR IGNORE INTO memory_context_terms(memory_id,term_type,term_key,term_value)
                SELECT NEW.id,'system','',lower(trim(CAST(j.value AS TEXT)))
                FROM json_each(CASE WHEN json_valid(NEW.applicable_systems_json) THEN NEW.applicable_systems_json ELSE '[]' END) j;
              INSERT OR IGNORE INTO memory_context_terms(memory_id,term_type,term_key,term_value)
                SELECT NEW.id,'version','',lower(trim(CAST(j.value AS TEXT)))
                FROM json_each(CASE WHEN json_valid(NEW.applicable_versions_json) THEN NEW.applicable_versions_json ELSE '[]' END) j;
            END;
            CREATE TEMP TRIGGER cortex_local_context_terms_memory_update
            AFTER UPDATE OF context_mode,scope_json,entities_json,preconditions_json,
              applicable_systems_json,applicable_versions_json ON main.memories BEGIN
              DELETE FROM memory_context_terms WHERE memory_id=NEW.id;
              INSERT OR IGNORE INTO memory_context_terms(memory_id,term_type,term_key,term_value)
                VALUES(NEW.id,'mode','',lower(NEW.context_mode));
              INSERT OR IGNORE INTO memory_context_terms(memory_id,term_type,term_key,term_value)
                SELECT NEW.id,'scope',lower(j.key),lower(trim(CAST(j.value AS TEXT)))
                FROM json_each(CASE WHEN json_valid(NEW.scope_json) THEN NEW.scope_json ELSE '{}' END) j;
              INSERT OR IGNORE INTO memory_context_terms(memory_id,term_type,term_key,term_value)
                SELECT NEW.id,'entity','',lower(trim(CAST(j.value AS TEXT)))
                FROM json_each(CASE WHEN json_valid(NEW.entities_json) THEN NEW.entities_json ELSE '[]' END) j;
              INSERT OR IGNORE INTO memory_context_terms(memory_id,term_type,term_key,term_value)
                SELECT NEW.id,'precondition',lower(j.key),lower(trim(CAST(j.value AS TEXT)))
                FROM json_each(CASE WHEN json_valid(NEW.preconditions_json) THEN NEW.preconditions_json ELSE '{}' END) j;
              INSERT OR IGNORE INTO memory_context_terms(memory_id,term_type,term_key,term_value)
                SELECT NEW.id,'system','',lower(trim(CAST(j.value AS TEXT)))
                FROM json_each(CASE WHEN json_valid(NEW.applicable_systems_json) THEN NEW.applicable_systems_json ELSE '[]' END) j;
              INSERT OR IGNORE INTO memory_context_terms(memory_id,term_type,term_key,term_value)
                SELECT NEW.id,'version','',lower(trim(CAST(j.value AS TEXT)))
                FROM json_each(CASE WHEN json_valid(NEW.applicable_versions_json) THEN NEW.applicable_versions_json ELSE '[]' END) j;
            END;
            CREATE TEMP TRIGGER cortex_local_revision_context_outcome_insert
            AFTER INSERT ON main.memory_context_outcomes BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_revision_context_outcome_update
            AFTER UPDATE ON main.memory_context_outcomes BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_revision_context_outcome_delete
            AFTER DELETE ON main.memory_context_outcomes BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_revision_recall_set_insert
            AFTER INSERT ON main.memory_recall_sets BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_revision_recall_set_update
            AFTER UPDATE ON main.memory_recall_sets BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_revision_recall_set_delete
            AFTER DELETE ON main.memory_recall_sets BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_revision_recall_membership_insert
            AFTER INSERT ON main.memory_recall_memberships BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_revision_recall_membership_update
            AFTER UPDATE ON main.memory_recall_memberships BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_revision_recall_membership_delete
            AFTER DELETE ON main.memory_recall_memberships BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_revision_neighborhood_membership_insert
            AFTER INSERT ON main.memory_neighborhood_memberships BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_revision_neighborhood_membership_update
            AFTER UPDATE ON main.memory_neighborhood_memberships BEGIN SELECT cortex_bump_revision(); END;
            CREATE TEMP TRIGGER cortex_local_revision_neighborhood_membership_delete
            AFTER DELETE ON main.memory_neighborhood_memberships BEGIN SELECT cortex_bump_revision(); END;
            """
        )

    def _backfill_memory_features(self) -> None:
        rows = self._conn.execute(
            """SELECT m.id,m.content FROM memories m
               WHERE NOT EXISTS(SELECT 1 FROM memory_features f WHERE f.memory_id=m.id)"""
        ).fetchall()
        for row in rows:
            self._index_features_tx(self._conn, str(row["id"]), str(row["content"]))

    @staticmethod
    def _index_features_tx(conn: sqlite3.Connection, memory_id: str, content: str) -> None:
        old_features = [
            str(row["feature"])
            for row in conn.execute("SELECT feature FROM memory_features WHERE memory_id=?", (memory_id,)).fetchall()
        ]
        if old_features:
            conn.executemany(
                "UPDATE feature_stats SET document_frequency=MAX(0,document_frequency-1) WHERE feature=?",
                [(feature,) for feature in old_features],
            )
            conn.execute("DELETE FROM feature_stats WHERE document_frequency<=0")
        conn.execute("DELETE FROM memory_features WHERE memory_id=?", (memory_id,))
        indexed = list(semantic_features(content).items())
        conn.executemany(
            "INSERT INTO memory_features(memory_id,feature,weight) VALUES(?,?,?)",
            [(memory_id, feature, weight) for feature, weight in indexed],
        )
        conn.executemany(
            """INSERT INTO feature_stats(feature,document_frequency) VALUES(?,1)
               ON CONFLICT(feature) DO UPDATE SET document_frequency=document_frequency+1""",
            [(feature,) for feature, _weight in indexed],
        )

    def _rebuild_feature_stats(self) -> None:
        self._conn.execute("DELETE FROM feature_stats")
        self._conn.execute(
            """INSERT INTO feature_stats(feature,document_frequency)
               SELECT feature,COUNT(*) FROM memory_features GROUP BY feature"""
        )

    def _backfill_memory_context_terms(self) -> None:
        rows = self._conn.execute(
            """SELECT m.* FROM memories m
               WHERE NOT EXISTS(
                 SELECT 1 FROM memory_context_terms t WHERE t.memory_id=m.id
               )"""
        ).fetchall()
        for row in rows:
            memory = _decode_memory_metadata(row)
            self._index_context_terms_tx(
                self._conn,
                str(memory["id"]),
                context_mode=str(memory.get("context_mode") or "standalone"),
                scope=dict(memory.get("scope") or {}),
                entities=list(memory.get("entities") or []),
                preconditions=dict(memory.get("preconditions") or {}),
                applicable_systems=list(memory.get("applicable_systems") or []),
                applicable_versions=list(memory.get("applicable_versions") or []),
            )

    def _backfill_explainable_edge_evidence(self) -> None:
        """Recover only link reasons that legacy relational data can prove.

        Similarity-era ``related`` and ``co_observed`` edges intentionally stay
        unattributed.  Calling them explained would make the map more polished
        but less truthful.
        """

        rows = self._conn.execute(
            """SELECT e.*,src.subject src_subject,src.predicate src_predicate,
                      src.object_value src_object,src.valid_from src_valid_from,src.valid_to src_valid_to,
                      src.supersedes_id src_supersedes_id,
                      dst.subject dst_subject,dst.predicate dst_predicate,
                      dst.object_value dst_object,dst.valid_from dst_valid_from,dst.valid_to dst_valid_to,
                      dst.supersedes_id dst_supersedes_id
               FROM edges e
               JOIN memories src ON src.id=e.src_id
               JOIN memories dst ON dst.id=e.dst_id
               WHERE NOT EXISTS(
                 SELECT 1 FROM edge_evidence ev
                 WHERE ev.src_id=e.src_id AND ev.dst_id=e.dst_id AND ev.relation=e.relation
                   AND ev.evidence_type<>'legacy_unattributed'
               )"""
        ).fetchall()
        now = utc_now()
        for row in rows:
            src_id, dst_id, relation = str(row["src_id"]), str(row["dst_id"]), str(row["relation"])
            evidence: list[tuple[str, str, str, str | None, dict[str, Any]]] = []
            if relation == "vault_link":
                evidence.append(
                    (
                        "migrated_explicit_wikilink",
                        f"legacy-vault-link:{src_id}:{dst_id}",
                        "The legacy vault importer created this relation only for an explicit wikilink. "
                        "The original link label predates per-link evidence logging.",
                        None,
                        {"migration": "legacy vault_link relation"},
                    )
                )
            elif relation == "supersedes" and (
                str(row["src_supersedes_id"] or "") == dst_id
                or str(row["dst_supersedes_id"] or "") == src_id
            ):
                evidence.append(
                    (
                        "migrated_version_lineage",
                        f"legacy-supersedes:{src_id}:{dst_id}",
                        "The stored version-lineage field identifies one memory as the replacement for the other.",
                        None,
                        {"migration": "verified supersedes_id lineage"},
                    )
                )
            elif relation == "contradicts" and (
                row["src_subject"]
                and row["src_subject"] == row["dst_subject"]
                and row["src_predicate"] == row["dst_predicate"]
                and str(row["src_object"] or "") != str(row["dst_object"] or "")
                and _periods_overlap(
                    row["src_valid_from"],
                    row["src_valid_to"],
                    row["dst_valid_from"],
                    row["dst_valid_to"],
                )
            ):
                evidence.append(
                    (
                        "migrated_structured_claim_conflict",
                        f"legacy-claim-conflict:{src_id}:{dst_id}",
                        "Both memories make the same structured claim for overlapping validity windows but store different values.",
                        None,
                        {"migration": "verified structured claim conflict"},
                    )
                )
            elif relation == "co_used":
                tasks = self._conn.execute(
                    """SELECT DISTINCT a.task_id FROM usage_records a
                       JOIN usage_records b ON b.task_id=a.task_id
                       WHERE a.memory_id=? AND b.memory_id=? AND a.used=1 AND b.used=1
                         AND a.task_id IS NOT NULL AND a.task_id<>''
                       ORDER BY a.created_at DESC LIMIT 20""",
                    (src_id, dst_id),
                ).fetchall()
                evidence.extend(
                    (
                        "migrated_shared_task_use",
                        f"legacy-shared-task:{task['task_id']}",
                        "Both memories were attributed as used in the same recorded task.",
                        str(task["task_id"]),
                        {"migration": "verified shared usage task"},
                    )
                    for task in tasks
                )
            for evidence_type, evidence_key, summary, task_id, metadata in evidence:
                self._record_edge_evidence_tx(
                    self._conn,
                    src_id,
                    dst_id,
                    relation,
                    evidence_type=evidence_type,
                    evidence_key=evidence_key,
                    summary=summary,
                    task_id=task_id,
                    metadata=metadata,
                    created_at=now,
                )
            if evidence:
                self._conn.execute(
                    """UPDATE edges SET evidence_count=MAX(evidence_count,(
                         SELECT COUNT(*) FROM edge_evidence ev
                         WHERE ev.src_id=edges.src_id AND ev.dst_id=edges.dst_id
                           AND ev.relation=edges.relation
                       )) WHERE src_id=? AND dst_id=? AND relation=?""",
                    (src_id, dst_id, relation),
                )

    @staticmethod
    def _index_context_terms_tx(
        conn: sqlite3.Connection,
        memory_id: str,
        *,
        context_mode: str,
        scope: dict[str, Any],
        entities: Sequence[str],
        preconditions: dict[str, Any],
        applicable_systems: Sequence[str],
        applicable_versions: Sequence[str],
    ) -> None:
        conn.execute("DELETE FROM memory_context_terms WHERE memory_id=?", (memory_id,))
        terms: set[tuple[str, str, str]] = {
            ("mode", "", normalize_text(context_mode).casefold())
        }
        terms.update(
            ("scope", normalize_text(str(key)).casefold(), normalize_text(str(value)).casefold())
            for key, value in scope.items()
            if normalize_text(str(key)) and normalize_text(str(value))
        )
        terms.update(
            (
                "precondition",
                normalize_text(str(key)).casefold(),
                normalize_text(str(value)).casefold(),
            )
            for key, value in preconditions.items()
            if normalize_text(str(key)) and normalize_text(str(value))
        )
        for term_type, values in (
            ("entity", entities),
            ("system", applicable_systems),
            ("version", applicable_versions),
        ):
            terms.update(
                (term_type, "", normalize_text(str(value)).casefold())
                for value in values
                if normalize_text(str(value))
            )
        conn.executemany(
            """INSERT INTO memory_context_terms(memory_id,term_type,term_key,term_value)
               VALUES(?,?,?,?)""",
            [(memory_id, term_type, key, value) for term_type, key, value in sorted(terms)],
        )

    def _migrate_columns(self) -> None:
        """Add v2 columns safely when opening an early Cortex prototype DB."""
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(memories)")}
        additions = {
            "source_category": "TEXT NOT NULL DEFAULT 'AGENT_INFERENCE'",
            "origin_source_category": "TEXT NOT NULL DEFAULT 'AGENT_INFERENCE'",
            "approval_state": "TEXT NOT NULL DEFAULT 'unreviewed'",
            "observed_at": "TEXT",
            "subject": "TEXT",
            "predicate": "TEXT",
            "object_value": "TEXT",
            "extraction_method": "TEXT NOT NULL DEFAULT 'unknown'",
            "currentness_confidence": "REAL NOT NULL DEFAULT 0.7",
            "uniqueness": "REAL NOT NULL DEFAULT 1.0",
            "protected": "INTEGER NOT NULL DEFAULT 0",
            "dirty": "INTEGER NOT NULL DEFAULT 0",
            "dirty_reason": "TEXT",
            "supersedes_id": "TEXT",
            "selected_count": "INTEGER NOT NULL DEFAULT 0",
            "validated_count": "INTEGER NOT NULL DEFAULT 0",
            "helpful_count": "INTEGER NOT NULL DEFAULT 0",
            "harmful_count": "INTEGER NOT NULL DEFAULT 0",
            "last_helpful_at": "TEXT",
            "context_mode": "TEXT NOT NULL DEFAULT 'standalone'",
            "scope_json": "TEXT NOT NULL DEFAULT '{}'",
            "entities_json": "TEXT NOT NULL DEFAULT '[]'",
            "preconditions_json": "TEXT NOT NULL DEFAULT '{}'",
            "source_context": "TEXT",
            "applicable_systems_json": "TEXT NOT NULL DEFAULT '[]'",
            "applicable_versions_json": "TEXT NOT NULL DEFAULT '[]'",
            "metadata_completeness": "REAL NOT NULL DEFAULT 1.0",
            "record_role": "TEXT NOT NULL DEFAULT 'canonical'",
            "role_method": "TEXT NOT NULL DEFAULT 'legacy_default'",
            "role_version": "TEXT",
            "role_reviewed_at": "TEXT",
            "stranded": "INTEGER NOT NULL DEFAULT 0",
            "stranded_reason": "TEXT",
            "stranded_at": "TEXT",
        }
        for name, declaration in additions.items():
            if name not in columns:
                self._conn.execute(f"ALTER TABLE memories ADD COLUMN {name} {declaration}")
        self._conn.execute("UPDATE memories SET observed_at=COALESCE(observed_at, created_at)")
        self._conn.execute(
            """UPDATE memories
               SET origin_source_category=CASE
                 WHEN origin_source_category IS NULL OR origin_source_category='' THEN source_category
                 WHEN origin_source_category='AGENT_INFERENCE'
                      AND source_category<>'AGENT_INFERENCE' THEN source_category
                 ELSE origin_source_category END,
                   approval_state=CASE
                 WHEN approval_state IS NULL OR approval_state='' THEN
                   CASE
                     WHEN source_category='OPERATOR_APPROVED' THEN 'operator_approved'
                     WHEN source_category='AUTOMATIC_APPROVED' THEN 'automatic_approved'
                     ELSE 'unreviewed' END
                 WHEN approval_state='unreviewed' AND source_category='OPERATOR_APPROVED'
                   THEN 'operator_approved'
                 WHEN approval_state='unreviewed' AND source_category='AUTOMATIC_APPROVED'
                   THEN 'automatic_approved'
                 ELSE approval_state END"""
        )
        # Schema 23 did not persist claim origin separately from review
        # authority. Recover it only when the proposal decision proves that it
        # created this memory; duplicate reviews must never rewrite an existing
        # memory's provenance.
        self._conn.execute(
            """UPDATE memories
               SET origin_source_category=(
                 SELECT p.source_category
                 FROM memory_creation_proposals p
                 JOIN operator_review_decisions d ON d.review_id=p.review_id
                 WHERE p.result_memory_id=memories.id
                   AND json_extract(d.effect_json,'$.memory_created')=1
                 ORDER BY p.decided_at DESC LIMIT 1
               )
               WHERE origin_source_category IN (
                 'AGENT_INFERENCE','OPERATOR_APPROVED','AUTOMATIC_APPROVED'
               )
                 AND EXISTS(
                   SELECT 1
                   FROM memory_creation_proposals p
                   JOIN operator_review_decisions d ON d.review_id=p.review_id
                   WHERE p.result_memory_id=memories.id
                     AND json_extract(d.effect_json,'$.memory_created')=1
                 )"""
        )
        self._conn.execute(
            """UPDATE memories SET context_mode='standalone',scope_json='{}',entities_json='[]',
                 preconditions_json='{}',applicable_systems_json='[]',applicable_versions_json='[]',
                 metadata_completeness=1.0
               WHERE context_mode IS NULL OR context_mode NOT IN ('standalone','context_dependent')"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_context_mode ON memories(context_mode,state)")
        task_label_columns = {row[1] for row in self._conn.execute("PRAGMA table_info(task_outcome_labels)")}
        if "prior_outcome" not in task_label_columns:
            self._conn.execute(
                "ALTER TABLE task_outcome_labels ADD COLUMN prior_outcome TEXT NOT NULL DEFAULT 'used'"
            )
        if "reversed_by" not in task_label_columns:
            self._conn.execute("ALTER TABLE task_outcome_labels ADD COLUMN reversed_by TEXT")
        recall_columns = {row[1] for row in self._conn.execute("PRAGMA table_info(recall_runs)")}
        if "task_id" not in recall_columns:
            self._conn.execute("ALTER TABLE recall_runs ADD COLUMN task_id TEXT")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_recall_runs_task ON recall_runs(task_id)")
        trace_columns = {row[1] for row in self._conn.execute("PRAGMA table_info(memory_traces)")}
        if "retrieval_context_json" not in trace_columns:
            self._conn.execute(
                "ALTER TABLE memory_traces ADD COLUMN retrieval_context_json TEXT NOT NULL DEFAULT '{}'"
            )
        write_columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(memory_write_decisions)")
        }
        if "source_type" not in write_columns:
            self._conn.execute(
                "ALTER TABLE memory_write_decisions ADD COLUMN source_type TEXT NOT NULL DEFAULT 'conversation'"
            )
        if "quality_flags_json" not in write_columns:
            self._conn.execute(
                "ALTER TABLE memory_write_decisions ADD COLUMN quality_flags_json TEXT NOT NULL DEFAULT '[]'"
            )
        proposal_columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(memory_creation_proposals)")
        }
        proposal_additions = {
            "positive_feedback_count": "INTEGER NOT NULL DEFAULT 0",
            "strong_feedback_count": "INTEGER NOT NULL DEFAULT 0",
            "last_feedback_at": "TEXT",
            "review_started_at": "TEXT",
        }
        for name, declaration in proposal_additions.items():
            if name not in proposal_columns:
                self._conn.execute(
                    f"ALTER TABLE memory_creation_proposals ADD COLUMN {name} {declaration}"
                )
        review_columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(operator_review_decisions)")
        }
        if "decision_scope" not in review_columns:
            self._conn.execute(
                "ALTER TABLE operator_review_decisions "
                "ADD COLUMN decision_scope TEXT NOT NULL DEFAULT 'item_only'"
            )
            # Decisions made before explicit reach controls were presented as
            # training evidence. Preserve that meaning during migration while
            # defaulting every new dashboard review to a one-off action.
            self._conn.execute(
                "UPDATE operator_review_decisions SET decision_scope='policy_evidence'"
            )
        # Schema 26: Remove FK constraints from operator_review_decisions so that
        # sleep-proposal reviews can be stored alongside creation-proposal reviews.
        # SQLite does not support ALTER TABLE DROP CONSTRAINT, so we recreate the table.
        if "proposal_id TEXT REFERENCES" in (
            self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='operator_review_decisions'"
            ).fetchone() or [""]
        )[0]:
            self._conn.execute("PRAGMA foreign_keys=OFF")
            self._conn.executescript(
                """CREATE TABLE IF NOT EXISTS operator_review_decisions_v26 (
                    review_id TEXT PRIMARY KEY,
                    item_type TEXT NOT NULL,
                    item_key TEXT NOT NULL,
                    proposal_id TEXT,
                    src_id TEXT,
                    dst_id TEXT,
                    action TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    reason_text TEXT,
                    prior_json TEXT NOT NULL DEFAULT '{}',
                    effect_json TEXT NOT NULL DEFAULT '{}',
                    learning_signal_json TEXT NOT NULL DEFAULT '{}',
                    decision_scope TEXT NOT NULL DEFAULT 'item_only'
                      CHECK(decision_scope IN ('item_only','exact_duplicates','policy_evidence')),
                    actor TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    reversed_at TEXT
                );
                INSERT INTO operator_review_decisions_v26
                    SELECT * FROM operator_review_decisions;
                DROP TABLE operator_review_decisions;
                ALTER TABLE operator_review_decisions_v26 RENAME TO operator_review_decisions;
                CREATE INDEX IF NOT EXISTS idx_operator_review_created
                  ON operator_review_decisions(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_operator_review_item
                  ON operator_review_decisions(item_type,item_key,created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_operator_review_proposal
                  ON operator_review_decisions(proposal_id, reversed_at);"""
            )
            self._conn.execute("PRAGMA foreign_keys=ON")

    def _active_dependency_ids_tx(self, conn: sqlite3.Connection) -> set[str]:
        return {
            str(row["memory_id"])
            for row in conn.execute(
                "SELECT DISTINCT memory_id FROM memory_dependencies WHERE active=1"
            ).fetchall()
        }

    @staticmethod
    def _has_active_dependency_tx(conn: sqlite3.Connection, memory_id: str) -> bool:
        return bool(
            conn.execute(
                "SELECT 1 FROM memory_dependencies WHERE memory_id=? AND active=1 LIMIT 1",
                (memory_id,),
            ).fetchone()
        )

    def _classify_and_present_tx(
        self,
        conn: sqlite3.Connection,
        memory: dict[str, Any],
        *,
        has_active_dependencies: bool,
        explicit_role: str | None = None,
        role_method: str | None = None,
        storage_policy: str | None = None,
        preserve_operator_role: bool = True,
    ) -> dict[str, Any]:
        """Classify one record and refresh its derived presentation, in place.

        Never changes content, state, versions, IDs, or retrieval behavior.
        Operator-reviewed roles are preserved unless explicitly overridden.
        """

        memory_id = str(memory["id"])
        classification = classify_record_role(memory, has_active_dependencies=has_active_dependencies)
        current_method = str(memory.get("role_method") or LEGACY_ROLE_METHOD)
        operator_owned = preserve_operator_role and current_method == OPERATOR_ROLE_METHOD
        if explicit_role:
            if explicit_role not in RECORD_ROLES:
                raise ValueError(f"unsupported record role: {explicit_role}")
            next_role = explicit_role
            next_method = role_method or "ingest_default"
        elif operator_owned:
            next_role = str(memory.get("record_role") or "canonical")
            next_method = OPERATOR_ROLE_METHOD
        else:
            next_role = str(classification["record_role"])
            next_method = str(classification["role_method"])
            if (
                storage_policy == "automatic"
                and next_role == "canonical"
                and any(flag in CLARITY_FLAGS for flag in classification["readability_flags"])
            ):
                # Automatic capture must pass the canonical clarity checks;
                # flagged records stay reviewable claims instead.
                next_role = "claim"
                classification["reasons"].append(
                    "Automatic capture with readability flags is kept as a reviewable claim, "
                    "not a canonical memory."
                )
        classification = {**classification, "record_role": next_role}
        conn.execute(
            "UPDATE memories SET record_role=?,role_method=?,role_version=? WHERE id=?",
            (next_role, next_method, str(classification["role_version"]), memory_id),
        )
        presentation = build_presentation(
            {**memory, "record_role": next_role},
            classification,
            has_active_dependencies=has_active_dependencies,
        )
        now = utc_now()
        conn.execute(
            """INSERT INTO memory_presentations(
                 memory_id,display_title,display_summary,applies_when,retention_reason,
                 readability_flags_json,presentation_method,presentation_version,source_digest,
                 created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(memory_id) DO UPDATE SET
                 display_title=excluded.display_title,
                 display_summary=excluded.display_summary,
                 applies_when=excluded.applies_when,
                 retention_reason=excluded.retention_reason,
                 readability_flags_json=excluded.readability_flags_json,
                 presentation_method=excluded.presentation_method,
                 presentation_version=excluded.presentation_version,
                 source_digest=excluded.source_digest,
                 updated_at=excluded.updated_at""",
            (
                memory_id,
                str(presentation["display_title"])[:200],
                str(presentation["display_summary"])[:600],
                str(presentation["applies_when"])[:400],
                str(presentation["retention_reason"])[:400],
                _trace_json(list(presentation["readability_flags"])),
                str(presentation["presentation_method"]),
                str(presentation["presentation_version"]),
                str(memory.get("content_hash") or content_hash(str(memory.get("content") or ""))),
                now,
                now,
            ),
        )
        return classification

    def _backfill_refinery(self) -> None:
        """Version-keyed role and presentation backfill.

        Reclassifies rows that were never classified or that carry a stale
        classifier version, and rebuilds presentations whose source digest or
        generator version no longer matches. Operator-reviewed roles are never
        overwritten. The pass changes no state, content, version, ID,
        dependency, or retrieval-visible value.
        """

        row = self._conn.execute(
            "SELECT value FROM meta WHERE key=?", (REFINERY_BACKFILL_KEY,)
        ).fetchone()
        if row and str(row["value"]) == REFINERY_BACKFILL_VERSION:
            return
        dependency_ids = self._active_dependency_ids_tx(self._conn)
        memory_rows = self._conn.execute(
            """SELECT m.*, p.source_digest presentation_digest,
                      p.presentation_version presentation_version
               FROM memories m LEFT JOIN memory_presentations p ON p.memory_id=m.id"""
        ).fetchall()
        for raw in memory_rows:
            memory = dict(raw)
            stale_presentation = (
                memory.get("presentation_digest") != memory.get("content_hash")
                or memory.get("presentation_version") != PRESENTATION_VERSION
            )
            stale_role = (
                str(memory.get("role_method") or LEGACY_ROLE_METHOD) != OPERATOR_ROLE_METHOD
                and str(memory.get("role_version") or "") != ROLE_CLASSIFIER_VERSION
            )
            if not stale_presentation and not stale_role:
                continue
            self._classify_and_present_tx(
                self._conn,
                memory,
                has_active_dependencies=str(memory["id"]) in dependency_ids,
            )
        self._conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (REFINERY_BACKFILL_KEY, REFINERY_BACKFILL_VERSION),
        )

    def _backfill_recall_sets(self) -> None:
        """Create a behavior-preserving baseline recall set on upgrade.

        Every existing record enters the baseline as ``primary`` so merely
        opening an older database does not change any retrieval result.  The
        stricter boundary begins only after an operator explicitly activates a
        trained set from the dashboard.
        """

        active = self._conn.execute(
            "SELECT recall_set_id FROM memory_recall_sets WHERE status='active' LIMIT 1"
        ).fetchone()
        created_baseline = False
        if not active:
            now = utc_now()
            baseline_id = str(uuid.uuid4())
            self._conn.execute(
                """INSERT INTO memory_recall_sets(
                     recall_set_id,label,kind,status,actor,reason,created_at,activated_at
                   ) VALUES(?,?,'legacy','active','schema-migration',?,?,?)""",
                (
                    baseline_id,
                    "Legacy recall set",
                    "Behavior-preserving baseline created during schema upgrade.",
                    now,
                    now,
                ),
            )
            self._conn.execute(
                """INSERT INTO memory_recall_set_events(
                     event_id,event_type,recall_set_id,actor,reason,details_json,created_at
                   ) VALUES(?,'created',?,'schema-migration',?,'{}',?)""",
                (
                    str(uuid.uuid4()),
                    baseline_id,
                    "Created behavior-preserving baseline recall set.",
                    now,
                ),
            )
            active_id = baseline_id
            created_baseline = True
        else:
            active_id = str(active["recall_set_id"])
        if not created_baseline:
            return
        now = utc_now()
        self._conn.execute(
            """INSERT OR IGNORE INTO memory_recall_memberships(
                 recall_set_id,memory_id,eligibility,origin,actor,reason,created_at
               )
               SELECT ?,id,'primary','legacy_migration','schema-migration',
                      'Preserve pre-recall-set retrieval behavior.',?
               FROM memories""",
            (active_id, now),
        )

    @staticmethod
    def _neighborhood_slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", normalize_text(value).casefold()).strip("-")
        return slug[:80] or "other"

    @staticmethod
    def _ensure_neighborhood_tx(
        conn: sqlite3.Connection,
        *,
        slug: str,
        label: str,
        category: str,
        parent_slug: str | None = None,
        description: str = "",
        safety_class: str = "normal",
    ) -> str:
        neighborhood_id = f"neighborhood:{slug}"
        parent_id = f"neighborhood:{parent_slug}" if parent_slug else None
        conn.execute(
            """INSERT OR IGNORE INTO memory_neighborhoods(
                 neighborhood_id,slug,label,category,parent_neighborhood_id,
                 description,safety_class,created_at
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (
                neighborhood_id,
                slug,
                normalize_text(label)[:120],
                normalize_text(category)[:80] or "topic",
                parent_id,
                normalize_text(description)[:500],
                safety_class,
                utc_now(),
            ),
        )
        return neighborhood_id

    @staticmethod
    def _admitted_dynamic_neighborhoods_tx(
        conn: sqlite3.Connection,
        *,
        record_evaluations: bool = False,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Return stable named projects/services that earned map-level grouping.

        A name is not a neighborhood merely because it appeared in text or in
        one approved record. It needs five active operator-approved canonical
        memories across at least two sessions or source records. The original
        scope metadata remains searchable even when the name stays below this
        deliberately conservative boundary.
        """

        evaluations: list[dict[str, Any]] = []

        def admitted(
            rows: Sequence[sqlite3.Row],
            *,
            category: str,
            name_key: str,
            status_key: str | None = None,
        ) -> dict[str, str]:
            buckets: dict[str, dict[str, Any]] = {}
            for row in rows:
                label = normalize_text(str(row[name_key] or ""))[:120]
                if not label:
                    continue
                inactive = False
                if status_key:
                    status = normalize_text(str(row[status_key] or "active")).casefold()
                    if status in _INACTIVE_PROJECT_STATUSES:
                        inactive = True
                key = label.casefold()
                bucket = buckets.setdefault(
                    key,
                    {
                        "label": label,
                        "memories": set(),
                        "contexts": set(),
                        "inactive": False,
                    },
                )
                bucket["inactive"] = bool(bucket["inactive"] or inactive)
                if inactive:
                    continue
                bucket["memories"].add(str(row["id"]))
                session_id = normalize_text(str(row["session_id"] or ""))
                source_ref = normalize_text(str(row["source_ref"] or ""))
                if session_id:
                    bucket["contexts"].add(f"session:{session_id}")
                elif source_ref:
                    bucket["contexts"].add(f"source:{source_ref}")
            result: dict[str, str] = {}
            for key, bucket in buckets.items():
                memory_count = len(bucket["memories"])
                context_count = len(bucket["contexts"])
                if bucket["inactive"]:
                    decision = "inactive"
                    reason = (
                        f"Held back because {bucket['label']} is marked paused, inactive, "
                        "stopped, cancelled, or abandoned."
                    )
                elif (
                    memory_count >= DYNAMIC_NEIGHBORHOOD_MIN_MEMORIES
                    and context_count >= DYNAMIC_NEIGHBORHOOD_MIN_CONTEXTS
                ):
                    decision = "admitted"
                    reason = (
                        f"Admitted after {memory_count} approved memories across "
                        f"{context_count} independent contexts."
                    )
                    result[key] = str(bucket["label"])
                else:
                    decision = "held"
                    reason = (
                        f"Held at {memory_count}/{DYNAMIC_NEIGHBORHOOD_MIN_MEMORIES} approved "
                        f"memories and {context_count}/{DYNAMIC_NEIGHBORHOOD_MIN_CONTEXTS} "
                        "independent contexts."
                    )
                evaluations.append(
                    {
                        "category": category,
                        "name_key": key,
                        "display_name": str(bucket["label"]),
                        "memory_count": memory_count,
                        "context_count": context_count,
                        "decision": decision,
                        "reason": reason,
                    }
                )
            return result

        project_rows = conn.execute(
            """SELECT id,session_id,source_ref,
                      COALESCE(json_extract(scope_json,'$.project'),
                               json_extract(scope_json,'$.active_project')) project_name,
                      COALESCE(json_extract(scope_json,'$.project_status'),
                               json_extract(scope_json,'$.status'),'active') project_status
               FROM memories
               WHERE state IN ('active','cold')
                 AND approval_state='operator_approved'
                 AND record_role<>'reference'
                 AND COALESCE(json_extract(scope_json,'$.project'),
                              json_extract(scope_json,'$.active_project')) IS NOT NULL"""
        ).fetchall()
        service_rows = conn.execute(
            """SELECT m.id,m.session_id,m.source_ref,j.value service_name
               FROM memories m JOIN json_each(m.applicable_systems_json) j
               WHERE m.state IN ('active','cold')
                 AND m.approval_state='operator_approved'
                 AND m.record_role<>'reference'"""
        ).fetchall()
        admitted_projects = admitted(
            project_rows,
            category="project",
            name_key="project_name",
            status_key="project_status",
        )
        admitted_services = admitted(
            service_rows,
            category="service",
            name_key="service_name",
        )
        if record_evaluations:
            now = utc_now()
            current = {
                (str(row["category"]), str(row["name_key"])): dict(row)
                for row in conn.execute("SELECT * FROM memory_neighborhood_evaluations").fetchall()
            }
            seen: set[tuple[str, str]] = set()
            for evaluation in evaluations:
                identity = (evaluation["category"], evaluation["name_key"])
                seen.add(identity)
                prior = current.get(identity)
                changed = not prior or any(
                    prior.get(key) != evaluation[key]
                    for key in ("display_name", "memory_count", "context_count", "decision", "reason")
                )
                if changed:
                    conn.execute(
                        """INSERT INTO memory_neighborhood_decision_events(
                             event_id,category,name_key,display_name,prior_decision,decision,
                             memory_count,context_count,reason,created_at
                           ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                        (
                            str(uuid.uuid4()),
                            evaluation["category"],
                            evaluation["name_key"],
                            evaluation["display_name"],
                            prior.get("decision") if prior else None,
                            evaluation["decision"],
                            evaluation["memory_count"],
                            evaluation["context_count"],
                            evaluation["reason"],
                            now,
                        ),
                    )
                conn.execute(
                    """INSERT INTO memory_neighborhood_evaluations(
                         category,name_key,display_name,memory_count,context_count,
                         decision,reason,evaluated_at
                       ) VALUES(?,?,?,?,?,?,?,?)
                       ON CONFLICT(category,name_key) DO UPDATE SET
                         display_name=excluded.display_name,
                         memory_count=excluded.memory_count,
                         context_count=excluded.context_count,
                         decision=excluded.decision,
                         reason=excluded.reason,
                         evaluated_at=excluded.evaluated_at""",
                    (
                        evaluation["category"],
                        evaluation["name_key"],
                        evaluation["display_name"],
                        evaluation["memory_count"],
                        evaluation["context_count"],
                        evaluation["decision"],
                        evaluation["reason"],
                        now,
                    ),
                )
            for identity, prior in current.items():
                if identity in seen:
                    continue
                reason = "Removed because no currently eligible approved memory supports this name."
                conn.execute(
                    """INSERT INTO memory_neighborhood_decision_events(
                         event_id,category,name_key,display_name,prior_decision,decision,
                         memory_count,context_count,reason,created_at
                       ) VALUES(?,?,?,?,?,'inactive',0,0,?,?)""",
                    (
                        str(uuid.uuid4()),
                        identity[0],
                        identity[1],
                        str(prior["display_name"]),
                        str(prior["decision"]),
                        reason,
                        now,
                    ),
                )
                conn.execute(
                    "DELETE FROM memory_neighborhood_evaluations WHERE category=? AND name_key=?",
                    identity,
                )
        return admitted_projects, admitted_services

    def _assign_memory_neighborhoods_tx(
        self,
        conn: sqlite3.Connection,
        memory: dict[str, Any],
        *,
        admitted_projects: dict[str, str] | None = None,
        admitted_services: dict[str, str] | None = None,
    ) -> None:
        """Assign overlapping, explainable neighborhoods without creating graph edges."""

        now = utc_now()
        memberships: dict[str, tuple[float, str]] = {}

        def assign(slug: str, confidence: float, explanation: str) -> None:
            memberships[slug] = max(
                memberships.get(slug, (0.0, explanation)),
                (confidence, explanation),
                key=lambda item: item[0],
            )

        kind = str(memory.get("kind") or "semantic").casefold()
        kind_slugs = {
            "preference": "preferences",
            "procedure": "procedures",
            "decision": "decisions",
            "prospective": "plans",
            "operational": "operations",
            "identity": "people",
        }
        if kind in kind_slugs:
            assign(kind_slugs[kind], 0.96, f"The record type is {kind}.")
        source_type = str(memory.get("source_type") or "").casefold()
        if source_type.startswith("tool_"):
            assign("tools", 0.98, "This record came from a tool workflow.")
        if source_type == "vault_markdown" or str(memory.get("record_role")) == "reference":
            assign("references", 0.95, "This record is preserved source evidence.")

        content = normalize_text(str(memory.get("content") or ""))
        if _is_credential_reference(content):
            assign(
                "credentials",
                0.99,
                "This is a credential-location reference. Secret values are not allowed in Cortex.",
            )
            assign("tools", 0.84, "Credential references are used through authorized tools.")

        if admitted_projects is None or admitted_services is None:
            admitted_projects, admitted_services = self._admitted_dynamic_neighborhoods_tx(conn)

        scope = _trace_json_object(memory.get("scope_json"))
        project = normalize_text(str(scope.get("project") or scope.get("active_project") or ""))
        if project:
            assign("projects", 0.78, "This record has explicit project scope.")
            project_status = normalize_text(
                str(scope.get("project_status") or scope.get("status") or "active")
            ).casefold()
            admitted_project = admitted_projects.get(project.casefold())
            if admitted_project and project_status not in _INACTIVE_PROJECT_STATUSES:
                slug = f"project-{self._neighborhood_slug(admitted_project)}"
                self._ensure_neighborhood_tx(
                    conn,
                    slug=slug,
                    label=f"Project: {admitted_project}",
                    category="project",
                    parent_slug="projects",
                    description=(
                        "A stable project grouping supported by repeated approved memories "
                        "across independent contexts."
                    ),
                )
                assign(
                    slug,
                    0.94,
                    f"Repeated approved memories establish {admitted_project} as a stable project.",
                )
        systems = _trace_json_array(memory.get("applicable_systems_json"))
        if systems:
            assign("services", 0.76, "This record explicitly names an applicable system.")
        for raw_system in systems[:8]:
            system = normalize_text(str(raw_system))[:120]
            if not system:
                continue
            admitted_service = admitted_services.get(system.casefold())
            if admitted_service:
                slug = f"service-{self._neighborhood_slug(admitted_service)}"
                self._ensure_neighborhood_tx(
                    conn,
                    slug=slug,
                    label=f"Service: {admitted_service}",
                    category="service",
                    parent_slug="services",
                    description=(
                        "A stable service grouping supported by repeated approved memories "
                        "across independent contexts."
                    ),
                )
                assign(
                    slug,
                    0.94,
                    f"Repeated approved memories establish {admitted_service} as a stable service.",
                )

        for slug, (confidence, explanation) in memberships.items():
            conn.execute(
                """INSERT OR IGNORE INTO memory_neighborhood_memberships(
                     memory_id,neighborhood_id,confidence,origin,explanation,created_at
                   ) VALUES(?,?,?,'deterministic',?,?)""",
                (str(memory["id"]), f"neighborhood:{slug}", confidence, explanation, now),
            )

    def _refresh_dynamic_neighborhoods_tx(self, conn: sqlite3.Connection) -> None:
        """Rebuild generated names from current stable evidence, preserving memories."""

        conn.execute("DELETE FROM memory_neighborhoods WHERE category IN ('project','service')")
        admitted_projects, admitted_services = self._admitted_dynamic_neighborhoods_tx(
            conn, record_evaluations=True
        )
        for row in conn.execute("SELECT * FROM memories").fetchall():
            self._assign_memory_neighborhoods_tx(
                conn,
                dict(row),
                admitted_projects=admitted_projects,
                admitted_services=admitted_services,
            )

    def _backfill_memory_neighborhoods(self) -> None:
        base = (
            ("knowledge", "Knowledge", "root", None, "Durable approved knowledge.", "normal"),
            ("tools", "Tool use", "root", None, "Tools, workflows, and authorized integrations.", "normal"),
            (
                "credentials",
                "Credential references",
                "security",
                "tools",
                "Locations and authorized retrieval paths only; never secret values.",
                "secret_reference_only",
            ),
            ("projects", "Projects", "root", None, "Project-scoped knowledge.", "normal"),
            ("services", "Services", "root", None, "Service- and system-scoped knowledge.", "normal"),
            ("preferences", "Preferences", "knowledge", "knowledge", "Stable operator preferences.", "normal"),
            ("procedures", "Procedures", "knowledge", "knowledge", "Verified reusable procedures.", "normal"),
            ("decisions", "Decisions", "knowledge", "knowledge", "Durable choices and rationale.", "normal"),
            ("plans", "Plans", "knowledge", "knowledge", "Open prospective commitments.", "normal"),
            ("operations", "Operations", "knowledge", "tools", "Operational facts and constraints.", "normal"),
            ("people", "People", "knowledge", "knowledge", "Identity and relationship facts.", "normal"),
            ("references", "Reference evidence", "evidence", None, "Lookup-only source evidence.", "normal"),
        )
        for slug, label, category, parent, description, safety_class in base:
            self._ensure_neighborhood_tx(
                self._conn,
                slug=slug,
                label=label,
                category=category,
                parent_slug=parent,
                description=description,
                safety_class=safety_class,
            )
        # Recompute deterministic groupings on every upgrade/open. This removes
        # named project/service labels produced by older one-mention heuristics
        # while leaving the underlying memories and their metadata untouched.
        self._conn.execute(
            "DELETE FROM memory_neighborhood_memberships WHERE origin='deterministic'"
        )
        self._refresh_dynamic_neighborhoods_tx(self._conn)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._conn.in_transaction:
                savepoint = "cortex_" + uuid.uuid4().hex
                self._conn.execute(f"SAVEPOINT {savepoint}")
                try:
                    yield self._conn
                    self._conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                except Exception:
                    self._conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    self._conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                    raise
                return
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def assess_storage_candidate(
        self,
        content: str,
        *,
        kind: str = "semantic",
        context_mode: str = "standalone",
        scope: dict[str, Any] | None = None,
        entities: Sequence[str] | None = None,
        preconditions: dict[str, Any] | None = None,
        source_context: str | None = None,
        applicable_systems: Sequence[str] | None = None,
        applicable_versions: Sequence[str] | None = None,
        source_type: str = "conversation",
        source_category: str = "AGENT_INFERENCE",
        extraction_method: str = "unknown",
        confidence: float = 0.65,
        importance: float = 0.5,
        uniqueness: float = 1.0,
        volatility: float = 0.4,
        subject: str | None = None,
        predicate: str | None = None,
        object_value: str | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        automatic: bool = False,
    ) -> dict[str, Any]:
        """Evaluate a proposed write before mutation using inspectable signals."""

        clean = normalize_text(content)
        if not clean:
            raise ValueError("memory content cannot be empty")
        mode = _normalize_context_mode(context_mode)
        scope_value = _normalize_context_map(scope)
        entity_values = _normalize_context_list(entities)
        precondition_values = _normalize_context_map(preconditions)
        systems = _normalize_context_list(applicable_systems)
        versions = _normalize_context_list(applicable_versions)
        source_context_value = normalize_text(source_context or "")[:1000] or None
        completeness = _memory_metadata_completeness(
            mode,
            scope=scope_value,
            entities=entity_values,
            preconditions=precondition_values,
            source_context=source_context_value,
            applicable_systems=systems,
            applicable_versions=versions,
        )
        independently_understandable = not bool(_UNRESOLVED_REFERENCE.search(clean))
        if not independently_understandable and entity_values and source_context_value:
            independently_understandable = True
        source_type_value = normalize_text(source_type or "conversation").casefold()
        source_category_value = normalize_text(source_category or "AGENT_INFERENCE").upper()
        extraction_value = normalize_text(extraction_method or "unknown").casefold()
        semantic_token_count = len(query_tokens(clean))
        quality_flags: list[str] = []
        if source_type_value in _TOOL_TELEMETRY_SOURCE_TYPES or extraction_value in {
            "tool_outcome_observer_v1",
            "tool_outcome_aggregator_v1",
            "tool_workflow_aggregator_v1",
        }:
            quality_flags.append("dedicated_telemetry_not_memory")
        if _is_transient_automation_noise(clean, kind=kind, source_type=source_type_value):
            quality_flags.append("transient_automation_status")
        if _is_placeholder_only_content(clean):
            quality_flags.append("placeholder_content")
        if semantic_token_count < 5:
            quality_flags.append("low_information_density")
        if source_category_value == "AGENT_INFERENCE" and not source_context_value:
            quality_flags.append("inferred_without_source_context")
        type_prior = {
            "identity": 0.92,
            "preference": 0.88,
            "procedure": 0.84,
            "prospective": 0.80,
            "decision": 0.74,
            "semantic": 0.64,
            "operational": 0.54,
            "episode": 0.38,
            "schema": 0.86,
        }.get(kind, 0.58)
        reusable_score = _clamp(
            0.34 * type_prior
            + 0.24 * _clamp(importance)
            + 0.18 * _clamp(confidence)
            + 0.12 * _clamp(uniqueness)
            + 0.12 * (1.0 - _clamp(volatility))
        )
        durability = (
            "temporary"
            if kind == "episode"
            or float(volatility) >= 0.70
            or (kind == "operational" and bool(valid_to))
            or "dedicated_telemetry_not_memory" in quality_flags
            or "transient_automation_status" in quality_flags
            else "durable"
        )
        minimum_reusable_score = 0.45
        if automatic:
            minimum_reusable_score = 0.62 if durability == "temporary" else 0.52
        digest = content_hash(clean)
        scope_json = _trace_json(scope_value)
        preconditions_json = _trace_json(precondition_values)
        systems_json = _trace_json(systems)
        versions_json = _trace_json(versions)
        with self._lock:
            duplicate = self._conn.execute(
                """SELECT id FROM memories WHERE content_hash=? AND context_mode=? AND scope_json=?
                     AND preconditions_json=? AND applicable_systems_json=? AND applicable_versions_json=?
                   ORDER BY created_at LIMIT 1""",
                (digest, mode, scope_json, preconditions_json, systems_json, versions_json),
            ).fetchone()
            contradictions: list[str] = []
            if subject and predicate and object_value is not None:
                rows = self._conn.execute(
                    """SELECT id,object_value,valid_from,valid_to FROM memories
                       WHERE subject=? AND predicate=? AND state IN ('active','cold')""",
                    (subject, predicate),
                ).fetchall()
                contradictions = [
                    str(row["id"])
                    for row in rows
                    if str(row["object_value"] or "") != str(object_value)
                    and _periods_overlap(valid_from, valid_to, row["valid_from"], row["valid_to"])
                ]
        duplicate_id = str(duplicate["id"]) if duplicate else None
        if duplicate_id and "duplicate_candidate" not in quality_flags:
            quality_flags.append("duplicate_candidate")
        decision = "updated" if duplicate_id else "created"
        reason = (
            "exact candidate in the same context should update the existing memory"
            if duplicate_id
            else "candidate is sufficiently reusable and independently understandable"
        )
        if not independently_understandable:
            decision = "ignored" if automatic else "created"
            reason = (
                "unresolved reference cannot be understood independently"
                if automatic
                else "explicit write contains an unresolved reference; add entities and source context"
            )
        elif automatic and "dedicated_telemetry_not_memory" in quality_flags:
            decision = "ignored"
            reason = "tool execution telemetry belongs in the dedicated tool ledger, not recallable memory"
        elif automatic and "transient_automation_status" in quality_flags:
            decision = "ignored"
            reason = "transient execution status is not independently reusable knowledge"
        elif automatic and "placeholder_content" in quality_flags:
            decision = "ignored"
            reason = "placeholder text does not contain reusable knowledge"
        elif automatic and semantic_token_count < 5 and kind not in {"identity", "preference", "prospective"}:
            decision = "ignored"
            reason = "candidate has too little specific information to justify long-term storage"
        elif automatic and reusable_score < minimum_reusable_score:
            decision = "ignored"
            reason = (
                f"candidate reusable score {reusable_score:.2f} is below the "
                f"{minimum_reusable_score:.2f} automatic {durability} threshold"
            )
        elif mode == "context_dependent" and completeness < 0.50:
            decision = "ignored" if automatic else "created"
            reason = "context-dependent candidate lacks enough explicit scope metadata"
        elif contradictions:
            reason = "candidate may contradict active structured evidence and requires review"
        policy_effect = self.active_policy_adjustment(
            "admission",
            {
                "kind": kind,
                "source_type": source_type_value,
                "source_category": source_category_value,
                "quality_flags": quality_flags,
            },
        )
        if automatic and policy_effect.get("automatic_action") == "ignore":
            decision = "ignored"
            reason = (
                "an operator-trained admission policy blocks this matching automatic write; "
                f"policy versions: {', '.join(policy_effect['matched_versions'])}"
            )
        return {
            "candidate_hash": digest,
            "kind": kind,
            "source_type": source_type_value,
            "context_mode": mode,
            "reusable_score": round(reusable_score, 6),
            "likely_useful_again": reusable_score >= minimum_reusable_score,
            "durability": durability,
            "quality_flags": quality_flags,
            "semantic_token_count": semantic_token_count,
            "duplicate_memory_id": duplicate_id,
            "contradiction_ids": contradictions,
            "independently_understandable": independently_understandable,
            "metadata_completeness": completeness,
            "decision": decision,
            "reason": reason,
            "policy_effect": policy_effect,
        }

    def record_ignored_memory_candidate(
        self,
        assessment: dict[str, Any],
        *,
        session_id: str | None,
    ) -> str:
        if str(assessment.get("decision")) != "ignored":
            raise ValueError("only ignored storage assessments can be recorded without a memory")
        with self.transaction() as conn:
            return self._record_memory_write_decision_tx(
                conn,
                assessment,
                session_id=session_id,
                memory_id=None,
            )

    def propose_memory_creation(
        self,
        content: str,
        *,
        kind: str = "semantic",
        source_type: str = "conversation",
        source_category: str = "AGENT_INFERENCE",
        source_ref: str | None = None,
        session_id: str | None = None,
        context_mode: str = "standalone",
        scope: dict[str, Any] | None = None,
        entities: Sequence[str] | None = None,
        preconditions: dict[str, Any] | None = None,
        source_context: str | None = None,
        applicable_systems: Sequence[str] | None = None,
        applicable_versions: Sequence[str] | None = None,
        observed_at: str | None = None,
        confidence: float = 0.65,
        currentness_confidence: float = 0.75,
        importance: float = 0.5,
        uniqueness: float = 1.0,
        volatility: float = 0.4,
        trust: float = 0.7,
        valid_from: str | None = None,
        valid_to: str | None = None,
        subject: str | None = None,
        predicate: str | None = None,
        object_value: str | None = None,
        supersedes_id: str | None = None,
        evidence_ids: Sequence[str] | None = None,
        extraction_method: str = "unknown",
        quarantine_reason: str | None = None,
        storage_policy: str = "automatic",
        assessment: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Stage a non-recallable candidate for human review.

        Equivalent pending candidates coalesce into one review item.  This
        table is deliberately separate from ``memories`` and the FTS index, so
        staging a proposal can never make it available to recall.
        """

        sanitized = sanitize_memory(str(content or ""))
        clean = normalize_text(sanitized.text)[:8000]
        if not clean:
            raise ValueError("memory creation proposal content cannot be empty")
        kind_value = normalize_text(kind or "semantic").casefold()[:80] or "semantic"
        source_type_value = normalize_text(source_type or "conversation").casefold()[:120]
        source_category_value = normalize_text(source_category or "AGENT_INFERENCE").upper()[:120]
        source_ref_sanitized = sanitize_memory(str(source_ref or ""))
        source_ref_value = normalize_text(source_ref_sanitized.text)[:500] or None
        session_value = normalize_text(session_id or "")[:200] or None
        mode = _normalize_context_mode(context_mode)
        scope_value = _sanitize_creation_context_map(scope)
        entity_values = _sanitize_creation_context_list(entities)
        precondition_values = _sanitize_creation_context_map(preconditions)
        systems = _sanitize_creation_context_list(applicable_systems)
        versions = _sanitize_creation_context_list(applicable_versions)
        source_context_sanitized = sanitize_memory(str(source_context or ""))
        source_context_value = normalize_text(source_context_sanitized.text)[:1000] or None
        extraction_value = normalize_text(extraction_method or "unknown").casefold()[:120]
        policy_value = normalize_text(storage_policy or "automatic").casefold()[:80] or "automatic"
        supplied_quarantine = normalize_text(quarantine_reason or "")[:500]
        quarantine_reasons = list(
            dict.fromkeys(
                value
                for value in (
                    supplied_quarantine,
                    sanitized.quarantine_reason,
                    source_context_sanitized.quarantine_reason,
                    (
                        "secret value removed; save only a reference to an approved secret manager"
                        if sanitized.redacted
                        or source_ref_sanitized.redacted
                        or source_context_sanitized.redacted
                        else None
                    ),
                )
                if value
            )
        )
        quarantine_value = ", ".join(quarantine_reasons)[:500] or None
        redacted = bool(
            sanitized.redacted or source_ref_sanitized.redacted or source_context_sanitized.redacted
        )
        digest = content_hash(clean)
        candidate = {
            "kind": kind_value,
            "context_mode": mode,
            "scope": scope_value,
            "entities": entity_values,
            "preconditions": precondition_values,
            "source_context": source_context_value,
            "applicable_systems": systems,
            "applicable_versions": versions,
            "observed_at": normalize_text(observed_at or "")[:80] or None,
            "confidence": _clamp(confidence),
            "currentness_confidence": _clamp(currentness_confidence),
            "importance": _clamp(importance),
            "uniqueness": _clamp(uniqueness),
            "volatility": _clamp(volatility),
            "trust": _clamp(trust),
            "valid_from": normalize_text(valid_from or "")[:80] or None,
            "valid_to": normalize_text(valid_to or "")[:80] or None,
            "subject": sanitize_memory(str(subject or "")).text[:500] or None,
            "predicate": sanitize_memory(str(predicate or "")).text[:500] or None,
            "object_value": sanitize_memory(str(object_value or "")).text[:1000] or None,
            "supersedes_id": normalize_text(supersedes_id or "")[:100] or None,
            "evidence_ids": _normalize_context_list(evidence_ids)[:50],
            "extraction_method": extraction_value,
        }
        if assessment is None:
            assessment_value = self.assess_storage_candidate(
                clean,
                kind=kind_value,
                context_mode=mode,
                scope=scope_value,
                entities=entity_values,
                preconditions=precondition_values,
                source_context=source_context_value,
                applicable_systems=systems,
                applicable_versions=versions,
                source_type=source_type_value,
                source_category=source_category_value,
                extraction_method=extraction_value,
                confidence=float(candidate["confidence"]),
                importance=float(candidate["importance"]),
                uniqueness=float(candidate["uniqueness"]),
                volatility=float(candidate["volatility"]),
                subject=candidate["subject"],
                predicate=candidate["predicate"],
                object_value=candidate["object_value"],
                valid_from=candidate["valid_from"],
                valid_to=candidate["valid_to"],
                automatic=policy_value == "automatic",
            )
        else:
            assessment_value = _bounded_creation_assessment(assessment)
        assessment_value = {
            **assessment_value,
            "candidate_hash": digest,
            "kind": kind_value,
            "source_type": source_type_value,
            "context_mode": mode,
        }
        proposal_material = {
            "candidate_hash": digest,
            "kind": kind_value,
            "source_type": source_type_value,
            "source_category": source_category_value,
            "source_ref": source_ref_value or "",
            "context_mode": mode,
            "scope": scope_value,
            "preconditions": precondition_values,
            "applicable_systems": systems,
            "applicable_versions": versions,
        }
        proposal_key = hashlib.sha256(_trace_json(proposal_material).encode("utf-8")).hexdigest()
        now = utc_now()
        with self.transaction() as conn:
            existing = conn.execute(
                """SELECT proposal_id, status FROM memory_creation_proposals
                   WHERE proposal_key=? AND status IN ('pending','needs_context',
                       'remembered','evidence_only','rejected')""",
                (proposal_key,),
            ).fetchone()
            if existing:
                proposal_id = str(existing["proposal_id"])
                existing_status = str(existing["status"])
                new_status = "pending" if existing_status in (
                    "remembered", "evidence_only", "rejected"
                ) else existing_status
                conn.execute(
                    """UPDATE memory_creation_proposals
                       SET status=?,recurrence_count=recurrence_count+1,last_seen_at=?,
                           session_id=COALESCE(?,session_id),
                           assessment_json=?,quarantine_reason=COALESCE(quarantine_reason,?),
                           redacted=MAX(redacted,?),review_started_at=NULL
                       WHERE proposal_id=?""",
                    (
                        new_status,
                        now,
                        session_value,
                        _trace_json(assessment_value),
                        quarantine_value,
                        int(redacted),
                        proposal_id,
                    ),
                )
                created = False
            else:
                proposal_id = str(uuid.uuid4())
                conn.execute(
                    """INSERT INTO memory_creation_proposals(
                         proposal_id,proposal_key,candidate_hash,content,kind,source_type,source_category,
                         source_ref,session_id,context_mode,candidate_json,assessment_json,storage_policy,
                         quarantine_reason,redacted,first_seen_at,last_seen_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        proposal_id,
                        proposal_key,
                        digest,
                        clean,
                        kind_value,
                        source_type_value,
                        source_category_value,
                        source_ref_value,
                        session_value,
                        mode,
                        _trace_json(candidate),
                        _trace_json(assessment_value),
                        policy_value,
                        quarantine_value,
                        int(redacted),
                        now,
                        now,
                    ),
                )
                created = True
            occurrence_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO memory_experience_events(
                     event_id,event_type,proposal_id,session_id,source_type,source_category,
                     evidence_key,event_day,metadata_json,created_at
                   ) VALUES(?,'candidate_observed',?,?,?,?,?,?,?,?)""",
                (
                    occurrence_id,
                    proposal_id,
                    session_value,
                    source_type_value,
                    source_category_value,
                    f"candidate:{occurrence_id}",
                    now[:10],
                    _trace_json(
                        {
                            "context_mode": mode,
                            "kind": kind_value,
                            "reused_pending_proposal": not created,
                        }
                    ),
                    now,
                ),
            )
        proposal = self.get_memory_creation_proposal(proposal_id)
        if not proposal:  # pragma: no cover - committed row must be readable
            raise RuntimeError("memory creation proposal was not persisted")
        return {**proposal, "created": created}

    def get_memory_creation_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memory_creation_proposals WHERE proposal_id=?",
                (normalize_text(proposal_id),),
            ).fetchone()
        return _decode_creation_proposal(row) if row else None

    def _has_active_review_started_at(
        self, proposal_id: str, conn: sqlite3.Connection | None = None
    ) -> bool:
        """True if another reviewer is currently working on this proposal.

        Uses the current transaction connection when available so that the
        advisory flag is observed consistently with the rest of the review.
        """
        connection = conn if conn is not None else self._conn
        row = connection.execute(
            """SELECT review_started_at FROM memory_creation_proposals
               WHERE proposal_id=? AND status IN ('pending','needs_context')""",
            (normalize_text(proposal_id),),
        ).fetchone()
        if not row or not row["review_started_at"]:
            return False
        started = parse_iso8601(row["review_started_at"])
        if started is None:
            return False
        return (utc_now_dt() - started).total_seconds() < 300

    def list_memory_creation_proposals(
        self,
        *,
        status: str | None = "pending",
        limit: int = 200,
        offset: int = 0,
        oldest_first: bool = False,
    ) -> list[dict[str, Any]]:
        bounded = max(1, min(1000, int(limit)))
        start = max(0, int(offset))
        order = "ASC" if oldest_first else "DESC"
        values: tuple[Any, ...]
        if status is None or normalize_text(status).casefold() == "all":
            sql = (
                "SELECT * FROM memory_creation_proposals "
                f"ORDER BY last_seen_at {order}, rowid {order} LIMIT ? OFFSET ?"
            )
            values = (bounded, start)
        else:
            status_value = normalize_text(status).casefold()
            if status_value not in {"pending", "remembered", "evidence_only", "rejected", "needs_context"}:
                raise ValueError("invalid memory creation proposal status")
            sql = (
                "SELECT * FROM memory_creation_proposals WHERE status=? "
                f"ORDER BY last_seen_at {order}, rowid {order} LIMIT ? OFFSET ?"
            )
            values = (status_value, bounded, start)
        with self._lock:
            rows = self._conn.execute(sql, values).fetchall()
        return [_decode_creation_proposal(row) for row in rows]

    def record_memory_creation_feedback(
        self,
        proposal_ids: Sequence[str],
        *,
        session_id: str | None = None,
        strength: str = "positive",
    ) -> int:
        """Reinforce still-pending candidates after later positive user feedback.

        Only a compact signal is retained: which candidate was reinforced,
        whether the wording was strongly positive, and when it happened. Raw
        feedback text is deliberately excluded from this ledger. Feedback can
        influence a later judge, but it never makes a candidate recallable by
        itself and cannot revive an already-decided proposal.
        """

        strength_value = normalize_text(strength).casefold()
        if strength_value not in {"positive", "strong"}:
            raise ValueError("creation feedback strength must be positive or strong")
        ids = list(
            dict.fromkeys(
                normalize_text(str(proposal_id))[:100]
                for proposal_id in proposal_ids
                if normalize_text(str(proposal_id))
            )
        )[:50]
        if not ids:
            return 0
        session_value = normalize_text(session_id or "")[:200] or None
        now = utc_now()
        recorded = 0
        with self.transaction() as conn:
            for proposal_id in ids:
                waiting = conn.execute(
                    """SELECT 1 FROM memory_creation_proposals
                       WHERE proposal_id=? AND status IN ('pending','needs_context')""",
                    (proposal_id,),
                ).fetchone()
                if not waiting:
                    continue
                conn.execute(
                    """INSERT INTO memory_creation_feedback(
                         feedback_id,proposal_id,session_id,strength,source_type,created_at
                       ) VALUES(?,?,?,?,?,?)""",
                    (
                        str(uuid.uuid4()),
                        proposal_id,
                        session_value,
                        strength_value,
                        "implicit_user_feedback",
                        now,
                    ),
                )
                conn.execute(
                    """UPDATE memory_creation_proposals
                       SET positive_feedback_count=positive_feedback_count+1,
                           strong_feedback_count=strong_feedback_count+?,
                           last_feedback_at=?
                       WHERE proposal_id=?""",
                    (int(strength_value == "strong"), now, proposal_id),
                )
                recorded += 1
        return recorded

    def review_memory_creation(
        self,
        proposal_id: str,
        action: str,
        *,
        edited_content: str | None = None,
        reason_text: str = "",
        actor: str = "dashboard-operator",
        decision_scope: str = "item_only",
        approval_authority: str = "operator",
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        """Decide one staged candidate and apply its explicit recall boundary."""

        aliases = {
            "edited_remember": "remember_edited",
            "reference": "evidence_only",
        }
        action_value = aliases.get(normalize_text(action).casefold().replace(" ", "_"), normalize_text(action).casefold().replace(" ", "_"))
        if action_value not in {
            "remember",
            "remember_edited",
            "evidence_only",
            "reject",
            "needs_context",
        }:
            raise ValueError(
                "creation action must be remember, remember_edited, evidence_only, reject, or needs_context"
            )
        scope_value = _normalize_review_scope(decision_scope)
        actor_value = normalize_text(actor)[:80] or "dashboard-operator"
        authority_value = normalize_text(approval_authority).casefold() or "operator"
        if authority_value not in {"operator", "automatic"}:
            raise ValueError("creation approval authority must be operator or automatic")
        note_value = normalize_text(reason_text)[:1000]
        review_id = str(uuid.uuid4())
        memory_id: str | None = None
        memory_created = False
        memory_promoted = False
        promoted_memory_prior: dict[str, Any] | None = None
        remembered_content: str | None = None

        # Keep the trusted memory write and review ledger in one transaction.
        # Nested store writes use savepoints, so any finalization failure rolls
        # the entire admission back while the lock still serializes reviewers.
        with self.transaction():
            row = self._conn.execute(
                "SELECT * FROM memory_creation_proposals WHERE proposal_id=?",
                (normalize_text(proposal_id),),
            ).fetchone()
            if not row:
                logger.warning(
                    "review_memory_creation skipped: proposal %s does not exist", proposal_id
                )
                return {
                    "review_id": review_id,
                    "proposal_id": str(proposal_id),
                    "action": action_value,
                    "status": "skipped",
                    "memory_id": None,
                    "memory_created": False,
                    "memory_promoted": False,
                    "approval_authority": authority_value,
                }
            if str(row["status"]) not in {"pending", "needs_context"}:
                logger.warning(
                    "review_memory_creation skipped: proposal %s is no longer waiting for review (status=%s)",
                    proposal_id,
                    row["status"],
                )
                return {
                    "review_id": review_id,
                    "proposal_id": str(proposal_id),
                    "action": action_value,
                    "status": "skipped",
                    "memory_id": None,
                    "memory_created": False,
                    "memory_promoted": False,
                    "approval_authority": authority_value,
                }
            proposal = _decode_creation_proposal(row)
            if authority_value == "automatic":
                if not expected_revision:
                    raise ValueError("automatic creation review requires an expected revision")
                if proposal["status"] != "pending" or creation_proposal_revision(
                    proposal
                ) != normalize_text(expected_revision):
                    logger.warning(
                        "review_memory_creation skipped: proposal %s changed while automatic review was in flight",
                        proposal_id,
                    )
                    return {
                        "review_id": review_id,
                        "proposal_id": str(proposal_id),
                        "action": action_value,
                        "status": "skipped",
                        "memory_id": None,
                        "memory_created": False,
                        "memory_promoted": False,
                        "approval_authority": authority_value,
                    }
                if self._has_active_review_started_at(proposal_id, self._conn):
                    logger.warning(
                        "review_memory_creation skipped: proposal %s is already being reviewed",
                        proposal_id,
                    )
                    return {
                        "review_id": review_id,
                        "proposal_id": str(proposal_id),
                        "action": action_value,
                        "status": "skipped",
                        "memory_id": None,
                        "memory_created": False,
                        "memory_promoted": False,
                        "approval_authority": authority_value,
                    }
                self._conn.execute(
                    "UPDATE memory_creation_proposals SET review_started_at=? WHERE proposal_id=?",
                    (utc_now(), normalize_text(proposal_id)),
                )
            candidate = dict(proposal.get("candidate") or {})
            if action_value in {"remember", "remember_edited", "evidence_only"}:
                remembered_content = str(proposal["content"])
                approval_quarantine = normalize_text(str(proposal.get("quarantine_reason") or ""))
                if action_value == "remember_edited":
                    sanitized_edit = sanitize_memory(str(edited_content or ""))
                    remembered_content = normalize_text(sanitized_edit.text)[:8000]
                    if not remembered_content:
                        raise ValueError("edited memory content is required")
                    approval_quarantine = normalize_text(
                        sanitized_edit.quarantine_reason
                        or (
                            "secret value removed; save only a reference to an approved secret manager"
                            if sanitized_edit.redacted
                            else ""
                        )
                    )[:500]
                requested_eligibility = (
                    "evidence_only" if action_value == "evidence_only" else "primary"
                )
                duplicate_id = str((proposal.get("assessment") or {}).get("duplicate_memory_id") or "")
                duplicate = self.get_memory(duplicate_id) if duplicate_id else None
                duplicate_memory = dict(duplicate or {})
                exact_duplicate = bool(
                    duplicate
                    and action_value != "remember_edited"
                    and str(duplicate_memory.get("state") or "") in {"active", "cold"}
                    and content_hash(remembered_content)
                    == str(duplicate_memory.get("content_hash") or "")
                )
                if exact_duplicate:
                    memory_id = duplicate_id
                    if not self.is_memory_recall_eligible(
                        duplicate_id,
                        evidence_lookup=requested_eligibility == "evidence_only",
                    ):
                        active_recall_set = self._active_recall_set_tx(self._conn)
                        prior_membership = self._conn.execute(
                            """SELECT * FROM memory_recall_memberships
                               WHERE recall_set_id=? AND memory_id=?""",
                            (active_recall_set["recall_set_id"], duplicate_id),
                        ).fetchone()
                        promoted_memory_prior = {
                            "approval_state": duplicate_memory.get("approval_state"),
                            "origin_source_category": duplicate_memory.get("origin_source_category"),
                            "updated_at": duplicate_memory.get("updated_at"),
                            "membership": dict(prior_membership) if prior_membership else None,
                        }
                        promotion = self.promote_memory_to_active_recall_set(
                            duplicate_id,
                            eligibility=requested_eligibility,
                            actor=actor_value,
                            reason="Creation review approved an exact preserved legacy memory.",
                            review_id=review_id,
                            approval_authority=authority_value,
                            origin_source_category=str(
                                proposal.get("source_category") or "AGENT_INFERENCE"
                            ),
                        )
                        memory_promoted = bool(promotion["changed"])
                else:
                    # This is the sole call that may create indexed storage.
                    # Recall-set membership makes the result ordinary memory or
                    # lookup-only evidence atomically.
                    memory_id, memory_created = self.add_memory(
                        remembered_content,
                        kind=str(proposal.get("kind") or "semantic"),
                        source_type=str(proposal.get("source_type") or "conversation"),
                        source_category=(
                            "AUTOMATIC_APPROVED"
                            if authority_value == "automatic"
                            else "OPERATOR_APPROVED"
                        ),
                        origin_source_category=str(
                            proposal.get("source_category") or "AGENT_INFERENCE"
                        ),
                        approval_state=(
                            "automatic_approved"
                            if authority_value == "automatic"
                            else "operator_approved"
                        ),
                        source_ref=proposal.get("source_ref"),
                        session_id=proposal.get("session_id"),
                        context_mode=str(proposal.get("context_mode") or "standalone"),
                        scope=dict(candidate.get("scope") or {}),
                        entities=list(candidate.get("entities") or []),
                        preconditions=dict(candidate.get("preconditions") or {}),
                        source_context=candidate.get("source_context"),
                        applicable_systems=list(candidate.get("applicable_systems") or []),
                        applicable_versions=list(candidate.get("applicable_versions") or []),
                        observed_at=candidate.get("observed_at"),
                        confidence=float(candidate.get("confidence") or 0.65),
                        currentness_confidence=float(candidate.get("currentness_confidence") or 0.75),
                        importance=float(candidate.get("importance") or 0.5),
                        uniqueness=float(candidate.get("uniqueness") or 1.0),
                        volatility=float(candidate.get("volatility") or 0.4),
                        trust=float(candidate.get("trust") or 0.7),
                        quarantine_reason=approval_quarantine or None,
                        valid_from=candidate.get("valid_from"),
                        valid_to=candidate.get("valid_to"),
                        subject=candidate.get("subject"),
                        predicate=candidate.get("predicate"),
                        object_value=candidate.get("object_value"),
                        supersedes_id=candidate.get("supersedes_id"),
                        evidence_ids=list(candidate.get("evidence_ids") or []),
                        extraction_method=str(candidate.get("extraction_method") or "unknown"),
                        storage_policy=(
                            "automatic_approved"
                            if authority_value == "automatic"
                            else "operator_approved"
                        ),
                        record_role="reference" if action_value == "evidence_only" else "canonical",
                        recall_eligibility=requested_eligibility,
                        preserve_exact_duplicate=True,
                    )
                    if memory_created:
                        active_recall_set = self._active_recall_set_tx(self._conn)
                        membership_updated = self._conn.execute(
                            """UPDATE memory_recall_memberships
                               SET origin=?,review_id=?,actor=?,reason=?
                               WHERE recall_set_id=? AND memory_id=? AND revoked_at IS NULL""",
                            (
                                (
                                    "automatic_judgment"
                                    if authority_value == "automatic"
                                    else "operator_creation_review"
                                ),
                                review_id,
                                actor_value,
                                "Memory created by an approved creation review.",
                                active_recall_set["recall_set_id"],
                                memory_id,
                            ),
                        )
                        if membership_updated.rowcount != 1:
                            raise RuntimeError("approved creation did not produce one active recall membership")
                    if not self.is_memory_recall_eligible(
                        memory_id,
                        evidence_lookup=requested_eligibility == "evidence_only",
                    ):
                        # The duplicate exists but is not recall-eligible in the
                        # requested mode. Record the proposal as satisfied by the
                        # existing memory so it does not stay pending forever.
                        action_value = "evidence_only"
                        status_value = "evidence_only"
                        memory_created = False
                        memory_promoted = False
            status_value = {
                "remember": "remembered",
                "remember_edited": "remembered",
                "evidence_only": "evidence_only",
                "reject": "rejected",
                "needs_context": "needs_context",
            }[action_value]
            now = utc_now()
            effect = {
                "proposal_id": proposal["proposal_id"],
                "memory_id": memory_id,
                "memory_created": memory_created,
                "memory_promoted": memory_promoted,
                "edited": action_value == "remember_edited",
                "candidate_hash": proposal["candidate_hash"],
                "kind": proposal["kind"],
                "source_type": proposal["source_type"],
                "source_category": proposal["source_category"],
                "approval_authority": authority_value,
                "quality_flags": list((proposal.get("assessment") or {}).get("quality_flags") or []),
            }
            signal = {
                "item_type": "creation",
                "action": action_value,
                "reason_code": action_value,
                "decision_scope": scope_value,
                "kind": proposal["kind"],
                "source_type": proposal["source_type"],
                "source_category": proposal["source_category"],
                "approval_authority": authority_value,
                "quality_flags": effect["quality_flags"],
            }
            prior = {"status": proposal["status"], "content": proposal["content"]}
            if promoted_memory_prior:
                prior["memory"] = promoted_memory_prior
            with self.transaction() as conn:
                # Finalize the proposal before writing the decision row so that
                # the proposal_id foreign key in operator_review_decisions points
                # to an existing record.  With FOREIGN KEYS=ON, inserting a
                # decision for a proposal that was concurrently deleted/revoked
                # would otherwise raise an integrity error.
                updated = conn.execute(
                    """UPDATE memory_creation_proposals
                       SET status=?,decision_action=?,decision_note=?,result_memory_id=?,review_id=?,
                           actor=?,decided_at=?
                       WHERE proposal_id=? AND status IN ('pending','needs_context')""",
                    (
                        status_value,
                        action_value,
                        note_value or None,
                        memory_id,
                        review_id,
                        actor_value,
                        now,
                        proposal["proposal_id"],
                    ),
                )
                if not updated.rowcount:  # pragma: no cover - protected by store lock
                    raise RuntimeError("memory creation proposal changed during review")
                # When the review leaves no memory behind (reject, needs_context,
                # evidence_only with no duplicate), src_id must be NULL because
                # the existing FK expects the memory_id to exist in memories.
                src_id_value = memory_id if memory_id and self.get_memory(memory_id) else None
                conn.execute(
                    """INSERT INTO operator_review_decisions(
                       review_id,item_type,item_key,proposal_id,src_id,dst_id,action,reason_code,
                       reason_text,prior_json,effect_json,learning_signal_json,decision_scope,actor,created_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        review_id,
                        "creation",
                        f"creation:{proposal['proposal_id']}",
                        proposal["proposal_id"],
                        src_id_value,
                        None,
                        action_value,
                        action_value,
                        note_value or None,
                        _trace_json(prior),
                        _trace_json(effect),
                        _trace_json(signal),
                        scope_value,
                        actor_value,
                        now,
                    ),
                )
                self._compile_policy_candidates_tx(conn)
        return {
            "review_id": review_id,
            "proposal_id": str(proposal["proposal_id"]),
            "action": action_value,
            "status": status_value,
            "memory_id": memory_id,
            "memory_created": memory_created,
            "memory_promoted": memory_promoted,
            "approval_authority": authority_value,
        }

    def decide_memory_creation(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Compatibility alias for dashboard/API callers."""

        return self.review_memory_creation(*args, **kwargs)

    @staticmethod
    def _record_memory_write_decision_tx(
        conn: sqlite3.Connection,
        assessment: dict[str, Any],
        *,
        session_id: str | None,
        memory_id: str | None,
    ) -> str:
        decision_id = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO memory_write_decisions(
                 decision_id,session_id,candidate_hash,kind,source_type,context_mode,
                 reusable_score,durability,quality_flags_json,duplicate_memory_id,
                 contradiction_ids_json,independently_understandable,decision,reason,memory_id,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                decision_id,
                session_id,
                str(assessment.get("candidate_hash") or ""),
                str(assessment.get("kind") or "semantic"),
                str(assessment.get("source_type") or "conversation"),
                str(assessment.get("context_mode") or "standalone"),
                _clamp(float(assessment.get("reusable_score") or 0.0)),
                str(assessment.get("durability") or "durable"),
                _trace_json(list(assessment.get("quality_flags") or [])),
                assessment.get("duplicate_memory_id"),
                _trace_json(list(assessment.get("contradiction_ids") or [])),
                int(bool(assessment.get("independently_understandable"))),
                str(assessment.get("decision") or "ignored"),
                normalize_text(str(assessment.get("reason") or "No storage reason was recorded."))[:600],
                memory_id,
                utc_now(),
            ),
        )
        return decision_id

    def add_memory(
        self,
        content: str,
        *,
        kind: str = "semantic",
        source_type: str = "conversation",
        source_category: str = "AGENT_INFERENCE",
        origin_source_category: str | None = None,
        approval_state: str | None = None,
        source_ref: str | None = None,
        session_id: str | None = None,
        context_mode: str = "standalone",
        scope: dict[str, Any] | None = None,
        entities: Sequence[str] | None = None,
        preconditions: dict[str, Any] | None = None,
        source_context: str | None = None,
        applicable_systems: Sequence[str] | None = None,
        applicable_versions: Sequence[str] | None = None,
        observed_at: str | None = None,
        confidence: float = 0.65,
        currentness_confidence: float = 0.75,
        importance: float = 0.5,
        uniqueness: float = 1.0,
        volatility: float = 0.4,
        trust: float = 0.7,
        pinned: bool = False,
        protected: bool = False,
        state: str = "active",
        quarantine_reason: str | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        subject: str | None = None,
        predicate: str | None = None,
        object_value: str | None = None,
        extraction_method: str = "unknown",
        supersedes_id: str | None = None,
        evidence_ids: Sequence[str] | None = None,
        storage_policy: str = "trusted",
        record_role: str | None = None,
        recall_eligibility: str | None = None,
        preserve_exact_duplicate: bool = False,
    ) -> tuple[str, bool]:
        sanitized_content = sanitize_memory(str(content or ""))
        content = normalize_text(sanitized_content.text)
        if not content:
            raise ValueError("memory content cannot be empty")
        quarantine_parts = [
            normalize_text(quarantine_reason or ""),
            normalize_text(sanitized_content.quarantine_reason or ""),
            (
                "secret value removed; save only a reference to an approved secret manager"
                if sanitized_content.redacted
                else ""
            ),
        ]
        quarantine_reason = ", ".join(dict.fromkeys(part for part in quarantine_parts if part)) or None
        digest = content_hash(content)
        now = utc_now()
        state = "quarantine" if quarantine_reason else state
        context_mode_value = _normalize_context_mode(context_mode)
        scope_value = _normalize_context_map(scope)
        entity_values = _normalize_context_list(entities)
        precondition_values = _normalize_context_map(preconditions)
        system_values = _normalize_context_list(applicable_systems)
        version_values = _normalize_context_list(applicable_versions)
        source_context_value = normalize_text(source_context or "")[:1000] or None
        origin_source_category_value = normalize_text(
            origin_source_category or source_category or "AGENT_INFERENCE"
        ).upper()[:120] or "AGENT_INFERENCE"
        approval_state_value = normalize_text(
            approval_state
            or (
                "operator_approved"
                if source_category == "OPERATOR_APPROVED"
                else "automatic_approved"
                if source_category == "AUTOMATIC_APPROVED"
                else "unreviewed"
            )
        ).casefold()[:40]
        if approval_state_value not in {
            "unreviewed",
            "operator_approved",
            "automatic_approved",
            "trusted_import",
        }:
            raise ValueError(
                "approval_state must be unreviewed, operator_approved, automatic_approved, "
                "or trusted_import"
            )
        if context_mode_value == "context_dependent" and not (
            scope_value or entity_values or precondition_values or system_values or version_values
        ):
            raise ValueError(
                "context-dependent memory requires scope, entities, preconditions, systems, or versions"
            )
        completeness = _memory_metadata_completeness(
            context_mode_value,
            scope=scope_value,
            entities=entity_values,
            preconditions=precondition_values,
            source_context=source_context_value,
            applicable_systems=system_values,
            applicable_versions=version_values,
        )
        scope_json = _trace_json(scope_value)
        preconditions_json = _trace_json(precondition_values)
        systems_json = _trace_json(system_values)
        versions_json = _trace_json(version_values)
        assessment = self.assess_storage_candidate(
            content,
            kind=kind,
            context_mode=context_mode_value,
            scope=scope_value,
            entities=entity_values,
            preconditions=precondition_values,
            source_context=source_context_value,
            applicable_systems=system_values,
            applicable_versions=version_values,
            source_type=source_type,
            source_category=source_category,
            extraction_method=extraction_method,
            confidence=confidence,
            importance=importance,
            uniqueness=uniqueness,
            volatility=volatility,
            subject=subject,
            predicate=predicate,
            object_value=object_value,
            valid_from=valid_from,
            valid_to=valid_to,
            automatic=storage_policy == "automatic",
        )
        if assessment["decision"] == "ignored":
            # Record the rejection in its own committed transaction. Doing this
            # inside the creation transaction below would let the ValueError roll
            # back the very decision row we just wrote (the ledger would be empty).
            self.record_ignored_memory_candidate(assessment, session_id=session_id)
            raise ValueError(str(assessment["reason"]))
        with self.transaction() as conn:
            active_recall_set = self._active_recall_set_tx(conn)
            if recall_eligibility is None:
                eligibility_value = (
                    "evidence_only"
                    if str(active_recall_set["kind"]) == "trained"
                    and (record_role == "reference" or source_type == "vault_markdown")
                    else "primary"
                )
            else:
                eligibility_value = normalize_text(recall_eligibility).casefold()
                if eligibility_value not in {"primary", "evidence_only"}:
                    raise ValueError("recall eligibility must be primary or evidence_only")
            existing = conn.execute(
                """SELECT m.id,m.kind,m.entities_json FROM memories m
                   JOIN memory_recall_memberships rm
                     ON rm.memory_id=m.id AND rm.recall_set_id=?
                    AND rm.revoked_at IS NULL AND rm.eligibility=?
                   WHERE m.state IN ('active','cold')
                     AND m.content_hash=? AND m.context_mode=? AND m.scope_json=? AND m.preconditions_json=?
                     AND applicable_systems_json=? AND applicable_versions_json=?
                   ORDER BY m.created_at LIMIT 1""",
                (
                    active_recall_set["recall_set_id"],
                    eligibility_value,
                    digest,
                    context_mode_value,
                    scope_json,
                    preconditions_json,
                    systems_json,
                    versions_json,
                ),
            ).fetchone()
            if existing:
                memory_id = str(existing["id"])
                if preserve_exact_duplicate:
                    return memory_id, False
                assessment = {
                    **assessment,
                    "decision": "updated",
                    "duplicate_memory_id": memory_id,
                    "reason": "exact candidate in the same context updated the existing memory",
                }
                existing_entities = _json_string_list(existing["entities_json"])
                merged_entities = _normalize_context_list([*existing_entities, *entity_values])
                conn.execute(
                    """UPDATE memories
                       SET duplicate_count=duplicate_count+1, updated_at=?,
                           importance=MAX(importance, ?), confidence=MAX(confidence, ?),
                           pinned=MAX(pinned, ?), protected=MAX(protected, ?),
                           trust=MAX(trust, ?), uniqueness=MIN(uniqueness, ?),
                           entities_json=?,source_context=COALESCE(source_context,?),
                           metadata_completeness=MAX(metadata_completeness,?),
                           source_category=CASE
                             WHEN ?='USER_EXPLICIT' THEN 'USER_EXPLICIT'
                             WHEN ?='OPERATOR_APPROVED' AND approval_state<>'operator_approved'
                               THEN 'OPERATOR_APPROVED'
                             WHEN ?='AUTOMATIC_APPROVED' AND approval_state='unreviewed'
                               THEN 'AUTOMATIC_APPROVED'
                             ELSE source_category END,
                           origin_source_category=CASE
                             WHEN approval_state NOT IN
                               ('operator_approved','automatic_approved','trusted_import')
                               THEN ? ELSE origin_source_category END,
                           approval_state=CASE
                             WHEN ?='operator_approved' THEN 'operator_approved'
                             WHEN ?='automatic_approved' AND approval_state='unreviewed'
                               THEN 'automatic_approved'
                             ELSE approval_state END
                       WHERE id=?""",
                    (
                        now,
                        _clamp(importance),
                        _clamp(confidence),
                        int(pinned),
                        int(protected or pinned or kind == "prospective"),
                        _clamp(trust),
                        _clamp(uniqueness),
                        _trace_json(merged_entities),
                        source_context_value,
                        completeness,
                        source_category,
                        source_category,
                        source_category,
                        origin_source_category_value,
                        approval_state_value,
                        approval_state_value,
                        memory_id,
                    ),
                )
                for evidence_id in evidence_ids or ():
                    if (
                        evidence_id != memory_id
                        and conn.execute("SELECT 1 FROM memories WHERE id=?", (evidence_id,)).fetchone()
                    ):
                        conn.execute(
                            "INSERT OR IGNORE INTO memory_dependencies(memory_id,evidence_id,relation,weight,active,created_at) VALUES(?,?,'derived_from',1.0,1,?)",
                            (memory_id, evidence_id, now),
                        )
                self._index_context_terms_tx(
                    conn,
                    memory_id,
                    context_mode=context_mode_value,
                    scope=scope_value,
                    entities=merged_entities,
                    preconditions=precondition_values,
                    applicable_systems=system_values,
                    applicable_versions=version_values,
                )
                if str(existing["kind"]) == "prospective":
                    conn.execute(
                        """INSERT OR IGNORE INTO prospective_items(
                             memory_id,status,due_at,created_at,updated_at
                           ) VALUES(?,'open',?,?,?)""",
                        (memory_id, valid_from or valid_to, now, now),
                    )
                self._record_memory_write_decision_tx(
                    conn,
                    assessment,
                    session_id=session_id,
                    memory_id=memory_id,
                )
                merged_row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
                if merged_row:
                    # A duplicate write refreshes the presentation but must not
                    # let an automatic capture downgrade an existing record.
                    self._classify_and_present_tx(
                        conn,
                        dict(merged_row),
                        has_active_dependencies=self._has_active_dependency_tx(conn, memory_id),
                        explicit_role=record_role,
                    )
                    refreshed = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
                    if refreshed:
                        self._assign_memory_neighborhoods_tx(conn, dict(refreshed))
                return memory_id, False

            memory_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO memories(
                    id, kind, content, content_hash, source_type, source_category,
                    origin_source_category,approval_state,source_ref, session_id,
                    context_mode,scope_json,entities_json,preconditions_json,source_context,
                    applicable_systems_json,applicable_versions_json,metadata_completeness,
                    created_at, updated_at, observed_at, valid_from, valid_to, subject, predicate, object_value,
                    extraction_method, confidence, currentness_confidence, importance, uniqueness,
                    volatility, trust, state, pinned, protected, supersedes_id, quarantine_reason
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    memory_id,
                    kind,
                    content,
                    digest,
                    source_type,
                    source_category,
                    origin_source_category_value,
                    approval_state_value,
                    source_ref,
                    session_id,
                    context_mode_value,
                    scope_json,
                    _trace_json(entity_values),
                    preconditions_json,
                    source_context_value,
                    systems_json,
                    versions_json,
                    completeness,
                    now,
                    now,
                    observed_at or now,
                    valid_from,
                    valid_to,
                    subject,
                    predicate,
                    object_value,
                    extraction_method,
                    _clamp(confidence),
                    _clamp(currentness_confidence),
                    _clamp(importance),
                    _clamp(uniqueness),
                    _clamp(volatility),
                    _clamp(trust),
                    state,
                    int(pinned),
                    int(protected or pinned or kind == "prospective"),
                    supersedes_id,
                    quarantine_reason,
                ),
            )
            conn.execute(
                """INSERT INTO memory_versions(
                    memory_id, content, confidence, state, valid_from, valid_to,
                    system_from, reason, source_ref
                ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (memory_id, content, _clamp(confidence), state, valid_from, valid_to, now, "created", source_ref),
            )
            conn.execute("INSERT INTO memory_fts(memory_id, content) VALUES(?,?)", (memory_id, content))
            self._index_features_tx(conn, memory_id, content)
            self._index_context_terms_tx(
                conn,
                memory_id,
                context_mode=context_mode_value,
                scope=scope_value,
                entities=entity_values,
                preconditions=precondition_values,
                applicable_systems=system_values,
                applicable_versions=version_values,
            )
            for evidence_id in evidence_ids or ():
                if (
                    evidence_id != memory_id
                    and conn.execute("SELECT 1 FROM memories WHERE id=?", (evidence_id,)).fetchone()
                ):
                    conn.execute(
                        "INSERT OR IGNORE INTO memory_dependencies(memory_id,evidence_id,relation,weight,active,created_at) VALUES(?,?,'derived_from',1.0,1,?)",
                        (memory_id, evidence_id, now),
                    )
            if supersedes_id and conn.execute("SELECT 1 FROM memories WHERE id=?", (supersedes_id,)).fetchone():
                conn.execute(
                    "INSERT OR IGNORE INTO edges(src_id,dst_id,relation,weight,evidence_count,created_at,last_reinforced_at) VALUES(?,?,'supersedes',1.0,1,?,?)",
                    (memory_id, supersedes_id, now, now),
                )
                self._record_edge_evidence_tx(
                    conn,
                    memory_id,
                    supersedes_id,
                    "supersedes",
                    evidence_type="version_lineage",
                    evidence_key=f"{memory_id}:{supersedes_id}",
                    summary="The write explicitly identified this memory as the newer replacement for the connected memory.",
                    source_ref=source_ref,
                    metadata={"newer_memory_id": memory_id, "older_memory_id": supersedes_id},
                    created_at=now,
                )
            self._link_structured_contradictions(
                conn, memory_id, subject, predicate, object_value, valid_from, valid_to, now
            )
            if kind == "prospective":
                conn.execute(
                    """INSERT INTO prospective_items(memory_id,status,due_at,created_at,updated_at)
                       VALUES(?,'open',?,?,?)""",
                    (memory_id, valid_from or valid_to, now, now),
                )
            self._record_memory_write_decision_tx(
                conn,
                {**assessment, "decision": "created"},
                session_id=session_id,
                memory_id=memory_id,
            )
            created_row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            self._classify_and_present_tx(
                conn,
                dict(created_row),
                has_active_dependencies=self._has_active_dependency_tx(conn, memory_id),
                explicit_role=record_role,
                storage_policy=storage_policy,
            )
            refreshed = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            if refreshed:
                self._refresh_dynamic_neighborhoods_tx(conn)
            conn.execute(
                """UPDATE memory_recall_memberships
                   SET eligibility=?,origin=?,reason=?
                   WHERE recall_set_id=? AND memory_id=?""",
                (
                    eligibility_value,
                    "operator_approval" if storage_policy == "operator_approved" else "trusted_write",
                    (
                        "Approved for explicit lookup evidence only."
                        if eligibility_value == "evidence_only"
                        else "Approved for ordinary recall in the active set."
                    ),
                    active_recall_set["recall_set_id"],
                    memory_id,
                ),
            )
            return memory_id, True

    def correct_memory(
        self,
        memory_id: str,
        new_content: str,
        *,
        reason: str = "corrected",
        confidence: float | None = None,
        source_ref: str | None = None,
    ) -> bool:
        new_content = normalize_text(new_content)
        if not new_content:
            raise ValueError("corrected content cannot be empty")
        now = utc_now()
        from .research import record_reconsolidation_correction_tx

        with self.transaction() as conn:
            current = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            if not current:
                return False
            prior_version = conn.execute(
                """SELECT version_id FROM memory_versions
                   WHERE memory_id=? AND system_to IS NULL ORDER BY version_id DESC LIMIT 1""",
                (memory_id,),
            ).fetchone()
            next_confidence = _clamp(confidence if confidence is not None else max(0.55, current["confidence"]))
            conn.execute(
                "UPDATE memory_versions SET system_to=? WHERE memory_id=? AND system_to IS NULL",
                (now, memory_id),
            )
            inserted_version = conn.execute(
                """INSERT INTO memory_versions(
                    memory_id, content, confidence, state, valid_from, valid_to,
                    system_from, reason, source_ref
                ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    memory_id,
                    new_content,
                    next_confidence,
                    "active",
                    current["valid_from"],
                    current["valid_to"],
                    now,
                    reason,
                    source_ref,
                ),
            )
            conn.execute(
                """UPDATE memories SET content=?, content_hash=?, confidence=?, state='active',
                   updated_at=?, correction_count=correction_count+1,
                   quarantine_reason=NULL, protected=1 WHERE id=?""",
                (new_content, content_hash(new_content), next_confidence, now, memory_id),
            )
            conn.execute("DELETE FROM memory_fts WHERE memory_id=?", (memory_id,))
            conn.execute("INSERT INTO memory_fts(memory_id, content) VALUES(?,?)", (memory_id, new_content))
            self._index_features_tx(conn, memory_id, new_content)
            conn.execute(
                "INSERT INTO access_log(memory_id,event,query,created_at) VALUES(?,?,?,?)",
                (memory_id, "corrected", reason, now),
            )
            record_reconsolidation_correction_tx(
                conn,
                memory_id=memory_id,
                prior_version_id=int(prior_version["version_id"]) if prior_version else None,
                new_version_id=int(inserted_version.lastrowid) if inserted_version.lastrowid else None,
                trigger_access_at=current["last_used_at"] or current["last_retrieved_at"],
                corrected_at=now,
                source_ref=source_ref,
            )
            self._mark_dependents_dirty_tx(conn, memory_id, f"evidence corrected: {reason}")
            corrected_row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            if corrected_row:
                self._classify_and_present_tx(
                    conn,
                    dict(corrected_row),
                    has_active_dependencies=self._has_active_dependency_tx(conn, memory_id),
                )
            return True

    def set_state(
        self,
        memory_id: str,
        state: str,
        *,
        reason: str = "manual",
        retention_score: float | None = None,
    ) -> bool:
        if state not in {"active", "cold", "archived", "quarantine", "tombstoned"}:
            raise ValueError(f"invalid state: {state}")
        now = utc_now()
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            if not row:
                return False
            if row["state"] == state:
                return True
            conn.execute(
                "UPDATE memory_versions SET system_to=? WHERE memory_id=? AND system_to IS NULL", (now, memory_id)
            )
            conn.execute(
                """INSERT INTO memory_versions(memory_id,content,confidence,state,valid_from,valid_to,
                   system_from,reason,source_ref) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    memory_id,
                    row["content"],
                    row["confidence"],
                    state,
                    row["valid_from"],
                    row["valid_to"],
                    now,
                    reason,
                    row["source_ref"],
                ),
            )
            conn.execute("UPDATE memories SET state=?, updated_at=? WHERE id=?", (state, now, memory_id))
            conn.execute(
                """INSERT INTO lifecycle_events(
                   memory_id,from_state,to_state,reason,retention_score,created_at
                   ) VALUES(?,?,?,?,?,?)""",
                (memory_id, row["state"], state, reason, retention_score, now),
            )
            if state in {"archived", "quarantine", "tombstoned"}:
                self._mark_dependents_dirty_tx(conn, memory_id, f"evidence state changed to {state}")
            self._refresh_dynamic_neighborhoods_tx(conn)
            return True

    def set_pinned(self, memory_id: str, pinned: bool) -> bool:
        with self.transaction() as conn:
            result = conn.execute(
                "UPDATE memories SET pinned=?, updated_at=? WHERE id=?",
                (int(pinned), utc_now(), memory_id),
            )
            return result.rowcount > 0

    def set_uniqueness(self, memory_id: str, uniqueness: float) -> bool:
        with self.transaction() as conn:
            result = conn.execute("UPDATE memories SET uniqueness=? WHERE id=?", (_clamp(uniqueness), memory_id))
            return result.rowcount > 0

    def find_by_content(self, content: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE content_hash=? ORDER BY created_at LIMIT 1",
                (content_hash(content),),
            ).fetchone()
        return _decode_memory_metadata(row) if row else None

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        return _decode_memory_metadata(row) if row else None

    def resolve_id(self, memory_id_or_prefix: str) -> str | None:
        """Resolve a full UUID or an unambiguous displayed prefix."""
        value = (memory_id_or_prefix or "").strip()
        if not value:
            return None
        with self._lock:
            exact = self._conn.execute("SELECT id FROM memories WHERE id=?", (value,)).fetchone()
            if exact:
                return str(exact["id"])
            rows = self._conn.execute(
                "SELECT id FROM memories WHERE id LIKE ? ORDER BY created_at LIMIT 2", (value + "%",)
            ).fetchall()
        return str(rows[0]["id"]) if len(rows) == 1 else None

    def get_memories(self, memory_ids: Sequence[str]) -> list[dict[str, Any]]:
        if not memory_ids:
            return []
        placeholders = ",".join("?" for _ in memory_ids)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders})", tuple(memory_ids)
            ).fetchall()
        by_id = {row["id"]: _decode_memory_metadata(row) for row in rows}
        return [by_id[mid] for mid in memory_ids if mid in by_id]

    def retrieval_revision(self) -> tuple[int, int]:
        """Return the durable revision used to invalidate retrieval caches.

        The revision advances for material memory, association, learned-tool,
        and active operator-policy changes. Pure retrieval/injection counters intentionally do not advance
        it, so a short-lived cache can still be useful while every injection is
        recorded independently.
        """

        with self._lock:
            row = self._conn.execute("PRAGMA data_version").fetchone()
            data_version = int(row[0]) if row else 0
            return self._local_retrieval_revision, data_version

    def recommend_token_budget(
        self,
        task_type: str,
        mode: str,
        base_budget: int,
        *,
        min_budget: int = 160,
        max_budget: int = 700,
        min_samples: int = 8,
    ) -> dict[str, Any]:
        """Return a bounded token-budget recommendation from resolved outcomes.

        Learning is intentionally conservative. Exact task-type/mode evidence
        is preferred; a mode-wide fallback requires twice as many observations.
        No change is recommended until the evidence clears those gates.
        """

        maximum = max(1, int(max_budget))
        minimum = max(1, min(int(min_budget), maximum))
        baseline = max(minimum, min(maximum, int(base_budget)))
        required = max(2, int(min_samples))
        with self._lock:
            exact = self._conn.execute(
                """SELECT requested_budget,estimated_tokens,selected_count,used_count,outcome
                   FROM recall_budget_observations
                   WHERE task_type=? AND mode=? AND outcome<>'pending'
                   ORDER BY created_at DESC LIMIT 80""",
                (task_type, mode),
            ).fetchall()
            rows = exact
            scope = "task_mode"
            if len(rows) < required:
                rows = self._conn.execute(
                    """SELECT requested_budget,estimated_tokens,selected_count,used_count,outcome
                       FROM recall_budget_observations
                       WHERE mode=? AND outcome<>'pending'
                       ORDER BY created_at DESC LIMIT 120""",
                    (mode,),
                ).fetchall()
                scope = "mode"
                required *= 2

        sample_count = len(rows)
        used_count = sum(1 for row in rows if int(row["used_count"]) > 0)
        positive_count = sum(1 for row in rows if row["outcome"] in {"helpful", "validated"})
        negative_count = sum(1 for row in rows if row["outcome"] in {"harmful", "corrected"})
        ignored_count = sum(1 for row in rows if row["outcome"] == "ignored")
        fill_values = [
            min(1.0, int(row["estimated_tokens"]) / max(1, int(row["requested_budget"])))
            for row in rows
        ]
        used_ratio = used_count / sample_count if sample_count else 0.0
        ignored_ratio = ignored_count / sample_count if sample_count else 0.0
        average_fill = sum(fill_values) / sample_count if sample_count else 0.0

        budget = baseline
        adjustment = "hold"
        reason = f"learning gate needs {required} resolved outcomes"
        if sample_count >= required:
            reason = "resolved outcomes support the baseline budget"
            if negative_count / sample_count >= 0.20:
                budget = max(minimum, round(baseline * 0.85))
                adjustment = "shrink"
                reason = "harmful or corrected outcomes exceeded the conservative limit"
            elif ignored_ratio >= 0.75 and positive_count == 0:
                budget = max(minimum, round(baseline * 0.85))
                adjustment = "shrink"
                reason = "most retrieved context was ignored"
            elif ignored_ratio >= 0.65 and positive_count == 0:
                budget = max(minimum, round(baseline * 0.90))
                adjustment = "shrink"
                reason = "retrieved context was usually ignored"
            elif (
                positive_count >= 3
                and negative_count == 0
                and used_ratio >= 0.75
                and average_fill >= 0.80
            ):
                budget = min(maximum, round(baseline * 1.10))
                adjustment = "expand"
                reason = "helpful outcomes repeatedly saturated the available budget"

        return {
            "budget": budget,
            "base_budget": baseline,
            "adjustment": adjustment,
            "reason": reason,
            "scope": scope,
            "task_type": task_type,
            "mode": mode,
            "sample_count": sample_count,
            "required_samples": required,
            "used_ratio": round(used_ratio, 4),
            "ignored_ratio": round(ignored_ratio, 4),
            "average_fill": round(average_fill, 4),
            "positive_count": positive_count,
            "negative_count": negative_count,
        }

    def attention_recommendation(
        self,
        task_type: str,
        topics: Sequence[str],
        live_mode: str,
        base_budget: int,
        *,
        max_budget: int = 700,
        min_samples: int = 4,
    ) -> dict[str, Any]:
        """Return a decayed, shadow-only topic-salience recommendation."""

        from .cognition import ATTENTION_POLICY_VERSION

        task = normalize_text(task_type)[:80] or "general"
        topic_keys = list(
            dict.fromkeys(
                normalize_text(str(topic)).casefold()[:40]
                for topic in topics
                if normalize_text(str(topic))
            )
        )[:12]
        mode = normalize_text(live_mode).casefold()[:40] or "lean"
        maximum = max(1, int(max_budget))
        baseline = max(0, min(maximum, int(base_budget)))
        required = max(2, int(min_samples))
        rows: list[sqlite3.Row] = []
        if topic_keys:
            placeholders = ",".join("?" for _ in topic_keys)
            with self._lock:
                rows = self._conn.execute(
                    f"""SELECT * FROM attention_weights
                        WHERE task_type=? AND topic_key IN ({placeholders})
                        ORDER BY topic_key,mode""",
                    (task, *topic_keys),
                ).fetchall()

        now = datetime.now(timezone.utc)
        evidence: list[dict[str, Any]] = []
        for row in rows:
            last_seen = parse_iso8601(str(row["last_observed_at"] or ""))
            age_days = (
                max(0.0, (now - last_seen).total_seconds() / 86400.0)
                if last_seen is not None
                else 0.0
            )
            decay = 0.5 ** (age_days / _attentional_decay_days())
            samples = int(row["used_count"]) + int(row["ignored_count"])
            evidence.append(
                {
                    "topic_key": str(row["topic_key"]),
                    "mode": str(row["mode"]),
                    "used_count": int(row["used_count"]),
                    "helpful_count": int(row["helpful_count"]),
                    "ignored_count": int(row["ignored_count"]),
                    "harmful_count": int(row["harmful_count"]),
                    "sample_count": samples,
                    "weight_delta": round(float(row["weight_delta"]), 6),
                    "effective_delta": round(float(row["weight_delta"]) * decay, 6),
                    "decay_factor": round(decay, 6),
                    "eligible": samples >= required,
                }
            )

        eligible = [item for item in evidence if item["eligible"]]
        topic_deltas: list[float] = []
        for topic in topic_keys:
            topic_rows = [item for item in eligible if item["topic_key"] == topic]
            if topic_rows:
                topic_deltas.append(
                    max(topic_rows, key=lambda item: abs(float(item["effective_delta"])))[
                        "effective_delta"
                    ]
                )
        aggregate_delta = (
            max(-0.10, min(0.10, sum(topic_deltas) / len(topic_deltas)))
            if topic_deltas
            else 0.0
        )

        mode_rank = {"none": 0, "lean": 1, "focused": 2, "procedural": 3, "deep": 4}
        shadow_mode = mode
        mode_reason = "static attention gate remains authoritative"
        if mode != "none" and eligible:
            current_rows = [item for item in eligible if item["mode"] == mode]
            current_delta = (
                min((float(item["effective_delta"]) for item in current_rows), default=0.0)
            )
            if current_delta <= -0.05:
                shadow_mode = {
                    "deep": "focused",
                    "procedural": "lean",
                    "focused": "lean",
                }.get(mode, "lean")
                mode_reason = "repeated ignored outcomes support a one-level shadow downgrade"
            else:
                positive = [
                    item
                    for item in eligible
                    if float(item["effective_delta"]) >= 0.05
                    and mode_rank.get(str(item["mode"]), 0) > mode_rank.get(mode, 1)
                ]
                if positive:
                    best = max(
                        positive,
                        key=lambda item: (
                            float(item["effective_delta"]),
                            int(item["sample_count"]),
                            mode_rank.get(str(item["mode"]), 0),
                        ),
                    )
                    shadow_mode = str(best["mode"])
                    mode_reason = (
                        f"resolved outcomes for {best['topic_key']} support the previously useful "
                        f"{shadow_mode} mode"
                    )

        shadow_budget = baseline
        if mode != "none" and baseline > 0 and topic_deltas:
            shadow_budget = max(
                min(300, maximum),
                min(maximum, round(baseline * (1.0 + aggregate_delta))),
            )
        return {
            "policy_version": ATTENTION_POLICY_VERSION,
            "mode": "shadow",
            "task_type": task,
            "topics": topic_keys,
            "live_mode": mode,
            "shadow_mode": shadow_mode,
            "live_budget": baseline,
            "shadow_budget": shadow_budget,
            "budget_delta": round(aggregate_delta, 6),
            "sample_count": sum(int(item["sample_count"]) for item in eligible),
            "required_samples_per_topic_mode": required,
            "evidence": evidence,
            "reason": mode_reason,
            "applied": False,
        }

    def record_attention_observation(
        self,
        task_id: str,
        *,
        task_type: str,
        topics: Sequence[str],
        live_mode: str,
        live_budget: int,
        max_budget: int,
        selected_count: int,
        control: bool = False,
    ) -> dict[str, Any]:
        """Persist the policy decision before its attribution outcome exists."""

        recommendation = self.attention_recommendation(
            task_type,
            topics,
            live_mode,
            live_budget,
            max_budget=max_budget,
        )
        now = utc_now()
        initial = "control" if control else ("no_context" if selected_count <= 0 else "pending")
        with self.transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO attention_observations(
                     task_id,task_type,topics_json,live_mode,live_budget,shadow_mode,
                     shadow_budget,policy_version,selected_count,usage_outcome,created_at,resolved_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    normalize_text(task_id)[:100],
                    recommendation["task_type"],
                    _trace_json(recommendation["topics"]),
                    recommendation["live_mode"],
                    recommendation["live_budget"],
                    recommendation["shadow_mode"],
                    recommendation["shadow_budget"],
                    recommendation["policy_version"],
                    max(0, int(selected_count)),
                    initial,
                    now,
                    now if initial != "pending" else None,
                ),
            )
        return recommendation

    @staticmethod
    def _attention_weight_delta(
        used_count: int,
        ignored_count: int,
        helpful_count: int,
        harmful_count: int,
    ) -> float:
        samples = used_count + ignored_count
        if samples < 4:
            return 0.0
        used_rate = used_count / samples
        delta = 0.0
        if used_rate >= 0.80:
            delta = 0.10
        elif used_rate >= 0.65:
            delta = 0.05
        elif used_rate <= 0.20:
            delta = -0.10
        elif used_rate <= 0.35:
            delta = -0.05
        if helpful_count >= 2 and harmful_count == 0:
            delta = max(delta, 0.10)
        if harmful_count > 0 and harmful_count >= helpful_count:
            delta = min(delta, -0.10)
        return round(max(-0.10, min(0.10, delta)), 6)

    def _refresh_attention_weights_tx(
        self,
        conn: sqlite3.Connection,
        *,
        task_type: str,
        topics: Sequence[str],
        now: str,
    ) -> None:
        topic_keys = set(topics)
        if not topic_keys:
            return
        rows = conn.execute(
            """SELECT topics_json,live_mode,usage_outcome,final_outcome,
                      COALESCE(resolved_at,created_at) observed_at
               FROM attention_observations
               WHERE task_type=? AND usage_outcome IN ('used','ignored')
               ORDER BY created_at""",
            (task_type,),
        ).fetchall()
        aggregates: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            try:
                row_topics = json.loads(str(row["topics_json"] or "[]"))
            except json.JSONDecodeError:
                row_topics = []
            if not isinstance(row_topics, list):
                continue
            for topic in topic_keys.intersection(str(item) for item in row_topics):
                key = (topic, str(row["live_mode"]))
                values = aggregates.setdefault(
                    key,
                    {
                        "used": 0,
                        "ignored": 0,
                        "helpful": 0,
                        "harmful": 0,
                        "last": str(row["observed_at"]),
                    },
                )
                used = str(row["usage_outcome"]) == "used"
                values["used" if used else "ignored"] += 1
                final = str(row["final_outcome"])
                values["helpful"] += int(used and final in {"helpful", "validated"})
                values["harmful"] += int(used and final in {"harmful", "corrected"})
                values["last"] = max(values["last"], str(row["observed_at"]))

        for topic in topic_keys:
            conn.execute(
                "DELETE FROM attention_weights WHERE task_type=? AND topic_key=?",
                (task_type, topic),
            )
        for (topic, mode), values in aggregates.items():
            delta = self._attention_weight_delta(
                int(values["used"]),
                int(values["ignored"]),
                int(values["helpful"]),
                int(values["harmful"]),
            )
            conn.execute(
                """INSERT INTO attention_weights(
                     task_type,topic_key,mode,used_count,helpful_count,ignored_count,
                     harmful_count,weight_delta,last_observed_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    task_type,
                    topic,
                    mode,
                    values["used"],
                    values["helpful"],
                    values["ignored"],
                    values["harmful"],
                    delta,
                    values["last"],
                    now,
                ),
            )

    def _set_attention_final_outcome_tx(
        self,
        conn: sqlite3.Connection,
        *,
        task_id: str,
        outcome: str,
        now: str,
    ) -> None:
        attention = conn.execute(
            """SELECT task_type,topics_json FROM attention_observations
               WHERE task_id=? AND usage_outcome='used'""",
            (task_id,),
        ).fetchone()
        if not attention:
            return
        final = (
            outcome
            if outcome in {"helpful", "harmful", "validated", "corrected"}
            else "pending"
        )
        conn.execute(
            """UPDATE attention_observations SET final_outcome=?,resolved_at=?
               WHERE task_id=? AND usage_outcome='used'""",
            (final, now, task_id),
        )
        try:
            topics = json.loads(str(attention["topics_json"] or "[]"))
        except json.JSONDecodeError:
            topics = []
        self._refresh_attention_weights_tx(
            conn,
            task_type=str(attention["task_type"]),
            topics=topics if isinstance(topics, list) else [],
            now=now,
        )

    def attention_learning_summary(self, *, limit: int = 100) -> dict[str, Any]:
        """Expose bounded evidence without claiming the shadow policy improved recall."""

        from .cognition import ATTENTION_POLICY_VERSION

        bounded = max(1, min(500, int(limit)))
        with self._lock:
            weights = [
                dict(row)
                for row in self._conn.execute(
                    """SELECT * FROM attention_weights
                       ORDER BY ABS(weight_delta) DESC,
                                used_count+ignored_count DESC,last_observed_at DESC LIMIT ?""",
                    (bounded,),
                ).fetchall()
            ]
            summary = dict(
                self._conn.execute(
                    """SELECT COUNT(*) observation_count,
                              SUM(CASE WHEN usage_outcome IN ('used','ignored') THEN 1 ELSE 0 END)
                                resolved_count,
                              SUM(CASE WHEN usage_outcome='used' THEN 1 ELSE 0 END) used_count,
                              SUM(CASE WHEN usage_outcome='ignored' THEN 1 ELSE 0 END) ignored_count,
                              SUM(CASE WHEN usage_outcome='no_context' THEN 1 ELSE 0 END)
                                no_context_count,
                              SUM(CASE WHEN usage_outcome='control' THEN 1 ELSE 0 END) control_count,
                              COUNT(DISTINCT CASE
                                WHEN usage_outcome IN ('used','ignored')
                                THEN substr(COALESCE(resolved_at,created_at),1,10) END) observed_days
                       FROM attention_observations"""
                ).fetchone()
            )
            recent = [
                dict(row)
                for row in self._conn.execute(
                    """SELECT task_id,task_type,topics_json,live_mode,live_budget,shadow_mode,
                              shadow_budget,policy_version,selected_count,usage_outcome,
                              final_outcome,created_at,resolved_at
                       FROM attention_observations ORDER BY created_at DESC LIMIT 100"""
                ).fetchall()
            ]
        now = datetime.now(timezone.utc)
        for row in weights:
            last_seen = parse_iso8601(str(row.get("last_observed_at") or ""))
            age_days = (
                max(0.0, (now - last_seen).total_seconds() / 86400.0)
                if last_seen is not None
                else 0.0
            )
            row["sample_count"] = int(row["used_count"]) + int(row["ignored_count"])
            row["effective_delta"] = round(
                float(row["weight_delta"]) * (0.5 ** (age_days / _attentional_decay_days())),
                6,
            )
        for row in recent:
            raw_topics = row.pop("topics_json", "[]")
            try:
                topics = json.loads(str(raw_topics or "[]"))
            except json.JSONDecodeError:
                topics = []
            row["topics"] = topics if isinstance(topics, list) else []
        resolved = int(summary.get("resolved_count") or 0)
        used = int(summary.get("used_count") or 0)
        observed_days = int(summary.get("observed_days") or 0)
        changed = sum(
            int(row["live_mode"] != row["shadow_mode"] or row["live_budget"] != row["shadow_budget"])
            for row in recent
        )
        return {
            "policy_version": ATTENTION_POLICY_VERSION,
            "mode": "shadow",
            "summary": {
                **{key: int(value or 0) for key, value in summary.items()},
                "attributed_use_rate": round(used / resolved, 6) if resolved else None,
                "recent_shadow_changes": changed,
            },
            "weights": weights,
            "recent_observations": recent,
            "promotion_gate": {
                "ready": observed_days >= 7 and resolved >= 50,
                "checks": [
                    {
                        "key": "days",
                        "label": "Resolved observation days",
                        "value": observed_days,
                        "target": 7,
                        "passed": observed_days >= 7,
                    },
                    {
                        "key": "outcomes",
                        "label": "Resolved retrieval outcomes",
                        "value": resolved,
                        "target": 50,
                        "passed": resolved >= 50,
                    },
                ],
            },
            "definitions": {
                "weight_delta": "A bounded topic and task-type recommendation from resolved attributed use versus ignored context.",
                "effective_delta": "The weight delta after a 30-day half-life; stale topic salience fades automatically.",
            },
            "claim_boundary": (
                "Topic cues are observational and may be correlated. Shadow recommendations never "
                "change live recall, and precision improvement requires a later controlled exposure."
            ),
        }

    def calibrate_metacognitive_probability(
        self,
        raw_probability: float,
        *,
        task_type: str,
        source_category: str,
        min_samples: int = 8,
    ) -> dict[str, Any]:
        """Conservatively adjust a reliability estimate from comparable outcomes.

        Calibration stays dormant until a probability band has enough explicit
        helpful/validated or harmful/corrected outcomes. Exact task and source
        evidence is preferred, with progressively stricter gates for fallbacks.
        """

        raw = _clamp(raw_probability)
        bucket = min(4, int(raw * 5.0))
        task = normalize_text(task_type)[:80] or "general"
        source = normalize_text(source_category)[:80] or "AGENT_INFERENCE"
        required = max(4, int(min_samples))
        scopes = (
            ("task_source", "AND task_type=? AND source_category=?", (task, source), required),
            ("task", "AND task_type=?", (task,), required * 2),
            ("global", "", (), required * 3),
        )
        strongest_sample = 0
        with self._lock:
            for scope, clause, params, gate in scopes:
                row = self._conn.execute(
                    f"""SELECT COUNT(*) sample_count,
                               SUM(CASE WHEN outcome IN ('helpful','validated') THEN 1 ELSE 0 END) positive_count
                        FROM metacognitive_predictions
                        WHERE outcome IN ('helpful','validated','harmful','corrected')
                          AND CASE WHEN raw_probability>=1.0 THEN 4
                                   ELSE CAST(raw_probability*5 AS INTEGER) END=?
                          {clause}""",
                    (bucket, *params),
                ).fetchone()
                sample_count = int(row["sample_count"] or 0)
                strongest_sample = max(strongest_sample, sample_count)
                if sample_count < gate:
                    continue
                positive_count = int(row["positive_count"] or 0)
                observed_rate = (positive_count + 2.0) / (sample_count + 4.0)
                weight = min(0.45, sample_count / (sample_count + 24.0))
                probability = (1.0 - weight) * raw + weight * observed_rate
                return {
                    "probability": round(_clamp(probability), 6),
                    "raw_probability": round(raw, 6),
                    "scope": scope,
                    "sample_count": sample_count,
                    "positive_count": positive_count,
                    "observed_rate": round(observed_rate, 6),
                    "weight": round(weight, 6),
                }
        return {
            "probability": round(raw, 6),
            "raw_probability": round(raw, 6),
            "scope": "prior",
            "sample_count": strongest_sample,
            "positive_count": 0,
            "observed_rate": None,
            "weight": 0.0,
        }

    def metacognition_enforcement_gate(self) -> dict[str, Any]:
        """Return the evidence gate that must pass before abstentions can be enforced."""

        with self._lock:
            summary = self._conn.execute(
                """SELECT
                     SUM(CASE WHEN outcome IN ('helpful','validated','harmful','corrected') THEN 1 ELSE 0 END) labels,
                     SUM(CASE WHEN decision='use' AND outcome IN ('helpful','validated','harmful','corrected') THEN 1 ELSE 0 END) use_labels,
                     SUM(CASE WHEN decision='use' AND outcome IN ('harmful','corrected') THEN 1 ELSE 0 END) harmful_use,
                     AVG(CASE
                       WHEN outcome IN ('helpful','validated')
                         THEN (1.0-calibrated_probability)*(1.0-calibrated_probability)
                       WHEN outcome IN ('harmful','corrected')
                         THEN calibrated_probability*calibrated_probability
                     END) brier_score
                   FROM metacognitive_predictions"""
            ).fetchone()
            bins = self._conn.execute(
                """SELECT CASE WHEN calibrated_probability>=1.0 THEN 4
                                 ELSE CAST(calibrated_probability*5 AS INTEGER) END bucket,
                          COUNT(*) sample_count,AVG(calibrated_probability) avg_probability,
                          SUM(CASE WHEN outcome IN ('helpful','validated') THEN 1 ELSE 0 END) positive_count
                   FROM metacognitive_predictions
                   WHERE outcome IN ('helpful','validated','harmful','corrected')
                   GROUP BY bucket"""
            ).fetchall()
        labels = int(summary["labels"] or 0)
        use_labels = int(summary["use_labels"] or 0)
        harmful_use = int(summary["harmful_use"] or 0)
        brier = float(summary["brier_score"]) if summary["brier_score"] is not None else None
        ece = None
        if labels:
            ece = sum(
                int(row["sample_count"])
                * abs(
                    float(row["avg_probability"] or 0.0)
                    - int(row["positive_count"] or 0) / max(1, int(row["sample_count"]))
                )
                for row in bins
            ) / labels
        harmful_use_rate = harmful_use / use_labels if use_labels else None
        checks = [
            {
                "key": "labels",
                "label": "Representative labeled outcomes",
                "value": labels,
                "target": 50,
                "passed": labels >= 50,
            },
            {
                "key": "brier",
                "label": "Brier score",
                "value": round(brier, 6) if brier is not None else None,
                "target": 0.20,
                "direction": "at_most",
                "passed": brier is not None and brier <= 0.20,
            },
            {
                "key": "ece",
                "label": "Expected calibration error",
                "value": round(ece, 6) if ece is not None else None,
                "target": 0.15,
                "direction": "at_most",
                "passed": ece is not None and ece <= 0.15,
            },
            {
                "key": "selective_risk",
                "label": "Harmful rate among use decisions",
                "value": round(harmful_use_rate, 6) if harmful_use_rate is not None else None,
                "target": 0.10,
                "direction": "at_most",
                "passed": use_labels >= 20 and harmful_use_rate is not None and harmful_use_rate <= 0.10,
            },
        ]
        return {
            "ready": all(bool(item["passed"]) for item in checks),
            "requested_mode": "enforce",
            "effective_mode_until_ready": "shadow",
            "checks": checks,
            "labels": labels,
            "use_labels": use_labels,
            "harmful_use": harmful_use,
            "brier_score": round(brier, 6) if brier is not None else None,
            "expected_calibration_error": round(ece, 6) if ece is not None else None,
            "harmful_use_rate": round(harmful_use_rate, 6) if harmful_use_rate is not None else None,
            "claim_boundary": "Passing this gate permits a controlled enforcement trial; it does not prove introspection or consciousness.",
        }

    def claim_memories(
        self,
        subject: str,
        predicate: str,
        *,
        object_value: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM memories WHERE subject=? AND predicate=?"
        params: list[Any] = [subject, predicate]
        if object_value is not None:
            sql += " AND object_value=?"
            params.append(object_value)
        sql += " ORDER BY observed_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [_decode_memory_metadata(row) for row in rows]

    @staticmethod
    def _active_recall_set_tx(conn: sqlite3.Connection) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM memory_recall_sets WHERE status='active' LIMIT 1"
        ).fetchone()
        if not row:  # pragma: no cover - schema backfill guarantees one
            raise RuntimeError("Cortex has no active recall set")
        return row

    def active_recall_set_id(self) -> str:
        with self._lock:
            return str(self._active_recall_set_tx(self._conn)["recall_set_id"])

    def is_memory_recall_eligible(
        self,
        memory_id: str,
        *,
        evidence_lookup: bool = False,
        include_archived: bool = False,
    ) -> bool:
        states = ("active", "cold", "archived") if include_archived else ("active", "cold")
        state_placeholders = ",".join("?" for _ in states)
        eligibilities = ("primary", "evidence_only") if evidence_lookup else ("primary",)
        eligibility_placeholders = ",".join("?" for _ in eligibilities)
        with self._lock:
            row = self._conn.execute(
                f"""SELECT 1 FROM memories m
                    JOIN memory_recall_memberships rm ON rm.memory_id=m.id AND rm.revoked_at IS NULL
                    JOIN memory_recall_sets rs ON rs.recall_set_id=rm.recall_set_id AND rs.status='active'
                    WHERE m.id=? AND m.state IN ({state_placeholders})
                      AND rm.eligibility IN ({eligibility_placeholders}) LIMIT 1""",
                (memory_id, *states, *eligibilities),
            ).fetchone()
        return bool(row)

    def recall_set_snapshot(self) -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT rs.*,
                          SUM(CASE WHEN rm.revoked_at IS NULL THEN 1 ELSE 0 END) member_count,
                          SUM(CASE WHEN rm.revoked_at IS NULL AND rm.eligibility='primary'
                                    AND m.state IN ('active','cold') THEN 1 ELSE 0 END) primary_count,
                          SUM(CASE WHEN rm.revoked_at IS NULL AND rm.eligibility='evidence_only'
                                    AND m.state IN ('active','cold') THEN 1 ELSE 0 END) evidence_only_count
                   FROM memory_recall_sets rs
                   LEFT JOIN memory_recall_memberships rm ON rm.recall_set_id=rs.recall_set_id
                   LEFT JOIN memories m ON m.id=rm.memory_id
                   GROUP BY rs.recall_set_id ORDER BY rs.created_at"""
            ).fetchall()
            pending = self._conn.execute(
                """SELECT COUNT(*) count FROM memory_creation_proposals
                   WHERE status IN ('pending','needs_context')"""
            ).fetchone()
        decoded: list[dict[str, Any]] = []
        for raw in rows:
            item = dict(raw)
            item["member_count"] = int(item.get("member_count") or 0)
            item["primary_count"] = int(item.get("primary_count") or 0)
            item["evidence_only_count"] = int(item.get("evidence_only_count") or 0)
            item["eligible_count"] = item["primary_count"]
            decoded.append(item)
        active = next(item for item in decoded if item["status"] == "active")
        legacy = next((item for item in reversed(decoded) if item["kind"] == "legacy"), {})
        trained = next((item for item in reversed(decoded) if item["kind"] == "trained"), {})
        return {
            "active_set": str(active["kind"]),
            "active": active,
            "sets": {"legacy": legacy, "trained": trained},
            "recall_sets": decoded,
            "eligible_count": int(active["primary_count"]),
            "evidence_only_count": int(active["evidence_only_count"]),
            "pending_creation_count": int(pending["count"] or 0),
        }

    def preview_trained_recall_set(self) -> dict[str, Any]:
        with self._lock:
            active = self._active_recall_set_tx(self._conn)
            current = self._conn.execute(
                """SELECT
                     SUM(CASE WHEN rm.eligibility='primary' AND m.state IN ('active','cold') THEN 1 ELSE 0 END) primary_count,
                     SUM(CASE WHEN rm.eligibility='evidence_only' AND m.state IN ('active','cold') THEN 1 ELSE 0 END) evidence_count
                   FROM memory_recall_memberships rm JOIN memories m ON m.id=rm.memory_id
                   WHERE rm.recall_set_id=? AND rm.revoked_at IS NULL""",
                (active["recall_set_id"],),
            ).fetchone()
            carried = self._conn.execute(
                """SELECT COUNT(*) count FROM memories m
                   WHERE m.state IN ('active','cold')
                     AND m.record_role<>'reference' AND m.source_type<>'vault_markdown'
                     AND (
                       m.source_category='OPERATOR_APPROVED'
                       OR EXISTS(
                         SELECT 1 FROM memory_creation_proposals p
                         WHERE p.result_memory_id=m.id AND p.status='remembered'
                       )
                     )"""
            ).fetchone()
            references = self._conn.execute(
                """SELECT COUNT(*) count FROM memories
                   WHERE state IN ('active','cold')
                     AND (record_role='reference' OR source_type='vault_markdown')"""
            ).fetchone()
            pending = self._conn.execute(
                """SELECT COUNT(*) count FROM memory_creation_proposals
                   WHERE status IN ('pending','needs_context')"""
            ).fetchone()
        current_primary = int(current["primary_count"] or 0)
        current_evidence = int(current["evidence_count"] or 0)
        approved = int(carried["count"] or 0)
        reference_count = int(references["count"] or 0)
        trained_total = approved + reference_count
        held_back = max(0, current_primary + current_evidence - trained_total)
        preview_material = {
            "active": str(active["recall_set_id"]),
            "primary": current_primary,
            "evidence": current_evidence,
            "approved": approved,
            "references": reference_count,
            "pending": int(pending["count"] or 0),
        }
        preview_id = hashlib.sha256(_trace_json(preview_material).encode("utf-8")).hexdigest()[:20]
        return {
            "preview_id": preview_id,
            "summary": (
                "No memory content will be deleted. Legacy personal memories leave ordinary recall; "
                "operator-approved memories carry forward and references remain lookup-only."
            ),
            "legacy_eligible_count": current_primary + current_evidence,
            "trained_eligible_count": trained_total,
            "excluded_count": held_back,
            "held_back_count": held_back,
            "legacy_primary_excluded": max(
                0,
                current_primary
                - reference_count
                - approved,
            ),
            "references_kept_as_evidence": reference_count,
            "approved_memories_carried": approved,
            "pending_creation_count": int(pending["count"] or 0),
            "counts": {
                "legacy_eligible": current_primary + current_evidence,
                "trained_eligible": trained_total,
                "excluded": held_back,
                "pending_creation": int(pending["count"] or 0),
            },
        }

    def start_trained_recall_set(
        self,
        *,
        label: str = "Trained set 1",
        actor: str = "dashboard-operator",
        reason: str = "Operator activated a reversible clean start.",
        expected_preview_id: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        actor_value = normalize_text(actor)[:80] or "dashboard-operator"
        reason_value = normalize_text(reason)[:500]
        with self.transaction() as conn:
            confirmed_preview = self.preview_trained_recall_set()
            expected_value = normalize_text(expected_preview_id or "")[:40]
            if expected_value and expected_value != str(confirmed_preview["preview_id"]):
                raise ValueError(
                    "The clean-start preview changed. Review the updated counts before activating it."
                )
            prior = self._active_recall_set_tx(conn)
            if str(prior["kind"]) == "trained":
                return self.recall_set_snapshot()
            recall_set_id = str(uuid.uuid4())
            conn.execute(
                "UPDATE memory_recall_sets SET status='inactive',deactivated_at=? WHERE status='active'",
                (now,),
            )
            conn.execute(
                """INSERT INTO memory_recall_sets(
                     recall_set_id,label,kind,status,parent_recall_set_id,actor,reason,
                     created_at,activated_at
                   ) VALUES(?,?,'trained','active',?,?,?,?,?)""",
                (
                    recall_set_id,
                    normalize_text(label)[:120] or "Trained set 1",
                    prior["recall_set_id"],
                    actor_value,
                    reason_value,
                    now,
                    now,
                ),
            )
            # Reviewed creations and trusted imports are the personal seed.
            # Carry the prior membership's authority instead of attributing an
            # automatic judgment to the operator who activates this set.
            conn.execute(
                """INSERT OR IGNORE INTO memory_recall_memberships(
                     recall_set_id,memory_id,eligibility,origin,review_id,actor,reason,created_at
                   )
                   SELECT ?,m.id,'primary',
                          COALESCE(prior_membership.origin,
                            CASE WHEN m.approval_state='automatic_approved'
                              THEN 'automatic_judgment' ELSE 'operator_approval' END),
                          COALESCE(prior_membership.review_id,(
                            SELECT p.review_id FROM memory_creation_proposals p
                            WHERE p.result_memory_id=m.id AND p.status='remembered'
                            ORDER BY p.decided_at DESC LIMIT 1
                          )),
                          COALESCE(prior_membership.actor,(
                            SELECT p.actor FROM memory_creation_proposals p
                            WHERE p.result_memory_id=m.id AND p.status='remembered'
                            ORDER BY p.decided_at DESC LIMIT 1
                          ),?),
                          COALESCE(prior_membership.reason,
                            'Carried into the trained set from Creation review.'),?
                   FROM memories m
                   LEFT JOIN memory_recall_memberships prior_membership
                     ON prior_membership.recall_set_id=?
                    AND prior_membership.memory_id=m.id
                    AND prior_membership.revoked_at IS NULL
                   WHERE m.state IN ('active','cold')
                     AND m.record_role<>'reference' AND m.source_type<>'vault_markdown'
                     AND (
                       m.source_category='OPERATOR_APPROVED'
                       OR EXISTS(
                         SELECT 1 FROM memory_creation_proposals p
                         WHERE p.result_memory_id=m.id AND p.status='remembered'
                       )
                     )""",
                (recall_set_id, actor_value, now, prior["recall_set_id"]),
            )
            # Raw documents remain explicit-lookup only while retaining the
            # authority that admitted their prior membership.
            conn.execute(
                """INSERT OR IGNORE INTO memory_recall_memberships(
                     recall_set_id,memory_id,eligibility,origin,review_id,actor,reason,created_at
                   )
                   SELECT ?,m.id,'evidence_only',
                          COALESCE(prior_membership.origin,'reference_carryover'),
                          prior_membership.review_id,
                          COALESCE(prior_membership.actor,?),
                          COALESCE(prior_membership.reason,
                            'Preserved as explicit lookup evidence, not ordinary personal recall.'),?
                   FROM memories m
                   LEFT JOIN memory_recall_memberships prior_membership
                     ON prior_membership.recall_set_id=?
                    AND prior_membership.memory_id=m.id
                    AND prior_membership.revoked_at IS NULL
                   WHERE m.state IN ('active','cold')
                     AND (m.record_role='reference' OR m.source_type='vault_markdown')""",
                (
                    recall_set_id,
                    actor_value,
                    now,
                    prior["recall_set_id"],
                ),
            )
            conn.execute(
                """INSERT INTO memory_recall_set_events(
                     event_id,event_type,recall_set_id,prior_recall_set_id,actor,reason,details_json,created_at
                   ) VALUES(?,'activated',?,?,?,?,?,?)""",
                (
                    str(uuid.uuid4()),
                    recall_set_id,
                    prior["recall_set_id"],
                    actor_value,
                    reason_value,
                    _trace_json(
                        {
                            "reversible": True,
                            "content_changed": False,
                            "confirmed_preview_id": confirmed_preview["preview_id"],
                            "confirmed_counts": confirmed_preview["counts"],
                        }
                    ),
                    now,
                ),
            )
        return self.recall_set_snapshot()

    def activate_recall_set(
        self,
        recall_set_id: str,
        *,
        actor: str = "dashboard-operator",
        reason: str = "Operator switched the active recall set.",
    ) -> dict[str, Any]:
        target_id = normalize_text(recall_set_id)
        now = utc_now()
        with self.transaction() as conn:
            prior = self._active_recall_set_tx(conn)
            target = conn.execute(
                "SELECT * FROM memory_recall_sets WHERE recall_set_id=?", (target_id,)
            ).fetchone()
            if not target:
                raise ValueError("recall set not found")
            if str(prior["recall_set_id"]) == target_id:
                return self.recall_set_snapshot()
            conn.execute(
                "UPDATE memory_recall_sets SET status='inactive',deactivated_at=? WHERE status='active'",
                (now,),
            )
            conn.execute(
                """UPDATE memory_recall_sets SET status='active',activated_at=?,deactivated_at=NULL
                   WHERE recall_set_id=?""",
                (now, target_id),
            )
            conn.execute(
                """INSERT INTO memory_recall_set_events(
                     event_id,event_type,recall_set_id,prior_recall_set_id,actor,reason,details_json,created_at
                   ) VALUES(?,'switched',?,?,?,?,?,?)""",
                (
                    str(uuid.uuid4()),
                    target_id,
                    prior["recall_set_id"],
                    normalize_text(actor)[:80] or "dashboard-operator",
                    normalize_text(reason)[:500],
                    _trace_json({"reversible": True, "content_changed": False}),
                    now,
                ),
            )
        return self.recall_set_snapshot()

    def promote_memory_to_active_recall_set(
        self,
        memory_id: str,
        *,
        eligibility: str = "primary",
        actor: str = "dashboard-operator",
        reason: str = "Operator promoted this preserved memory into the active recall set.",
        review_id: str | None = None,
        origin_source_category: str | None = None,
        approval_authority: str = "operator",
    ) -> dict[str, Any]:
        eligibility_value = normalize_text(eligibility).casefold()
        if eligibility_value not in {"primary", "evidence_only"}:
            raise ValueError("recall eligibility must be primary or evidence_only")
        authority_value = normalize_text(approval_authority).casefold() or "operator"
        if authority_value not in {"operator", "automatic"}:
            raise ValueError("recall approval authority must be operator or automatic")
        approval_state = (
            "automatic_approved" if authority_value == "automatic" else "operator_approved"
        )
        membership_origin = (
            "automatic_judgment" if authority_value == "automatic" else "operator_promotion"
        )
        now = utc_now()
        with self.transaction() as conn:
            memory = conn.execute("SELECT id FROM memories WHERE id=?", (memory_id,)).fetchone()
            if not memory:
                raise ValueError("memory not found")
            origin_value = normalize_text(origin_source_category or "").upper()[:120]
            conn.execute(
                """UPDATE memories SET approval_state=CASE
                       WHEN ?='automatic_approved'
                        AND approval_state IN ('operator_approved','trusted_import')
                         THEN approval_state
                       ELSE ? END,
                     origin_source_category=CASE WHEN ?<>'' THEN ? ELSE origin_source_category END,
                     updated_at=? WHERE id=?""",
                (approval_state, approval_state, origin_value, origin_value, now, memory_id),
            )
            active = self._active_recall_set_tx(conn)
            existing = conn.execute(
                """SELECT eligibility,revoked_at FROM memory_recall_memberships
                   WHERE recall_set_id=? AND memory_id=?""",
                (active["recall_set_id"], memory_id),
            ).fetchone()
            changed = not existing or str(existing["eligibility"]) != eligibility_value or bool(existing["revoked_at"])
            conn.execute(
                """INSERT INTO memory_recall_memberships(
                     recall_set_id,memory_id,eligibility,origin,review_id,actor,reason,created_at,revoked_at
                   ) VALUES(?,?,?,?,?,?,?,?,NULL)
                   ON CONFLICT(recall_set_id,memory_id) DO UPDATE SET
                     eligibility=excluded.eligibility,origin=excluded.origin,review_id=excluded.review_id,
                     actor=excluded.actor,reason=excluded.reason,revoked_at=NULL""",
                (
                    active["recall_set_id"],
                    memory_id,
                    eligibility_value,
                    membership_origin,
                    review_id,
                    normalize_text(actor)[:80] or "dashboard-operator",
                    normalize_text(reason)[:500],
                    now,
                ),
            )
            if changed:
                conn.execute(
                    """INSERT INTO memory_recall_set_events(
                         event_id,event_type,recall_set_id,memory_id,eligibility,actor,reason,
                         details_json,created_at
                       ) VALUES(?,'promoted',?,?,?,?,?,'{}',?)""",
                    (
                        str(uuid.uuid4()),
                        active["recall_set_id"],
                        memory_id,
                        eligibility_value,
                        normalize_text(actor)[:80] or "dashboard-operator",
                        normalize_text(reason)[:500],
                        now,
                    ),
                )
        return {
            "changed": changed,
            "memory_id": memory_id,
            "eligibility": eligibility_value,
            "active_recall_set_id": str(active["recall_set_id"]),
        }

    def neighborhood_search(
        self,
        query: str,
        *,
        limit: int = 40,
        include_archived: bool = False,
        evidence_lookup: bool = False,
    ) -> list[dict[str, Any]]:
        """Expand only through explicitly named semantic neighborhoods."""

        tokens = set(query_tokens(query))
        selected: set[str] = set()
        if re.search(r"\b(?:passwords?|credentials?|logins?|secret manager|1password|bitwarden)\b", query, re.I):
            selected.add("credentials")
        if re.search(r"\b(?:tools?|commands?|integrations?|workflows?)\b", query, re.I):
            selected.add("tools")
        with self._lock:
            neighborhoods = self._conn.execute(
                "SELECT neighborhood_id,slug,label,category FROM memory_neighborhoods"
            ).fetchall()
            for row in neighborhoods:
                label_tokens = set(query_tokens(str(row["label"])))
                if (
                    str(row["category"]) in {"project", "service"}
                    and label_tokens
                    and any(token in tokens for token in label_tokens if len(token) >= 4)
                ):
                    selected.add(str(row["slug"]))
            if not selected:
                return []
            neighborhood_ids = [f"neighborhood:{slug}" for slug in sorted(selected)]
            states = ("active", "cold", "archived") if include_archived else ("active", "cold")
            eligibilities = ("primary", "evidence_only") if evidence_lookup else ("primary",)
            neighborhood_placeholders = ",".join("?" for _ in neighborhood_ids)
            state_placeholders = ",".join("?" for _ in states)
            eligibility_placeholders = ",".join("?" for _ in eligibilities)
            rows = self._conn.execute(
                f"""SELECT m.*,MAX(nm.confidence) neighborhood_candidate_score,
                           GROUP_CONCAT(n.label, ', ') neighborhood_labels
                    FROM memory_neighborhood_memberships nm
                    JOIN memory_neighborhoods n ON n.neighborhood_id=nm.neighborhood_id
                    JOIN memories m ON m.id=nm.memory_id
                    JOIN memory_recall_memberships rm
                      ON rm.memory_id=m.id AND rm.revoked_at IS NULL
                    JOIN memory_recall_sets rs
                      ON rs.recall_set_id=rm.recall_set_id AND rs.status='active'
                    WHERE nm.neighborhood_id IN ({neighborhood_placeholders})
                      AND m.state IN ({state_placeholders})
                      AND rm.eligibility IN ({eligibility_placeholders})
                    GROUP BY m.id
                    ORDER BY neighborhood_candidate_score DESC,m.importance DESC,m.updated_at DESC
                    LIMIT ?""",
                (*neighborhood_ids, *states, *eligibilities, max(1, int(limit))),
            ).fetchall()
        return [_decode_memory_metadata(row) for row in rows]

    def due_prospective_memories(self, *, limit: int = 2) -> list[dict[str, Any]]:
        """Return only open commitments whose explicit due time has arrived."""

        now = utc_now()
        with self._lock:
            rows = self._conn.execute(
                """SELECT m.*,1.0 prospective_due_score
                   FROM prospective_items p JOIN memories m ON m.id=p.memory_id
                   JOIN memory_recall_memberships rm
                     ON rm.memory_id=m.id AND rm.revoked_at IS NULL AND rm.eligibility='primary'
                   JOIN memory_recall_sets rs
                     ON rs.recall_set_id=rm.recall_set_id AND rs.status='active'
                   WHERE p.status='open' AND p.due_at IS NOT NULL AND p.due_at<>''
                     AND p.due_at<=? AND m.state IN ('active','cold')
                   ORDER BY p.due_at,m.importance DESC LIMIT ?""",
                (now, max(1, min(int(limit), 5))),
            ).fetchall()
        return [_decode_memory_metadata(row) for row in rows]

    def has_due_prospective_memory(self) -> bool:
        return bool(self.due_prospective_memories(limit=1))

    def neighborhood_training_snapshot(self, *, limit: int = 100) -> dict[str, Any]:
        """Explain every current and recent named-neighborhood admission decision."""

        bounded = max(1, min(int(limit), 1000))
        with self._lock:
            evaluations = self._conn.execute(
                """SELECT * FROM memory_neighborhood_evaluations
                   ORDER BY decision='admitted' DESC,category,display_name LIMIT ?""",
                (bounded,),
            ).fetchall()
            events = self._conn.execute(
                """SELECT * FROM memory_neighborhood_decision_events
                   ORDER BY created_at DESC LIMIT ?""",
                (bounded,),
            ).fetchall()
        return {
            "minimum_memories": DYNAMIC_NEIGHBORHOOD_MIN_MEMORIES,
            "minimum_contexts": DYNAMIC_NEIGHBORHOOD_MIN_CONTEXTS,
            "evaluations": [dict(row) for row in evaluations],
            "recent_events": [dict(row) for row in events],
            "rule": (
                "Names become project or service neighborhoods only after repeated operator-approved "
                "canonical memories establish them across independent contexts."
            ),
        }

    def memory_experience_strengths(
        self, memory_ids: Sequence[str]
    ) -> dict[str, dict[str, Any]]:
        """Return bounded, outcome-backed reinforcement evidence for retrieval.

        Selection and injection are intentionally not positive evidence. Credit
        comes from actual use and from helpful/validated outcomes across
        independent tasks, sessions, and days; repetition in one burst has
        diminishing value.
        """

        ids = list(dict.fromkeys(str(value) for value in memory_ids if value))
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT memory_id,
                       COUNT(DISTINCT CASE WHEN event_type='memory_used' THEN task_id END) used_tasks,
                       COUNT(DISTINCT CASE WHEN event_type='memory_used' THEN session_id END) used_sessions,
                       COUNT(DISTINCT CASE WHEN event_type='memory_used' THEN event_day END) used_days,
                       COUNT(DISTINCT CASE WHEN event_type='outcome_labeled'
                         AND outcome IN ('helpful','validated') THEN task_id END) positive_tasks,
                       COUNT(DISTINCT CASE WHEN event_type='outcome_labeled'
                         AND outcome IN ('helpful','validated') THEN event_day END) positive_days,
                       SUM(CASE WHEN event_type='outcome_labeled'
                         AND outcome IN ('harmful','corrected') THEN 1 ELSE 0 END) negative_outcomes,
                       SUM(CASE WHEN event_type='memory_selected' THEN 1 ELSE 0 END) selections,
                       SUM(CASE WHEN event_type='memory_used' THEN 1 ELSE 0 END) uses
                    FROM memory_experience_events
                    WHERE memory_id IN ({placeholders}) GROUP BY memory_id""",
                tuple(ids),
            ).fetchall()
        result: dict[str, dict[str, Any]] = {}
        for raw in rows:
            row = dict(raw)
            positive_tasks = int(row.get("positive_tasks") or 0)
            positive_days = int(row.get("positive_days") or 0)
            used_tasks = int(row.get("used_tasks") or 0)
            used_days = int(row.get("used_days") or 0)
            independent_contexts = max(
                used_tasks,
                int(row.get("used_sessions") or 0),
            )
            meaningful = max(positive_tasks, min(used_tasks, 2))
            spacing = max(positive_days, min(used_days, 2))
            strength = min(
                1.0,
                0.28 * math.log1p(meaningful)
                + 0.22 * min(3, spacing) / 3.0
                + 0.18 * min(3, independent_contexts) / 3.0,
            )
            selections = int(row.get("selections") or 0)
            uses = int(row.get("uses") or 0)
            result[str(row["memory_id"])] = {
                "spaced_reinforcement": round(strength, 6),
                "used_tasks": used_tasks,
                "used_days": used_days,
                "positive_tasks": positive_tasks,
                "positive_days": positive_days,
                "independent_contexts": independent_contexts,
                "selected_unused": max(0, selections - uses),
                "negative_outcomes": int(row.get("negative_outcomes") or 0),
            }
        return result

    def learning_experience_dataset(
        self, *, limit: int = 10000, include_text: bool = False
    ) -> dict[str, Any]:
        """Export privacy-safe supervised experiences for offline replay/training.

        The default representation contains decisions, outcome labels, bounded
        metadata, and stable local identifiers but no memory or conversation
        text. Text is available only through an explicit local opt-in.
        """

        bounded = max(1, min(int(limit), 100000))
        with self._lock:
            rows = self._conn.execute(
                """SELECT e.*,m.kind memory_kind,m.origin_source_category,m.approval_state,
                          CASE WHEN ? THEN m.content ELSE NULL END memory_text,
                          CASE WHEN ? THEN p.content ELSE NULL END proposal_text
                   FROM memory_experience_events e
                   LEFT JOIN memories m ON m.id=e.memory_id
                   LEFT JOIN memory_creation_proposals p ON p.proposal_id=e.proposal_id
                   ORDER BY e.created_at DESC LIMIT ?""",
                (int(include_text), int(include_text), bounded),
            ).fetchall()
        experiences: list[dict[str, Any]] = []
        for raw in rows:
            item = dict(raw)
            material = str(item["evidence_key"])
            bucket = int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:8], 16) % 10
            item["split"] = "train" if bucket < 8 else "validation" if bucket == 8 else "test"
            item["metadata"] = _trace_json_object(item.pop("metadata_json"))
            if not include_text:
                item.pop("memory_text", None)
                item.pop("proposal_text", None)
            experiences.append(item)
        counts: dict[str, int] = {}
        split_counts: dict[str, int] = {}
        for item in experiences:
            key = str(item["event_type"])
            counts[key] = counts.get(key, 0) + 1
            split = str(item["split"])
            split_counts[split] = split_counts.get(split, 0) + 1
        return {
            "schema": "cortex-learning-experience-v1",
            "privacy_mode": "explicit_text_opt_in" if include_text else "no_memory_text",
            "claim_boundary": (
                "These examples train and evaluate Cortex admission, retrieval, connection, and retention "
                "policies. They do not fine-tune the agent model unless a separate approved pipeline does so."
            ),
            "counts": counts,
            "split_counts": split_counts,
            "experiences": experiences,
        }

    def decision_log(self, *, limit: int = 200) -> list[dict[str, Any]]:
        """Compose the append-only ledgers into one human-readable timeline."""

        bounded = max(1, min(int(limit), 2000))
        with self._lock:
            reviews = self._conn.execute(
                "SELECT * FROM operator_review_decisions ORDER BY created_at DESC LIMIT ?",
                (bounded,),
            ).fetchall()
            lifecycle = self._conn.execute(
                "SELECT * FROM lifecycle_events ORDER BY created_at DESC LIMIT ?", (bounded,)
            ).fetchall()
            neighborhoods = self._conn.execute(
                "SELECT * FROM memory_neighborhood_decision_events ORDER BY created_at DESC LIMIT ?",
                (bounded,),
            ).fetchall()
            recall_sets = self._conn.execute(
                "SELECT * FROM memory_recall_set_events ORDER BY created_at DESC LIMIT ?", (bounded,)
            ).fetchall()
            policies = self._conn.execute(
                "SELECT * FROM policy_events ORDER BY created_at DESC LIMIT ?", (bounded,)
            ).fetchall()
            writes = self._conn.execute(
                "SELECT * FROM memory_write_decisions ORDER BY created_at DESC LIMIT ?", (bounded,)
            ).fetchall()
            mechanics = self._conn.execute(
                """SELECT decision_id id,'semantic_consolidation' mechanic,status action,
                          reason summary,created_at,left_id first_id,right_id second_id,
                          result_memory_id result_id,reversed_at
                   FROM semantic_consolidation_decisions
                   UNION ALL
                   SELECT decision_id,'adaptive_pruning',status,reason,created_at,
                          memory_id,NULL,NULL,reversed_at
                   FROM adaptive_pruning_decisions
                   UNION ALL
                   SELECT proposal_id,'scoring_weights',status,reason,created_at,
                          task_type,NULL,NULL,rolled_back_at
                   FROM scoring_weight_proposals
                   UNION ALL
                   SELECT proposal_id,'adaptive_reconsolidation',status,reason,created_at,
                          memory_id,evidence_memory_id,NULL,reversed_at
                   FROM adaptive_reconsolidation_proposals
                   UNION ALL
                   SELECT proposal_id,'schema_formation',status,reason,created_at,
                          result_memory_id,NULL,result_memory_id,reversed_at
                   FROM schema_formation_proposals
                   ORDER BY created_at DESC LIMIT ?""",
                (bounded,),
            ).fetchall()
        entries: list[dict[str, Any]] = []
        for raw in reviews:
            row = dict(raw)
            entries.append(
                {
                    "id": str(row["review_id"]),
                    "category": "operator_review",
                    "action": str(row["action"]),
                    "summary": f"{str(row['item_type']).replace('_', ' ').title()}: {str(row['reason_code']).replace('_', ' ')}",
                    "actor": str(row["actor"] or "operator"),
                    "created_at": row["created_at"],
                    "affected_ids": [value for value in (row.get("src_id"), row.get("dst_id"), row.get("proposal_id")) if value],
                    "before": _trace_json_object(row.get("prior_json")),
                    "after": _trace_json_object(row.get("effect_json")),
                    "scope": str(row.get("decision_scope") or "item_only"),
                    "reversible": True,
                    "reversed_at": row.get("reversed_at"),
                }
            )
        for raw in lifecycle:
            row = dict(raw)
            entries.append(
                {
                    "id": f"lifecycle:{row['event_id']}",
                    "category": "lifecycle",
                    "action": str(row["to_state"]),
                    "summary": str(row["reason"] or "Memory lifecycle state changed."),
                    "actor": "cortex",
                    "created_at": row["created_at"],
                    "affected_ids": [str(row["memory_id"])],
                    "before": {"state": row["from_state"]},
                    "after": {"state": row["to_state"]},
                    "scope": "item_only",
                    "reversible": str(row["to_state"]) != "tombstoned",
                    "reversed_at": None,
                }
            )
        for raw in neighborhoods:
            row = dict(raw)
            entries.append(
                {
                    "id": str(row["event_id"]),
                    "category": "neighborhood",
                    "action": str(row["decision"]),
                    "summary": f"{row['category'].title()} {row['display_name']}: {row['reason']}",
                    "actor": "cortex-rule",
                    "created_at": row["created_at"],
                    "affected_ids": [f"{row['category']}:{row['name_key']}"],
                    "before": {"decision": row["prior_decision"]},
                    "after": {"decision": row["decision"], "memory_count": row["memory_count"], "context_count": row["context_count"]},
                    "scope": "schema",
                    "reversible": True,
                    "reversed_at": None,
                }
            )
        for raw in recall_sets:
            row = dict(raw)
            details = _trace_json_object(row.get("details_json"))
            entries.append(
                {
                    "id": str(row["event_id"]),
                    "category": "recall_set",
                    "action": str(row["event_type"]),
                    "summary": str(row["reason"] or "Recall-set membership changed."),
                    "actor": str(row["actor"] or "cortex"),
                    "created_at": row["created_at"],
                    "affected_ids": [value for value in (row.get("recall_set_id"), row.get("memory_id")) if value],
                    "before": {"recall_set_id": row.get("prior_recall_set_id")},
                    "after": details,
                    "scope": "recall_boundary",
                    "reversible": bool(details.get("reversible", True)),
                    "reversed_at": None,
                }
            )
        for raw in policies:
            row = dict(raw)
            entries.append(
                {
                    "id": str(row["event_id"]),
                    "category": "policy",
                    "action": str(row["event_type"]),
                    "summary": f"Policy {str(row['event_type']).replace('_', ' ')}.",
                    "actor": str(row["actor"] or "cortex"),
                    "created_at": row["created_at"],
                    "affected_ids": [value for value in (row.get("candidate_id"), row.get("version_id")) if value],
                    "before": {},
                    "after": _trace_json_object(row.get("details_json")),
                    "scope": "policy",
                    "reversible": str(row["event_type"]) == "promoted",
                    "reversed_at": None,
                }
            )
        for raw in writes:
            row = dict(raw)
            entries.append(
                {
                    "id": str(row["decision_id"]),
                    "category": "admission",
                    "action": str(row["decision"]),
                    "summary": str(row["reason"]),
                    "actor": "cortex-admission",
                    "created_at": row["created_at"],
                    "affected_ids": [value for value in (row.get("memory_id"), row.get("candidate_hash")) if value],
                    "before": {},
                    "after": {"durability": row["durability"], "reusable_score": row["reusable_score"]},
                    "scope": "admission",
                    "reversible": str(row["decision"]) != "ignored",
                    "reversed_at": None,
                }
            )
        for raw in mechanics:
            row = dict(raw)
            entries.append(
                {
                    "id": str(row["id"]),
                    "category": "brain_mechanics",
                    "action": str(row["action"]),
                    "summary": f"{str(row['mechanic']).replace('_', ' ').title()}: {row['summary']}",
                    "actor": "cortex-auto-judge",
                    "created_at": row["created_at"],
                    "affected_ids": [
                        value
                        for value in (
                            row.get("first_id"),
                            row.get("second_id"),
                            row.get("result_id"),
                        )
                        if value
                    ],
                    "before": {},
                    "after": {"mechanic": row["mechanic"], "status": row["action"]},
                    "scope": "proposal_then_explicit_gate",
                    "reversible": True,
                    "reversed_at": row.get("reversed_at"),
                }
            )
        entries.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return entries[:bounded]

    def fts_search(
        self,
        query: str,
        *,
        limit: int = 40,
        include_archived: bool = False,
        evidence_lookup: bool = False,
    ) -> list[dict[str, Any]]:
        tokens = query_tokens(query)
        states = ("active", "cold", "archived") if include_archived else ("active", "cold")
        eligibilities = ("primary", "evidence_only") if evidence_lookup else ("primary",)
        eligibility_placeholders = ",".join("?" for _ in eligibilities)
        if not tokens:
            placeholders = ",".join("?" for _ in states)
            with self._lock:
                rows = self._conn.execute(
                    f"""SELECT m.*, 8.0 AS fts_rank FROM memories m
                        JOIN memory_recall_memberships rm
                          ON rm.memory_id=m.id AND rm.revoked_at IS NULL
                        JOIN memory_recall_sets rs
                          ON rs.recall_set_id=rm.recall_set_id AND rs.status='active'
                        WHERE m.state IN ({placeholders})
                          AND rm.eligibility IN ({eligibility_placeholders})
                        ORDER BY m.pinned DESC,m.updated_at DESC LIMIT ?""",
                    (*states, *eligibilities, limit),
                ).fetchall()
            return [_decode_memory_metadata(r) for r in rows]
        match = " OR ".join(f'"{t.replace(chr(34), "")}"' for t in tokens)
        placeholders = ",".join("?" for _ in states)
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT m.*, bm25(memory_fts) AS fts_rank
                    FROM memory_fts JOIN memories m ON m.id=memory_fts.memory_id
                    JOIN memory_recall_memberships rm
                      ON rm.memory_id=m.id AND rm.revoked_at IS NULL
                    JOIN memory_recall_sets rs
                      ON rs.recall_set_id=rm.recall_set_id AND rs.status='active'
                    WHERE memory_fts MATCH ? AND m.state IN ({placeholders})
                      AND rm.eligibility IN ({eligibility_placeholders})
                    ORDER BY bm25(memory_fts) LIMIT ?""",
                (match, *states, *eligibilities, limit),
            ).fetchall()
        return [_decode_memory_metadata(r) for r in rows]

    def feature_search(
        self,
        query: str,
        *,
        limit: int = 40,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        """Search the transparent secondary semantic-feature index."""

        features = list(semantic_features(query, max_features=96).items())
        if not features:
            return []
        feature_placeholders = ",".join("?" for _ in features)
        with self._lock:
            frequency_rows = self._conn.execute(
                f"SELECT feature,document_frequency FROM feature_stats WHERE feature IN ({feature_placeholders})",
                tuple(feature for feature, _weight in features),
            ).fetchall()
        frequencies = {str(row["feature"]): int(row["document_frequency"]) for row in frequency_rows}
        # The highest-value rare cues do nearly all of the discriminative work.
        # Bounding the query feature set avoids joining common trigrams across
        # the full corpus on every turn.
        features = [item for item in features if item[0] in frequencies]
        features.sort(
            key=lambda item: (
                item[1] / (1.0 + 0.035 * max(0, frequencies[item[0]] - 1)),
                item[1],
            ),
            reverse=True,
        )
        features = features[:8]
        if not features:
            return []
        states = ("active", "cold", "archived") if include_archived else ("active", "cold")
        values = ",".join("(?,?)" for _ in features)
        state_placeholders = ",".join("?" for _ in states)
        params: list[Any] = []
        for feature, weight in features:
            params.extend((feature, weight))
        params.extend(states)
        params.append("primary")
        params.append(max(1, limit))
        with self._lock:
            rows = self._conn.execute(
                f"""WITH query_features(feature,qweight) AS (VALUES {values}),
                    scored AS (
                      SELECT f.memory_id,
                             SUM(MIN(f.weight,q.qweight) /
                                 (1.0 + 0.035 * MAX(0,fs.document_frequency-1))) AS feature_score,
                             COUNT(*) AS feature_matches
                      FROM query_features q
                      JOIN memory_features f ON f.feature=q.feature
                      JOIN feature_stats fs ON fs.feature=f.feature
                      GROUP BY f.memory_id
                    )
                    SELECT m.*,s.feature_score,s.feature_matches,8.0 AS fts_rank
                    FROM scored s
                    JOIN memories m ON m.id=s.memory_id
                    JOIN memory_recall_memberships rm
                      ON rm.memory_id=m.id AND rm.revoked_at IS NULL
                    JOIN memory_recall_sets rs
                      ON rs.recall_set_id=rm.recall_set_id AND rs.status='active'
                    WHERE m.state IN ({state_placeholders})
                      AND rm.eligibility=?
                    ORDER BY s.feature_score DESC,s.feature_matches DESC,m.pinned DESC
                    LIMIT ?""",
                tuple(params),
            ).fetchall()
        return [_decode_memory_metadata(row) for row in rows]

    def context_search(
        self,
        *,
        active_project: str | None = None,
        entities: Sequence[str] = (),
        scope: dict[str, Any] | None = None,
        system_state: dict[str, Any] | None = None,
        applicable_systems: Sequence[str] = (),
        applicable_versions: Sequence[str] = (),
        include_archived: bool = False,
        evidence_lookup: bool = False,
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        """Find scoped candidates without requiring a text-similarity hit."""

        requested_scope = _normalize_context_map(scope)
        if active_project:
            requested_scope.setdefault("project", normalize_text(active_project)[:300])
        requested_entities = {item.casefold() for item in _normalize_context_list(entities)}
        requested_state = _normalize_context_map(system_state)
        requested_systems = {item.casefold() for item in _normalize_context_list(applicable_systems)}
        requested_versions = {item.casefold() for item in _normalize_context_list(applicable_versions)}
        terms: list[tuple[str, str, str]] = []
        terms.extend(("scope", key, value.casefold()) for key, value in requested_scope.items())
        terms.extend(("entity", "", value) for value in requested_entities)
        terms.extend(("precondition", key, value.casefold()) for key, value in requested_state.items())
        terms.extend(("system", "", value) for value in requested_systems)
        terms.extend(("version", "", value) for value in requested_versions)
        if not terms:
            return []
        states = ("active", "cold", "archived") if include_archived else ("active", "cold")
        state_placeholders = ",".join("?" for _ in states)
        eligibilities = ("primary", "evidence_only") if evidence_lookup else ("primary",)
        eligibility_placeholders = ",".join("?" for _ in eligibilities)
        term_clauses = " OR ".join(
            "(t.term_type=? AND t.term_key=? AND t.term_value=?)" for _ in terms
        )
        term_params = [item for term in terms for item in term]
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT m.*,
                           SUM(CASE WHEN t.term_type IN ('scope','precondition') THEN 2.0 ELSE 1.0 END)
                             context_candidate_score
                    FROM memory_context_terms t JOIN memories m ON m.id=t.memory_id
                    JOIN memory_recall_memberships rm
                      ON rm.memory_id=m.id AND rm.revoked_at IS NULL
                    JOIN memory_recall_sets rs
                      ON rs.recall_set_id=rm.recall_set_id AND rs.status='active'
                    WHERE m.context_mode='context_dependent'
                      AND m.state IN ({state_placeholders})
                      AND rm.eligibility IN ({eligibility_placeholders})
                      AND ({term_clauses})
                    GROUP BY m.id
                    ORDER BY context_candidate_score DESC,m.metadata_completeness DESC,
                             m.importance DESC,m.updated_at DESC
                    LIMIT ?""",
                (*states, *eligibilities, *term_params, max(1, int(limit))),
            ).fetchall()
        return [_decode_memory_metadata(row) for row in rows]

    def association_scores(
        self,
        seed_scores: dict[str, float],
        *,
        depth: int = 1,
        max_nodes: int = 96,
        iterations: int = 7,
        restart: float = 0.74,
    ) -> dict[str, float]:
        """Return bounded Personalized PageRank scores around retrieval seeds."""

        if not seed_scores or depth <= 0:
            return {}
        nodes = set(seed_scores)
        frontier = set(seed_scores)
        with self._lock:
            for _ in range(min(2, depth)):
                if not frontier or len(nodes) >= max_nodes:
                    break
                placeholders = ",".join("?" for _ in frontier)
                rows = self._conn.execute(
                    f"""SELECT e.src_id,e.dst_id,e.relation,e.weight,e.evidence_count FROM edges e
                        JOIN memory_recall_memberships sm
                          ON sm.memory_id=e.src_id AND sm.revoked_at IS NULL AND sm.eligibility='primary'
                        JOIN memory_recall_memberships dm
                          ON dm.memory_id=e.dst_id AND dm.recall_set_id=sm.recall_set_id
                         AND dm.revoked_at IS NULL AND dm.eligibility='primary'
                        JOIN memory_recall_sets rs
                          ON rs.recall_set_id=sm.recall_set_id AND rs.status='active'
                        WHERE e.src_id IN ({placeholders}) OR e.dst_id IN ({placeholders})
                        ORDER BY e.weight DESC,e.evidence_count DESC LIMIT ?""",
                    (*frontier, *frontier, max_nodes * 3),
                ).fetchall()
                expanded: set[str] = set()
                for row in rows:
                    for source, target, _weight in _association_transitions(dict(row)):
                        if source in frontier:
                            expanded.add(target)
                expanded -= nodes
                room = max_nodes - len(nodes)
                frontier = set(sorted(expanded)[:room])
                nodes.update(frontier)
            if not nodes:
                return {}
            placeholders = ",".join("?" for _ in nodes)
            edge_rows = self._conn.execute(
                f"""SELECT e.src_id,e.dst_id,e.relation,e.weight,e.evidence_count FROM edges e
                    JOIN memory_recall_memberships sm
                      ON sm.memory_id=e.src_id AND sm.revoked_at IS NULL AND sm.eligibility='primary'
                    JOIN memory_recall_memberships dm
                      ON dm.memory_id=e.dst_id AND dm.recall_set_id=sm.recall_set_id
                     AND dm.revoked_at IS NULL AND dm.eligibility='primary'
                    JOIN memory_recall_sets rs
                      ON rs.recall_set_id=sm.recall_set_id AND rs.status='active'
                    WHERE e.src_id IN ({placeholders}) AND e.dst_id IN ({placeholders})
                    LIMIT ?""",
                (*nodes, *nodes, max_nodes * 12),
            ).fetchall()

        adjacency: dict[str, list[tuple[str, float]]] = {node: [] for node in nodes}
        for row in edge_rows:
            for source, target, semantic_weight in _association_transitions(dict(row)):
                if source not in adjacency or target not in adjacency:
                    continue
                weight = semantic_weight * min(2.0, math.log1p(int(row["evidence_count"])))
                adjacency[source].append((target, weight))
        seed_total = sum(max(0.0, score) for score in seed_scores.values()) or 1.0
        preference = {node: max(0.0, seed_scores.get(node, 0.0)) / seed_total for node in nodes}
        rank = dict(preference)
        for _ in range(max(1, iterations)):
            next_rank = {node: restart * preference.get(node, 0.0) for node in nodes}
            for node, links in adjacency.items():
                total = sum(weight for _, weight in links)
                if not links or total <= 0:
                    continue
                share = (1.0 - restart) * rank.get(node, 0.0)
                for neighbor, weight in links:
                    next_rank[neighbor] += share * weight / total
            rank = next_rank
        peak = max(rank.values(), default=0.0) or 1.0
        return {node: min(1.0, value / peak) for node, value in rank.items() if value > 0}

    def superseded_ids(self, memory_ids: Sequence[str]) -> set[str]:
        if not memory_ids:
            return set()
        placeholders = ",".join("?" for _ in memory_ids)
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT dst_id FROM edges WHERE relation='supersedes'
                    AND dst_id IN ({placeholders})""",
                tuple(memory_ids),
            ).fetchall()
        return {str(row["dst_id"]) for row in rows}

    def contradicted_ids(self, memory_ids: Sequence[str]) -> set[str]:
        if not memory_ids:
            return set()
        placeholders = ",".join("?" for _ in memory_ids)
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT src_id,dst_id FROM edges WHERE relation='contradicts'
                    AND (src_id IN ({placeholders}) OR dst_id IN ({placeholders}))""",
                (*memory_ids, *memory_ids),
            ).fetchall()
        result: set[str] = set()
        requested = set(memory_ids)
        for row in rows:
            if str(row["src_id"]) in requested:
                result.add(str(row["src_id"]))
            if str(row["dst_id"]) in requested:
                result.add(str(row["dst_id"]))
        return result

    def add_edge(
        self,
        src_id: str,
        dst_id: str,
        relation: str,
        *,
        weight: float = 0.25,
        evidence_type: str | None = None,
        explanation: str | None = None,
        evidence_key: str | None = None,
        source_ref: str | None = None,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Add or reinforce a typed connection with inspectable evidence.

        Connections are not hidden similarity guesses.  Every new edge records
        either caller-supplied evidence or an explicit ``legacy_unattributed``
        marker so the dashboard can say exactly how strong the claim is.
        """
        if not src_id or not dst_id or src_id == dst_id:
            return False
        src_id, dst_id = (
            sorted((src_id, dst_id))
            if relation in {"related", "co_used", "co_observed", "sleep_replay"}
            else (src_id, dst_id)
        )
        now = utc_now()
        evidence_type_value = normalize_text(evidence_type or "legacy_unattributed")[:80]
        explanation_value = normalize_text(
            explanation or _default_edge_explanation(relation, evidence_count=1)
        )[:600]
        evidence_key_value = normalize_text(evidence_key or str(uuid.uuid4()))[:240]
        with self.transaction() as conn:
            if not all(
                conn.execute("SELECT 1 FROM memories WHERE id=?", (memory_id,)).fetchone()
                for memory_id in (src_id, dst_id)
            ):
                return False
            evidence_added = self._record_edge_evidence_tx(
                conn,
                src_id,
                dst_id,
                relation,
                evidence_type=evidence_type_value,
                evidence_key=evidence_key_value,
                summary=explanation_value,
                source_ref=source_ref,
                task_id=task_id,
                metadata=metadata,
                created_at=now,
            )
            existing = conn.execute(
                "SELECT 1 FROM edges WHERE src_id=? AND dst_id=? AND relation=?",
                (src_id, dst_id, relation),
            ).fetchone()
            if not existing:
                evidence_count = conn.execute(
                    """SELECT COUNT(*) count FROM edge_evidence
                       WHERE src_id=? AND dst_id=? AND relation=?""",
                    (src_id, dst_id, relation),
                ).fetchone()["count"]
                conn.execute(
                    """INSERT INTO edges(
                       src_id,dst_id,relation,weight,evidence_count,created_at,last_reinforced_at
                       ) VALUES(?,?,?,?,?,?,?)""",
                    (src_id, dst_id, relation, _clamp(weight), max(1, int(evidence_count)), now, now),
                )
            elif evidence_added:
                conn.execute(
                    """UPDATE edges SET weight=MIN(1.0,weight+?),
                       evidence_count=evidence_count+1,last_reinforced_at=?
                       WHERE src_id=? AND dst_id=? AND relation=?""",
                    (_clamp(weight), now, src_id, dst_id, relation),
                )
            return True

    @staticmethod
    def _record_edge_evidence_tx(
        conn: sqlite3.Connection,
        src_id: str,
        dst_id: str,
        relation: str,
        *,
        evidence_type: str,
        evidence_key: str,
        summary: str,
        source_ref: str | None = None,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> bool:
        evidence_type_value = normalize_text(evidence_type)[:80] or "legacy_unattributed"
        evidence_key_value = normalize_text(evidence_key)[:240] or str(uuid.uuid4())
        summary_value = normalize_text(summary)[:600] or _default_edge_explanation(
            relation,
            evidence_count=1,
        )
        source_ref_value = normalize_text(source_ref or "")[:500] or None
        task_id_value = normalize_text(task_id or "")[:160] or None
        metadata_json_value = _trace_json(metadata or {})
        existing = conn.execute(
            """SELECT evidence_id FROM edge_evidence
               WHERE src_id=? AND dst_id=? AND relation=?
                 AND evidence_type=? AND evidence_key=?""",
            (src_id, dst_id, relation, evidence_type_value, evidence_key_value),
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE edge_evidence
                   SET summary=?,source_ref=?,task_id=?,metadata_json=?
                   WHERE evidence_id=?""",
                (
                    summary_value,
                    source_ref_value,
                    task_id_value,
                    metadata_json_value,
                    existing["evidence_id"],
                ),
            )
            return False

        conn.execute(
            """INSERT INTO edge_evidence(
               evidence_id,src_id,dst_id,relation,evidence_type,evidence_key,summary,
               source_ref,task_id,metadata_json,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                str(uuid.uuid4()),
                src_id,
                dst_id,
                relation,
                evidence_type_value,
                evidence_key_value,
                summary_value,
                source_ref_value,
                task_id_value,
                metadata_json_value,
                created_at or utc_now(),
            ),
        )
        return True

    def edge_evidence(
        self,
        src_id: str,
        dst_id: str,
        relation: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if relation in {"related", "co_used", "co_observed", "sleep_replay"}:
            src_id, dst_id = sorted((src_id, dst_id))
        with self._lock:
            rows = self._conn.execute(
                """SELECT evidence_id,evidence_type,evidence_key,summary,source_ref,task_id,
                          metadata_json,created_at
                   FROM edge_evidence WHERE src_id=? AND dst_id=? AND relation=?
                   ORDER BY created_at DESC LIMIT ?""",
                (src_id, dst_id, relation, max(1, min(int(limit), 100))),
            ).fetchall()
        return [
            {**dict(row), "metadata": _trace_json_object(row["metadata_json"])}
            for row in rows
        ]

    def resolve_contradiction(self, first_id: str, second_id: str, resolution: str) -> bool:
        """Resolve one contradictory pair while preserving an auditable history."""
        if resolution not in {"keep_first", "keep_second", "both_valid", "trash_both"}:
            raise ValueError("invalid contradiction resolution")
        if not first_id or not second_id or first_id == second_id:
            raise ValueError("two different memory IDs are required")
        now = utc_now()
        with self.transaction() as conn:
            edge = conn.execute(
                """SELECT * FROM edges WHERE relation='contradicts'
                   AND ((src_id=? AND dst_id=?) OR (src_id=? AND dst_id=?))""",
                (first_id, second_id, second_id, first_id),
            ).fetchone()
            rows = conn.execute(
                "SELECT * FROM memories WHERE id IN (?,?)",
                (first_id, second_id),
            ).fetchall()
            memories = {str(row["id"]): row for row in rows}
            if not edge or len(memories) != 2:
                return False
            if any(memories[memory_id]["state"] not in {"active", "cold"} for memory_id in memories):
                return False

            conn.execute(
                """DELETE FROM edges WHERE relation='contradicts'
                   AND ((src_id=? AND dst_id=?) OR (src_id=? AND dst_id=?))""",
                (first_id, second_id, second_id, first_id),
            )
            if resolution == "trash_both":
                for memory_id in (first_id, second_id):
                    memory = memories[memory_id]
                    conn.execute(
                        "UPDATE memory_versions SET system_to=? WHERE memory_id=? AND system_to IS NULL",
                        (now, memory_id),
                    )
                    conn.execute(
                        """INSERT INTO memory_versions(memory_id,content,confidence,state,valid_from,valid_to,
                           system_from,reason,source_ref) VALUES(?,?,?,?,?,?,?,?,?)""",
                        (
                            memory_id, memory["content"], memory["confidence"], "tombstoned",
                            memory["valid_from"], memory["valid_to"], now,
                            "dashboard conflict review: both rejected", memory["source_ref"],
                        ),
                    )
                    conn.execute("UPDATE memories SET state='tombstoned', updated_at=? WHERE id=?", (now, memory_id))
                    conn.execute(
                        """INSERT INTO lifecycle_events(
                           memory_id,from_state,to_state,reason,retention_score,created_at
                           ) VALUES(?,?,?,?,?,?)""",
                        (memory_id, memory["state"], "tombstoned", "dashboard conflict review: both rejected", None, now),
                    )
                    self._mark_dependents_dirty_tx(conn, memory_id, "evidence rejected in dashboard conflict review")
            elif resolution == "both_valid":
                edge_src, edge_dst = sorted((first_id, second_id))
                conn.execute(
                    """INSERT INTO edges(src_id,dst_id,relation,weight,evidence_count,created_at,last_reinforced_at)
                       VALUES(?,?,'contextual',?,?,?,?)
                       ON CONFLICT(src_id,dst_id,relation) DO UPDATE SET
                         weight=MAX(edges.weight,excluded.weight),
                         evidence_count=MAX(edges.evidence_count,excluded.evidence_count),
                         last_reinforced_at=excluded.last_reinforced_at""",
                    (
                        edge_src,
                        edge_dst,
                        min(1.0, max(0.25, float(edge["weight"]))),
                        int(edge["evidence_count"]),
                        str(edge["created_at"]),
                        now,
                    ),
                )
                self._record_edge_evidence_tx(
                    conn,
                    edge_src,
                    edge_dst,
                    "contextual",
                    evidence_type="operator_review",
                    evidence_key=f"both_valid:{first_id}:{second_id}:{now}",
                    summary=(
                        "An operator reviewed the apparent conflict and confirmed that both memories "
                        "can be valid in different situations."
                    ),
                    metadata={"resolution": resolution},
                    created_at=now,
                )
            else:
                winner_id = first_id if resolution == "keep_first" else second_id
                loser_id = second_id if resolution == "keep_first" else first_id
                loser = memories[loser_id]
                conn.execute(
                    "UPDATE memory_versions SET system_to=? WHERE memory_id=? AND system_to IS NULL",
                    (now, loser_id),
                )
                conn.execute(
                    """INSERT INTO memory_versions(memory_id,content,confidence,state,valid_from,valid_to,
                       system_from,reason,source_ref) VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        loser_id,
                        loser["content"],
                        loser["confidence"],
                        "archived",
                        loser["valid_from"],
                        loser["valid_to"],
                        now,
                        "dashboard conflict review: superseded",
                        loser["source_ref"],
                    ),
                )
                conn.execute("UPDATE memories SET state='archived', updated_at=? WHERE id=?", (now, loser_id))
                conn.execute(
                    """INSERT INTO lifecycle_events(
                       memory_id,from_state,to_state,reason,retention_score,created_at
                       ) VALUES(?,?,?,?,?,?)""",
                    (
                        loser_id,
                        loser["state"],
                        "archived",
                        "dashboard conflict review: superseded",
                        None,
                        now,
                    ),
                )
                self._mark_dependents_dirty_tx(conn, loser_id, "evidence superseded in dashboard conflict review")
                conn.execute(
                    """INSERT INTO edges(src_id,dst_id,relation,weight,evidence_count,created_at,last_reinforced_at)
                       VALUES(?,?,'supersedes',1.0,1,?,?)
                       ON CONFLICT(src_id,dst_id,relation) DO UPDATE SET
                         weight=1.0,evidence_count=edges.evidence_count+1,last_reinforced_at=excluded.last_reinforced_at""",
                    (winner_id, loser_id, now, now),
                )
                self._record_edge_evidence_tx(
                    conn,
                    winner_id,
                    loser_id,
                    "supersedes",
                    evidence_type="operator_review",
                    evidence_key=f"{resolution}:{first_id}:{second_id}:{now}",
                    summary=(
                        "An operator reviewed both memories, selected this one as current, and archived "
                        "the connected memory as superseded."
                    ),
                    metadata={"resolution": resolution},
                    created_at=now,
                )

            conn.executemany(
                "INSERT INTO access_log(memory_id,event,query,created_at) VALUES(?,?,?,?)",
                [
                    (first_id, "conflict_reviewed", resolution, now),
                    (second_id, "conflict_reviewed", resolution, now),
                ],
            )
            conn.execute(
                "INSERT INTO maintenance_log(action,details,dry_run,created_at) VALUES(?,?,0,?)",
                (
                    "dashboard_conflict_review",
                    json.dumps(
                        {"first_id": first_id, "second_id": second_id, "resolution": resolution},
                        sort_keys=True,
                    ),
                    now,
                ),
            )
        return True

    def review_inference(
        self,
        memory_id: str,
        resolution: str,
        *,
        decision_scope: str = "item_only",
    ) -> bool:
        """Review an unsupported inference while preserving the legacy bool API."""

        return bool(
            self.review_inference_with_scope(
                memory_id,
                resolution,
                decision_scope=decision_scope,
            )
        )

    def review_inference_with_scope(
        self,
        memory_id: str,
        resolution: str,
        *,
        decision_scope: str = "item_only",
    ) -> list[str]:
        """Review one inference and, when requested, its eligible exact duplicates."""
        if resolution not in {"confirm", "archive", "trash"}:
            raise ValueError("invalid inference resolution")
        scope_value = _normalize_review_scope(decision_scope)
        if scope_value == "policy_evidence":
            scope_value = "item_only"
        now = utc_now()
        with self.transaction() as conn:
            memory = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            if not memory or memory["source_category"] not in {"AGENT_INFERENCE", "REFLECTION"}:
                return []
            if memory["state"] not in {"active", "cold"}:
                return []
            supported = conn.execute(
                "SELECT 1 FROM memory_dependencies WHERE memory_id=? AND active=1 LIMIT 1",
                (memory_id,),
            ).fetchone()
            if supported:
                return []

            memories = {memory_id: memory}
            if scope_value == "exact_duplicates":
                duplicate_rows = conn.execute(
                    """SELECT * FROM memories
                       WHERE content_hash=? AND context_mode=? AND scope_json=? AND preconditions_json=?
                         AND source_type=? AND source_category=? AND COALESCE(source_ref,'')=?
                         AND id<>? AND state IN ('active','cold')
                         AND NOT EXISTS(
                           SELECT 1 FROM memory_dependencies d
                           WHERE d.memory_id=memories.id AND d.active=1
                         )
                       ORDER BY created_at,id""",
                    (
                        memory["content_hash"], memory["context_mode"], memory["scope_json"],
                        memory["preconditions_json"], memory["source_type"],
                        memory["source_category"], memory["source_ref"] or "", memory_id,
                    ),
                ).fetchall()
                memories.update({str(row["id"]): row for row in duplicate_rows})

            for target_id, target in memories.items():
                if resolution == "confirm":
                    conn.execute(
                        """UPDATE memories SET source_category='USER_EXPLICIT',
                           confidence=MAX(confidence,0.85),trust=MAX(trust,0.85),
                           confirmed_count=confirmed_count+1,protected=1,updated_at=? WHERE id=?""",
                        (now, target_id),
                    )
                    event = "confirmed"
                else:
                    next_state = "tombstoned" if resolution == "trash" else "archived"
                    conn.execute(
                        "UPDATE memory_versions SET system_to=? WHERE memory_id=? AND system_to IS NULL",
                        (now, target_id),
                    )
                    conn.execute(
                        """INSERT INTO memory_versions(memory_id,content,confidence,state,valid_from,valid_to,
                           system_from,reason,source_ref) VALUES(?,?,?,?,?,?,?,?,?)""",
                        (
                            target_id,
                            target["content"],
                            target["confidence"],
                            next_state,
                            target["valid_from"],
                            target["valid_to"],
                            now,
                            f"dashboard inference review: {resolution}",
                            target["source_ref"],
                        ),
                    )
                    conn.execute(
                        "UPDATE memories SET state=?, updated_at=? WHERE id=?",
                        (next_state, now, target_id),
                    )
                    conn.execute(
                        """INSERT INTO lifecycle_events(
                           memory_id,from_state,to_state,reason,retention_score,created_at
                           ) VALUES(?,?,?,?,?,?)""",
                        (
                            target_id,
                            target["state"],
                            next_state,
                            f"dashboard inference review: {resolution}",
                            None,
                            now,
                        ),
                    )
                    self._mark_dependents_dirty_tx(
                        conn, target_id, "inference archived in dashboard review"
                    )
                    event = next_state

                conn.execute(
                    "INSERT INTO access_log(memory_id,event,query,created_at) VALUES(?,?,?,?)",
                    (target_id, event, "dashboard health review", now),
                )
            conn.execute(
                "INSERT INTO maintenance_log(action,details,dry_run,created_at) VALUES(?,?,0,?)",
                (
                    "dashboard_inference_review",
                    json.dumps(
                        {
                            "memory_id": memory_id,
                            "memory_ids": sorted(memories),
                            "resolution": resolution,
                            "decision_scope": scope_value,
                        },
                        sort_keys=True,
                    ),
                    now,
                ),
            )
        return sorted(memories)

    def add_dependency(
        self,
        memory_id: str,
        evidence_id: str,
        *,
        relation: str = "derived_from",
        weight: float = 1.0,
    ) -> bool:
        if memory_id == evidence_id:
            return False
        with self.transaction() as conn:
            if not conn.execute("SELECT 1 FROM memories WHERE id=?", (memory_id,)).fetchone():
                return False
            if not conn.execute("SELECT 1 FROM memories WHERE id=?", (evidence_id,)).fetchone():
                return False
            conn.execute(
                """INSERT INTO memory_dependencies(memory_id,evidence_id,relation,weight,active,created_at)
                   VALUES(?,?,?,?,1,?) ON CONFLICT(memory_id,evidence_id,relation) DO UPDATE SET
                   weight=excluded.weight, active=1""",
                (memory_id, evidence_id, relation, _clamp(weight), utc_now()),
            )
            return True

    def dependencies(self, memory_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT d.*, e.state AS evidence_state, e.confidence AS evidence_confidence,
                   e.currentness_confidence AS evidence_currentness, e.content AS evidence_content
                   FROM memory_dependencies d JOIN memories e ON e.id=d.evidence_id
                   WHERE d.memory_id=? ORDER BY d.weight DESC""",
                (memory_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_dependents_dirty(self, evidence_id: str, reason: str) -> int:
        with self.transaction() as conn:
            return self._mark_dependents_dirty_tx(conn, evidence_id, reason)

    def _mark_dependents_dirty_tx(self, conn: sqlite3.Connection, evidence_id: str, reason: str) -> int:
        result = conn.execute(
            """UPDATE memories SET dirty=1, dirty_reason=? WHERE id IN (
                   SELECT memory_id FROM memory_dependencies WHERE evidence_id=? AND active=1
               )""",
            (reason, evidence_id),
        )
        return result.rowcount

    def repair_dependencies(self, *, dry_run: bool = True) -> dict[str, Any]:
        """Stage or apply confidence repair for derived memories with changed evidence."""
        with self._lock:
            dirty_rows = self._conn.execute("SELECT * FROM memories WHERE dirty=1").fetchall()
        proposals: list[dict[str, Any]] = []
        for row in dirty_rows:
            deps = self.dependencies(row["id"])
            active = [
                d
                for d in deps
                if d["active"]
                and (
                    d["evidence_state"] in {"active", "cold"}
                    or (
                        d["relation"] == "consolidated_from"
                        and d["evidence_state"] == "archived"
                    )
                )
            ]
            if not active:
                proposals.append(
                    {"memory_id": row["id"], "action": "quarantine", "confidence": 0.0, "reason": "no active evidence"}
                )
                continue
            total_weight = sum(float(d["weight"]) for d in active) or 1.0
            confidence = (
                sum(
                    float(d["weight"]) * float(d["evidence_confidence"]) * float(d["evidence_currentness"])
                    for d in active
                )
                / total_weight
            )
            proposals.append(
                {
                    "memory_id": row["id"],
                    "action": "recalculate",
                    "confidence": round(confidence, 6),
                    "reason": row["dirty_reason"],
                }
            )
        if not dry_run:
            with self.transaction() as conn:
                for proposal in proposals:
                    if proposal["action"] == "quarantine":
                        conn.execute(
                            "UPDATE memories SET state='quarantine', confidence=0.0, dirty=0, dirty_reason=NULL, updated_at=? WHERE id=?",
                            (utc_now(), proposal["memory_id"]),
                        )
                    else:
                        conn.execute(
                            "UPDATE memories SET confidence=?, dirty=0, dirty_reason=NULL, updated_at=? WHERE id=?",
                            (proposal["confidence"], utc_now(), proposal["memory_id"]),
                        )
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO maintenance_log(action,details,dry_run,created_at) VALUES(?,?,?,?)",
                ("dependency_repair", json.dumps(proposals, sort_keys=True), int(dry_run), utc_now()),
            )
        return {"dry_run": dry_run, "proposals": proposals, "count": len(proposals)}

    def _link_structured_contradictions(
        self,
        conn: sqlite3.Connection,
        memory_id: str,
        subject: str | None,
        predicate: str | None,
        object_value: str | None,
        valid_from: str | None,
        valid_to: str | None,
        now: str,
    ) -> None:
        if not subject or not predicate or object_value is None:
            return
        rows = conn.execute(
            """SELECT id,object_value,valid_from,valid_to FROM memories
               WHERE id<>? AND subject=? AND predicate=? AND object_value<>?
                 AND state IN ('active','cold')""",
            (memory_id, subject, predicate, object_value),
        ).fetchall()
        for row in rows:
            if not _periods_overlap(valid_from, valid_to, row["valid_from"], row["valid_to"]):
                continue
            src, dst = sorted((memory_id, row["id"]))
            conn.execute(
                """INSERT OR IGNORE INTO edges(src_id,dst_id,relation,weight,evidence_count,created_at,last_reinforced_at)
                   VALUES(?,?,\'contradicts\',1.0,1,?,?)""",
                (src, dst, now, now),
            )
            self._record_edge_evidence_tx(
                conn,
                str(src),
                str(dst),
                "contradicts",
                evidence_type="structured_claim_conflict",
                evidence_key=f"{subject}:{predicate}:{memory_id}:{row['id']}",
                summary=(
                    f"Both memories make a structured claim about {subject} / {predicate}, "
                    "but their values differ during overlapping validity periods."
                ),
                metadata={"subject": subject, "predicate": predicate},
                created_at=now,
            )

    def neighbors(self, memory_ids: Sequence[str], *, limit: int = 30) -> list[dict[str, Any]]:
        if not memory_ids:
            return []
        placeholders = ",".join("?" for _ in memory_ids)
        params = (*memory_ids, *memory_ids, *memory_ids, limit)
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT e.*,
                    CASE WHEN e.src_id IN ({placeholders}) THEN e.dst_id ELSE e.src_id END AS neighbor_id
                    FROM edges e
                    WHERE e.src_id IN ({placeholders}) OR e.dst_id IN ({placeholders})
                    ORDER BY e.weight DESC, e.evidence_count DESC LIMIT ?""",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def log_access(
        self,
        memory_id: str,
        event: str,
        *,
        query: str | None = None,
        session_id: str | None = None,
        score: float | None = None,
    ) -> None:
        columns = {
            "retrieved": ("retrieved_count", "last_retrieved_at"),
            "selected": ("selected_count", "last_retrieved_at"),
            "injected": ("injected_count", "last_injected_at"),
            "used": ("used_count", "last_used_at"),
            "successful": ("success_count", "last_used_at"),
            "confirmed": ("confirmed_count", "last_used_at"),
            "validated": ("validated_count", "last_helpful_at"),
            "helpful": ("helpful_count", "last_helpful_at"),
            "wrong": ("harmful_count", "last_retrieved_at"),
            "irrelevant": ("false_positive_count", "last_retrieved_at"),
        }
        now = utc_now()
        from .research import record_reconsolidation_reuse_tx

        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO access_log(memory_id,event,query,session_id,score,created_at) VALUES(?,?,?,?,?,?)",
                (memory_id, event, query, session_id, score, now),
            )
            if event in columns:
                count_col, time_col = columns[event]
                conn.execute(
                    f"UPDATE memories SET {count_col}={count_col}+1, {time_col}=?, updated_at=updated_at WHERE id=?",
                    (now, memory_id),
                )
                if event == "successful":
                    conn.execute(
                        "UPDATE memories SET helpful_count=helpful_count+1,last_helpful_at=? WHERE id=?",
                        (now, memory_id),
                    )
                elif event == "confirmed":
                    conn.execute(
                        "UPDATE memories SET validated_count=validated_count+1,last_helpful_at=? WHERE id=?",
                        (now, memory_id),
                    )
                elif event == "wrong":
                    conn.execute(
                        "UPDATE memories SET false_positive_count=false_positive_count+1 WHERE id=?",
                        (memory_id,),
                    )
            if event in {"used", "successful", "confirmed", "validated", "helpful"}:
                record_reconsolidation_reuse_tx(conn, memory_id, now)

    def record_recall_run(
        self,
        *,
        session_id: str | None,
        query: str,
        mode: str,
        reason: str,
        requested_limit: int,
        token_budget: int,
        candidate_count: int,
        selected_count: int,
        estimated_tokens: int,
        prepare_ms: float,
        abstained: bool,
        task_id: str | None = None,
    ) -> str:
        recall_id = str(uuid.uuid4())
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO recall_runs(
                   recall_id,task_id,session_id,query,mode,reason,requested_limit,token_budget,
                   candidate_count,selected_count,estimated_tokens,prepare_ms,abstained,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    recall_id,
                    task_id,
                    session_id,
                    normalize_text(query)[:500],
                    mode,
                    reason,
                    int(requested_limit),
                    int(token_budget),
                    int(candidate_count),
                    int(selected_count),
                    int(estimated_tokens),
                    max(0.0, float(prepare_ms)),
                    int(abstained),
                    utc_now(),
                ),
            )
        return recall_id

    def record_memory_trace_decision(
        self,
        *,
        task_id: str,
        session_id: str | None,
        goal: str,
        context_summary: str,
        task_type: str,
        recall_mode: str,
        retrieval_used: bool,
        retrieval_reason: str,
        queries: Sequence[str],
        candidate_memories: Sequence[dict[str, Any]],
        retrieval_context: dict[str, Any] | None = None,
    ) -> str:
        """Record one inspectable retrieval decision and append its JSON event.

        The payload is a concise operational explanation. It deliberately
        stores component scores and decision reason codes, not hidden model
        reasoning or chain-of-thought.
        """

        task_id_value = normalize_text(task_id)[:120]
        if not task_id_value:
            raise ValueError("task_id is required for a memory trace")
        candidates = [_normalize_trace_candidate(candidate) for candidate in candidate_memories[:100]]
        selected_ids = [
            str(candidate["memory_id"]) for candidate in candidates if bool(candidate.get("selected"))
        ]
        rejected_ids = [
            str(candidate["memory_id"]) for candidate in candidates if not bool(candidate.get("selected"))
        ]
        now = utc_now()
        trace_id = str(uuid.uuid4())
        goal_value = normalize_text(goal)[:1000] or "Unspecified task"
        context_value = normalize_text(context_summary)[:1000] or "No additional task context was provided."
        task_type_value = normalize_text(task_type)[:80] or "general"
        retrieval_context_value = _normalize_retrieval_context(
            {**dict(retrieval_context or {}), "task_type": task_type_value}
        )
        recall_mode_value = normalize_text(recall_mode)[:40] or "unknown"
        reason_value = normalize_text(retrieval_reason)[:600] or "No retrieval reason was recorded."
        query_values = [normalize_text(query)[:1000] for query in queries if normalize_text(query)][:8]
        payload = {
            "goal": goal_value,
            "context_summary": context_value,
            "retrieval_context": retrieval_context_value,
            "task_type": task_type_value,
            "recall_mode": recall_mode_value,
            "retrieval_used": bool(retrieval_used),
            "retrieval_reason": reason_value,
            "queries": query_values,
            "candidate_memories": candidates,
            "selected_memory_ids": selected_ids,
            "rejected_memory_ids": rejected_ids,
        }
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT trace_id,created_at FROM memory_traces WHERE task_id=?", (task_id_value,)
            ).fetchone()
            if existing:
                trace_id = str(existing["trace_id"])
                created_at = str(existing["created_at"])
            else:
                created_at = now
            conn.execute(
                """INSERT INTO memory_traces(
                     trace_id,task_id,session_id,goal,context_summary,retrieval_context_json,task_type,recall_mode,
                     retrieval_used,retrieval_reason,queries_json,candidate_memories_json,
                     selected_memory_ids_json,rejected_memory_ids_json,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(task_id) DO UPDATE SET
                     session_id=excluded.session_id,goal=excluded.goal,
                     context_summary=excluded.context_summary,
                     retrieval_context_json=excluded.retrieval_context_json,task_type=excluded.task_type,
                     recall_mode=excluded.recall_mode,retrieval_used=excluded.retrieval_used,
                     retrieval_reason=excluded.retrieval_reason,queries_json=excluded.queries_json,
                     candidate_memories_json=excluded.candidate_memories_json,
                     selected_memory_ids_json=excluded.selected_memory_ids_json,
                     rejected_memory_ids_json=excluded.rejected_memory_ids_json,
                     updated_at=excluded.updated_at""",
                (
                    trace_id,
                    task_id_value,
                    session_id,
                    goal_value,
                    context_value,
                    _trace_json(retrieval_context_value),
                    task_type_value,
                    recall_mode_value,
                    int(retrieval_used),
                    reason_value,
                    _trace_json(query_values),
                    _trace_json(candidates),
                    _trace_json(selected_ids),
                    _trace_json(rejected_ids),
                    created_at,
                    now,
                ),
            )
            self._append_memory_trace_event_tx(conn, task_id_value, "retrieval_decision", payload, now)
        return trace_id

    def memory_traces(self, *, limit: int = 100, task_id: str | None = None) -> list[dict[str, Any]]:
        """Return parsed task-level trace snapshots, newest first."""

        safe_limit = max(1, min(50000, int(limit)))
        with self._lock:
            if task_id:
                rows = self._conn.execute(
                    "SELECT * FROM memory_traces WHERE task_id=? ORDER BY created_at DESC LIMIT ?",
                    (task_id, safe_limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM memory_traces ORDER BY created_at DESC LIMIT ?", (safe_limit,)
                ).fetchall()
        parsed: list[dict[str, Any]] = []
        json_fields = {
            "queries_json": "queries",
            "candidate_memories_json": "candidate_memories",
            "selected_memory_ids_json": "selected_memory_ids",
            "rejected_memory_ids_json": "rejected_memory_ids",
            "influence_json": "influence",
            "evaluations_json": "evaluations",
            "memory_actions_json": "memory_actions",
        }
        for row in rows:
            item = dict(row)
            item["retrieval_used"] = bool(item.get("retrieval_used"))
            item["retrieval_context"] = _trace_json_object(
                item.pop("retrieval_context_json", "{}")
            )
            for source, target in json_fields.items():
                item[target] = _trace_json_array(item.pop(source, "[]"))
            parsed.append(item)
        return parsed

    def memory_trace_jsonl(self, *, limit: int = 1000, task_id: str | None = None) -> str:
        """Export the append-only decision ledger as one JSON object per line."""

        safe_limit = max(1, min(10000, int(limit)))
        with self._lock:
            if task_id:
                rows = self._conn.execute(
                    """SELECT event_id,task_id,event_type,payload_json,created_at
                       FROM memory_trace_events WHERE task_id=?
                       ORDER BY sequence_id DESC LIMIT ?""",
                    (task_id, safe_limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT event_id,task_id,event_type,payload_json,created_at
                       FROM memory_trace_events ORDER BY sequence_id DESC LIMIT ?""",
                    (safe_limit,),
                ).fetchall()
        lines: list[str] = []
        for row in reversed(rows):
            try:
                payload = json.loads(str(row["payload_json"]))
            except (json.JSONDecodeError, TypeError):
                payload = {"parse_error": "invalid stored trace payload"}
            lines.append(
                _trace_json(
                    {
                        "event_id": str(row["event_id"]),
                        "task_id": str(row["task_id"]),
                        "event_type": str(row["event_type"]),
                        "created_at": str(row["created_at"]),
                        "payload": payload,
                    }
                )
            )
        return "\n".join(lines)

    def memory_trace_summary(self, *, limit: int = 1000) -> dict[str, Any]:
        """Summarize observable trace quality without claiming task causality."""

        traces = self.memory_traces(limit=limit)
        selected = sum(len(trace["selected_memory_ids"]) for trace in traces)
        influenced = sum(
            int(bool(item.get("influenced")))
            for trace in traces
            for item in trace["influence"]
        )
        ratings: dict[str, int] = {}
        rejection_reasons: dict[str, int] = {}
        context_failures = 0
        for trace in traces:
            for evaluation in trace["evaluations"]:
                rating = str(evaluation.get("rating") or "Neutral")
                ratings[rating] = ratings.get(rating, 0) + 1
            for candidate in trace["candidate_memories"]:
                if candidate.get("selected"):
                    continue
                reason = str(candidate.get("reason") or "unspecified")
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                if float((candidate.get("components") or {}).get("context_gate", 1.0)) < 1.0:
                    context_failures += 1
        positive_ratings = ratings.get("Essential", 0) + ratings.get("Helpful", 0)
        negative_ratings = (
            ratings.get("Irrelevant", 0) + ratings.get("Misleading", 0) + ratings.get("Harmful", 0)
        )
        resolved_ratings = positive_ratings + negative_ratings
        evaluated_ratings = sum(ratings.values())
        candidate_count = sum(len(trace["candidate_memories"]) for trace in traces)
        return {
            "tasks": len(traces),
            "retrieval_used_tasks": sum(int(trace["retrieval_used"]) for trace in traces),
            "abstained_tasks": sum(int(not trace["retrieval_used"]) for trace in traces),
            "selected_memories": selected,
            "influenced_memories": influenced,
            "observed_selection_precision": round(influenced / selected, 6) if selected else None,
            "resolved_memory_usefulness": (
                round(positive_ratings / resolved_ratings, 6) if resolved_ratings else None
            ),
            "false_positive_memories": negative_ratings,
            "false_positive_rate": (
                round(negative_ratings / evaluated_ratings, 6) if evaluated_ratings else None
            ),
            "context_failures": context_failures,
            "context_failure_rate": (
                round(context_failures / candidate_count, 6) if candidate_count else None
            ),
            "ratings": ratings,
            "rejection_reasons": rejection_reasons,
            "claim_boundary": (
                "Selection precision means attributed answer use among selected memories; "
                "it does not by itself prove answer correctness or task causality."
            ),
        }

    def scoring_health(self, *, weeks: int = 12, limit: int = 10000) -> dict[str, Any]:
        """Return resolved weekly retrieval-quality trends and signal evidence.

        Missing attribution is kept separate from false positives. That avoids
        making an unresolved task look like evidence that a selected memory was
        irrelevant.
        """

        from .retrieval import (
            LIVE_SCORING_POLICY_VERSION,
            SCORE_SIGNAL_WEIGHTS,
            SHADOW_SCORING_POLICY_VERSION,
            SHADOW_SCORE_SIGNAL_WEIGHTS,
        )

        safe_weeks = max(1, min(104, int(weeks)))
        traces = self.memory_traces(limit=max(1, min(50000, int(limit))))
        positive_ratings = {"Essential", "Helpful"}
        negative_ratings = {"Irrelevant", "Misleading", "Harmful"}
        weekly: dict[str, dict[str, Any]] = {}
        signal_totals = {
            signal: {"used": 0.0, "unused": 0.0, "used_samples": 0, "unused_samples": 0}
            for signal in SCORE_SIGNAL_WEIGHTS
        }
        policy_versions: dict[str, int] = {}
        observed_days: set[str] = set()
        shadow_deltas = {"used": [], "unused": [], "all": []}

        def new_bucket(label: str) -> dict[str, Any]:
            return {
                "week": label,
                "tasks": 0,
                "injections": 0,
                "resolved_injections": 0,
                "pending_injections": 0,
                "used_injections": 0,
                "resolved_tokens": 0,
                "wasted_tokens": 0,
                "evaluated_ratings": 0,
                "helpful_ratings": 0,
                "false_positive_ratings": 0,
            }

        for trace in traces:
            try:
                created_at = datetime.fromisoformat(str(trace.get("created_at") or ""))
            except ValueError:
                continue
            iso_year, iso_week, _weekday = created_at.isocalendar()
            week_label = f"{iso_year}-W{iso_week:02d}"
            bucket = weekly.setdefault(week_label, new_bucket(week_label))
            bucket["tasks"] += 1
            observed_days.add(created_at.date().isoformat())
            influence = {
                str(item.get("memory_id") or ""): bool(item.get("influenced"))
                for item in trace.get("influence", [])
                if item.get("memory_id")
            }
            evaluations = {
                str(item.get("memory_id") or ""): str(item.get("rating") or "Neutral")
                for item in trace.get("evaluations", [])
                if item.get("memory_id")
            }
            for candidate in trace.get("candidate_memories", []):
                if not bool(candidate.get("selected")):
                    continue
                memory_id = str(candidate.get("memory_id") or "")
                bucket["injections"] += 1
                policy_version = str(
                    candidate.get("scoring_policy_version") or "legacy_unversioned"
                )
                policy_versions[policy_version] = policy_versions.get(policy_version, 0) + 1
                if memory_id not in influence:
                    bucket["pending_injections"] += 1
                    continue
                used = bool(influence[memory_id])
                tokens = max(0, int(candidate.get("estimated_tokens") or 0))
                bucket["resolved_injections"] += 1
                bucket["resolved_tokens"] += tokens
                if used:
                    bucket["used_injections"] += 1
                else:
                    bucket["wasted_tokens"] += tokens
                rating = evaluations.get(memory_id)
                if rating is not None:
                    bucket["evaluated_ratings"] += 1
                    bucket["helpful_ratings"] += int(rating in positive_ratings)
                    bucket["false_positive_ratings"] += int(rating in negative_ratings)
                components = (
                    candidate.get("components")
                    if isinstance(candidate.get("components"), dict)
                    else {}
                )
                for signal, weight in SCORE_SIGNAL_WEIGHTS.items():
                    try:
                        observed_weight = float(
                            components.get(f"live_weight_{signal}", weight)
                        )
                        contribution = max(
                            0.0,
                            observed_weight * float(components.get(signal) or 0.0),
                        )
                    except (TypeError, ValueError):
                        contribution = 0.0
                    target = "used" if used else "unused"
                    signal_totals[signal][target] += contribution
                    signal_totals[signal][f"{target}_samples"] += 1
                try:
                    shadow_delta = float(components["shadow_score_delta"])
                except (KeyError, TypeError, ValueError):
                    continue
                shadow_deltas["all"].append(shadow_delta)
                shadow_deltas["used" if used else "unused"].append(shadow_delta)

        def finalize(bucket: dict[str, Any]) -> dict[str, Any]:
            resolved = int(bucket["resolved_injections"])
            evaluated = int(bucket["evaluated_ratings"])
            resolved_tokens = int(bucket["resolved_tokens"])
            return {
                **bucket,
                "precision": (
                    round(int(bucket["used_injections"]) / resolved, 6) if resolved else None
                ),
                "helpfulness": (
                    round(int(bucket["helpful_ratings"]) / evaluated, 6) if evaluated else None
                ),
                "false_positive_rate": (
                    round(int(bucket["false_positive_ratings"]) / evaluated, 6)
                    if evaluated
                    else None
                ),
                "waste_rate": (
                    round(int(bucket["wasted_tokens"]) / resolved_tokens, 6)
                    if resolved_tokens
                    else None
                ),
            }

        weekly_rows = [finalize(weekly[key]) for key in sorted(weekly)[-safe_weeks:]]
        overall = new_bucket("all")
        for row in weekly_rows:
            for field in (
                "tasks",
                "injections",
                "resolved_injections",
                "pending_injections",
                "used_injections",
                "resolved_tokens",
                "wasted_tokens",
                "evaluated_ratings",
                "helpful_ratings",
                "false_positive_ratings",
            ):
                overall[field] += int(row[field])
        current = finalize(overall)
        used_total = sum(float(item["used"]) for item in signal_totals.values())
        unused_total = sum(float(item["unused"]) for item in signal_totals.values())
        signals = []
        for signal, values in signal_totals.items():
            signals.append(
                {
                    "signal": signal,
                    "weight": SCORE_SIGNAL_WEIGHTS[signal],
                    "shadow_weight": SHADOW_SCORE_SIGNAL_WEIGHTS[signal],
                    "used_contribution_share": (
                        round(float(values["used"]) / used_total, 6) if used_total else None
                    ),
                    "unused_contribution_share": (
                        round(float(values["unused"]) / unused_total, 6)
                        if unused_total
                        else None
                    ),
                    "used_samples": int(values["used_samples"]),
                    "unused_samples": int(values["unused_samples"]),
                }
            )
        signals.sort(
            key=lambda item: float(item.get("unused_contribution_share") or 0.0),
            reverse=True,
        )
        top_false = next(
            (item for item in signals if item["unused_samples"] > 0),
            None,
        )
        best_true = max(
            (item for item in signals if item["used_samples"] > 0),
            key=lambda item: float(item.get("used_contribution_share") or 0.0),
            default=None,
        )

        def mean(values: list[float]) -> float | None:
            return round(sum(values) / len(values), 6) if values else None

        return {
            "generated_at": utc_now(),
            "current": current,
            "weekly": weekly_rows,
            "signals": signals,
            "top_false_signal": top_false,
            "best_true_signal": best_true,
            "policy": {
                "live_version": LIVE_SCORING_POLICY_VERSION,
                "shadow_version": SHADOW_SCORING_POLICY_VERSION,
                "live_weights": dict(SCORE_SIGNAL_WEIGHTS),
                "shadow_weights": dict(SHADOW_SCORE_SIGNAL_WEIGHTS),
                "observed_policy_versions": policy_versions,
                "mode": "shadow",
            },
            "shadow": {
                "resolved_observations": len(shadow_deltas["all"]),
                "observed_days": len(observed_days),
                "mean_score_delta": mean(shadow_deltas["all"]),
                "mean_used_score_delta": mean(shadow_deltas["used"]),
                "mean_unused_score_delta": mean(shadow_deltas["unused"]),
                "ready_for_selection_experiment": (
                    len(observed_days) >= 7 and len(shadow_deltas["all"]) >= 50
                ),
            },
            "definitions": {
                "precision": "Attributed answer use divided by resolved selected memories.",
                "helpfulness": "Helpful or Essential ratings divided by all evaluated selected memories.",
                "false_positive_rate": "Irrelevant, Misleading, or Harmful ratings divided by all evaluated selected memories.",
                "waste_rate": "Estimated tokens from resolved selected memories with no answer-use attribution divided by all resolved selected-memory tokens.",
                "signal_breakdown": "Share of weighted component contribution among used or unused resolved selections; observational, not causal.",
            },
            "claim_boundary": (
                "The shadow policy currently records score deltas only. It cannot claim better selection "
                "precision until a gated selection experiment exposes shadow-ranked memories to real tasks."
            ),
        }

    def memory_write_decisions(self, *, limit: int = 100) -> list[dict[str, Any]]:
        safe_limit = max(1, min(1000, int(limit)))
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memory_write_decisions ORDER BY created_at DESC,rowid DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["independently_understandable"] = bool(item["independently_understandable"])
            item["contradiction_ids"] = _trace_json_array(item.pop("contradiction_ids_json", "[]"))
            item["quality_flags"] = _trace_json_array(item.pop("quality_flags_json", "[]"))
            result.append(item)
        return result

    def memory_write_summary(self, *, limit: int = 1000) -> dict[str, Any]:
        rows = self.memory_write_decisions(limit=limit)
        decisions: dict[str, int] = {}
        durability: dict[str, int] = {}
        quality_flags: dict[str, int] = {}
        for row in rows:
            decision = str(row.get("decision") or "unknown")
            decisions[decision] = decisions.get(decision, 0) + 1
            durability_key = str(row.get("durability") or "unknown")
            durability[durability_key] = durability.get(durability_key, 0) + 1
            for flag in row.get("quality_flags") or []:
                flag_key = str(flag)
                quality_flags[flag_key] = quality_flags.get(flag_key, 0) + 1
        return {
            "candidates": len(rows),
            "decisions": decisions,
            "durability": durability,
            "quality_flags": quality_flags,
            "duplicate_updates": sum(int(bool(row.get("duplicate_memory_id"))) for row in rows),
            "contradiction_candidates": sum(int(bool(row.get("contradiction_ids"))) for row in rows),
            "contextless_candidates": sum(
                int(not bool(row.get("independently_understandable"))) for row in rows
            ),
            "average_reusable_score": round(
                sum(float(row.get("reusable_score") or 0.0) for row in rows) / len(rows), 6
            )
            if rows
            else None,
        }

    def context_feedback(
        self,
        memory_id: str,
        retrieval_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Return outcome-backed usefulness for one stable retrieval context."""

        context_key, context_value = _retrieval_context_key(retrieval_context)
        with self._lock:
            row = self._conn.execute(
                """SELECT COUNT(*) observations,
                          SUM(selected) selected_count,
                          SUM(used) used_count,
                          SUM(CASE WHEN outcome IN ('helpful','validated') THEN 1 ELSE 0 END) positive_count,
                          SUM(CASE WHEN outcome IN ('harmful','corrected') THEN 1 ELSE 0 END) negative_count,
                          SUM(CASE WHEN outcome='ignored' THEN 1 ELSE 0 END) irrelevant_count
                   FROM memory_context_outcomes WHERE memory_id=? AND context_key=?""",
                (memory_id, context_key),
            ).fetchone()
        observations = int(row["observations"] or 0)
        selected = int(row["selected_count"] or 0)
        used = int(row["used_count"] or 0)
        positive = int(row["positive_count"] or 0)
        negative = int(row["negative_count"] or 0)
        irrelevant = int(row["irrelevant_count"] or 0)
        if not observations:
            usefulness = 0.5
        else:
            # Positive task outcomes earn credit; repeatedly selected-but-unused
            # memories lose credit. Bayesian smoothing avoids overreacting once.
            usefulness = _clamp(
                (positive + 1.5) / (selected + 3.0)
                + 0.15 * used / max(1, selected)
                - 0.45 * negative / max(2, selected)
                - 0.25 * irrelevant / max(2, selected)
            )
        return {
            "context_key": context_key,
            "context": context_value,
            "observations": observations,
            "selected_count": selected,
            "used_count": used,
            "positive_count": positive,
            "negative_count": negative,
            "irrelevant_count": irrelevant,
            "usefulness": round(usefulness, 6),
        }

    def context_feedback_summary(self, *, limit: int = 1000) -> dict[str, Any]:
        safe_limit = max(1, min(10000, int(limit)))
        with self._lock:
            rows = self._conn.execute(
                """SELECT memory_id,context_key,context_json,COUNT(*) observations,
                          SUM(selected) selected_count,SUM(used) used_count,
                          SUM(CASE WHEN outcome IN ('helpful','validated') THEN 1 ELSE 0 END) positive_count,
                          SUM(CASE WHEN outcome IN ('harmful','corrected') THEN 1 ELSE 0 END) negative_count,
                          SUM(CASE WHEN outcome='ignored' THEN 1 ELSE 0 END) irrelevant_count,
                          MAX(updated_at) updated_at
                   FROM memory_context_outcomes
                   GROUP BY memory_id,context_key
                   ORDER BY MAX(updated_at) DESC LIMIT ?""",
                (safe_limit,),
            ).fetchall()
        buckets: list[dict[str, Any]] = []
        for row in rows:
            selected = int(row["selected_count"] or 0)
            positive = int(row["positive_count"] or 0)
            negative = int(row["negative_count"] or 0)
            irrelevant = int(row["irrelevant_count"] or 0)
            usefulness = _clamp(
                (positive + 1.5) / (selected + 3.0)
                + 0.15 * int(row["used_count"] or 0) / max(1, selected)
                - 0.45 * negative / max(2, selected)
                - 0.25 * irrelevant / max(2, selected)
            )
            buckets.append(
                {
                    "memory_id": str(row["memory_id"]),
                    "context_key": str(row["context_key"]),
                    "context": _trace_json_object(row["context_json"]),
                    "observations": int(row["observations"] or 0),
                    "selected_count": selected,
                    "used_count": int(row["used_count"] or 0),
                    "positive_count": positive,
                    "negative_count": negative,
                    "irrelevant_count": irrelevant,
                    "usefulness": round(usefulness, 6),
                    "updated_at": str(row["updated_at"]),
                }
            )
        return {
            "buckets": len(buckets),
            "positive_buckets": sum(int(item["usefulness"] > 0.6) for item in buckets),
            "downweighted_buckets": sum(int(item["usefulness"] < 0.4) for item in buckets),
            "items": buckets,
            "claim_boundary": (
                "Context usefulness reflects observed selection, answer use, and labeled outcomes; "
                "it is not proof that a memory caused the task result."
            ),
        }

    @staticmethod
    def _upsert_context_outcomes_for_task_tx(
        conn: sqlite3.Connection,
        task_id: str,
        candidates: Sequence[dict[str, Any]],
        attribution_by_id: dict[str, float],
        retrieval_context: dict[str, Any],
        now: str,
    ) -> None:
        context_key, context_value = _retrieval_context_key(retrieval_context)
        for candidate in candidates:
            if not bool(candidate.get("selected")):
                continue
            memory_id = str(candidate.get("memory_id") or "")
            if not memory_id:
                continue
            used = _clamp(float(attribution_by_id.get(memory_id, 0.0))) > 0.0
            conn.execute(
                """INSERT INTO memory_context_outcomes(
                     task_id,memory_id,context_key,context_json,selected,used,outcome,updated_at
                   ) VALUES(?,?,?,?,1,?,?,?)
                   ON CONFLICT(task_id,memory_id) DO UPDATE SET
                     context_key=excluded.context_key,context_json=excluded.context_json,
                     selected=1,used=excluded.used,outcome=excluded.outcome,updated_at=excluded.updated_at""",
                (
                    task_id,
                    memory_id,
                    context_key,
                    _trace_json(context_value),
                    int(used),
                    "used" if used else "ignored",
                    now,
                ),
            )

    @staticmethod
    def _append_memory_trace_event_tx(
        conn: sqlite3.Connection,
        task_id: str,
        event_type: str,
        payload: dict[str, Any],
        created_at: str,
    ) -> None:
        conn.execute(
            """INSERT INTO memory_trace_events(event_id,task_id,event_type,payload_json,created_at)
               VALUES(?,?,?,?,?)""",
            (str(uuid.uuid4()), task_id, event_type, _trace_json(payload), created_at),
        )

    def _resolve_memory_trace_usage_tx(
        self,
        conn: sqlite3.Connection,
        task_id: str,
        attribution_by_id: dict[str, float],
        memory_actions: Sequence[dict[str, Any]],
        now: str,
    ) -> None:
        row = conn.execute("SELECT * FROM memory_traces WHERE task_id=?", (task_id,)).fetchone()
        if not row:
            return
        candidates = _trace_json_list(row["candidate_memories_json"])
        retrieval_context = _trace_json_object(row["retrieval_context_json"])
        self._upsert_context_outcomes_for_task_tx(
            conn,
            task_id,
            candidates,
            attribution_by_id,
            retrieval_context,
            now,
        )
        influence: list[dict[str, Any]] = []
        evaluations: list[dict[str, Any]] = []
        for candidate in candidates:
            if not bool(candidate.get("selected")):
                continue
            memory_id = str(candidate.get("memory_id") or "")
            attribution = _clamp(float(attribution_by_id.get(memory_id, 0.0)))
            influenced = attribution > 0.0
            influence.append(
                {
                    "memory_id": memory_id,
                    "influenced": influenced,
                    "attribution_score": round(attribution, 6),
                    "reason": (
                        "distinctive answer evidence matched this memory"
                        if influenced
                        else "retrieved but no answer-use evidence was observed"
                    ),
                }
            )
            evaluations.append(
                {
                    "memory_id": memory_id,
                    "rating": "Neutral" if influenced else "Irrelevant",
                    "improved_outcome": False,
                    "prevented_error": False,
                    "introduced_incorrect_assumption": False,
                    "duplicated_another_memory": False,
                    "retrieved_but_never_used": not influenced,
                    "reason": (
                        "The memory influenced the answer, but no explicit outcome is available yet."
                        if influenced
                        else "The memory was selected but not used in the answer."
                    ),
                }
            )
        actions = [_normalize_trace_action(action) for action in memory_actions[:100]]
        if not actions:
            actions = [
                {
                    "action": "ignored",
                    "memory_id": None,
                    "reason": "no durable memory candidate met storage criteria",
                }
            ]
        payload = {
            "influence": influence,
            "evaluations": evaluations,
            "memory_actions": actions,
            "outcome": "completed_unlabeled",
        }
        conn.execute(
            """UPDATE memory_traces SET influence_json=?,evaluations_json=?,memory_actions_json=?,
                 outcome='completed_unlabeled',updated_at=?,completed_at=? WHERE task_id=?""",
            (
                _trace_json(influence),
                _trace_json(evaluations),
                _trace_json(actions),
                now,
                now,
                task_id,
            ),
        )
        self._append_memory_trace_event_tx(conn, task_id, "task_evaluation", payload, now)

    def _update_memory_trace_outcome_tx(
        self,
        conn: sqlite3.Connection,
        task_id: str,
        outcome: str,
        now: str,
        *,
        source: str,
    ) -> None:
        row = conn.execute(
            "SELECT evaluations_json,influence_json FROM memory_traces WHERE task_id=?", (task_id,)
        ).fetchone()
        if not row:
            return
        influence_by_id = {
            str(item.get("memory_id") or ""): bool(item.get("influenced"))
            for item in _trace_json_list(row["influence_json"])
        }
        rating_by_outcome = {
            "helpful": "Helpful",
            "validated": "Essential",
            "corrected": "Misleading",
            "harmful": "Harmful",
        }
        updated: list[dict[str, Any]] = []
        for evaluation in _trace_json_list(row["evaluations_json"]):
            item = dict(evaluation)
            memory_id = str(item.get("memory_id") or "")
            if not influence_by_id.get(memory_id, False):
                updated.append(item)
                continue
            item.update(
                {
                    "rating": rating_by_outcome[outcome],
                    "improved_outcome": outcome in {"helpful", "validated"},
                    "prevented_error": outcome == "validated",
                    "introduced_incorrect_assumption": outcome in {"harmful", "corrected"},
                    "reason": {
                        "helpful": "Explicit feedback says the used memory improved the task outcome.",
                        "validated": "Independent outcome evidence validated the used memory.",
                        "corrected": "The task exposed this used memory as stale or incorrect.",
                        "harmful": "Explicit feedback says this used memory harmed the task outcome.",
                    }[outcome],
                }
            )
            updated.append(item)
        conn.execute(
            """UPDATE memory_traces SET evaluations_json=?,outcome=?,updated_at=?,completed_at=COALESCE(completed_at,?)
               WHERE task_id=?""",
            (_trace_json(updated), outcome, now, now, task_id),
        )
        conn.execute(
            """UPDATE memory_context_outcomes SET outcome=?,updated_at=?
               WHERE task_id=? AND used=1""",
            (outcome, now, task_id),
        )
        self._append_memory_trace_event_tx(
            conn,
            task_id,
            "outcome_feedback",
            {"outcome": outcome, "source": source, "evaluations": updated},
            now,
        )

    def _restore_memory_trace_outcome_tx(
        self,
        conn: sqlite3.Connection,
        task_id: str,
        prior_outcome: str,
        now: str,
    ) -> None:
        if prior_outcome in {"helpful", "harmful", "validated", "corrected"}:
            self._update_memory_trace_outcome_tx(
                conn,
                task_id,
                prior_outcome,
                now,
                source="dashboard_undo_restored_prior",
            )
            return
        row = conn.execute(
            "SELECT evaluations_json,influence_json FROM memory_traces WHERE task_id=?", (task_id,)
        ).fetchone()
        if not row:
            return
        influence_by_id = {
            str(item.get("memory_id") or ""): bool(item.get("influenced"))
            for item in _trace_json_list(row["influence_json"])
        }
        restored: list[dict[str, Any]] = []
        for evaluation in _trace_json_list(row["evaluations_json"]):
            item = dict(evaluation)
            influenced = influence_by_id.get(str(item.get("memory_id") or ""), False)
            item.update(
                {
                    "rating": "Neutral" if influenced else "Irrelevant",
                    "improved_outcome": False,
                    "prevented_error": False,
                    "introduced_incorrect_assumption": False,
                    "reason": (
                        "The memory influenced the answer, but the explicit outcome label was undone."
                        if influenced
                        else "The memory was selected but not used in the answer."
                    ),
                }
            )
            restored.append(item)
        restored_outcome = "completed_unlabeled" if prior_outcome in {"used", "ignored", "pending"} else prior_outcome
        conn.execute(
            "UPDATE memory_traces SET evaluations_json=?,outcome=?,updated_at=? WHERE task_id=?",
            (_trace_json(restored), restored_outcome, now, task_id),
        )
        conn.execute(
            """UPDATE memory_context_outcomes SET outcome=?,updated_at=?
               WHERE task_id=? AND used=1""",
            (
                prior_outcome if prior_outcome in {"helpful", "harmful", "validated", "corrected"} else "used",
                now,
                task_id,
            ),
        )
        self._append_memory_trace_event_tx(
            conn,
            task_id,
            "outcome_feedback_undone",
            {"restored_outcome": restored_outcome, "evaluations": restored},
            now,
        )

    def record_pruning_regret(
        self,
        memory_id: str,
        *,
        query: str,
        score: float,
        restore: bool = False,
    ) -> bool:
        memory = self.get_memory(memory_id)
        if not memory or (
            memory["state"] != "archived" and not bool(memory.get("stranded"))
        ):
            return False
        strand_decision_id: str | None = None
        if bool(memory.get("stranded")):
            with self._lock:
                strand = self._conn.execute(
                    """SELECT decision_id FROM memory_strands
                       WHERE memory_id=? AND status='active'
                       ORDER BY created_at DESC LIMIT 1""",
                    (memory_id,),
                ).fetchone()
            strand_decision_id = str(strand["decision_id"]) if strand else None
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO pruning_regret(memory_id,query,score,restored,created_at)
                   VALUES(?,?,?,?,?)""",
                (memory_id, normalize_text(query)[:500], _clamp(score), int(restore), utc_now()),
            )
        if restore:
            if strand_decision_id:
                return bool(
                    self.undo_adaptive_pruning(
                        strand_decision_id,
                        via_regret=True,
                    ).get("restored")
                )
            return self.set_state(memory_id, "active", reason="automatic pruning-regret restoration")
        return True

    def feedback(self, memory_ids: Sequence[str], outcome: str, *, session_id: str | None = None) -> int:
        event = {
            "useful": "used",
            "helpful": "helpful",
            "successful": "successful",
            "confirmed": "confirmed",
            "irrelevant": "irrelevant",
            "wrong": "wrong",
            "outdated": "wrong",
        }.get(outcome)
        if not event:
            raise ValueError(
                "outcome must be useful, helpful, successful, confirmed, irrelevant, wrong, or outdated"
            )
        count = 0
        for memory_id in dict.fromkeys(memory_ids):
            if self.get_memory(memory_id):
                self.log_access(memory_id, event, session_id=session_id)
                count += 1
        return count

    def trace_memory_feedback(self, task_id: str) -> dict[str, dict[str, Any]]:
        """Return the latest audited per-memory label for one recall trace."""

        task_id_value = normalize_text(task_id)[:120]
        if not re.fullmatch(r"[0-9A-Za-z_-]{8,120}", task_id_value):
            return {}
        prefix = f"trace:{task_id_value}:memory:"
        escaped_prefix = (
            prefix.replace("!", "!!").replace("%", "!%").replace("_", "!_")
        )
        with self._lock:
            rows = self._conn.execute(
                """SELECT review_id,item_key,action,reason_code,actor,created_at
                   FROM operator_review_decisions
                   WHERE item_type='memory_feedback' AND reversed_at IS NULL
                     AND item_key LIKE ? ESCAPE '!'
                   ORDER BY created_at,review_id""",
                (f"{escaped_prefix}%",),
            ).fetchall()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            item_key = str(row["item_key"])
            memory_id = item_key[len(prefix) :]
            if memory_id:
                result[memory_id] = {
                    "review_id": str(row["review_id"]),
                    "label": str(row["action"]),
                    "reason_code": str(row["reason_code"]),
                    "actor": str(row["actor"]),
                    "created_at": str(row["created_at"]),
                }
        return result

    def set_trace_memory_feedback(
        self,
        task_id: str,
        memory_id: str,
        label: str | None,
        *,
        actor: str = "dashboard-operator",
    ) -> dict[str, Any]:
        """Set, replace, or clear one audited trace-memory feedback label."""

        task_id_value = normalize_text(task_id)[:120]
        memory_id_value = normalize_text(memory_id)
        label_value = normalize_text(label or "").casefold()
        if not re.fullmatch(r"[0-9A-Za-z_-]{8,120}", task_id_value):
            raise ValueError("a valid task_id is required")
        if label_value not in {"", "helpful", "irrelevant", "wrong", "outdated"}:
            raise ValueError("label must be helpful, irrelevant, wrong, outdated, or clear")

        item_key = f"trace:{task_id_value}:memory:{memory_id_value}"
        session_id = f"dashboard-trace:{task_id_value}"
        now = utc_now()
        new_review_id = str(uuid.uuid4())
        event_by_label = {
            "helpful": "helpful",
            "irrelevant": "irrelevant",
            "wrong": "wrong",
            "outdated": "wrong",
        }
        replaced_label: str | None = None

        with self.transaction() as conn:
            memory = conn.execute(
                """SELECT id,helpful_count,harmful_count,false_positive_count,
                          last_helpful_at,last_retrieved_at
                   FROM memories WHERE id=?""",
                (memory_id_value,),
            ).fetchone()
            if not memory:
                raise ValueError("memory does not exist")
            existing = conn.execute(
                """SELECT * FROM operator_review_decisions
                   WHERE item_type='memory_feedback' AND item_key=? AND reversed_at IS NULL
                   ORDER BY created_at DESC,review_id DESC LIMIT 1""",
                (item_key,),
            ).fetchone()
            if existing and str(existing["action"]) == label_value:
                return {
                    "changed": False,
                    "review_id": str(existing["review_id"]),
                    "task_id": task_id_value,
                    "memory_id": memory_id_value,
                    "label": label_value,
                    "replaced_label": None,
                }

            if existing:
                replaced_label = str(existing["action"])
                try:
                    existing_prior = json.loads(str(existing["prior_json"] or "{}"))
                    existing_effect = json.loads(str(existing["effect_json"] or "{}"))
                except json.JSONDecodeError:
                    existing_prior = {}
                    existing_effect = {}
                old_event = event_by_label.get(replaced_label)
                if old_event == "helpful":
                    conn.execute(
                        """UPDATE memories
                           SET helpful_count=MAX(0,helpful_count-1),
                               last_helpful_at=CASE
                                 WHEN last_helpful_at=? THEN ? ELSE last_helpful_at END
                           WHERE id=?""",
                        (
                            existing_effect.get("feedback_at"),
                            existing_prior.get("last_helpful_at"),
                            memory_id_value,
                        ),
                    )
                    for prior_event in existing_effect.get("reconsolidation_prior", []):
                        if not isinstance(prior_event, dict):
                            continue
                        conn.execute(
                            """UPDATE reconsolidation_events
                               SET status=?,first_reused_at=?
                               WHERE event_id=? AND status='reexposed' AND first_reused_at=?""",
                            (
                                prior_event.get("status") or "pending",
                                prior_event.get("first_reused_at"),
                                prior_event.get("event_id"),
                                existing_effect.get("feedback_at"),
                            ),
                        )
                elif old_event == "irrelevant":
                    conn.execute(
                        """UPDATE memories
                           SET false_positive_count=MAX(0,false_positive_count-1),
                               last_retrieved_at=CASE
                                 WHEN last_retrieved_at=? THEN ? ELSE last_retrieved_at END
                           WHERE id=?""",
                        (
                            existing_effect.get("feedback_at"),
                            existing_prior.get("last_retrieved_at"),
                            memory_id_value,
                        ),
                    )
                elif old_event == "wrong":
                    conn.execute(
                        """UPDATE memories
                           SET harmful_count=MAX(0,harmful_count-1),
                               false_positive_count=MAX(0,false_positive_count-1),
                               last_retrieved_at=CASE
                                 WHEN last_retrieved_at=? THEN ? ELSE last_retrieved_at END
                           WHERE id=?""",
                        (
                            existing_effect.get("feedback_at"),
                            existing_prior.get("last_retrieved_at"),
                            memory_id_value,
                        ),
                    )
                conn.execute(
                    "UPDATE operator_review_decisions SET reversed_at=? WHERE review_id=?",
                    (now, existing["review_id"]),
                )
                conn.execute(
                    """INSERT INTO access_log(memory_id,event,query,session_id,score,created_at)
                       VALUES(?,'feedback_reversed',?,?,NULL,?)""",
                    (memory_id_value, replaced_label, session_id, now),
                )

            if not label_value:
                return {
                    "changed": bool(existing),
                    "review_id": None,
                    "task_id": task_id_value,
                    "memory_id": memory_id_value,
                    "label": None,
                    "replaced_label": replaced_label,
                }

            current = conn.execute(
                """SELECT helpful_count,harmful_count,false_positive_count,
                          last_helpful_at,last_retrieved_at
                   FROM memories WHERE id=?""",
                (memory_id_value,),
            ).fetchone()
            prior = dict(current)
            event = event_by_label[label_value]
            cursor = conn.execute(
                """INSERT INTO access_log(memory_id,event,query,session_id,score,created_at)
                   VALUES(?,?,?,?,NULL,?)""",
                (memory_id_value, event, new_review_id, session_id, now),
            )
            reconsolidation_prior: list[dict[str, Any]] = []
            if event == "helpful":
                reconsolidation_prior = [
                    dict(row)
                    for row in conn.execute(
                        """SELECT event_id,status,first_reused_at
                           FROM reconsolidation_events
                           WHERE memory_id=? AND corrected_at<=? AND status='pending'""",
                        (memory_id_value, now),
                    ).fetchall()
                ]
                conn.execute(
                    """UPDATE memories SET helpful_count=helpful_count+1,last_helpful_at=?
                       WHERE id=?""",
                    (now, memory_id_value),
                )
                from .research import record_reconsolidation_reuse_tx

                record_reconsolidation_reuse_tx(conn, memory_id_value, now)
            elif event == "irrelevant":
                conn.execute(
                    """UPDATE memories
                       SET false_positive_count=false_positive_count+1,last_retrieved_at=?
                       WHERE id=?""",
                    (now, memory_id_value),
                )
            else:
                conn.execute(
                    """UPDATE memories
                       SET harmful_count=harmful_count+1,
                           false_positive_count=false_positive_count+1,last_retrieved_at=?
                       WHERE id=?""",
                    (now, memory_id_value),
                )

            effect = {
                "task_id": task_id_value,
                "memory_id": memory_id_value,
                "label": label_value,
                "event": event,
                "feedback_at": now,
                "access_log_id": int(cursor.lastrowid),
                "reconsolidation_prior": reconsolidation_prior,
            }
            signal = {
                "item_type": "memory_feedback",
                "action": label_value,
                "reason_code": f"trace_{label_value}",
                "decision_scope": "item_only",
            }
            conn.execute(
                """INSERT INTO operator_review_decisions(
                   review_id,item_type,item_key,proposal_id,src_id,dst_id,action,reason_code,
                   reason_text,prior_json,effect_json,learning_signal_json,decision_scope,actor,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    new_review_id,
                    "memory_feedback",
                    item_key,
                    None,
                    None,
                    None,
                    label_value,
                    f"trace_{label_value}",
                    None,
                    json.dumps(prior, sort_keys=True),
                    json.dumps(effect, sort_keys=True),
                    json.dumps(signal, sort_keys=True),
                    "item_only",
                    normalize_text(actor)[:80] or "dashboard-operator",
                    now,
                ),
            )

        return {
            "changed": True,
            "review_id": new_review_id,
            "task_id": task_id_value,
            "memory_id": memory_id_value,
            "label": label_value,
            "replaced_label": replaced_label,
        }

    def create_usage_batch(
        self,
        items: Sequence[tuple[str, float]],
        *,
        query: str,
        session_id: str | None,
        task_type: str | None = None,
        recall_mode: str | None = None,
        requested_budget: int = 0,
        estimated_tokens: int = 0,
        metacognitive_assessments: Sequence[dict[str, Any]] = (),
        metacognition_mode: str = "shadow",
        task_id: str | None = None,
    ) -> str:
        task_id = task_id or str(uuid.uuid4())
        now = utc_now()
        task_type_value = normalize_text(task_type or "general")[:80] or "general"
        recall_mode_value = normalize_text(recall_mode or "unknown")[:40] or "unknown"
        monitor_mode = "enforce" if str(metacognition_mode).casefold() == "enforce" else "shadow"
        with self.transaction() as conn:
            for memory_id, score in items:
                conn.execute(
                    """INSERT INTO usage_records(
                       usage_id,task_id,memory_id,session_id,query,score,selected,created_at
                       ) VALUES(?,?,?,?,?,?,1,?)""",
                    (str(uuid.uuid4()), task_id, memory_id, session_id, query, score, now),
                )
                conn.execute(
                    """INSERT OR IGNORE INTO memory_experience_events(
                         event_id,event_type,memory_id,task_id,session_id,evidence_key,event_day,
                         metadata_json,created_at
                       ) VALUES(?,'memory_selected',?,?,?,?,?,?,?)""",
                    (
                        str(uuid.uuid4()),
                        memory_id,
                        task_id,
                        session_id,
                        f"selected:{task_id}:{memory_id}",
                        now[:10],
                        _trace_json(
                            {
                                "score": round(float(score), 6),
                                "task_type": task_type_value,
                                "recall_mode": recall_mode_value,
                            }
                        ),
                        now,
                    ),
                )
            if items and task_type and recall_mode:
                conn.execute(
                    """INSERT INTO recall_budget_observations(
                       task_id,task_type,mode,requested_budget,estimated_tokens,
                       selected_count,created_at
                       ) VALUES(?,?,?,?,?,?,?)""",
                    (
                        task_id,
                        task_type_value,
                        recall_mode_value,
                        max(1, int(requested_budget)),
                        max(0, int(estimated_tokens)),
                        len(items),
                        now,
                    ),
                )
            for assessment in metacognitive_assessments:
                memory_id = str(assessment.get("memory_id") or "")
                if not memory_id:
                    continue
                decision = str(assessment.get("decision") or "verify").casefold()
                if decision not in {"use", "verify", "abstain"}:
                    decision = "verify"
                applied = bool(assessment.get("applied"))
                initial_outcome = "withheld" if applied and decision == "abstain" else "pending"
                conn.execute(
                    """INSERT INTO metacognitive_predictions(
                       prediction_id,task_id,memory_id,task_type,recall_mode,monitor_mode,
                       source_category,raw_probability,calibrated_probability,decision,
                       applied,reason,calibration_scope,calibration_samples,features_json,
                       outcome,created_at,resolved_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        str(uuid.uuid4()),
                        task_id,
                        memory_id,
                        task_type_value,
                        recall_mode_value,
                        monitor_mode,
                        normalize_text(str(assessment.get("source_category") or "AGENT_INFERENCE"))[:80],
                        _clamp(float(assessment.get("raw_probability") or 0.0)),
                        _clamp(float(assessment.get("calibrated_probability") or 0.0)),
                        decision,
                        int(applied),
                        normalize_text(str(assessment.get("reason") or "No rationale recorded."))[:500],
                        normalize_text(str(assessment.get("calibration_scope") or "prior"))[:40],
                        max(0, int(assessment.get("calibration_samples") or 0)),
                        json.dumps(assessment.get("features") or {}, sort_keys=True),
                        initial_outcome,
                        now,
                        now if initial_outcome == "withheld" else None,
                    ),
                )
        return task_id

    def resolve_usage(
        self,
        task_id: str,
        attribution_by_id: dict[str, float],
        *,
        memory_actions: Sequence[dict[str, Any]] = (),
    ) -> int:
        now = utc_now()
        resolved = 0
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT memory_id FROM usage_records WHERE task_id=? AND outcome='pending'", (task_id,)
            ).fetchall()
            for row in rows:
                memory_id = row["memory_id"]
                attribution = _clamp(attribution_by_id.get(memory_id, 0.0))
                used = attribution > 0.0
                conn.execute(
                    """UPDATE usage_records SET used=?,attribution=?,outcome=?,resolved_at=?
                       WHERE task_id=? AND memory_id=?""",
                    (int(used), attribution, "used" if used else "ignored", now, task_id, memory_id),
                )
                conn.execute(
                    """UPDATE metacognitive_predictions SET outcome=?,resolved_at=?
                       WHERE task_id=? AND memory_id=? AND outcome='pending'""",
                    ("used" if used else "ignored", now, task_id, memory_id),
                )
                if used:
                    conn.execute(
                        """INSERT OR IGNORE INTO memory_experience_events(
                             event_id,event_type,memory_id,task_id,evidence_key,event_day,
                             outcome,metadata_json,created_at
                           ) VALUES(?,'memory_used',?,?,?,?,'used',?,?)""",
                        (
                            str(uuid.uuid4()),
                            memory_id,
                            task_id,
                            f"used:{task_id}:{memory_id}",
                            now[:10],
                            _trace_json({"attribution": round(attribution, 6)}),
                            now,
                        ),
                    )
                resolved += 1
            used_count = sum(
                1
                for row in rows
                if _clamp(attribution_by_id.get(str(row["memory_id"]), 0.0)) > 0.0
            )
            conn.execute(
                """UPDATE recall_budget_observations
                   SET used_count=?,outcome=?,resolved_at=?
                   WHERE task_id=? AND outcome='pending'""",
                (used_count, "used" if used_count else "ignored", now, task_id),
            )
            attention = conn.execute(
                """SELECT task_type,topics_json FROM attention_observations
                   WHERE task_id=? AND usage_outcome='pending'""",
                (task_id,),
            ).fetchone()
            if attention:
                conn.execute(
                    """UPDATE attention_observations SET usage_outcome=?,resolved_at=?
                       WHERE task_id=? AND usage_outcome='pending'""",
                    ("used" if used_count else "ignored", now, task_id),
                )
                try:
                    topics = json.loads(str(attention["topics_json"] or "[]"))
                except json.JSONDecodeError:
                    topics = []
                self._refresh_attention_weights_tx(
                    conn,
                    task_type=str(attention["task_type"]),
                    topics=topics if isinstance(topics, list) else [],
                    now=now,
                )
            self._resolve_memory_trace_usage_tx(
                conn,
                task_id,
                attribution_by_id,
                memory_actions,
                now,
            )
        return resolved

    def apply_task_outcome(self, task_id: str, outcome: str) -> list[str]:
        if outcome not in {"helpful", "harmful", "validated", "corrected"}:
            raise ValueError("invalid task outcome")
        from .research import sync_task_outcome_tx

        now = utc_now()
        with self.transaction() as conn:
            rows = conn.execute("SELECT memory_id FROM usage_records WHERE task_id=? AND used=1", (task_id,)).fetchall()
            ids = [str(row["memory_id"]) for row in rows]
            conn.execute(
                "UPDATE usage_records SET outcome=?,resolved_at=? WHERE task_id=? AND used=1",
                (outcome, now, task_id),
            )
            if ids:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"""UPDATE metacognitive_predictions SET outcome=?,resolved_at=?
                        WHERE task_id=? AND memory_id IN ({placeholders})
                          AND outcome IN ('pending','used')""",
                    (outcome, now, task_id, *ids),
                )
            conn.execute(
                """UPDATE recall_budget_observations SET outcome=?,resolved_at=?
                   WHERE task_id=? AND used_count>0""",
                (outcome, now, task_id),
            )
            attention = conn.execute(
                """SELECT task_type,topics_json FROM attention_observations
                   WHERE task_id=? AND usage_outcome='used'""",
                (task_id,),
            ).fetchone()
            if attention:
                conn.execute(
                    """UPDATE attention_observations SET final_outcome=?,resolved_at=?
                       WHERE task_id=? AND usage_outcome='used'""",
                    (outcome, now, task_id),
                )
                try:
                    topics = json.loads(str(attention["topics_json"] or "[]"))
                except json.JSONDecodeError:
                    topics = []
                self._refresh_attention_weights_tx(
                    conn,
                    task_type=str(attention["task_type"]),
                    topics=topics if isinstance(topics, list) else [],
                    now=now,
                )
            sync_task_outcome_tx(
                conn,
                task_id,
                outcome,
                source="conversation_feedback",
                resolved_at=now,
            )
            self._update_memory_trace_outcome_tx(
                conn,
                task_id,
                outcome,
                now,
                source="conversation_feedback",
            )
            for memory_id in ids:
                conn.execute(
                    """INSERT OR IGNORE INTO memory_experience_events(
                         event_id,event_type,memory_id,task_id,evidence_key,event_day,
                         outcome,metadata_json,created_at
                       ) VALUES(?,'outcome_labeled',?,?,?,? ,?,'{}',?)""",
                    (
                        str(uuid.uuid4()),
                        memory_id,
                        task_id,
                        f"outcome:{task_id}:{memory_id}:{outcome}",
                        now[:10],
                        outcome,
                        now,
                    ),
                )
        for memory_id in ids:
            self.log_access(memory_id, outcome if outcome != "harmful" else "wrong")
        return ids

    def label_task_outcome(self, task_id: str, outcome: str, *, actor: str) -> dict[str, Any]:
        """Apply one auditable, reversible outcome label to a used recall task."""

        if outcome not in {"helpful", "harmful", "validated", "corrected"}:
            raise ValueError("outcome must be helpful, harmful, validated, or corrected")
        actor_value = normalize_text(actor)[:80] or "dashboard-operator"
        now = utc_now()
        from .research import sync_task_outcome_tx

        with self.transaction() as conn:
            rows = conn.execute(
                """SELECT u.memory_id,u.query,u.session_id,u.outcome,m.content
                   FROM usage_records u JOIN memories m ON m.id=u.memory_id
                   WHERE u.task_id=? AND u.used=1 ORDER BY u.created_at,u.memory_id""",
                (task_id,),
            ).fetchall()
            agent_task = conn.execute(
                "SELECT * FROM agent_task_observations WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if not rows and not agent_task:
                raise ValueError("task not found")
            active = conn.execute(
                "SELECT * FROM task_outcome_labels WHERE task_id=? AND active=1", (task_id,)
            ).fetchone()
            if active and str(active["outcome"]) == outcome:
                return {"changed": False, "label_id": active["label_id"], "outcome": outcome, "task_id": task_id}
            prior_outcome = str(
                active["prior_outcome"]
                if active
                else (rows[0]["outcome"] if rows else agent_task["outcome"] if agent_task else "used")
                or "used"
            )
            if prior_outcome not in {"used", "helpful", "harmful", "validated", "corrected"}:
                prior_outcome = "used"
            if active:
                self._reverse_task_outcome_tx(conn, dict(active), now, reversed_by=actor_value)
            label_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO task_outcome_labels(
                     label_id,task_id,outcome,actor,source,prior_outcome,active,created_at
                   ) VALUES(?,?,?,?, 'dashboard',?,1,?)""",
                (label_id, task_id, outcome, actor_value, prior_outcome, now),
            )
            memory_ids = [str(row["memory_id"]) for row in rows]
            placeholders = ",".join("?" for _ in memory_ids)
            if memory_ids:
                conn.execute(
                    f"UPDATE usage_records SET outcome=?,resolved_at=? WHERE task_id=? AND memory_id IN ({placeholders})",
                    (outcome, now, task_id, *memory_ids),
                )
                conn.execute(
                    f"""UPDATE metacognitive_predictions SET outcome=?,resolved_at=?
                        WHERE task_id=? AND memory_id IN ({placeholders})""",
                    (outcome, now, task_id, *memory_ids),
                )
            conn.execute(
                "UPDATE recall_budget_observations SET outcome=?,resolved_at=? WHERE task_id=?",
                (outcome, now, task_id),
            )
            self._set_attention_final_outcome_tx(
                conn,
                task_id=task_id,
                outcome=outcome,
                now=now,
            )
            count_columns = {
                "helpful": "helpful_count",
                "harmful": "harmful_count",
                "validated": "validated_count",
                "corrected": "correction_count",
            }
            if memory_ids and prior_outcome != outcome:
                if prior_outcome in count_columns:
                    prior_column = count_columns[prior_outcome]
                    conn.execute(
                        f"UPDATE memories SET {prior_column}=MAX(0,{prior_column}-1) "
                        f"WHERE id IN ({placeholders})",
                        memory_ids,
                    )
                outcome_column = count_columns[outcome]
                conn.execute(
                    f"UPDATE memories SET {outcome_column}={outcome_column}+1 "
                    f"WHERE id IN ({placeholders})",
                    memory_ids,
                )
            for memory_id in memory_ids:
                conn.execute(
                    """INSERT OR IGNORE INTO memory_experience_events(
                         event_id,event_type,memory_id,task_id,evidence_key,event_day,
                         outcome,metadata_json,created_at
                       ) VALUES(?,'outcome_labeled',?,?,?,?,?,'{}',?)""",
                    (
                        str(uuid.uuid4()),
                        memory_id,
                        task_id,
                        f"outcome-label:{label_id}:{memory_id}",
                        now[:10],
                        outcome,
                        now,
                    ),
                )
            event = "wrong" if outcome == "harmful" else outcome
            marker = f"task-outcome:{label_id}"
            conn.executemany(
                "INSERT INTO access_log(memory_id,event,query,session_id,score,created_at) VALUES(?,?,?,?,NULL,?)",
                [
                    (
                        memory_id,
                        event,
                        marker,
                        rows[0]["session_id"] if rows else agent_task["session_id"],
                        now,
                    )
                    for memory_id in memory_ids
                ],
            )
            task_type_row = conn.execute(
                "SELECT task_type FROM recall_budget_observations WHERE task_id=?", (task_id,)
            ).fetchone()
            if outcome in {"helpful", "validated"} and memory_ids:
                conn.execute(
                    """INSERT INTO evaluation_cases(
                         case_id,task_id,query,relevant_memory_ids,task_type,source_label_id,active,created_at,updated_at
                       ) VALUES(?,?,?,?,?,?,1,?,?)
                       ON CONFLICT(task_id) DO UPDATE SET
                         query=excluded.query,relevant_memory_ids=excluded.relevant_memory_ids,
                         task_type=excluded.task_type,source_label_id=excluded.source_label_id,
                         active=1,updated_at=excluded.updated_at""",
                    (
                        str(uuid.uuid4()),
                        task_id,
                        str(rows[0]["query"] if rows else agent_task["query_preview"] or ""),
                        json.dumps(memory_ids),
                        str(task_type_row["task_type"] if task_type_row else "general"),
                        label_id,
                        now,
                        now,
                    ),
                )
            else:
                conn.execute("UPDATE evaluation_cases SET active=0,updated_at=? WHERE task_id=?", (now, task_id))
            conn.execute(
                "UPDATE tool_guidance_exposures SET task_outcome=? WHERE task_id=?",
                (outcome, task_id),
            )
            sync_task_outcome_tx(conn, task_id, outcome, source="dashboard", resolved_at=now)
            self._update_memory_trace_outcome_tx(
                conn,
                task_id,
                outcome,
                now,
                source="dashboard",
            )
        return {
            "changed": True,
            "label_id": label_id,
            "task_id": task_id,
            "outcome": outcome,
            "memory_ids": memory_ids,
        }

    def undo_task_outcome_label(self, task_id: str, *, actor: str) -> bool:
        """Undo the active dashboard label and restore its preceding task outcome."""

        actor_value = normalize_text(actor)[:80] or "dashboard-operator"
        now = utc_now()
        with self.transaction() as conn:
            active = conn.execute(
                "SELECT * FROM task_outcome_labels WHERE task_id=? AND active=1", (task_id,)
            ).fetchone()
            if not active:
                return False
            self._reverse_task_outcome_tx(conn, dict(active), now, reversed_by=actor_value)
        return True

    def _reverse_task_outcome_tx(
        self,
        conn: sqlite3.Connection,
        label: dict[str, Any],
        now: str,
        *,
        reversed_by: str,
    ) -> None:
        task_id = str(label["task_id"])
        outcome = str(label["outcome"])
        prior_outcome = str(label.get("prior_outcome") or "used")
        rows = conn.execute(
            "SELECT memory_id FROM usage_records WHERE task_id=? AND used=1", (task_id,)
        ).fetchall()
        memory_ids = [str(row["memory_id"]) for row in rows]
        if memory_ids and prior_outcome != outcome:
            placeholders = ",".join("?" for _ in memory_ids)
            count_columns = {
                "helpful": "helpful_count",
                "harmful": "harmful_count",
                "validated": "validated_count",
                "corrected": "correction_count",
            }
            count_column = count_columns[outcome]
            conn.execute(
                f"UPDATE memories SET {count_column}=MAX(0,{count_column}-1) WHERE id IN ({placeholders})",
                memory_ids,
            )
            if prior_outcome in count_columns:
                prior_column = count_columns[prior_outcome]
                conn.execute(
                    f"UPDATE memories SET {prior_column}={prior_column}+1 WHERE id IN ({placeholders})",
                    memory_ids,
                )
        conn.execute(
            "UPDATE usage_records SET outcome=?,resolved_at=? WHERE task_id=? AND used=1",
            (prior_outcome, now, task_id),
        )
        conn.execute(
            """UPDATE metacognitive_predictions SET outcome=?,resolved_at=?
               WHERE task_id=? AND outcome IN ('helpful','harmful','validated','corrected')""",
            (prior_outcome, now, task_id),
        )
        conn.execute(
            "UPDATE recall_budget_observations SET outcome=?,resolved_at=? WHERE task_id=?",
            (prior_outcome, now, task_id),
        )
        self._set_attention_final_outcome_tx(
            conn,
            task_id=task_id,
            outcome=prior_outcome,
            now=now,
        )
        conn.execute("DELETE FROM access_log WHERE query=?", (f"task-outcome:{label['label_id']}",))
        conn.execute(
            "UPDATE task_outcome_labels SET active=0,reversed_at=?,reversed_by=? WHERE label_id=?",
            (now, reversed_by, label["label_id"]),
        )
        conn.execute("UPDATE evaluation_cases SET active=0,updated_at=? WHERE task_id=?", (now, task_id))
        conn.execute(
            "UPDATE tool_guidance_exposures SET task_outcome=? WHERE task_id=?",
            (prior_outcome if prior_outcome != "used" else None, task_id),
        )
        from .research import sync_task_outcome_tx

        sync_task_outcome_tx(
            conn,
            task_id,
            prior_outcome,
            source=None if prior_outcome == "used" else "restored_prior",
            resolved_at=now,
        )
        self._restore_memory_trace_outcome_tx(conn, task_id, prior_outcome, now)

    def record_episode(self, user_content: str, assistant_content: str, *, session_id: str | None = None) -> bool:
        user_content = normalize_text(user_content)
        assistant_content = normalize_text(assistant_content)
        digest = hashlib.sha256(f"{session_id}\0{user_content}\0{assistant_content}".encode()).hexdigest()
        with self.transaction() as conn:
            result = conn.execute(
                "INSERT OR IGNORE INTO episodes(session_id,user_content,assistant_content,created_at,content_hash) VALUES(?,?,?,?,?)",
                (session_id, user_content, assistant_content, utc_now(), digest),
            )
            return result.rowcount > 0

    def record_tool_execution(self, execution: Any) -> tuple[bool, dict[str, Any]]:
        """Persist a redacted tool outcome and update its task-specific aggregate."""
        now = utc_now()
        with self.transaction() as conn:
            result = conn.execute(
                """INSERT OR IGNORE INTO tool_executions(
                   execution_id,session_id,task_type,task_context,tool_name,argument_keys,
                   success,error_type,result_summary,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    execution.execution_id,
                    execution.session_id,
                    execution.task_type,
                    execution.task_context,
                    execution.tool_name,
                    json.dumps(execution.argument_keys),
                    int(execution.success),
                    execution.error_type,
                    execution.result_summary,
                    now,
                ),
            )
            created = result.rowcount > 0
            if created:
                conn.execute(
                    """INSERT INTO tool_stats(
                       task_type,tool_name,success_count,failure_count,argument_keys,
                       last_success_at,last_failure_at,last_error_type,updated_at
                       ) VALUES(?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(task_type,tool_name) DO UPDATE SET
                         success_count=tool_stats.success_count+excluded.success_count,
                         failure_count=tool_stats.failure_count+excluded.failure_count,
                         argument_keys=excluded.argument_keys,
                         last_success_at=COALESCE(excluded.last_success_at,tool_stats.last_success_at),
                         last_failure_at=COALESCE(excluded.last_failure_at,tool_stats.last_failure_at),
                         last_error_type=COALESCE(excluded.last_error_type,tool_stats.last_error_type),
                         updated_at=excluded.updated_at""",
                    (
                        execution.task_type,
                        execution.tool_name,
                        int(execution.success),
                        int(not execution.success),
                        json.dumps(execution.argument_keys),
                        now if execution.success else None,
                        now if not execution.success else None,
                        execution.error_type,
                        now,
                    ),
                )
            stats = conn.execute(
                """SELECT s.*,
                   (SELECT COUNT(DISTINCT NULLIF(e.task_context,'')) FROM tool_executions e
                    WHERE e.task_type=s.task_type AND e.tool_name=s.tool_name) distinct_tasks
                   FROM tool_stats s WHERE s.task_type=? AND s.tool_name=?""",
                (execution.task_type, execution.tool_name),
            ).fetchone()
        return created, dict(stats) if stats else {}

    def tool_guidance(self, task_type: str, *, limit: int = 4) -> list[dict[str, Any]]:
        """Return only reinforced tool evidence; single attempts do not generalize."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT s.*,
                   CAST(s.success_count AS REAL)/MAX(1,s.success_count+s.failure_count) reliability,
                   (SELECT COUNT(DISTINCT NULLIF(e.task_context,'')) FROM tool_executions e
                    WHERE e.task_type=s.task_type AND e.tool_name=s.tool_name) distinct_tasks
                   FROM tool_stats s WHERE s.task_type=? AND s.success_count+s.failure_count>=2
                   AND (SELECT COUNT(DISTINCT NULLIF(e.task_context,'')) FROM tool_executions e
                        WHERE e.task_type=s.task_type AND e.tool_name=s.tool_name)>=2
                   ORDER BY reliability DESC, success_count DESC, failure_count ASC LIMIT ?""",
                (task_type, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def record_tool_workflow(self, workflow: Any) -> tuple[bool, dict[str, Any]]:
        now = utc_now()
        steps_json = json.dumps(workflow.steps, sort_keys=True)
        with self.transaction() as conn:
            result = conn.execute(
                """INSERT OR IGNORE INTO tool_workflows(
                   workflow_id,session_id,task_type,task_fingerprint,task_context,steps_json,
                   workflow_key,success,error_type,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    workflow.workflow_id,
                    workflow.session_id,
                    workflow.task_type,
                    workflow.task_fingerprint,
                    workflow.task_context,
                    steps_json,
                    workflow.workflow_key,
                    int(workflow.success),
                    workflow.error_type,
                    now,
                ),
            )
            created = result.rowcount > 0
            if created:
                conn.execute(
                    """INSERT INTO tool_workflow_stats(
                       task_type,task_fingerprint,workflow_key,steps_json,success_count,failure_count,
                       distinct_tasks,last_error_type,updated_at
                       ) VALUES(?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(task_type,task_fingerprint,workflow_key) DO UPDATE SET
                         success_count=tool_workflow_stats.success_count+excluded.success_count,
                         failure_count=tool_workflow_stats.failure_count+excluded.failure_count,
                         steps_json=excluded.steps_json,
                         last_error_type=COALESCE(excluded.last_error_type,tool_workflow_stats.last_error_type),
                         updated_at=excluded.updated_at""",
                    (
                        workflow.task_type,
                        workflow.task_fingerprint,
                        workflow.workflow_key,
                        steps_json,
                        int(workflow.success),
                        int(not workflow.success),
                        1,
                        workflow.error_type,
                        now,
                    ),
                )
                distinct = conn.execute(
                    """SELECT COUNT(DISTINCT NULLIF(task_context,'')) count FROM tool_workflows
                       WHERE task_type=? AND task_fingerprint=? AND workflow_key=?""",
                    (workflow.task_type, workflow.task_fingerprint, workflow.workflow_key),
                ).fetchone()["count"]
                conn.execute(
                    """UPDATE tool_workflow_stats SET distinct_tasks=?
                       WHERE task_type=? AND task_fingerprint=? AND workflow_key=?""",
                    (distinct, workflow.task_type, workflow.task_fingerprint, workflow.workflow_key),
                )
            stats = conn.execute(
                """SELECT * FROM tool_workflow_stats
                   WHERE task_type=? AND task_fingerprint=? AND workflow_key=?""",
                (workflow.task_type, workflow.task_fingerprint, workflow.workflow_key),
            ).fetchone()
        return created, dict(stats) if stats else {}

    def tool_workflow_guidance(
        self,
        task_type: str,
        task_fingerprint: str,
        *,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        query_features = set((task_fingerprint or "").split("|"))
        with self._lock:
            rows = self._conn.execute(
                """SELECT *,CAST(success_count AS REAL)/MAX(1,success_count+failure_count) reliability
                   FROM tool_workflow_stats WHERE task_type=?
                   AND success_count+failure_count>=2 AND distinct_tasks>=2
                   ORDER BY reliability DESC,success_count DESC,failure_count ASC LIMIT 30""",
                (task_type,),
            ).fetchall()
        ranked: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            item = dict(row)
            stored = set(str(item["task_fingerprint"] or "").split("|"))
            similarity = len(query_features & stored) / max(1, len(query_features | stored))
            if similarity <= 0 and task_fingerprint != "general":
                continue
            item["fingerprint_similarity"] = round(similarity, 6)
            ranked.append((0.65 * float(item["reliability"]) + 0.35 * similarity, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in ranked[:limit]]

    def record_tool_guidance_exposures(
        self,
        *,
        session_id: str | None,
        task_id: str | None,
        task_type: str,
        tool_guidance: Sequence[dict[str, Any]] = (),
        workflow_guidance: Sequence[dict[str, Any]] = (),
    ) -> int:
        """Record which reinforced guidance was actually shown before a tool turn."""

        now = utc_now()
        rows: list[tuple[Any, ...]] = []
        for guidance in tool_guidance:
            successes = int(guidance.get("success_count") or 0)
            failures = int(guidance.get("failure_count") or 0)
            reliability = successes / max(1, successes + failures)
            tool_name = str(guidance.get("tool_name") or "")
            if not tool_name:
                continue
            rows.append(
                (
                    str(uuid.uuid4()), session_id, task_id, task_type, "tool",
                    f"{task_type}:{tool_name}", tool_name, reliability, now,
                )
            )
        for guidance in workflow_guidance:
            successes = int(guidance.get("success_count") or 0)
            failures = int(guidance.get("failure_count") or 0)
            reliability = successes / max(1, successes + failures)
            workflow_key = str(guidance.get("workflow_key") or "")
            if not workflow_key:
                continue
            rows.append(
                (
                    str(uuid.uuid4()), session_id, task_id, task_type, "workflow",
                    workflow_key, None, reliability, now,
                )
            )
        if not rows:
            return 0
        with self.transaction() as conn:
            conn.executemany(
                """INSERT INTO tool_guidance_exposures(
                     exposure_id,session_id,task_id,task_type,guidance_type,guidance_key,
                     recommended_tool,predicted_reliability,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?)""",
                rows,
            )
        return len(rows)

    def resolve_tool_guidance_exposures(
        self,
        *,
        session_id: str,
        executions: Sequence[Any],
        workflow: Any | None,
    ) -> int:
        """Resolve recent guidance as followed or not followed without treating either as causal."""

        with self.transaction() as conn:
            rows = conn.execute(
                """SELECT * FROM tool_guidance_exposures
                   WHERE session_id=? AND resolved_at IS NULL
                   ORDER BY created_at DESC LIMIT 100""",
                (session_id,),
            ).fetchall()
            now = utc_now()
            for row in rows:
                if row["guidance_type"] == "tool":
                    matching = [item for item in executions if item.tool_name == row["recommended_tool"]]
                    followed = bool(matching)
                    success = int(any(item.success for item in matching)) if matching else None
                else:
                    followed = bool(workflow and workflow.workflow_key == row["guidance_key"])
                    success = int(bool(workflow.success)) if followed else None
                conn.execute(
                    """UPDATE tool_guidance_exposures SET followed=?,success=?,resolved_at=?
                       WHERE exposure_id=?""",
                    (int(followed), success, now, row["exposure_id"]),
                )
        return len(rows)

    def document_manifest(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM document_sources").fetchall()
        return {str(row["source_path"]): dict(row) for row in rows}

    def document_chunks(self, source_path: str, *, active_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM document_chunks WHERE source_path=?"
        if active_only:
            sql += " AND active=1"
        sql += " ORDER BY ordinal,chunk_key"
        with self._lock:
            rows = self._conn.execute(sql, (source_path,)).fetchall()
        return [dict(row) for row in rows]

    def upsert_document_source(
        self,
        *,
        source_path: str,
        title: str,
        content_hash: str,
        modified_at: str,
        size_bytes: int,
        status: str = "active",
    ) -> None:
        now = utc_now()
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO document_sources(
                   source_path,source_type,title,content_hash,modified_at,size_bytes,status,imported_at,last_seen_at
                   ) VALUES(?,\'obsidian_markdown\',?,?,?,?,?,?,?)
                   ON CONFLICT(source_path) DO UPDATE SET
                     title=excluded.title,content_hash=excluded.content_hash,
                     modified_at=excluded.modified_at,size_bytes=excluded.size_bytes,
                     status=excluded.status,last_seen_at=excluded.last_seen_at""",
                (source_path, title, content_hash, modified_at, int(size_bytes), status, now, now),
            )

    def upsert_document_chunk(
        self,
        *,
        source_path: str,
        chunk_key: str,
        memory_id: str,
        chunk_hash: str,
        heading: str,
        ordinal: int,
        active: bool = True,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO document_chunks(
                   source_path,chunk_key,memory_id,chunk_hash,heading,ordinal,active,imported_at
                   ) VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(source_path,chunk_key) DO UPDATE SET
                     memory_id=excluded.memory_id,chunk_hash=excluded.chunk_hash,
                     heading=excluded.heading,ordinal=excluded.ordinal,
                     active=excluded.active,imported_at=excluded.imported_at""",
                (
                    source_path,
                    chunk_key,
                    memory_id,
                    chunk_hash,
                    heading,
                    int(ordinal),
                    int(active),
                    utc_now(),
                ),
            )

    def update_document_memory(
        self,
        memory_id: str,
        content: str,
        *,
        kind: str,
        source_ref: str,
        entities: Sequence[str],
        source_context: str,
        observed_at: str,
        confidence: float,
        currentness_confidence: float,
        importance: float,
        uniqueness: float,
        volatility: float,
        trust: float,
        state: str,
        quarantine_reason: str | None,
        valid_from: str | None,
        subject: str,
        predicate: str,
        object_value: str,
        extraction_method: str,
        reason: str,
    ) -> dict[str, bool]:
        """Revise one vault chunk in place while preserving version history.

        A document edit is a new version of the same chunk, not a brand-new
        recallable memory.  Keeping the stable memory ID prevents the Index
        from filling with archived copies of one evolving file.
        """

        clean = normalize_text(content)
        if not clean:
            raise ValueError("document memory content cannot be empty")
        entity_values = _normalize_context_list(entities)
        source_context_value = normalize_text(source_context)[:1000] or None
        completeness = _memory_metadata_completeness(
            "standalone",
            scope={},
            entities=entity_values,
            preconditions={},
            source_context=source_context_value,
            applicable_systems=[],
            applicable_versions=[],
        )
        now = utc_now()
        with self.transaction() as conn:
            current = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            if not current or str(current["source_type"]) != "vault_markdown":
                return {"updated": False, "reactivated": False}
            prior_state = str(current["state"])
            next_state = "quarantine" if quarantine_reason else state
            content_changed = str(current["content_hash"]) != content_hash(clean)
            state_changed = prior_state != next_state
            if not content_changed and not state_changed:
                return {"updated": False, "reactivated": False}
            conn.execute(
                "UPDATE memory_versions SET system_to=? WHERE memory_id=? AND system_to IS NULL",
                (now, memory_id),
            )
            conn.execute(
                """INSERT INTO memory_versions(
                   memory_id,content,confidence,state,valid_from,valid_to,
                   system_from,reason,source_ref
                   ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    memory_id,
                    clean,
                    _clamp(confidence),
                    next_state,
                    valid_from,
                    current["valid_to"],
                    now,
                    normalize_text(reason)[:500],
                    source_ref,
                ),
            )
            conn.execute(
                """UPDATE memories SET kind=?,content=?,content_hash=?,source_ref=?,
                   source_category='DOCUMENT_EXTRACTED',context_mode='standalone',scope_json='{}',
                   entities_json=?,preconditions_json='{}',source_context=?,
                   applicable_systems_json='[]',applicable_versions_json='[]',metadata_completeness=?,
                   updated_at=?,observed_at=?,valid_from=?,subject=?,predicate=?,object_value=?,
                   extraction_method=?,confidence=?,currentness_confidence=?,importance=?,uniqueness=?,
                   volatility=?,trust=?,state=?,quarantine_reason=? WHERE id=?""",
                (
                    kind,
                    clean,
                    content_hash(clean),
                    source_ref,
                    _trace_json(entity_values),
                    source_context_value,
                    completeness,
                    now,
                    observed_at,
                    valid_from,
                    subject,
                    predicate,
                    object_value,
                    extraction_method,
                    _clamp(confidence),
                    _clamp(currentness_confidence),
                    _clamp(importance),
                    _clamp(uniqueness),
                    _clamp(volatility),
                    _clamp(trust),
                    next_state,
                    quarantine_reason,
                    memory_id,
                ),
            )
            if content_changed:
                conn.execute("DELETE FROM memory_fts WHERE memory_id=?", (memory_id,))
                conn.execute("INSERT INTO memory_fts(memory_id,content) VALUES(?,?)", (memory_id, clean))
                self._index_features_tx(conn, memory_id, clean)
                self._mark_dependents_dirty_tx(conn, memory_id, "vault source chunk revised")
            self._index_context_terms_tx(
                conn,
                memory_id,
                context_mode="standalone",
                scope={},
                entities=entity_values,
                preconditions={},
                applicable_systems=[],
                applicable_versions=[],
            )
            if state_changed:
                conn.execute(
                    """INSERT INTO lifecycle_events(
                       memory_id,from_state,to_state,reason,retention_score,created_at
                       ) VALUES(?,?,?,?,?,?)""",
                    (memory_id, prior_state, next_state, normalize_text(reason)[:500], None, now),
                )
            revised_row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            if revised_row:
                self._classify_and_present_tx(
                    conn,
                    dict(revised_row),
                    has_active_dependencies=self._has_active_dependency_tx(conn, memory_id),
                )
            return {
                "updated": content_changed,
                "reactivated": prior_state in {"archived", "cold"} and next_state == "active",
            }

    def deactivate_document_chunk(self, source_path: str, chunk_key: str) -> bool:
        with self.transaction() as conn:
            result = conn.execute(
                "UPDATE document_chunks SET active=0,imported_at=? WHERE source_path=? AND chunk_key=?",
                (utc_now(), source_path, chunk_key),
            )
            return result.rowcount > 0

    def mark_document_missing(self, source_path: str) -> list[str]:
        """Archive active chunks for a source no longer present in the vault."""
        chunks = self.document_chunks(source_path, active_only=True)
        archived: list[str] = []
        for chunk in chunks:
            if self.set_state(chunk["memory_id"], "archived", reason="vault source removed"):
                archived.append(chunk["memory_id"])
            self.deactivate_document_chunk(source_path, chunk["chunk_key"])
        with self.transaction() as conn:
            conn.execute(
                "UPDATE document_sources SET status='missing',last_seen_at=? WHERE source_path=?",
                (utc_now(), source_path),
            )
        return archived

    def clear_edges(self, relation: str) -> int:
        with self.transaction() as conn:
            result = conn.execute("DELETE FROM edges WHERE relation=?", (relation,))
            return result.rowcount

    def versions(self, memory_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memory_versions WHERE memory_id=? ORDER BY system_from", (memory_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def latest_lifecycle_event(self, memory_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """SELECT * FROM lifecycle_events WHERE memory_id=?
                   ORDER BY event_id DESC LIMIT 1""",
                (memory_id,),
            ).fetchone()
        return dict(row) if row else None

    def explain(self, memory_id: str) -> dict[str, Any] | None:
        memory = self.get_memory(memory_id)
        if not memory:
            return None
        with self._lock:
            edges = self._conn.execute(
                """SELECT e.*,
                          CASE WHEN e.src_id=? THEN dst.content ELSE src.content END peer_content,
                          CASE WHEN e.src_id=? THEN dst.kind ELSE src.kind END peer_kind,
                          CASE WHEN e.src_id=? THEN dst.state ELSE src.state END peer_state,
                          CASE WHEN e.src_id=? THEN dst.source_ref ELSE src.source_ref END peer_source_ref,
                          COALESCE((SELECT ev.summary FROM edge_evidence ev
                            WHERE ev.src_id=e.src_id AND ev.dst_id=e.dst_id AND ev.relation=e.relation
                            ORDER BY ev.created_at DESC LIMIT 1),'') explanation,
                          COALESCE((SELECT ev.evidence_type FROM edge_evidence ev
                            WHERE ev.src_id=e.src_id AND ev.dst_id=e.dst_id AND ev.relation=e.relation
                            ORDER BY ev.created_at DESC LIMIT 1),'legacy_unattributed') evidence_type,
                          (SELECT COUNT(*) FROM edge_evidence ev
                            WHERE ev.src_id=e.src_id AND ev.dst_id=e.dst_id AND ev.relation=e.relation) evidence_records
                   FROM edges e
                   JOIN memories src ON src.id=e.src_id
                   JOIN memories dst ON dst.id=e.dst_id
                   WHERE e.src_id=? OR e.dst_id=?
                   ORDER BY e.weight DESC,e.evidence_count DESC LIMIT 20""",
                (memory_id, memory_id, memory_id, memory_id, memory_id, memory_id),
            ).fetchall()
            accesses = self._conn.execute(
                "SELECT event,query,session_id,score,created_at FROM access_log WHERE memory_id=? ORDER BY id DESC LIMIT 20",
                (memory_id,),
            ).fetchall()
            neighborhoods = self._conn.execute(
                """SELECT n.slug,n.label,n.category,n.safety_class,n.description,
                          nm.confidence,nm.origin,nm.explanation,nm.created_at,nm.reviewed_at
                   FROM memory_neighborhood_memberships nm
                   JOIN memory_neighborhoods n ON n.neighborhood_id=nm.neighborhood_id
                   WHERE nm.memory_id=? ORDER BY nm.confidence DESC,n.label""",
                (memory_id,),
            ).fetchall()
        edge_items: list[dict[str, Any]] = []
        for row in edges:
            item = _decode_edge(row)
            item["evidence"] = self.edge_evidence(
                str(row["src_id"]), str(row["dst_id"]), str(row["relation"]), limit=20
            )
            edge_items.append(item)
        with self._lock:
            presentation_row = self._conn.execute(
                "SELECT * FROM memory_presentations WHERE memory_id=?", (memory_id,)
            ).fetchone()
        presentation = dict(presentation_row) if presentation_row else None
        if presentation:
            presentation["readability_flags"] = _json_string_list(
                presentation.pop("readability_flags_json", "[]")
            )
        return {
            "memory": memory,
            "presentation": presentation,
            "versions": self.versions(memory_id),
            "edges": edge_items,
            "dependencies": self.dependencies(memory_id),
            "neighborhoods": [dict(row) for row in neighborhoods],
            "recent_access": [dict(r) for r in accesses],
        }

    def outcome_lab_snapshot(self, *, limit: int = 40) -> dict[str, Any]:
        """Return private task-level outcomes and the measurement readiness model."""

        bounded = max(1, min(int(limit), 500))
        with self._lock:
            task_rows = self._conn.execute(
                """SELECT u.task_id,MAX(u.query) query,MIN(u.session_id) session_id,
                          MIN(u.created_at) created_at,COUNT(*) selected_count,SUM(u.used) used_count,
                          b.task_type,b.mode recall_mode,b.estimated_tokens,
                          l.label_id,l.outcome label_outcome,l.actor,l.created_at labeled_at
                   FROM usage_records u
                   LEFT JOIN recall_budget_observations b ON b.task_id=u.task_id
                   LEFT JOIN task_outcome_labels l ON l.task_id=u.task_id AND l.active=1
                   GROUP BY u.task_id
                   HAVING SUM(u.used)>0
                   ORDER BY MIN(u.created_at) DESC LIMIT ?""",
                (bounded,),
            ).fetchall()
            eligible_count = int(
                self._conn.execute(
                    "SELECT COUNT(DISTINCT task_id) count FROM usage_records WHERE used=1"
                ).fetchone()["count"]
            )
            label_summary = self._conn.execute(
                """SELECT COUNT(*) labeled_count,
                          SUM(CASE WHEN outcome IN ('helpful','validated') THEN 1 ELSE 0 END) positive_count,
                          SUM(CASE WHEN outcome IN ('harmful','corrected') THEN 1 ELSE 0 END) negative_count
                   FROM task_outcome_labels WHERE active=1"""
            ).fetchone()
            evaluation_case_count = int(
                self._conn.execute("SELECT COUNT(*) count FROM evaluation_cases WHERE active=1").fetchone()["count"]
            )
            tasks: list[dict[str, Any]] = []
            for row in task_rows:
                item = dict(row)
                memory_rows = self._conn.execute(
                    """SELECT u.memory_id,u.attribution,u.score,m.kind,m.state,m.content,m.source_category
                       FROM usage_records u JOIN memories m ON m.id=u.memory_id
                       WHERE u.task_id=? AND u.used=1 ORDER BY u.attribution DESC,u.score DESC LIMIT 8""",
                    (row["task_id"],),
                ).fetchall()
                item["memories"] = [dict(memory) for memory in memory_rows]
                trace = self._conn.execute(
                    """SELECT retrieval_reason,candidate_memories_json,retrieval_context_json
                       FROM memory_traces WHERE task_id=?""",
                    (row["task_id"],),
                ).fetchone()
                item["why_recalled"] = (
                    {
                        "reason": str(trace["retrieval_reason"]),
                        "candidates": _trace_json_list(trace["candidate_memories_json"]),
                        "context": _trace_json_object(trace["retrieval_context_json"]),
                    }
                    if trace
                    else None
                )
                tasks.append(item)
        labeled = int(label_summary["labeled_count"] or 0)
        positive = int(label_summary["positive_count"] or 0)
        negative = int(label_summary["negative_count"] or 0)
        return {
            "tasks": tasks,
            "eligible_tasks": eligible_count,
            "labeled_tasks": labeled,
            "unlabeled_tasks": max(0, eligible_count - labeled),
            "label_coverage": round(labeled / eligible_count, 6) if eligible_count else 0.0,
            "positive_count": positive,
            "negative_count": negative,
            "observed_helpfulness": round(positive / max(1, positive + negative), 6) if labeled else None,
            "evaluation_case_count": evaluation_case_count,
            "evaluation_min_cases": 8,
            "metacognition_gate": self.metacognition_enforcement_gate(),
            "definitions": {
                "observed_helpfulness": "Helpful or validated labels divided by all explicit positive and negative task labels.",
                "label_coverage": "Used recall tasks with an explicit operator outcome divided by all used recall tasks.",
                "causal_status": "Outcome labels are observational. Only paired randomized conditions support causal claims.",
            },
        }

    def active_evaluation_cases(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT c.case_id,c.task_id,c.query,c.relevant_memory_ids,c.task_type,c.created_at,c.updated_at
                   FROM evaluation_cases c
                   JOIN task_outcome_labels l ON l.label_id=c.source_label_id AND l.active=1
                   WHERE c.active=1 ORDER BY c.created_at"""
            ).fetchall()
        cases: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                ids = json.loads(str(item.pop("relevant_memory_ids") or "[]"))
            except json.JSONDecodeError:
                ids = []
            item["relevant_memory_ids"] = [str(value) for value in ids if isinstance(value, str)]
            if item["query"] and item["relevant_memory_ids"]:
                cases.append(item)
        return cases

    def begin_evaluation_run(self, *, suite: str, suite_version: str, case_count: int) -> str:
        run_id = str(uuid.uuid4())
        with self._lock:
            self._conn.execute(
                """INSERT INTO evaluation_runs(
                     run_id,suite,suite_version,status,phase,progress,message,case_count,started_at
                   ) VALUES(?,?,?,'running','snapshot',1,'Opening the private evaluation set.',?,?)""",
                (run_id, suite, suite_version, max(1, int(case_count)), utc_now()),
            )
            self._conn.commit()
        return run_id

    def update_evaluation_progress(
        self,
        run_id: str,
        *,
        phase: str,
        progress: int,
        message: str,
    ) -> bool:
        with self._lock:
            result = self._conn.execute(
                """UPDATE evaluation_runs SET phase=?,progress=?,message=?
                   WHERE run_id=? AND status='running'""",
                (str(phase)[:64], max(0, min(99, int(progress))), str(message)[:300], run_id),
            )
            self._conn.commit()
        return bool(result.rowcount)

    def complete_evaluation_run(self, run_id: str, report: dict[str, Any]) -> bool:
        adaptive = report.get("conditions", {}).get("adaptive", {}).get("summary", {})
        fixed = report.get("conditions", {}).get("fixed", {}).get("summary", {})
        delta = report.get("adaptive_minus_fixed", {})
        with self._lock:
            result = self._conn.execute(
                """UPDATE evaluation_runs SET
                     status='completed',phase='completed',progress=100,message=?,
                     adaptive_hit_at_k=?,adaptive_recall_at_k=?,adaptive_mrr=?,
                     fixed_hit_at_k=?,fixed_recall_at_k=?,fixed_mrr=?,
                     context_delta_p50=?,latency_delta_p95=?,result_json=?,error=NULL,completed_at=?
                   WHERE run_id=? AND status='running'""",
                (
                    f"Private evaluation complete across {int(report.get('case_count') or 0)} labeled cases.",
                    adaptive.get("hit_at_k"), adaptive.get("mean_recall_at_k"), adaptive.get("mrr"),
                    fixed.get("hit_at_k"), fixed.get("mean_recall_at_k"), fixed.get("mrr"),
                    delta.get("context_tokens_p50"), delta.get("retrieval_p95_ms"),
                    json.dumps(report, sort_keys=True), utc_now(), run_id,
                ),
            )
            self._conn.commit()
        return bool(result.rowcount)

    def fail_evaluation_run(self, run_id: str, error: str) -> bool:
        with self._lock:
            result = self._conn.execute(
                """UPDATE evaluation_runs SET status='failed',phase='failed',progress=100,
                       message='Private evaluation stopped before completion.',error=?,completed_at=?
                   WHERE run_id=? AND status='running'""",
                (str(error)[:500], utc_now(), run_id),
            )
            self._conn.commit()
        return bool(result.rowcount)

    def abandon_active_evaluations(self) -> int:
        with self._lock:
            result = self._conn.execute(
                """UPDATE evaluation_runs SET status='failed',phase='interrupted',progress=100,
                       message='Dashboard restarted before this run completed.',
                       error='interrupted by dashboard restart',completed_at=?
                   WHERE status='running'""",
                (utc_now(),),
            )
            self._conn.commit()
        return int(result.rowcount)

    def evaluation_snapshot(self, *, limit: int = 40) -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT run_id,suite,suite_version,status,phase,progress,message,case_count,
                          adaptive_hit_at_k,adaptive_recall_at_k,adaptive_mrr,
                          fixed_hit_at_k,fixed_recall_at_k,fixed_mrr,
                          context_delta_p50,latency_delta_p95,result_json,error,started_at,completed_at
                   FROM evaluation_runs ORDER BY started_at DESC LIMIT ?""",
                (max(1, min(int(limit), 100)),),
            ).fetchall()
            case_count = int(
                self._conn.execute("SELECT COUNT(*) count FROM evaluation_cases WHERE active=1").fetchone()["count"]
            )
        runs: list[dict[str, Any]] = []
        latest_result: dict[str, Any] = {}
        for row in rows:
            item = dict(row)
            raw = item.pop("result_json", "{}")
            if not latest_result and item.get("status") == "completed":
                try:
                    parsed = json.loads(str(raw or "{}"))
                except json.JSONDecodeError:
                    parsed = {}
                latest_result = parsed if isinstance(parsed, dict) else {}
            runs.append(item)
        completed = [row for row in runs if row.get("status") == "completed"]
        return {
            "case_count": case_count,
            "minimum_cases": 8,
            "ready": case_count >= 8,
            "runs": runs,
            "active": next((row for row in runs if row.get("status") == "running"), None),
            "latest": completed[0] if completed else None,
            "latest_result": latest_result,
            "completed_count": len(completed),
            "claim_boundary": "This is private retrieval evaluation, not complete-agent accuracy or inference speed.",
        }

    def begin_benchmark_run(
        self,
        *,
        suite: str,
        suite_version: str,
        corpus_memories: int,
        queries: int,
    ) -> str:
        """Persist a bounded dashboard benchmark before its worker starts."""

        run_id = str(uuid.uuid4())
        now = utc_now()
        with self._lock:
            self._conn.execute(
                """INSERT INTO benchmark_runs(
                       run_id,suite,suite_version,status,phase,progress,message,
                       corpus_memories,queries,started_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    suite,
                    suite_version,
                    "running",
                    "preparing",
                    1,
                    "Opening the fixed local retrieval suite.",
                    max(1, int(corpus_memories)),
                    max(1, int(queries)),
                    now,
                ),
            )
            self._conn.commit()
        return run_id

    def update_benchmark_progress(
        self,
        run_id: str,
        *,
        phase: str,
        progress: int,
        message: str,
    ) -> bool:
        with self._lock:
            result = self._conn.execute(
                """UPDATE benchmark_runs
                   SET status='running',phase=?,progress=?,message=?
                   WHERE run_id=? AND status='running'""",
                (str(phase)[:64], max(0, min(99, int(progress))), str(message)[:300], run_id),
            )
            self._conn.commit()
        return bool(result.rowcount)

    def complete_benchmark_run(self, run_id: str, report: dict[str, Any]) -> bool:
        summary = report.get("dashboard_summary") or {}
        with self._lock:
            result = self._conn.execute(
                """UPDATE benchmark_runs SET
                     status='completed',phase='completed',progress=100,message=?,
                     score=?,quality_score=?,speed_score=?,recall_at_k=?,mrr=?,precision_at_k=?,
                     p50_ms=?,p95_ms=?,context_tokens_p50=?,default_coverage=?,
                     corpus_memories=?,queries=?,result_json=?,error=NULL,completed_at=?
                   WHERE run_id=? AND status='running'""",
                (
                    f"Benchmark complete. Cortex scored {float(summary.get('score') or 0.0):.1f} out of 100.",
                    summary.get("score"),
                    summary.get("quality_score"),
                    summary.get("speed_score"),
                    summary.get("recall_at_k"),
                    summary.get("mrr"),
                    summary.get("precision_at_k"),
                    summary.get("p50_ms"),
                    summary.get("p95_ms"),
                    summary.get("context_tokens_p50"),
                    summary.get("default_coverage"),
                    max(1, int(summary.get("corpus_memories") or 1)),
                    max(1, int(summary.get("queries") or 1)),
                    json.dumps(report, sort_keys=True),
                    utc_now(),
                    run_id,
                ),
            )
            self._conn.commit()
        return bool(result.rowcount)

    def fail_benchmark_run(self, run_id: str, error: str) -> bool:
        with self._lock:
            result = self._conn.execute(
                """UPDATE benchmark_runs
                   SET status='failed',phase='failed',progress=100,
                       message='Benchmark stopped before completion.',error=?,completed_at=?
                   WHERE run_id=? AND status='running'""",
                (str(error)[:500], utc_now(), run_id),
            )
            self._conn.commit()
        return bool(result.rowcount)

    def abandon_active_benchmarks(self) -> int:
        """Close benchmark rows left running by an interrupted dashboard."""

        with self._lock:
            result = self._conn.execute(
                """UPDATE benchmark_runs
                   SET status='failed',phase='interrupted',progress=100,
                       message='Dashboard restarted before this run completed.',
                       error='interrupted by dashboard restart',completed_at=?
                   WHERE status='running'""",
                (utc_now(),),
            )
            self._conn.commit()
        return int(result.rowcount)

    def benchmark_snapshot(self, *, limit: int = 40) -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT run_id,suite,suite_version,status,phase,progress,message,score,
                          quality_score,speed_score,recall_at_k,mrr,precision_at_k,p50_ms,p95_ms,
                          context_tokens_p50,default_coverage,corpus_memories,queries,
                          result_json,error,started_at,completed_at
                   FROM benchmark_runs ORDER BY started_at DESC LIMIT ?""",
                (max(1, min(int(limit), 100)),),
            ).fetchall()
        runs: list[dict[str, Any]] = []
        latest_result: dict[str, Any] = {}
        for row in rows:
            item = dict(row)
            raw_result = item.pop("result_json", "{}")
            if not latest_result and item.get("status") == "completed":
                try:
                    parsed = json.loads(str(raw_result or "{}"))
                except json.JSONDecodeError:
                    parsed = {}
                latest_result = parsed if isinstance(parsed, dict) else {}
            runs.append(item)
        active = next((row for row in runs if row.get("status") == "running"), None)
        completed = [row for row in runs if row.get("status") == "completed"]
        latest = completed[0] if completed else None
        previous = None
        if latest:
            previous = next(
                (
                    row
                    for row in completed[1:]
                    if row.get("suite") == latest.get("suite")
                    and row.get("suite_version") == latest.get("suite_version")
                ),
                None,
            )
        delta = None
        if latest and previous:
            delta = {
                "score": round(float(latest.get("score") or 0.0) - float(previous.get("score") or 0.0), 3),
                "recall_at_k": round(
                    float(latest.get("recall_at_k") or 0.0) - float(previous.get("recall_at_k") or 0.0),
                    6,
                ),
                "mrr": round(float(latest.get("mrr") or 0.0) - float(previous.get("mrr") or 0.0), 6),
                "p95_ms": round(float(latest.get("p95_ms") or 0.0) - float(previous.get("p95_ms") or 0.0), 3),
            }
        return {
            "runs": runs,
            "active": active,
            "latest": latest,
            "previous": previous,
            "delta": delta,
            "latest_result": latest_result,
            "completed_count": len(completed),
        }

    def tool_evaluation_snapshot(self, *, limit: int = 80) -> dict[str, Any]:
        """Summarize whether reinforced guidance was followed and what happened next."""

        with self._lock:
            aggregate = self._conn.execute(
                """SELECT COUNT(*) exposures,
                          SUM(CASE WHEN resolved_at IS NOT NULL THEN 1 ELSE 0 END) resolved,
                          SUM(CASE WHEN followed=1 THEN 1 ELSE 0 END) followed,
                          SUM(CASE WHEN followed=0 THEN 1 ELSE 0 END) not_followed,
                          SUM(CASE WHEN followed=1 AND success=1 THEN 1 ELSE 0 END) followed_success,
                          SUM(CASE WHEN followed=1 AND success=0 THEN 1 ELSE 0 END) followed_failure,
                          SUM(CASE WHEN task_outcome IN ('helpful','validated') THEN 1 ELSE 0 END) positive_outcomes,
                          SUM(CASE WHEN task_outcome IN ('harmful','corrected') THEN 1 ELSE 0 END) negative_outcomes,
                          AVG(predicted_reliability) avg_predicted_reliability
                   FROM tool_guidance_exposures"""
            ).fetchone()
            by_type = self._conn.execute(
                """SELECT guidance_type,COUNT(*) exposures,
                          SUM(CASE WHEN followed=1 THEN 1 ELSE 0 END) followed,
                          SUM(CASE WHEN followed=1 AND success=1 THEN 1 ELSE 0 END) followed_success,
                          AVG(predicted_reliability) avg_predicted_reliability
                   FROM tool_guidance_exposures GROUP BY guidance_type ORDER BY guidance_type"""
            ).fetchall()
            recent = self._conn.execute(
                """SELECT exposure_id,task_id,task_type,guidance_type,guidance_key,recommended_tool,
                          predicted_reliability,followed,success,task_outcome,created_at,resolved_at
                   FROM tool_guidance_exposures ORDER BY created_at DESC LIMIT ?""",
                (max(1, min(int(limit), 250)),),
            ).fetchall()
            execution = self._conn.execute(
                "SELECT COUNT(*) count,SUM(success) successes FROM tool_executions"
            ).fetchone()
            workflow = self._conn.execute(
                """SELECT COUNT(*) strategies,
                          SUM(CASE WHEN success_count+failure_count>=4 AND distinct_tasks>=3 THEN 1 ELSE 0 END) evaluable
                   FROM tool_workflow_stats"""
            ).fetchone()
        exposures = int(aggregate["exposures"] or 0)
        resolved = int(aggregate["resolved"] or 0)
        followed = int(aggregate["followed"] or 0)
        followed_success = int(aggregate["followed_success"] or 0)
        return {
            "exposures": exposures,
            "resolved": resolved,
            "followed": followed,
            "not_followed": int(aggregate["not_followed"] or 0),
            "follow_rate": round(followed / resolved, 6) if resolved else None,
            "followed_success_rate": round(followed_success / followed, 6) if followed else None,
            "followed_success": followed_success,
            "followed_failure": int(aggregate["followed_failure"] or 0),
            "positive_task_outcomes": int(aggregate["positive_outcomes"] or 0),
            "negative_task_outcomes": int(aggregate["negative_outcomes"] or 0),
            "avg_predicted_reliability": (
                round(float(aggregate["avg_predicted_reliability"]), 6)
                if aggregate["avg_predicted_reliability"] is not None else None
            ),
            "tool_executions": int(execution["count"] or 0),
            "tool_execution_success_rate": round(
                int(execution["successes"] or 0) / max(1, int(execution["count"] or 0)), 6
            ),
            "workflow_strategies": int(workflow["strategies"] or 0),
            "evaluable_workflows": int(workflow["evaluable"] or 0),
            "by_type": [dict(row) for row in by_type],
            "recent": [dict(row) for row in recent],
            "ready_for_comparison": followed >= 20,
            "claim_boundary": (
                "Follow-through is observational: it shows whether reinforced guidance preceded a matching tool or "
                "workflow and its recorded result. It does not prove that guidance caused the outcome."
            ),
        }

    def evidence_hierarchy_snapshot(self, *, limit: int = 24) -> dict[str, Any]:
        """Build the hierarchy of raw evidence, supported claims, and reviewed bundles."""

        bounded = max(1, min(int(limit), 80))
        with self._lock:
            supported_rows = self._conn.execute(
                """SELECT m.id,m.kind,m.content,m.confidence,m.currentness_confidence,m.dirty,
                          COUNT(d.evidence_id) evidence_count
                   FROM memories m JOIN memory_dependencies d ON d.memory_id=m.id AND d.active=1
                   WHERE m.state IN ('active','cold')
                   GROUP BY m.id HAVING COUNT(d.evidence_id)>0
                   ORDER BY m.dirty ASC,COUNT(d.evidence_id) DESC,m.updated_at DESC LIMIT ?""",
                (bounded,),
            ).fetchall()
            structured_rows = self._conn.execute(
                """SELECT subject,predicate,COUNT(*) member_count,AVG(confidence) avg_confidence,
                          SUM(CASE WHEN dirty=1 THEN 1 ELSE 0 END) dirty_count
                   FROM memories
                   WHERE state IN ('active','cold') AND subject IS NOT NULL AND subject<>''
                   GROUP BY subject,predicate HAVING COUNT(*)>=2
                   ORDER BY COUNT(*) DESC,subject LIMIT ?""",
                (bounded,),
            ).fetchall()
            source_rows = self._conn.execute(
                """SELECT source_category,kind,COUNT(*) member_count,AVG(confidence) avg_confidence
                   FROM memories WHERE state IN ('active','cold')
                   GROUP BY source_category,kind HAVING COUNT(*)>=3
                   ORDER BY COUNT(*) DESC LIMIT ?""",
                (bounded,),
            ).fetchall()
            total_memories = int(
                self._conn.execute("SELECT COUNT(*) count FROM memories WHERE state IN ('active','cold')").fetchone()["count"]
            )
            dependency_count = int(
                self._conn.execute("SELECT COUNT(*) count FROM memory_dependencies WHERE active=1").fetchone()["count"]
            )
            summary_candidate_count = int(
                self._conn.execute("SELECT COUNT(*) count FROM summary_candidates").fetchone()["count"]
            )
            supported: list[dict[str, Any]] = []
            for row in supported_rows:
                item = dict(row)
                evidence = self._conn.execute(
                    """SELECT e.id,e.kind,e.state,e.content,e.source_category,d.relation,d.weight
                       FROM memory_dependencies d JOIN memories e ON e.id=d.evidence_id
                       WHERE d.memory_id=? AND d.active=1 ORDER BY d.weight DESC,e.updated_at DESC LIMIT 8""",
                    (row["id"],),
                ).fetchall()
                item["evidence"] = [dict(record) for record in evidence]
                item["status"] = "needs_repair" if item["dirty"] else "supported"
                supported.append(item)
        return {
            "raw_evidence_count": total_memories,
            "dependency_count": dependency_count,
            "supported_claims": supported,
            "structured_bundles": [dict(row) for row in structured_rows],
            "source_bundles": [dict(row) for row in source_rows],
            "summary_candidates": summary_candidate_count,
            "levels": [
                {"level": "raw", "label": "Raw episodes and observations", "mutable": False},
                {"level": "claim", "label": "Claims with explicit evidence links", "mutable": True},
                {"level": "bundle", "label": "Cited summaries after operator approval", "mutable": True},
            ],
            "claim_boundary": (
                "Candidates are non-recallable until an operator approves them. Every approved summary keeps "
                "dependency links to its cited raw evidence, and no summary is written automatically."
            ),
        }

    def sleep_hypotheses_snapshot(self, *, limit: int = 300) -> dict[str, list[dict[str, Any]]]:
        """Translate Sleep proposals into testable post-run observational hypotheses."""

        with self._lock:
            rows = self._conn.execute(
                """SELECT p.proposal_id,p.run_id,p.kind,p.src_id,p.dst_id,p.score,p.evidence_count,
                          p.rationale,p.created_at,r.mode,r.completed_at,
                          (SELECT COUNT(DISTINCT u.task_id) FROM usage_records u
                           WHERE u.used=1 AND u.memory_id IN (p.src_id,p.dst_id)
                             AND u.created_at>=COALESCE(r.completed_at,r.started_at)) post_use_tasks,
                          (SELECT COUNT(DISTINCT u.task_id) FROM usage_records u
                           WHERE u.used=1 AND u.outcome IN ('helpful','validated')
                             AND u.memory_id IN (p.src_id,p.dst_id)
                             AND COALESCE(u.resolved_at,u.created_at)>=COALESCE(r.completed_at,r.started_at)) positive_tasks,
                          (SELECT COUNT(DISTINCT u.task_id) FROM usage_records u
                           WHERE u.used=1 AND u.outcome IN ('harmful','corrected')
                             AND u.memory_id IN (p.src_id,p.dst_id)
                             AND COALESCE(u.resolved_at,u.created_at)>=COALESCE(r.completed_at,r.started_at)) negative_tasks,
                          CASE WHEN EXISTS(
                            SELECT 1 FROM sleep_edge_changes c WHERE c.run_id=p.run_id
                              AND c.src_id=p.src_id AND c.dst_id=p.dst_id
                          ) OR EXISTS(
                            SELECT 1 FROM sleep_state_changes s WHERE s.run_id=p.run_id
                              AND s.memory_id=p.src_id
                          ) THEN 1 ELSE 0 END applied
                   FROM sleep_proposals p JOIN sleep_runs r ON r.run_id=p.run_id
                   ORDER BY p.created_at DESC,p.score DESC LIMIT ?""",
                (max(1, min(int(limit), 1000)),),
            ).fetchall()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            item = dict(row)
            applied = bool(item["applied"])
            observations = int(item["post_use_tasks"] or 0)
            item["hypothesis"] = _sleep_hypothesis_text(str(item["kind"] or "proposal"))
            item["success_metric"] = "Helpful or validated future use without harmful/corrected use."
            item["causal_status"] = (
                "observational_after_apply" if applied else "proposal_only_no_exposure"
            )
            item["evidence_ready"] = applied and observations >= 8
            grouped.setdefault(str(item["run_id"]), []).append(item)
        return grouped

    def stats(self) -> dict[str, Any]:
        with self._lock:
            state_rows = self._conn.execute("SELECT state,COUNT(*) AS n FROM memories GROUP BY state").fetchall()
            kind_rows = self._conn.execute("SELECT kind,COUNT(*) AS n FROM memories GROUP BY kind").fetchall()
            role_rows = self._conn.execute(
                "SELECT record_role,COUNT(*) AS n FROM memories GROUP BY record_role"
            ).fetchall()
            counts = self._conn.execute(
                "SELECT (SELECT COUNT(*) FROM memories) memories, (SELECT COUNT(*) FROM edges) edges, "
                "(SELECT COUNT(*) FROM edge_evidence) edge_evidence, "
                "(SELECT COUNT(*) FROM memories WHERE context_mode='context_dependent') context_dependent_memories, "
                "(SELECT COUNT(*) FROM memories WHERE context_mode='standalone') standalone_memories, "
                "(SELECT COUNT(*) FROM episodes) episodes, (SELECT COUNT(*) FROM access_log) accesses, "
                "(SELECT COUNT(*) FROM usage_records WHERE outcome='pending') pending_usage, "
                "(SELECT COUNT(*) FROM memories WHERE dirty=1) dirty_memories, "
                "(SELECT COUNT(*) FROM tool_executions) tool_executions, "
                "(SELECT COUNT(*) FROM tool_stats) tool_strategies, "
                "(SELECT COUNT(*) FROM tool_workflows) tool_workflows, "
                "(SELECT COUNT(*) FROM recall_runs) recall_runs, "
                "(SELECT COUNT(*) FROM memory_traces) memory_traces, "
                "(SELECT COUNT(*) FROM memory_trace_events) memory_trace_events, "
                "(SELECT COUNT(*) FROM memory_write_decisions) memory_write_decisions, "
                "(SELECT COUNT(*) FROM memory_creation_proposals) memory_creation_proposals, "
                "(SELECT COUNT(*) FROM memory_creation_proposals WHERE status IN ('pending','needs_context')) pending_memory_creation_proposals, "
                "(SELECT COUNT(*) FROM memory_context_outcomes) memory_context_outcomes, "
                "(SELECT COUNT(*) FROM memory_context_terms) memory_context_terms, "
                "(SELECT COUNT(*) FROM recall_budget_observations WHERE outcome<>'pending') budget_observations, "
                "(SELECT COUNT(*) FROM metacognitive_predictions) metacognitive_predictions, "
                "(SELECT COUNT(*) FROM metacognitive_predictions "
                " WHERE outcome IN ('helpful','validated','harmful','corrected')) metacognitive_labels, "
                "(SELECT COUNT(*) FROM task_outcome_labels WHERE active=1) task_outcome_labels, "
                "(SELECT COUNT(*) FROM evaluation_cases WHERE active=1) evaluation_cases, "
                "(SELECT COUNT(*) FROM evaluation_runs WHERE status='completed') evaluation_runs, "
                "(SELECT COUNT(*) FROM tool_guidance_exposures) tool_guidance_exposures, "
                "(SELECT COUNT(*) FROM benchmark_runs WHERE status='completed') benchmark_runs, "
                "(SELECT COUNT(*) FROM pruning_regret) pruning_regrets, "
                "(SELECT COUNT(*) FROM consolidation_runs WHERE dry_run=0) consolidations, "
                "(SELECT COUNT(*) FROM sleep_runs) sleep_runs, "
                "(SELECT COUNT(*) FROM sleep_proposals WHERE status='proposed') sleep_proposals, "
                "(SELECT COUNT(*) FROM agent_task_observations) agent_tasks, "
                "(SELECT COUNT(*) FROM controlled_experiments WHERE status='active') active_experiments, "
                "(SELECT COUNT(*) FROM sleep_trials) sleep_trials, "
                "(SELECT COUNT(*) FROM summary_candidates WHERE status='proposed') summary_candidates, "
                "(SELECT COUNT(*) FROM prospective_items WHERE status='open') prospective_open, "
                "(SELECT COUNT(*) FROM reconsolidation_events) reconsolidation_events, "
                "(SELECT COUNT(*) FROM document_sources WHERE status='active') documents, "
                "(SELECT COUNT(*) FROM document_chunks WHERE active=1) document_chunks, "
                "(SELECT COUNT(*) FROM memory_presentations) memory_presentations, "
                "(SELECT COUNT(*) FROM memory_refinery_proposals) refinery_proposals, "
                "(SELECT COUNT(*) FROM memory_recall_sets) recall_sets, "
                "(SELECT COUNT(*) FROM memory_recall_memberships WHERE revoked_at IS NULL) recall_memberships, "
                "(SELECT COUNT(*) FROM memory_neighborhoods) neighborhoods, "
                "(SELECT COUNT(*) FROM memory_neighborhood_evaluations) neighborhood_evaluations, "
                "(SELECT COUNT(*) FROM memory_neighborhood_decision_events) neighborhood_decision_events, "
                "(SELECT COUNT(*) FROM memory_experience_events) memory_experience_events, "
                "(SELECT COUNT(*) FROM memory_recall_memberships rm "
                " JOIN memory_recall_sets rs ON rs.recall_set_id=rm.recall_set_id "
                " WHERE rs.status='active' AND rm.revoked_at IS NULL "
                " AND rm.eligibility='primary') active_recallable_memories"
            ).fetchone()
        return {
            **dict(counts),
            "states": {r["state"]: r["n"] for r in state_rows},
            "kinds": {r["kind"]: r["n"] for r in kind_rows},
            "roles": {r["record_role"]: r["n"] for r in role_rows},
            "db_path": str(self.path),
            "db_bytes": self.path.stat().st_size if self.path.exists() else 0,
            "schema_version": SCHEMA_VERSION,
        }

    def _operator_policy_contributions_tx(
        self, conn: sqlite3.Connection
    ) -> list[dict[str, Any]]:
        """Translate reversible operator decisions into bounded policy evidence."""

        rows = conn.execute(
            """SELECT d.*,p.kind proposal_kind,
                      c.kind creation_kind,c.source_type creation_source_type,
                      c.source_category creation_source_category,
                      c.candidate_json creation_candidate_json,
                      c.assessment_json creation_assessment_json,
                      c.session_id creation_session_id,c.source_ref creation_source_ref
               FROM operator_review_decisions d
               LEFT JOIN sleep_proposals p ON p.proposal_id=d.proposal_id
               LEFT JOIN memory_creation_proposals c
                 ON d.item_key=('creation:' || c.proposal_id)
               WHERE d.reversed_at IS NULL AND d.decision_scope='policy_evidence'
               ORDER BY d.created_at"""
        ).fetchall()
        prepared: list[tuple[dict[str, Any], set[str]]] = []
        all_memory_ids: set[str] = set()
        for row in rows:
            item = dict(row)
            effect = _trace_json_object(item.get("effect_json"))
            memory_ids = {
                str(value)
                for value in (item.get("src_id"), item.get("dst_id"))
                if value
            }
            memory_ids.update(str(value) for value in effect.get("memory_ids", []) if value)
            all_memory_ids.update(memory_ids)
            prepared.append((item, memory_ids))
        memory_cache: dict[str, dict[str, Any]] = {}
        ordered_ids = sorted(all_memory_ids)
        for start in range(0, len(ordered_ids), 800):
            batch = ordered_ids[start : start + 800]
            placeholders = ",".join("?" for _ in batch)
            for memory in conn.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders})", tuple(batch)
            ).fetchall():
                memory_cache[str(memory["id"])] = dict(memory)
        contributions: list[dict[str, Any]] = []
        for item, memory_ids in prepared:
            action = str(item.get("action") or "").casefold()
            reason = str(item.get("reason_code") or "unspecified").casefold()
            item_type = str(item.get("item_type") or "").casefold()
            proposal_kind = str(item.get("proposal_kind") or "").casefold()
            memories = [memory_cache[memory_id] for memory_id in sorted(memory_ids) if memory_id in memory_cache]

            if item_type == "creation" and action == "reject" and item.get("creation_kind"):
                assessment = _trace_json_object(item.get("creation_assessment_json"))
                raw_quality_flags = assessment.get("quality_flags")
                quality_flags = (
                    list(raw_quality_flags) if isinstance(raw_quality_flags, list) else []
                )
                quality_flag = str(quality_flags[0]) if quality_flags else "operator_rejected_candidate"
                candidate = _trace_json_object(item.get("creation_candidate_json"))
                context_memory = {
                    "kind": item.get("creation_kind"),
                    "source_type": item.get("creation_source_type"),
                    "source_ref": item.get("creation_source_ref"),
                    "session_id": item.get("creation_session_id"),
                    "scope_json": _trace_json(candidate.get("scope") or {}),
                    "applicable_systems_json": _trace_json(
                        candidate.get("applicable_systems") or []
                    ),
                }
                contributions.append(
                    _policy_contribution(
                        review_id=str(item["review_id"]),
                        created_at=str(item["created_at"]),
                        domain="admission",
                        lever="automatic_write_gate",
                        selector={
                            "kind": str(item.get("creation_kind") or "semantic"),
                            "source_type": str(item.get("creation_source_type") or "conversation"),
                            "quality_flag": quality_flag,
                        },
                        direction="ignore",
                        context_key=_policy_context_key(
                            context_memory, fallback=str(item["review_id"])
                        ),
                        reason=reason,
                    )
                )

            if proposal_kind in {"association", "association_reinforcement"} and len(memories) >= 2:
                selector = connection_policy_selector(memories[0], memories[1])
                contributions.append(
                    _policy_contribution(
                        review_id=str(item["review_id"]),
                        created_at=str(item["created_at"]),
                        domain="connection",
                        lever="independent_witness_requirement",
                        selector=selector,
                        direction="stricter" if action == "deny" else "baseline",
                        context_key=_policy_context_key(memories[0], fallback=str(item["review_id"])),
                        reason=reason,
                    )
                )

            for memory in memories:
                kind = str(memory.get("kind") or "semantic")
                source_type = str(memory.get("source_type") or "conversation")
                source_category = str(memory.get("source_category") or "AGENT_INFERENCE")
                context_key = _policy_context_key(memory, fallback=str(item["review_id"]))

                retrieval_direction: str | None = None
                if item_type == "outcome":
                    retrieval_direction = "boost" if action in {"helpful", "validated"} else "downrank"
                elif item_type == "inference":
                    retrieval_direction = "boost" if action == "confirm" else "downrank"
                if retrieval_direction:
                    contributions.append(
                        _policy_contribution(
                            review_id=str(item["review_id"]),
                            created_at=str(item["created_at"]),
                            domain="retrieval",
                            lever="operator_rank_adjustment",
                            selector={"kind": kind, "source_category": source_category},
                            direction=retrieval_direction,
                            context_key=context_key,
                            reason=reason,
                        )
                    )

                if item_type == "proposal" and proposal_kind not in {
                    "association",
                    "association_reinforcement",
                    "edge_downscale",
                    "interference_review",
                }:
                    if action in {"keep", "confirm"}:
                        retention_direction = "preserve"
                    elif action in {"archive", "trash", "quarantine"}:
                        retention_direction = "cool_faster"
                    else:
                        retention_direction = None
                    if retention_direction:
                        contributions.append(
                            _policy_contribution(
                                review_id=str(item["review_id"]),
                                created_at=str(item["created_at"]),
                                domain="retention",
                                lever="operator_retention_adjustment",
                                selector={"kind": kind, "source_type": source_type},
                                direction=retention_direction,
                                context_key=context_key,
                                reason=reason,
                            )
                        )

                admission_flags = {
                    "transient": "transient_automation_status",
                    "duplicate": "duplicate_candidate",
                    "not_durable": "low_durability",
                    "unsupported_guess": "inferred_without_source_context",
                }
                quality_flag = admission_flags.get(reason)
                if quality_flag and item_type in {"proposal", "inference"}:
                    contributions.append(
                        _policy_contribution(
                            review_id=str(item["review_id"]),
                            created_at=str(item["created_at"]),
                            domain="admission",
                            lever="automatic_write_gate",
                            selector={
                                "kind": kind,
                                "source_type": source_type,
                                "quality_flag": quality_flag,
                            },
                            direction=(
                                "ignore"
                                if action in {"archive", "trash", "quarantine"}
                                else "baseline"
                            ),
                            context_key=context_key,
                            reason=reason,
                        )
                    )
        return contributions

    def _compile_policy_candidates_tx(self, conn: sqlite3.Connection) -> dict[str, int]:
        contributions = self._operator_policy_contributions_tx(conn)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in contributions:
            grouped.setdefault(str(item["signal_key"]), []).append(item)
        created = 0
        updated = 0
        now = utc_now()
        for signal_key, rows in grouped.items():
            exemplar = rows[0]
            directions: dict[str, list[dict[str, Any]]] = {}
            for item in rows:
                directions.setdefault(str(item["direction"]), []).append(item)
            direction, support_rows = max(
                directions.items(), key=lambda pair: (len({row["review_id"] for row in pair[1]}), pair[0])
            )
            support_ids = sorted({str(row["review_id"]) for row in support_rows})
            opposing_rows = [row for value, items in directions.items() if value != direction for row in items]
            oppose_ids = sorted({str(row["review_id"]) for row in opposing_rows})
            support_count = len(support_ids)
            oppose_count = len(oppose_ids)
            consistency = support_count / max(1, support_count + oppose_count)
            contexts = sorted({str(row["context_key"]) for row in support_rows})
            domain = str(exemplar["domain"])
            if domain in {"connection", "admission"} and direction == "baseline":
                continue
            selector = dict(exemplar["selector"])
            title, explanation, change = _policy_candidate_copy(domain, direction, selector)
            existing = conn.execute(
                "SELECT * FROM policy_candidates WHERE signal_key=?", (signal_key,)
            ).fetchone()
            evidence = {
                "supporting_review_ids": support_ids[-100:],
                "opposing_review_ids": oppose_ids[-100:],
                "contexts": contexts[-30:],
                "reasons": sorted({str(row["reason"]) for row in support_rows}),
            }
            base_stage = (
                "replay_ready"
                if support_count >= POLICY_MIN_SUPPORT and consistency >= POLICY_MIN_CONSISTENCY
                else "collecting"
            )
            if existing:
                existing_stage = str(existing["stage"])
                existing_direction = str(existing["direction"])
                replay = _trace_json_object(existing["replay_json"])
                shadow = _trace_json_object(existing["shadow_json"])
                stage = existing_stage
                if existing_direction != direction and existing_stage not in {"promoted", "rejected"}:
                    stage, replay, shadow = base_stage, {}, {}
                elif existing_stage in {"collecting", "replay_ready"}:
                    stage = base_stage
                elif existing_stage == "shadow":
                    started_at = str(shadow.get("started_at") or "")
                    baseline_ids = {str(value) for value in shadow.get("baseline_review_ids", [])}
                    shadow_rows = [
                        row
                        for row in rows
                        if str(row["review_id"]) not in baseline_ids
                        and (not started_at or str(row["created_at"]) >= started_at)
                    ]
                    shadow_support = len(
                        {str(row["review_id"]) for row in shadow_rows if row["direction"] == direction}
                    )
                    shadow_oppose = len(
                        {str(row["review_id"]) for row in shadow_rows if row["direction"] != direction}
                    )
                    shadow_total = shadow_support + shadow_oppose
                    shadow_consistency = shadow_support / max(1, shadow_total)
                    shadow.update(
                        {
                            "observation_count": shadow_total,
                            "support_count": shadow_support,
                            "oppose_count": shadow_oppose,
                            "consistency": round(shadow_consistency, 6),
                            "required_observations": POLICY_SHADOW_MIN_OBSERVATIONS,
                        }
                    )
                    if shadow_total >= POLICY_SHADOW_MIN_OBSERVATIONS:
                        stage = "ready" if shadow_consistency >= POLICY_MIN_CONSISTENCY else "paused"
                conn.execute(
                    """UPDATE policy_candidates SET domain=?,title=?,explanation=?,selector_json=?,
                         change_json=?,direction=?,support_count=?,oppose_count=?,context_count=?,
                         consistency=?,stage=?,evidence_json=?,replay_json=?,shadow_json=?,updated_at=?
                       WHERE candidate_id=?""",
                    (
                        domain,
                        title,
                        explanation,
                        _trace_json(selector),
                        _trace_json(change),
                        direction,
                        support_count,
                        oppose_count,
                        len(contexts),
                        round(consistency, 6),
                        stage,
                        _trace_json(evidence),
                        _trace_json(replay),
                        _trace_json(shadow),
                        now,
                        existing["candidate_id"],
                    ),
                )
                updated += 1
            else:
                candidate_id = str(uuid.uuid4())
                conn.execute(
                    """INSERT INTO policy_candidates(
                       candidate_id,signal_key,domain,title,explanation,selector_json,change_json,
                       direction,support_count,oppose_count,context_count,consistency,stage,
                       evidence_json,replay_json,shadow_json,created_at,updated_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        candidate_id,
                        signal_key,
                        domain,
                        title,
                        explanation,
                        _trace_json(selector),
                        _trace_json(change),
                        direction,
                        support_count,
                        oppose_count,
                        len(contexts),
                        round(consistency, 6),
                        base_stage,
                        _trace_json(evidence),
                        "{}",
                        "{}",
                        now,
                        now,
                    ),
                )
                conn.execute(
                    """INSERT INTO policy_events(event_id,candidate_id,event_type,details_json,actor,created_at)
                       VALUES(?,?,'compiled',?,'cortex-feedback-compiler',?)""",
                    (
                        str(uuid.uuid4()),
                        candidate_id,
                        _trace_json({"signal_key": signal_key, "support_count": support_count}),
                        now,
                    ),
                )
                created += 1
        return {"mapped_evidence": len(contributions), "created": created, "updated": updated}

    def compile_policy_candidates(self) -> dict[str, Any]:
        with self.transaction() as conn:
            result = self._compile_policy_candidates_tx(conn)
        return {**result, "training": self.policy_training_snapshot()}

    def evaluate_policy_candidate(
        self, candidate_id: str, *, actor: str = "dashboard-operator"
    ) -> dict[str, Any]:
        """Run an inspectable ledger counterfactual before any shadow trial."""

        now = utc_now()
        with self.transaction() as conn:
            candidate = conn.execute(
                "SELECT * FROM policy_candidates WHERE candidate_id=?", (candidate_id,)
            ).fetchone()
            if not candidate:
                raise ValueError("policy candidate not found")
            if int(candidate["support_count"]) < POLICY_MIN_SUPPORT:
                raise ValueError(f"collect at least {POLICY_MIN_SUPPORT} supporting reviews first")
            selector = _trace_json_object(candidate["selector_json"])
            memories = [
                dict(row)
                for row in conn.execute("SELECT * FROM memories").fetchall()
                if _policy_selector_matches(selector, dict(row))
            ]
            if str(candidate["domain"]) == "admission":
                matched_writes = []
                for row in conn.execute("SELECT * FROM memory_write_decisions").fetchall():
                    write = dict(row)
                    write["quality_flags"] = _trace_json_array(write.get("quality_flags_json"))
                    if _policy_selector_matches(selector, write):
                        matched_writes.append(write)
                matched_memory_ids = {
                    str(row["memory_id"]) for row in matched_writes if row.get("memory_id")
                }
                memories = [
                    dict(row)
                    for row in conn.execute("SELECT * FROM memories").fetchall()
                    if str(row["id"]) in matched_memory_ids
                ]
            positive = sum(
                int(row.get("helpful_count", 0)) + int(row.get("validated_count", 0))
                for row in memories
            )
            negative = sum(
                int(row.get("harmful_count", 0))
                + int(row.get("correction_count", 0))
                + int(row.get("false_positive_count", 0))
                for row in memories
            )
            consistency = float(candidate["consistency"])
            passed = consistency >= POLICY_MIN_CONSISTENCY
            projected = len(matched_writes) if str(candidate["domain"]) == "admission" else len(memories)
            domain = str(candidate["domain"])
            if domain == "connection":
                projected = int(candidate["support_count"])
            replay = {
                "method": "operator_evidence_counterfactual_v1",
                "passed": passed,
                "evaluated_at": now,
                "affected_records": projected,
                "historical_positive_outcomes": positive,
                "historical_negative_outcomes": negative,
                "support_count": int(candidate["support_count"]),
                "oppose_count": int(candidate["oppose_count"]),
                "consistency": consistency,
                "summary": (
                    f"The proposed {domain} standard agrees with "
                    f"{consistency:.0%} of matching operator evidence across {projected} affected records."
                ),
                "claim_boundary": (
                    "This replays stored decisions and outcomes. It does not prove whole-answer accuracy; "
                    "new reviews must still agree during a shadow trial."
                ),
            }
            next_stage = "shadow_ready" if passed else "collecting"
            conn.execute(
                "UPDATE policy_candidates SET replay_json=?,stage=?,updated_at=? WHERE candidate_id=?",
                (_trace_json(replay), next_stage, now, candidate_id),
            )
            conn.execute(
                """INSERT INTO policy_events(event_id,candidate_id,event_type,details_json,actor,created_at)
                   VALUES(?,?,'replayed',?,?,?)""",
                (str(uuid.uuid4()), candidate_id, _trace_json(replay), normalize_text(actor)[:80], now),
            )
        return replay

    def start_policy_shadow(
        self, candidate_id: str, *, actor: str = "dashboard-operator"
    ) -> dict[str, Any]:
        now = utc_now()
        shadow = {
            "started_at": now,
            "baseline_review_ids": [],
            "observation_count": 0,
            "support_count": 0,
            "oppose_count": 0,
            "consistency": 0.0,
            "required_observations": POLICY_SHADOW_MIN_OBSERVATIONS,
        }
        with self.transaction() as conn:
            candidate = conn.execute(
                "SELECT stage FROM policy_candidates WHERE candidate_id=?", (candidate_id,)
            ).fetchone()
            if not candidate or str(candidate["stage"]) != "shadow_ready":
                raise ValueError("run and pass the evidence replay before starting shadow observation")
            evidence = _trace_json_object(
                self._conn.execute(
                    "SELECT evidence_json FROM policy_candidates WHERE candidate_id=?", (candidate_id,)
                ).fetchone()["evidence_json"]
            )
            shadow["baseline_review_ids"] = sorted(
                {
                    str(value)
                    for value in evidence.get("supporting_review_ids", []) + evidence.get("opposing_review_ids", [])
                }
            )
            conn.execute(
                "UPDATE policy_candidates SET stage='shadow',shadow_json=?,updated_at=? WHERE candidate_id=?",
                (_trace_json(shadow), now, candidate_id),
            )
            conn.execute(
                """INSERT INTO policy_events(event_id,candidate_id,event_type,details_json,actor,created_at)
                   VALUES(?,?,'shadow_started',?,?,?)""",
                (str(uuid.uuid4()), candidate_id, _trace_json(shadow), normalize_text(actor)[:80], now),
            )
        return shadow

    def _apply_connection_policy_backlog_tx(
        self,
        conn: sqlite3.Connection,
        *,
        version_id: str,
        selector: dict[str, Any],
        change: dict[str, Any],
        now: str,
    ) -> dict[str, Any]:
        """Withdraw pending weak proposals newly covered by an approved connection policy."""

        delta = max(0, int(change.get("min_independent_witnesses_delta") or 0))
        if not delta:
            return {"filtered": 0, "required_witness_delta": 0}
        filtered_ids: list[str] = []
        proposals = conn.execute(
            """SELECT * FROM sleep_proposals
               WHERE status='proposed' AND kind IN ('association','association_reinforcement')
               ORDER BY created_at"""
        ).fetchall()
        for proposal in proposals:
            memories = conn.execute(
                "SELECT * FROM memories WHERE id IN (?,?)",
                (proposal["src_id"], proposal["dst_id"]),
            ).fetchall()
            if len(memories) != 2:
                continue
            features = connection_policy_selector(dict(memories[0]), dict(memories[1]))
            if not _policy_selector_matches(selector, features):
                continue
            details = _trace_json_object(proposal["details_json"])
            witnesses = int(details.get("distinct_witnesses") or 0)
            if not witnesses:
                witnesses = int(
                    conn.execute(
                        """SELECT COUNT(DISTINCT witness_key) count
                           FROM sleep_association_evidence WHERE src_id=? AND dst_id=?""",
                        tuple(sorted((str(proposal["src_id"]), str(proposal["dst_id"])))),
                    ).fetchone()["count"]
                )
            prior_required = max(2, int(details.get("required_witnesses") or 2))
            if witnesses >= prior_required + delta:
                continue
            conn.execute(
                "UPDATE sleep_proposals SET status='policy_filtered' WHERE proposal_id=? AND status='proposed'",
                (proposal["proposal_id"],),
            )
            conn.execute(
                """INSERT INTO policy_proposal_effects(
                   version_id,proposal_id,prior_status,next_status,reason,created_at
                   ) VALUES(?,?,'proposed','policy_filtered',?,?)""",
                (
                    version_id,
                    proposal["proposal_id"],
                    f"approved connection policy requires {prior_required + delta} independent witnesses",
                    now,
                ),
            )
            filtered_ids.append(str(proposal["proposal_id"]))
        return {
            "filtered": len(filtered_ids),
            "required_witness_delta": delta,
            "proposal_ids": filtered_ids,
        }

    def promote_policy_candidate(
        self,
        candidate_id: str,
        *,
        activation_scope: str = "scoped",
        actor: str = "dashboard-operator",
    ) -> dict[str, Any]:
        scope = normalize_text(activation_scope).casefold()
        if scope not in {"scoped", "core"}:
            raise ValueError("activation_scope must be scoped or core")
        now = utc_now()
        version_id = str(uuid.uuid4())
        backlog_effect: dict[str, Any] = {"filtered": 0}
        with self.transaction() as conn:
            candidate = conn.execute(
                "SELECT * FROM policy_candidates WHERE candidate_id=?", (candidate_id,)
            ).fetchone()
            if not candidate or str(candidate["stage"]) != "ready":
                raise ValueError("the candidate must pass replay and shadow evidence before promotion")
            if scope == "core" and (
                int(candidate["support_count"]) < POLICY_CORE_MIN_SUPPORT
                or int(candidate["context_count"]) < POLICY_CORE_MIN_CONTEXTS
            ):
                raise ValueError(
                    f"core promotion needs {POLICY_CORE_MIN_SUPPORT} supporting reviews across "
                    f"{POLICY_CORE_MIN_CONTEXTS} contexts"
                )
            evidence = {
                "candidate_support": int(candidate["support_count"]),
                "candidate_opposition": int(candidate["oppose_count"]),
                "candidate_consistency": float(candidate["consistency"]),
                "replay": _trace_json_object(candidate["replay_json"]),
                "shadow": _trace_json_object(candidate["shadow_json"]),
            }
            selector = _trace_json_object(candidate["selector_json"])
            change = _trace_json_object(candidate["change_json"])
            conn.execute(
                """INSERT INTO policy_versions(
                   version_id,candidate_id,domain,activation_scope,selector_json,change_json,
                   evidence_json,status,activated_by,activated_at
                   ) VALUES(?,?,?,?,?,?,?,'active',?,?)""",
                (
                    version_id,
                    candidate_id,
                    candidate["domain"],
                    scope,
                    candidate["selector_json"],
                    candidate["change_json"],
                    _trace_json(evidence),
                    normalize_text(actor)[:80],
                    now,
                ),
            )
            conn.execute(
                "UPDATE policy_candidates SET stage='promoted',decided_at=?,updated_at=? WHERE candidate_id=?",
                (now, now, candidate_id),
            )
            if str(candidate["domain"]) == "connection":
                backlog_effect = self._apply_connection_policy_backlog_tx(
                    conn,
                    version_id=version_id,
                    selector=selector,
                    change=change,
                    now=now,
                )
                evidence["pending_backlog_effect"] = backlog_effect
                conn.execute(
                    "UPDATE policy_versions SET evidence_json=? WHERE version_id=?",
                    (_trace_json(evidence), version_id),
                )
            conn.execute(
                """INSERT INTO policy_events(event_id,candidate_id,version_id,event_type,details_json,actor,created_at)
                   VALUES(?,?,?,'promoted',?,?,?)""",
                (
                    str(uuid.uuid4()),
                    candidate_id,
                    version_id,
                    _trace_json({"activation_scope": scope, "pending_backlog_effect": backlog_effect}),
                    normalize_text(actor)[:80],
                    now,
                ),
            )
        self._active_policy_cache = None
        self._bump_local_retrieval_revision()
        return {
            "version_id": version_id,
            "candidate_id": candidate_id,
            "activation_scope": scope,
            "pending_backlog_effect": backlog_effect,
        }

    def reject_policy_candidate(
        self, candidate_id: str, *, reason: str = "operator rejected", actor: str = "dashboard-operator"
    ) -> bool:
        now = utc_now()
        with self.transaction() as conn:
            result = conn.execute(
                """UPDATE policy_candidates SET stage='rejected',decided_at=?,updated_at=?
                   WHERE candidate_id=? AND stage<>'promoted'""",
                (now, now, candidate_id),
            )
            if not result.rowcount:
                return False
            conn.execute(
                """INSERT INTO policy_events(event_id,candidate_id,event_type,details_json,actor,created_at)
                   VALUES(?,?,'rejected',?,?,?)""",
                (
                    str(uuid.uuid4()),
                    candidate_id,
                    _trace_json({"reason": normalize_text(reason)[:500]}),
                    normalize_text(actor)[:80],
                    now,
                ),
            )
        return True

    def rollback_policy_version(
        self, version_id: str, *, reason: str, actor: str = "dashboard-operator"
    ) -> bool:
        now = utc_now()
        with self.transaction() as conn:
            version = conn.execute(
                "SELECT * FROM policy_versions WHERE version_id=? AND status='active'", (version_id,)
            ).fetchone()
            if not version:
                return False
            proposal_effects = conn.execute(
                """SELECT * FROM policy_proposal_effects
                   WHERE version_id=? AND reversed_at IS NULL""",
                (version_id,),
            ).fetchall()
            restored_proposals = 0
            for effect in proposal_effects:
                result = conn.execute(
                    """UPDATE sleep_proposals SET status=?
                       WHERE proposal_id=? AND status=?""",
                    (effect["prior_status"], effect["proposal_id"], effect["next_status"]),
                )
                restored_proposals += int(result.rowcount > 0)
            conn.execute(
                "UPDATE policy_proposal_effects SET reversed_at=? WHERE version_id=? AND reversed_at IS NULL",
                (now, version_id),
            )
            conn.execute(
                """UPDATE policy_versions SET status='rolled_back',deactivated_at=?,deactivated_by=?,
                     rollback_reason=? WHERE version_id=?""",
                (now, normalize_text(actor)[:80], normalize_text(reason)[:500], version_id),
            )
            conn.execute(
                "UPDATE policy_candidates SET stage='ready',updated_at=? WHERE candidate_id=?",
                (now, version["candidate_id"]),
            )
            conn.execute(
                """INSERT INTO policy_events(event_id,candidate_id,version_id,event_type,details_json,actor,created_at)
                   VALUES(?,?,?,'rolled_back',?,?,?)""",
                (
                    str(uuid.uuid4()),
                    version["candidate_id"],
                    version_id,
                    _trace_json(
                        {
                            "reason": normalize_text(reason)[:500],
                            "restored_pending_proposals": restored_proposals,
                        }
                    ),
                    normalize_text(actor)[:80],
                    now,
                ),
            )
        self._active_policy_cache = None
        self._bump_local_retrieval_revision()
        return True

    def active_policy_adjustment(self, domain: str, features: dict[str, Any]) -> dict[str, Any]:
        """Return the combined, versioned adjustment for one core policy decision."""

        if self._active_policy_cache is None:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM policy_versions WHERE status='active' ORDER BY activated_at"
                ).fetchall()
            self._active_policy_cache = [dict(row) for row in rows]
        result: dict[str, Any] = {
            "score_adjustment": 0.0,
            "min_independent_witnesses_delta": 0,
            "automatic_action": None,
            "matched_versions": [],
        }
        for row in self._active_policy_cache:
            if str(row.get("domain")) != domain:
                continue
            selector = _trace_json_object(row.get("selector_json"))
            if not _policy_selector_matches(selector, features):
                continue
            change = _trace_json_object(row.get("change_json"))
            result["score_adjustment"] += float(change.get("score_adjustment") or 0.0)
            result["min_independent_witnesses_delta"] = max(
                int(result["min_independent_witnesses_delta"]),
                int(change.get("min_independent_witnesses_delta") or 0),
            )
            if change.get("automatic_action"):
                result["automatic_action"] = str(change["automatic_action"])
            result["matched_versions"].append(str(row["version_id"]))
        result["score_adjustment"] = max(-0.20, min(0.12, float(result["score_adjustment"])))
        return result

    def policy_training_snapshot(self) -> dict[str, Any]:
        with self._lock:
            candidate_rows = self._conn.execute(
                "SELECT * FROM policy_candidates ORDER BY updated_at DESC"
            ).fetchall()
            version_rows = self._conn.execute(
                "SELECT * FROM policy_versions ORDER BY activated_at DESC LIMIT 100"
            ).fetchall()
            event_rows = self._conn.execute(
                "SELECT * FROM policy_events ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
            decision_counts = self._conn.execute(
                """SELECT COUNT(*) total,
                          SUM(CASE WHEN decision_scope='policy_evidence' THEN 1 ELSE 0 END) training,
                          SUM(CASE WHEN decision_scope='item_only' THEN 1 ELSE 0 END) item_only,
                          SUM(CASE WHEN decision_scope='exact_duplicates' THEN 1 ELSE 0 END) exact_duplicates
                   FROM operator_review_decisions WHERE reversed_at IS NULL"""
            ).fetchone()
            decision_count = int(decision_counts["training"] or 0)
            contributions = self._operator_policy_contributions_tx(self._conn)
        candidates: list[dict[str, Any]] = []
        for row in candidate_rows:
            item = dict(row)
            for field in ("selector_json", "change_json", "evidence_json", "replay_json", "shadow_json"):
                item[field.removesuffix("_json")] = _trace_json_object(item.pop(field))
            item["scoped_gate_ready"] = (
                int(item["support_count"]) >= POLICY_MIN_SUPPORT
                and float(item["consistency"]) >= POLICY_MIN_CONSISTENCY
            )
            item["core_gate_ready"] = (
                int(item["support_count"]) >= POLICY_CORE_MIN_SUPPORT
                and int(item["context_count"]) >= POLICY_CORE_MIN_CONTEXTS
                and float(item["consistency"]) >= POLICY_MIN_CONSISTENCY
            )
            candidates.append(item)
        versions: list[dict[str, Any]] = []
        for row in version_rows:
            item = dict(row)
            for field in ("selector_json", "change_json", "evidence_json"):
                item[field.removesuffix("_json")] = _trace_json_object(item.pop(field))
            versions.append(item)
        events: list[dict[str, Any]] = []
        for row in event_rows:
            item = dict(row)
            item["details"] = _trace_json_object(item.pop("details_json"))
            events.append(item)
        active_versions = [row for row in versions if row["status"] == "active"]
        mapped_decisions = len({str(row["review_id"]) for row in contributions})
        ready_for_replay = sum(row["stage"] == "replay_ready" for row in candidates)
        replayed = sum(bool(row["replay"]) for row in candidates)
        shadowing = sum(row["stage"] == "shadow" for row in candidates)
        ready = sum(row["stage"] == "ready" for row in candidates)
        monitored = 0
        if active_versions:
            first_activation = min(str(row["activated_at"]) for row in active_versions)
            with self._lock:
                monitored = int(
                    self._conn.execute(
                        """SELECT COUNT(*) count FROM operator_review_decisions
                           WHERE reversed_at IS NULL AND decision_scope='policy_evidence'
                             AND created_at>?""",
                        (first_activation,),
                    ).fetchone()["count"]
                )
        steps = [
            {
                "key": "review",
                "label": "Review real examples",
                "description": "Choose Teach Kaya only when this decision should inform similar cases.",
                "current": decision_count,
                "target": POLICY_MIN_SUPPORT,
            },
            {
                "key": "compile",
                "label": "Form a proposed standard",
                "description": "Five consistent decisions can become one scoped, explainable policy candidate.",
                "current": sum(row["scoped_gate_ready"] for row in candidates),
                "target": 1,
            },
            {
                "key": "test",
                "label": "Replay and observe in shadow",
                "description": "Test stored evidence, then collect three new matching decisions without changing Kaya.",
                "current": max(replayed, shadowing, ready),
                "target": 1,
            },
            {
                "key": "approve",
                "label": "Approve a versioned policy",
                "description": "You promote the standard only after its replay and shadow gates pass.",
                "current": len(active_versions),
                "target": 1,
            },
            {
                "key": "monitor",
                "label": "Monitor and correct",
                "description": "Keep reviewing after activation so regressions are visible and rollback stays available.",
                "current": monitored,
                "target": POLICY_MIN_SUPPORT,
            },
        ]
        for step in steps:
            step["progress"] = round(min(1.0, int(step["current"]) / max(1, int(step["target"]))), 6)
            step["status"] = "complete" if step["progress"] >= 1.0 else "current"
        current_index = next((index for index, step in enumerate(steps) if step["progress"] < 1.0), len(steps) - 1)
        for index, step in enumerate(steps):
            if index < current_index:
                step["status"] = "complete"
            elif index == current_index:
                step["status"] = "current"
            else:
                step["status"] = "waiting"
        if decision_count < POLICY_MIN_SUPPORT:
            next_action = {
                "title": "Review real Kaya examples",
                "description": (
                    f"Mark {POLICY_MIN_SUPPORT - decision_count} more matching decisions Teach Kaya "
                    "to give the compiler its first useful pattern."
                ),
                "action": "review",
            }
        elif ready_for_replay:
            next_action = {
                "title": "Run the first evidence replay",
                "description": "A proposed standard has enough consistent support to test against stored evidence.",
                "action": "replay",
            }
        elif any(row["stage"] == "shadow_ready" for row in candidates):
            next_action = {
                "title": "Start shadow observation",
                "description": "The replay passed. Cortex can now watch three new matching decisions without changing behavior.",
                "action": "shadow",
            }
        elif shadowing:
            next_action = {
                "title": "Keep reviewing matching examples",
                "description": "Shadow mode needs three new decisions before the standard can be approved.",
                "action": "review",
            }
        elif ready:
            next_action = {
                "title": "Approve a tested policy",
                "description": "Replay and shadow evidence agree. Review the exact rule and activate it when ready.",
                "action": "approve",
            }
        elif active_versions and monitored < POLICY_MIN_SUPPORT:
            next_action = {
                "title": "Monitor the active standard",
                "description": f"Add {POLICY_MIN_SUPPORT - monitored} post-activation reviews so regressions can be detected.",
                "action": "review",
            }
        else:
            next_action = {
                "title": "Keep building representative evidence",
                "description": "Review different memory types and situations so the next proposal is not based on one narrow pattern.",
                "action": "review",
            }
        return {
            "overall_progress": round(sum(float(step["progress"]) for step in steps) / len(steps), 6),
            "steps": steps,
            "next_action": next_action,
            "counts": {
                "decisions": decision_count,
                "all_decisions": int(decision_counts["total"] or 0),
                "item_only_decisions": int(decision_counts["item_only"] or 0),
                "exact_duplicate_decisions": int(decision_counts["exact_duplicates"] or 0),
                "mapped_decisions": mapped_decisions,
                "candidates": len(candidates),
                "ready_for_replay": ready_for_replay,
                "shadowing": shadowing,
                "ready": ready,
                "active_policies": len(active_versions),
                "post_activation_reviews": monitored,
            },
            "thresholds": {
                "scoped_support": POLICY_MIN_SUPPORT,
                "consistency": POLICY_MIN_CONSISTENCY,
                "shadow_observations": POLICY_SHADOW_MIN_OBSERVATIONS,
                "core_support": POLICY_CORE_MIN_SUPPORT,
                "core_contexts": POLICY_CORE_MIN_CONTEXTS,
            },
            "candidates": candidates,
            "versions": versions,
            "events": events,
            "claim_boundary": (
                "Operator reviews generate explainable policy candidates. Only tested, explicitly promoted versions "
                "affect the admission, connection, retrieval, or retention core, and every active version can be rolled back."
            ),
        }

    # ------------------------------------------------------------------
    # Memory Refinery: roles, presentations, proposals, actions, and undo
    # ------------------------------------------------------------------

    _REFINERY_ACTIONS = {
        "keep_canonical",
        "keep_reference",
        "rewrite",
        "split",
        "archive",
        "trash",
    }
    _REFINERY_ROLE_ACTIONS = {"keep_canonical": "canonical", "keep_reference": "reference"}

    @staticmethod
    def _clarity_flag_placeholders() -> tuple[str, tuple[str, ...]]:
        return ",".join("?" for _ in CLARITY_FLAGS), tuple(CLARITY_FLAGS)

    def _clarity_predicate_sql(self) -> tuple[str, tuple[str, ...]]:
        placeholders, params = self._clarity_flag_placeholders()
        sql = (
            "m.state IN ('active','cold') AND m.record_role<>'reference' "
            "AND m.role_reviewed_at IS NULL AND EXISTS ("
            "  SELECT 1 FROM json_each(COALESCE(p.readability_flags_json,'[]')) "
            f"  WHERE json_each.value IN ({placeholders})"
            ")"
        )
        return sql, params

    def refinery_summary(self) -> dict[str, Any]:
        """Aggregate role, readability, presentation, and migration status.

        Aggregates only: no memory content, source paths, or identifiers.
        """

        clarity_sql, clarity_params = self._clarity_predicate_sql()
        with self._lock:
            role_state_rows = self._conn.execute(
                "SELECT record_role,state,COUNT(*) n FROM memories GROUP BY record_role,state"
            ).fetchall()
            flag_rows = self._conn.execute(
                """SELECT json_each.value flag,COUNT(*) n
                   FROM memory_presentations,json_each(memory_presentations.readability_flags_json)
                   GROUP BY json_each.value ORDER BY n DESC"""
            ).fetchall()
            clarity_count = int(
                self._conn.execute(
                    f"""SELECT COUNT(*) n FROM memories m
                        LEFT JOIN memory_presentations p ON p.memory_id=m.id
                        WHERE {clarity_sql}""",
                    clarity_params,
                ).fetchone()["n"]
            )
            presentation_counts = self._conn.execute(
                """SELECT COUNT(*) total,
                          SUM(CASE WHEN p.source_digest<>m.content_hash
                                     OR p.presentation_version<>? THEN 1 ELSE 0 END) stale
                   FROM memory_presentations p JOIN memories m ON m.id=p.memory_id""",
                (PRESENTATION_VERSION,),
            ).fetchone()
            missing_presentations = int(
                self._conn.execute(
                    """SELECT COUNT(*) n FROM memories m
                       WHERE NOT EXISTS(SELECT 1 FROM memory_presentations p WHERE p.memory_id=m.id)"""
                ).fetchone()["n"]
            )
            proposal_rows = self._conn.execute(
                "SELECT status,COUNT(*) n FROM memory_refinery_proposals GROUP BY status"
            ).fetchall()
            role_usage_rows = self._conn.execute(
                """SELECT m.record_role role,
                          COUNT(*) selected,
                          SUM(u.used) used,
                          SUM(CASE WHEN u.outcome IN ('helpful','validated') THEN 1 ELSE 0 END) helpful,
                          SUM(CASE WHEN u.outcome IN ('harmful','corrected') THEN 1 ELSE 0 END) harmful,
                          SUM(CASE WHEN u.outcome='ignored' THEN 1 ELSE 0 END) ignored
                   FROM usage_records u JOIN memories m ON m.id=u.memory_id
                   GROUP BY m.record_role"""
            ).fetchall()
            backfill_row = self._conn.execute(
                "SELECT value FROM meta WHERE key=?", (REFINERY_BACKFILL_KEY,)
            ).fetchone()
        role_counts: dict[str, int] = {}
        recallable_role_counts: dict[str, int] = {}
        role_state_counts: dict[str, dict[str, int]] = {}
        for row in role_state_rows:
            role = str(row["record_role"])
            state = str(row["state"])
            count = int(row["n"])
            role_counts[role] = role_counts.get(role, 0) + count
            role_state_counts.setdefault(role, {})[state] = count
            if state in {"active", "cold"}:
                recallable_role_counts[role] = recallable_role_counts.get(role, 0) + count
        total = sum(role_counts.values())
        backfill_version = str(backfill_row["value"]) if backfill_row else ""
        return {
            "classifier_version": ROLE_CLASSIFIER_VERSION,
            "presentation_version": PRESENTATION_VERSION,
            "backfill": {
                "recorded_version": backfill_version,
                "current_version": REFINERY_BACKFILL_VERSION,
                "complete": backfill_version == REFINERY_BACKFILL_VERSION,
            },
            "role_counts": role_counts,
            "role_state_counts": role_state_counts,
            "recallable_role_counts": recallable_role_counts,
            "readability_flag_counts": {str(row["flag"]): int(row["n"]) for row in flag_rows},
            "needs_clarity_count": clarity_count,
            "view_counts": {
                "readable": role_counts.get("canonical", 0),
                "reference": role_counts.get("reference", 0),
                "clarity": clarity_count,
                "all": total,
            },
            "presentations": {
                "total": int(presentation_counts["total"] or 0),
                "stale": int(presentation_counts["stale"] or 0),
                "missing": missing_presentations,
            },
            "proposals": {str(row["status"]): int(row["n"]) for row in proposal_rows},
            "role_usage": [dict(row) for row in role_usage_rows],
        }

    def refinery_items(
        self,
        *,
        view: str = "readable",
        limit: int = 60,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Bounded, filtered rows for one refinery Index view."""

        view_value = str(view or "readable").casefold()
        if view_value not in {"readable", "reference", "clarity", "all"}:
            raise ValueError("view must be readable, reference, clarity, or all")
        bounded_limit = max(1, min(int(limit), 200))
        bounded_offset = max(0, min(int(offset), 100_000))
        clarity_sql, clarity_params = self._clarity_predicate_sql()
        predicates = {
            "readable": ("m.record_role='canonical'", ()),
            "reference": ("m.record_role='reference'", ()),
            "clarity": (clarity_sql, clarity_params),
            "all": ("1=1", ()),
        }
        where_sql, params = predicates[view_value]
        with self._lock:
            total = int(
                self._conn.execute(
                    f"""SELECT COUNT(*) n FROM memories m
                        LEFT JOIN memory_presentations p ON p.memory_id=m.id
                        WHERE {where_sql}""",
                    params,
                ).fetchone()["n"]
            )
            rows = self._conn.execute(
                f"""SELECT m.*,
                           p.display_title,p.display_summary,p.applies_when,p.retention_reason,
                           p.readability_flags_json,p.presentation_method,p.presentation_version,
                           p.source_digest presentation_digest,p.updated_at presentation_updated_at
                    FROM memories m
                    LEFT JOIN memory_presentations p ON p.memory_id=m.id
                    WHERE {where_sql}
                    ORDER BY m.observed_at DESC,m.id
                    LIMIT ? OFFSET ?""",
                (*params, bounded_limit, bounded_offset),
            ).fetchall()
        items = [self._decode_refinery_row(row) for row in rows]
        return {
            "view": view_value,
            "total": total,
            "limit": bounded_limit,
            "offset": bounded_offset,
            "items": items,
        }

    @staticmethod
    def _decode_refinery_row(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        item = _decode_memory_metadata(row)
        flags = _json_string_list(item.pop("readability_flags_json", "[]"))
        item["readability_flags"] = flags
        item["needs_clarity"] = (
            needs_clarity(str(item.get("record_role") or "canonical"), flags)
            and str(item.get("state")) in {"active", "cold"}
            and not item.get("role_reviewed_at")
        )
        return item

    def refinery_classification_report(self) -> dict[str, Any]:
        """Dry-run classification over every record, without any mutation.

        Returns aggregate counts plus bounded, redacted examples (ID prefixes
        and flags only — never content or source paths).
        """

        with self._lock:
            rows = self._conn.execute("SELECT * FROM memories").fetchall()
            dependency_ids = self._active_dependency_ids_tx(self._conn)
        current_counts: dict[str, int] = {}
        proposed_counts: dict[str, int] = {}
        flag_counts: dict[str, int] = {}
        source_counts: dict[str, dict[str, int]] = {}
        changed = 0
        examples: list[dict[str, Any]] = []
        for raw in rows:
            memory = dict(raw)
            current_role = str(memory.get("record_role") or "canonical")
            classification = classify_record_role(
                memory, has_active_dependencies=str(memory["id"]) in dependency_ids
            )
            proposed_role = (
                current_role
                if str(memory.get("role_method") or "") == OPERATOR_ROLE_METHOD
                else str(classification["record_role"])
            )
            current_counts[current_role] = current_counts.get(current_role, 0) + 1
            proposed_counts[proposed_role] = proposed_counts.get(proposed_role, 0) + 1
            for flag in classification["readability_flags"]:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
            source = str(memory.get("source_category") or "UNKNOWN")
            source_counts.setdefault(source, {})
            source_counts[source][proposed_role] = source_counts[source].get(proposed_role, 0) + 1
            if proposed_role != current_role:
                changed += 1
                if len(examples) < 20:
                    examples.append(
                        {
                            "memory_id_prefix": str(memory["id"])[:8],
                            "current_role": current_role,
                            "proposed_role": proposed_role,
                            "readability_flags": list(classification["readability_flags"]),
                        }
                    )
        return {
            "classifier_version": ROLE_CLASSIFIER_VERSION,
            "records": len(rows),
            "current_role_counts": current_counts,
            "proposed_role_counts": proposed_counts,
            "records_with_changed_role": changed,
            "readability_flag_counts": dict(sorted(flag_counts.items(), key=lambda p: (-p[1], p[0]))),
            "proposed_roles_by_source_category": source_counts,
            "redacted_examples": examples,
            "mutations": 0,
        }

    def refinery_preview(
        self,
        kind: str,
        memory_id: str,
        *,
        target_role: str | None = None,
    ) -> dict[str, Any]:
        """Deterministic, mutation-free preview for rewrite, split, or role change."""

        kind_value = str(kind or "").casefold()
        if kind_value not in {"rewrite", "split", "role_change"}:
            raise ValueError("preview kind must be rewrite, split, or role_change")
        memory = self.get_memory(memory_id)
        if not memory:
            raise ValueError("memory not found")
        with self._lock:
            has_dependencies = self._has_active_dependency_tx(self._conn, memory_id)
            presentation_row = self._conn.execute(
                "SELECT * FROM memory_presentations WHERE memory_id=?", (memory_id,)
            ).fetchone()
        classification = classify_record_role(memory, has_active_dependencies=has_dependencies)
        readability_evidence = {
            "readability_flags": list(classification["readability_flags"]),
            "reasons": list(classification["reasons"]),
            "structures": list(classification["structures"]),
        }
        base = {
            "kind": kind_value,
            "memory_id": memory_id,
            "current_role": str(memory.get("record_role") or "canonical"),
            "source_dependencies": [memory_id],
            "readability_evidence": readability_evidence,
            "requires_confirmation": True,
            "presentation": dict(presentation_row) if presentation_row else None,
        }
        content = str(memory.get("content") or "")
        if kind_value == "rewrite":
            base["proposed_records"] = [
                {
                    "content": deterministic_rewrite_preview(content, memory=memory),
                    "kind": str(memory.get("kind") or "semantic"),
                    "editable": True,
                }
            ]
            base["rationale"] = (
                "Deterministic starting point for an operator rewrite. Words are preserved exactly; "
                "edit the text, then confirm to store it as explicit operator evidence linked to the source."
            )
        elif kind_value == "split":
            parts = deterministic_split_preview(content, memory=memory)
            if len(parts) < 2:
                raise ValueError(
                    "this record does not have a safe deterministic split boundary; use rewrite instead"
                )
            base["proposed_records"] = [
                {"content": part, "kind": str(memory.get("kind") or "semantic"), "editable": True}
                for part in parts
            ]
            base["rationale"] = (
                "Deterministic split on paragraph, bullet, or sentence boundaries. Edit each part, remove "
                "any that should not become memories, then confirm."
            )
        else:
            role = str(target_role or "").casefold()
            if role not in RECORD_ROLES:
                raise ValueError(f"target_role must be one of {', '.join(RECORD_ROLES)}")
            base["proposed_role"] = role
            base["rationale"] = (
                f"Change how this record is governed and presented: {base['current_role']} → {role}. "
                "Raw content, provenance, state, and history do not change."
            )
        return base

    def _insert_operator_memory_tx(
        self,
        conn: sqlite3.Connection,
        content: str,
        *,
        kind: str,
        source_memory: dict[str, Any],
        actor: str,
        now: str,
    ) -> str:
        """Insert one operator-authored memory inside an open transaction."""

        clean = normalize_text(content)
        if not clean:
            raise ValueError("proposed memory content cannot be empty")
        if len(clean) > 4000:
            raise ValueError("proposed memory content is longer than the 4000 character bound")
        digest = content_hash(clean)
        context_mode = _normalize_context_mode(
            str(source_memory.get("context_mode") or "standalone")
        )
        scope = _normalize_context_map(_trace_json_object(source_memory.get("scope_json")))
        entities = _normalize_context_list(_json_string_list(source_memory.get("entities_json")))
        preconditions = _normalize_context_map(
            _trace_json_object(source_memory.get("preconditions_json"))
        )
        applicable_systems = _normalize_context_list(
            _json_string_list(source_memory.get("applicable_systems_json"))
        )
        applicable_versions = _normalize_context_list(
            _json_string_list(source_memory.get("applicable_versions_json"))
        )
        scope_json = _trace_json(scope)
        preconditions_json = _trace_json(preconditions)
        systems_json = _trace_json(applicable_systems)
        versions_json = _trace_json(applicable_versions)
        duplicate = conn.execute(
            """SELECT id FROM memories WHERE content_hash=? AND context_mode=?
                 AND scope_json=? AND preconditions_json=?
                 AND applicable_systems_json=? AND applicable_versions_json=? LIMIT 1""",
            (
                digest,
                context_mode,
                scope_json,
                preconditions_json,
                systems_json,
                versions_json,
            ),
        ).fetchone()
        if duplicate:
            raise ValueError("an identical memory already exists; edit the text before confirming")
        memory_id = str(uuid.uuid4())
        source_ref = f"refinery:{source_memory['id']}"
        operator_context = f"Rewritten by {actor} from a stored record during Clarity review."
        prior_source_context = normalize_text(str(source_memory.get("source_context") or ""))
        source_context = normalize_text(
            f"{prior_source_context} {operator_context}" if prior_source_context else operator_context
        )
        calculated_completeness = _memory_metadata_completeness(
            context_mode,
            scope=scope,
            entities=entities,
            preconditions=preconditions,
            source_context=source_context,
            applicable_systems=applicable_systems,
            applicable_versions=applicable_versions,
        )
        metadata_completeness = _clamp(
            float(source_memory.get("metadata_completeness") or calculated_completeness)
        )
        conn.execute(
            """INSERT INTO memories(
                id, kind, content, content_hash, source_type, source_category, source_ref, session_id,
                context_mode,scope_json,entities_json,preconditions_json,source_context,
                applicable_systems_json,applicable_versions_json,metadata_completeness,
                created_at, updated_at, observed_at, valid_from, valid_to, subject, predicate, object_value,
                extraction_method, confidence, currentness_confidence, importance, uniqueness,
                volatility, trust, state, pinned, protected, supersedes_id, quarantine_reason,
                record_role, role_method, role_version, role_reviewed_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                memory_id,
                kind,
                clean,
                digest,
                "operator_rewrite",
                "USER_EXPLICIT",
                source_ref,
                None,
                context_mode,
                scope_json,
                _trace_json(entities),
                preconditions_json,
                source_context,
                systems_json,
                versions_json,
                metadata_completeness,
                now,
                now,
                str(source_memory.get("observed_at") or now),
                source_memory.get("valid_from"),
                source_memory.get("valid_to"),
                None,
                None,
                None,
                "refinery_rewrite_v1",
                0.85,
                _clamp(float(source_memory.get("currentness_confidence") or 0.75)),
                _clamp(float(source_memory.get("importance") or 0.5)),
                1.0,
                _clamp(float(source_memory.get("volatility") or 0.4)),
                0.9,
                "active",
                0,
                0,
                None,
                None,
                "canonical",
                OPERATOR_ROLE_METHOD,
                ROLE_CLASSIFIER_VERSION,
                now,
            ),
        )
        conn.execute(
            """INSERT INTO memory_versions(
                memory_id, content, confidence, state, valid_from, valid_to,
                system_from, reason, source_ref
            ) VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                memory_id,
                clean,
                0.85,
                "active",
                source_memory.get("valid_from"),
                source_memory.get("valid_to"),
                now,
                "created by operator refinery review",
                source_ref,
            ),
        )
        conn.execute("INSERT INTO memory_fts(memory_id, content) VALUES(?,?)", (memory_id, clean))
        self._index_features_tx(conn, memory_id, clean)
        self._index_context_terms_tx(
            conn,
            memory_id,
            context_mode=context_mode,
            scope=scope,
            entities=entities,
            preconditions=preconditions,
            applicable_systems=applicable_systems,
            applicable_versions=applicable_versions,
        )
        conn.execute(
            """INSERT OR IGNORE INTO memory_dependencies(
                 memory_id,evidence_id,relation,weight,active,created_at
               ) VALUES(?,?,'derived_from',1.0,1,?)""",
            (memory_id, str(source_memory["id"]), now),
        )
        self._record_memory_write_decision_tx(
            conn,
            {
                "candidate_hash": digest,
                "kind": kind,
                "source_type": "operator_rewrite",
                "context_mode": context_mode,
                "reusable_score": 0.85,
                "durability": "durable",
                "quality_flags": [],
                "duplicate_memory_id": None,
                "contradiction_ids": [],
                "independently_understandable": True,
                "decision": "created",
                "reason": "operator-confirmed refinery rewrite preserving the source dependency",
            },
            session_id=None,
            memory_id=memory_id,
        )
        row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        self._classify_and_present_tx(
            conn,
            dict(row),
            has_active_dependencies=True,
            explicit_role="canonical",
            role_method=OPERATOR_ROLE_METHOD,
            preserve_operator_role=False,
        )
        return memory_id

    def _refinery_state_change_tx(
        self,
        conn: sqlite3.Connection,
        memory: dict[str, Any],
        next_state: str,
        explanation: str,
        now: str,
    ) -> None:
        memory_id = str(memory["id"])
        conn.execute(
            "UPDATE memory_versions SET system_to=? WHERE memory_id=? AND system_to IS NULL",
            (now, memory_id),
        )
        conn.execute("UPDATE memories SET state=?,updated_at=? WHERE id=?", (next_state, now, memory_id))
        conn.execute(
            """INSERT INTO memory_versions(memory_id,content,confidence,state,valid_from,valid_to,
               system_from,reason,source_ref) VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                memory_id,
                memory["content"],
                memory["confidence"],
                next_state,
                memory["valid_from"],
                memory["valid_to"],
                now,
                explanation,
                memory["source_ref"],
            ),
        )
        conn.execute(
            """INSERT INTO lifecycle_events(
               memory_id,from_state,to_state,reason,retention_score,created_at
               ) VALUES(?,?,?,?,NULL,?)""",
            (memory_id, memory["state"], next_state, explanation, now),
        )
        if next_state in {"archived", "quarantine", "tombstoned"}:
            self._mark_dependents_dirty_tx(conn, memory_id, f"refinery review moved evidence to {next_state}")

    def apply_refinery_action(
        self,
        action: str,
        memory_id: str,
        *,
        proposed_records: Sequence[dict[str, Any]] | None = None,
        reason_code: str = "unspecified",
        reason_text: str = "",
        actor: str = "dashboard-operator",
        decision_scope: str = "item_only",
    ) -> dict[str, Any]:
        """Apply one confirmed, audited, reversible refinery decision."""

        action_value = str(action or "").casefold()
        if action_value not in self._REFINERY_ACTIONS:
            raise ValueError(
                "refinery actions are keep_canonical, keep_reference, rewrite, split, archive, or trash"
            )
        scope_value = _normalize_review_scope(decision_scope)
        if scope_value == "exact_duplicates" and action_value in {"rewrite", "split"}:
            raise ValueError("exact-duplicate reach is not available for rewrite or split")
        reason_value = normalize_text(reason_code)[:80] or "unspecified"
        actor_value = normalize_text(actor)[:80] or "dashboard-operator"
        review_id = str(uuid.uuid4())
        proposal_id = str(uuid.uuid4())
        now = utc_now()
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            if not row:
                raise ValueError("memory not found")
            memory = dict(row)
            if str(memory["state"]) not in {"active", "cold"}:
                raise ValueError("only active or cold records can be reviewed here")
            target_ids = [memory_id]
            memories = {memory_id: memory}
            if scope_value == "exact_duplicates":
                duplicate_rows = conn.execute(
                    """SELECT * FROM memories
                       WHERE content_hash=? AND context_mode=? AND scope_json=? AND preconditions_json=?
                         AND source_type=? AND source_category=? AND COALESCE(source_ref,'')=?
                         AND id<>? AND state IN ('active','cold')
                       ORDER BY created_at,id""",
                    (
                        memory["content_hash"], memory["context_mode"], memory["scope_json"],
                        memory["preconditions_json"], memory["source_type"],
                        memory["source_category"], memory["source_ref"] or "", memory_id,
                    ),
                ).fetchall()
                for duplicate in duplicate_rows:
                    memories[str(duplicate["id"])] = dict(duplicate)
                    target_ids.append(str(duplicate["id"]))
            prior = {
                "memories": {
                    target_id: {
                        "record_role": memories[target_id].get("record_role"),
                        "role_method": memories[target_id].get("role_method"),
                        "role_version": memories[target_id].get("role_version"),
                        "role_reviewed_at": memories[target_id].get("role_reviewed_at"),
                        "state": memories[target_id].get("state"),
                        "content_hash": memories[target_id].get("content_hash"),
                    }
                    for target_id in target_ids
                }
            }
            effect: dict[str, Any] = {"action": action_value}
            created_ids: list[str] = []
            classification = classify_record_role(
                memory,
                has_active_dependencies=self._has_active_dependency_tx(conn, memory_id),
            )

            if action_value in self._REFINERY_ROLE_ACTIONS:
                next_role = self._REFINERY_ROLE_ACTIONS[action_value]
                for target_id in target_ids:
                    conn.execute(
                        """UPDATE memories SET record_role=?,role_method=?,role_version=?,
                           role_reviewed_at=?,updated_at=? WHERE id=?""",
                        (next_role, OPERATOR_ROLE_METHOD, ROLE_CLASSIFIER_VERSION, now, now, target_id),
                    )
                    refreshed = conn.execute("SELECT * FROM memories WHERE id=?", (target_id,)).fetchone()
                    self._classify_and_present_tx(
                        conn,
                        dict(refreshed),
                        has_active_dependencies=self._has_active_dependency_tx(conn, target_id),
                        explicit_role=next_role,
                        role_method=OPERATOR_ROLE_METHOD,
                        preserve_operator_role=False,
                    )
                effect["role_changes"] = {target_id: next_role for target_id in target_ids}
                proposal_kind = "role_change"
            elif action_value in {"rewrite", "split"}:
                records = [dict(record) for record in (proposed_records or []) if isinstance(record, dict)]
                minimum = 1 if action_value == "rewrite" else 2
                maximum = 1 if action_value == "rewrite" else 8
                if not (minimum <= len(records) <= maximum):
                    raise ValueError(
                        "rewrite requires exactly one proposed record; split requires two to eight"
                    )
                for record in records:
                    created_ids.append(
                        self._insert_operator_memory_tx(
                            conn,
                            str(record.get("content") or ""),
                            kind=str(record.get("kind") or memory.get("kind") or "semantic"),
                            source_memory=memory,
                            actor=actor_value,
                            now=now,
                        )
                    )
                conn.execute(
                    """UPDATE memories SET record_role='reference',role_method=?,role_version=?,
                       role_reviewed_at=?,updated_at=? WHERE id=?""",
                    (OPERATOR_ROLE_METHOD, ROLE_CLASSIFIER_VERSION, now, now, memory_id),
                )
                refreshed = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
                self._classify_and_present_tx(
                    conn,
                    dict(refreshed),
                    has_active_dependencies=self._has_active_dependency_tx(conn, memory_id),
                    explicit_role="reference",
                    role_method=OPERATOR_ROLE_METHOD,
                    preserve_operator_role=False,
                )
                effect["created_memory_ids"] = list(created_ids)
                effect["source_role"] = "reference"
                proposal_kind = action_value
            else:
                next_state = "archived" if action_value == "archive" else "tombstoned"
                for target_id in target_ids:
                    self._refinery_state_change_tx(
                        conn,
                        memories[target_id],
                        next_state,
                        f"refinery review: {reason_value}",
                        now,
                    )
                    conn.execute(
                        "UPDATE memories SET role_reviewed_at=?,role_method=? WHERE id=?",
                        (now, OPERATOR_ROLE_METHOD, target_id),
                    )
                effect["state_changes"] = {target_id: next_state for target_id in target_ids}
                proposal_kind = action_value

            undo = {
                "prior_memories": prior["memories"],
                "created_memory_ids": created_ids,
                "created_content_hashes": {
                    created_id: str(
                        conn.execute(
                            "SELECT content_hash FROM memories WHERE id=?", (created_id,)
                        ).fetchone()["content_hash"]
                    )
                    for created_id in created_ids
                },
                "post_action": effect,
            }
            conn.execute(
                """INSERT INTO memory_refinery_proposals(
                     proposal_id,proposal_kind,source_memory_ids_json,proposed_records_json,
                     dependencies_json,rationale,readability_evidence_json,status,actor,review_id,
                     before_state_json,undo_json,created_at,decided_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    proposal_id,
                    proposal_kind,
                    _trace_json(target_ids),
                    _trace_json(
                        [
                            {"memory_id": created_id}
                            for created_id in created_ids
                        ]
                    ),
                    _trace_json([memory_id]),
                    normalize_text(
                        f"Operator {action_value} decision during Clarity review. "
                        + " ".join(classification.get("reasons") or [])
                    )[:1000],
                    _trace_json(list(classification.get("readability_flags") or [])),
                    "applied",
                    actor_value,
                    review_id,
                    json.dumps(prior, sort_keys=True),
                    json.dumps(undo, sort_keys=True),
                    now,
                    now,
                ),
            )
            signal = {
                "item_type": "refinery",
                "action": action_value,
                "reason_code": reason_value,
                "decision_scope": scope_value,
            }
            conn.execute(
                """INSERT INTO operator_review_decisions(
                   review_id,item_type,item_key,proposal_id,src_id,dst_id,action,reason_code,
                   reason_text,prior_json,effect_json,learning_signal_json,decision_scope,actor,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    review_id,
                    "refinery",
                    f"refinery:{memory_id}",
                    None,
                    memory_id,
                    None,
                    action_value,
                    reason_value,
                    normalize_text(reason_text)[:1000] or None,
                    json.dumps(prior, sort_keys=True),
                    json.dumps({**effect, "refinery_proposal_id": proposal_id}, sort_keys=True),
                    json.dumps(signal, sort_keys=True),
                    scope_value,
                    actor_value,
                    now,
                ),
            )
            self._compile_policy_candidates_tx(conn)
        return {
            "review_id": review_id,
            "proposal_id": proposal_id,
            "action": action_value,
            "decision_scope": scope_value,
            "affected_memory_ids": target_ids,
            "created_memory_ids": created_ids,
        }

    def undo_refinery_action(self, review_id: str, *, actor: str = "dashboard-operator") -> bool:
        """Reverse one refinery decision without overwriting later changes."""

        now = utc_now()
        actor_value = normalize_text(actor)[:80] or "dashboard-operator"
        with self.transaction() as conn:
            decision = conn.execute(
                """SELECT * FROM operator_review_decisions
                   WHERE review_id=? AND item_type='refinery' AND reversed_at IS NULL""",
                (review_id,),
            ).fetchone()
            if not decision:
                return False
            proposal = conn.execute(
                "SELECT * FROM memory_refinery_proposals WHERE review_id=? AND status='applied'",
                (review_id,),
            ).fetchone()
            if not proposal:
                return False
            try:
                undo = json.loads(str(proposal["undo_json"] or "{}"))
            except json.JSONDecodeError:
                return False
            post_action = dict(undo.get("post_action") or {})
            state_changes = dict(post_action.get("state_changes") or {})
            role_changes = dict(post_action.get("role_changes") or {})
            source_role = post_action.get("source_role")
            for memory_id, values in dict(undo.get("prior_memories") or {}).items():
                current = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
                if not current:
                    continue
                restore_role_fields = False
                expected_state = state_changes.get(memory_id)
                if expected_state:
                    # Archive/trash undo: only if the state is still what the
                    # action set, so later changes are never overwritten.
                    if str(current["state"]) != expected_state:
                        continue
                    self._refinery_state_change_tx(
                        conn,
                        dict(current),
                        str(values.get("state") or "active"),
                        f"refinery undo by {actor_value}",
                        now,
                    )
                    restore_role_fields = True
                expected_role = role_changes.get(memory_id) or source_role
                if expected_role and str(current["record_role"]) == expected_role:
                    restore_role_fields = True
                if restore_role_fields:
                    conn.execute(
                        """UPDATE memories SET record_role=?,role_method=?,role_version=?,
                           role_reviewed_at=?,updated_at=? WHERE id=?""",
                        (
                            str(values.get("record_role") or "canonical"),
                            str(values.get("role_method") or LEGACY_ROLE_METHOD),
                            values.get("role_version"),
                            values.get("role_reviewed_at"),
                            now,
                            memory_id,
                        ),
                    )
                    refreshed = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
                    self._classify_and_present_tx(
                        conn,
                        dict(refreshed),
                        has_active_dependencies=self._has_active_dependency_tx(conn, memory_id),
                        explicit_role=str(values.get("record_role") or "canonical"),
                        role_method=str(values.get("role_method") or LEGACY_ROLE_METHOD),
                        preserve_operator_role=False,
                    )
            created_hashes = dict(undo.get("created_content_hashes") or {})
            for created_id in list(undo.get("created_memory_ids") or []):
                current = conn.execute("SELECT * FROM memories WHERE id=?", (created_id,)).fetchone()
                if not current:
                    continue
                unchanged = str(current["content_hash"]) == str(created_hashes.get(created_id) or "")
                if unchanged and str(current["state"]) == "active":
                    self._refinery_state_change_tx(
                        conn,
                        dict(current),
                        "tombstoned",
                        f"refinery undo by {actor_value}",
                        now,
                    )
            conn.execute(
                "UPDATE memory_refinery_proposals SET status='undone',undone_at=? WHERE proposal_id=?",
                (now, proposal["proposal_id"]),
            )
            conn.execute(
                "UPDATE operator_review_decisions SET reversed_at=? WHERE review_id=?", (now, review_id)
            )
            self._compile_policy_candidates_tx(conn)
        return True

    def rebuild_presentations(
        self,
        *,
        batch_size: int = 200,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Bounded administrative rebuild of every derived presentation."""

        bounded_batch = max(10, min(int(batch_size), 500))
        with self._lock:
            ids = [
                str(row["id"])
                for row in self._conn.execute("SELECT id FROM memories ORDER BY id").fetchall()
            ]
        rebuilt = 0
        for start in range(0, len(ids), bounded_batch):
            batch = ids[start : start + bounded_batch]
            with self.transaction() as conn:
                dependency_ids = self._active_dependency_ids_tx(conn)
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"SELECT * FROM memories WHERE id IN ({placeholders})", tuple(batch)
                ).fetchall()
                for row in rows:
                    memory = dict(row)
                    self._classify_and_present_tx(
                        conn,
                        memory,
                        has_active_dependencies=str(memory["id"]) in dependency_ids,
                    )
                    rebuilt += 1
            if callable(progress_callback):
                progress_callback(
                    {
                        "phase": "rebuilding",
                        "progress": int(100 * min(1.0, (start + len(batch)) / max(1, len(ids)))),
                        "message": f"Rebuilt {rebuilt} of {len(ids)} presentations.",
                    }
                )
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (REFINERY_BACKFILL_KEY, REFINERY_BACKFILL_VERSION),
            )
            self._conn.commit()
        return {"rebuilt": rebuilt, "total": len(ids)}

    def exact_duplicate_ids(self, memory_id: str) -> list[str]:
        """Return other recallable memories with the same normalized content hash."""

        with self._lock:
            memory = self._conn.execute(
                """SELECT content_hash,context_mode,scope_json,preconditions_json,
                          source_type,source_category,COALESCE(source_ref,'') source_ref
                   FROM memories WHERE id=?""",
                (memory_id,),
            ).fetchone()
            if not memory:
                return []
            rows = self._conn.execute(
                """SELECT id FROM memories
                   WHERE content_hash=? AND context_mode=? AND scope_json=? AND preconditions_json=?
                     AND source_type=? AND source_category=? AND COALESCE(source_ref,'')=?
                     AND id<>? AND state IN ('active','cold')
                   ORDER BY created_at,id""",
                (
                    memory["content_hash"], memory["context_mode"], memory["scope_json"],
                    memory["preconditions_json"], memory["source_type"],
                    memory["source_category"], memory["source_ref"], memory_id,
                ),
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def review_inbox_snapshot(self, *, limit: int = 600) -> dict[str, Any]:
        """Return one decision-ready queue plus the operator-learning trail."""

        bounded = max(1, min(int(limit), 1000))
        with self._lock:
            creation_rows = self._conn.execute(
                """SELECT * FROM memory_creation_proposals
                   WHERE status IN ('pending','needs_context')
                   ORDER BY last_seen_at DESC LIMIT ?""",
                (bounded,),
            ).fetchall()
            proposal_rows = self._conn.execute(
                """SELECT proposal_id,run_id,kind,src_id,dst_id,status,score,evidence_count,
                          rationale,details_json,created_at
                   FROM sleep_proposals WHERE status='proposed'
                   ORDER BY created_at DESC,score DESC LIMIT ?""",
                (bounded,),
            ).fetchall()
            proposal_ids = {
                str(value)
                for row in proposal_rows
                for value in (row["src_id"], row["dst_id"])
                if value
            }
            memory_rows: dict[str, dict[str, Any]] = {}
            if proposal_ids:
                placeholders = ",".join("?" for _ in proposal_ids)
                for row in self._conn.execute(
                    f"""SELECT m.*,p.display_title,p.display_summary,p.applies_when,p.retention_reason
                        FROM memories m LEFT JOIN memory_presentations p ON p.memory_id=m.id
                        WHERE m.id IN ({placeholders})""",
                    tuple(proposal_ids),
                ).fetchall():
                    memory_rows[str(row["id"])] = _decode_memory_metadata(row)
            contradiction_rows = self._conn.execute(
                """SELECT e.src_id,e.dst_id,e.weight,e.evidence_count,e.created_at,
                          src.content src_content,dst.content dst_content
                   FROM edges e
                   JOIN memories src ON src.id=e.src_id
                   JOIN memories dst ON dst.id=e.dst_id
                   JOIN memory_recall_memberships sm
                     ON sm.memory_id=src.id AND sm.revoked_at IS NULL AND sm.eligibility='primary'
                   JOIN memory_recall_memberships dm
                     ON dm.memory_id=dst.id AND dm.recall_set_id=sm.recall_set_id
                    AND dm.revoked_at IS NULL AND dm.eligibility='primary'
                   JOIN memory_recall_sets rs
                     ON rs.recall_set_id=sm.recall_set_id AND rs.status='active'
                   WHERE e.relation='contradicts'
                     AND src.state IN ('active','cold') AND dst.state IN ('active','cold')
                   ORDER BY e.last_reinforced_at DESC LIMIT 500"""
            ).fetchall()
            unsupported_rows = self._conn.execute(
                """SELECT m.* FROM memories m
                   JOIN memory_recall_memberships rm
                     ON rm.memory_id=m.id AND rm.revoked_at IS NULL AND rm.eligibility='primary'
                   JOIN memory_recall_sets rs
                     ON rs.recall_set_id=rm.recall_set_id AND rs.status='active'
                   WHERE m.source_category IN ('AGENT_INFERENCE','REFLECTION')
                     AND m.state IN ('active','cold')
                     AND NOT EXISTS(
                       SELECT 1 FROM memory_dependencies d
                       WHERE d.memory_id=m.id AND d.active=1
                     )
                   ORDER BY m.updated_at DESC LIMIT 500"""
            ).fetchall()
            clarity_sql, clarity_params = self._clarity_predicate_sql()
            clarity_rows = self._conn.execute(
                f"""SELECT m.*,p.display_title,p.display_summary,p.applies_when,p.retention_reason,
                          p.readability_flags_json
                    FROM memories m LEFT JOIN memory_presentations p ON p.memory_id=m.id
                    WHERE {clarity_sql}
                      AND EXISTS(
                        SELECT 1 FROM memory_recall_memberships rm
                        JOIN memory_recall_sets rs ON rs.recall_set_id=rm.recall_set_id
                        WHERE rm.memory_id=m.id AND rm.revoked_at IS NULL
                          AND rm.eligibility='primary' AND rs.status='active'
                      )
                    ORDER BY m.updated_at DESC LIMIT 300""",
                clarity_params,
            ).fetchall()
            history_rows = self._conn.execute(
                """SELECT * FROM operator_review_decisions
                   ORDER BY created_at DESC LIMIT 100"""
            ).fetchall()
            connection_training_rows = self._conn.execute(
                """SELECT d.action,d.reason_code,
                          src.kind src_kind,src.source_category src_source_category,
                          dst.kind dst_kind,dst.source_category dst_source_category
                   FROM operator_review_decisions d
                   JOIN sleep_proposals p ON p.proposal_id=d.proposal_id
                   JOIN memories src ON src.id=d.src_id
                   JOIN memories dst ON dst.id=d.dst_id
                   WHERE d.reversed_at IS NULL AND d.decision_scope='policy_evidence'
                     AND p.kind IN ('association','association_reinforcement')"""
            ).fetchall()
            admitted_projects, admitted_services = self._admitted_dynamic_neighborhoods_tx(
                self._conn
            )
            neighborhood_evaluation_rows = self._conn.execute(
                """SELECT * FROM memory_neighborhood_evaluations
                   ORDER BY decision='admitted' DESC,category,display_name"""
            ).fetchall()
            neighborhood_event_rows = self._conn.execute(
                """SELECT * FROM memory_neighborhood_decision_events
                   ORDER BY created_at DESC LIMIT 100"""
            ).fetchall()

        items: list[dict[str, Any]] = []
        for row in creation_rows:
            proposal = _decode_creation_proposal(row)
            assessment = dict(proposal.get("assessment") or {})
            recurrence_count = int(proposal.get("recurrence_count") or 1)
            items.append(
                {
                    "item_key": f"creation:{proposal['proposal_id']}",
                    "item_type": "creation",
                    "category": "creation",
                    "proposal_kind": "memory_creation",
                    "title": (
                        "Add the missing context before Kaya remembers this"
                        if proposal["status"] == "needs_context"
                        else "Should Kaya remember this?"
                    ),
                    "question": str(
                        assessment.get("reason")
                        or "This candidate is staged outside recall until you decide."
                    ),
                    "score": float(assessment.get("reusable_score") or 0.0),
                    "evidence_count": recurrence_count,
                    "created_at": proposal.get("last_seen_at"),
                    "proposal_id": proposal["proposal_id"],
                    "memories": [],
                    "evidence": [],
                    "creation": {
                        "proposal_id": proposal["proposal_id"],
                        "content": proposal["content"],
                        "kind": proposal["kind"],
                        "source_type": proposal["source_type"],
                        "source_category": proposal["source_category"],
                        "source_ref": proposal.get("source_ref"),
                        "session_id": proposal.get("session_id"),
                        "context_mode": proposal["context_mode"],
                        "candidate": dict(proposal.get("candidate") or {}),
                        "assessment": assessment,
                        "recurrence_count": recurrence_count,
                        "redacted": bool(proposal.get("redacted")),
                        "quarantine_reason": proposal.get("quarantine_reason"),
                        "status": proposal["status"],
                        "first_seen_at": proposal.get("first_seen_at"),
                        "last_seen_at": proposal.get("last_seen_at"),
                        "proposed_neighborhoods": _creation_neighborhood_preview(
                            proposal,
                            admitted_projects=admitted_projects,
                            admitted_services=admitted_services,
                        ),
                    },
                }
            )
        connection_training: dict[str, dict[str, int]] = {}
        for row in connection_training_rows:
            selector = connection_policy_selector(
                {"kind": row["src_kind"], "source_category": row["src_source_category"]},
                {"kind": row["dst_kind"], "source_category": row["dst_source_category"]},
            )
            key = _trace_json(selector)
            counts = connection_training.setdefault(key, {"approved": 0, "denied": 0})
            counts["approved" if str(row["action"]) == "approve" else "denied"] += 1
        connection_kinds = {"association", "association_reinforcement", "edge_downscale"}
        cleanup_kinds = {"consolidation", "lifecycle", "dependency_repair"}
        for row in proposal_rows:
            proposal = dict(row)
            try:
                details = json.loads(str(proposal.pop("details_json") or "{}"))
            except json.JSONDecodeError:
                details = {}
            src = memory_rows.get(str(proposal.get("src_id") or ""))
            dst = memory_rows.get(str(proposal.get("dst_id") or ""))
            if src and str(src.get("state")) not in {"active", "cold"}:
                continue
            if dst and str(dst.get("state")) not in {"active", "cold"}:
                continue
            if src and not self.is_memory_recall_eligible(str(src["id"])):
                continue
            if dst and not self.is_memory_recall_eligible(str(dst["id"])):
                continue
            kind = str(proposal["kind"])
            category = (
                "connections"
                if kind in connection_kinds
                else "cleanup"
                if kind in cleanup_kinds
                else "conflicts"
                if kind == "interference_review"
                else "claims"
            )
            titles = {
                "association": "Should these memories be connected?",
                "association_reinforcement": "Should this connection become stronger?",
                "edge_downscale": "Should this weak connection count less?",
                "consolidation": "Are these memories duplicates?",
                "lifecycle": "Should this memory leave normal recall?",
                "dependency_repair": "This memory lost supporting evidence",
                "interference_review": "These memories disagree",
                "context_review": "Does this memory need narrower context?",
            }
            witnesses: list[dict[str, Any]] = []
            if kind in {"association", "association_reinforcement"} and src and dst:
                left, right = sorted((str(src["id"]), str(dst["id"])))
                with self._lock:
                    witness_rows = self._conn.execute(
                        """SELECT witness_key,evidence_kind,score,episode_id,created_at
                           FROM sleep_association_evidence
                           WHERE src_id=? AND dst_id=?
                           ORDER BY created_at DESC LIMIT 12""",
                        (left, right),
                    ).fetchall()
                witnesses = [dict(witness) for witness in witness_rows]
            connection_review: dict[str, Any] | None = None
            if kind in {"association", "association_reinforcement"} and src and dst:
                selector = connection_policy_selector(src, dst)
                selector_key = _trace_json(selector)
                trained = connection_training.get(selector_key, {"approved": 0, "denied": 0})
                witness_kinds: dict[str, int] = {}
                for witness in witnesses:
                    evidence_kind = str(witness.get("evidence_kind") or "unknown")
                    witness_kinds[evidence_kind] = witness_kinds.get(evidence_kind, 0) + 1
                first_id, second_id = str(src["id"]), str(dst["id"])
                with self._lock:
                    existing_edge_row = self._conn.execute(
                        """SELECT e.src_id,e.dst_id,e.relation,e.weight,e.evidence_count,
                                  e.last_reinforced_at,
                                  COALESCE((SELECT ev.summary FROM edge_evidence ev
                                    WHERE ev.src_id=e.src_id AND ev.dst_id=e.dst_id
                                      AND ev.relation=e.relation
                                    ORDER BY ev.created_at DESC LIMIT 1),'') explanation,
                                  COALESCE((SELECT ev.evidence_type FROM edge_evidence ev
                                    WHERE ev.src_id=e.src_id AND ev.dst_id=e.dst_id
                                      AND ev.relation=e.relation
                                    ORDER BY ev.created_at DESC LIMIT 1),'legacy_unattributed') evidence_type,
                                  (SELECT COUNT(*) FROM edge_evidence ev
                                    WHERE ev.src_id=e.src_id AND ev.dst_id=e.dst_id
                                      AND ev.relation=e.relation) evidence_records
                           FROM edges e
                           WHERE (e.src_id=? AND e.dst_id=?) OR (e.src_id=? AND e.dst_id=?)
                           ORDER BY (e.relation='sleep_replay'),e.weight DESC,e.evidence_count DESC
                           LIMIT 1""",
                        (first_id, second_id, second_id, first_id),
                    ).fetchone()
                existing_edge = _decode_edge(existing_edge_row) if existing_edge_row else None
                connection_review = {
                    "pattern_key": hashlib.sha256(selector_key.encode("utf-8")).hexdigest()[:16],
                    "pattern_label": connection_pattern_label(selector),
                    "selector": selector,
                    "training": {
                        "approved": int(trained.get("approved", 0)),
                        "denied": int(trained.get("denied", 0)),
                        "total": int(trained.get("approved", 0)) + int(trained.get("denied", 0)),
                        "target": POLICY_MIN_SUPPORT,
                    },
                    "evidence_summary": {
                        "distinct_witnesses": int(details.get("distinct_witnesses") or len({w.get("witness_key") for w in witnesses})),
                        "required_witnesses": int(details.get("required_witnesses") or 2),
                        "helpful_co_use": int(witness_kinds.get("helpful_co_use", 0)),
                        "episode_replay": int(witness_kinds.get("episode_replay", 0)),
                    },
                    "shared_signals": _connection_review_signals(src, dst),
                    "existing_edge": existing_edge,
                    "decision_rule": (
                        "Approve only when you can name a stable relationship that would make recalling "
                        "the second memory useful after the first. Co-occurrence by itself is not enough."
                    ),
                }
            items.append(
                {
                    "item_key": f"proposal:{proposal['proposal_id']}",
                    "item_type": "proposal",
                    "category": category,
                    "proposal_kind": kind,
                    "title": titles.get(kind, "Review this Cortex proposal"),
                    "question": str(proposal["rationale"]),
                    "score": float(proposal["score"] or 0.0),
                    "evidence_count": int(proposal["evidence_count"] or 0),
                    "created_at": proposal["created_at"],
                    "proposal_id": proposal["proposal_id"],
                    "run_id": proposal["run_id"],
                    "details": details if isinstance(details, dict) else {},
                    "memories": [memory for memory in (src, dst) if memory],
                    "evidence": witnesses,
                    "connection_review": connection_review,
                }
            )

        known_memory_ids = set(memory_rows)
        for row in contradiction_rows:
            pair_ids = [str(row["src_id"]), str(row["dst_id"])]
            pair_memories: list[dict[str, Any]] = []
            for memory_id in pair_ids:
                if memory_id not in known_memory_ids:
                    memory = self.get_memory(memory_id)
                    if memory:
                        memory_rows[memory_id] = memory
                        known_memory_ids.add(memory_id)
                if memory_id in memory_rows:
                    pair_memories.append(memory_rows[memory_id])
            items.append(
                {
                    "item_key": f"conflict:{':'.join(sorted(pair_ids))}",
                    "item_type": "conflict",
                    "category": "conflicts",
                    "title": "Which statement should Kaya trust now?",
                    "question": "A persistent contradiction link says these statements cannot both be used without context.",
                    "score": float(row["weight"] or 0.0),
                    "evidence_count": int(row["evidence_count"] or 0),
                    "created_at": row["created_at"],
                    "memories": pair_memories,
                    "evidence": [],
                }
            )

        for row in unsupported_rows:
            memory = _decode_memory_metadata(row)
            items.append(
                {
                    "item_key": f"inference:{memory['id']}",
                    "item_type": "inference",
                    "category": "claims",
                    "title": "Did Kaya infer this correctly?",
                    "question": "This claim was generated by an agent or reflection and has no active evidence dependency.",
                    "score": float(memory.get("confidence") or 0.0),
                    "evidence_count": 0,
                    "created_at": memory.get("updated_at"),
                    "memories": [memory],
                    "evidence": [],
                }
            )

        inference_item_ids = {str(row["id"]) for row in unsupported_rows}
        for row in clarity_rows:
            memory = self._decode_refinery_row(row)
            if str(memory["id"]) in inference_item_ids:
                # The unsupported-claim queue already offers a decision for
                # this record; one memory gets one inbox entry.
                continue
            flags = [str(flag) for flag in memory.get("readability_flags") or []]
            flag_text = ", ".join(flag.replace("_", " ") for flag in flags) or "readability concerns"
            items.append(
                {
                    "item_key": f"clarity:{memory['id']}",
                    "item_type": "clarity",
                    "category": "clarity",
                    "title": "Make this record readable, or file it as reference",
                    "question": (
                        "Deterministic readability checks flagged this record: "
                        f"{flag_text}. Decide how Kaya should present and govern it."
                    ),
                    "score": float(len(flags)) / 10.0,
                    "evidence_count": len(flags),
                    "created_at": memory.get("updated_at"),
                    "memories": [memory],
                    "evidence": [],
                    "readability_flags": flags,
                }
            )

        outcome_lab = self.outcome_lab_snapshot(limit=500)
        outcome_memory_ids = {
            str(memory.get("memory_id"))
            for task in outcome_lab.get("tasks", [])
            for memory in task.get("memories", [])
            if memory.get("memory_id")
        }
        outcome_memories: dict[str, dict[str, Any]] = {}
        if outcome_memory_ids:
            placeholders = ",".join("?" for _ in outcome_memory_ids)
            with self._lock:
                rows = self._conn.execute(
                    f"SELECT * FROM memories WHERE id IN ({placeholders})", tuple(outcome_memory_ids)
                ).fetchall()
            outcome_memories = {str(row["id"]): _decode_memory_metadata(row) for row in rows}
        for task in outcome_lab.get("tasks", []):
            if task.get("label_outcome"):
                continue
            items.append(
                {
                    "item_key": f"outcome:{task['task_id']}",
                    "item_type": "outcome",
                    "category": "accuracy",
                    "title": "Did memory help with this answer?",
                    "question": str(task.get("query") or "Private memory-bearing task"),
                    "score": 0.0,
                    "evidence_count": int(task.get("used_count") or 0),
                    "created_at": task.get("created_at"),
                    "task": task,
                    "memories": [
                        {
                            **outcome_memories.get(str(memory.get("memory_id")), {}),
                            "id": memory.get("memory_id"),
                            "attribution": memory.get("attribution"),
                            "usage_score": memory.get("score"),
                        }
                        for memory in task.get("memories", [])
                    ],
                    "evidence": [],
                }
            )

        single_hashes = {
            str(item["memories"][0].get("content_hash") or "")
            for item in items
            if len(item.get("memories") or []) == 1
            and item["memories"][0].get("content_hash")
        }
        duplicate_totals: dict[tuple[str, ...], int] = {}
        if single_hashes:
            placeholders = ",".join("?" for _ in single_hashes)
            with self._lock:
                rows = self._conn.execute(
                    f"""SELECT content_hash,context_mode,scope_json,preconditions_json,
                               source_type,source_category,COALESCE(source_ref,'') source_ref,
                               COUNT(*) count FROM memories
                        WHERE state IN ('active','cold') AND content_hash IN ({placeholders})
                        GROUP BY content_hash,context_mode,scope_json,preconditions_json,
                                 source_type,source_category,COALESCE(source_ref,'')""",
                    tuple(single_hashes),
                ).fetchall()
            duplicate_totals = {
                (
                    str(row["content_hash"]), str(row["context_mode"]), str(row["scope_json"]),
                    str(row["preconditions_json"]), str(row["source_type"]),
                    str(row["source_category"]), str(row["source_ref"]),
                ): int(row["count"])
                for row in rows
            }
        for item in items:
            memories = item.get("memories") or []
            duplicate_key = (
                str(memories[0].get("content_hash") or ""),
                str(memories[0].get("context_mode") or "standalone"),
                str(memories[0].get("scope_json") or "{}"),
                str(memories[0].get("preconditions_json") or "{}"),
                str(memories[0].get("source_type") or "conversation"),
                str(memories[0].get("source_category") or "AGENT_INFERENCE"),
                str(memories[0].get("source_ref") or ""),
            ) if len(memories) == 1 else ()
            item["exact_duplicate_count"] = (
                max(0, duplicate_totals.get(duplicate_key, 1) - 1)
                if len(memories) == 1
                else 0
            )

        category_order = {
            "creation": 0,
            "conflicts": 1,
            "cleanup": 2,
            "clarity": 3,
            "connections": 4,
            "claims": 5,
            "accuracy": 6,
        }
        items.sort(
            key=lambda item: (
                category_order.get(str(item.get("category")), 9),
                -float(item.get("score") or 0.0),
                str(item.get("created_at") or ""),
            )
        )
        counts: dict[str, int] = {}
        for item in items:
            category = str(item["category"])
            counts[category] = counts.get(category, 0) + 1
        connection_groups: dict[str, dict[str, Any]] = {}
        for item in items:
            review = item.get("connection_review")
            if not isinstance(review, dict):
                continue
            pattern_key = str(review.get("pattern_key") or "")
            group = connection_groups.setdefault(
                pattern_key,
                {
                    "pattern_key": pattern_key,
                    "pattern_label": str(review.get("pattern_label") or "Matching connections"),
                    "pending": 0,
                    "training": dict(review.get("training") or {}),
                },
            )
            group["pending"] = int(group["pending"]) + 1

        history: list[dict[str, Any]] = []
        learning_counts: dict[str, int] = {}
        for row in history_rows:
            item = dict(row)
            for field in ("prior_json", "effect_json", "learning_signal_json"):
                try:
                    item[field.removesuffix("_json")] = json.loads(str(item.pop(field) or "{}"))
                except json.JSONDecodeError:
                    item[field.removesuffix("_json")] = {}
            history.append(item)
            if not item.get("reversed_at") and item.get("decision_scope") == "policy_evidence":
                key = f"{item['item_type']}:{item['action']}:{item['reason_code']}"
                learning_counts[key] = learning_counts.get(key, 0) + 1
        return {
            "items": items[:bounded],
            "total": len(items),
            "counts": counts,
            "recent_decisions": history,
            "learning_signals": [
                {"signal": key, "count": count}
                for key, count in sorted(learning_counts.items(), key=lambda pair: (-pair[1], pair[0]))
            ],
            "connection_groups": sorted(
                connection_groups.values(),
                key=lambda group: (
                    -int(group.get("pending") or 0),
                    str(group.get("pattern_label") or ""),
                ),
            ),
            "neighborhood_training": {
                "minimum_memories": DYNAMIC_NEIGHBORHOOD_MIN_MEMORIES,
                "minimum_contexts": DYNAMIC_NEIGHBORHOOD_MIN_CONTEXTS,
                "evaluations": [dict(row) for row in neighborhood_evaluation_rows],
                "recent_events": [dict(row) for row in neighborhood_event_rows],
                "rule": (
                    "Named projects and services remain ordinary scoped details until repeated "
                    "operator-approved memories establish a stable schema across independent contexts."
                ),
            },
            "standards": {
                "creation": "Candidates remain outside recall until you explicitly remember them. Rejecting or asking for context creates audit evidence but no memory.",
                "trash": "Tombstones the memory and removes it from normal recall; content and provenance remain restorable.",
                "archive": "Removes the memory from normal recall while preserving its full history.",
                "connection": "Approval creates an explained link backed by this operator decision; denial stores why it was rejected.",
                "learning": "Only decisions marked Teach Kaya become policy evidence. One-off and exact-duplicate actions stay out of proposed standards.",
                "clarity": "Clarity decisions change how a record is presented and governed. Rewrite and split show an editable preview first, keep the raw record as linked evidence, and remain reversible.",
            },
        }

    def record_review_copilot_interpretation(
        self,
        *,
        proposal_id: str,
        operator_text: str,
        conversation: list[dict[str, Any]],
        response_mode: str,
        response: dict[str, Any],
        provider: str,
        model: str,
        usage: dict[str, Any] | None = None,
    ) -> str:
        """Keep copilot dialogue in the audit ledger, never recallable memory."""

        mode = normalize_text(response_mode).casefold()
        if mode not in {"clarify", "recommendation"}:
            raise ValueError("copilot response mode must be clarify or recommendation")
        note = str(operator_text or "").replace("\x00", "").strip()[:2000]
        if not note:
            raise ValueError("copilot operator text is required")
        with self._lock:
            proposal = self._conn.execute(
                "SELECT proposal_id FROM sleep_proposals WHERE proposal_id=? AND status='proposed'",
                (proposal_id,),
            ).fetchone()
        if not proposal:
            raise ValueError("this proposal is no longer waiting for review")
        usage = usage or {}

        def token_value(key: str) -> int | None:
            try:
                raw = usage.get(key)
                return max(0, int(raw)) if raw is not None else None
            except (TypeError, ValueError):
                return None

        interpretation_id = str(uuid.uuid4())
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO review_copilot_interpretations(
                   interpretation_id,proposal_id,item_key,operator_text,conversation_json,
                   response_mode,response_json,provider,model,input_tokens,output_tokens,total_tokens,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    interpretation_id,
                    proposal_id,
                    f"proposal:{proposal_id}",
                    note,
                    json.dumps(conversation[-8:], sort_keys=True, ensure_ascii=False),
                    mode,
                    json.dumps(response, sort_keys=True, ensure_ascii=False),
                    normalize_text(provider)[:200] or "configured provider",
                    normalize_text(model)[:200] or "configured model",
                    token_value("input_tokens"),
                    token_value("output_tokens"),
                    token_value("total_tokens"),
                    utc_now(),
                ),
            )
        return interpretation_id

    def review_copilot_interpretation(self, interpretation_id: str) -> dict[str, Any] | None:
        """Return one local audit record for tests and operator inspection."""

        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM review_copilot_interpretations WHERE interpretation_id=?",
                (interpretation_id,),
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        for key in ("conversation_json", "response_json"):
            try:
                item[key.removesuffix("_json")] = json.loads(str(item.pop(key) or "{}"))
            except json.JSONDecodeError:
                item[key.removesuffix("_json")] = [] if key == "conversation_json" else {}
        return item

    def record_operator_review(
        self,
        *,
        item_type: str,
        item_key: str,
        action: str,
        reason_code: str,
        reason_text: str = "",
        actor: str = "dashboard-operator",
        proposal_id: str | None = None,
        src_id: str | None = None,
        dst_id: str | None = None,
        prior: dict[str, Any] | None = None,
        effect: dict[str, Any] | None = None,
        decision_scope: str = "item_only",
    ) -> str:
        review_id = str(uuid.uuid4())
        scope_value = _normalize_review_scope(decision_scope)
        signal = {
            "item_type": normalize_text(item_type)[:80],
            "action": normalize_text(action)[:80],
            "reason_code": normalize_text(reason_code)[:80] or "unspecified",
            "decision_scope": scope_value,
        }
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO operator_review_decisions(
                   review_id,item_type,item_key,proposal_id,src_id,dst_id,action,reason_code,
                   reason_text,prior_json,effect_json,learning_signal_json,decision_scope,actor,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    review_id,
                    normalize_text(item_type)[:80],
                    normalize_text(item_key)[:500],
                    proposal_id,
                    src_id,
                    dst_id,
                    normalize_text(action)[:80],
                    normalize_text(reason_code)[:80] or "unspecified",
                    normalize_text(reason_text)[:1000] or None,
                    json.dumps(prior or {}, sort_keys=True),
                    json.dumps(effect or {}, sort_keys=True),
                    json.dumps(signal, sort_keys=True),
                    scope_value,
                    normalize_text(actor)[:80] or "dashboard-operator",
                    utc_now(),
                ),
            )
            self._compile_policy_candidates_tx(conn)
        return review_id

    def decide_review_proposal(
        self,
        proposal_id: str,
        action: str,
        *,
        reason_code: str,
        reason_text: str = "",
        actor: str = "dashboard-operator",
        decision_scope: str = "item_only",
        copilot_interpretation_id: str | None = None,
    ) -> dict[str, Any]:
        """Apply or deny one Sleep proposal and preserve enough state to undo it."""

        action_value = normalize_text(action).casefold()
        reason_value = normalize_text(reason_code)[:80] or "unspecified"
        scope_value = _normalize_review_scope(decision_scope)
        review_id = str(uuid.uuid4())
        now = utc_now()
        with self.transaction() as conn:
            proposal = conn.execute(
                "SELECT * FROM sleep_proposals WHERE proposal_id=? AND status='proposed'", (proposal_id,)
            ).fetchone()
            if not proposal:
                raise ValueError("this proposal is no longer waiting for review")
            copilot_row = None
            if copilot_interpretation_id:
                copilot_row = conn.execute(
                    """SELECT * FROM review_copilot_interpretations
                       WHERE interpretation_id=? AND confirmed_review_id IS NULL""",
                    (copilot_interpretation_id,),
                ).fetchone()
                if not copilot_row or str(copilot_row["proposal_id"] or "") != proposal_id:
                    raise ValueError("this copilot recommendation does not belong to the active review")
                if str(copilot_row["response_mode"]) != "recommendation":
                    raise ValueError("a clarifying question cannot be confirmed as a review decision")
                try:
                    copilot_response = json.loads(str(copilot_row["response_json"] or "{}"))
                except json.JSONDecodeError as error:
                    raise ValueError("the copilot recommendation audit record is invalid") from error
                recommendation = copilot_response.get("recommendation") if isinstance(copilot_response, dict) else None
                if not isinstance(recommendation, dict):
                    raise ValueError("the copilot recommendation audit record is incomplete")
                expected = (
                    str(recommendation.get("api_action") or ""),
                    str(recommendation.get("reason_code") or ""),
                    str(recommendation.get("decision_scope") or ""),
                )
                if expected != (action_value, reason_value, scope_value):
                    raise ValueError(
                        "the review choices changed after the copilot preview; confirm manually or request a new preview"
                    )
            kind = str(proposal["kind"])
            try:
                details = json.loads(str(proposal["details_json"] or "{}"))
            except json.JSONDecodeError:
                details = {}
            ids = [str(value) for value in (proposal["src_id"], proposal["dst_id"]) if value]
            memories = {
                str(row["id"]): dict(row)
                for row in conn.execute(
                    f"SELECT * FROM memories WHERE id IN ({','.join('?' for _ in ids)})", ids
                ).fetchall()
            } if ids else {}
            scope_target_ids = list(ids)
            if scope_value == "exact_duplicates":
                if len(ids) != 1 or ids[0] not in memories:
                    raise ValueError("exact-duplicate reach is available only for a single-memory review")
                primary = memories[ids[0]]
                duplicate_rows = conn.execute(
                    """SELECT * FROM memories
                       WHERE content_hash=? AND context_mode=? AND scope_json=? AND preconditions_json=?
                         AND source_type=? AND source_category=? AND COALESCE(source_ref,'')=?
                         AND id<>? AND state IN ('active','cold')
                       ORDER BY created_at,id""",
                    (
                        primary["content_hash"], primary["context_mode"], primary["scope_json"],
                        primary["preconditions_json"], primary["source_type"],
                        primary["source_category"], primary["source_ref"] or "", ids[0],
                    ),
                ).fetchall()
                for row in duplicate_rows:
                    memories[str(row["id"])] = dict(row)
                scope_target_ids.extend(str(row["id"]) for row in duplicate_rows)
            prior: dict[str, Any] = {
                "proposal_status": str(proposal["status"]),
                "memories": {
                    memory_id: {
                        key: memory[key]
                        for key in ("state", "protected", "source_category", "confidence", "trust", "dirty")
                    }
                    for memory_id, memory in memories.items()
                },
            }
            effect: dict[str, Any] = {"proposal_kind": kind}

            def change_state(memory_id: str, next_state: str, explanation: str) -> None:
                memory = memories.get(memory_id)
                if not memory or str(memory["state"]) not in {"active", "cold"}:
                    raise ValueError("the memory is no longer eligible for this decision")
                conn.execute(
                    "UPDATE memory_versions SET system_to=? WHERE memory_id=? AND system_to IS NULL",
                    (now, memory_id),
                )
                conn.execute("UPDATE memories SET state=?,updated_at=? WHERE id=?", (next_state, now, memory_id))
                conn.execute(
                    """INSERT INTO memory_versions(memory_id,content,confidence,state,valid_from,valid_to,
                       system_from,reason,source_ref) VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        memory_id, memory["content"], memory["confidence"], next_state,
                        memory["valid_from"], memory["valid_to"], now, explanation, memory["source_ref"],
                    ),
                )
                conn.execute(
                    """INSERT INTO lifecycle_events(
                       memory_id,from_state,to_state,reason,retention_score,created_at
                       ) VALUES(?,?,?,?,NULL,?)""",
                    (memory_id, memory["state"], next_state, explanation, now),
                )
                if next_state in {"archived", "quarantine", "tombstoned"}:
                    self._mark_dependents_dirty_tx(conn, memory_id, f"operator review moved evidence to {next_state}")
                effect.setdefault("state_changes", {})[memory_id] = next_state

            if kind in {"association", "association_reinforcement"}:
                if action_value not in {"approve", "deny"}:
                    raise ValueError("connection proposals can only be approved or denied")
                if action_value == "approve":
                    if len(ids) != 2:
                        raise ValueError("connection proposal is missing one memory")
                    memory_a, memory_b = str(proposal["src_id"]), str(proposal["dst_id"])
                    existing_connection = conn.execute(
                        """SELECT * FROM edges
                           WHERE (src_id=? AND dst_id=?) OR (src_id=? AND dst_id=?)
                           ORDER BY (relation='sleep_replay'),weight DESC,evidence_count DESC LIMIT 1""",
                        (memory_a, memory_b, memory_b, memory_a),
                    ).fetchone()
                    relation = "operator_link"
                    src_id, dst_id = sorted((memory_a, memory_b))
                    explanation = "You approved this pair as a durable connection Kaya should be able to follow."
                    if (
                        kind == "association_reinforcement"
                        and existing_connection
                        and reason_value == "reinforce_existing"
                    ):
                        src_id = str(existing_connection["src_id"])
                        dst_id = str(existing_connection["dst_id"])
                        relation = str(existing_connection["relation"])
                        explanation = (
                            f"You confirmed that the existing {relation.replace('_', ' ')} connection "
                            "is still useful and deserves stronger influence."
                        )
                    elif reason_value in {"a_supports_b", "meaningful_dependency"}:
                        src_id, dst_id, relation = memory_a, memory_b, "supports"
                        explanation = "You approved this because Memory A supports or explains Memory B."
                    elif reason_value == "b_supports_a":
                        src_id, dst_id, relation = memory_b, memory_a, "supports"
                        explanation = "You approved this because Memory B supports or explains Memory A."
                    elif reason_value in {"same_subject", "same_context"}:
                        relation = "same_subject" if reason_value == "same_subject" else "same_context"
                        explanation = (
                            "You approved this because both memories describe the same durable subject."
                            if reason_value == "same_subject"
                            else "You approved this because both memories are useful in the same durable context."
                        )
                    elif reason_value == "useful_together":
                        relation = "useful_together"
                        explanation = "You approved this because recalling either memory should make the other useful."
                    operator_note = normalize_text(reason_text)[:300]
                    if operator_note:
                        explanation = f"{explanation} Your note: {operator_note}"
                    edge = conn.execute(
                        "SELECT * FROM edges WHERE src_id=? AND dst_id=? AND relation=?",
                        (src_id, dst_id, relation),
                    ).fetchone()
                    prior["edge"] = dict(edge) if edge else None
                    weight = min(1.0, max(0.25, float(proposal["score"] or 0.0)))
                    conn.execute(
                        """INSERT INTO edges(src_id,dst_id,relation,weight,evidence_count,created_at,last_reinforced_at)
                           VALUES(?,?,?,?,1,?,?)
                           ON CONFLICT(src_id,dst_id,relation) DO UPDATE SET
                             weight=MIN(1.0,MAX(edges.weight,excluded.weight)),
                             evidence_count=edges.evidence_count+1,
                             last_reinforced_at=excluded.last_reinforced_at""",
                        (src_id, dst_id, relation, weight, now, now),
                    )
                    self._record_edge_evidence_tx(
                        conn, src_id, dst_id, relation,
                        evidence_type="operator_review", evidence_key=review_id,
                        summary=explanation,
                        source_ref=f"review:{review_id}",
                        metadata={
                            "proposal_id": proposal_id,
                            "reason_code": reason_value,
                            "proposal_evidence_count": int(proposal["evidence_count"] or 0),
                            "memory_a": memory_a,
                            "memory_b": memory_b,
                        },
                        created_at=now,
                    )
                    current_edge = conn.execute(
                        "SELECT weight,evidence_count FROM edges WHERE src_id=? AND dst_id=? AND relation=?",
                        (src_id, dst_id, relation),
                    ).fetchone()
                    effect["edge"] = {
                        "src_id": src_id,
                        "dst_id": dst_id,
                        "relation": relation,
                        "explanation": explanation,
                        "memory_a": memory_a,
                        "memory_b": memory_b,
                        "created": edge is None,
                        "weight": float(current_edge["weight"] if current_edge else weight),
                        "evidence_count": int(current_edge["evidence_count"] if current_edge else 1),
                    }
                    next_status = "operator_approved"
                else:
                    next_status = "operator_denied"
            elif kind == "edge_downscale":
                if action_value not in {"approve", "deny"}:
                    raise ValueError("connection-weight proposals can only be approved or denied")
                if action_value == "approve":
                    relation = str(details.get("relation") or "related")
                    edge = conn.execute(
                        "SELECT * FROM edges WHERE src_id=? AND dst_id=? AND relation=?",
                        (proposal["src_id"], proposal["dst_id"], relation),
                    ).fetchone()
                    if not edge:
                        raise ValueError("the connection no longer exists")
                    prior["edge"] = dict(edge)
                    conn.execute(
                        "UPDATE edges SET weight=? WHERE src_id=? AND dst_id=? AND relation=?",
                        (float(details.get("next_weight") or edge["weight"]), edge["src_id"], edge["dst_id"], relation),
                    )
                    effect["edge"] = {"src_id": edge["src_id"], "dst_id": edge["dst_id"], "relation": relation}
                    next_status = "operator_approved"
                else:
                    next_status = "operator_denied"
            elif kind == "consolidation":
                if action_value not in {"keep", "archive", "trash"}:
                    raise ValueError("duplicate proposals support keep, archive, or trash")
                if action_value in {"archive", "trash"}:
                    change_state(str(proposal["dst_id"]), "archived" if action_value == "archive" else "tombstoned", f"operator review: {reason_value}")
                    next_status = "operator_approved"
                else:
                    next_status = "operator_denied"
            elif kind in {"lifecycle", "dependency_repair", "context_review"}:
                allowed = {"keep", "archive", "trash", "quarantine"}
                if action_value not in allowed:
                    raise ValueError("memory proposals support keep, archive, trash, or quarantine")
                if action_value == "keep":
                    conn.executemany(
                        "UPDATE memories SET protected=1,updated_at=? WHERE id=?",
                        [(now, memory_id) for memory_id in scope_target_ids],
                    )
                    effect["protected"] = scope_target_ids
                    next_status = "operator_denied"
                else:
                    for memory_id in scope_target_ids:
                        change_state(
                            memory_id,
                            "tombstoned" if action_value == "trash" else action_value,
                            f"operator review: {reason_value}",
                        )
                    next_status = "operator_approved"
            elif kind == "interference_review":
                if action_value not in {"keep_first", "keep_second", "both_valid", "trash_both"}:
                    raise ValueError("conflict proposals require a current statement, both valid, or trash both")
                if len(ids) != 2:
                    raise ValueError("conflict proposal is missing one memory")
                first_id, second_id = ids
                if action_value == "trash_both":
                    change_state(first_id, "tombstoned", f"operator conflict review: {reason_value}")
                    change_state(second_id, "tombstoned", f"operator conflict review: {reason_value}")
                elif action_value == "both_valid":
                    src_id, dst_id = sorted(ids)
                    edge = conn.execute(
                        "SELECT * FROM edges WHERE src_id=? AND dst_id=? AND relation='contextual'",
                        (src_id, dst_id),
                    ).fetchone()
                    prior["edge"] = dict(edge) if edge else None
                    conn.execute(
                        """INSERT INTO edges(src_id,dst_id,relation,weight,evidence_count,created_at,last_reinforced_at)
                           VALUES(?,?,'contextual',0.7,1,?,?)
                           ON CONFLICT(src_id,dst_id,relation) DO UPDATE SET
                             weight=MAX(edges.weight,0.7),evidence_count=edges.evidence_count+1,
                             last_reinforced_at=excluded.last_reinforced_at""",
                        (src_id, dst_id, now, now),
                    )
                    self._record_edge_evidence_tx(
                        conn, src_id, dst_id, "contextual",
                        evidence_type="operator_review", evidence_key=review_id,
                        summary="An operator confirmed that both statements are valid in different contexts.",
                        source_ref=f"review:{review_id}", metadata={"reason_code": reason_value}, created_at=now,
                    )
                    effect["edge"] = {"src_id": src_id, "dst_id": dst_id, "relation": "contextual"}
                else:
                    loser_id = second_id if action_value == "keep_first" else first_id
                    change_state(loser_id, "archived", f"operator conflict review: {reason_value}")
                next_status = "operator_approved"
            else:
                if action_value not in {"keep", "archive", "trash"}:
                    raise ValueError("unsupported proposal decision")
                if action_value == "keep":
                    next_status = "operator_denied"
                else:
                    for memory_id in scope_target_ids:
                        change_state(
                            memory_id,
                            "archived" if action_value == "archive" else "tombstoned",
                            f"operator review: {reason_value}",
                        )
                    next_status = "operator_approved"

            conn.execute("UPDATE sleep_proposals SET status=? WHERE proposal_id=?", (next_status, proposal_id))
            signal = {
                "proposal_kind": kind,
                "action": action_value,
                "reason_code": reason_value,
                "decision_scope": scope_value,
            }
            conn.execute(
                """INSERT INTO operator_review_decisions(
                   review_id,item_type,item_key,proposal_id,src_id,dst_id,action,reason_code,
                   reason_text,prior_json,effect_json,learning_signal_json,decision_scope,actor,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    review_id, "proposal", f"proposal:{proposal_id}", proposal_id,
                    proposal["src_id"], proposal["dst_id"], action_value, reason_value,
                    normalize_text(reason_text)[:1000] or None,
                    json.dumps(prior, sort_keys=True), json.dumps(effect, sort_keys=True),
                    json.dumps(signal, sort_keys=True), scope_value,
                    normalize_text(actor)[:80] or "dashboard-operator", now,
                ),
            )
            if copilot_row is not None:
                conn.execute(
                    """UPDATE review_copilot_interpretations
                       SET confirmed_review_id=?,confirmed_at=?
                       WHERE interpretation_id=? AND confirmed_review_id IS NULL""",
                    (review_id, now, copilot_interpretation_id),
                )
            if effect.get("state_changes"):
                self._refresh_dynamic_neighborhoods_tx(conn)
            self._compile_policy_candidates_tx(conn)
        affected_ids = sorted(
            set(effect.get("state_changes", {}))
            | set(effect.get("protected", []) if isinstance(effect.get("protected"), list) else [])
        )
        return {
            "review_id": review_id,
            "proposal_id": proposal_id,
            "action": action_value,
            "decision_scope": scope_value,
            "copilot_interpretation_id": copilot_interpretation_id,
            "affected_memory_ids": affected_ids,
            "edge": effect.get("edge") if isinstance(effect, dict) else None,
        }

    def undo_review_decision(self, review_id: str, *, actor: str = "dashboard-operator") -> bool:
        """Reverse an operator proposal decision using its preserved before-state."""

        now = utc_now()
        with self.transaction() as conn:
            decision = conn.execute(
                "SELECT * FROM operator_review_decisions WHERE review_id=? AND reversed_at IS NULL", (review_id,)
            ).fetchone()
            if not decision:
                return False
            try:
                prior = json.loads(str(decision["prior_json"] or "{}"))
                effect = json.loads(str(decision["effect_json"] or "{}"))
            except json.JSONDecodeError:
                return False
            if str(decision["item_type"]) == "creation":
                item_key = str(decision["item_key"] or "")
                proposal_id = item_key.removeprefix("creation:")
                proposal = conn.execute(
                    "SELECT * FROM memory_creation_proposals WHERE proposal_id=?", (proposal_id,)
                ).fetchone()
                if not proposal:
                    return False
                memory_id = str(effect.get("memory_id") or "")
                if memory_id and bool(effect.get("memory_promoted")):
                    memory_prior = dict(prior.get("memory") or {})
                    membership_prior = memory_prior.get("membership")
                    if isinstance(membership_prior, dict) and membership_prior.get(
                        "recall_set_id"
                    ):
                        restored = conn.execute(
                            """UPDATE memory_recall_memberships
                               SET eligibility=?,origin=?,review_id=?,actor=?,reason=?,
                                   created_at=?,revoked_at=?
                               WHERE recall_set_id=? AND memory_id=?""",
                            (
                                membership_prior.get("eligibility"),
                                membership_prior.get("origin"),
                                membership_prior.get("review_id"),
                                membership_prior.get("actor"),
                                membership_prior.get("reason"),
                                membership_prior.get("created_at"),
                                membership_prior.get("revoked_at"),
                                membership_prior.get("recall_set_id"),
                                memory_id,
                            ),
                        )
                        if restored.rowcount != 1:
                            raise RuntimeError(
                                "creation review undo could not restore prior recall membership"
                            )
                    else:
                        conn.execute(
                            """UPDATE memory_recall_memberships SET revoked_at=?
                               WHERE memory_id=? AND review_id=? AND revoked_at IS NULL""",
                            (now, memory_id, review_id),
                        )
                    if memory_prior:
                        conn.execute(
                            """UPDATE memories
                               SET approval_state=?,origin_source_category=?,updated_at=?
                               WHERE id=?""",
                            (
                                memory_prior.get("approval_state") or "unreviewed",
                                memory_prior.get("origin_source_category") or "AGENT_INFERENCE",
                                memory_prior.get("updated_at") or now,
                                memory_id,
                            ),
                        )
                if memory_id and bool(effect.get("memory_created")):
                    memory = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
                    if memory and str(memory["state"]) in {"active", "cold"}:
                        conn.execute(
                            "UPDATE memory_versions SET system_to=? WHERE memory_id=? AND system_to IS NULL",
                            (now, memory_id),
                        )
                        conn.execute(
                            "UPDATE memories SET state='archived',updated_at=? WHERE id=?",
                            (now, memory_id),
                        )
                        conn.execute(
                            """INSERT INTO memory_versions(
                                 memory_id,content,confidence,state,valid_from,valid_to,
                                 system_from,reason,source_ref
                               ) VALUES(?,?,?,'archived',?,?,?,?,?)""",
                            (
                                memory_id,
                                memory["content"],
                                memory["confidence"],
                                memory["valid_from"],
                                memory["valid_to"],
                                now,
                                f"creation review undone by {normalize_text(actor)[:80]}",
                                memory["source_ref"],
                            ),
                        )
                        conn.execute(
                            """INSERT INTO lifecycle_events(
                                 memory_id,from_state,to_state,reason,retention_score,created_at
                               ) VALUES(?,?,'archived',?,NULL,?)""",
                            (
                                memory_id,
                                memory["state"],
                                f"creation review undone by {normalize_text(actor)[:80]}",
                                now,
                            ),
                        )
                        conn.execute(
                            """UPDATE memory_recall_memberships SET revoked_at=?
                               WHERE memory_id=? AND revoked_at IS NULL""",
                            (now, memory_id),
                        )
                conn.execute(
                    """UPDATE memory_creation_proposals
                       SET status='pending',decision_action=NULL,decision_note=NULL,
                           result_memory_id=NULL,review_id=NULL,actor=NULL,decided_at=NULL,
                           review_started_at=NULL
                       WHERE proposal_id=?""",
                    (proposal_id,),
                )
                conn.execute(
                    "UPDATE operator_review_decisions SET reversed_at=? WHERE review_id=?",
                    (now, review_id),
                )
                self._refresh_dynamic_neighborhoods_tx(conn)
                self._compile_policy_candidates_tx(conn)
                return True
            if not decision["proposal_id"]:
                return False
            for memory_id, values in dict(prior.get("memories") or {}).items():
                current = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
                if not current:
                    continue
                conn.execute(
                    """UPDATE memories SET state=?,protected=?,source_category=?,confidence=?,trust=?,dirty=?,updated_at=?
                       WHERE id=?""",
                    (
                        values.get("state", "active"), values.get("protected", 0),
                        values.get("source_category", "AGENT_INFERENCE"), values.get("confidence", 0.6),
                        values.get("trust", 0.7), values.get("dirty", 0), now, memory_id,
                    ),
                )
                if str(current["state"]) != str(values.get("state")):
                    conn.execute(
                        "UPDATE memory_versions SET system_to=? WHERE memory_id=? AND system_to IS NULL",
                        (now, memory_id),
                    )
                    conn.execute(
                        """INSERT INTO memory_versions(memory_id,content,confidence,state,valid_from,valid_to,
                           system_from,reason,source_ref) VALUES(?,?,?,?,?,?,?,?,?)""",
                        (
                            memory_id, current["content"], values.get("confidence", current["confidence"]),
                            values.get("state", "active"), current["valid_from"], current["valid_to"],
                            now, f"review undo by {normalize_text(actor)[:80]}", current["source_ref"],
                        ),
                    )
                    conn.execute(
                        """INSERT INTO lifecycle_events(
                           memory_id,from_state,to_state,reason,retention_score,created_at
                           ) VALUES(?,?,?,?,NULL,?)""",
                        (memory_id, current["state"], values.get("state", "active"), f"review undo by {normalize_text(actor)[:80]}", now),
                    )
            edge_effect = effect.get("edge") if isinstance(effect, dict) else None
            if isinstance(edge_effect, dict):
                edge_prior = prior.get("edge")
                key = (edge_effect.get("src_id"), edge_effect.get("dst_id"), edge_effect.get("relation"))
                conn.execute("DELETE FROM edge_evidence WHERE evidence_key=?", (review_id,))
                if edge_prior:
                    conn.execute(
                        """UPDATE edges SET weight=?,evidence_count=?,created_at=?,last_reinforced_at=?
                           WHERE src_id=? AND dst_id=? AND relation=?""",
                        (
                            edge_prior["weight"], edge_prior["evidence_count"], edge_prior["created_at"],
                            edge_prior["last_reinforced_at"], *key,
                        ),
                    )
                else:
                    conn.execute("DELETE FROM edges WHERE src_id=? AND dst_id=? AND relation=?", key)
            conn.execute("UPDATE sleep_proposals SET status='proposed' WHERE proposal_id=?", (decision["proposal_id"],))
            conn.execute("UPDATE operator_review_decisions SET reversed_at=? WHERE review_id=?", (now, review_id))
            self._refresh_dynamic_neighborhoods_tx(conn)
            self._compile_policy_candidates_tx(conn)
        return True

    def dashboard_snapshot(self, *, memory_limit: int = 1000) -> dict[str, Any]:
        """Return a bounded read-only snapshot for the local visualization dashboard."""
        with self._lock:
            memory_rows = self._conn.execute(
                """SELECT m.id,m.kind,m.content,m.source_type,m.source_category,
                   m.origin_source_category,m.approval_state,m.source_ref,m.session_id,
                   m.created_at,m.updated_at,m.observed_at,m.valid_from,m.valid_to,m.subject,m.predicate,
                   m.object_value,
                   m.extraction_method,m.confidence,m.currentness_confidence,m.importance,m.uniqueness,
                   m.volatility,
                   m.trust,m.state,m.pinned,m.protected,m.dirty,m.dirty_reason,m.supersedes_id,
                   m.quarantine_reason,
                   m.context_mode,m.scope_json,m.entities_json,m.preconditions_json,m.source_context,
                   m.applicable_systems_json,m.applicable_versions_json,m.metadata_completeness,
                   m.retrieved_count,m.selected_count,m.injected_count,m.used_count,m.success_count,
                   m.confirmed_count,
                   m.validated_count,m.helpful_count,m.harmful_count,m.correction_count,
                   m.false_positive_count,
                   m.duplicate_count,m.last_retrieved_at,m.last_injected_at,m.last_used_at,m.last_helpful_at,
                   m.record_role,m.role_method,m.role_version,m.role_reviewed_at,
                   p.display_title,p.display_summary,p.applies_when,p.retention_reason,
                   p.readability_flags_json,rm.eligibility recall_eligibility
                   FROM memories m
                   LEFT JOIN memory_presentations p ON p.memory_id=m.id
                   LEFT JOIN memory_recall_memberships rm
                     ON rm.memory_id=m.id AND rm.revoked_at IS NULL
                    AND rm.recall_set_id=(
                      SELECT recall_set_id FROM memory_recall_sets
                      WHERE status='active' ORDER BY activated_at DESC LIMIT 1
                    )
                   ORDER BY m.observed_at DESC LIMIT ?""",
                (max(1, min(memory_limit, 2000)),),
            ).fetchall()
            ids = [str(row["id"]) for row in memory_rows]
            edge_rows: list[sqlite3.Row] = []
            dependency_rows: list[sqlite3.Row] = []
            neighborhood_rows: list[sqlite3.Row] = []
            if ids:
                placeholders = ",".join("?" for _ in ids)
                edge_rows = self._conn.execute(
                    f"""SELECT e.src_id,e.dst_id,e.relation,e.weight,e.evidence_count,
                              e.last_reinforced_at,
                              COALESCE((SELECT ev.summary FROM edge_evidence ev
                                WHERE ev.src_id=e.src_id AND ev.dst_id=e.dst_id AND ev.relation=e.relation
                                ORDER BY ev.created_at DESC LIMIT 1),'') explanation,
                              COALESCE((SELECT ev.evidence_type FROM edge_evidence ev
                                WHERE ev.src_id=e.src_id AND ev.dst_id=e.dst_id AND ev.relation=e.relation
                                ORDER BY ev.created_at DESC LIMIT 1),'legacy_unattributed') evidence_type,
                              (SELECT COUNT(*) FROM edge_evidence ev
                                WHERE ev.src_id=e.src_id AND ev.dst_id=e.dst_id AND ev.relation=e.relation) evidence_records
                       FROM edges e WHERE e.src_id IN ({placeholders}) AND e.dst_id IN ({placeholders})
                       ORDER BY e.weight DESC,e.evidence_count DESC LIMIT 4000""",
                    (*ids, *ids),
                ).fetchall()
                dependency_rows = self._conn.execute(
                    f"""SELECT memory_id,evidence_id,relation,weight,active,created_at
                       FROM memory_dependencies
                       WHERE memory_id IN ({placeholders}) AND evidence_id IN ({placeholders})
                       LIMIT 4000""",
                    (*ids, *ids),
                ).fetchall()
                neighborhood_rows = self._conn.execute(
                    f"""SELECT nm.memory_id,n.slug,n.label,n.category,n.safety_class,
                               nm.confidence,nm.explanation
                        FROM memory_neighborhood_memberships nm
                        JOIN memory_neighborhoods n ON n.neighborhood_id=nm.neighborhood_id
                        WHERE nm.memory_id IN ({placeholders})
                        ORDER BY nm.memory_id,
                          CASE WHEN n.slug='credentials' THEN 0
                               WHEN n.category='project' THEN 1
                               WHEN n.category='service' THEN 2
                               WHEN n.slug='tools' THEN 3 ELSE 4 END,
                          nm.confidence DESC,n.label""",
                    tuple(ids),
                ).fetchall()
            tool_rows = self._conn.execute(
                """SELECT task_type,tool_name,success_count,failure_count,helpful_count,harmful_count,
                   argument_keys,last_success_at,last_failure_at,last_error_type,updated_at
                   FROM tool_stats ORDER BY success_count+failure_count DESC,updated_at DESC LIMIT 500"""
            ).fetchall()
            access_rows = self._conn.execute(
                """SELECT substr(created_at,1,10) day,event,COUNT(*) count
                   FROM access_log GROUP BY day,event ORDER BY day DESC LIMIT 500"""
            ).fetchall()
            memory_timeline_rows = self._conn.execute(
                """SELECT substr(observed_at,1,10) day,kind,COUNT(*) count
                   FROM memories
                   WHERE observed_at IS NOT NULL AND observed_at<>''
                   GROUP BY day,kind ORDER BY day,kind"""
            ).fetchall()
            memory_trend_rows = self._conn.execute(
                """SELECT substr(created_at,1,10) day,COUNT(*) count
                   FROM memories GROUP BY day ORDER BY day DESC LIMIT 365"""
            ).fetchall()
            connection_trend_rows = self._conn.execute(
                """SELECT substr(created_at,1,10) day,COUNT(*) count
                   FROM edges GROUP BY day ORDER BY day DESC LIMIT 365"""
            ).fetchall()
            prune_trend_rows = self._conn.execute(
                """SELECT substr(created_at,1,10) day,COUNT(*) count
                   FROM lifecycle_events
                   WHERE to_state IN ('cold','archived','tombstoned')
                   GROUP BY day ORDER BY day DESC LIMIT 365"""
            ).fetchall()
            tool_trend_rows = self._conn.execute(
                """SELECT substr(created_at,1,10) day,COUNT(*) count
                   FROM tool_executions GROUP BY day ORDER BY day DESC LIMIT 365"""
            ).fetchall()
            usage_rows = self._conn.execute(
                "SELECT outcome,COUNT(*) count FROM usage_records GROUP BY outcome"
            ).fetchall()
            source_rows = self._conn.execute(
                """SELECT source_path,source_type,title,modified_at,size_bytes,status,imported_at,last_seen_at
                   FROM document_sources ORDER BY modified_at DESC LIMIT 1000"""
            ).fetchall()
            recent_access_rows = self._conn.execute(
                """SELECT a.memory_id,a.event,a.score,a.created_at,m.kind,m.state,m.source_ref,m.content
                   FROM access_log a LEFT JOIN memories m ON m.id=a.memory_id
                   ORDER BY a.id DESC LIMIT 80"""
            ).fetchall()
            version_count = self._conn.execute("SELECT COUNT(*) count FROM memory_versions").fetchone()["count"]
            contradiction_rows = self._conn.execute(
                """SELECT e.src_id,e.dst_id,e.weight,e.evidence_count,e.created_at,e.last_reinforced_at,
                          src.content src_content,src.kind src_kind,src.source_category src_source_category,
                          src.source_ref src_source_ref,src.observed_at src_observed_at,
                          src.valid_from src_valid_from,src.valid_to src_valid_to,
                          src.confidence src_confidence,src.subject src_subject,src.predicate src_predicate,
                          src.object_value src_object_value,
                          dst.content dst_content,dst.kind dst_kind,dst.source_category dst_source_category,
                          dst.source_ref dst_source_ref,dst.observed_at dst_observed_at,
                          dst.valid_from dst_valid_from,dst.valid_to dst_valid_to,
                          dst.confidence dst_confidence,dst.subject dst_subject,dst.predicate dst_predicate,
                          dst.object_value dst_object_value
                   FROM edges e
                   JOIN memories src ON src.id=e.src_id
                   JOIN memories dst ON dst.id=e.dst_id
                   WHERE e.relation='contradicts'
                     AND src.state IN ('active','cold') AND dst.state IN ('active','cold')
                   ORDER BY e.last_reinforced_at DESC LIMIT 500"""
            ).fetchall()
            contradiction_count = self._conn.execute(
                """SELECT COUNT(*) count FROM edges e
                   JOIN memories src ON src.id=e.src_id
                   JOIN memories dst ON dst.id=e.dst_id
                   WHERE e.relation='contradicts'
                     AND src.state IN ('active','cold') AND dst.state IN ('active','cold')"""
            ).fetchone()["count"]
            unsupported_inference_rows = self._conn.execute(
                """SELECT m.id,m.content,m.kind,m.source_category,m.source_ref,m.observed_at,
                          m.valid_from,m.valid_to,m.confidence,m.subject,m.predicate,m.object_value
                   FROM memories m
                   WHERE m.source_category IN ('AGENT_INFERENCE','REFLECTION')
                   AND m.state IN ('active','cold')
                   AND NOT EXISTS(
                       SELECT 1 FROM memory_dependencies d
                       WHERE d.memory_id=m.id AND d.active=1
                   )
                   ORDER BY m.updated_at DESC LIMIT 2000"""
            ).fetchall()
            recall_rows = self._conn.execute(
                """SELECT recall_id,task_id,mode,reason,requested_limit,token_budget,candidate_count,
                          selected_count,estimated_tokens,prepare_ms,abstained,created_at
                   FROM recall_runs ORDER BY created_at DESC LIMIT 1000"""
            ).fetchall()
            recall_mode_rows = self._conn.execute(
                """SELECT mode,COUNT(*) count,SUM(abstained) abstained,
                          AVG(estimated_tokens) avg_tokens,AVG(prepare_ms) avg_prepare_ms
                   FROM recall_runs GROUP BY mode ORDER BY count DESC"""
            ).fetchall()
            budget_rows = self._conn.execute(
                """SELECT task_type,mode,COUNT(*) sample_count,
                          SUM(CASE WHEN used_count>0 THEN 1 ELSE 0 END) used_count,
                          SUM(CASE WHEN outcome IN ('helpful','validated') THEN 1 ELSE 0 END) helpful_count,
                          SUM(CASE WHEN outcome IN ('harmful','corrected') THEN 1 ELSE 0 END) harmful_count,
                          SUM(CASE WHEN outcome='ignored' THEN 1 ELSE 0 END) ignored_count,
                          AVG(requested_budget) avg_requested_budget,
                          AVG(estimated_tokens) avg_estimated_tokens,
                          MAX(COALESCE(resolved_at,created_at)) most_recent_at
                   FROM recall_budget_observations WHERE outcome<>'pending'
                   GROUP BY task_type,mode
                   ORDER BY sample_count DESC,most_recent_at DESC LIMIT 250"""
            ).fetchall()
            recall_day_rows = self._conn.execute(
                """SELECT substr(created_at,1,10) day,COUNT(*) count,SUM(abstained) abstained,
                          SUM(estimated_tokens) estimated_tokens,AVG(prepare_ms) avg_prepare_ms
                   FROM recall_runs GROUP BY day ORDER BY day DESC LIMIT 90"""
            ).fetchall()
            lifecycle_rows = self._conn.execute(
                """SELECT l.event_id,l.memory_id,l.from_state,l.to_state,l.reason,l.retention_score,
                          l.created_at,m.kind,m.content
                   FROM lifecycle_events l LEFT JOIN memories m ON m.id=l.memory_id
                   ORDER BY l.event_id DESC LIMIT 250"""
            ).fetchall()
            regret_rows = self._conn.execute(
                """SELECT r.regret_id,r.memory_id,r.query,r.score,r.restored,r.created_at,m.kind,m.content
                   FROM pruning_regret r LEFT JOIN memories m ON m.id=r.memory_id
                   ORDER BY r.regret_id DESC LIMIT 100"""
            ).fetchall()
            consolidation_rows = self._conn.execute(
                """SELECT run_id,dry_run,cluster_count,member_count,created_at
                   FROM consolidation_runs ORDER BY created_at DESC LIMIT 100"""
            ).fetchall()
            workflow_rows = self._conn.execute(
                """SELECT task_type,task_fingerprint,workflow_key,steps_json,success_count,failure_count,
                          distinct_tasks,last_error_type,updated_at
                   FROM tool_workflow_stats
                   ORDER BY success_count+failure_count DESC,updated_at DESC LIMIT 250"""
            ).fetchall()
            sleep_rows = self._conn.execute(
                """SELECT run_id,mode,status,cutoff_at,episodes_scanned,episodes_replayed,
                          association_proposals,interference_proposals,edge_decay_proposals,
                          lifecycle_candidates,consolidation_candidates,dependency_candidates,
                          applied_changes,reflection_token_budget,reflection_estimated_tokens,
                          reflection_billed_tokens,reflection_status,report_json,error,started_at,completed_at
                   FROM sleep_runs ORDER BY started_at DESC LIMIT 100"""
            ).fetchall()
            sleep_proposal_rows = self._conn.execute(
                """SELECT p.proposal_id,p.run_id,p.kind,p.src_id,p.dst_id,p.status,p.score,
                          p.evidence_count,p.rationale,p.details_json,p.created_at,
                          src.kind src_kind,src.state src_state,src.content src_content,
                          dst.kind dst_kind,dst.state dst_state,dst.content dst_content
                   FROM sleep_proposals p
                   LEFT JOIN memories src ON src.id=p.src_id
                   LEFT JOIN memories dst ON dst.id=p.dst_id
                   ORDER BY p.created_at DESC LIMIT 1000"""
            ).fetchall()
            sleep_edge_change_rows = self._conn.execute(
                """SELECT c.change_id,c.run_id,c.src_id,c.dst_id,c.relation,c.prior_exists,
                          c.prior_weight,c.prior_evidence_count,c.next_weight,
                          c.next_evidence_count,c.created_at,c.reversed_at,
                          src.kind src_kind,src.content src_content,
                          dst.kind dst_kind,dst.content dst_content
                   FROM sleep_edge_changes c
                   JOIN memories src ON src.id=c.src_id
                   JOIN memories dst ON dst.id=c.dst_id
                   ORDER BY c.change_id DESC LIMIT 1000"""
            ).fetchall()
            sleep_state_change_rows = self._conn.execute(
                """SELECT c.change_id,c.run_id,c.memory_id,c.prior_state,c.next_state,
                          c.created_at,c.reversed_at,m.kind memory_kind,m.content memory_content
                   FROM sleep_state_changes c
                   JOIN memories m ON m.id=c.memory_id
                   ORDER BY c.change_id DESC LIMIT 1000"""
            ).fetchall()
            sleep_effect_rows = self._conn.execute(
                """SELECT r.run_id,
                          (SELECT COUNT(*) FROM recall_runs rr
                           WHERE rr.created_at>=COALESCE(r.completed_at,r.started_at)
                             AND rr.created_at<COALESCE(
                               (SELECT MIN(n.started_at) FROM sleep_runs n WHERE n.started_at>r.started_at),
                               '9999-12-31T23:59:59.999+00:00')) recall_runs_after,
                          (SELECT COUNT(*) FROM recall_budget_observations b
                           WHERE b.outcome IN ('helpful','validated')
                             AND COALESCE(b.resolved_at,b.created_at)>=COALESCE(r.completed_at,r.started_at)
                             AND COALESCE(b.resolved_at,b.created_at)<COALESCE(
                               (SELECT MIN(n.started_at) FROM sleep_runs n WHERE n.started_at>r.started_at),
                               '9999-12-31T23:59:59.999+00:00')) helpful_outcomes_after,
                          (SELECT COUNT(*) FROM recall_budget_observations b
                           WHERE b.outcome IN ('harmful','corrected')
                             AND COALESCE(b.resolved_at,b.created_at)>=COALESCE(r.completed_at,r.started_at)
                             AND COALESCE(b.resolved_at,b.created_at)<COALESCE(
                               (SELECT MIN(n.started_at) FROM sleep_runs n WHERE n.started_at>r.started_at),
                               '9999-12-31T23:59:59.999+00:00')) harmful_outcomes_after,
                          (SELECT COUNT(*) FROM pruning_regret p
                           WHERE p.created_at>=COALESCE(r.completed_at,r.started_at)
                             AND p.created_at<COALESCE(
                               (SELECT MIN(n.started_at) FROM sleep_runs n WHERE n.started_at>r.started_at),
                               '9999-12-31T23:59:59.999+00:00')) pruning_regrets_after,
                          (SELECT COUNT(*) FROM sleep_edge_changes c
                           WHERE c.run_id=r.run_id AND c.reversed_at IS NULL) live_edge_changes,
                          (SELECT COUNT(*) FROM sleep_state_changes c
                           WHERE c.run_id=r.run_id AND c.reversed_at IS NULL) live_state_changes,
                          (SELECT COUNT(*) FROM sleep_edge_changes c
                           WHERE c.run_id=r.run_id AND c.reversed_at IS NOT NULL) reversed_edge_changes,
                          (SELECT COUNT(*) FROM sleep_state_changes c
                           WHERE c.run_id=r.run_id AND c.reversed_at IS NOT NULL) reversed_state_changes
                   FROM sleep_runs r ORDER BY r.started_at DESC LIMIT 100"""
            ).fetchall()
            capacity_impact_rows = self._conn.execute(
                """WITH RECURSIVE
                   observed_days(day) AS (
                     SELECT substr(created_at,1,10) FROM memories
                     UNION SELECT substr(created_at,1,10) FROM recall_runs
                     UNION SELECT substr(COALESCE(resolved_at,created_at),1,10)
                           FROM recall_budget_observations WHERE outcome<>'pending'
                   ),
                   bounds(start_day,end_day) AS (
                     SELECT MAX(COALESCE(MIN(day),date('now','-89 days')),date('now','-364 days')),
                            date('now') FROM observed_days WHERE day<>''
                   ),
                   days(day) AS (
                     SELECT start_day FROM bounds
                     UNION ALL SELECT date(day,'+1 day') FROM days,bounds WHERE day<end_day
                   )
                   SELECT d.day,
                          (SELECT COUNT(*) FROM memories m
                           WHERE m.created_at<datetime(d.day,'+1 day')) stored_capacity,
                          (SELECT COUNT(*) FROM memories m
                           WHERE substr(m.created_at,1,10)=d.day) memories_added,
                          (SELECT COUNT(*) FROM recall_runs r
                           WHERE substr(r.created_at,1,10)=d.day) recall_runs,
                          (SELECT COALESCE(SUM(r.abstained),0) FROM recall_runs r
                           WHERE substr(r.created_at,1,10)=d.day) abstained,
                          (SELECT AVG(r.estimated_tokens) FROM recall_runs r
                           WHERE substr(r.created_at,1,10)=d.day) avg_context_tokens,
                          (SELECT AVG(r.prepare_ms) FROM recall_runs r
                           WHERE substr(r.created_at,1,10)=d.day) avg_prepare_ms,
                          (SELECT COUNT(*) FROM recall_budget_observations b
                           WHERE b.outcome<>'pending'
                             AND substr(COALESCE(b.resolved_at,b.created_at),1,10)=d.day) resolved_outcomes,
                          (SELECT COUNT(*) FROM recall_budget_observations b
                           WHERE b.outcome IN ('helpful','validated')
                             AND substr(COALESCE(b.resolved_at,b.created_at),1,10)=d.day) helpful_outcomes,
                          (SELECT COUNT(*) FROM recall_budget_observations b
                           WHERE b.outcome IN ('harmful','corrected')
                             AND substr(COALESCE(b.resolved_at,b.created_at),1,10)=d.day) harmful_outcomes,
                          (SELECT COUNT(*) FROM recall_budget_observations b
                           WHERE b.outcome='ignored'
                             AND substr(COALESCE(b.resolved_at,b.created_at),1,10)=d.day) ignored_outcomes,
                          (SELECT COUNT(*) FROM sleep_runs s
                           WHERE substr(s.started_at,1,10)=d.day) sleep_runs
                   FROM days d ORDER BY d.day"""
            ).fetchall()
            metacognition_summary_row = self._conn.execute(
                """SELECT COUNT(*) prediction_count,
                          SUM(CASE WHEN decision='use' THEN 1 ELSE 0 END) use_count,
                          SUM(CASE WHEN decision='verify' THEN 1 ELSE 0 END) verify_count,
                          SUM(CASE WHEN decision='abstain' THEN 1 ELSE 0 END) abstain_count,
                          SUM(applied) applied_count,
                          SUM(CASE WHEN outcome IN ('helpful','validated','harmful','corrected') THEN 1 ELSE 0 END) labeled_count,
                          SUM(CASE WHEN outcome IN ('helpful','validated') THEN 1 ELSE 0 END) positive_count,
                          SUM(CASE WHEN outcome IN ('harmful','corrected') THEN 1 ELSE 0 END) negative_count,
                          SUM(CASE WHEN outcome='ignored' THEN 1 ELSE 0 END) ignored_count,
                          AVG(CASE
                            WHEN outcome IN ('helpful','validated')
                              THEN (1.0-calibrated_probability)*(1.0-calibrated_probability)
                            WHEN outcome IN ('harmful','corrected')
                              THEN calibrated_probability*calibrated_probability
                          END) brier_score,
                          AVG(calibrated_probability) avg_probability,
                          MAX(created_at) latest_prediction_at,
                          (SELECT monitor_mode FROM metacognitive_predictions
                           ORDER BY created_at DESC LIMIT 1) latest_mode
                   FROM metacognitive_predictions"""
            ).fetchone()
            metacognition_bin_rows = self._conn.execute(
                """SELECT CASE WHEN calibrated_probability>=1.0 THEN 4
                                 ELSE CAST(calibrated_probability*5 AS INTEGER) END bucket,
                          COUNT(*) sample_count,
                          AVG(calibrated_probability) avg_probability,
                          SUM(CASE WHEN outcome IN ('helpful','validated') THEN 1 ELSE 0 END) positive_count,
                          SUM(CASE WHEN outcome IN ('harmful','corrected') THEN 1 ELSE 0 END) negative_count
                   FROM metacognitive_predictions
                   WHERE outcome IN ('helpful','validated','harmful','corrected')
                   GROUP BY bucket ORDER BY bucket"""
            ).fetchall()
            metacognition_day_rows = self._conn.execute(
                """SELECT substr(created_at,1,10) day,COUNT(*) prediction_count,
                          SUM(CASE WHEN decision='use' THEN 1 ELSE 0 END) use_count,
                          SUM(CASE WHEN decision='verify' THEN 1 ELSE 0 END) verify_count,
                          SUM(CASE WHEN decision='abstain' THEN 1 ELSE 0 END) abstain_count,
                          SUM(CASE WHEN outcome IN ('helpful','validated','harmful','corrected') THEN 1 ELSE 0 END) labeled_count,
                          SUM(CASE WHEN outcome IN ('helpful','validated') THEN 1 ELSE 0 END) positive_count,
                          SUM(CASE WHEN outcome IN ('harmful','corrected') THEN 1 ELSE 0 END) negative_count,
                          AVG(calibrated_probability) avg_probability,
                          AVG(CASE
                            WHEN outcome IN ('helpful','validated')
                              THEN (1.0-calibrated_probability)*(1.0-calibrated_probability)
                            WHEN outcome IN ('harmful','corrected')
                              THEN calibrated_probability*calibrated_probability
                          END) brier_score
                   FROM metacognitive_predictions
                   GROUP BY day ORDER BY day DESC LIMIT 365"""
            ).fetchall()
            metacognition_source_rows = self._conn.execute(
                """SELECT source_category,COUNT(*) prediction_count,
                          AVG(calibrated_probability) avg_probability,
                          SUM(CASE WHEN outcome IN ('helpful','validated','harmful','corrected') THEN 1 ELSE 0 END) labeled_count,
                          SUM(CASE WHEN outcome IN ('helpful','validated') THEN 1 ELSE 0 END) positive_count,
                          SUM(CASE WHEN outcome IN ('harmful','corrected') THEN 1 ELSE 0 END) negative_count
                   FROM metacognitive_predictions
                   GROUP BY source_category ORDER BY prediction_count DESC"""
            ).fetchall()
            metacognition_recent_rows = self._conn.execute(
                """SELECT p.prediction_id,p.task_id,p.memory_id,p.task_type,p.recall_mode,
                          p.monitor_mode,p.source_category,p.raw_probability,p.calibrated_probability,
                          p.decision,p.applied,p.reason,p.calibration_scope,p.calibration_samples,
                          p.features_json,p.outcome,p.created_at,p.resolved_at,
                          m.kind memory_kind,m.state memory_state,m.content memory_content
                   FROM metacognitive_predictions p
                   JOIN memories m ON m.id=p.memory_id
                   ORDER BY p.created_at DESC LIMIT 250"""
            ).fetchall()
        prepare_samples = sorted(float(row["prepare_ms"]) for row in recall_rows)
        token_samples = sorted(int(row["estimated_tokens"]) for row in recall_rows)
        trend_days: dict[str, dict[str, Any]] = {}
        for field, rows in (
            ("memories_made", memory_trend_rows),
            ("connections_made", connection_trend_rows),
            ("memories_pruned", prune_trend_rows),
            ("tool_calls", tool_trend_rows),
        ):
            for row in rows:
                day = str(row["day"] or "")
                if not day:
                    continue
                trend_days.setdefault(
                    day,
                    {
                        "day": day,
                        "memories_made": 0,
                        "connections_made": 0,
                        "memories_pruned": 0,
                        "tool_calls": 0,
                    },
                )[field] = int(row["count"])
        sleep_runs: list[dict[str, Any]] = []
        for row in sleep_rows:
            item = dict(row)
            report = item.pop("report_json", "{}")
            try:
                parsed_report = json.loads(str(report or "{}"))
            except json.JSONDecodeError:
                parsed_report = {}
            item["report"] = parsed_report if isinstance(parsed_report, dict) else {}
            sleep_runs.append(item)
        metacognition_bins = [dict(row) for row in metacognition_bin_rows]
        labeled_count = sum(int(row["sample_count"] or 0) for row in metacognition_bin_rows)
        expected_calibration_error = (
            sum(
                int(row["sample_count"] or 0)
                * abs(
                    float(row["avg_probability"] or 0.0)
                    - int(row["positive_count"] or 0) / max(1, int(row["sample_count"] or 0))
                )
                for row in metacognition_bin_rows
            )
            / labeled_count
            if labeled_count
            else None
        )
        metacognition_summary = dict(metacognition_summary_row)
        metacognition_summary["expected_calibration_error"] = (
            round(expected_calibration_error, 6) if expected_calibration_error is not None else None
        )
        metacognition_recent: list[dict[str, Any]] = []
        for row in metacognition_recent_rows:
            item = dict(row)
            features = item.pop("features_json", "{}")
            try:
                parsed_features = json.loads(str(features or "{}"))
            except json.JSONDecodeError:
                parsed_features = {}
            item["features"] = parsed_features if isinstance(parsed_features, dict) else {}
            metacognition_recent.append(item)
        from .research import research_snapshot

        neighborhoods_by_memory: dict[str, list[dict[str, Any]]] = {}
        for row in neighborhood_rows:
            item = dict(row)
            neighborhoods_by_memory.setdefault(str(item.pop("memory_id")), []).append(item)
        decoded_memories: list[dict[str, Any]] = []
        for row in memory_rows:
            memory = self._decode_refinery_row(row)
            neighborhoods = neighborhoods_by_memory.get(str(memory["id"]), [])
            memory["neighborhoods"] = neighborhoods
            memory["primary_neighborhood"] = (
                str(neighborhoods[0]["label"]) if neighborhoods else "Ungrouped"
            )
            decoded_memories.append(memory)

        return {
            "generated_at": utc_now(),
            "stats": self.stats(),
            "refinery": self.refinery_summary(),
            "memories": decoded_memories,
            "recall_set": self.recall_set_snapshot(),
            "edges": [_decode_edge(row) for row in edge_rows],
            "dependencies": [dict(row) for row in dependency_rows],
            "tools": [dict(row) for row in tool_rows],
            "sources": [dict(row) for row in source_rows],
            "recent_access": [dict(row) for row in recent_access_rows],
            "access_by_day": [dict(row) for row in access_rows],
            "memory_timeline_by_day_kind": [dict(row) for row in memory_timeline_rows],
            "activity_trends": [trend_days[day] for day in sorted(trend_days)],
            "usage_outcomes": {str(row["outcome"]): int(row["count"]) for row in usage_rows},
            "recall_runs": [dict(row) for row in recall_rows],
            "recall_modes": [dict(row) for row in recall_mode_rows],
            "recall_budgets": [dict(row) for row in budget_rows],
            "recall_by_day": [dict(row) for row in recall_day_rows],
            "recall_summary": {
                "count": len(recall_rows),
                "abstained": sum(int(row["abstained"]) for row in recall_rows),
                "p50_prepare_ms": _percentile(prepare_samples, 0.50),
                "p95_prepare_ms": _percentile(prepare_samples, 0.95),
                "p50_tokens": _percentile(token_samples, 0.50),
                "p95_tokens": _percentile(token_samples, 0.95),
            },
            "memory_traces": self.memory_traces(limit=100),
            "memory_trace_summary": self.memory_trace_summary(limit=1000),
            "scoring_health": self.scoring_health(),
            "attention_learning": self.attention_learning_summary(),
            "semantic_consolidation": self.semantic_consolidation_snapshot(),
            "adaptive_pruning": self.adaptive_pruning_snapshot(),
            "scoring_weights": self.scoring_weight_snapshot(),
            "adaptive_reconsolidation": self.adaptive_reconsolidation_snapshot(),
            "schema_formation": self.schema_formation_snapshot(),
            "memory_write_decisions": self.memory_write_decisions(limit=100),
            "memory_write_summary": self.memory_write_summary(limit=1000),
            "context_feedback": self.context_feedback_summary(limit=1000),
            "lifecycle_events": [dict(row) for row in lifecycle_rows],
            "pruning_regrets": [dict(row) for row in regret_rows],
            "consolidation_runs": [dict(row) for row in consolidation_rows],
            "tool_workflows": [dict(row) for row in workflow_rows],
            "sleep_runs": sleep_runs,
            "sleep_proposals": [dict(row) for row in sleep_proposal_rows],
            "sleep_edge_changes": [dict(row) for row in sleep_edge_change_rows],
            "sleep_state_changes": [dict(row) for row in sleep_state_change_rows],
            "sleep_effects": {str(row["run_id"]): dict(row) for row in sleep_effect_rows},
            "sleep_hypotheses": self.sleep_hypotheses_snapshot(),
            "capacity_impact_by_day": [dict(row) for row in capacity_impact_rows],
            "outcome_lab": self.outcome_lab_snapshot(),
            "evaluations": self.evaluation_snapshot(),
            "evidence_hierarchy": self.evidence_hierarchy_snapshot(),
            "tool_evaluation": self.tool_evaluation_snapshot(),
            "research": research_snapshot(self),
            "metacognition": {
                "mode": str(metacognition_summary.get("latest_mode") or "shadow"),
                "summary": metacognition_summary,
                "enforcement_gate": self.metacognition_enforcement_gate(),
                "calibration_bins": metacognition_bins,
                "by_day": [dict(row) for row in metacognition_day_rows],
                "by_source": [dict(row) for row in metacognition_source_rows],
                "recent_predictions": metacognition_recent,
                "definitions": {
                    "positive_outcomes": ["helpful", "validated"],
                    "negative_outcomes": ["harmful", "corrected"],
                    "brier_score": "Mean squared error between predicted reliability and labeled outcome; lower is better.",
                    "expected_calibration_error": "Weighted gap between predicted reliability and observed helpfulness across five probability bands; lower is better.",
                },
            },
            "benchmarks": self.benchmark_snapshot(),
            "version_count": int(version_count),
            "contradiction_count": int(contradiction_count),
            "unsupported_inference_ids": [str(row["id"]) for row in unsupported_inference_rows],
            "health_reviews": {
                "contradictions": [dict(row) for row in contradiction_rows],
                "unsupported_inferences": [dict(row) for row in unsupported_inference_rows],
            },
            "review_inbox": self.review_inbox_snapshot(),
            "policy_training": self.policy_training_snapshot(),
            "decision_log": self.decision_log(limit=160),
            "learning_experiences": self.learning_experience_dataset(limit=160),
            "memory_hygiene": self.memory_hygiene_summary(),
            "audit": self.audit(),
        }

    def memory_quality_report(self) -> dict[str, Any]:
        """Return one compact, observable report for the memory feedback loop."""

        snapshot = self.dashboard_snapshot(memory_limit=1)
        context_feedback = snapshot["context_feedback"]
        context_items = list(context_feedback.get("items") or [])
        proposal_counts: dict[str, int] = {}
        for proposal in snapshot["sleep_proposals"]:
            if str(proposal.get("status")) not in {"proposed", "applied", "revert_review"}:
                continue
            kind = str(proposal.get("kind") or "unknown")
            proposal_counts[kind] = proposal_counts.get(kind, 0) + 1
        latest_sleep = snapshot["sleep_runs"][0] if snapshot["sleep_runs"] else None
        latest_effects = (
            snapshot["sleep_effects"].get(str(latest_sleep["run_id"])) if latest_sleep else None
        )
        states = dict(snapshot["stats"].get("states") or {})
        hygiene = snapshot["memory_hygiene"]
        return {
            "generated_at": snapshot["generated_at"],
            "retrieval": snapshot["memory_trace_summary"],
            "scoring": snapshot["scoring_health"],
            "attention": snapshot["attention_learning"],
            "storage": snapshot["memory_write_summary"],
            "context_adaptation": {
                "buckets": context_feedback.get("buckets", 0),
                "positive_buckets": context_feedback.get("positive_buckets", 0),
                "downweighted_buckets": context_feedback.get("downweighted_buckets", 0),
                "most_positive": sorted(
                    context_items, key=lambda item: float(item.get("usefulness") or 0.0), reverse=True
                )[:10],
                "most_downweighted": sorted(
                    context_items, key=lambda item: float(item.get("usefulness") or 0.0)
                )[:10],
            },
            "health": {
                "duplicate_review_candidates": proposal_counts.get("consolidation", 0),
                "active_contradictions": snapshot["contradiction_count"],
                "stale_or_inactive_items": int(states.get("cold", 0)) + int(states.get("archived", 0)),
                "context_review_candidates": proposal_counts.get("context_review", 0),
                "contextless_memories": snapshot["audit"]["contextless_memories"],
                "unsupported_inferences": len(snapshot["unsupported_inference_ids"]),
                "pruning_candidates": hygiene["candidate_count"],
                "pruning_candidates_by_reason": hygiene["by_reason"],
                "unexplained_links": hygiene["links"]["unexplained"],
                "weak_single_witness_links": hygiene["links"]["weak_single_witness"],
                "audit_ok": bool(snapshot["audit"]["ok"]),
            },
            "sleep": {
                "runs": len(snapshot["sleep_runs"]),
                "open_proposals_by_kind": proposal_counts,
                "latest_run": latest_sleep,
                "latest_observed_effect_window": latest_effects,
                "hypotheses": snapshot["sleep_hypotheses"].get(
                    str(latest_sleep["run_id"]), []
                )
                if latest_sleep
                else [],
            },
            "claim_boundary": (
                "This report measures storage, selection, attributed use, labeled outcomes, and maintenance "
                "effects. It does not claim that Cortex caused an answer or task outcome without a controlled test."
            ),
        }

    def memory_hygiene_summary(self) -> dict[str, Any]:
        """Profile deterministic memory noise and connection explainability."""

        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM memories WHERE pinned=0 AND protected=0
                   AND state IN ('active','cold')"""
            ).fetchall()
            edge_total = int(self._conn.execute("SELECT COUNT(*) n FROM edges").fetchone()["n"])
            unexplained = int(
                self._conn.execute(
                    """SELECT COUNT(*) n FROM edges e WHERE NOT EXISTS(
                         SELECT 1 FROM edge_evidence ev
                         WHERE ev.src_id=e.src_id AND ev.dst_id=e.dst_id AND ev.relation=e.relation
                           AND ev.evidence_type<>'legacy_unattributed'
                       )"""
                ).fetchone()["n"]
            )
            weak_single = int(
                self._conn.execute(
                    """SELECT COUNT(*) n FROM edges
                       WHERE relation IN ('related','co_observed','co_used','sleep_replay')
                         AND evidence_count<=1 AND weight<0.35"""
                ).fetchone()["n"]
            )
        candidates: list[dict[str, Any]] = []
        by_reason: dict[str, int] = {}
        by_state: dict[str, int] = {}
        for row in rows:
            item = dict(row)
            action = _memory_quality_pruning_action(item)
            if not action:
                continue
            next_state, reason = action
            if str(item.get("state") or "active") == next_state:
                continue
            source_type = str(item.get("source_type") or "")
            if source_type.casefold() in _TOOL_TELEMETRY_SOURCE_TYPES:
                category = "legacy_tool_telemetry"
            elif _is_placeholder_only_content(str(item.get("content") or "")):
                category = "placeholder_content"
            else:
                category = "transient_automation_status"
            by_reason[category] = by_reason.get(category, 0) + 1
            by_state[next_state] = by_state.get(next_state, 0) + 1
            candidates.append(
                {
                    "memory_id": str(item["id"]),
                    "source_type": source_type,
                    "kind": str(item.get("kind") or "semantic"),
                    "current_state": str(item.get("state") or "active"),
                    "recommended_state": next_state,
                    "reason_category": category,
                    "reason": reason,
                }
            )
        return {
            "candidate_count": len(candidates),
            "by_reason": by_reason,
            "by_recommended_state": by_state,
            "candidates": candidates[:500],
            "links": {
                "total": edge_total,
                "explained": max(0, edge_total - unexplained),
                "unexplained": unexplained,
                "weak_single_witness": weak_single,
            },
            "standards": {
                "tool_telemetry": "stored only in the dedicated tool ledger",
                "weak_association": "similarity alone does not create a persistent link",
                "deletion": "never automatic; archive and cold states remain reversible",
            },
        }

    def audit(self) -> dict[str, Any]:
        with self._lock:
            duplicate_groups = self._conn.execute(
                """SELECT content_hash,context_mode,scope_json,preconditions_json,
                          applicable_systems_json,applicable_versions_json,COUNT(*) n
                   FROM memories
                   GROUP BY content_hash,context_mode,scope_json,preconditions_json,
                            applicable_systems_json,applicable_versions_json
                   HAVING n>1"""
            ).fetchall()
            orphan_fts = self._conn.execute(
                "SELECT COUNT(*) n FROM memory_fts f LEFT JOIN memories m ON m.id=f.memory_id WHERE m.id IS NULL"
            ).fetchone()["n"]
            orphan_features = self._conn.execute(
                """SELECT COUNT(*) n FROM memory_features f
                   LEFT JOIN memories m ON m.id=f.memory_id WHERE m.id IS NULL"""
            ).fetchone()["n"]
            missing_fts = self._conn.execute(
                "SELECT COUNT(*) n FROM memories m LEFT JOIN memory_fts f ON f.memory_id=m.id WHERE f.memory_id IS NULL"
            ).fetchone()["n"]
            missing_features = self._conn.execute(
                """SELECT COUNT(*) n FROM memories m
                   WHERE NOT EXISTS(SELECT 1 FROM memory_features f WHERE f.memory_id=m.id)"""
            ).fetchone()["n"]
            missing_context_terms = self._conn.execute(
                """SELECT COUNT(*) n FROM memories m WHERE NOT EXISTS(
                     SELECT 1 FROM memory_context_terms t WHERE t.memory_id=m.id
                   )"""
            ).fetchone()["n"]
            orphan_context_terms = self._conn.execute(
                """SELECT COUNT(*) n FROM memory_context_terms t
                   LEFT JOIN memories m ON m.id=t.memory_id WHERE m.id IS NULL"""
            ).fetchone()["n"]
            invalid_context_terms = self._conn.execute(
                """SELECT COUNT(*) n FROM memory_context_terms
                   WHERE term_type NOT IN ('mode','scope','entity','precondition','system','version')
                      OR term_value=''"""
            ).fetchone()["n"]
            feature_stat_mismatches = self._conn.execute(
                """SELECT COUNT(*) n FROM (
                     SELECT f.feature,COUNT(*) actual,MAX(s.document_frequency) recorded
                     FROM memory_features f LEFT JOIN feature_stats s ON s.feature=f.feature
                     GROUP BY f.feature HAVING recorded IS NULL OR actual<>recorded
                   )"""
            ).fetchone()["n"]
            orphan_feature_stats = self._conn.execute(
                """SELECT COUNT(*) n FROM feature_stats s
                   WHERE NOT EXISTS(SELECT 1 FROM memory_features f WHERE f.feature=s.feature)"""
            ).fetchone()["n"]
            invalid = self._conn.execute(
                """SELECT COUNT(*) n FROM memories WHERE confidence<0 OR confidence>1 OR importance<0 OR importance>1
                   OR currentness_confidence<0 OR currentness_confidence>1 OR uniqueness<0 OR uniqueness>1"""
            ).fetchone()["n"]
            invalid_context_metadata = self._conn.execute(
                """SELECT COUNT(*) n FROM memories
                   WHERE context_mode NOT IN ('standalone','context_dependent')
                      OR metadata_completeness<0 OR metadata_completeness>1
                      OR NOT json_valid(scope_json) OR NOT json_valid(entities_json)
                      OR NOT json_valid(preconditions_json) OR NOT json_valid(applicable_systems_json)
                      OR NOT json_valid(applicable_versions_json)"""
            ).fetchone()["n"]
            contextless_memories = self._conn.execute(
                """SELECT COUNT(*) n FROM memories WHERE context_mode='context_dependent'
                   AND scope_json='{}' AND entities_json='[]' AND preconditions_json='{}'
                   AND applicable_systems_json='[]' AND applicable_versions_json='[]'"""
            ).fetchone()["n"]
            unsupported = self._conn.execute(
                """SELECT COUNT(*) n FROM memories m WHERE m.source_category IN ('AGENT_INFERENCE','REFLECTION')
                   AND m.state IN ('active','cold')
                   AND NOT EXISTS(SELECT 1 FROM memory_dependencies d WHERE d.memory_id=m.id AND d.active=1)"""
            ).fetchone()["n"]
            stuck_usage = self._conn.execute(
                "SELECT COUNT(*) n FROM usage_records WHERE outcome='pending' AND created_at < datetime('now','-1 day')"
            ).fetchone()["n"]
            orphan_document_chunks = self._conn.execute(
                """SELECT COUNT(*) n FROM document_chunks c
                   LEFT JOIN memories m ON m.id=c.memory_id WHERE m.id IS NULL"""
            ).fetchone()["n"]
            orphan_sleep_proposals = self._conn.execute(
                """SELECT COUNT(*) n FROM sleep_proposals p
                   LEFT JOIN sleep_runs r ON r.run_id=p.run_id WHERE r.run_id IS NULL"""
            ).fetchone()["n"]
            orphan_experiment_assignments = self._conn.execute(
                """SELECT COUNT(*) n FROM recall_experiment_assignments a
                   LEFT JOIN controlled_experiments e ON e.experiment_id=a.experiment_id
                   WHERE e.experiment_id IS NULL"""
            ).fetchone()["n"]
            orphan_agent_assignments = self._conn.execute(
                """SELECT COUNT(*) n FROM agent_task_observations t
                   LEFT JOIN recall_experiment_assignments a ON a.assignment_id=t.experiment_assignment_id
                   WHERE t.experiment_assignment_id IS NOT NULL AND a.assignment_id IS NULL"""
            ).fetchone()["n"]
            orphan_prospective_items = self._conn.execute(
                """SELECT COUNT(*) n FROM prospective_items p
                   LEFT JOIN memories m ON m.id=p.memory_id WHERE m.id IS NULL OR m.kind<>'prospective'"""
            ).fetchone()["n"]
            orphan_trace_events = self._conn.execute(
                """SELECT COUNT(*) n FROM memory_trace_events e
                   LEFT JOIN memory_traces t ON t.task_id=e.task_id WHERE t.task_id IS NULL"""
            ).fetchone()["n"]
            invalid_write_decisions = self._conn.execute(
                """SELECT COUNT(*) n FROM memory_write_decisions
                   WHERE decision NOT IN ('created','updated','ignored')
                      OR durability NOT IN ('temporary','durable')
                      OR reusable_score<0 OR reusable_score>1
                      OR source_type='' OR NOT json_valid(quality_flags_json)"""
            ).fetchone()["n"]
            orphan_write_decisions = self._conn.execute(
                """SELECT COUNT(*) n FROM memory_write_decisions d
                   LEFT JOIN memories m ON m.id=d.memory_id
                   WHERE d.memory_id IS NOT NULL AND m.id IS NULL"""
            ).fetchone()["n"]
            invalid_context_outcomes = self._conn.execute(
                """SELECT COUNT(*) n FROM memory_context_outcomes
                   WHERE selected NOT IN (0,1) OR used NOT IN (0,1) OR used>selected
                      OR outcome NOT IN ('pending','used','ignored','helpful','validated','harmful','corrected')
                      OR NOT json_valid(context_json)"""
            ).fetchone()["n"]
            orphan_context_outcomes = self._conn.execute(
                """SELECT COUNT(*) n FROM memory_context_outcomes o
                   LEFT JOIN memory_traces t ON t.task_id=o.task_id
                   LEFT JOIN memories m ON m.id=o.memory_id
                   WHERE t.task_id IS NULL OR m.id IS NULL"""
            ).fetchone()["n"]
            invalid_edge_evidence = self._conn.execute(
                """SELECT COUNT(*) n FROM edge_evidence
                   WHERE evidence_type='' OR evidence_key='' OR summary='' OR NOT json_valid(metadata_json)"""
            ).fetchone()["n"]
            unexplained_edges = self._conn.execute(
                """SELECT COUNT(*) n FROM edges e WHERE NOT EXISTS(
                     SELECT 1 FROM edge_evidence ev
                     WHERE ev.src_id=e.src_id AND ev.dst_id=e.dst_id AND ev.relation=e.relation
                       AND ev.evidence_type<>'legacy_unattributed'
                   )"""
            ).fetchone()["n"]
            invalid_record_roles = self._conn.execute(
                """SELECT COUNT(*) n FROM memories
                   WHERE record_role NOT IN ('canonical','reference','event','claim')
                      OR role_method=''"""
            ).fetchone()["n"]
            invalid_provenance = self._conn.execute(
                """SELECT COUNT(*) n FROM memories
                   WHERE origin_source_category IS NULL OR origin_source_category=''
                      OR approval_state NOT IN
                        ('unreviewed','operator_approved','automatic_approved','trusted_import')"""
            ).fetchone()["n"]
            invalid_experience_events = self._conn.execute(
                """SELECT COUNT(*) n FROM memory_experience_events
                   WHERE event_type NOT IN ('candidate_observed','memory_selected','memory_used','outcome_labeled')
                      OR (proposal_id IS NULL AND memory_id IS NULL)
                      OR evidence_key='' OR event_day='' OR NOT json_valid(metadata_json)"""
            ).fetchone()["n"]
            invalid_neighborhood_evaluations = self._conn.execute(
                """SELECT COUNT(*) n FROM memory_neighborhood_evaluations
                   WHERE category NOT IN ('project','service')
                      OR decision NOT IN ('admitted','held','inactive')
                      OR memory_count<0 OR context_count<0 OR reason=''"""
            ).fetchone()["n"]
            invalid_neighborhood_events = self._conn.execute(
                """SELECT COUNT(*) n FROM memory_neighborhood_decision_events
                   WHERE category NOT IN ('project','service')
                      OR decision NOT IN ('admitted','held','inactive')
                      OR memory_count<0 OR context_count<0 OR reason=''"""
            ).fetchone()["n"]
            orphan_presentations = self._conn.execute(
                """SELECT COUNT(*) n FROM memory_presentations p
                   LEFT JOIN memories m ON m.id=p.memory_id WHERE m.id IS NULL"""
            ).fetchone()["n"]
            missing_presentations = self._conn.execute(
                """SELECT COUNT(*) n FROM memories m
                   WHERE NOT EXISTS(SELECT 1 FROM memory_presentations p WHERE p.memory_id=m.id)"""
            ).fetchone()["n"]
            stale_presentations = self._conn.execute(
                """SELECT COUNT(*) n FROM memory_presentations p
                   JOIN memories m ON m.id=p.memory_id
                   WHERE p.source_digest<>m.content_hash"""
            ).fetchone()["n"]
            orphan_refinery_proposals = self._conn.execute(
                """SELECT COUNT(*) n FROM memory_refinery_proposals r
                   WHERE r.review_id IS NOT NULL AND NOT EXISTS(
                     SELECT 1 FROM operator_review_decisions d WHERE d.review_id=r.review_id
                   )"""
            ).fetchone()["n"]
            invalid_adaptive_pruning = self._conn.execute(
                """SELECT COUNT(*) n FROM adaptive_pruning_decisions
                   WHERE NOT json_valid(score_evidence_json)
                      OR NOT json_valid(edge_snapshot_json)
                      OR relevance_score<0 OR relevance_score>1
                      OR confidence<0 OR confidence>1"""
            ).fetchone()["n"]
            invalid_scoring_weights = self._conn.execute(
                """SELECT COUNT(*) n FROM scoring_weight_proposals
                   WHERE NOT json_valid(baseline_weights_json)
                      OR NOT json_valid(proposed_weights_json)
                      OR NOT json_valid(evidence_json)
                      OR confidence<0 OR confidence>1
                      OR max_deviation<0
                   """
            ).fetchone()["n"]
            invalid_active_scoring_weights = self._conn.execute(
                """SELECT COUNT(*) n FROM scoring_weight_history h
                   WHERE h.status='active' AND CASE
                     WHEN NOT json_valid(h.weights_json) THEN 1
                     ELSE EXISTS(
                       SELECT 1 FROM json_each(h.weights_json) w
                       WHERE CAST(w.value AS REAL)<0.02 OR CAST(w.value AS REAL)>0.30
                     )
                   END"""
            ).fetchone()["n"]
            invalid_adaptive_reconsolidation = self._conn.execute(
                """SELECT COUNT(*) n FROM adaptive_reconsolidation_proposals
                   WHERE NOT json_valid(source_snapshot_json)
                      OR NOT json_valid(edge_snapshot_json)
                      OR confidence<0 OR confidence>1
                      OR (action='supersede' AND COALESCE(replacement_content,'')='')"""
            ).fetchone()["n"]
            invalid_schema_formations = self._conn.execute(
                """SELECT COUNT(*) n FROM schema_formation_proposals p
                   LEFT JOIN memories m ON m.id=p.result_memory_id
                   WHERE NOT json_valid(p.source_ids_json)
                      OR NOT json_valid(p.included_source_ids_json)
                      OR NOT json_valid(p.source_hashes_json)
                      OR NOT json_valid(p.evidence_json)
                      OR p.confidence<0 OR p.confidence>1
                      OR (p.action IN ('abstract','partial') AND COALESCE(p.abstract_content,'')='')
                      OR (p.status='applied' AND (m.id IS NULL OR m.kind<>'schema'))
                      OR (p.status='applied' AND (
                           SELECT COUNT(*) FROM memory_dependencies d
                           WHERE d.memory_id=p.result_memory_id
                             AND d.relation='schema_source' AND d.active=1
                         )<3)"""
            ).fetchone()["n"]
            invalid_strands = self._conn.execute(
                """SELECT COUNT(*) n FROM memory_strands s
                   JOIN memories m ON m.id=s.memory_id
                   WHERE NOT json_valid(s.edge_snapshot_json)
                      OR (s.status='active' AND m.stranded<>1)
                      OR (s.status<>'active' AND m.stranded=1)"""
            ).fetchone()["n"]
        return {
            "ok": not (
                duplicate_groups
                or orphan_fts
                or orphan_features
                or missing_fts
                or missing_features
                or missing_context_terms
                or orphan_context_terms
                or invalid_context_terms
                or feature_stat_mismatches
                or orphan_feature_stats
                or invalid
                or invalid_context_metadata
                or contextless_memories
                or stuck_usage
                or orphan_document_chunks
                or orphan_sleep_proposals
                or orphan_experiment_assignments
                or orphan_agent_assignments
                or orphan_prospective_items
                or orphan_trace_events
                or invalid_write_decisions
                or orphan_write_decisions
                or invalid_context_outcomes
                or orphan_context_outcomes
                or invalid_edge_evidence
                or invalid_record_roles
                or invalid_provenance
                or invalid_experience_events
                or invalid_neighborhood_evaluations
                or invalid_neighborhood_events
                or orphan_presentations
                or missing_presentations
                or stale_presentations
                or orphan_refinery_proposals
                or invalid_adaptive_pruning
                or invalid_scoring_weights
                or invalid_active_scoring_weights
                or invalid_adaptive_reconsolidation
                or invalid_schema_formations
                or invalid_strands
            ),
            "duplicate_groups": len(duplicate_groups),
            "orphan_fts_rows": orphan_fts,
            "orphan_feature_rows": orphan_features,
            "missing_fts_rows": missing_fts,
            "missing_feature_memories": missing_features,
            "missing_memory_context_terms": missing_context_terms,
            "orphan_memory_context_terms": orphan_context_terms,
            "invalid_memory_context_terms": invalid_context_terms,
            "feature_stat_mismatches": feature_stat_mismatches,
            "orphan_feature_stats": orphan_feature_stats,
            "invalid_score_rows": invalid,
            "invalid_context_metadata": invalid_context_metadata,
            "contextless_memories": contextless_memories,
            "unsupported_inferences": unsupported,
            "stuck_pending_usage": stuck_usage,
            "orphan_document_chunks": orphan_document_chunks,
            "orphan_sleep_proposals": orphan_sleep_proposals,
            "orphan_experiment_assignments": orphan_experiment_assignments,
            "orphan_agent_assignments": orphan_agent_assignments,
            "orphan_prospective_items": orphan_prospective_items,
            "orphan_memory_trace_events": orphan_trace_events,
            "invalid_memory_write_decisions": invalid_write_decisions,
            "orphan_memory_write_decisions": orphan_write_decisions,
            "invalid_memory_context_outcomes": invalid_context_outcomes,
            "orphan_memory_context_outcomes": orphan_context_outcomes,
            "invalid_edge_evidence": invalid_edge_evidence,
            "unexplained_edges": unexplained_edges,
            "invalid_record_roles": invalid_record_roles,
            "invalid_memory_provenance": invalid_provenance,
            "invalid_memory_experience_events": invalid_experience_events,
            "invalid_neighborhood_evaluations": invalid_neighborhood_evaluations,
            "invalid_neighborhood_decision_events": invalid_neighborhood_events,
            "orphan_memory_presentations": orphan_presentations,
            "missing_memory_presentations": missing_presentations,
            "stale_memory_presentations": stale_presentations,
            "orphan_refinery_proposals": orphan_refinery_proposals,
            "invalid_adaptive_pruning": invalid_adaptive_pruning,
            "invalid_scoring_weights": invalid_scoring_weights,
            "invalid_active_scoring_weights": invalid_active_scoring_weights,
            "invalid_adaptive_reconsolidation": invalid_adaptive_reconsolidation,
            "invalid_schema_formations": invalid_schema_formations,
            "invalid_memory_strands": invalid_strands,
        }

    def semantic_consolidation_candidates(
        self,
        *,
        limit: int = 5,
        scan_limit: int = 500,
        lexical_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Return deterministic, safety-filtered pairs for model review.

        This is intentionally separate from :meth:`consolidate`, which keeps
        handling near-exact duplicate folding.  Candidate generation never
        mutates memories and excludes protected, recall-ineligible, quarantined,
        and contradiction-bearing records.
        """

        bounded_limit = max(1, min(int(limit), 5))
        bounded_scan = max(bounded_limit, min(int(scan_limit), 2000))
        threshold = max(0.0, min(float(lexical_threshold), 1.0))
        with self._lock:
            rows = [
                _decode_memory_metadata(row)
                for row in self._conn.execute(
                    """SELECT DISTINCT m.*,
                              (SELECT COUNT(*) FROM edges e
                               WHERE e.src_id=m.id OR e.dst_id=m.id) edge_count
                       FROM memories m
                       JOIN memory_recall_memberships rm
                         ON rm.memory_id=m.id AND rm.revoked_at IS NULL
                        AND rm.eligibility='primary'
                       JOIN memory_recall_sets rs
                         ON rs.recall_set_id=rm.recall_set_id AND rs.status='active'
                       WHERE m.state IN ('active','cold')
                         AND m.pinned=0 AND m.protected=0
                         AND m.quarantine_reason IS NULL
                         AND m.kind NOT IN ('identity','preference','prospective')
                       ORDER BY m.created_at ASC,m.id ASC LIMIT ?""",
                    (bounded_scan,),
                ).fetchall()
            ]
            contradiction_ids = {
                str(row["memory_id"])
                for row in self._conn.execute(
                    """SELECT src_id memory_id FROM edges WHERE relation='contradicts'
                       UNION SELECT dst_id memory_id FROM edges WHERE relation='contradicts'"""
                ).fetchall()
            }
            graph_pairs = {
                tuple(sorted((str(row["src_id"]), str(row["dst_id"]))))
                for row in self._conn.execute(
                    """SELECT src_id,dst_id FROM edges
                       WHERE relation IN ('supports','extends')"""
                ).fetchall()
            }
            decided_pairs = {
                tuple(sorted((str(row["left_id"]), str(row["right_id"]))))
                for row in self._conn.execute(
                    """SELECT left_id,right_id FROM semantic_consolidation_decisions
                       WHERE status IN ('proposed','applied','kept_separate','linked')"""
                ).fetchall()
            }

        eligible = [row for row in rows if str(row["id"]) not in contradiction_ids]
        candidates: list[dict[str, Any]] = []
        for left_index, left in enumerate(eligible):
            left_id = str(left["id"])
            left_tokens = set(query_tokens(str(left.get("content") or "")))
            left_entities = {item.casefold() for item in left.get("entities") or []}
            for right in eligible[left_index + 1 :]:
                right_id = str(right["id"])
                pair_key = tuple(sorted((left_id, right_id)))
                if pair_key in decided_pairs:
                    continue
                if _memory_applicability_key(left) != _memory_applicability_key(right):
                    continue
                # A different value for the same structured claim is a conflict,
                # not consolidation material, even if a legacy database lacks
                # an explicit contradiction edge.
                same_subject = bool(left.get("subject")) and (
                    str(left.get("subject")).casefold() == str(right.get("subject") or "").casefold()
                )
                same_predicate = bool(left.get("predicate")) and (
                    str(left.get("predicate")).casefold()
                    == str(right.get("predicate") or "").casefold()
                )
                if (
                    same_subject
                    and same_predicate
                    and left.get("object_value") is not None
                    and right.get("object_value") is not None
                    and str(left.get("object_value")).casefold()
                    != str(right.get("object_value")).casefold()
                ):
                    continue

                right_tokens = set(query_tokens(str(right.get("content") or "")))
                union = left_tokens | right_tokens
                lexical_overlap = len(left_tokens & right_tokens) / len(union) if union else 0.0
                right_entities = {item.casefold() for item in right.get("entities") or []}
                shared_entities = sorted(left_entities & right_entities)
                evidence: list[str] = []
                if str(left.get("kind")) == str(right.get("kind")) and lexical_overlap >= threshold:
                    evidence.append("same_kind_lexical_overlap")
                if same_subject and not same_predicate:
                    evidence.append("shared_subject_supplementary_claims")
                if shared_entities and (
                    str(left.get("predicate") or "").casefold()
                    != str(right.get("predicate") or "").casefold()
                    or lexical_overlap >= 0.2
                ):
                    evidence.append("shared_entities_related_predicates")
                if pair_key in graph_pairs:
                    evidence.append("supports_or_extends_graph_neighbors")
                if not evidence:
                    continue
                candidates.append(
                    {
                        "pair_id": f"{left_id}:{right_id}",
                        "left": left,
                        "right": right,
                        "lexical_overlap": round(lexical_overlap, 6),
                        "shared_entities": shared_entities[:20],
                        "candidate_evidence": evidence,
                        "oldest_at": min(str(left["created_at"]), str(right["created_at"])),
                    }
                )

        candidates.sort(
            key=lambda item: (
                str(item["oldest_at"]),
                str(item["left"]["id"]),
                str(item["right"]["id"]),
            )
        )
        return candidates[:bounded_limit]

    def semantic_consolidation_snapshot(self, *, limit: int = 100) -> dict[str, Any]:
        bounded = max(1, min(int(limit), 500))
        with self._lock:
            runs = [
                dict(row)
                for row in self._conn.execute(
                    """SELECT * FROM semantic_consolidation_runs
                       ORDER BY started_at DESC LIMIT ?""",
                    (bounded,),
                ).fetchall()
            ]
            decisions = [
                dict(row)
                for row in self._conn.execute(
                    """SELECT d.*,lm.content left_content,rm.content right_content,
                              nm.content result_content,
                              (SELECT f.label FROM semantic_consolidation_feedback f
                               WHERE f.decision_id=d.decision_id
                               ORDER BY f.created_at DESC LIMIT 1) feedback_label
                       FROM semantic_consolidation_decisions d
                       JOIN memories lm ON lm.id=d.left_id
                       JOIN memories rm ON rm.id=d.right_id
                       LEFT JOIN memories nm ON nm.id=d.result_memory_id
                       ORDER BY d.created_at DESC LIMIT ?""",
                    (bounded,),
                ).fetchall()
            ]
            count_rows = self._conn.execute(
                """SELECT status,COUNT(*) count FROM semantic_consolidation_decisions
                   GROUP BY status"""
            ).fetchall()
            feedback_rows = self._conn.execute(
                """SELECT label,COUNT(*) count FROM semantic_consolidation_feedback
                   GROUP BY label"""
            ).fetchall()
        for run in runs:
            run["usage"] = _json_object(run.pop("usage_json", "{}"))
        for decision in decisions:
            decision["candidate_evidence"] = _json_string_list(
                decision.pop("candidate_evidence_json", "[]")
            )
            decision.pop("source_snapshot_json", None)
        counts = {
            str(row["status"]): int(row["count"] or 0)
            for row in count_rows
        }
        feedback = {
            str(row["label"]): int(row["count"] or 0)
            for row in feedback_rows
        }
        reviewed = sum(feedback.values())
        correctness = feedback.get("correct", 0) / reviewed if reviewed else None
        return {
            "policy_version": "semantic_consolidation_v1",
            "default_mode": "shadow",
            "batch_limit": 5,
            "counts": counts,
            "feedback": {
                **feedback,
                "reviewed": reviewed,
                "correctness": round(correctness, 6) if correctness is not None else None,
                "target": 0.8,
                "ready": reviewed > 0 and correctness is not None and correctness >= 0.8,
            },
            "runs": runs,
            "decisions": decisions,
            "claim_boundary": (
                "Proposed model judgments are not applied merges. Applied decisions remain "
                "separately labeled and reversible."
            ),
        }

    @staticmethod
    def _refresh_semantic_consolidation_run_tx(
        conn: sqlite3.Connection,
        run_id: str,
    ) -> None:
        conn.execute(
            """UPDATE semantic_consolidation_runs
               SET judgment_count=(
                     SELECT COUNT(*) FROM semantic_consolidation_decisions WHERE run_id=?
                   ),
                   merge_count=(
                     SELECT COUNT(*) FROM semantic_consolidation_decisions
                     WHERE run_id=? AND action='merge'
                   ),
                   keep_count=(
                     SELECT COUNT(*) FROM semantic_consolidation_decisions
                     WHERE run_id=? AND action='keep_separate'
                   ),
                   link_count=(
                     SELECT COUNT(*) FROM semantic_consolidation_decisions
                     WHERE run_id=? AND action='link_as_related'
                   ),
                   applied_count=(
                     SELECT COUNT(*) FROM semantic_consolidation_decisions
                     WHERE run_id=? AND status IN ('applied','kept_separate','linked')
                   )
               WHERE run_id=?""",
            (run_id, run_id, run_id, run_id, run_id, run_id),
        )

    def apply_semantic_consolidation(
        self,
        decision_id: str,
        *,
        actor: str = "cortex-auto-judge",
    ) -> dict[str, Any]:
        """Apply one previously recorded semantic decision after stale checks."""

        with self._lock:
            raw_decision = self._conn.execute(
                """SELECT * FROM semantic_consolidation_decisions
                   WHERE decision_id=?""",
                (decision_id,),
            ).fetchone()
        if not raw_decision:
            raise ValueError("semantic consolidation decision not found")
        decision = dict(raw_decision)
        if str(decision["status"]) != "proposed":
            raise ValueError("semantic consolidation decision is not proposed")
        left = self.get_memory(str(decision["left_id"]))
        right = self.get_memory(str(decision["right_id"]))
        if not left or not right:
            raise ValueError("semantic consolidation source memory is unavailable")
        if (
            not self.is_memory_recall_eligible(str(left["id"]))
            or not self.is_memory_recall_eligible(str(right["id"]))
            or not _semantic_consolidation_sources_are_safe(left, right, decision)
        ):
            with self.transaction() as conn:
                conn.execute(
                    """UPDATE semantic_consolidation_decisions
                       SET status='skipped' WHERE decision_id=? AND status='proposed'""",
                    (decision_id,),
                )
                self._refresh_semantic_consolidation_run_tx(conn, str(decision["run_id"]))
            return {"decision_id": decision_id, "status": "skipped", "reason": "stale or unsafe sources"}

        action = str(decision["action"])
        now = utc_now()
        if action == "keep_separate":
            with self.transaction() as conn:
                conn.execute(
                    """UPDATE semantic_consolidation_decisions
                       SET status='kept_separate',applied_at=?
                       WHERE decision_id=? AND status='proposed'""",
                    (now, decision_id),
                )
                self._refresh_semantic_consolidation_run_tx(conn, str(decision["run_id"]))
            return {"decision_id": decision_id, "status": "kept_separate"}
        if action == "link_as_related":
            created = self.add_edge(
                str(left["id"]),
                str(right["id"]),
                "related",
                weight=max(0.1, min(float(decision["confidence"]), 0.7)),
                evidence_type="semantic_consolidation_judgment",
                evidence_key=decision_id,
                explanation=str(decision["reason"]),
                source_ref=f"semantic-consolidation:{decision_id}",
                metadata={"decision_id": decision_id, "actor": actor},
            )
            with self.transaction() as conn:
                conn.execute(
                    """UPDATE semantic_consolidation_decisions
                       SET status='linked',applied_at=?
                       WHERE decision_id=? AND status='proposed'""",
                    (now, decision_id),
                )
                self._refresh_semantic_consolidation_run_tx(conn, str(decision["run_id"]))
            return {"decision_id": decision_id, "status": "linked", "edge_created": bool(created)}

        merged_content = normalize_text(str(decision.get("merged_content") or ""))
        if not merged_content:
            raise ValueError("merge decision is missing consolidated content")
        entities = _normalize_context_list(
            [*(left.get("entities") or []), *(right.get("entities") or [])]
        )
        source_snapshot = str(decision.get("source_snapshot_json") or "")
        if not _json_object(source_snapshot):
            source_snapshot = json.dumps(
                {"left": left, "right": right, "edges": [], "edge_evidence": []},
                ensure_ascii=True,
                sort_keys=True,
                default=str,
            )
        result_memory_id, created = self.add_memory(
            merged_content,
            kind=str(left["kind"]) if left["kind"] == right["kind"] else "semantic",
            source_type="semantic_consolidation",
            source_category="AUTOMATIC_APPROVED",
            origin_source_category="AUTOMATIC_APPROVED",
            approval_state="automatic_approved",
            source_ref=f"semantic-consolidation:{decision_id}",
            context_mode=str(left.get("context_mode") or "standalone"),
            scope=dict(left.get("scope") or {}),
            entities=entities,
            preconditions=dict(left.get("preconditions") or {}),
            source_context=(
                f"Consolidated from {str(left['id'])[:8]} and {str(right['id'])[:8]}; "
                "the decision ledger retains both source snapshots and their edge evidence."
            ),
            applicable_systems=list(left.get("applicable_systems") or []),
            applicable_versions=list(left.get("applicable_versions") or []),
            confidence=max(float(left.get("confidence") or 0.0), float(right.get("confidence") or 0.0)),
            currentness_confidence=max(
                float(left.get("currentness_confidence") or 0.0),
                float(right.get("currentness_confidence") or 0.0),
            ),
            importance=max(float(left.get("importance") or 0.0), float(right.get("importance") or 0.0)),
            uniqueness=max(float(left.get("uniqueness") or 0.0), float(right.get("uniqueness") or 0.0)),
            volatility=max(float(left.get("volatility") or 0.0), float(right.get("volatility") or 0.0)),
            trust=max(float(left.get("trust") or 0.0), float(right.get("trust") or 0.0)),
            subject=left.get("subject") if left.get("subject") == right.get("subject") else None,
            predicate=left.get("predicate") if left.get("predicate") == right.get("predicate") else None,
            object_value=(
                left.get("object_value")
                if left.get("object_value") == right.get("object_value")
                else None
            ),
            extraction_method="autojudge_semantic_consolidation",
            evidence_ids=(str(left["id"]), str(right["id"])),
            storage_policy="trusted",
            record_role="canonical",
            preserve_exact_duplicate=True,
        )
        with self.transaction() as conn:
            current_left = conn.execute("SELECT * FROM memories WHERE id=?", (left["id"],)).fetchone()
            current_right = conn.execute("SELECT * FROM memories WHERE id=?", (right["id"],)).fetchone()
            if not current_left or not current_right or not _semantic_consolidation_sources_are_safe(
                dict(current_left), dict(current_right), decision
            ):
                self._refinery_state_change_tx(
                    conn,
                    dict(conn.execute("SELECT * FROM memories WHERE id=?", (result_memory_id,)).fetchone()),
                    "archived",
                    f"semantic consolidation {decision_id[:8]} became stale before apply",
                    now,
                )
                conn.execute(
                    """UPDATE semantic_consolidation_decisions
                       SET status='skipped',result_memory_id=?,source_snapshot_json=?
                       WHERE decision_id=?""",
                    (result_memory_id, source_snapshot, decision_id),
                )
                self._refresh_semantic_consolidation_run_tx(conn, str(decision["run_id"]))
                return {
                    "decision_id": decision_id,
                    "status": "skipped",
                    "reason": "sources changed during apply",
                    "result_memory_id": result_memory_id,
                }
            for source in (dict(current_left), dict(current_right)):
                self._refinery_state_change_tx(
                    conn,
                    source,
                    "archived",
                    f"reversible semantic consolidation into {result_memory_id[:8]}",
                    now,
                )
            # These dependencies are preserved lineage, not evidence lost by
            # the intentional source archive. Future corrections still dirty
            # the result, while dependency repair recognizes archived
            # consolidated sources as retained evidence.
            conn.execute(
                """UPDATE memory_dependencies
                   SET relation='consolidated_from'
                   WHERE memory_id=? AND evidence_id IN (?,?)
                     AND relation='derived_from'""",
                (result_memory_id, left["id"], right["id"]),
            )
            conn.execute(
                """UPDATE memories SET dirty=0,dirty_reason=NULL
                   WHERE id=?""",
                (result_memory_id,),
            )
            conn.execute(
                """UPDATE semantic_consolidation_decisions
                   SET status='applied',result_memory_id=?,source_snapshot_json=?,applied_at=?
                   WHERE decision_id=? AND status='proposed'""",
                (result_memory_id, source_snapshot, now, decision_id),
            )
            self._refresh_semantic_consolidation_run_tx(conn, str(decision["run_id"]))
        for source in (left, right):
            self.add_edge(
                result_memory_id,
                str(source["id"]),
                "consolidates",
                weight=max(0.1, min(float(decision["confidence"]), 1.0)),
                evidence_type="semantic_consolidation_judgment",
                evidence_key=f"{decision_id}:{source['id']}",
                explanation=str(decision["reason"]),
                source_ref=f"semantic-consolidation:{decision_id}",
                metadata={"decision_id": decision_id, "actor": actor},
            )
        return {
            "decision_id": decision_id,
            "status": "applied",
            "result_memory_id": result_memory_id,
            "created": bool(created),
        }

    def undo_semantic_consolidation(self, decision_id: str) -> dict[str, Any]:
        """Restore both source memories and retire the consolidated result."""

        with self._lock:
            raw = self._conn.execute(
                """SELECT * FROM semantic_consolidation_decisions
                   WHERE decision_id=? AND status='applied'""",
                (decision_id,),
            ).fetchone()
        if not raw:
            return {
                "decision_id": decision_id,
                "restored": 0,
                "error": "applied semantic consolidation decision not found",
            }
        decision = dict(raw)
        snapshots = _json_object(decision.get("source_snapshot_json"))
        now = utc_now()
        restored = 0
        with self.transaction() as conn:
            for label, memory_id in (
                ("left", str(decision["left_id"])),
                ("right", str(decision["right_id"])),
            ):
                current = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
                prior = snapshots.get(label) if isinstance(snapshots.get(label), dict) else {}
                prior_state = str(prior.get("state") or "active")
                if current and str(current["state"]) != prior_state:
                    self._refinery_state_change_tx(
                        conn,
                        dict(current),
                        prior_state,
                        f"undo semantic consolidation {decision_id[:8]}",
                        now,
                    )
                    restored += 1
            result_id = str(decision.get("result_memory_id") or "")
            result = conn.execute("SELECT * FROM memories WHERE id=?", (result_id,)).fetchone()
            if result and str(result["state"]) != "archived":
                self._refinery_state_change_tx(
                    conn,
                    dict(result),
                    "archived",
                    f"reversed semantic consolidation {decision_id[:8]}",
                    now,
                )
            conn.execute(
                """UPDATE semantic_consolidation_decisions
                   SET status='reversed',reversed_at=? WHERE decision_id=?""",
                (now, decision_id),
            )
            self._refresh_semantic_consolidation_run_tx(conn, str(decision["run_id"]))
        return {
            "decision_id": decision_id,
            "restored": restored,
            "result_memory_id": decision.get("result_memory_id"),
            "status": "reversed",
        }

    def record_semantic_consolidation_feedback(
        self,
        decision_id: str,
        label: str,
        *,
        reason: str = "",
        actor: str = "cortex-operator",
    ) -> dict[str, Any]:
        """Record human review; a wrong applied merge is immediately undone."""

        label_value = normalize_text(label).casefold()
        if label_value not in {"correct", "wrong"}:
            raise ValueError("feedback label must be correct or wrong")
        with self._lock:
            row = self._conn.execute(
                """SELECT * FROM semantic_consolidation_decisions
                   WHERE decision_id=?""",
                (decision_id,),
            ).fetchone()
        if not row:
            raise ValueError("semantic consolidation decision not found")
        decision = dict(row)
        undo_result: dict[str, Any] | None = None
        if label_value == "wrong" and str(decision["status"]) == "applied":
            undo_result = self.undo_semantic_consolidation(decision_id)
        elif label_value == "wrong" and str(decision["status"]) == "proposed":
            with self.transaction() as conn:
                conn.execute(
                    """UPDATE semantic_consolidation_decisions SET status='skipped'
                       WHERE decision_id=? AND status='proposed'""",
                    (decision_id,),
                )
                self._refresh_semantic_consolidation_run_tx(conn, str(decision["run_id"]))
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO semantic_consolidation_feedback(
                   feedback_id,decision_id,label,reason,actor,created_at
                   ) VALUES(?,?,?,?,?,?)""",
                (
                    str(uuid.uuid4()),
                    decision_id,
                    label_value,
                    normalize_text(reason)[:600],
                    normalize_text(actor)[:160] or "cortex-operator",
                    utc_now(),
                ),
            )
        return {
            "decision_id": decision_id,
            "label": label_value,
            "undo": undo_result,
        }

    def consolidate(self, *, dry_run: bool = True, similarity_threshold: float = 0.78) -> dict[str, Any]:
        """Fold safe near-duplicates behind a canonical memory, reversibly."""

        with self._lock:
            rows = [
                dict(row)
                for row in self._conn.execute(
                    """SELECT * FROM memories WHERE state IN ('active','cold')
                       ORDER BY pinned DESC,protected DESC,validated_count+confirmed_count+success_count DESC,
                                confidence*trust DESC,importance DESC,created_at ASC LIMIT 2500"""
                ).fetchall()
            ]
        buckets: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for row in rows:
            applicability = (
                str(row.get("context_mode") or "standalone"),
                str(row.get("scope_json") or "{}"),
                str(row.get("preconditions_json") or "{}"),
                str(row.get("applicable_systems_json") or "[]"),
                str(row.get("applicable_versions_json") or "[]"),
            )
            if row.get("subject") and row.get("predicate") and row.get("object_value") is not None:
                key = (
                    "claim",
                    str(row["kind"]),
                    *applicability,
                    str(row["subject"]).casefold(),
                    str(row["predicate"]).casefold(),
                    str(row["object_value"]).casefold(),
                )
            else:
                features = semantic_features(str(row["content"]), max_features=32)
                concepts = sorted(feature for feature in features if feature.startswith("concept:"))[:2]
                tokens = sorted(feature for feature in features if feature.startswith("tok:"))[:3]
                anchors = sorted(
                    feature
                    for feature in features
                    if feature.startswith("tok:") and any(mark in feature[4:] for mark in (".", "/", ":", "@"))
                )
                key = (
                    "text",
                    str(row["kind"]),
                    *applicability,
                    *((anchors[:1] or concepts or tokens[:2])),
                )
            buckets.setdefault(key, []).append(row)

        clusters: list[dict[str, Any]] = []
        claimed: set[str] = set()
        for bucket in buckets.values():
            if len(bucket) < 2:
                continue
            for canonical in bucket:
                canonical_id = str(canonical["id"])
                if canonical_id in claimed:
                    continue
                members: list[dict[str, Any]] = []
                for candidate in bucket:
                    member_id = str(candidate["id"])
                    if member_id == canonical_id or member_id in claimed:
                        continue
                    if candidate["pinned"] or candidate["protected"]:
                        continue
                    similarity = feature_similarity(str(canonical["content"]), str(candidate["content"]))
                    if similarity >= similarity_threshold:
                        members.append(
                            {
                                "memory_id": member_id,
                                "similarity": round(similarity, 6),
                                "prior_state": str(candidate["state"]),
                            }
                        )
                if members:
                    claimed.update(member["memory_id"] for member in members)
                    clusters.append({"canonical_id": canonical_id, "members": members})

        run_id = str(uuid.uuid4())
        member_count = sum(len(cluster["members"]) for cluster in clusters)
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO consolidation_runs(run_id,dry_run,cluster_count,member_count,details,created_at)
                   VALUES(?,?,?,?,?,?)""",
                (run_id, int(dry_run), len(clusters), member_count, json.dumps(clusters, sort_keys=True), utc_now()),
            )
            for cluster in clusters:
                for member in cluster["members"]:
                    conn.execute(
                        """INSERT INTO consolidation_members(
                           run_id,canonical_id,member_id,similarity,prior_state
                           ) VALUES(?,?,?,?,?)""",
                        (
                            run_id,
                            cluster["canonical_id"],
                            member["memory_id"],
                            member["similarity"],
                            member["prior_state"],
                        ),
                    )
        if not dry_run:
            for cluster in clusters:
                for member in cluster["members"]:
                    self.add_edge(
                        cluster["canonical_id"],
                        member["memory_id"],
                        "consolidates",
                        weight=max(0.1, float(member["similarity"])),
                        evidence_type="consolidation_similarity",
                        evidence_key=f"{run_id}:{cluster['canonical_id']}:{member['memory_id']}",
                        explanation=(
                            f"A reversible consolidation review measured {float(member['similarity']):.0%} "
                            "feature similarity and selected the first memory as canonical."
                        ),
                        source_ref=f"consolidation:{run_id}",
                        metadata={"similarity": float(member["similarity"]), "run_id": run_id},
                    )
                    self.set_state(
                        member["memory_id"],
                        "cold",
                        reason=f"reversible consolidation into {cluster['canonical_id'][:8]}",
                    )
        return {
            "run_id": run_id,
            "dry_run": dry_run,
            "cluster_count": len(clusters),
            "member_count": member_count,
            "clusters": clusters,
        }

    def undo_consolidation(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._conn.execute(
                "SELECT * FROM consolidation_runs WHERE run_id=? AND dry_run=0", (run_id,)
            ).fetchone()
            members = self._conn.execute(
                "SELECT * FROM consolidation_members WHERE run_id=?", (run_id,)
            ).fetchall()
        if not run:
            return {"run_id": run_id, "restored": 0, "error": "applied consolidation run not found"}
        restored = 0
        for member in members:
            if self.set_state(
                str(member["member_id"]),
                str(member["prior_state"]),
                reason=f"undo consolidation {run_id[:8]}",
            ):
                restored += 1
        return {"run_id": run_id, "restored": restored}

    def adaptive_pruning_candidates(
        self,
        *,
        relevance_threshold: float = 0.25,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return low-relevance memories with inspectable activity evidence."""

        threshold = _clamp(float(relevance_threshold))
        bounded = max(1, min(int(limit), 50))
        with self._lock:
            rows = [
                dict(row)
                for row in self._conn.execute(
                    """SELECT DISTINCT m.*,
                              (SELECT COUNT(*) FROM edges e
                               WHERE e.src_id=m.id OR e.dst_id=m.id) edge_count
                       FROM memories m
                       JOIN memory_recall_memberships rm
                         ON rm.memory_id=m.id AND rm.revoked_at IS NULL
                        AND rm.eligibility='primary'
                       JOIN memory_recall_sets rs
                         ON rs.recall_set_id=rm.recall_set_id AND rs.status='active'
                       WHERE m.state IN ('active','cold')
                         AND m.pinned=0 AND m.protected=0 AND m.stranded=0
                         AND m.quarantine_reason IS NULL
                         AND m.kind NOT IN ('identity','preference','prospective')
                         AND NOT (
                           m.kind='schema'
                           AND julianday('now')-julianday(m.created_at)<7
                         )
                       ORDER BY m.created_at ASC,m.id ASC LIMIT 5000"""
                ).fetchall()
            ]
            pending = {
                (str(row["memory_id"]), str(row["memory_hash"]))
                for row in self._conn.execute(
                    """SELECT memory_id,memory_hash FROM adaptive_pruning_decisions
                       WHERE status IN ('proposed','applied','kept')"""
                ).fetchall()
            }
        experience = self.memory_experience_strengths([str(row["id"]) for row in rows])
        candidates: list[dict[str, Any]] = []
        for raw in rows:
            memory_id = str(raw["id"])
            if (memory_id, str(raw["content_hash"])) in pending:
                continue
            item = {**raw, **experience.get(memory_id, {})}
            score, evidence = _pruning_relevance_score(item)
            if score > threshold:
                continue
            candidates.append(
                {
                    "memory": _decode_memory_metadata(item),
                    "relevance_score": score,
                    "score_evidence": evidence,
                }
            )
        candidates.sort(
            key=lambda item: (
                float(item["relevance_score"]),
                str(item["memory"]["created_at"]),
                str(item["memory"]["id"]),
            )
        )
        return candidates[:bounded]

    def adaptive_pruning_snapshot(self, *, limit: int = 100) -> dict[str, Any]:
        bounded = max(1, min(int(limit), 500))
        with self._lock:
            runs = [
                dict(row)
                for row in self._conn.execute(
                    """SELECT * FROM adaptive_pruning_runs
                       ORDER BY started_at DESC LIMIT ?""",
                    (bounded,),
                ).fetchall()
            ]
            decisions = [
                dict(row)
                for row in self._conn.execute(
                    """SELECT d.*,m.content,m.kind,m.stranded,m.stranded_reason
                       FROM adaptive_pruning_decisions d
                       JOIN memories m ON m.id=d.memory_id
                       ORDER BY d.created_at DESC LIMIT ?""",
                    (bounded,),
                ).fetchall()
            ]
            count_rows = self._conn.execute(
                """SELECT status,COUNT(*) count FROM adaptive_pruning_decisions
                   GROUP BY status"""
            ).fetchall()
            action_rows = self._conn.execute(
                """SELECT action,COUNT(*) count FROM adaptive_pruning_decisions
                   GROUP BY action"""
            ).fetchall()
            strand_rows = self._conn.execute(
                """SELECT status,COUNT(*) count FROM memory_strands GROUP BY status"""
            ).fetchall()
            applied = int(
                self._conn.execute(
                    """SELECT COUNT(*) count FROM adaptive_pruning_decisions
                       WHERE status IN ('applied','reversed')"""
                ).fetchone()["count"]
            )
            regrets = int(
                self._conn.execute(
                    """SELECT COUNT(DISTINCT r.memory_id) count
                       FROM pruning_regret r
                       WHERE EXISTS(
                         SELECT 1 FROM adaptive_pruning_decisions d
                         WHERE d.memory_id=r.memory_id
                           AND d.status IN ('applied','reversed')
                           AND r.created_at>=d.applied_at
                       )"""
                ).fetchone()["count"]
            )
        for run in runs:
            run["usage"] = _json_object(run.pop("usage_json", "{}"))
        for decision in decisions:
            decision["score_evidence"] = _json_object(
                decision.pop("score_evidence_json", "{}")
            )
            decision.pop("edge_snapshot_json", None)
        return {
            "policy_version": "adaptive_pruning_v1",
            "default_mode": "shadow",
            "candidate_limit": 50,
            "counts": {
                str(row["status"]): int(row["count"] or 0)
                for row in count_rows
            },
            "actions": {
                str(row["action"]): int(row["count"] or 0)
                for row in action_rows
            },
            "strands": {
                str(row["status"]): int(row["count"] or 0)
                for row in strand_rows
            },
            "regret": {
                "applied_memories": applied,
                "regretted_memories": regrets,
                "rate": round(regrets / applied, 6) if applied else None,
                "target": 0.05,
                "passed": applied > 0 and regrets / applied <= 0.05,
            },
            "runs": runs,
            "decisions": decisions,
            "claim_boundary": (
                "Low relevance is a review signal, not proof that a memory is useless. "
                "No pruning action hard-deletes memory content."
            ),
        }

    @staticmethod
    def _refresh_adaptive_pruning_run_tx(
        conn: sqlite3.Connection,
        run_id: str,
    ) -> None:
        conn.execute(
            """UPDATE adaptive_pruning_runs
               SET judgment_count=(
                     SELECT COUNT(*) FROM adaptive_pruning_decisions WHERE run_id=?
                   ),
                   applied_count=(
                     SELECT COUNT(*) FROM adaptive_pruning_decisions
                     WHERE run_id=? AND status IN ('applied','kept')
                   )
               WHERE run_id=?""",
            (run_id, run_id, run_id),
        )

    def apply_adaptive_pruning(
        self,
        decision_id: str,
        *,
        actor: str = "cortex-operator",
    ) -> dict[str, Any]:
        """Apply one pruning proposal after current-state and hash checks."""

        with self._lock:
            raw = self._conn.execute(
                """SELECT * FROM adaptive_pruning_decisions WHERE decision_id=?""",
                (decision_id,),
            ).fetchone()
        if not raw:
            raise ValueError("adaptive pruning decision not found")
        decision = dict(raw)
        if str(decision["status"]) != "proposed":
            raise ValueError("adaptive pruning decision is not proposed")
        memory = self.get_memory(str(decision["memory_id"]))
        if (
            not memory
            or str(memory.get("content_hash") or "") != str(decision["memory_hash"])
            or str(memory.get("state") or "") not in {"active", "cold"}
            or bool(memory.get("pinned"))
            or bool(memory.get("protected"))
            or str(memory.get("kind") or "") in {"identity", "preference", "prospective"}
        ):
            with self.transaction() as conn:
                conn.execute(
                    """UPDATE adaptive_pruning_decisions SET status='skipped'
                       WHERE decision_id=? AND status='proposed'""",
                    (decision_id,),
                )
                self._refresh_adaptive_pruning_run_tx(conn, str(decision["run_id"]))
            return {"decision_id": decision_id, "status": "skipped", "reason": "stale or protected memory"}

        action = str(decision["action"])
        now = utc_now()
        if action == "keep":
            with self.transaction() as conn:
                conn.execute(
                    """UPDATE adaptive_pruning_decisions
                       SET status='kept',result_state=?,applied_at=?
                       WHERE decision_id=? AND status='proposed'""",
                    (memory["state"], now, decision_id),
                )
                self._refresh_adaptive_pruning_run_tx(conn, str(decision["run_id"]))
            return {"decision_id": decision_id, "status": "kept", "state": memory["state"]}
        if action == "quarantine" and not (
            int(memory.get("harmful_count") or 0)
            or int(memory.get("false_positive_count") or 0)
        ):
            with self.transaction() as conn:
                conn.execute(
                    """UPDATE adaptive_pruning_decisions SET status='skipped'
                       WHERE decision_id=? AND status='proposed'""",
                    (decision_id,),
                )
                self._refresh_adaptive_pruning_run_tx(conn, str(decision["run_id"]))
            return {
                "decision_id": decision_id,
                "status": "skipped",
                "reason": "quarantine requires harmful or false-positive evidence",
            }

        if action == "orphan_strand":
            with self.transaction() as conn:
                current = conn.execute(
                    "SELECT * FROM memories WHERE id=?",
                    (memory["id"],),
                ).fetchone()
                edge_rows = [
                    dict(row)
                    for row in conn.execute(
                        """SELECT * FROM edges
                           WHERE src_id=? OR dst_id=?
                           ORDER BY created_at,src_id,dst_id,relation""",
                        (memory["id"], memory["id"]),
                    ).fetchall()
                ]
                snapshot = json.dumps(edge_rows, ensure_ascii=True, sort_keys=True)
                conn.execute(
                    "DELETE FROM edges WHERE src_id=? OR dst_id=?",
                    (memory["id"], memory["id"]),
                )
                if str(current["state"]) != "cold":
                    self._refinery_state_change_tx(
                        conn,
                        dict(current),
                        "cold",
                        f"reversible relevance stranding {decision_id[:8]}",
                        now,
                    )
                conn.execute(
                    """UPDATE memories SET stranded=1,stranded_reason=?,stranded_at=?,updated_at=?
                       WHERE id=?""",
                    (str(decision["reason"])[:600], now, now, memory["id"]),
                )
                conn.execute(
                    """INSERT INTO memory_strands(
                       strand_id,decision_id,memory_id,prior_state,reason,
                       edge_snapshot_json,status,created_at
                       ) VALUES(?,?,?,?,?,?,'active',?)""",
                    (
                        str(uuid.uuid4()),
                        decision_id,
                        memory["id"],
                        memory["state"],
                        str(decision["reason"])[:600],
                        snapshot,
                        now,
                    ),
                )
                conn.execute(
                    """UPDATE adaptive_pruning_decisions
                       SET status='applied',result_state='cold',
                           edge_snapshot_json=?,applied_at=?
                       WHERE decision_id=? AND status='proposed'""",
                    (snapshot, now, decision_id),
                )
                self._refresh_adaptive_pruning_run_tx(conn, str(decision["run_id"]))
            return {
                "decision_id": decision_id,
                "status": "applied",
                "action": action,
                "state": "cold",
                "edges_removed": len(edge_rows),
            }

        target_state = {
            "cool": "cold",
            "archive": "archived" if str(memory["state"]) == "cold" else "cold",
            "quarantine": "quarantine",
        }[action]
        with self.transaction() as conn:
            current = conn.execute(
                "SELECT * FROM memories WHERE id=?",
                (memory["id"],),
            ).fetchone()
            if current and str(current["state"]) != target_state:
                self._refinery_state_change_tx(
                    conn,
                    dict(current),
                    target_state,
                    f"reversible adaptive pruning {decision_id[:8]}: {decision['reason']}",
                    now,
                )
            conn.execute(
                """UPDATE adaptive_pruning_decisions
                   SET status='applied',result_state=?,applied_at=?
                   WHERE decision_id=? AND status='proposed'""",
                (target_state, now, decision_id),
            )
            self._refresh_adaptive_pruning_run_tx(conn, str(decision["run_id"]))
        return {
            "decision_id": decision_id,
            "status": "applied",
            "action": action,
            "state": target_state,
        }

    def undo_adaptive_pruning(
        self,
        decision_id: str,
        *,
        via_regret: bool = False,
    ) -> dict[str, Any]:
        """Restore lifecycle state and any graph edges removed by stranding."""

        with self._lock:
            raw = self._conn.execute(
                """SELECT * FROM adaptive_pruning_decisions
                   WHERE decision_id=? AND status='applied'""",
                (decision_id,),
            ).fetchone()
        if not raw:
            return {
                "decision_id": decision_id,
                "restored": False,
                "error": "applied adaptive pruning decision not found",
            }
        decision = dict(raw)
        memory = self.get_memory(str(decision["memory_id"]))
        if not memory:
            return {
                "decision_id": decision_id,
                "restored": False,
                "error": "pruned memory not found",
            }
        now = utc_now()
        restored_edges = 0
        with self.transaction() as conn:
            current = conn.execute(
                "SELECT * FROM memories WHERE id=?",
                (memory["id"],),
            ).fetchone()
            prior_state = str(decision["prior_state"])
            if current and str(current["state"]) != prior_state:
                self._refinery_state_change_tx(
                    conn,
                    dict(current),
                    prior_state,
                    (
                        f"pruning regret restoration {decision_id[:8]}"
                        if via_regret
                        else f"undo adaptive pruning {decision_id[:8]}"
                    ),
                    now,
                )
            edge_snapshot = json.loads(str(decision.get("edge_snapshot_json") or "[]"))
            if isinstance(edge_snapshot, list):
                for edge in edge_snapshot:
                    if not isinstance(edge, dict):
                        continue
                    inserted = conn.execute(
                        """INSERT OR IGNORE INTO edges(
                           src_id,dst_id,relation,weight,evidence_count,
                           created_at,last_reinforced_at
                           ) VALUES(?,?,?,?,?,?,?)""",
                        (
                            edge.get("src_id"),
                            edge.get("dst_id"),
                            edge.get("relation"),
                            edge.get("weight"),
                            edge.get("evidence_count"),
                            edge.get("created_at"),
                            edge.get("last_reinforced_at"),
                        ),
                    )
                    restored_edges += int(inserted.rowcount > 0)
            conn.execute(
                """UPDATE memories SET stranded=0,stranded_reason=NULL,stranded_at=NULL,updated_at=?
                   WHERE id=?""",
                (now, memory["id"]),
            )
            conn.execute(
                """UPDATE memory_strands
                   SET status=?,restored_at=?
                   WHERE decision_id=? AND status='active'""",
                ("restored" if via_regret else "reversed", now, decision_id),
            )
            conn.execute(
                """UPDATE adaptive_pruning_decisions
                   SET status='reversed',reversed_at=? WHERE decision_id=?""",
                (now, decision_id),
            )
            self._refresh_adaptive_pruning_run_tx(conn, str(decision["run_id"]))
        return {
            "decision_id": decision_id,
            "restored": True,
            "state": decision["prior_state"],
            "edges_restored": restored_edges,
            "via_regret": via_regret,
        }

    def scoring_weight_evidence(self, *, lookback_days: int = 7) -> dict[str, Any]:
        """Summarize resolved score components by task type without claiming causality."""

        from .retrieval import SCORE_SIGNAL_WEIGHTS

        bounded_days = max(1, min(90, int(lookback_days)))
        cutoff = datetime.now(timezone.utc).timestamp() - bounded_days * 86400
        buckets: dict[str, dict[str, Any]] = {}
        traces = self.memory_traces(limit=50000)
        for trace in traces:
            try:
                created = datetime.fromisoformat(str(trace.get("created_at") or "")).timestamp()
            except (TypeError, ValueError):
                continue
            if created < cutoff:
                continue
            task_type = normalize_text(str(trace.get("task_type") or "general"))[:80] or "general"
            influence = {
                str(item.get("memory_id") or ""): bool(item.get("influenced"))
                for item in trace.get("influence", [])
                if item.get("memory_id")
            }
            bucket = buckets.setdefault(
                task_type,
                {
                    "task_type": task_type,
                    "tasks": 0,
                    "resolved": 0,
                    "used": 0,
                    "signals": {
                        signal: {
                            "used_total": 0.0,
                            "unused_total": 0.0,
                            "used_samples": 0,
                            "unused_samples": 0,
                        }
                        for signal in SCORE_SIGNAL_WEIGHTS
                    },
                },
            )
            bucket["tasks"] += 1
            for candidate in trace.get("candidate_memories", []):
                if not bool(candidate.get("selected")):
                    continue
                memory_id = str(candidate.get("memory_id") or "")
                if memory_id not in influence:
                    continue
                used = influence[memory_id]
                bucket["resolved"] += 1
                bucket["used"] += int(used)
                components = candidate.get("components")
                if not isinstance(components, dict):
                    components = {}
                for signal in SCORE_SIGNAL_WEIGHTS:
                    try:
                        value = _clamp(float(components.get(signal) or 0.0))
                    except (TypeError, ValueError):
                        value = 0.0
                    target = "used" if used else "unused"
                    bucket["signals"][signal][f"{target}_total"] += value
                    bucket["signals"][signal][f"{target}_samples"] += 1

        rows: list[dict[str, Any]] = []
        for bucket in buckets.values():
            resolved = int(bucket["resolved"])
            signals: list[dict[str, Any]] = []
            for signal, values in bucket["signals"].items():
                used_samples = int(values["used_samples"])
                unused_samples = int(values["unused_samples"])
                used_mean = (
                    float(values["used_total"]) / used_samples if used_samples else None
                )
                unused_mean = (
                    float(values["unused_total"]) / unused_samples if unused_samples else None
                )
                signals.append(
                    {
                        "signal": signal,
                        "current_weight": float(SCORE_SIGNAL_WEIGHTS[signal]),
                        "used_mean": round(used_mean, 6) if used_mean is not None else None,
                        "unused_mean": (
                            round(unused_mean, 6) if unused_mean is not None else None
                        ),
                        "separation": (
                            round(used_mean - unused_mean, 6)
                            if used_mean is not None and unused_mean is not None
                            else None
                        ),
                        "used_samples": used_samples,
                        "unused_samples": unused_samples,
                    }
                )
            rows.append(
                {
                    "task_type": bucket["task_type"],
                    "tasks": int(bucket["tasks"]),
                    "resolved": resolved,
                    "used": int(bucket["used"]),
                    "precision": (
                        round(int(bucket["used"]) / resolved, 6) if resolved else None
                    ),
                    "signals": signals,
                }
            )
        rows.sort(key=lambda row: (-int(row["resolved"]), str(row["task_type"])))
        return {
            "lookback_days": bounded_days,
            "task_types": rows,
            "minimum_resolved": 8,
            "claim_boundary": (
                "Signal separation is observational evidence from attributed selected memories. "
                "It does not prove that changing a weight will improve precision."
            ),
        }

    def active_scoring_weights(self, task_type: str | None) -> dict[str, Any]:
        """Return an explicitly approved task profile or immutable code defaults."""

        from .retrieval import LIVE_SCORING_POLICY_VERSION, SCORE_SIGNAL_WEIGHTS

        normalized = normalize_text(str(task_type or "general"))[:80] or "general"
        with self._lock:
            row = self._conn.execute(
                """SELECT * FROM scoring_weight_history
                   WHERE status='active' AND task_type IN (?, '__all__')
                   ORDER BY CASE WHEN task_type=? THEN 0 ELSE 1 END,activated_at DESC
                   LIMIT 1""",
                (normalized, normalized),
            ).fetchone()
        if not row:
            return {
                "task_type": normalized,
                "policy_version": LIVE_SCORING_POLICY_VERSION,
                "weights": dict(SCORE_SIGNAL_WEIGHTS),
                "source": "factory_defaults",
            }
        item = dict(row)
        weights = _json_object(item.get("weights_json") or "{}")
        if set(weights) != set(SCORE_SIGNAL_WEIGHTS):
            weights = dict(SCORE_SIGNAL_WEIGHTS)
            source = "factory_defaults_invalid_profile"
            version = LIVE_SCORING_POLICY_VERSION
        else:
            weights = {key: float(weights[key]) for key in SCORE_SIGNAL_WEIGHTS}
            source = "approved_profile"
            version = str(item["policy_version"])
        return {
            "task_type": normalized,
            "matched_task_type": str(item["task_type"]),
            "policy_version": version,
            "weights": weights,
            "source": source,
            "activated_at": item.get("activated_at"),
            "history_id": item.get("history_id"),
        }

    def apply_scoring_weight_proposal(
        self,
        proposal_id: str,
        *,
        actor: str = "cortex-operator",
        note: str = "",
        confirm_large_change: bool = False,
    ) -> dict[str, Any]:
        """Explicitly activate one staged task profile with stale-baseline checks."""

        from .retrieval import SCORE_SIGNAL_WEIGHTS

        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM scoring_weight_proposals WHERE proposal_id=?",
                (proposal_id,),
            ).fetchone()
        if not row:
            raise ValueError("scoring weight proposal not found")
        proposal = dict(row)
        if str(proposal["status"]) != "proposed":
            raise ValueError("scoring weight proposal is not proposed")
        if bool(proposal["explicit_confirmation_required"]) and not confirm_large_change:
            raise ValueError(
                "proposal changes at least one weight by 0.05; explicit confirmation is required"
            )
        baseline = {
            key: float(value)
            for key, value in _json_object(proposal["baseline_weights_json"]).items()
        }
        proposed = {
            key: float(value)
            for key, value in _json_object(proposal["proposed_weights_json"]).items()
        }
        if set(proposed) != set(SCORE_SIGNAL_WEIGHTS):
            raise ValueError("proposal does not contain the complete scoring signal set")
        if any(value < 0.02 or value > 0.30 for value in proposed.values()):
            raise ValueError("proposed weights must remain between 0.02 and 0.30")
        active = self.active_scoring_weights(str(proposal["task_type"]))
        if any(
            abs(float(active["weights"].get(key, -1.0)) - float(baseline.get(key, -2.0)))
            > 0.000001
            for key in SCORE_SIGNAL_WEIGHTS
        ):
            raise ValueError("proposal baseline is stale; run weight tuning again")
        now = utc_now()
        policy_version = f"adaptive_weights:{proposal_id[:12]}"
        history_id = str(uuid.uuid4())
        with self.transaction() as conn:
            conn.execute(
                """UPDATE scoring_weight_history
                   SET status='superseded',deactivated_at=?
                   WHERE task_type=? AND status='active'""",
                (now, proposal["task_type"]),
            )
            conn.execute(
                """INSERT INTO scoring_weight_history(
                   history_id,proposal_id,task_type,policy_version,weights_json,status,
                   precision_at_activation,activation_sample_count,activated_at,actor,note
                   ) VALUES(?,?,?,?,?,'active',?,?,?,?,?)""",
                (
                    history_id,
                    proposal_id,
                    proposal["task_type"],
                    policy_version,
                    json.dumps(proposed, sort_keys=True),
                    proposal["baseline_precision"],
                    int(_json_object(proposal["evidence_json"]).get("resolved") or 0),
                    now,
                    normalize_text(actor)[:120] or "cortex-operator",
                    normalize_text(note)[:600],
                ),
            )
            conn.execute(
                """UPDATE scoring_weight_proposals
                   SET status='approved',decided_at=?,decided_by=?,decision_note=?
                   WHERE proposal_id=? AND status='proposed'""",
                (
                    now,
                    normalize_text(actor)[:120] or "cortex-operator",
                    normalize_text(note)[:600],
                    proposal_id,
                ),
            )
        return {
            "proposal_id": proposal_id,
            "status": "approved",
            "task_type": proposal["task_type"],
            "policy_version": policy_version,
            "history_id": history_id,
        }

    def reject_scoring_weight_proposal(
        self,
        proposal_id: str,
        *,
        actor: str = "cortex-operator",
        note: str = "",
    ) -> dict[str, Any]:
        now = utc_now()
        with self.transaction() as conn:
            updated = conn.execute(
                """UPDATE scoring_weight_proposals
                   SET status='rejected',decided_at=?,decided_by=?,decision_note=?
                   WHERE proposal_id=? AND status='proposed'""",
                (
                    now,
                    normalize_text(actor)[:120] or "cortex-operator",
                    normalize_text(note)[:600],
                    proposal_id,
                ),
            )
        if not updated.rowcount:
            raise ValueError("proposed scoring weight proposal not found")
        return {"proposal_id": proposal_id, "status": "rejected"}

    def rollback_scoring_weights(
        self,
        proposal_id: str,
        *,
        reason: str,
        automatic: bool = False,
        actor: str = "cortex-weight-monitor",
    ) -> dict[str, Any]:
        """Deactivate an approved profile and restore its recorded baseline."""

        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM scoring_weight_proposals WHERE proposal_id=?",
                (proposal_id,),
            ).fetchone()
        if not row or str(row["status"]) != "approved":
            raise ValueError("approved scoring weight proposal not found")
        proposal = dict(row)
        baseline = _json_object(proposal["baseline_weights_json"])
        now = utc_now()
        restored_id = str(uuid.uuid4())
        status = "auto_reverted" if automatic else "rolled_back"
        with self.transaction() as conn:
            active = conn.execute(
                """UPDATE scoring_weight_history
                   SET status='rolled_back',deactivated_at=?
                   WHERE proposal_id=? AND status='active'""",
                (now, proposal_id),
            )
            if not active.rowcount:
                raise ValueError("proposal is not the active task profile")
            conn.execute(
                """INSERT INTO scoring_weight_history(
                   history_id,proposal_id,task_type,policy_version,weights_json,status,
                   precision_at_activation,activation_sample_count,activated_at,actor,note
                   ) VALUES(?,?,?,?,?,'active',NULL,0,?,?,?)""",
                (
                    restored_id,
                    None,
                    proposal["task_type"],
                    f"adaptive_rollback:{proposal_id[:12]}",
                    json.dumps(baseline, sort_keys=True),
                    now,
                    normalize_text(actor)[:120] or "cortex-weight-monitor",
                    normalize_text(reason)[:600],
                ),
            )
            conn.execute(
                """UPDATE scoring_weight_proposals
                   SET status=?,rolled_back_at=?,rollback_reason=?
                   WHERE proposal_id=?""",
                (status, now, normalize_text(reason)[:600], proposal_id),
            )
        return {
            "proposal_id": proposal_id,
            "status": status,
            "restored_history_id": restored_id,
        }

    def factory_reset_scoring_weights(
        self,
        task_type: str,
        *,
        actor: str = "cortex-operator",
        note: str = "factory reset",
    ) -> dict[str, Any]:
        """Restore code defaults for one task type while preserving history."""

        from .retrieval import LIVE_SCORING_POLICY_VERSION, SCORE_SIGNAL_WEIGHTS

        normalized = normalize_text(task_type)[:80] or "general"
        now = utc_now()
        history_id = str(uuid.uuid4())
        with self.transaction() as conn:
            conn.execute(
                """UPDATE scoring_weight_history
                   SET status='superseded',deactivated_at=?
                   WHERE task_type=? AND status='active'""",
                (now, normalized),
            )
            conn.execute(
                """INSERT INTO scoring_weight_history(
                   history_id,proposal_id,task_type,policy_version,weights_json,status,
                   precision_at_activation,activation_sample_count,activated_at,actor,note
                   ) VALUES(?,NULL,?,?,?,'active',NULL,0,?,?,?)""",
                (
                    history_id,
                    normalized,
                    f"{LIVE_SCORING_POLICY_VERSION}:factory_reset",
                    json.dumps(SCORE_SIGNAL_WEIGHTS, sort_keys=True),
                    now,
                    normalize_text(actor)[:120] or "cortex-operator",
                    normalize_text(note)[:600],
                ),
            )
        return {
            "task_type": normalized,
            "status": "factory_reset",
            "history_id": history_id,
        }

    def scoring_weight_snapshot(self, *, limit: int = 100) -> dict[str, Any]:
        from .retrieval import SCORE_SIGNAL_WEIGHTS

        bounded = max(1, min(500, int(limit)))
        with self._lock:
            runs = [
                dict(row)
                for row in self._conn.execute(
                    "SELECT * FROM scoring_weight_runs ORDER BY started_at DESC LIMIT ?",
                    (bounded,),
                ).fetchall()
            ]
            proposals = [
                dict(row)
                for row in self._conn.execute(
                    """SELECT * FROM scoring_weight_proposals
                       ORDER BY created_at DESC LIMIT ?""",
                    (bounded,),
                ).fetchall()
            ]
            history = [
                dict(row)
                for row in self._conn.execute(
                    """SELECT * FROM scoring_weight_history
                       ORDER BY activated_at DESC LIMIT ?""",
                    (bounded,),
                ).fetchall()
            ]
        for run in runs:
            run["usage"] = _json_object(run.pop("usage_json", "{}"))
        for proposal in proposals:
            proposal["baseline_weights"] = _json_object(
                proposal.pop("baseline_weights_json", "{}")
            )
            proposal["proposed_weights"] = _json_object(
                proposal.pop("proposed_weights_json", "{}")
            )
            proposal["evidence"] = _json_object(proposal.pop("evidence_json", "{}"))
            proposal["explicit_confirmation_required"] = bool(
                proposal["explicit_confirmation_required"]
            )
        for item in history:
            item["weights"] = _json_object(item.pop("weights_json", "{}"))
        active = [item for item in history if item["status"] == "active"]
        return {
            "policy_version": "adaptive_scoring_weights_v1",
            "default_mode": "shadow",
            "factory_defaults": dict(SCORE_SIGNAL_WEIGHTS),
            "bounds": {"floor": 0.02, "ceiling": 0.30, "explicit_delta": 0.05},
            "runs": runs,
            "proposals": proposals,
            "history": history,
            "active_profiles": active,
            "claim_boundary": (
                "A staged weight proposal has no retrieval effect. Only an explicit approval "
                "creates a live task profile, and post-activation precision can trigger rollback."
            ),
        }

    def auto_revert_scoring_weights(
        self,
        *,
        minimum_resolved: int = 10,
        allowed_drop: float = 0.02,
    ) -> list[dict[str, Any]]:
        """Roll back active proposals after enough worse post-activation outcomes."""

        traces = self.memory_traces(limit=50000)
        with self._lock:
            active = [
                dict(row)
                for row in self._conn.execute(
                    """SELECT h.*,p.baseline_precision
                       FROM scoring_weight_history h
                       JOIN scoring_weight_proposals p ON p.proposal_id=h.proposal_id
                       WHERE h.status='active' AND p.status='approved'"""
                ).fetchall()
            ]
        results: list[dict[str, Any]] = []
        for profile in active:
            if profile.get("baseline_precision") is None:
                continue
            resolved = 0
            used = 0
            for trace in traces:
                if str(trace.get("task_type") or "general") != str(profile["task_type"]):
                    continue
                if str(trace.get("created_at") or "") < str(profile["activated_at"]):
                    continue
                influence = {
                    str(item.get("memory_id") or ""): bool(item.get("influenced"))
                    for item in trace.get("influence", [])
                    if item.get("memory_id")
                }
                for candidate in trace.get("candidate_memories", []):
                    memory_id = str(candidate.get("memory_id") or "")
                    if bool(candidate.get("selected")) and memory_id in influence:
                        resolved += 1
                        used += int(influence[memory_id])
            precision = used / resolved if resolved else None
            if (
                resolved >= max(1, int(minimum_resolved))
                and precision is not None
                and precision + max(0.0, float(allowed_drop))
                < float(profile["baseline_precision"])
            ):
                results.append(
                    self.rollback_scoring_weights(
                        str(profile["proposal_id"]),
                        reason=(
                            f"automatic rollback: precision {precision:.3f} fell below "
                            f"baseline {float(profile['baseline_precision']):.3f}"
                        ),
                        automatic=True,
                    )
                )
        return results

    def adaptive_reconsolidation_candidates(
        self,
        *,
        task_id: str | None = None,
        lability_minutes: int = 30,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Find new same-task evidence for memories that actually influenced the task."""

        bounded_minutes = max(1, min(1440, int(lability_minutes)))
        bounded_limit = max(1, min(100, int(limit)))
        now = datetime.now(timezone.utc)
        traces = self.memory_traces(limit=1000, task_id=task_id)
        candidates: list[dict[str, Any]] = []
        with self._lock:
            pending = {
                (
                    str(row["task_id"]),
                    str(row["memory_id"]),
                    str(row["evidence_memory_id"]),
                )
                for row in self._conn.execute(
                    """SELECT task_id,memory_id,evidence_memory_id
                       FROM adaptive_reconsolidation_proposals
                       WHERE status IN ('proposed','applied')"""
                ).fetchall()
            }
        for trace in traces:
            if str(trace.get("outcome") or "") == "pending":
                continue
            reference = trace.get("completed_at") or trace.get("updated_at")
            try:
                reference_at = datetime.fromisoformat(str(reference))
                if reference_at.tzinfo is None:
                    reference_at = reference_at.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
            age_minutes = (now - reference_at).total_seconds() / 60.0
            if age_minutes < 0 or age_minutes > bounded_minutes:
                continue
            used_ids = {
                str(item.get("memory_id") or "")
                for item in trace.get("influence", [])
                if item.get("memory_id") and bool(item.get("influenced"))
            }
            evidence_ids = {
                str(item.get("memory_id") or "")
                for item in trace.get("memory_actions", [])
                if item.get("memory_id")
                and str(item.get("action") or "") in {"created", "updated"}
            }
            for memory_id in sorted(used_ids):
                memory = self.get_memory(memory_id)
                if not memory or str(memory.get("state")) not in {"active", "cold"}:
                    continue
                for evidence_id in sorted(evidence_ids - {memory_id}):
                    evidence = self.get_memory(evidence_id)
                    key = (str(trace["task_id"]), memory_id, evidence_id)
                    if (
                        not evidence
                        or key in pending
                        or str(evidence.get("state")) not in {"active", "cold"}
                    ):
                        continue
                    candidates.append(
                        {
                            "task_id": str(trace["task_id"]),
                            "task_type": str(trace.get("task_type") or "general"),
                            "goal": str(trace.get("goal") or "")[:500],
                            "used_memory": memory,
                            "new_evidence": evidence,
                            "used_at": reference,
                            "lability_age_minutes": round(age_minutes, 3),
                            "user_correction_wins": True,
                        }
                    )
                    if len(candidates) >= bounded_limit:
                        return candidates
        return candidates

    def adaptive_reconsolidation_snapshot(self, *, limit: int = 100) -> dict[str, Any]:
        bounded = max(1, min(500, int(limit)))
        with self._lock:
            runs = [
                dict(row)
                for row in self._conn.execute(
                    """SELECT * FROM adaptive_reconsolidation_runs
                       ORDER BY started_at DESC LIMIT ?""",
                    (bounded,),
                ).fetchall()
            ]
            proposals = [
                dict(row)
                for row in self._conn.execute(
                    """SELECT p.*,m.content memory_content,m.kind memory_kind,
                              e.content evidence_content,e.kind evidence_kind
                       FROM adaptive_reconsolidation_proposals p
                       JOIN memories m ON m.id=p.memory_id
                       JOIN memories e ON e.id=p.evidence_memory_id
                       ORDER BY p.created_at DESC LIMIT ?""",
                    (bounded,),
                ).fetchall()
            ]
            counts = self._conn.execute(
                """SELECT status,COUNT(*) count
                   FROM adaptive_reconsolidation_proposals GROUP BY status"""
            ).fetchall()
            feedback = self._conn.execute(
                """SELECT feedback_label,COUNT(*) count
                   FROM adaptive_reconsolidation_proposals
                   WHERE feedback_label IS NOT NULL GROUP BY feedback_label"""
            ).fetchall()
        for run in runs:
            run["usage"] = _json_object(run.pop("usage_json", "{}"))
        for proposal in proposals:
            proposal["protected_confirmation_required"] = bool(
                proposal["protected_confirmation_required"]
            )
            proposal.pop("source_snapshot_json", None)
            proposal.pop("edge_snapshot_json", None)
        labels = {str(row["feedback_label"]): int(row["count"]) for row in feedback}
        reviewed = sum(labels.values())
        correctness = labels.get("correct", 0) / reviewed if reviewed else None
        return {
            "policy_version": "adaptive_reconsolidation_v1",
            "default_mode": "shadow",
            "lability_minutes": 30,
            "counts": {str(row["status"]): int(row["count"]) for row in counts},
            "feedback": {
                **labels,
                "reviewed": reviewed,
                "correctness": round(correctness, 6) if correctness is not None else None,
                "target": 0.90,
                "ready": bool(reviewed and correctness is not None and correctness >= 0.90),
            },
            "runs": runs,
            "proposals": proposals,
            "claim_boundary": (
                "Retrieval plus same-task evidence opens a bounded review window; it does "
                "not authorize a rewrite. User corrections remain authoritative."
            ),
        }

    def apply_adaptive_reconsolidation(
        self,
        proposal_id: str,
        *,
        actor: str = "cortex-operator",
        confirm_protected: bool = False,
    ) -> dict[str, Any]:
        """Apply one reviewed reconsolidation proposal with stale-source guards."""

        with self._lock:
            raw = self._conn.execute(
                """SELECT * FROM adaptive_reconsolidation_proposals
                   WHERE proposal_id=?""",
                (proposal_id,),
            ).fetchone()
        if not raw:
            raise ValueError("adaptive reconsolidation proposal not found")
        proposal = dict(raw)
        if str(proposal["status"]) != "proposed":
            raise ValueError("adaptive reconsolidation proposal is not proposed")
        if bool(proposal["protected_confirmation_required"]) and not confirm_protected:
            raise ValueError("protected identity or preference requires explicit confirmation")
        memory = self.get_memory(str(proposal["memory_id"]))
        evidence = self.get_memory(str(proposal["evidence_memory_id"]))
        if (
            not memory
            or not evidence
            or str(memory.get("content_hash")) != str(proposal["memory_hash"])
            or str(evidence.get("content_hash")) != str(proposal["evidence_hash"])
        ):
            with self.transaction() as conn:
                conn.execute(
                    """UPDATE adaptive_reconsolidation_proposals SET status='skipped'
                       WHERE proposal_id=? AND status='proposed'""",
                    (proposal_id,),
                )
            return {"proposal_id": proposal_id, "status": "skipped", "reason": "source changed"}
        action = str(proposal["action"])
        now = utc_now()
        if action == "supersede":
            replacement = normalize_text(str(proposal.get("replacement_content") or ""))
            if not replacement:
                raise ValueError("supersede proposal has no replacement content")
            changed = self.correct_memory(
                str(memory["id"]),
                replacement,
                reason=f"reviewed adaptive reconsolidation {proposal_id[:8]}",
                confidence=max(float(memory["confidence"]), float(evidence["confidence"])),
                source_ref=f"memory:{evidence['id']}",
            )
            if not changed:
                raise ValueError("reconsolidation target could not be corrected")
        else:
            relation = "extends" if action == "extend" else "contradicts"
            src_id = str(evidence["id"])
            dst_id = str(memory["id"])
            with self._lock:
                edge = self._conn.execute(
                    """SELECT * FROM edges
                       WHERE src_id=? AND dst_id=? AND relation=?""",
                    (src_id, dst_id, relation),
                ).fetchone()
            edge_snapshot = json.dumps(dict(edge) if edge else {}, sort_keys=True)
            self.add_edge(
                src_id,
                dst_id,
                relation,
                weight=0.55 if action == "extend" else 0.70,
                evidence_type="reviewed_reconsolidation",
                evidence_key=proposal_id,
                explanation=str(proposal["reason"]),
                source_ref=f"memory:{evidence['id']}",
                task_id=str(proposal["task_id"]),
                metadata={"proposal_id": proposal_id, "action": action},
            )
            with self.transaction() as conn:
                conn.execute(
                    """UPDATE adaptive_reconsolidation_proposals SET edge_snapshot_json=?
                       WHERE proposal_id=?""",
                    (edge_snapshot, proposal_id),
                )
        with self.transaction() as conn:
            conn.execute(
                """UPDATE adaptive_reconsolidation_proposals
                   SET status='applied',applied_at=?,applied_by=?
                   WHERE proposal_id=? AND status='proposed'""",
                (now, normalize_text(actor)[:120], proposal_id),
            )
        return {"proposal_id": proposal_id, "status": "applied", "action": action}

    def undo_adaptive_reconsolidation(self, proposal_id: str) -> dict[str, Any]:
        """Restore the prior version or edge state recorded by an applied proposal."""

        with self._lock:
            raw = self._conn.execute(
                """SELECT * FROM adaptive_reconsolidation_proposals
                   WHERE proposal_id=? AND status='applied'""",
                (proposal_id,),
            ).fetchone()
        if not raw:
            return {"proposal_id": proposal_id, "reversed": False}
        proposal = dict(raw)
        action = str(proposal["action"])
        if action == "supersede":
            snapshot = _json_object(proposal["source_snapshot_json"])
            prior = snapshot.get("memory") if isinstance(snapshot.get("memory"), dict) else {}
            content = normalize_text(str(prior.get("content") or ""))
            if not content:
                raise ValueError("reconsolidation source snapshot is unavailable")
            self.correct_memory(
                str(proposal["memory_id"]),
                content,
                reason=f"undo adaptive reconsolidation {proposal_id[:8]}",
                confidence=float(prior.get("confidence") or 0.6),
                source_ref=f"reconsolidation-undo:{proposal_id}",
            )
            with self.transaction() as conn:
                conn.execute(
                    """UPDATE memories SET state=?,protected=?,updated_at=updated_at
                       WHERE id=?""",
                    (
                        str(prior.get("state") or "active"),
                        int(bool(prior.get("protected"))),
                        proposal["memory_id"],
                    ),
                )
        else:
            relation = "extends" if action == "extend" else "contradicts"
            src_id = str(proposal["evidence_memory_id"])
            dst_id = str(proposal["memory_id"])
            snapshot = _json_object(proposal["edge_snapshot_json"])
            with self.transaction() as conn:
                conn.execute(
                    "DELETE FROM edge_evidence WHERE evidence_key=?",
                    (proposal_id,),
                )
                if snapshot:
                    conn.execute(
                        """INSERT INTO edges(
                           src_id,dst_id,relation,weight,evidence_count,
                           created_at,last_reinforced_at
                           ) VALUES(?,?,?,?,?,?,?)
                           ON CONFLICT(src_id,dst_id,relation) DO UPDATE SET
                             weight=excluded.weight,
                             evidence_count=excluded.evidence_count,
                             created_at=excluded.created_at,
                             last_reinforced_at=excluded.last_reinforced_at""",
                        (
                            snapshot["src_id"],
                            snapshot["dst_id"],
                            snapshot["relation"],
                            snapshot["weight"],
                            snapshot["evidence_count"],
                            snapshot["created_at"],
                            snapshot["last_reinforced_at"],
                        ),
                    )
                else:
                    remaining = conn.execute(
                        """SELECT COUNT(*) count FROM edge_evidence
                           WHERE src_id=? AND dst_id=? AND relation=?""",
                        (src_id, dst_id, relation),
                    ).fetchone()["count"]
                    if not remaining:
                        conn.execute(
                            """DELETE FROM edges
                               WHERE src_id=? AND dst_id=? AND relation=?""",
                            (src_id, dst_id, relation),
                        )
        with self.transaction() as conn:
            conn.execute(
                """UPDATE adaptive_reconsolidation_proposals
                   SET status='reversed',reversed_at=? WHERE proposal_id=?""",
                (utc_now(), proposal_id),
            )
        return {"proposal_id": proposal_id, "reversed": True, "action": action}

    def record_adaptive_reconsolidation_feedback(
        self,
        proposal_id: str,
        label: str,
        *,
        reason: str = "",
    ) -> dict[str, Any]:
        label_value = normalize_text(label).casefold()
        if label_value not in {"correct", "wrong"}:
            raise ValueError("feedback label must be correct or wrong")
        with self._lock:
            row = self._conn.execute(
                """SELECT status FROM adaptive_reconsolidation_proposals
                   WHERE proposal_id=?""",
                (proposal_id,),
            ).fetchone()
        if not row:
            raise ValueError("adaptive reconsolidation proposal not found")
        undone = False
        if label_value == "wrong" and str(row["status"]) == "applied":
            undone = bool(self.undo_adaptive_reconsolidation(proposal_id)["reversed"])
        elif label_value == "wrong" and str(row["status"]) == "proposed":
            with self.transaction() as conn:
                conn.execute(
                    """UPDATE adaptive_reconsolidation_proposals
                       SET status='rejected' WHERE proposal_id=? AND status='proposed'""",
                    (proposal_id,),
                )
        with self.transaction() as conn:
            conn.execute(
                """UPDATE adaptive_reconsolidation_proposals
                   SET feedback_label=?,feedback_reason=?,feedback_at=?
                   WHERE proposal_id=?""",
                (label_value, normalize_text(reason)[:600], utc_now(), proposal_id),
            )
        return {
            "proposal_id": proposal_id,
            "label": label_value,
            "automatically_undone": undone,
            "feedback": self.adaptive_reconsolidation_snapshot(limit=1)["feedback"],
        }

    def schema_formation_candidates(
        self,
        *,
        minimum_cluster: int = 3,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        """Build evidence-qualified related-memory clusters for model review."""

        minimum = max(3, min(10, int(minimum_cluster)))
        bounded = max(1, min(50, int(limit)))
        with self._lock:
            rows = self._conn.execute(
                """SELECT m.*,u.task_id,t.task_type
                   FROM memories m
                   JOIN usage_records u ON u.memory_id=m.id AND u.used=1
                   LEFT JOIN memory_traces t ON t.task_id=u.task_id
                   WHERE m.state IN ('active','cold')
                     AND m.kind NOT IN ('schema','identity','preference','prospective')
                     AND m.quarantine_reason IS NULL
                     AND NOT EXISTS(
                       SELECT 1 FROM memory_dependencies d
                       JOIN memories s ON s.id=d.memory_id
                       WHERE d.evidence_id=m.id AND d.relation='schema_source'
                         AND d.active=1 AND s.state IN ('active','cold')
                     )
                   ORDER BY m.created_at DESC LIMIT 2000"""
            ).fetchall()
            edge_rows = self._conn.execute(
                """SELECT src_id,dst_id FROM edges
                   WHERE relation IN ('related','extends','supports','co_used','generalizes')"""
            ).fetchall()
            prior_rows = self._conn.execute(
                """SELECT cluster_signature FROM schema_formation_proposals
                   WHERE status IN ('proposed','applied')"""
            ).fetchall()
        by_memory: dict[str, dict[str, Any]] = {}
        for raw in rows:
            row = dict(raw)
            memory_id = str(row["id"])
            item = by_memory.setdefault(
                memory_id,
                {
                    "memory": _decode_memory_metadata(row),
                    "tasks": set(),
                    "task_types": set(),
                },
            )
            item["tasks"].add(str(row["task_id"]))
            if row.get("task_type"):
                item["task_types"].add(str(row["task_type"]))
        ids = sorted(by_memory)
        if len(ids) < minimum:
            return []
        edge_pairs = {
            frozenset((str(row["src_id"]), str(row["dst_id"]))) for row in edge_rows
        }
        parent = {memory_id: memory_id for memory_id in ids}

        def find(memory_id: str) -> str:
            while parent[memory_id] != memory_id:
                parent[memory_id] = parent[parent[memory_id]]
                memory_id = parent[memory_id]
            return memory_id

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        token_sets = {
            memory_id: set(query_tokens(str(by_memory[memory_id]["memory"]["content"])))
            for memory_id in ids
        }
        for index, left_id in enumerate(ids):
            left = by_memory[left_id]["memory"]
            for right_id in ids[index + 1 :]:
                right = by_memory[right_id]["memory"]
                left_tokens = token_sets[left_id]
                right_tokens = token_sets[right_id]
                overlap = (
                    len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
                )
                same_claim = bool(
                    left.get("subject")
                    and left.get("predicate")
                    and left.get("subject") == right.get("subject")
                    and left.get("predicate") == right.get("predicate")
                )
                if (
                    overlap >= 0.30
                    or same_claim
                    or frozenset((left_id, right_id)) in edge_pairs
                ):
                    union(left_id, right_id)
        groups: dict[str, list[str]] = {}
        for memory_id in ids:
            groups.setdefault(find(memory_id), []).append(memory_id)
        prior = {str(row["cluster_signature"]) for row in prior_rows}
        candidates: list[dict[str, Any]] = []
        for group_ids in groups.values():
            if len(group_ids) < minimum:
                continue
            # Keep provider payloads bounded while retaining deterministic clusters.
            for offset in range(0, len(group_ids), 8):
                source_ids = sorted(group_ids[offset : offset + 8])
                if len(source_ids) < minimum:
                    continue
                tasks = sorted(
                    {
                        task
                        for memory_id in source_ids
                        for task in by_memory[memory_id]["tasks"]
                    }
                )
                task_types = sorted(
                    {
                        task_type
                        for memory_id in source_ids
                        for task_type in by_memory[memory_id]["task_types"]
                    }
                )
                episode_count = sum(
                    str(by_memory[memory_id]["memory"].get("kind")) == "episode"
                    for memory_id in source_ids
                )
                if len(tasks) < 3 or (len(task_types) < 2 and episode_count < 2):
                    continue
                signature = hashlib.sha256("|".join(source_ids).encode("utf-8")).hexdigest()
                if signature in prior:
                    continue
                memories = [by_memory[memory_id]["memory"] for memory_id in source_ids]
                candidates.append(
                    {
                        "cluster_signature": signature,
                        "source_ids": source_ids,
                        "sources": memories,
                        "evidence": {
                            "source_count": len(source_ids),
                            "distinct_tasks": len(tasks),
                            "task_ids": tasks,
                            "distinct_task_types": len(task_types),
                            "task_types": task_types,
                            "episode_count": episode_count,
                            "minimum_cluster": minimum,
                        },
                    }
                )
        candidates.sort(
            key=lambda item: (
                -int(item["evidence"]["distinct_tasks"]),
                -int(item["evidence"]["source_count"]),
                item["cluster_signature"],
            )
        )
        return candidates[:bounded]

    def active_schema_source_ids(self, memory_ids: Sequence[str]) -> set[str]:
        if not memory_ids:
            return set()
        placeholders = ",".join("?" for _ in memory_ids)
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT DISTINCT d.evidence_id
                    FROM memory_dependencies d JOIN memories s ON s.id=d.memory_id
                    WHERE d.relation='schema_source' AND d.active=1
                      AND s.state IN ('active','cold')
                      AND d.evidence_id IN ({placeholders})""",
                tuple(memory_ids),
            ).fetchall()
        return {str(row["evidence_id"]) for row in rows}

    def schema_formation_snapshot(self, *, limit: int = 100) -> dict[str, Any]:
        bounded = max(1, min(500, int(limit)))
        with self._lock:
            runs = [
                dict(row)
                for row in self._conn.execute(
                    """SELECT * FROM schema_formation_runs
                       ORDER BY started_at DESC LIMIT ?""",
                    (bounded,),
                ).fetchall()
            ]
            proposals = [
                dict(row)
                for row in self._conn.execute(
                    """SELECT p.*,m.content result_content,m.state result_state,m.dirty result_dirty,
                              m.dirty_reason result_dirty_reason
                       FROM schema_formation_proposals p
                       LEFT JOIN memories m ON m.id=p.result_memory_id
                       ORDER BY p.created_at DESC LIMIT ?""",
                    (bounded,),
                ).fetchall()
            ]
            counts = self._conn.execute(
                """SELECT status,COUNT(*) count FROM schema_formation_proposals
                   GROUP BY status"""
            ).fetchall()
            feedback = self._conn.execute(
                """SELECT feedback_label,COUNT(*) count FROM schema_formation_proposals
                   WHERE feedback_label IS NOT NULL GROUP BY feedback_label"""
            ).fetchall()
        for run in runs:
            run["usage"] = _json_object(run.pop("usage_json", "{}"))
        for proposal in proposals:
            proposal["source_ids"] = _trace_json_array(proposal.pop("source_ids_json"))
            proposal["included_source_ids"] = _trace_json_array(
                proposal.pop("included_source_ids_json")
            )
            proposal["source_hashes"] = _json_object(proposal.pop("source_hashes_json"))
            proposal["evidence"] = _json_object(proposal.pop("evidence_json"))
            proposal["sources"] = [
                self.get_memory(memory_id)
                for memory_id in proposal["included_source_ids"]
                if self.get_memory(memory_id)
            ]
        labels = {str(row["feedback_label"]): int(row["count"]) for row in feedback}
        reviewed = sum(labels.values())
        accuracy = labels.get("correct", 0) / reviewed if reviewed else None
        return {
            "policy_version": "schema_formation_v1",
            "default_mode": "shadow",
            "minimum_cluster": 3,
            "counts": {str(row["status"]): int(row["count"]) for row in counts},
            "feedback": {
                **labels,
                "reviewed": reviewed,
                "accuracy": round(accuracy, 6) if accuracy is not None else None,
                "target": 0.70,
                "ready": bool(reviewed and accuracy is not None and accuracy >= 0.70),
            },
            "runs": runs,
            "proposals": proposals,
            "claim_boundary": (
                "A cluster and model abstraction are review evidence, not a proven rule. "
                "Source memories remain independently inspectable and recallable."
            ),
        }

    def apply_schema_formation(
        self,
        proposal_id: str,
        *,
        actor: str = "cortex-operator",
    ) -> dict[str, Any]:
        """Apply one reviewed schema abstraction without replacing its examples."""

        with self._lock:
            raw = self._conn.execute(
                "SELECT * FROM schema_formation_proposals WHERE proposal_id=?",
                (proposal_id,),
            ).fetchone()
        if not raw:
            raise ValueError("schema formation proposal not found")
        proposal = dict(raw)
        if str(proposal["status"]) != "proposed":
            raise ValueError("schema formation proposal is not proposed")
        if str(proposal["action"]) == "no_schema":
            with self.transaction() as conn:
                conn.execute(
                    """UPDATE schema_formation_proposals
                       SET status='dismissed',applied_at=?,applied_by=?
                       WHERE proposal_id=? AND status='proposed'""",
                    (utc_now(), normalize_text(actor)[:120], proposal_id),
                )
            return {"proposal_id": proposal_id, "status": "dismissed"}
        included = _trace_json_array(proposal["included_source_ids_json"])
        hashes = _json_object(proposal["source_hashes_json"])
        if len(included) < 3:
            raise ValueError("schema abstraction requires at least three source memories")
        sources = [self.get_memory(str(memory_id)) for memory_id in included]
        if any(
            not source
            or str(source.get("content_hash")) != str(hashes.get(str(source["id"])))
            or str(source.get("state")) not in {"active", "cold"}
            for source in sources
            if source is not None
        ) or len(sources) != len(included) or any(source is None for source in sources):
            with self.transaction() as conn:
                conn.execute(
                    """UPDATE schema_formation_proposals SET status='skipped'
                       WHERE proposal_id=? AND status='proposed'""",
                    (proposal_id,),
                )
            return {"proposal_id": proposal_id, "status": "skipped", "reason": "source changed"}
        abstract_content = normalize_text(str(proposal.get("abstract_content") or ""))
        if not abstract_content:
            raise ValueError("schema abstraction content is unavailable")
        schema_id, created = self.add_memory(
            abstract_content,
            kind="schema",
            source_type="schema_formation",
            source_category="OPERATOR_APPROVED",
            origin_source_category="AGENT_INFERENCE",
            approval_state="operator_approved",
            source_ref=f"schema-proposal:{proposal_id}",
            extraction_method="reviewed_schema_formation_v1",
            confidence=max(0.55, min(float(proposal["confidence"]), 0.9)),
            importance=0.72,
            uniqueness=0.8,
            volatility=0.25,
            trust=0.76,
            protected=False,
        )
        if not created:
            with self.transaction() as conn:
                conn.execute(
                    """UPDATE schema_formation_proposals SET status='skipped'
                       WHERE proposal_id=? AND status='proposed'""",
                    (proposal_id,),
                )
            return {
                "proposal_id": proposal_id,
                "status": "skipped",
                "reason": "identical schema memory already exists",
            }
        for source in sources:
            source_id = str(source["id"])
            self.add_dependency(schema_id, source_id, relation="schema_source", weight=1.0)
            self.add_edge(
                schema_id,
                source_id,
                "abstracts",
                weight=0.65,
                evidence_type="reviewed_schema_formation",
                evidence_key=proposal_id,
                explanation=str(proposal["reason"]),
                source_ref=f"schema-proposal:{proposal_id}",
                metadata={"proposal_id": proposal_id},
            )
            self.add_edge(
                source_id,
                schema_id,
                "example_of",
                weight=0.65,
                evidence_type="reviewed_schema_formation",
                evidence_key=proposal_id,
                explanation=str(proposal["reason"]),
                source_ref=f"schema-proposal:{proposal_id}",
                metadata={"proposal_id": proposal_id},
            )
        with self.transaction() as conn:
            conn.execute(
                """UPDATE schema_formation_proposals
                   SET status='applied',result_memory_id=?,applied_at=?,applied_by=?
                   WHERE proposal_id=? AND status='proposed'""",
                (schema_id, utc_now(), normalize_text(actor)[:120], proposal_id),
            )
        return {
            "proposal_id": proposal_id,
            "status": "applied",
            "schema_memory_id": schema_id,
            "source_count": len(sources),
        }

    def undo_schema_formation(self, proposal_id: str) -> dict[str, Any]:
        with self._lock:
            raw = self._conn.execute(
                """SELECT * FROM schema_formation_proposals
                   WHERE proposal_id=? AND status='applied'""",
                (proposal_id,),
            ).fetchone()
        if not raw:
            return {"proposal_id": proposal_id, "reversed": False}
        proposal = dict(raw)
        schema_id = str(proposal["result_memory_id"])
        now = utc_now()
        with self.transaction() as conn:
            conn.execute(
                """UPDATE memory_dependencies SET active=0
                   WHERE memory_id=? AND relation='schema_source'""",
                (schema_id,),
            )
            edge_keys = conn.execute(
                """SELECT src_id,dst_id,relation FROM edge_evidence
                   WHERE evidence_key=?""",
                (proposal_id,),
            ).fetchall()
            conn.execute("DELETE FROM edge_evidence WHERE evidence_key=?", (proposal_id,))
            for edge in edge_keys:
                remaining = conn.execute(
                    """SELECT COUNT(*) count FROM edge_evidence
                       WHERE src_id=? AND dst_id=? AND relation=?""",
                    (edge["src_id"], edge["dst_id"], edge["relation"]),
                ).fetchone()["count"]
                if not remaining:
                    conn.execute(
                        """DELETE FROM edges WHERE src_id=? AND dst_id=? AND relation=?""",
                        (edge["src_id"], edge["dst_id"], edge["relation"]),
                    )
            current = conn.execute("SELECT * FROM memories WHERE id=?", (schema_id,)).fetchone()
            if current and str(current["state"]) != "archived":
                self._refinery_state_change_tx(
                    conn,
                    dict(current),
                    "archived",
                    f"undo schema formation {proposal_id[:8]}",
                    now,
                )
            conn.execute(
                """UPDATE schema_formation_proposals
                   SET status='reversed',reversed_at=? WHERE proposal_id=?""",
                (now, proposal_id),
            )
        return {"proposal_id": proposal_id, "reversed": True, "schema_memory_id": schema_id}

    def record_schema_formation_feedback(
        self,
        proposal_id: str,
        label: str,
        *,
        reason: str = "",
    ) -> dict[str, Any]:
        label_value = normalize_text(label).casefold()
        if label_value not in {"correct", "wrong"}:
            raise ValueError("feedback label must be correct or wrong")
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM schema_formation_proposals WHERE proposal_id=?",
                (proposal_id,),
            ).fetchone()
        if not row:
            raise ValueError("schema formation proposal not found")
        undone = False
        if label_value == "wrong" and str(row["status"]) == "applied":
            undone = bool(self.undo_schema_formation(proposal_id)["reversed"])
        elif label_value == "wrong" and str(row["status"]) == "proposed":
            with self.transaction() as conn:
                conn.execute(
                    """UPDATE schema_formation_proposals
                       SET status='rejected' WHERE proposal_id=? AND status='proposed'""",
                    (proposal_id,),
                )
        with self.transaction() as conn:
            conn.execute(
                """UPDATE schema_formation_proposals
                   SET feedback_label=?,feedback_reason=?,feedback_at=?
                   WHERE proposal_id=?""",
                (label_value, normalize_text(reason)[:600], utc_now(), proposal_id),
            )
        return {
            "proposal_id": proposal_id,
            "label": label_value,
            "automatically_undone": undone,
            "feedback": self.schema_formation_snapshot(limit=1)["feedback"],
        }

    def maintenance(
        self,
        *,
        dry_run: bool = True,
        cold_after_days: int = 90,
        archive_after_days: int = 180,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        candidates: dict[str, list[str]] = {"cold": [], "archived": []}
        retention_scores: dict[str, float] = {}
        reasons: dict[str, str] = {}
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM memories WHERE pinned=0 AND protected=0 AND state IN ('active','cold')
                   AND importance<0.85 AND kind NOT IN ('identity','preference','prospective')"""
            ).fetchall()
        experience = self.memory_experience_strengths([str(row["id"]) for row in rows])
        for raw in rows:
            row = {**dict(raw), **experience.get(str(raw["id"]), {})}
            quality_action = _memory_quality_pruning_action(row)
            if quality_action:
                next_state, quality_reason = quality_action
                if str(row["state"]) != next_state:
                    candidates[next_state].append(str(row["id"]))
                    retention_scores[str(row["id"])] = 0.0 if next_state == "archived" else 0.2
                    reasons[str(row["id"])] = quality_reason
                continue
            reference = row["last_helpful_at"] or row["last_used_at"] or row["updated_at"]
            try:
                age = (now - datetime.fromisoformat(reference)).total_seconds() / 86400
            except (TypeError, ValueError):
                continue
            retention = _retention_score(row)
            policy_effect = self.active_policy_adjustment(
                "retention",
                {
                    "kind": str(row["kind"]),
                    "source_type": str(row["source_type"]),
                    "source_category": str(row["source_category"]),
                },
            )
            retention = _clamp(retention + float(policy_effect.get("score_adjustment") or 0.0))
            retention_scores[str(row["id"])] = round(retention, 6)
            cold_threshold = cold_after_days * (0.55 + retention)
            archive_threshold = archive_after_days * (0.65 + retention)
            if row["state"] == "active" and age >= cold_threshold and retention < 0.78:
                candidates["cold"].append(row["id"])
                reasons[str(row["id"])] = (
                    "Low retention evidence plus age make this memory eligible to cool; recency alone is not enough."
                    + (
                        " An active operator-trained retention policy contributed to this score."
                        if policy_effect.get("matched_versions")
                        else ""
                    )
                )
            elif row["state"] == "cold" and age >= archive_threshold and retention < 0.58:
                candidates["archived"].append(row["id"])
                reasons[str(row["id"])] = (
                    "The memory was already cold and still has low usefulness evidence after the archive window."
                    + (
                        " An active operator-trained retention policy contributed to this score."
                        if policy_effect.get("matched_versions")
                        else ""
                    )
                )
        if not dry_run:
            for state, ids in candidates.items():
                for memory_id in ids:
                    self.set_state(
                        memory_id,
                        state,
                        reason=reasons.get(str(memory_id), "utility and age lifecycle maintenance"),
                        retention_score=retention_scores.get(str(memory_id)),
                    )
        details = {k: len(v) for k, v in candidates.items()}
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO maintenance_log(action,details,dry_run,created_at) VALUES(?,?,?,?)",
                ("lifecycle", json.dumps(details, sort_keys=True), int(dry_run), utc_now()),
            )
        return {
            "dry_run": dry_run,
            "candidates": details,
            "memory_ids": candidates,
            "retention_scores": {memory_id: retention_scores[memory_id] for ids in candidates.values() for memory_id in ids},
            "reasons": {memory_id: reasons[memory_id] for ids in candidates.values() for memory_id in ids},
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _trace_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _association_transitions(edge: dict[str, Any]) -> list[tuple[str, str, float]]:
    """Translate a typed edge into allowed positive activation paths.

    Connections are not interchangeable. Contradictions and consolidation
    lineage never spread positive activation. A superseded memory may lead to
    its replacement, never the reverse. A claim may lead to the memory that
    supports it; evidence does not pull the claim in merely because it exists.
    """

    src = str(edge.get("src_id") or "")
    dst = str(edge.get("dst_id") or "")
    if not src or not dst or src == dst:
        return []
    relation = normalize_text(str(edge.get("relation") or "")).casefold()
    base = _clamp(float(edge.get("weight") or 0.0))
    symmetric_factors = {
        "related": 0.55,
        "co_used": 0.90,
        "co_observed": 0.30,
        "sleep_replay": 0.60,
        "vault_link": 0.80,
        "same_subject": 0.90,
        "same_context": 0.75,
        "useful_together": 1.00,
        "operator_link": 0.90,
        "contextual": 0.70,
    }
    if relation in symmetric_factors:
        weight = base * symmetric_factors[relation]
        return [(src, dst, weight), (dst, src, weight)] if weight > 0 else []
    if relation == "supports":
        # src supports/explains dst, so recalling the claim (dst) may bring in
        # its evidence (src). The reverse would let evidence invent a claim.
        return [(dst, src, base)] if base > 0 else []
    if relation == "supersedes":
        # src is the newer memory and dst is the preserved older memory.
        return [(dst, src, base)] if base > 0 else []
    # contradicts, consolidates, dependencies, and unknown relations are
    # inspectable lineage or inhibition signals, not positive associations.
    return []


def _default_edge_explanation(relation: str, *, evidence_count: int) -> str:
    """Return a truthful fallback for migrated edges without a reason ledger."""

    count = max(1, int(evidence_count or 1))
    explanations = {
        "vault_link": "The source vault note contains an explicit wikilink to the connected note.",
        "supersedes": "The newer memory records that it replaces the earlier memory while preserving history.",
        "contradicts": (
            "The memories make different structured claims about the same subject and property "
            "during overlapping validity periods."
        ),
        "contextual": "A review determined that both memories can be valid in different contexts.",
        "co_used": f"Both memories were attributed to the same task outcome ({count} recorded witness{'es' if count != 1 else ''}).",
        "co_observed": (
            "The memories were captured in the same turn. This is weak provenance context, not proof "
            "that their meanings are related."
        ),
        "related": (
            "This is a legacy similarity link without preserved supporting evidence; treat it as weak "
            "and pending review."
        ),
        "sleep_replay": f"Offline replay found the pair together in independent evidence ({count} recorded witness{'es' if count != 1 else ''}).",
        "consolidates": "A reversible consolidation review identified the memories as near-duplicates.",
        "supports": "One memory was recorded as supporting evidence for the other.",
        "same_subject": "A reviewed connection says both memories describe the same durable subject.",
        "same_context": "A reviewed connection says both memories are useful in the same durable context.",
        "useful_together": "A reviewed connection says recalling either memory should make the other useful.",
        "operator_link": "An operator approved this as a durable relationship after reviewing both memories.",
        "dependency": "The derived memory depends on the connected evidence memory.",
    }
    return explanations.get(
        normalize_text(relation).casefold(),
        "This migrated connection has no preserved rationale; its type and weight are visible for review.",
    )


def _normalize_context_mode(value: str) -> str:
    normalized = normalize_text(value or "standalone").casefold().replace("-", "_")
    if normalized not in {"standalone", "context_dependent"}:
        raise ValueError("context_mode must be standalone or context_dependent")
    return normalized


def _normalize_context_map(value: dict[str, Any] | None) -> dict[str, str]:
    if not value:
        return {}
    if not isinstance(value, dict):
        raise ValueError("scope and preconditions must be JSON objects")
    normalized: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = normalize_text(str(raw_key)).casefold().replace(" ", "_")[:80]
        item = normalize_text(str(raw_value))[:300]
        if key and item:
            normalized[key] = item
    return dict(sorted(normalized.items()))


def _normalize_context_list(values: Sequence[str] | None) -> list[str]:
    if isinstance(values, str):
        values = (values,)
    normalized: list[str] = []
    for raw in values or ():
        item = normalize_text(str(raw))[:200]
        if item and item.casefold() not in {existing.casefold() for existing in normalized}:
            normalized.append(item)
    return sorted(normalized, key=str.casefold)[:100]


def _json_string_list(value: Any) -> list[str]:
    try:
        parsed = json.loads(str(value or "[]"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [normalize_text(str(item))[:200] for item in parsed if normalize_text(str(item))]


def _json_object(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or "{}"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _memory_applicability_key(memory: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(memory.get("context_mode") or "standalone"),
        str(memory.get("scope_json") or "{}"),
        str(memory.get("preconditions_json") or "{}"),
        str(memory.get("applicable_systems_json") or "[]"),
        str(memory.get("applicable_versions_json") or "[]"),
    )


def _semantic_consolidation_sources_are_safe(
    left: dict[str, Any],
    right: dict[str, Any],
    decision: dict[str, Any],
) -> bool:
    for source, hash_key in ((left, "left_hash"), (right, "right_hash")):
        if str(source.get("state") or "") not in {"active", "cold"}:
            return False
        if bool(source.get("pinned")) or bool(source.get("protected")):
            return False
        if str(source.get("kind") or "") in {"identity", "preference", "prospective"}:
            return False
        if source.get("quarantine_reason"):
            return False
        if str(source.get("content_hash") or "") != str(decision.get(hash_key) or ""):
            return False
    return _memory_applicability_key(left) == _memory_applicability_key(right)


def _sanitize_creation_context_map(value: dict[str, Any] | None) -> dict[str, str]:
    """Normalize and redact bounded proposal metadata before it reaches the ledger."""

    normalized = _normalize_context_map(value)
    return {
        key: sanitize_memory(item).text[:300]
        for key, item in normalized.items()
        if sanitize_memory(item).text
    }


def _sanitize_creation_context_list(values: Sequence[str] | None) -> list[str]:
    sanitized = [sanitize_memory(item).text[:200] for item in _normalize_context_list(values)]
    return _normalize_context_list([item for item in sanitized if item])


def _bounded_creation_assessment(value: dict[str, Any]) -> dict[str, Any]:
    """Keep only inspectable assessment fields with bounded, JSON-safe values."""

    if not isinstance(value, dict):
        raise ValueError("creation assessment must be a JSON object")
    return {
        "candidate_hash": normalize_text(str(value.get("candidate_hash") or ""))[:128],
        "kind": normalize_text(str(value.get("kind") or "semantic"))[:80],
        "source_type": normalize_text(str(value.get("source_type") or "conversation"))[:120],
        "context_mode": normalize_text(str(value.get("context_mode") or "standalone"))[:40],
        "reusable_score": _clamp(float(value.get("reusable_score") or 0.0)),
        "likely_useful_again": bool(value.get("likely_useful_again")),
        "durability": normalize_text(str(value.get("durability") or "unknown"))[:40],
        "quality_flags": _normalize_context_list(value.get("quality_flags") or [])[:20],
        "semantic_token_count": max(0, min(100000, int(value.get("semantic_token_count") or 0))),
        "duplicate_memory_id": normalize_text(str(value.get("duplicate_memory_id") or ""))[:100] or None,
        "contradiction_ids": _normalize_context_list(value.get("contradiction_ids") or [])[:20],
        "independently_understandable": bool(value.get("independently_understandable")),
        "metadata_completeness": _clamp(float(value.get("metadata_completeness") or 0.0)),
        "decision": normalize_text(str(value.get("decision") or "review"))[:40],
        "reason": normalize_text(str(value.get("reason") or "Candidate is waiting for review."))[:600],
        "policy_effect": {
            "automatic_action": normalize_text(
                str(dict(value.get("policy_effect") or {}).get("automatic_action") or "")
            )[:80]
            or None,
            "matched_versions": _normalize_context_list(
                dict(value.get("policy_effect") or {}).get("matched_versions") or []
            )[:20],
        },
    }


def _decode_creation_proposal(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["candidate"] = _trace_json_object(item.pop("candidate_json", "{}"))
    item["assessment"] = _trace_json_object(item.pop("assessment_json", "{}"))
    item["redacted"] = bool(item.get("redacted"))
    item["recurrence_count"] = int(item.get("recurrence_count") or 1)
    item["positive_feedback_count"] = int(item.get("positive_feedback_count") or 0)
    item["strong_feedback_count"] = int(item.get("strong_feedback_count") or 0)
    return item


def creation_proposal_revision(proposal: dict[str, Any]) -> str:
    """Return a stable token for every field that can change an automatic decision."""

    value = {
        "proposal_id": proposal.get("proposal_id"),
        "candidate_hash": proposal.get("candidate_hash"),
        "content": proposal.get("content"),
        "kind": proposal.get("kind"),
        "source_type": proposal.get("source_type"),
        "source_category": proposal.get("source_category"),
        "context_mode": proposal.get("context_mode"),
        "candidate": proposal.get("candidate"),
        "assessment": proposal.get("assessment"),
        "quarantine_reason": proposal.get("quarantine_reason"),
        "redacted": bool(proposal.get("redacted")),
        "recurrence_count": int(proposal.get("recurrence_count") or 0),
        "positive_feedback_count": int(proposal.get("positive_feedback_count") or 0),
        "strong_feedback_count": int(proposal.get("strong_feedback_count") or 0),
        "last_feedback_at": proposal.get("last_feedback_at"),
        "status": proposal.get("status"),
        "last_seen_at": proposal.get("last_seen_at"),
    }
    return hashlib.sha256(_trace_json(value).encode("utf-8")).hexdigest()


def _creation_neighborhood_preview(
    proposal: dict[str, Any],
    *,
    admitted_projects: dict[str, str] | None = None,
    admitted_services: dict[str, str] | None = None,
) -> list[str]:
    """Describe likely grouping before the candidate becomes a memory."""

    admitted_projects = admitted_projects or {}
    admitted_services = admitted_services or {}
    candidate = dict(proposal.get("candidate") or {})
    labels: list[str] = []
    content = normalize_text(str(proposal.get("content") or ""))
    if _is_credential_reference(content):
        labels.extend(["Credential references", "Tool use"])
    elif str(proposal.get("source_type") or "").casefold().startswith("tool_"):
        labels.append("Tool use")
    kind_labels = {
        "preference": "Preferences",
        "procedure": "Procedures",
        "decision": "Decisions",
        "prospective": "Plans",
        "operational": "Operations",
        "identity": "People",
    }
    kind_label = kind_labels.get(str(proposal.get("kind") or "").casefold())
    if kind_label:
        labels.append(kind_label)
    scope = dict(candidate.get("scope") or {})
    project = normalize_text(str(scope.get("project") or scope.get("active_project") or ""))
    if project:
        labels.append("Projects")
        admitted_project = admitted_projects.get(project.casefold())
        if admitted_project:
            labels.append(f"Project: {admitted_project}")
    systems = list(candidate.get("applicable_systems") or [])
    if systems:
        labels.append("Services")
    for system in systems[:8]:
        normalized = normalize_text(str(system))
        admitted_service = admitted_services.get(normalized.casefold())
        if admitted_service:
            labels.append(f"Service: {admitted_service}")
    return list(dict.fromkeys(labels))[:8]


def _is_credential_reference(content: str) -> bool:
    """Recognize safe credential locations without classifying policy prose."""

    text = normalize_text(content)
    if not re.search(r"\b(?:passwords?|credentials?|logins?|tokens?|api keys?|secrets?)\b", text, re.I):
        return False
    if re.search(
        r"\b(?:never|must\s+not|should\s+not|do(?:es)?\s+not|don['’]t|cannot|can['’]t)\b"
        r".{0,80}\b(?:store|save|keep|remember|retain|record)\w*\b",
        text,
        re.I,
    ):
        return False
    manager = r"(?:1password|bitwarden|password manager|secret manager|keychain|vault item)"
    return bool(
        re.search(
            rf"\b(?:stored|saved|kept|managed|located|available|retrieved|accessed)\b"
            rf".{{0,100}}\b(?:in|under|from|via|through|at)\s+(?:the\s+)?{manager}\b",
            text,
            re.I,
        )
        or re.search(
            rf"\b{manager}\b.{{0,100}}\b(?:item|entry|under|named|called|for)\b",
            text,
            re.I,
        )
    )


def _memory_metadata_completeness(
    context_mode: str,
    *,
    scope: dict[str, str],
    entities: Sequence[str],
    preconditions: dict[str, str],
    source_context: str | None,
    applicable_systems: Sequence[str],
    applicable_versions: Sequence[str],
) -> float:
    if context_mode == "standalone":
        signals = [bool(entities), bool(source_context)]
        if applicable_systems or applicable_versions or preconditions:
            signals.extend(
                [
                    bool(applicable_systems),
                    bool(preconditions),
                    bool(applicable_versions) or not applicable_systems,
                ]
            )
        return round(0.7 + 0.3 * sum(signals) / max(1, len(signals)), 6)
    signals = [
        bool(scope),
        bool(entities),
        bool(preconditions) or bool(scope),
        bool(source_context),
    ]
    if applicable_systems or applicable_versions:
        signals.extend([bool(applicable_systems), bool(applicable_versions) or not applicable_systems])
    return round(sum(signals) / len(signals), 6)


def _decode_memory_metadata(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    try:
        scope = json.loads(str(item.get("scope_json") or "{}"))
    except (json.JSONDecodeError, TypeError, ValueError):
        scope = {}
    try:
        preconditions = json.loads(str(item.get("preconditions_json") or "{}"))
    except (json.JSONDecodeError, TypeError, ValueError):
        preconditions = {}
    item["scope"] = scope if isinstance(scope, dict) else {}
    item["entities"] = _json_string_list(item.get("entities_json"))
    item["preconditions"] = preconditions if isinstance(preconditions, dict) else {}
    item["applicable_systems"] = _json_string_list(item.get("applicable_systems_json"))
    item["applicable_versions"] = _json_string_list(item.get("applicable_versions_json"))
    return item


def _decode_edge(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    explanation = normalize_text(str(item.get("explanation") or ""))
    if not explanation:
        explanation = _default_edge_explanation(
            str(item.get("relation") or "related"),
            evidence_count=int(item.get("evidence_count") or 1),
        )
    item["explanation"] = explanation
    item["evidence_type"] = str(item.get("evidence_type") or "legacy_unattributed")
    item["evidence_records"] = int(item.get("evidence_records") or 0)
    item["explainable"] = item["evidence_type"] != "legacy_unattributed"
    return item


def _trace_json_list(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in _trace_json_array(value) if isinstance(item, dict)]


def _trace_json_array(value: Any) -> list[Any]:
    try:
        parsed = json.loads(str(value or "[]"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return parsed


def _trace_json_object(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or "{}"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def connection_policy_selector(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, str]:
    """Return the scoped, order-independent pattern used for connection training."""

    endpoints = sorted(
        (
            normalize_text(str(memory.get("kind") or "semantic")).casefold(),
            normalize_text(str(memory.get("source_category") or "AGENT_INFERENCE")).upper(),
        )
        for memory in (left, right)
    )
    return {
        "proposal_kind": "association",
        "src_kind": endpoints[0][0],
        "src_source_category": endpoints[0][1],
        "dst_kind": endpoints[1][0],
        "dst_source_category": endpoints[1][1],
    }


def connection_pattern_label(selector: dict[str, Any]) -> str:
    def endpoint(prefix: str) -> str:
        source = normalize_text(str(selector.get(f"{prefix}_source_category") or "matching"))
        kind = normalize_text(str(selector.get(f"{prefix}_kind") or "memory"))
        return f"{source.replace('_', ' ').title()} {kind.replace('_', ' ')}"

    return f"{endpoint('src')} ↔ {endpoint('dst')}"


def _connection_review_signals(
    left: dict[str, Any], right: dict[str, Any]
) -> list[dict[str, str]]:
    """Describe inspectable overlap without claiming that overlap proves a link."""

    signals: list[dict[str, str]] = []
    left_entities = {normalize_text(str(value)) for value in left.get("entities", []) if value}
    right_entities = {normalize_text(str(value)) for value in right.get("entities", []) if value}
    shared_entities = sorted(value for value in left_entities & right_entities if value)
    if shared_entities:
        signals.append({"label": "Shared entities", "value": ", ".join(shared_entities[:6])})
    left_systems = {normalize_text(str(value)) for value in left.get("applicable_systems", []) if value}
    right_systems = {normalize_text(str(value)) for value in right.get("applicable_systems", []) if value}
    shared_systems = sorted(value for value in left_systems & right_systems if value)
    if shared_systems:
        signals.append({"label": "Shared systems", "value": ", ".join(shared_systems[:6])})
    left_scope = dict(left.get("scope") or {})
    right_scope = dict(right.get("scope") or {})
    shared_scope = [
        f"{key.replace('_', ' ')}={left_scope[key]}"
        for key in sorted(set(left_scope) & set(right_scope))
        if normalize_text(str(left_scope.get(key)))
        and normalize_text(str(left_scope.get(key))).casefold()
        == normalize_text(str(right_scope.get(key))).casefold()
    ]
    if shared_scope:
        signals.append({"label": "Same scope", "value": ", ".join(shared_scope[:6])})
    if left.get("subject") and normalize_text(str(left.get("subject"))).casefold() == normalize_text(
        str(right.get("subject"))
    ).casefold():
        signals.append({"label": "Same structured subject", "value": normalize_text(str(left["subject"]))})
    shared_terms = sorted(
        term
        for term in set(query_tokens(str(left.get("content") or "")))
        & set(query_tokens(str(right.get("content") or "")))
        if len(term) >= 4 and not term.isdigit()
    )
    if shared_terms:
        signals.append({"label": "Shared wording", "value": ", ".join(shared_terms[:8])})
    return signals


def _policy_context_key(memory: dict[str, Any], *, fallback: str) -> str:
    scope = _trace_json_object(memory.get("scope_json"))
    project = normalize_text(str(scope.get("project") or scope.get("active_project") or ""))
    systems = _trace_json_array(memory.get("applicable_systems_json"))
    source_ref = normalize_text(str(memory.get("source_ref") or ""))
    session_id = normalize_text(str(memory.get("session_id") or ""))
    if project:
        return f"project:{project.casefold()}"
    if systems:
        return f"system:{normalize_text(str(systems[0])).casefold()}"
    if source_ref:
        return f"source:{source_ref.split('#', 1)[0].casefold()[:160]}"
    if session_id:
        return f"session:{session_id.casefold()[:160]}"
    return (
        f"unscoped:{normalize_text(str(memory.get('source_type') or 'unknown')).casefold()}:"
        f"{normalize_text(str(memory.get('kind') or 'memory')).casefold()}"
    )


def _policy_contribution(
    *,
    review_id: str,
    created_at: str,
    domain: str,
    lever: str,
    selector: dict[str, Any],
    direction: str,
    context_key: str,
    reason: str,
) -> dict[str, Any]:
    signature = _trace_json({"domain": domain, "lever": lever, "selector": selector})
    signal_key = f"{domain}:{lever}:{hashlib.sha256(signature.encode('utf-8')).hexdigest()[:20]}"
    return {
        "review_id": review_id,
        "created_at": created_at,
        "domain": domain,
        "lever": lever,
        "selector": selector,
        "direction": direction,
        "context_key": context_key,
        "reason": reason,
        "signal_key": signal_key,
    }


def _normalize_review_scope(value: str | None) -> str:
    scope = normalize_text(str(value or "item_only")).casefold()
    if scope not in {"item_only", "exact_duplicates", "policy_evidence"}:
        raise ValueError("decision scope must be item_only, exact_duplicates, or policy_evidence")
    return scope


def _policy_candidate_copy(
    domain: str, direction: str, selector: dict[str, Any]
) -> tuple[str, str, dict[str, Any]]:
    kind = str(selector.get("kind") or "memory")
    source = str(selector.get("source_category") or selector.get("source_type") or "matching")
    if domain == "retrieval":
        if direction == "boost":
            return (
                f"Give helpful {source} {kind} memories a small ranking lift",
                "Repeated outcome reviews say this source-and-kind pattern tends to help Kaya. "
                "The adjustment is bounded and still cannot bypass scope, relevance, or context gates.",
                {"score_adjustment": 0.035},
            )
        return (
            f"Downrank misleading {source} {kind} memories",
            "Repeated outcome reviews say this source-and-kind pattern needs more caution. "
            "The adjustment lowers rank without deleting the underlying evidence.",
            {"score_adjustment": -0.05},
        )
    if domain == "retention":
        if direction == "preserve":
            return (
                f"Preserve useful {source} {kind} memories longer",
                "Repeated cleanup decisions say this pattern is being proposed for pruning too aggressively.",
                {"score_adjustment": 0.08},
            )
        return (
            f"Move low-value {source} {kind} memories toward review sooner",
            "Repeated cleanup decisions say this pattern consumes memory capacity without enough durable value. "
            "The adjustment changes lifecycle scoring but still produces reviewable proposals.",
            {"score_adjustment": -0.08},
        )
    if domain == "connection":
        pattern = connection_pattern_label(selector)
        return (
            f"Require another independent witness for {pattern} links",
            "Repeated denials say co-occurrence alone is producing links that do not make conceptual sense. "
            "Future Sleep proposals in this pattern must clear a higher evidence threshold, and promotion "
            "withdraws still-pending weak proposals that no longer meet it.",
            {"min_independent_witnesses_delta": 1},
        )
    quality_flag = str(selector.get("quality_flag") or "low-quality")
    return (
        f"Block automatic {source} {kind} writes flagged {quality_flag}",
        "Repeated cleanup decisions identify a reusable admission failure. Explicit writes remain possible, "
        "but matching automatic candidates are kept out of recallable memory.",
        {"automatic_action": "ignore", "quality_flag": quality_flag},
    )


def _policy_selector_matches(selector: dict[str, Any], features: dict[str, Any]) -> bool:
    for key, expected in selector.items():
        if key == "quality_flag":
            flags = features.get("quality_flags") or features.get("quality_flag") or []
            if isinstance(flags, str):
                flags = [flags]
            if str(expected).casefold() not in {str(value).casefold() for value in flags}:
                return False
            continue
        actual = features.get(key)
        if isinstance(actual, (list, tuple, set)):
            if str(expected).casefold() not in {str(value).casefold() for value in actual}:
                return False
        elif str(actual or "").casefold() != str(expected or "").casefold():
            return False
    return True


def _normalize_retrieval_context(value: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(value or {})
    active_project = normalize_text(str(raw.get("active_project") or ""))[:200]
    scope = _normalize_context_map(
        raw.get("scope") if isinstance(raw.get("scope"), dict) else {}
    )
    task_type = normalize_text(
        str(raw.get("task_type") or scope.get("task_type") or "general")
    )[:80] or "general"
    return {
        "active_project": active_project or None,
        "task_type": task_type,
        "entities": _normalize_context_list(raw.get("entities")),
        "scope": scope,
        "system_state": _normalize_context_map(
            raw.get("system_state") if isinstance(raw.get("system_state"), dict) else {}
        ),
        "applicable_systems": _normalize_context_list(raw.get("applicable_systems")),
        "applicable_versions": _normalize_context_list(raw.get("applicable_versions")),
    }


def _retrieval_context_key(value: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    normalized = _normalize_retrieval_context(value)
    # Session and conversation identifiers make every task unique and therefore
    # cannot support longitudinal learning. Keep them in the trace, but remove
    # them from the context bucket used for adaptive usefulness.
    stable_scope = {
        key: item
        for key, item in dict(normalized["scope"]).items()
        if key not in {"conversation", "session", "session_id", "thread", "task_id"}
    }
    signature = {**normalized, "scope": stable_scope}
    encoded = _trace_json(signature)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), signature


def _normalize_trace_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    components: dict[str, float] = {}
    raw_components = candidate.get("components")
    if isinstance(raw_components, dict):
        for key, value in raw_components.items():
            try:
                components[normalize_text(str(key))[:80]] = round(float(value), 6)
            except (TypeError, ValueError):
                continue
    try:
        score = round(_clamp(float(candidate.get("score") or 0.0)), 6)
    except (TypeError, ValueError):
        score = 0.0
    try:
        rank = max(0, int(candidate.get("rank") or 0))
    except (TypeError, ValueError):
        rank = 0
    try:
        estimated_tokens = max(0, int(candidate.get("estimated_tokens") or 0))
    except (TypeError, ValueError):
        estimated_tokens = 0
    context_mode = str(candidate.get("context_mode") or "standalone")
    try:
        context_mode = _normalize_context_mode(context_mode)
    except ValueError:
        context_mode = "standalone"
    return {
        "memory_id": normalize_text(str(candidate.get("memory_id") or ""))[:120],
        "kind": normalize_text(str(candidate.get("kind") or "semantic"))[:40],
        "context_mode": context_mode,
        "scope": _normalize_context_map(
            candidate.get("scope") if isinstance(candidate.get("scope"), dict) else {}
        ),
        "entities": _normalize_context_list(
            candidate.get("entities") if isinstance(candidate.get("entities"), (list, tuple)) else ()
        ),
        "preconditions": _normalize_context_map(
            candidate.get("preconditions") if isinstance(candidate.get("preconditions"), dict) else {}
        ),
        "applicable_systems": _normalize_context_list(
            candidate.get("applicable_systems")
            if isinstance(candidate.get("applicable_systems"), (list, tuple))
            else ()
        ),
        "applicable_versions": _normalize_context_list(
            candidate.get("applicable_versions")
            if isinstance(candidate.get("applicable_versions"), (list, tuple))
            else ()
        ),
        "content_preview": normalize_text(str(candidate.get("content_preview") or ""))[:180],
        "rank": rank,
        "selected": bool(candidate.get("selected")),
        "score": score,
        "estimated_tokens": estimated_tokens,
        "scoring_policy_version": normalize_text(
            str(candidate.get("scoring_policy_version") or "legacy_unversioned")
        )[:80],
        "shadow_scoring_policy_version": normalize_text(
            str(candidate.get("shadow_scoring_policy_version") or "")
        )[:80],
        "components": components,
        "reason": normalize_text(str(candidate.get("reason") or "unspecified decision"))[:300],
        **(
            {"metacognition": dict(candidate["metacognition"])}
            if isinstance(candidate.get("metacognition"), dict)
            else {}
        ),
    }


def _normalize_trace_action(action: dict[str, Any]) -> dict[str, Any]:
    action_name = normalize_text(str(action.get("action") or "ignored")).casefold()
    if action_name not in {"created", "updated", "ignored", "proposed"}:
        action_name = "updated" if action_name else "ignored"
    memory_id = normalize_text(str(action.get("memory_id") or ""))[:120] or None
    result = {
        "action": action_name,
        "memory_id": memory_id,
        "reason": normalize_text(str(action.get("reason") or "No storage reason was recorded."))[:400],
    }
    if action.get("kind"):
        result["kind"] = normalize_text(str(action["kind"]))[:40]
    if action.get("state"):
        result["state"] = normalize_text(str(action["state"]))[:40]
    if action.get("proposal_id"):
        result["proposal_id"] = normalize_text(str(action["proposal_id"]))[:120]
    return result


def _clamp(value: float) -> float:
    if not math.isfinite(float(value)):
        raise ValueError("score must be finite")
    return max(0.0, min(1.0, float(value)))


def _attentional_decay_days() -> float:
    try:
        value = float(os.environ.get("CORTEX_ATTENTIONAL_DECAY_DAYS", "30"))
    except ValueError:
        value = 30.0
    return max(1.0, min(365.0, value))


def _percentile(values: Sequence[float | int], quantile: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * max(0.0, min(1.0, quantile))
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(float(values[lower]), 4)
    fraction = position - lower
    return round(float(values[lower]) + (float(values[upper]) - float(values[lower])) * fraction, 4)


def _memory_quality_pruning_action(memory: dict[str, Any]) -> tuple[str, str] | None:
    """Return an immediate reversible lifecycle action for deterministic noise.

    This is intentionally narrow.  It never uses lack of retrieval by itself,
    and durable preferences, identity, prospective items, pinned memories, and
    protected memories are filtered before this helper is called.
    """

    source_type = str(memory.get("source_type") or "").casefold()
    extraction_method = str(memory.get("extraction_method") or "").casefold()
    content = normalize_text(str(memory.get("content") or ""))
    positive = sum(
        int(memory.get(field, 0) or 0)
        for field in ("helpful_count", "validated_count", "confirmed_count", "success_count")
    )
    if source_type in _TOOL_TELEMETRY_SOURCE_TYPES or extraction_method in {
        "tool_outcome_observer_v1",
        "tool_outcome_aggregator_v1",
        "tool_workflow_aggregator_v1",
    }:
        if positive <= 0:
            return (
                "archived",
                "Legacy tool-call telemetry is already preserved in the tool execution/statistics ledger; "
                "archive the duplicate recallable memory without deleting its history.",
            )
    if (
        str(memory.get("source_type") or "") == "vault_markdown"
        and _is_placeholder_only_content(content)
        and int(memory.get("used_count", 0) or 0) == 0
        and positive <= 0
    ):
        return (
            "cold",
            "The vault chunk is placeholder text with no attributed use; cool it for review instead of deleting it.",
        )
    if (
        _is_transient_automation_noise(
            content,
            kind=str(memory.get("kind") or "semantic"),
            source_type=source_type,
        )
        and int(memory.get("used_count", 0) or 0) == 0
        and positive <= 0
    ):
        return (
            "cold",
            "The memory is a transient execution status with no positive outcome evidence; cool it for review.",
        )
    return None


def _is_placeholder_only_content(content: str) -> bool:
    """Distinguish an empty template from a substantive note mentioning one."""

    clean = normalize_text(content)
    if not _PLACEHOLDER_CONTENT.search(clean):
        return False
    body = re.sub(
        r"^Vault note:[^\n]*\nPath:[^\n]*\nSection:[^\n]*\n?",
        "",
        clean,
        flags=re.I,
    )
    without_markers = _PLACEHOLDER_CONTENT.sub(" ", body)
    meaningful = [
        token.casefold().strip("'-")
        for token in _TOKEN.findall(without_markers)
        if token.casefold().strip("'-") not in _STOP
    ]
    return len(meaningful) < 5


def _is_transient_automation_noise(content: str, *, kind: str, source_type: str) -> bool:
    """Recognize execution status while preserving durable runbooks and fixes."""

    clean = normalize_text(content)
    if not _AUTOMATION_NOISE.search(clean):
        return False
    source = normalize_text(source_type).casefold()
    if source in _TOOL_TELEMETRY_SOURCE_TYPES:
        return True
    if re.search(r"^tool execution observation:|\bexpected argument keys\b|^reinforced [a-z0-9_-]+ workflow:", clean, re.I):
        return True
    return kind == "episode"


def _retention_score(memory: dict[str, Any]) -> float:
    # Raw selections/injections never earn retention. Even actual use is
    # burst-capped; stronger credit comes from helpful use across independent
    # tasks and days in the experience ledger.
    used = min(2, int(memory.get("used_count", 0)))
    positive = (
        int(memory.get("success_count", 0))
        + int(memory.get("confirmed_count", 0))
        + int(memory.get("validated_count", 0))
        + int(memory.get("helpful_count", 0))
    )
    harmful = int(memory.get("harmful_count", 0)) + int(memory.get("false_positive_count", 0))
    utility = max(0.0, min(1.0, (positive + 1.0) / (used + 2.0) - harmful / max(3.0, used + 2.0)))
    frequency = min(1.0, math.log1p(used + positive) / math.log(12.0))
    spaced = _clamp(float(memory.get("spaced_reinforcement") or 0.0))
    unused_inhibition = min(1.0, float(memory.get("selected_unused") or 0.0) / 5.0)
    score = (
        0.24 * float(memory.get("importance", 0.5))
        + 0.18 * float(memory.get("confidence", 0.6)) * float(memory.get("trust", 0.7))
        + 0.12 * float(memory.get("currentness_confidence", 0.7))
        + 0.15 * float(memory.get("uniqueness", 1.0))
        + 0.22 * utility
        + 0.05 * frequency
        + 0.10 * spaced
        - 0.08 * unused_inhibition
    )
    if int(memory.get("duplicate_count", 0)) > 0:
        score -= min(0.12, 0.03 * int(memory["duplicate_count"]))
    return _clamp(score)


def _pruning_relevance_score(memory: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    now = datetime.now(timezone.utc)

    def days_since(value: Any, fallback: float) -> float:
        parsed = parse_iso8601(str(value or ""))
        if not parsed:
            return fallback
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (now - parsed).total_seconds() / 86400.0)

    age_days = days_since(memory.get("created_at"), 3650.0)
    last_recall_days = days_since(
        memory.get("last_helpful_at")
        or memory.get("last_used_at")
        or memory.get("last_retrieved_at")
        or memory.get("updated_at"),
        age_days,
    )
    retrieved = max(0, int(memory.get("retrieved_count") or 0))
    used = max(0, int(memory.get("used_count") or 0))
    positive = sum(
        max(0, int(memory.get(key) or 0))
        for key in ("success_count", "confirmed_count", "validated_count", "helpful_count")
    )
    harmful = max(0, int(memory.get("harmful_count") or 0))
    false_positive = max(0, int(memory.get("false_positive_count") or 0))
    recency_signal = math.exp(-last_recall_days / 120.0)
    age_signal = math.exp(-age_days / 730.0)
    retrieval_signal = min(1.0, math.log1p(retrieved) / math.log(20.0))
    helpfulness_ratio = positive / max(1.0, used + positive + harmful + false_positive)
    successful_use_signal = min(1.0, math.log1p(positive) / math.log(8.0))
    harmful_rate = (harmful + false_positive) / max(
        1.0,
        used + positive + harmful + false_positive,
    )
    retention = _retention_score(memory)
    relevance = _clamp(
        0.40 * retention
        + 0.18 * recency_signal
        + 0.10 * retrieval_signal
        + 0.14 * helpfulness_ratio
        + 0.10 * successful_use_signal
        + 0.08 * age_signal
        - 0.18 * harmful_rate
    )
    evidence = {
        "days_since_last_recall": round(last_recall_days, 3),
        "age_days": round(age_days, 3),
        "retrieval_count": retrieved,
        "used_count": used,
        "positive_outcomes": positive,
        "harmful_outcomes": harmful,
        "false_positive_count": false_positive,
        "helpfulness_ratio": round(helpfulness_ratio, 6),
        "harmful_rate": round(harmful_rate, 6),
        "retention_score": round(retention, 6),
        "recency_signal": round(recency_signal, 6),
        "retrieval_signal": round(retrieval_signal, 6),
        "successful_use_signal": round(successful_use_signal, 6),
        "age_signal": round(age_signal, 6),
    }
    return round(relevance, 6), evidence


def _sleep_hypothesis_text(kind: str) -> str:
    category = kind.casefold()
    if category in {"association", "association_reinforcement"}:
        return "If this evidence-backed connection is applied, future cues should retrieve both memories more reliably."
    if category in {"edge_downscale", "lifecycle"}:
        return "If this cooling or pruning change is applied, context cost should fall without later pruning regret."
    if category == "consolidation":
        return "If these duplicates are consolidated, retrieval redundancy should fall without losing labeled recall."
    if category in {"interference_review", "conflict"}:
        return "If this interference is resolved, competing-fact errors should fall on time-labeled questions."
    if category == "dependency_repair":
        return "If dependent evidence is repaired, unsupported active inference should fall without hiding valid claims."
    return "If this proposal is applied, its named quality metric should improve without harming retrieval or reversibility."


def _periods_overlap(
    left_start: str | None,
    left_end: str | None,
    right_start: str | None,
    right_end: str | None,
) -> bool:
    """Return whether two optional ISO validity intervals overlap."""
    try:
        ls = datetime.fromisoformat(left_start) if left_start else datetime.min.replace(tzinfo=timezone.utc)
        le = datetime.fromisoformat(left_end) if left_end else datetime.max.replace(tzinfo=timezone.utc)
        rs = datetime.fromisoformat(right_start) if right_start else datetime.min.replace(tzinfo=timezone.utc)
        re_ = datetime.fromisoformat(right_end) if right_end else datetime.max.replace(tzinfo=timezone.utc)
        for value_name, value in (("ls", ls), ("le", le), ("rs", rs), ("re", re_)):
            if value.tzinfo is None:
                if value_name == "ls":
                    ls = value.replace(tzinfo=timezone.utc)
                elif value_name == "le":
                    le = value.replace(tzinfo=timezone.utc)
                elif value_name == "rs":
                    rs = value.replace(tzinfo=timezone.utc)
                else:
                    re_ = value.replace(tzinfo=timezone.utc)
        return ls <= re_ and rs <= le
    except ValueError:
        return True
