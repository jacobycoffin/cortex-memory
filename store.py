"""SQLite persistence for Cortex adaptive memory."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from .security import normalize_text
from .semantics import feature_similarity, semantic_features


SCHEMA_VERSION = 6
_TOKEN = re.compile(r"[\w'-]{2,}", re.UNICODE)
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


def content_hash(content: str) -> str:
    return hashlib.sha256(normalize_text(content).casefold().encode("utf-8")).hexdigest()


def query_tokens(text: str) -> list[str]:
    tokens = [t.casefold().strip("'-") for t in _TOKEN.findall(text or "")]
    return list(dict.fromkeys(t for t in tokens if t and t not in _STOP))[:24]


class CortexStore:
    """Thread-safe, auditable SQLite store with explicit lifecycle states."""

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._local_retrieval_revision = 0
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
                source_ref TEXT,
                session_id TEXT,
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
                state TEXT NOT NULL DEFAULT 'active',
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

            CREATE TABLE IF NOT EXISTS feature_stats (
                feature TEXT PRIMARY KEY,
                document_frequency INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        self._migrate_columns()
        self._backfill_memory_features()
        self._rebuild_feature_stats()
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_claim ON memories(subject, predicate, state, valid_from, valid_to)"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_dirty ON memories(dirty, state)")
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )
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
            AFTER UPDATE OF kind,content,source_category,valid_from,valid_to,subject,predicate,object_value,
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

    def _migrate_columns(self) -> None:
        """Add v2 columns safely when opening an early Cortex prototype DB."""
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(memories)")}
        additions = {
            "source_category": "TEXT NOT NULL DEFAULT 'AGENT_INFERENCE'",
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
        }
        for name, declaration in additions.items():
            if name not in columns:
                self._conn.execute(f"ALTER TABLE memories ADD COLUMN {name} {declaration}")
        self._conn.execute("UPDATE memories SET observed_at=COALESCE(observed_at, created_at)")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def add_memory(
        self,
        content: str,
        *,
        kind: str = "semantic",
        source_type: str = "conversation",
        source_category: str = "AGENT_INFERENCE",
        source_ref: str | None = None,
        session_id: str | None = None,
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
    ) -> tuple[str, bool]:
        content = normalize_text(content)
        if not content:
            raise ValueError("memory content cannot be empty")
        digest = content_hash(content)
        now = utc_now()
        state = "quarantine" if quarantine_reason else state
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT id FROM memories WHERE content_hash=? ORDER BY created_at LIMIT 1", (digest,)
            ).fetchone()
            if existing:
                memory_id = str(existing["id"])
                conn.execute(
                    """UPDATE memories
                       SET duplicate_count=duplicate_count+1, updated_at=?,
                           importance=MAX(importance, ?), confidence=MAX(confidence, ?),
                           pinned=MAX(pinned, ?), protected=MAX(protected, ?),
                           trust=MAX(trust, ?), uniqueness=MIN(uniqueness, ?),
                           source_category=CASE WHEN ?='USER_EXPLICIT' THEN 'USER_EXPLICIT' ELSE source_category END
                       WHERE id=?""",
                    (
                        now,
                        _clamp(importance),
                        _clamp(confidence),
                        int(pinned),
                        int(protected or pinned or kind == "prospective"),
                        _clamp(trust),
                        _clamp(uniqueness),
                        source_category,
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
                return memory_id, False

            memory_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO memories(
                    id, kind, content, content_hash, source_type, source_category, source_ref, session_id,
                    created_at, updated_at, observed_at, valid_from, valid_to, subject, predicate, object_value,
                    extraction_method, confidence, currentness_confidence, importance, uniqueness,
                    volatility, trust, state, pinned, protected, supersedes_id, quarantine_reason
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    memory_id,
                    kind,
                    content,
                    digest,
                    source_type,
                    source_category,
                    source_ref,
                    session_id,
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
            self._link_structured_contradictions(
                conn, memory_id, subject, predicate, object_value, valid_from, valid_to, now
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
        with self.transaction() as conn:
            current = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            if not current:
                return False
            next_confidence = _clamp(confidence if confidence is not None else max(0.55, current["confidence"]))
            conn.execute(
                "UPDATE memory_versions SET system_to=? WHERE memory_id=? AND system_to IS NULL",
                (now, memory_id),
            )
            conn.execute(
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
            self._mark_dependents_dirty_tx(conn, memory_id, f"evidence corrected: {reason}")
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
        return dict(row) if row else None

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        return dict(row) if row else None

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
        by_id = {row["id"]: dict(row) for row in rows}
        return [by_id[mid] for mid in memory_ids if mid in by_id]

    def retrieval_revision(self) -> tuple[int, int]:
        """Return the durable revision used to invalidate retrieval caches.

        The revision advances for material memory, association, and learned-tool
        changes. Pure retrieval/injection counters intentionally do not advance
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
        return [dict(row) for row in rows]

    def fts_search(self, query: str, *, limit: int = 40, include_archived: bool = False) -> list[dict[str, Any]]:
        tokens = query_tokens(query)
        states = ("active", "cold", "archived") if include_archived else ("active", "cold")
        if not tokens:
            placeholders = ",".join("?" for _ in states)
            with self._lock:
                rows = self._conn.execute(
                    f"SELECT m.*, 8.0 AS fts_rank FROM memories m WHERE state IN ({placeholders}) "
                    "ORDER BY pinned DESC, updated_at DESC LIMIT ?",
                    (*states, limit),
                ).fetchall()
            return [dict(r) for r in rows]
        match = " OR ".join(f'"{t.replace(chr(34), "")}"' for t in tokens)
        placeholders = ",".join("?" for _ in states)
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT m.*, bm25(memory_fts) AS fts_rank
                    FROM memory_fts JOIN memories m ON m.id=memory_fts.memory_id
                    WHERE memory_fts MATCH ? AND m.state IN ({placeholders})
                    ORDER BY bm25(memory_fts) LIMIT ?""",
                (match, *states, limit),
            ).fetchall()
        return [dict(r) for r in rows]

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
                    WHERE m.state IN ({state_placeholders})
                    ORDER BY s.feature_score DESC,s.feature_matches DESC,m.pinned DESC
                    LIMIT ?""",
                tuple(params),
            ).fetchall()
        return [dict(row) for row in rows]

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
                    f"""SELECT src_id,dst_id FROM edges
                        WHERE src_id IN ({placeholders}) OR dst_id IN ({placeholders})
                        ORDER BY weight DESC,evidence_count DESC LIMIT ?""",
                    (*frontier, *frontier, max_nodes * 3),
                ).fetchall()
                expanded = {str(row["src_id"]) for row in rows} | {str(row["dst_id"]) for row in rows}
                expanded -= nodes
                room = max_nodes - len(nodes)
                frontier = set(sorted(expanded)[:room])
                nodes.update(frontier)
            if not nodes:
                return {}
            placeholders = ",".join("?" for _ in nodes)
            edge_rows = self._conn.execute(
                f"""SELECT src_id,dst_id,weight,evidence_count FROM edges
                    WHERE src_id IN ({placeholders}) AND dst_id IN ({placeholders})
                    LIMIT ?""",
                (*nodes, *nodes, max_nodes * 12),
            ).fetchall()

        adjacency: dict[str, list[tuple[str, float]]] = {node: [] for node in nodes}
        for row in edge_rows:
            left, right = str(row["src_id"]), str(row["dst_id"])
            weight = float(row["weight"]) * min(2.0, math.log1p(int(row["evidence_count"])))
            adjacency[left].append((right, weight))
            adjacency[right].append((left, weight))
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

    def add_edge(self, src_id: str, dst_id: str, relation: str, *, weight: float = 0.25) -> bool:
        if not src_id or not dst_id or src_id == dst_id:
            return False
        src_id, dst_id = (
            sorted((src_id, dst_id))
            if relation in {"related", "co_used", "co_observed", "sleep_replay"}
            else (src_id, dst_id)
        )
        now = utc_now()
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO edges(src_id,dst_id,relation,weight,evidence_count,created_at,last_reinforced_at)
                   VALUES(?,?,?,?,1,?,?)
                   ON CONFLICT(src_id,dst_id,relation) DO UPDATE SET
                     weight=MIN(1.0, edges.weight + excluded.weight),
                     evidence_count=edges.evidence_count+1,
                     last_reinforced_at=excluded.last_reinforced_at""",
                (src_id, dst_id, relation, _clamp(weight), now, now),
            )
            return True

    def resolve_contradiction(self, first_id: str, second_id: str, resolution: str) -> bool:
        """Resolve one contradictory pair while preserving an auditable history."""
        if resolution not in {"keep_first", "keep_second", "both_valid"}:
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
            if resolution == "both_valid":
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

    def review_inference(self, memory_id: str, resolution: str) -> bool:
        """Confirm or archive one unsupported inference from the Health reviewer."""
        if resolution not in {"confirm", "archive"}:
            raise ValueError("invalid inference resolution")
        now = utc_now()
        with self.transaction() as conn:
            memory = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            if not memory or memory["source_category"] not in {"AGENT_INFERENCE", "REFLECTION"}:
                return False
            if memory["state"] not in {"active", "cold"}:
                return False
            supported = conn.execute(
                "SELECT 1 FROM memory_dependencies WHERE memory_id=? AND active=1 LIMIT 1",
                (memory_id,),
            ).fetchone()
            if supported:
                return False

            if resolution == "confirm":
                conn.execute(
                    """UPDATE memories SET source_category='USER_EXPLICIT',
                       confidence=MAX(confidence,0.85),trust=MAX(trust,0.85),
                       confirmed_count=confirmed_count+1,protected=1,updated_at=? WHERE id=?""",
                    (now, memory_id),
                )
                event = "confirmed"
            else:
                conn.execute(
                    "UPDATE memory_versions SET system_to=? WHERE memory_id=? AND system_to IS NULL",
                    (now, memory_id),
                )
                conn.execute(
                    """INSERT INTO memory_versions(memory_id,content,confidence,state,valid_from,valid_to,
                       system_from,reason,source_ref) VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        memory_id,
                        memory["content"],
                        memory["confidence"],
                        "archived",
                        memory["valid_from"],
                        memory["valid_to"],
                        now,
                        "dashboard inference review: not retained",
                        memory["source_ref"],
                    ),
                )
                conn.execute("UPDATE memories SET state='archived', updated_at=? WHERE id=?", (now, memory_id))
                conn.execute(
                    """INSERT INTO lifecycle_events(
                       memory_id,from_state,to_state,reason,retention_score,created_at
                       ) VALUES(?,?,?,?,?,?)""",
                    (
                        memory_id,
                        memory["state"],
                        "archived",
                        "dashboard inference review: not retained",
                        None,
                        now,
                    ),
                )
                self._mark_dependents_dirty_tx(conn, memory_id, "inference archived in dashboard review")
                event = "archived"

            conn.execute(
                "INSERT INTO access_log(memory_id,event,query,created_at) VALUES(?,?,?,?)",
                (memory_id, event, "dashboard health review", now),
            )
            conn.execute(
                "INSERT INTO maintenance_log(action,details,dry_run,created_at) VALUES(?,?,0,?)",
                (
                    "dashboard_inference_review",
                    json.dumps({"memory_id": memory_id, "resolution": resolution}, sort_keys=True),
                    now,
                ),
            )
        return True

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
            active = [d for d in deps if d["active"] and d["evidence_state"] in {"active", "cold"}]
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
            "wrong": ("harmful_count", "last_used_at"),
            "irrelevant": ("false_positive_count", "last_used_at"),
        }
        now = utc_now()
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
    ) -> str:
        recall_id = str(uuid.uuid4())
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO recall_runs(
                   recall_id,session_id,query,mode,reason,requested_limit,token_budget,
                   candidate_count,selected_count,estimated_tokens,prepare_ms,abstained,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    recall_id,
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

    def record_pruning_regret(
        self,
        memory_id: str,
        *,
        query: str,
        score: float,
        restore: bool = False,
    ) -> bool:
        memory = self.get_memory(memory_id)
        if not memory or memory["state"] != "archived":
            return False
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO pruning_regret(memory_id,query,score,restored,created_at)
                   VALUES(?,?,?,?,?)""",
                (memory_id, normalize_text(query)[:500], _clamp(score), int(restore), utc_now()),
            )
        if restore:
            return self.set_state(memory_id, "active", reason="automatic pruning-regret restoration")
        return True

    def feedback(self, memory_ids: Sequence[str], outcome: str, *, session_id: str | None = None) -> int:
        event = {
            "useful": "used",
            "successful": "successful",
            "confirmed": "confirmed",
            "irrelevant": "irrelevant",
            "wrong": "wrong",
        }.get(outcome)
        if not event:
            raise ValueError("outcome must be useful, successful, confirmed, irrelevant, or wrong")
        count = 0
        for memory_id in dict.fromkeys(memory_ids):
            if self.get_memory(memory_id):
                self.log_access(memory_id, event, session_id=session_id)
                count += 1
        return count

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
    ) -> str:
        task_id = str(uuid.uuid4())
        now = utc_now()
        with self.transaction() as conn:
            for memory_id, score in items:
                conn.execute(
                    """INSERT INTO usage_records(
                       usage_id,task_id,memory_id,session_id,query,score,selected,created_at
                       ) VALUES(?,?,?,?,?,?,1,?)""",
                    (str(uuid.uuid4()), task_id, memory_id, session_id, query, score, now),
                )
            if items and task_type and recall_mode:
                conn.execute(
                    """INSERT INTO recall_budget_observations(
                       task_id,task_type,mode,requested_budget,estimated_tokens,
                       selected_count,created_at
                       ) VALUES(?,?,?,?,?,?,?)""",
                    (
                        task_id,
                        normalize_text(task_type)[:80],
                        normalize_text(recall_mode)[:40],
                        max(1, int(requested_budget)),
                        max(0, int(estimated_tokens)),
                        len(items),
                        now,
                    ),
                )
        return task_id

    def resolve_usage(self, task_id: str, attribution_by_id: dict[str, float]) -> int:
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
        return resolved

    def apply_task_outcome(self, task_id: str, outcome: str) -> list[str]:
        if outcome not in {"helpful", "harmful", "validated", "corrected"}:
            raise ValueError("invalid task outcome")
        with self.transaction() as conn:
            rows = conn.execute("SELECT memory_id FROM usage_records WHERE task_id=? AND used=1", (task_id,)).fetchall()
            ids = [str(row["memory_id"]) for row in rows]
            conn.execute(
                "UPDATE usage_records SET outcome=?,resolved_at=? WHERE task_id=? AND used=1",
                (outcome, utc_now(), task_id),
            )
            conn.execute(
                """UPDATE recall_budget_observations SET outcome=?,resolved_at=?
                   WHERE task_id=? AND used_count>0""",
                (outcome, utc_now(), task_id),
            )
        for memory_id in ids:
            self.log_access(memory_id, outcome if outcome != "harmful" else "wrong")
        return ids

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

    def explain(self, memory_id: str) -> dict[str, Any] | None:
        memory = self.get_memory(memory_id)
        if not memory:
            return None
        with self._lock:
            edges = self._conn.execute(
                "SELECT * FROM edges WHERE src_id=? OR dst_id=? ORDER BY weight DESC LIMIT 20",
                (memory_id, memory_id),
            ).fetchall()
            accesses = self._conn.execute(
                "SELECT event,query,session_id,score,created_at FROM access_log WHERE memory_id=? ORDER BY id DESC LIMIT 20",
                (memory_id,),
            ).fetchall()
        return {
            "memory": memory,
            "versions": self.versions(memory_id),
            "edges": [dict(r) for r in edges],
            "dependencies": self.dependencies(memory_id),
            "recent_access": [dict(r) for r in accesses],
        }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            state_rows = self._conn.execute("SELECT state,COUNT(*) AS n FROM memories GROUP BY state").fetchall()
            kind_rows = self._conn.execute("SELECT kind,COUNT(*) AS n FROM memories GROUP BY kind").fetchall()
            counts = self._conn.execute(
                "SELECT (SELECT COUNT(*) FROM memories) memories, (SELECT COUNT(*) FROM edges) edges, "
                "(SELECT COUNT(*) FROM episodes) episodes, (SELECT COUNT(*) FROM access_log) accesses, "
                "(SELECT COUNT(*) FROM usage_records WHERE outcome='pending') pending_usage, "
                "(SELECT COUNT(*) FROM memories WHERE dirty=1) dirty_memories, "
                "(SELECT COUNT(*) FROM tool_executions) tool_executions, "
                "(SELECT COUNT(*) FROM tool_stats) tool_strategies, "
                "(SELECT COUNT(*) FROM tool_workflows) tool_workflows, "
                "(SELECT COUNT(*) FROM recall_runs) recall_runs, "
                "(SELECT COUNT(*) FROM recall_budget_observations WHERE outcome<>'pending') budget_observations, "
                "(SELECT COUNT(*) FROM pruning_regret) pruning_regrets, "
                "(SELECT COUNT(*) FROM consolidation_runs WHERE dry_run=0) consolidations, "
                "(SELECT COUNT(*) FROM sleep_runs) sleep_runs, "
                "(SELECT COUNT(*) FROM sleep_proposals WHERE status='proposed') sleep_proposals, "
                "(SELECT COUNT(*) FROM document_sources WHERE status='active') documents, "
                "(SELECT COUNT(*) FROM document_chunks WHERE active=1) document_chunks"
            ).fetchone()
        return {
            **dict(counts),
            "states": {r["state"]: r["n"] for r in state_rows},
            "kinds": {r["kind"]: r["n"] for r in kind_rows},
            "db_path": str(self.path),
            "db_bytes": self.path.stat().st_size if self.path.exists() else 0,
            "schema_version": SCHEMA_VERSION,
        }

    def dashboard_snapshot(self, *, memory_limit: int = 1000) -> dict[str, Any]:
        """Return a bounded read-only snapshot for the local visualization dashboard."""
        with self._lock:
            memory_rows = self._conn.execute(
                """SELECT id,kind,content,source_type,source_category,source_ref,session_id,
                   created_at,updated_at,observed_at,valid_from,valid_to,subject,predicate,object_value,
                   extraction_method,confidence,currentness_confidence,importance,uniqueness,volatility,
                   trust,state,pinned,protected,dirty,dirty_reason,supersedes_id,quarantine_reason,
                   retrieved_count,selected_count,injected_count,used_count,success_count,confirmed_count,
                   validated_count,helpful_count,harmful_count,correction_count,false_positive_count,
                   duplicate_count,last_retrieved_at,last_injected_at,last_used_at,last_helpful_at
                   FROM memories ORDER BY observed_at DESC LIMIT ?""",
                (max(1, min(memory_limit, 2000)),),
            ).fetchall()
            ids = [str(row["id"]) for row in memory_rows]
            edge_rows: list[sqlite3.Row] = []
            dependency_rows: list[sqlite3.Row] = []
            if ids:
                placeholders = ",".join("?" for _ in ids)
                edge_rows = self._conn.execute(
                    f"""SELECT src_id,dst_id,relation,weight,evidence_count,last_reinforced_at
                       FROM edges WHERE src_id IN ({placeholders}) AND dst_id IN ({placeholders})
                       ORDER BY weight DESC LIMIT 4000""",
                    (*ids, *ids),
                ).fetchall()
                dependency_rows = self._conn.execute(
                    f"""SELECT memory_id,evidence_id,relation,weight,active,created_at
                       FROM memory_dependencies
                       WHERE memory_id IN ({placeholders}) AND evidence_id IN ({placeholders})
                       LIMIT 4000""",
                    (*ids, *ids),
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
                """SELECT recall_id,mode,reason,requested_limit,token_budget,candidate_count,
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
                          reflection_billed_tokens,reflection_status,error,started_at,completed_at
                   FROM sleep_runs ORDER BY started_at DESC LIMIT 100"""
            ).fetchall()
            sleep_proposal_rows = self._conn.execute(
                """SELECT proposal_id,run_id,kind,src_id,dst_id,status,score,evidence_count,
                          rationale,created_at
                   FROM sleep_proposals ORDER BY created_at DESC LIMIT 250"""
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
        return {
            "generated_at": utc_now(),
            "stats": self.stats(),
            "memories": [dict(row) for row in memory_rows],
            "edges": [dict(row) for row in edge_rows],
            "dependencies": [dict(row) for row in dependency_rows],
            "tools": [dict(row) for row in tool_rows],
            "sources": [dict(row) for row in source_rows],
            "recent_access": [dict(row) for row in recent_access_rows],
            "access_by_day": [dict(row) for row in access_rows],
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
            "lifecycle_events": [dict(row) for row in lifecycle_rows],
            "pruning_regrets": [dict(row) for row in regret_rows],
            "consolidation_runs": [dict(row) for row in consolidation_rows],
            "tool_workflows": [dict(row) for row in workflow_rows],
            "sleep_runs": [dict(row) for row in sleep_rows],
            "sleep_proposals": [dict(row) for row in sleep_proposal_rows],
            "version_count": int(version_count),
            "contradiction_count": int(contradiction_count),
            "unsupported_inference_ids": [str(row["id"]) for row in unsupported_inference_rows],
            "health_reviews": {
                "contradictions": [dict(row) for row in contradiction_rows],
                "unsupported_inferences": [dict(row) for row in unsupported_inference_rows],
            },
            "audit": self.audit(),
        }

    def audit(self) -> dict[str, Any]:
        with self._lock:
            duplicate_groups = self._conn.execute(
                "SELECT content_hash,COUNT(*) n FROM memories GROUP BY content_hash HAVING n>1"
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
        return {
            "ok": not (
                duplicate_groups
                or orphan_fts
                or orphan_features
                or missing_fts
                or missing_features
                or feature_stat_mismatches
                or orphan_feature_stats
                or invalid
                or stuck_usage
                or orphan_document_chunks
                or orphan_sleep_proposals
            ),
            "duplicate_groups": len(duplicate_groups),
            "orphan_fts_rows": orphan_fts,
            "orphan_feature_rows": orphan_features,
            "missing_fts_rows": missing_fts,
            "missing_feature_memories": missing_features,
            "feature_stat_mismatches": feature_stat_mismatches,
            "orphan_feature_stats": orphan_feature_stats,
            "invalid_score_rows": invalid,
            "unsupported_inferences": unsupported,
            "stuck_pending_usage": stuck_usage,
            "orphan_document_chunks": orphan_document_chunks,
            "orphan_sleep_proposals": orphan_sleep_proposals,
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
            if row.get("subject") and row.get("predicate") and row.get("object_value") is not None:
                key = (
                    "claim",
                    str(row["kind"]),
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
                key = ("text", str(row["kind"]), *((anchors[:1] or concepts or tokens[:2])))
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
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM memories WHERE pinned=0 AND protected=0 AND state IN ('active','cold')
                   AND importance<0.85 AND kind NOT IN ('identity','preference','prospective')"""
            ).fetchall()
        for row in rows:
            reference = row["last_used_at"] or row["last_injected_at"] or row["updated_at"]
            try:
                age = (now - datetime.fromisoformat(reference)).total_seconds() / 86400
            except (TypeError, ValueError):
                continue
            retention = _retention_score(dict(row))
            retention_scores[str(row["id"])] = round(retention, 6)
            cold_threshold = cold_after_days * (0.55 + retention)
            archive_threshold = archive_after_days * (0.65 + retention)
            if row["state"] == "active" and age >= cold_threshold and retention < 0.78:
                candidates["cold"].append(row["id"])
            elif row["state"] == "cold" and age >= archive_threshold and retention < 0.58:
                candidates["archived"].append(row["id"])
        if not dry_run:
            for state, ids in candidates.items():
                for memory_id in ids:
                    self.set_state(
                        memory_id,
                        state,
                        reason="utility and age lifecycle maintenance",
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
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _clamp(value: float) -> float:
    if not math.isfinite(float(value)):
        raise ValueError("score must be finite")
    return max(0.0, min(1.0, float(value)))


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


def _retention_score(memory: dict[str, Any]) -> float:
    used = int(memory.get("used_count", 0))
    positive = (
        int(memory.get("success_count", 0))
        + int(memory.get("confirmed_count", 0))
        + int(memory.get("validated_count", 0))
        + int(memory.get("helpful_count", 0))
    )
    harmful = int(memory.get("harmful_count", 0)) + int(memory.get("false_positive_count", 0))
    utility = max(0.0, min(1.0, (positive + 1.0) / (used + 2.0) - harmful / max(3.0, used + 2.0)))
    frequency = min(1.0, math.log1p(used + positive) / math.log(12.0))
    score = (
        0.24 * float(memory.get("importance", 0.5))
        + 0.18 * float(memory.get("confidence", 0.6)) * float(memory.get("trust", 0.7))
        + 0.12 * float(memory.get("currentness_confidence", 0.7))
        + 0.15 * float(memory.get("uniqueness", 1.0))
        + 0.22 * utility
        + 0.09 * frequency
    )
    if int(memory.get("duplicate_count", 0)) > 0:
        score -= min(0.12, 0.03 * int(memory["duplicate_count"]))
    return _clamp(score)


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
