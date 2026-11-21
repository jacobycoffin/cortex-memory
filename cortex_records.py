"""Memory records and version history behind the CortexStore facade.

Stage 2 of docs/STORE_SPLIT_PLAN.md. Every function receives the owning store
and uses its existing connection, lock, and transaction/savepoint boundaries.
No connection is opened here. Shared store helpers are imported at call time
so the public facade can re-export and delegate without a circular import.
"""

from __future__ import annotations

import uuid
from typing import Any, Sequence

from .security import normalize_text, neutralize_role_tags, sanitize_memory


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
    from .store import (
        _clamp,
        _json_string_list,
        _memory_metadata_completeness,
        _normalize_context_list,
        _normalize_context_map,
        _normalize_context_mode,
        _trace_json,
        content_hash,
        utc_now,
    )

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
    # Provenance fields are prompt-visible metadata, so they get the same
    # care as content: redact secrets on the way in, and keep role-tag
    # markers out of the stored labels (kind, source_type, source_ref) so
    # none of them can impersonate a chat turn beside recalled content.
    kind = normalize_text(neutralize_role_tags(str(kind or "semantic"))).casefold()[:80] or "semantic"
    source_ref = normalize_text(neutralize_role_tags(sanitize_memory(str(source_ref or "")).text))[:500] or None
    source_type = (
        normalize_text(neutralize_role_tags(sanitize_memory(str(source_type or "")).text))[:120]
        or "conversation"
    )
    subject = sanitize_memory(str(subject or "")).text[:500] or None
    predicate = sanitize_memory(str(predicate or "")).text[:500] or None
    object_value = sanitize_memory(str(object_value or "")).text[:1000] or None
    digest = content_hash(content)
    now = utc_now()
    state = "quarantine" if quarantine_reason else state
    context_mode_value = _normalize_context_mode(context_mode)
    scope_value = _normalize_context_map(scope)
    entity_values = _normalize_context_list(entities)
    precondition_values = _normalize_context_map(preconditions)
    system_values = _normalize_context_list(applicable_systems)
    version_values = _normalize_context_list(applicable_versions)
    source_context_value = normalize_text(sanitize_memory(str(source_context or "")).text)[:1000] or None
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
    from .store import _clamp, content_hash, utc_now

    sanitized_content = sanitize_memory(str(new_content or ""))
    new_content = normalize_text(sanitized_content.text)
    if not new_content:
        raise ValueError("corrected content cannot be empty")
    # Ledger text is durable and exportable, so it crosses the same secret
    # gate as content: a user may name the secret they are correcting.
    reason = normalize_text(sanitize_memory(str(reason or "")).text)[:500]
    # A correction is a write, so it gets the same treatment as add_memory:
    # secrets never reach storage through this path, and a correction that
    # carried a secret or an injection marker lands in quarantine instead
    # of ordinary recall.
    correction_quarantine = (
        ", ".join(
            dict.fromkeys(
                part
                for part in (
                    normalize_text(sanitized_content.quarantine_reason or ""),
                    (
                        "secret value removed; save only a reference to an approved secret manager"
                        if sanitized_content.redacted
                        else ""
                    ),
                )
                if part
            )
        )[:500]
        or None
    )
    correction_state = "quarantine" if correction_quarantine else "active"
    source_ref = normalize_text(neutralize_role_tags(sanitize_memory(str(source_ref or "")).text))[:500] or None
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
                correction_state,
                current["valid_from"],
                current["valid_to"],
                now,
                reason,
                source_ref,
            ),
        )
        conn.execute(
            """UPDATE memories SET content=?, content_hash=?, confidence=?, state=?,
                   updated_at=?, correction_count=correction_count+1,
                   quarantine_reason=?, protected=1 WHERE id=?""",
            (
                new_content,
                content_hash(new_content),
                next_confidence,
                correction_state,
                now,
                correction_quarantine,
                memory_id,
            ),
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
    from .store import utc_now

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
    from .store import utc_now

    with self.transaction() as conn:
        result = conn.execute(
            "UPDATE memories SET pinned=?, updated_at=? WHERE id=?",
            (int(pinned), utc_now(), memory_id),
        )
        return result.rowcount > 0


def set_uniqueness(self, memory_id: str, uniqueness: float) -> bool:
    from .store import _clamp

    with self.transaction() as conn:
        result = conn.execute("UPDATE memories SET uniqueness=? WHERE id=?", (_clamp(uniqueness), memory_id))
        return result.rowcount > 0


def find_by_content(self, content: str) -> dict[str, Any] | None:
    from .store import _decode_memory_metadata, content_hash

    with self._lock:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE content_hash=? ORDER BY created_at LIMIT 1",
            (content_hash(content),),
        ).fetchone()
    return _decode_memory_metadata(row) if row else None


def get_memory(self, memory_id: str) -> dict[str, Any] | None:
    from .store import _decode_memory_metadata

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
    from .store import _decode_memory_metadata

    if not memory_ids:
        return []
    placeholders = ",".join("?" for _ in memory_ids)
    with self._lock:
        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", tuple(memory_ids)
        ).fetchall()
    by_id = {row["id"]: _decode_memory_metadata(row) for row in rows}
    return [by_id[mid] for mid in memory_ids if mid in by_id]


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
