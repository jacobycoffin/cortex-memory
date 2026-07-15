"""Operator-controlled experiments and longitudinal learning evidence.

This module deliberately keeps causal evidence separate from observational
telemetry.  Randomized assignments are recorded before recall, while accuracy
uses only explicit dashboard labels.  Inferred conversational feedback remains
visible as diagnostic evidence but never unlocks a causal claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Sequence

from .security import normalize_text
from .store import CortexStore, utc_now


RECALL_EXPERIMENT_KEY = "adaptive-vs-fixed-vs-no-memory-v1"
RECALL_CONDITIONS = ("adaptive", "fixed", "no_memory")
POSITIVE_OUTCOMES = {"helpful", "validated"}
NEGATIVE_OUTCOMES = {"harmful", "corrected"}


def _query_hash(query: str) -> str:
    return hashlib.sha256(normalize_text(query).casefold().encode("utf-8")).hexdigest()


def set_recall_experiment(store: CortexStore, *, active: bool) -> dict[str, Any]:
    """Start a fresh controlled recall experiment or stop the active one."""

    now = utc_now()
    with store.transaction() as conn:
        current = conn.execute(
            """SELECT * FROM controlled_experiments
               WHERE experiment_key=? AND status='active' ORDER BY started_at DESC LIMIT 1""",
            (RECALL_EXPERIMENT_KEY,),
        ).fetchone()
        if not active:
            if current:
                conn.execute(
                    "UPDATE controlled_experiments SET status='stopped',stopped_at=? WHERE experiment_id=?",
                    (now, current["experiment_id"]),
                )
            return {"changed": bool(current), "active": False}
        if current:
            return {"changed": False, "active": True, "experiment_id": str(current["experiment_id"])}
        experiment_id = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO controlled_experiments(
                 experiment_id,experiment_key,name,status,conditions_json,assignment_rule,
                 primary_metric,minimum_labeled_per_condition,started_at,created_at
               ) VALUES(?,?,?,'active',?,?,?,?,?,?)""",
            (
                experiment_id,
                RECALL_EXPERIMENT_KEY,
                "Controlled recall policy comparison",
                json.dumps(RECALL_CONDITIONS),
                "balanced randomized blocks within task type; assignment occurs before recall",
                "explicit helpful-or-validated rate per assigned task",
                8,
                now,
                now,
            ),
        )
    return {"changed": True, "active": True, "experiment_id": experiment_id}


def assign_recall_condition(
    store: CortexStore,
    *,
    session_id: str,
    query: str,
    task_type: str,
) -> dict[str, Any] | None:
    """Assign one task using a balanced randomized block inside its task type."""

    with store.transaction() as conn:
        experiment = conn.execute(
            """SELECT * FROM controlled_experiments
               WHERE experiment_key=? AND status='active' ORDER BY started_at DESC LIMIT 1""",
            (RECALL_EXPERIMENT_KEY,),
        ).fetchone()
        if not experiment:
            return None
        rows = conn.execute(
            """SELECT condition,COUNT(*) count FROM recall_experiment_assignments
               WHERE experiment_id=? AND task_type=? GROUP BY condition""",
            (experiment["experiment_id"], task_type),
        ).fetchall()
        counts = {condition: 0 for condition in RECALL_CONDITIONS}
        counts.update({str(row["condition"]): int(row["count"]) for row in rows})
        minimum = min(counts.values())
        eligible = [condition for condition in RECALL_CONDITIONS if counts[condition] == minimum]
        salt = f"{experiment['experiment_id']}\0{session_id}\0{_query_hash(query)}\0{sum(counts.values())}"
        digest = hashlib.sha256(salt.encode("utf-8")).hexdigest()
        bucket = int(digest[:12], 16)
        condition = eligible[bucket % len(eligible)]
        assignment_id = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO recall_experiment_assignments(
                 assignment_id,experiment_id,session_id,query_hash,task_type,condition,
                 random_bucket,created_at
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (
                assignment_id,
                experiment["experiment_id"],
                session_id,
                _query_hash(query),
                task_type,
                condition,
                bucket,
                utc_now(),
            ),
        )
    return {
        "assignment_id": assignment_id,
        "experiment_id": str(experiment["experiment_id"]),
        "condition": condition,
        "random_bucket": bucket,
    }


def record_agent_task_start(
    store: CortexStore,
    *,
    task_id: str,
    session_id: str,
    task_type: str,
    query: str,
    recall_condition: str,
    recall_mode: str,
    memory_count: int,
    context_tokens: int,
    prepare_ms: float,
    assignment_id: str | None = None,
) -> None:
    now = utc_now()
    with store.transaction() as conn:
        conn.execute(
            """INSERT INTO agent_task_observations(
                 task_id,session_id,task_type,query_hash,query_preview,recall_condition,
                 recall_mode,memory_count,context_tokens,prepare_ms,experiment_assignment_id,started_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(task_id) DO UPDATE SET
                 recall_condition=excluded.recall_condition,recall_mode=excluded.recall_mode,
                 memory_count=excluded.memory_count,context_tokens=excluded.context_tokens,
                 prepare_ms=excluded.prepare_ms,
                 experiment_assignment_id=COALESCE(excluded.experiment_assignment_id,agent_task_observations.experiment_assignment_id)""",
            (
                task_id,
                session_id,
                task_type,
                _query_hash(query),
                normalize_text(query)[:240],
                recall_condition,
                recall_mode,
                max(0, int(memory_count)),
                max(0, int(context_tokens)),
                max(0.0, float(prepare_ms)),
                assignment_id,
                now,
            ),
        )
        if assignment_id:
            conn.execute(
                "UPDATE recall_experiment_assignments SET task_id=? WHERE assignment_id=?",
                (task_id, assignment_id),
            )


def complete_agent_tasks(
    store: CortexStore,
    task_ids: Sequence[str],
    *,
    response_ms: float,
    tool_calls: int,
    tool_successes: int,
) -> None:
    ids = list(dict.fromkeys(str(task_id) for task_id in task_ids if task_id))
    if not ids:
        return
    placeholders = ",".join("?" for _ in ids)
    with store.transaction() as conn:
        conn.execute(
            f"""UPDATE agent_task_observations SET response_ms=?,tool_calls=?,tool_successes=?,completed_at=?
                WHERE task_id IN ({placeholders})""",
            (
                max(0.0, float(response_ms)),
                max(0, int(tool_calls)),
                max(0, int(tool_successes)),
                utc_now(),
                *ids,
            ),
        )


def sync_task_outcome_tx(
    conn: Any,
    task_id: str,
    outcome: str,
    *,
    source: str | None,
    resolved_at: str,
) -> None:
    """Keep task, randomized assignment, and reconsolidation evidence aligned."""

    correction = int(outcome == "corrected")
    conn.execute(
        """UPDATE agent_task_observations
           SET outcome=?,outcome_source=?,correction_detected=MAX(correction_detected,?),resolved_at=?
           WHERE task_id=?""",
        (outcome, source, correction, resolved_at, task_id),
    )
    conn.execute(
        """UPDATE recall_experiment_assignments
           SET outcome=?,outcome_source=?,resolved_at=? WHERE task_id=?""",
        (outcome, source, resolved_at, task_id),
    )
    memory_rows = conn.execute(
        "SELECT memory_id FROM usage_records WHERE task_id=? AND used=1",
        (task_id,),
    ).fetchall()
    for row in memory_rows:
        memory_id = str(row["memory_id"])
        if outcome in POSITIVE_OUTCOMES:
            conn.execute(
                """UPDATE reconsolidation_events SET status='stabilized',outcome=?,stabilized_at=?
                   WHERE memory_id=? AND status IN ('pending','reexposed')""",
                (outcome, resolved_at, memory_id),
            )
        elif outcome in NEGATIVE_OUTCOMES:
            conn.execute(
                """UPDATE reconsolidation_events SET status='failed',outcome=?,failed_at=?
                   WHERE memory_id=? AND status IN ('pending','reexposed')""",
                (outcome, resolved_at, memory_id),
            )


def record_reconsolidation_correction_tx(
    conn: Any,
    *,
    memory_id: str,
    prior_version_id: int | None,
    new_version_id: int | None,
    trigger_access_at: str | None,
    corrected_at: str,
    source_ref: str | None,
) -> None:
    conn.execute(
        """INSERT INTO reconsolidation_events(
             event_id,memory_id,prior_version_id,new_version_id,trigger_access_at,
             corrected_at,status,source_ref
           ) VALUES(?,?,?,?,?,?,'pending',?)""",
        (
            str(uuid.uuid4()),
            memory_id,
            prior_version_id,
            new_version_id,
            trigger_access_at,
            corrected_at,
            source_ref,
        ),
    )


def record_reconsolidation_reuse_tx(conn: Any, memory_id: str, at: str) -> None:
    conn.execute(
        """UPDATE reconsolidation_events
           SET first_reused_at=COALESCE(first_reused_at,?),status=CASE WHEN status='pending' THEN 'reexposed' ELSE status END
           WHERE memory_id=? AND corrected_at<=? AND status IN ('pending','reexposed')""",
        (at, memory_id, at),
    )


def generate_summary_candidates(store: CortexStore, *, limit: int = 6) -> dict[str, Any]:
    """Create extractive, source-cited candidates without making recallable memory."""

    bounded = max(1, min(int(limit), 20))
    with store._lock:
        rows = store._conn.execute(
            """SELECT e.src_id,e.dst_id,e.relation,e.weight,e.evidence_count,
                      s.content src_content,s.kind src_kind,s.subject src_subject,
                      s.predicate src_predicate,s.object_value src_object,
                      d.content dst_content,d.kind dst_kind,d.subject dst_subject,
                      d.predicate dst_predicate,d.object_value dst_object
               FROM edges e
               JOIN memories s ON s.id=e.src_id JOIN memories d ON d.id=e.dst_id
               WHERE s.state IN ('active','cold') AND d.state IN ('active','cold')
                 AND e.relation NOT IN ('contradicts','supersedes','consolidates')
                 AND (e.relation IN ('derived_from','supports','sleep_replay','co_used')
                      OR e.evidence_count>=2 OR e.weight>=0.30)
               ORDER BY e.evidence_count DESC,e.weight DESC,e.last_reinforced_at DESC LIMIT 200"""
        ).fetchall()
    created = 0
    skipped = 0
    for row in rows:
        if created >= bounded:
            break
        source_ids = sorted((str(row["src_id"]), str(row["dst_id"])))
        signature = hashlib.sha256("\0".join(source_ids).encode("utf-8")).hexdigest()
        left = _claim_from_row(row, "src")
        right = _claim_from_row(row, "dst")
        if left.casefold() == right.casefold():
            skipped += 1
            continue
        shared_subject = str(row["src_subject"] or row["dst_subject"] or "").strip()
        title = (
            f"Evidence bundle · {shared_subject[:80]}"
            if shared_subject
            else f"Evidence bundle · {str(row['relation']).replace('_', ' ')}"
        )
        claims = ((left, str(row["src_id"]), str(row["src_content"])), (right, str(row["dst_id"]), str(row["dst_content"])))
        summary_text = " ".join(f"{claim} [M:{memory_id[:8]}]" for claim, memory_id, _ in claims)
        candidate_id = str(uuid.uuid4())
        with store.transaction() as conn:
            result = conn.execute(
                """INSERT OR IGNORE INTO summary_candidates(
                     candidate_id,source_signature,title,summary_text,status,created_at
                   ) VALUES(?,?,?,?,'proposed',?)""",
                (candidate_id, signature, title, summary_text, utc_now()),
            )
            if not result.rowcount:
                skipped += 1
                continue
            for ordinal, (claim, memory_id, excerpt) in enumerate(claims, 1):
                conn.execute(
                    "INSERT INTO summary_candidate_claims(candidate_id,ordinal,claim_text) VALUES(?,?,?)",
                    (candidate_id, ordinal, claim),
                )
                conn.execute(
                    """INSERT INTO summary_candidate_sources(
                         candidate_id,claim_ordinal,memory_id,excerpt
                       ) VALUES(?,?,?,?)""",
                    (candidate_id, ordinal, memory_id, normalize_text(excerpt)[:320]),
                )
        created += 1
    return {"created": created, "skipped": skipped}


def review_summary_candidate(
    store: CortexStore,
    candidate_id: str,
    *,
    action: str,
    actor: str,
) -> dict[str, Any]:
    if action not in {"approve", "reject"}:
        raise ValueError("summary action must be approve or reject")
    with store._lock:
        candidate = store._conn.execute(
            "SELECT * FROM summary_candidates WHERE candidate_id=?",
            (candidate_id,),
        ).fetchone()
        sources = store._conn.execute(
            """SELECT s.memory_id,c.claim_text FROM summary_candidate_sources s
               JOIN summary_candidate_claims c
                 ON c.candidate_id=s.candidate_id AND c.ordinal=s.claim_ordinal
               JOIN memories m ON m.id=s.memory_id
               WHERE s.candidate_id=? AND m.state IN ('active','cold')
               ORDER BY s.claim_ordinal""",
            (candidate_id,),
        ).fetchall()
    if not candidate or str(candidate["status"]) != "proposed":
        raise ValueError("this summary candidate is no longer awaiting review")
    now = utc_now()
    if action == "reject":
        with store.transaction() as conn:
            conn.execute(
                """UPDATE summary_candidates SET status='rejected',reviewed_at=?,reviewed_by=?
                   WHERE candidate_id=? AND status='proposed'""",
                (now, normalize_text(actor)[:80], candidate_id),
            )
        return {"candidate_id": candidate_id, "status": "rejected"}
    if len({str(row["memory_id"]) for row in sources}) < 2:
        raise ValueError("approval requires at least two active cited source memories")
    memory_id, _ = store.add_memory(
        str(candidate["summary_text"]),
        kind="semantic",
        source_type="summary_approval",
        source_category="USER_EXPLICIT",
        source_ref=candidate_id,
        confidence=0.9,
        currentness_confidence=0.85,
        importance=0.75,
        trust=0.92,
        protected=True,
        extraction_method="operator_approved_summary_v1",
        evidence_ids=[str(row["memory_id"]) for row in sources],
    )
    with store.transaction() as conn:
        conn.execute(
            """UPDATE summary_candidates SET status='approved',approved_memory_id=?,reviewed_at=?,reviewed_by=?
               WHERE candidate_id=? AND status='proposed'""",
            (memory_id, now, normalize_text(actor)[:80], candidate_id),
        )
    return {"candidate_id": candidate_id, "status": "approved", "memory_id": memory_id}


def create_prospective_item(
    store: CortexStore,
    *,
    content: str,
    due_at: str | None,
    actor: str,
) -> dict[str, Any]:
    clean = normalize_text(content)
    if not clean:
        raise ValueError("prospective memory content is required")
    due = _validate_optional_iso(due_at)
    memory_id, created = store.add_memory(
        clean,
        kind="prospective",
        source_type="dashboard",
        source_category="USER_EXPLICIT",
        source_ref=f"prospective:{normalize_text(actor)[:80]}",
        confidence=0.95,
        importance=0.85,
        trust=0.95,
        pinned=True,
        protected=True,
        valid_from=due,
        extraction_method="dashboard_prospective_v1",
    )
    now = utc_now()
    with store.transaction() as conn:
        conn.execute(
            """INSERT INTO prospective_items(memory_id,status,due_at,created_at,updated_at)
               VALUES(?,'open',?,?,?)
               ON CONFLICT(memory_id) DO UPDATE SET due_at=excluded.due_at,updated_at=excluded.updated_at""",
            (memory_id, due, now, now),
        )
    return {"memory_id": memory_id, "created": created, "status": "open", "due_at": due}


def update_prospective_item(
    store: CortexStore,
    memory_id: str,
    *,
    status: str,
    due_at: str | None = None,
) -> dict[str, Any]:
    if status not in {"open", "completed", "abandoned"}:
        raise ValueError("prospective status must be open, completed, or abandoned")
    due = _validate_optional_iso(due_at)
    now = utc_now()
    with store.transaction() as conn:
        memory = conn.execute(
            "SELECT id,kind,valid_from,valid_to FROM memories WHERE id=?",
            (memory_id,),
        ).fetchone()
        if not memory or str(memory["kind"]) != "prospective":
            raise ValueError("prospective memory not found")
        existing = conn.execute("SELECT * FROM prospective_items WHERE memory_id=?", (memory_id,)).fetchone()
        next_due = due if due_at is not None else (existing["due_at"] if existing else memory["valid_from"] or memory["valid_to"])
        conn.execute(
            """INSERT INTO prospective_items(
                 memory_id,status,due_at,completed_at,abandoned_at,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(memory_id) DO UPDATE SET status=excluded.status,due_at=excluded.due_at,
                 completed_at=excluded.completed_at,abandoned_at=excluded.abandoned_at,
                 updated_at=excluded.updated_at""",
            (
                memory_id,
                status,
                next_due,
                now if status == "completed" else None,
                now if status == "abandoned" else None,
                now,
                now,
            ),
        )
        conn.execute(
            "UPDATE memories SET valid_from=?,updated_at=? WHERE id=?",
            (next_due, now, memory_id),
        )
    store.set_state(
        memory_id,
        "active" if status == "open" else "archived",
        reason=f"prospective memory marked {status}",
    )
    return {"memory_id": memory_id, "status": status, "due_at": next_due}


def start_sleep_apply_trial(store: CortexStore, *, pair_limit: int = 8) -> dict[str, Any]:
    """Randomize matched association proposals and expose treatment only."""

    bounded_pairs = max(1, min(int(pair_limit), 20))
    with store._lock:
        rows = store._conn.execute(
            """SELECT p.* FROM sleep_proposals p
               WHERE p.status='proposed' AND p.kind IN ('association','association_reinforcement')
                 AND p.src_id IS NOT NULL AND p.dst_id IS NOT NULL
                 AND NOT EXISTS(SELECT 1 FROM sleep_trial_items i WHERE i.proposal_id=p.proposal_id)
               ORDER BY p.kind,p.evidence_count DESC,p.score DESC,p.created_at DESC LIMIT 100"""
        ).fetchall()
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["kind"]), int(row["evidence_count"] or 0))].append(dict(row))
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for items in groups.values():
        for index in range(0, len(items) - 1, 2):
            pairs.append((items[index], items[index + 1]))
            if len(pairs) >= bounded_pairs:
                break
        if len(pairs) >= bounded_pairs:
            break
    if not pairs:
        raise ValueError("run shadow Sleep until at least two matched association proposals are available")
    trial_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    now = utc_now()
    assignments: list[tuple[dict[str, Any], str, str]] = []
    for index, pair in enumerate(pairs):
        pair_key = f"{trial_id[:8]}-{index + 1:02d}"
        randomized = int(hashlib.sha256(pair_key.encode("utf-8")).hexdigest()[:8], 16) % 2
        assignments.extend(
            [
                (pair[randomized], "treatment", pair_key),
                (pair[1 - randomized], "control", pair_key),
            ]
        )
    with store.transaction() as conn:
        conn.execute(
            """INSERT INTO sleep_runs(
                 run_id,mode,status,cutoff_at,association_proposals,applied_changes,
                 reflection_status,report_json,started_at,completed_at
               ) VALUES(?,'apply','completed',?,?,?,?,?,?,?)""",
            (
                run_id,
                now,
                len(pairs),
                len(pairs),
                "disabled",
                json.dumps({"controlled_sleep_trial": trial_id, "matched_pairs": len(pairs)}),
                now,
                now,
            ),
        )
        conn.execute(
            """INSERT INTO sleep_trials(
                 trial_id,sleep_run_id,status,assignment_rule,primary_metric,pair_count,
                 minimum_labeled_per_arm,started_at
               ) VALUES(?,?,'collecting','randomized within kind and evidence-count matched pairs',?,?,8,?)""",
            (
                trial_id,
                run_id,
                "explicit helpful-or-validated future tasks touching proposal memories",
                len(pairs),
                now,
            ),
        )
        for proposal, assignment, pair_key in assignments:
            conn.execute(
                """INSERT INTO sleep_trial_items(
                     item_id,trial_id,pair_key,proposal_id,assignment,exposure,src_id,dst_id,score,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    str(uuid.uuid4()),
                    trial_id,
                    pair_key,
                    proposal["proposal_id"],
                    assignment,
                    "applied" if assignment == "treatment" else "withheld",
                    proposal["src_id"],
                    proposal["dst_id"],
                    float(proposal["score"] or 0.0),
                    now,
                ),
            )
    from .sleep import _apply_association  # Local import avoids a module cycle.

    try:
        for proposal, assignment, _pair_key in assignments:
            if assignment != "treatment":
                continue
            _apply_association(
                store,
                run_id,
                str(proposal["proposal_id"]),
                str(proposal["src_id"]),
                str(proposal["dst_id"]),
                int(proposal["evidence_count"] or 0),
            )
    except Exception:
        with store.transaction() as conn:
            conn.execute("UPDATE sleep_trials SET status='failed',completed_at=? WHERE trial_id=?", (utc_now(), trial_id))
            conn.execute("UPDATE sleep_runs SET status='failed' WHERE run_id=?", (run_id,))
        raise
    return {"trial_id": trial_id, "sleep_run_id": run_id, "pair_count": len(pairs), "status": "collecting"}


def undo_sleep_apply_trial(store: CortexStore, trial_id: str) -> dict[str, Any]:
    with store._lock:
        trial = store._conn.execute("SELECT * FROM sleep_trials WHERE trial_id=?", (trial_id,)).fetchone()
    if not trial or not trial["sleep_run_id"]:
        raise ValueError("sleep trial not found")
    from .sleep import undo_sleep

    result = undo_sleep(store, str(trial["sleep_run_id"]))
    with store.transaction() as conn:
        conn.execute(
            "UPDATE sleep_trials SET status=?,completed_at=? WHERE trial_id=?",
            ("revert_partial" if result.get("conflicts") else "reverted", utc_now(), trial_id),
        )
    return {"trial_id": trial_id, **result}


def research_snapshot(store: CortexStore) -> dict[str, Any]:
    """Return bounded source-backed datasets and plain metric definitions."""

    with store._lock:
        experiment_rows = store._conn.execute(
            "SELECT * FROM controlled_experiments ORDER BY created_at DESC LIMIT 20"
        ).fetchall()
        assignment_rows = store._conn.execute(
            """SELECT a.*,t.memory_count,t.context_tokens,t.prepare_ms,t.response_ms,
                      t.tool_calls,t.tool_successes,t.query_preview,t.started_at task_started_at,
                      t.completed_at task_completed_at
               FROM recall_experiment_assignments a
               LEFT JOIN agent_task_observations t ON t.task_id=a.task_id
               ORDER BY a.created_at DESC LIMIT 500"""
        ).fetchall()
        task_rows = store._conn.execute(
            """SELECT t.*,
                      COUNT(DISTINCT u.memory_id) attributed_memories,
                      GROUP_CONCAT(DISTINCT m.source_category) source_categories,
                      AVG(julianday(t.started_at)-julianday(m.observed_at)) avg_memory_age_days,
                      (SELECT COUNT(*) FROM memories cap WHERE cap.created_at<=t.started_at) stored_capacity
               FROM agent_task_observations t
               LEFT JOIN usage_records u ON u.task_id=t.task_id AND u.used=1
               LEFT JOIN memories m ON m.id=u.memory_id
               GROUP BY t.task_id ORDER BY t.started_at DESC LIMIT 1000"""
        ).fetchall()
        summary_rows = store._conn.execute(
            "SELECT * FROM summary_candidates ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
        summary_source_rows = store._conn.execute(
            """SELECT s.candidate_id,s.claim_ordinal,s.memory_id,s.excerpt,c.claim_text,
                      m.kind,m.source_category,m.source_ref,m.state
               FROM summary_candidate_sources s
               JOIN summary_candidate_claims c
                 ON c.candidate_id=s.candidate_id AND c.ordinal=s.claim_ordinal
               JOIN memories m ON m.id=s.memory_id
               ORDER BY s.candidate_id,s.claim_ordinal LIMIT 500"""
        ).fetchall()
        prospective_rows = store._conn.execute(
            """SELECT m.id memory_id,m.content,m.state memory_state,m.valid_from,m.valid_to,
                      COALESCE(p.status,'open') status,COALESCE(p.due_at,m.valid_from,m.valid_to) due_at,
                      p.completed_at,p.abandoned_at,p.note,COALESCE(p.updated_at,m.updated_at) updated_at
               FROM memories m LEFT JOIN prospective_items p ON p.memory_id=m.id
               WHERE m.kind='prospective' AND m.state<>'tombstoned'
               ORDER BY CASE WHEN COALESCE(p.status,'open')='open' THEN 0 ELSE 1 END,
                        COALESCE(p.due_at,m.valid_from,m.valid_to,'9999') LIMIT 500"""
        ).fetchall()
        reconsolidation_rows = store._conn.execute(
            """SELECT r.*,m.content,m.kind,m.source_category,
                      pv.content prior_content,nv.content corrected_content
               FROM reconsolidation_events r JOIN memories m ON m.id=r.memory_id
               LEFT JOIN memory_versions pv ON pv.version_id=r.prior_version_id
               LEFT JOIN memory_versions nv ON nv.version_id=r.new_version_id
               ORDER BY r.corrected_at DESC LIMIT 250"""
        ).fetchall()
        trial_rows = store._conn.execute(
            "SELECT * FROM sleep_trials ORDER BY started_at DESC LIMIT 50"
        ).fetchall()
        trial_item_rows = store._conn.execute(
            """SELECT i.*,
                      (SELECT COUNT(DISTINCT u.task_id) FROM usage_records u
                       JOIN task_outcome_labels l ON l.task_id=u.task_id AND l.active=1
                       WHERE u.used=1 AND u.memory_id IN (i.src_id,i.dst_id)
                         AND l.created_at>=tr.started_at) labeled_tasks,
                      (SELECT COUNT(DISTINCT u.task_id) FROM usage_records u
                       JOIN task_outcome_labels l ON l.task_id=u.task_id AND l.active=1
                       WHERE u.used=1 AND u.memory_id IN (i.src_id,i.dst_id)
                         AND l.outcome IN ('helpful','validated') AND l.created_at>=tr.started_at) positive_tasks,
                      (SELECT COUNT(DISTINCT u.task_id) FROM usage_records u
                       JOIN task_outcome_labels l ON l.task_id=u.task_id AND l.active=1
                       WHERE u.used=1 AND u.memory_id IN (i.src_id,i.dst_id)
                         AND l.outcome IN ('harmful','corrected') AND l.created_at>=tr.started_at) negative_tasks
               FROM sleep_trial_items i JOIN sleep_trials tr ON tr.trial_id=i.trial_id
               ORDER BY i.created_at DESC LIMIT 500"""
        ).fetchall()
        trial_arm_rows = store._conn.execute(
            """SELECT i.trial_id,i.assignment,COUNT(DISTINCT i.item_id) items,
                      COUNT(DISTINCT CASE WHEN l.label_id IS NOT NULL THEN u.task_id END) labeled_tasks,
                      COUNT(DISTINCT CASE WHEN l.outcome IN ('helpful','validated') THEN u.task_id END) positive_tasks,
                      COUNT(DISTINCT CASE WHEN l.outcome IN ('harmful','corrected') THEN u.task_id END) negative_tasks
               FROM sleep_trial_items i
               JOIN sleep_trials tr ON tr.trial_id=i.trial_id
               LEFT JOIN usage_records u ON u.used=1 AND u.memory_id IN (i.src_id,i.dst_id)
               LEFT JOIN task_outcome_labels l ON l.task_id=u.task_id AND l.active=1
                 AND l.created_at>=tr.started_at
               GROUP BY i.trial_id,i.assignment"""
        ).fetchall()

    experiments = [dict(row) for row in experiment_rows]
    assignments = [dict(row) for row in assignment_rows]
    active = next((row for row in experiments if row["status"] == "active"), None)
    focus_experiment = active or (experiments[0] if experiments else None)
    active_id = str(focus_experiment["experiment_id"]) if focus_experiment else None
    condition_stats = []
    for condition in RECALL_CONDITIONS:
        rows = [row for row in assignments if row["experiment_id"] == active_id and row["condition"] == condition]
        explicit = [row for row in rows if row["outcome_source"] == "dashboard" and row["outcome"] in POSITIVE_OUTCOMES | NEGATIVE_OUTCOMES]
        positive = sum(row["outcome"] in POSITIVE_OUTCOMES for row in explicit)
        rate = positive / len(explicit) if explicit else None
        low, high = _wilson_interval(positive, len(explicit))
        condition_stats.append(
            {
                "condition": condition,
                "assignments": len(rows),
                "explicit_labels": len(explicit),
                "positive": positive,
                "accuracy": rate,
                "ci_low": low,
                "ci_high": high,
                "avg_context_tokens": _average(row.get("context_tokens") for row in rows),
                "avg_prepare_ms": _average(row.get("prepare_ms") for row in rows),
                "avg_response_ms": _average(row.get("response_ms") for row in rows),
            }
        )
    minimum = int(focus_experiment["minimum_labeled_per_condition"] if focus_experiment else 8)
    causal_ready = bool(focus_experiment) and all(int(row["explicit_labels"]) >= minimum for row in condition_stats)

    tasks = [dict(row) for row in task_rows]
    daily: dict[str, dict[str, Any]] = {}
    for task in reversed(tasks):
        day = str(task["started_at"] or "")[:10]
        if not day:
            continue
        item = daily.setdefault(
            day,
            {
                "day": day,
                "tasks": 0,
                "completed": 0,
                "explicit_labels": 0,
                "positive": 0,
                "negative": 0,
                "context_tokens_total": 0.0,
                "prepare_ms_total": 0.0,
                "response_ms_total": 0.0,
                "response_samples": 0,
                "tool_calls": 0,
                "tool_successes": 0,
                "corrections": 0,
            },
        )
        item["tasks"] += 1
        item["completed"] += int(bool(task.get("completed_at")))
        item["context_tokens_total"] += float(task.get("context_tokens") or 0)
        item["prepare_ms_total"] += float(task.get("prepare_ms") or 0)
        if task.get("response_ms") is not None:
            item["response_ms_total"] += float(task["response_ms"])
            item["response_samples"] += 1
        item["tool_calls"] += int(task.get("tool_calls") or 0)
        item["tool_successes"] += int(task.get("tool_successes") or 0)
        item["corrections"] += int(task.get("correction_detected") or 0)
        if task.get("outcome_source") == "dashboard" and task.get("outcome") in POSITIVE_OUTCOMES | NEGATIVE_OUTCOMES:
            item["explicit_labels"] += 1
            if task["outcome"] in POSITIVE_OUTCOMES:
                item["positive"] += 1
            else:
                item["negative"] += 1
    daily_rows = []
    for item in daily.values():
        task_count = max(1, item["tasks"])
        labels = item["explicit_labels"]
        calls = item["tool_calls"]
        daily_rows.append(
            {
                "day": item["day"],
                "tasks": item["tasks"],
                "completion_rate": item["completed"] / task_count,
                "explicit_labels": labels,
                "accuracy": item["positive"] / labels if labels else None,
                "positive": item["positive"],
                "negative": item["negative"],
                "avg_context_tokens": item["context_tokens_total"] / task_count,
                "avg_prepare_ms": item["prepare_ms_total"] / task_count,
                "avg_response_ms": item["response_ms_total"] / item["response_samples"] if item["response_samples"] else None,
                "tool_success_rate": item["tool_successes"] / calls if calls else None,
                "tool_calls": calls,
                "corrections": item["corrections"],
            }
        )

    failures = []
    for task in tasks:
        if not (
            task.get("outcome") in NEGATIVE_OUTCOMES
            or int(task.get("correction_detected") or 0)
            or int(task.get("tool_calls") or 0) > int(task.get("tool_successes") or 0)
        ):
            continue
        failures.append({**task, **_failure_cause(task)})
        if len(failures) >= 250:
            break

    sources_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_source_rows:
        sources_by_candidate[str(row["candidate_id"])].append(dict(row))
    summaries = []
    for row in summary_rows:
        item = dict(row)
        item["sources"] = sources_by_candidate.get(str(item["candidate_id"]), [])
        summaries.append(item)

    prospective = []
    now = datetime.now(timezone.utc)
    for row in prospective_rows:
        item = dict(row)
        item["display_status"] = item["status"]
        if item["status"] == "open" and item.get("due_at"):
            try:
                due = datetime.fromisoformat(str(item["due_at"]).replace("Z", "+00:00"))
                if due.tzinfo is None:
                    due = due.replace(tzinfo=timezone.utc)
                if due < now:
                    item["display_status"] = "overdue"
            except ValueError:
                pass
        prospective.append(item)

    trials = [dict(row) for row in trial_rows]
    trial_items = [dict(row) for row in trial_item_rows]
    trial_arms = [dict(row) for row in trial_arm_rows]
    for trial in trials:
        arms = []
        for assignment in ("treatment", "control"):
            arm = next(
                (
                    row for row in trial_arms
                    if row["trial_id"] == trial["trial_id"] and row["assignment"] == assignment
                ),
                {},
            )
            labeled = int(arm.get("labeled_tasks") or 0)
            positive = int(arm.get("positive_tasks") or 0)
            arms.append(
                {
                    "assignment": assignment,
                    "items": int(arm.get("items") or 0),
                    "labeled_tasks": labeled,
                    "positive_tasks": positive,
                    "negative_tasks": int(arm.get("negative_tasks") or 0),
                    "positive_rate": positive / labeled if labeled else None,
                }
            )
        trial["arms"] = arms
        trial["causal_ready"] = all(
            arm["labeled_tasks"] >= int(trial["minimum_labeled_per_arm"]) for arm in arms
        )

    explicit_tasks = [task for task in tasks if task.get("outcome_source") == "dashboard" and task.get("outcome") in POSITIVE_OUTCOMES | NEGATIVE_OUTCOMES]
    positive_tasks = sum(task["outcome"] in POSITIVE_OUTCOMES for task in explicit_tasks)
    recon = []
    for row in reconsolidation_rows:
        item = dict(row)
        item["time_to_reuse_hours"] = _hours_between(item.get("corrected_at"), item.get("first_reused_at"))
        item["time_to_stabilize_hours"] = _hours_between(item.get("corrected_at"), item.get("stabilized_at"))
        recon.append(item)
    return {
        "experiments": {
            "active": active,
            "latest": focus_experiment,
            "condition_stats": condition_stats,
            "recent_assignments": [row for row in assignments if row["experiment_id"] == active_id][:100],
            "causal_ready": causal_ready,
            "minimum_labeled_per_condition": minimum,
            "claim": (
                "Randomized comparison has enough explicit outcomes for an initial causal estimate."
                if causal_ready
                else "Collect explicit labels in every arm before interpreting condition differences causally."
            ),
        },
        "agent_evaluation": {
            "summary": {
                "tasks": len(tasks),
                "completed": sum(bool(task.get("completed_at")) for task in tasks),
                "explicit_labels": len(explicit_tasks),
                "accuracy": positive_tasks / len(explicit_tasks) if explicit_tasks else None,
                "tool_calls": sum(int(task.get("tool_calls") or 0) for task in tasks),
                "tool_successes": sum(int(task.get("tool_successes") or 0) for task in tasks),
                "corrections": sum(int(task.get("correction_detected") or 0) for task in tasks),
            },
            "by_day": daily_rows,
            "recent_tasks": tasks[:250],
            "definitions": {
                "accuracy": "Explicit helpful or validated labels divided by all explicit positive and negative labels.",
                "completion": "Turn sync completed after a recorded prefetch; this is instrumentation completion, not task correctness.",
                "tool_success": "Observed tool results without a detected error divided by observed tool results.",
                "latency": "Provider prefetch preparation plus wall time from prefetch instrumentation to turn sync; model and network time are not isolated.",
            },
        },
        "failures": failures,
        "summary_candidates": summaries,
        "prospective": prospective,
        "reconsolidation": {
            "events": recon,
            "counts": {
                state: sum(str(row.get("status")) == state for row in recon)
                for state in ("pending", "reexposed", "stabilized", "failed")
            },
            "after_recall": sum(bool(row.get("trigger_access_at")) for row in recon),
        },
        "sleep_trials": {"trials": trials, "items": trial_items},
    }


def _claim_from_row(row: Any, prefix: str) -> str:
    subject = normalize_text(str(row[f"{prefix}_subject"] or ""))
    predicate = normalize_text(str(row[f"{prefix}_predicate"] or "")).replace("_", " ")
    value = normalize_text(str(row[f"{prefix}_object"] or ""))
    if subject and predicate and value:
        return f"{subject} {predicate} {value}."[:320]
    content = normalize_text(str(row[f"{prefix}_content"] or ""))[:300]
    return content if content.endswith((".", "!", "?")) else f"{content}."


def _validate_optional_iso(value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    clean = str(value).strip()
    try:
        parsed = datetime.fromisoformat(clean.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("due_at must be an ISO date or timestamp") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds")


def _average(values: Sequence[Any] | Any) -> float | None:
    samples = [float(value) for value in values if value is not None]
    return sum(samples) / len(samples) if samples else None


def _wilson_interval(successes: int, samples: int) -> tuple[float | None, float | None]:
    if samples <= 0:
        return None, None
    z = 1.96
    p = successes / samples
    denominator = 1 + z * z / samples
    center = (p + z * z / (2 * samples)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * samples)) / samples) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _failure_cause(task: dict[str, Any]) -> dict[str, str]:
    tool_calls = int(task.get("tool_calls") or 0)
    tool_successes = int(task.get("tool_successes") or 0)
    age = float(task.get("avg_memory_age_days") or 0.0)
    sources = str(task.get("source_categories") or "")
    if tool_calls > tool_successes:
        return {
            "cause": "tool_execution",
            "cause_label": "A tool returned an error",
            "lever": "Inspect the failed tool and reinforced workflow evidence before changing recall capacity.",
        }
    if str(task.get("recall_condition")) == "no_memory":
        return {
            "cause": "no_memory_control",
            "cause_label": "No-memory control task",
            "lever": "Compare its explicit outcome with matched adaptive and fixed tasks; do not tune from one case.",
        }
    if int(task.get("memory_count") or 0) == 0:
        return {
            "cause": "recall_abstention",
            "cause_label": "Recall supplied no memory",
            "lever": "Review the task type and abstention reason before lowering the retrieval threshold.",
        }
    if age >= 90:
        return {
            "cause": "stale_evidence",
            "cause_label": "Older recalled evidence",
            "lever": "Inspect validity windows and currentness before increasing capacity or context budget.",
        }
    if "AGENT_INFERENCE" in sources or "REFLECTION" in sources:
        return {
            "cause": "weak_source",
            "cause_label": "Inference or reflection evidence",
            "lever": "Add cited source evidence or tighten metacognitive verification for this task type.",
        }
    return {
        "cause": "unclassified_negative",
        "cause_label": "Outcome needs review",
        "lever": "Open the task and its attributed memories; one negative label is diagnostic, not a causal conclusion.",
    }


def _hours_between(start: Any, end: Any) -> float | None:
    if not start or not end:
        return None
    try:
        first = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        second = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (second - first).total_seconds() / 3600.0)
