from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path

from tests import _bootstrap  # noqa: F401

from cortex import CortexMemoryProvider
from cortex.research import (
    assign_recall_condition,
    complete_agent_tasks,
    create_prospective_item,
    generate_summary_candidates,
    record_agent_task_start,
    research_snapshot,
    review_summary_candidate,
    set_recall_experiment,
    start_sleep_apply_trial,
    undo_sleep_apply_trial,
    update_prospective_item,
)
from cortex.store import CortexStore, utc_now


class ResearchLabTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "cortex.db"
        self.store = CortexStore(self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        self.temp.cleanup()

    def test_controlled_experiment_balances_and_uses_explicit_labels(self) -> None:
        started = set_recall_experiment(self.store, active=True)
        self.assertTrue(started["active"])
        assignments = []
        for index in range(24):
            assignment = assign_recall_condition(
                self.store,
                session_id="experiment-session",
                query=f"Complete controlled task {index}",
                task_type="general",
            )
            self.assertIsNotNone(assignment)
            task_id = str(uuid.uuid4())
            record_agent_task_start(
                self.store,
                task_id=task_id,
                session_id="experiment-session",
                task_type="general",
                query=f"Complete controlled task {index}",
                recall_condition=str(assignment["condition"]),
                recall_mode=str(assignment["condition"]),
                memory_count=0,
                context_tokens=0,
                prepare_ms=0.5,
                assignment_id=str(assignment["assignment_id"]),
            )
            complete_agent_tasks(
                self.store,
                [task_id],
                response_ms=20.0,
                tool_calls=0,
                tool_successes=0,
            )
            # No-memory tasks have no usage rows, but must remain labelable.
            self.store.label_task_outcome(task_id, "helpful", actor="test-operator")
            assignments.append(str(assignment["condition"]))

        self.assertEqual({condition: assignments.count(condition) for condition in set(assignments)}, {
            "adaptive": 8,
            "fixed": 8,
            "no_memory": 8,
        })
        snapshot = research_snapshot(self.store)
        self.assertTrue(snapshot["experiments"]["causal_ready"])
        self.assertTrue(all(row["accuracy"] == 1.0 for row in snapshot["experiments"]["condition_stats"]))
        self.assertEqual(snapshot["agent_evaluation"]["summary"]["explicit_labels"], 24)

    def test_provider_really_withholds_the_no_memory_condition(self) -> None:
        self.store.close()
        provider = CortexMemoryProvider(
            {
                "db_path": str(self.db_path),
                "auto_capture": False,
                "retrieval_threshold": 0.01,
                "top_k": 4,
                "token_budget": 500,
            }
        )
        provider.initialize("experiment-session", hermes_home=self.temp.name)
        provider._store.add_memory(
            "Deployment verification requires checking the live service health endpoint.",
            kind="procedure",
            source_category="USER_EXPLICIT",
            confidence=0.95,
            importance=0.9,
        )
        set_recall_experiment(provider._store, active=True)
        outputs = []
        for index in range(3):
            query = f"How should memory guide deployment verification task {index}?"
            output = provider.prefetch(query, session_id="experiment-session")
            provider.sync_turn(
                query,
                "Check the live service health endpoint.",
                session_id="experiment-session",
            )
            outputs.append(output)
        with provider._store._lock:
            rows = provider._store._conn.execute(
                """SELECT a.condition,t.query_preview FROM recall_experiment_assignments a
                   JOIN agent_task_observations t ON t.task_id=a.task_id"""
            ).fetchall()
        conditions = [str(row["condition"]) for row in rows]
        self.assertEqual(set(conditions), {"adaptive", "fixed", "no_memory"})
        no_memory_row = next(row for row in rows if row["condition"] == "no_memory")
        output_index = int(str(no_memory_row["query_preview"]).rsplit(" ", 1)[-1].rstrip("?"))
        self.assertEqual(outputs[output_index], "")
        provider.shutdown()

    def test_summary_prospective_and_reconsolidation_keep_evidence(self) -> None:
        first, _ = self.store.add_memory(
            "Deployments require a live health check.",
            kind="procedure",
            source_category="USER_EXPLICIT",
        )
        second, _ = self.store.add_memory(
            "A deployment is complete only after the live health endpoint passes.",
            kind="procedure",
            source_category="USER_EXPLICIT",
        )
        self.store.add_edge(first, second, "supports", weight=0.7)
        generated = generate_summary_candidates(self.store)
        self.assertEqual(generated["created"], 1)
        candidate = research_snapshot(self.store)["summary_candidates"][0]
        approved = review_summary_candidate(
            self.store,
            candidate["candidate_id"],
            action="approve",
            actor="test-operator",
        )
        explanation = self.store.explain(approved["memory_id"])
        self.assertEqual(len(explanation["dependencies"]), 2)

        prospective = create_prospective_item(
            self.store,
            content="Review the controlled recall experiment",
            due_at="2026-07-16T12:00:00Z",
            actor="test-operator",
        )
        update_prospective_item(self.store, prospective["memory_id"], status="completed")
        tracked = next(
            row for row in research_snapshot(self.store)["prospective"]
            if row["memory_id"] == prospective["memory_id"]
        )
        self.assertEqual(tracked["status"], "completed")

        self.store.log_access(first, "retrieved")
        self.assertTrue(self.store.correct_memory(first, "Deployments require a passing live health check."))
        self.store.log_access(first, "injected")
        pending = research_snapshot(self.store)["reconsolidation"]
        self.assertEqual(pending["events"][0]["status"], "pending")
        self.store.log_access(first, "used")
        reconsolidation = research_snapshot(self.store)["reconsolidation"]
        self.assertEqual(reconsolidation["after_recall"], 1)
        self.assertEqual(reconsolidation["events"][0]["status"], "reexposed")
        self.assertIn("live health check", reconsolidation["events"][0]["corrected_content"])
        self.assertIn("live health check", reconsolidation["events"][0]["prior_content"])

    def test_sleep_trial_applies_one_matched_arm_and_undoes_it(self) -> None:
        memory_ids = [
            self.store.add_memory(
                f"Distinct matched Sleep trial memory {index}.",
                source_category="USER_EXPLICIT",
            )[0]
            for index in range(8)
        ]
        shadow_run = str(uuid.uuid4())
        now = utc_now()
        with self.store.transaction() as conn:
            conn.execute(
                """INSERT INTO sleep_runs(run_id,mode,status,cutoff_at,started_at,completed_at)
                   VALUES(?,'shadow','completed',?,?,?)""",
                (shadow_run, now, now, now),
            )
            for index in range(4):
                conn.execute(
                    """INSERT INTO sleep_proposals(
                         proposal_id,run_id,kind,src_id,dst_id,status,score,evidence_count,rationale,created_at
                       ) VALUES(?,?,'association',?,?,'proposed',0.8,3,'matched evidence',?)""",
                    (
                        str(uuid.uuid4()),
                        shadow_run,
                        memory_ids[index * 2],
                        memory_ids[index * 2 + 1],
                        now,
                    ),
                )
        trial = start_sleep_apply_trial(self.store, pair_limit=2)
        snapshot = research_snapshot(self.store)["sleep_trials"]
        items = [row for row in snapshot["items"] if row["trial_id"] == trial["trial_id"]]
        self.assertEqual(sum(row["assignment"] == "treatment" for row in items), 2)
        self.assertEqual(sum(row["assignment"] == "control" for row in items), 2)
        with self.store._lock:
            live_changes = self.store._conn.execute(
                "SELECT COUNT(*) count FROM sleep_edge_changes WHERE run_id=? AND reversed_at IS NULL",
                (trial["sleep_run_id"],),
            ).fetchone()["count"]
        self.assertEqual(live_changes, 2)
        undone = undo_sleep_apply_trial(self.store, trial["trial_id"])
        self.assertEqual(undone["restored_edges"], 2)


if __name__ == "__main__":
    unittest.main()
