from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tests._bootstrap import ROOT

from cortex.sleep import SleepConfig, run_sleep, undo_sleep
from cortex.store import CortexStore


class CortexSleepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _memory_pair(self) -> tuple[str, str]:
        first, _ = self.store.add_memory(
            "Hermes database backups use restic snapshots with encrypted retention.",
            kind="procedure",
            source_category="TOOL_VERIFIED",
        )
        second, _ = self.store.add_memory(
            "The nightly backup verification checks the most recent snapshot before pruning.",
            kind="procedure",
            source_category="TOOL_VERIFIED",
        )
        return first, second

    def _two_helpful_witnesses(self, memory_ids: tuple[str, str]) -> None:
        for session_id in ("session-a", "session-b"):
            task_id = self.store.create_usage_batch(
                [(memory_ids[0], 0.8), (memory_ids[1], 0.75)],
                query="verify the backup procedure",
                session_id=session_id,
            )
            self.store.resolve_usage(task_id, {memory_ids[0]: 0.9, memory_ids[1]: 0.8})
            self.store.apply_task_outcome(task_id, "helpful")

    def test_shadow_replay_requires_independent_witnesses_and_is_idempotent(self) -> None:
        first, second = self._memory_pair()
        self._two_helpful_witnesses((first, second))

        report = run_sleep(self.store, SleepConfig(mode="shadow"))
        self.assertEqual(report["association_proposals"], 1)
        self.assertEqual(report["applied_changes"], 0)
        with self.store._lock:
            edge = self.store._conn.execute(
                "SELECT * FROM edges WHERE relation='sleep_replay'"
            ).fetchone()
            proposal = self.store._conn.execute(
                "SELECT * FROM sleep_proposals WHERE run_id=? AND kind='association'",
                (report["run_id"],),
            ).fetchone()
        self.assertIsNone(edge)
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal["status"], "proposed")

        repeated = run_sleep(self.store, SleepConfig(mode="shadow"))
        self.assertEqual(repeated["usage_tasks_replayed"], 0)
        self.assertEqual(repeated["association_proposals"], 0)

    def test_apply_association_is_explicit_and_reversible(self) -> None:
        first, second = self._memory_pair()
        self._two_helpful_witnesses((first, second))
        run_sleep(self.store, SleepConfig(mode="shadow"))

        applied = run_sleep(self.store, SleepConfig(mode="apply"))
        self.assertEqual(applied["association_proposals"], 1)
        self.assertEqual(applied["applied_changes"], 1)
        with self.store._lock:
            edge = self.store._conn.execute(
                "SELECT * FROM edges WHERE relation='sleep_replay'"
            ).fetchone()
        self.assertIsNotNone(edge)

        restored = undo_sleep(self.store, applied["run_id"])
        self.assertEqual(restored["restored_edges"], 1)
        with self.store._lock:
            edge = self.store._conn.execute(
                "SELECT * FROM edges WHERE relation='sleep_replay'"
            ).fetchone()
        self.assertIsNone(edge)

    def test_dashboard_snapshot_exposes_sleep_evidence_changes_and_effect_windows(self) -> None:
        first, second = self._memory_pair()
        self._two_helpful_witnesses((first, second))

        report = run_sleep(self.store, SleepConfig(mode="apply"))
        snapshot = self.store.dashboard_snapshot()

        run = next(item for item in snapshot["sleep_runs"] if item["run_id"] == report["run_id"])
        proposal = next(item for item in snapshot["sleep_proposals"] if item["run_id"] == report["run_id"])
        change = next(item for item in snapshot["sleep_edge_changes"] if item["run_id"] == report["run_id"])
        effects = snapshot["sleep_effects"][report["run_id"]]

        self.assertEqual(run["report"]["usage_tasks_replayed"], 2)
        self.assertEqual(
            {proposal["src_content"], proposal["dst_content"]},
            {
                "Hermes database backups use restic snapshots with encrypted retention.",
                "The nightly backup verification checks the most recent snapshot before pruning.",
            },
        )
        self.assertEqual(change["relation"], "sleep_replay")
        self.assertEqual(effects["live_edge_changes"], 1)
        self.assertEqual(snapshot["sleep_state_changes"], [])

    def test_sleep_flags_contextless_memory_and_creates_only_source_cited_summary_candidates(self) -> None:
        ambiguous, _ = self.store.add_memory(
            "Use it after the service restart.",
            context_mode="context_dependent",
            entities=["service"],
        )
        first, _ = self.store.add_memory("Deployments require a live health check.")
        second, _ = self.store.add_memory(
            "A release is complete only after the live health endpoint passes."
        )
        self.store.add_edge(first, second, "supports", weight=0.7)
        self.store.add_memory(
            "The service listens on port 3000.",
            subject="service",
            predicate="port",
            object_value="3000",
        )
        self.store.add_memory(
            "The service listens on port 3001.",
            subject="service",
            predicate="port",
            object_value="3001",
        )

        report = run_sleep(self.store, SleepConfig(mode="shadow"))

        self.assertEqual(report["context_review_candidates"], 1)
        self.assertEqual(report["summary_candidates"], 1)
        proposal = self.store._conn.execute(
            "SELECT * FROM sleep_proposals WHERE run_id=? AND kind='context_review'",
            (report["run_id"],),
        ).fetchone()
        self.assertEqual(proposal["src_id"], ambiguous)
        self.assertEqual(proposal["status"], "proposed")
        candidate = self.store._conn.execute(
            "SELECT candidate_id,approved_memory_id,status FROM summary_candidates"
        ).fetchone()
        source_count = self.store._conn.execute(
            "SELECT COUNT(DISTINCT memory_id) n FROM summary_candidate_sources WHERE candidate_id=?",
            (candidate["candidate_id"],),
        ).fetchone()["n"]
        self.assertEqual(candidate["status"], "proposed")
        self.assertIsNone(candidate["approved_memory_id"])
        self.assertEqual(source_count, 2)
        summary_sources = {
            str(row["memory_id"])
            for row in self.store._conn.execute(
                "SELECT memory_id FROM summary_candidate_sources WHERE candidate_id=?",
                (candidate["candidate_id"],),
            ).fetchall()
        }
        self.assertEqual(summary_sources, {first, second})

    def test_undo_refuses_to_overwrite_a_later_edge_change(self) -> None:
        first, second = self._memory_pair()
        self._two_helpful_witnesses((first, second))
        applied = run_sleep(self.store, SleepConfig(mode="apply"))
        self.store.add_edge(first, second, "sleep_replay", weight=0.1)

        restored = undo_sleep(self.store, applied["run_id"])
        self.assertEqual(restored["restored_edges"], 0)
        self.assertEqual(restored["conflicts"], 1)
        with self.store._lock:
            edge = self.store._conn.execute(
                "SELECT * FROM edges WHERE relation='sleep_replay'"
            ).fetchone()
        self.assertIsNotNone(edge)
        self.assertGreater(edge["weight"], 0.1)

    def test_old_episode_is_replayed_once_and_recent_episode_waits(self) -> None:
        self._memory_pair()
        self.store.record_episode(
            "How do backups work?",
            "Restic creates encrypted snapshots and verifies the newest snapshot before retention pruning.",
            session_id="old-session",
        )
        self.store.record_episode("Hello", "Hello!", session_id="recent-session")
        old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        with self.store.transaction() as conn:
            conn.execute("UPDATE episodes SET created_at=? WHERE session_id='old-session'", (old,))

        report = run_sleep(self.store, SleepConfig(mode="shadow", min_episode_age_hours=12))
        self.assertEqual(report["episodes_scanned"], 1)
        self.assertEqual(report["episodes_replayed"], 1)
        repeated = run_sleep(self.store, SleepConfig(mode="shadow", min_episode_age_hours=12))
        self.assertEqual(repeated["episodes_replayed"], 0)

    def test_stale_weak_edge_downscales_without_deletion_and_undo_restores_it(self) -> None:
        first, second = self._memory_pair()
        self.store.add_edge(first, second, "related", weight=0.3)
        old = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()
        with self.store.transaction() as conn:
            conn.execute("UPDATE edges SET last_reinforced_at=? WHERE relation='related'", (old,))

        shadow = run_sleep(self.store, SleepConfig(mode="shadow", decay_after_days=120))
        repeated = run_sleep(self.store, SleepConfig(mode="shadow", decay_after_days=120))
        self.assertEqual(shadow["edge_decay_proposals"], 1)
        self.assertEqual(repeated["edge_decay_proposals"], 0)

        report = run_sleep(self.store, SleepConfig(mode="apply", decay_after_days=120))
        with self.store._lock:
            downscaled = self.store._conn.execute(
                "SELECT weight FROM edges WHERE relation='related'"
            ).fetchone()["weight"]
        self.assertEqual(report["edge_decay_proposals"], 1)
        self.assertAlmostEqual(downscaled, 0.285)

        undo_sleep(self.store, report["run_id"])
        with self.store._lock:
            restored = self.store._conn.execute(
                "SELECT weight FROM edges WHERE relation='related'"
            ).fetchone()["weight"]
        self.assertAlmostEqual(restored, 0.3)

    def test_reflection_spends_only_an_explicit_budget_and_writes_proposals_only(self) -> None:
        first, second = self._memory_pair()
        self._two_helpful_witnesses((first, second))
        response = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"proposals":[{"kind":"associate","memory_ids":["'
                            + first
                            + '","'
                            + second
                            + '"],"confidence":0.82,"rationale":"Repeated backup evidence supports review."}]}'
                        )
                    }
                }
            ],
            "usage": {"total_tokens": 321},
        }
        with patch.dict(os.environ, {"TEST_SLEEP_KEY": "secret"}), patch(
            "cortex.sleep._post_chat", return_value=response
        ) as post:
            report = run_sleep(
                self.store,
                SleepConfig(
                    mode="shadow",
                    reflection_token_budget=4000,
                    reflection_endpoint="https://provider.example/v1/chat/completions",
                    reflection_model="example/model",
                    reflection_api_key_env="TEST_SLEEP_KEY",
                ),
            )
        self.assertEqual(report["reflection_status"], "completed")
        self.assertEqual(report["reflection_billed_tokens"], 321)
        payload = post.call_args.args[2]
        self.assertLessEqual(payload["max_tokens"], 4000)
        with self.store._lock:
            reflection = self.store._conn.execute(
                "SELECT * FROM sleep_proposals WHERE run_id=? AND kind='reflection_associate'",
                (report["run_id"],),
            ).fetchone()
            edge = self.store._conn.execute(
                "SELECT * FROM edges WHERE relation='sleep_replay'"
            ).fetchone()
        self.assertIsNotNone(reflection)
        self.assertEqual(reflection["status"], "proposed")
        self.assertIsNone(edge)

    def test_zero_budget_never_calls_a_provider(self) -> None:
        with patch("cortex.sleep._post_chat") as post:
            report = run_sleep(self.store, SleepConfig(reflection_token_budget=0))
        post.assert_not_called()
        self.assertEqual(report["reflection_status"], "disabled")
        self.assertEqual(report["reflection_estimated_tokens"], 0)

    def test_progress_callback_reports_bounded_review_phases(self) -> None:
        updates: list[dict[str, object]] = []
        report = run_sleep(
            self.store,
            SleepConfig(mode="shadow", reflection_token_budget=0),
            progress_callback=updates.append,
        )

        self.assertEqual(report["status"], "completed")
        self.assertEqual(updates[0]["phase"], "starting")
        self.assertEqual(updates[-1]["phase"], "completed")
        self.assertEqual(updates[-1]["progress"], 100)
        self.assertIn("connections", {update["phase"] for update in updates})
        self.assertIn("pruning", {update["phase"] for update in updates})
        self.assertTrue(all(0 <= int(update["progress"]) <= 100 for update in updates))

    def test_provider_failure_does_not_discard_deterministic_sleep(self) -> None:
        first, second = self._memory_pair()
        self._two_helpful_witnesses((first, second))
        with patch.dict(os.environ, {"TEST_SLEEP_KEY": "secret"}), patch(
            "cortex.sleep._post_chat", side_effect=RuntimeError("provider offline")
        ):
            report = run_sleep(
                self.store,
                SleepConfig(
                    reflection_token_budget=2000,
                    reflection_endpoint="https://provider.example/v1/chat/completions",
                    reflection_model="example/model",
                    reflection_api_key_env="TEST_SLEEP_KEY",
                ),
            )
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["reflection_status"], "error")
        self.assertEqual(report["association_proposals"], 1)


if __name__ == "__main__":
    unittest.main()
