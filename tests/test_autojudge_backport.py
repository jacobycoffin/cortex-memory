"""Regression tests for the auto-judge hotfixes back-ported from the plugin.

Each test here pins one behaviour that only existed in the installed plugin
until the back-port commit:

1. Chunked judging: a run fetches up to ``max_proposals`` candidates but
   evaluates them in bounded LLM calls, sums usage across chunks, and only
   commits after every chunk has been validated.
2. ``AutoJudgeConfig.validate`` accepts ``max_proposals`` in 1..50.
3. Newest-first proposal selection (``oldest_first=False``).
4. Date-aware system prompt plus a ``first_seen_at`` candidate field.
5. ``link_orphan_memories`` only evaluates live (active/cold) memories.

Every test is written to fail on the pre-fix module (see the guard-proof run
in the task report). No network calls are made: a stub ``provider_call`` is
supplied to each test.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.autojudge import (
    AutoJudge,
    AutoJudgeConfig,
    AutoJudgeError,
    link_orphan_memories,
)
from cortex.store import CortexStore


class AutoJudgeBackportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    @staticmethod
    def config(**overrides) -> AutoJudgeConfig:
        values = {
            "enabled": True,
            "endpoint": "http://127.0.0.1:9999/v1/chat/completions",
            "model": "synthetic-memory-judge",
            "api_key_env": "",
            "credential_file": None,
            "timeout_seconds": 5.0,
            "max_output_tokens": 1200,
            "max_proposals": 12,
            "minimum_age_seconds": 0,
            "keep_threshold": 0.80,
            "decision_threshold": 0.72,
            "strong_feedback_boost": 0.08,
            "positive_feedback_boost": 0.02,
        }
        values.update(overrides)
        return AutoJudgeConfig(**values)

    def _seed_proposals(self, count: int) -> list[dict]:
        return [
            self.store.propose_memory_creation(
                f"Synthetic back-port backlog candidate {index}.",
                source_type="user_turn",
                source_category="USER_STATED",
                confidence=0.9,
                importance=0.8,
            )
            for index in range(count)
        ]

    # ------------------------------------------------------------------
    # Fix 2: validate() accepts max_proposals up to 50.
    # ------------------------------------------------------------------
    def test_config_accepts_max_proposals_up_to_fifty(self) -> None:
        # Values inside the widened bound must validate cleanly.
        self.config(max_proposals=1).validate()
        self.config(max_proposals=13).validate()
        self.config(max_proposals=50).validate()

        # The old ceiling (12) is now legal; zero and fifty-one are not, and
        # the error text must advertise the widened bound.
        for bad in (0, 51):
            with self.subTest(max_proposals=bad):
                with self.assertRaisesRegex(
                    AutoJudgeError, "auto-judge max proposals must be between 1 and 50"
                ):
                    self.config(max_proposals=bad).validate()

    # ------------------------------------------------------------------
    # Fix 3: newest-first proposal selection.
    # ------------------------------------------------------------------
    def test_run_selects_proposals_newest_first(self) -> None:
        self._seed_proposals(1)
        captured: dict = {}
        original = self.store.list_memory_creation_proposals

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return original(*args, **kwargs)

        self.store.list_memory_creation_proposals = spy  # type: ignore[assignment]

        def provider_call(endpoint, api_key, payload, timeout):
            return {"choices": [{"message": {"content": '{"decisions":[]}'}}]}

        AutoJudge(
            self.config(max_proposals=7), provider_call=provider_call
        ).run(self.store)

        self.assertEqual(captured.get("status"), "pending")
        self.assertEqual(captured.get("limit"), 7)
        self.assertIs(captured.get("oldest_first"), False)

    # ------------------------------------------------------------------
    # Fix 1a: chunked provider calls with a bounded chunk size, usage summed.
    # ------------------------------------------------------------------
    def test_run_chunks_provider_calls_and_sums_usage(self) -> None:
        self._seed_proposals(25)
        chunk_sizes: list[int] = []

        def provider_call(endpoint, api_key, payload, timeout):
            records = json.loads(payload["messages"][1]["content"])["candidates"]
            chunk_sizes.append(len(records))
            return {
                "choices": [{"message": {"content": '{"decisions":[]}'}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }

        report = AutoJudge(
            self.config(max_proposals=25), provider_call=provider_call
        ).run(self.store)

        self.assertEqual(report["selected"], 25)
        # 25 candidates must be evaluated in bounded calls of at most 10.
        self.assertEqual(chunk_sizes, [10, 10, 5])
        # Usage is the sum across every chunk, not just the last one.
        self.assertEqual(
            report["usage"],
            {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45},
        )

    # ------------------------------------------------------------------
    # Fix 1b: decisions collected from every chunk are committed in one pass.
    # ------------------------------------------------------------------
    def test_run_commits_decisions_from_every_chunk(self) -> None:
        proposals = self._seed_proposals(25)
        call_count = 0

        def provider_call(endpoint, api_key, payload, timeout):
            nonlocal call_count
            call_count += 1
            records = json.loads(payload["messages"][1]["content"])["candidates"]
            decisions = [
                {
                    "proposal_id": record["proposal_id"],
                    "action": "remember",
                    "confidence": 0.95,
                    "reason": "Synthetic but durable back-port fixture.",
                }
                for record in records
            ]
            return {
                "choices": [
                    {"message": {"content": json.dumps({"decisions": decisions})}}
                ]
            }

        report = AutoJudge(
            self.config(max_proposals=25), provider_call=provider_call
        ).run(self.store)

        self.assertEqual(call_count, 3)
        self.assertEqual(report["applied"], 25)
        self.assertEqual(report["remembered"], 25)
        for proposal in proposals:
            self.assertEqual(
                self.store.get_memory_creation_proposal(proposal["proposal_id"])[
                    "status"
                ],
                "remembered",
            )

    # ------------------------------------------------------------------
    # Fix 1c: a failed chunk aborts the run and leaves every proposal pending.
    # Uses max_proposals=12 (legal pre-fix too) so the test isolates the
    # chunking behaviour rather than the widened bound.
    # ------------------------------------------------------------------
    def test_failed_chunk_aborts_run_and_leaves_proposals_pending(self) -> None:
        proposals = self._seed_proposals(12)
        call_count = 0

        def provider_call(endpoint, api_key, payload, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                records = json.loads(payload["messages"][1]["content"])["candidates"]
                decisions = [
                    {
                        "proposal_id": record["proposal_id"],
                        "action": "remember",
                        "confidence": 0.95,
                        "reason": "First chunk would be admissible on its own.",
                    }
                    for record in records
                ]
                return {
                    "choices": [
                        {"message": {"content": json.dumps({"decisions": decisions})}}
                    ]
                }
            raise RuntimeError("synthetic provider failure on a later chunk")

        with self.assertRaises(AutoJudgeError):
            AutoJudge(
                self.config(max_proposals=12), provider_call=provider_call
            ).run(self.store)

        # Two chunk calls were attempted (10 + 2), so the failure hit a later
        # chunk; the first chunk's decisions must NOT have been committed.
        self.assertEqual(call_count, 2)
        self.assertEqual(self.store.stats()["memories"], 0)
        for proposal in proposals:
            self.assertEqual(
                self.store.get_memory_creation_proposal(proposal["proposal_id"])[
                    "status"
                ],
                "pending",
            )

    # ------------------------------------------------------------------
    # Fix 4: date-aware system prompt plus a first_seen_at candidate field.
    # ------------------------------------------------------------------
    def test_system_prompt_is_date_aware_and_candidates_carry_first_seen_at(
        self,
    ) -> None:
        proposal = self.store.propose_memory_creation(
            "Synthetic fixture describing the current deployment version.",
            source_type="user_turn",
            source_category="USER_STATED",
        )
        captured: dict = {}

        def provider_call(endpoint, api_key, payload, timeout):
            captured["system"] = payload["messages"][0]["content"]
            captured["candidate"] = json.loads(payload["messages"][1]["content"])[
                "candidates"
            ][0]
            return {"choices": [{"message": {"content": '{"decisions":[]}'}}]}

        AutoJudge(self.config(), provider_call=provider_call).run(self.store)

        today = datetime.now(timezone.utc).date().isoformat()
        system_prompt = captured["system"]
        self.assertIn(f"Today is {today}.", system_prompt)
        self.assertIn("30 days", system_prompt)
        self.assertIn("stale", system_prompt)

        candidate = captured["candidate"]
        self.assertIn("first_seen_at", candidate)
        # The field is bounded (limit=40) before transport.
        self.assertLessEqual(len(candidate["first_seen_at"]), 40)
        self.assertEqual(candidate["first_seen_at"], proposal["first_seen_at"])

    # ------------------------------------------------------------------
    # Fix 5: orphan linking only evaluates live (active/cold) memories.
    # ------------------------------------------------------------------
    def test_link_orphan_memories_skips_archived_orphans(self) -> None:
        active_id, _ = self.store.add_memory(
            "An active orphan memory about the blue deployment target.",
            kind="semantic",
            source_type="test",
            source_category="USER_STATED",
            state="active",
        )
        archived_id, _ = self.store.add_memory(
            "An archived orphan memory about the red deployment target.",
            kind="semantic",
            source_type="test",
            source_category="USER_STATED",
            state="archived",
        )

        seen_ids: list[str] = []

        def provider_call(endpoint, api_key, payload, timeout):
            for record in json.loads(payload["messages"][1]["content"])["candidates"]:
                seen_ids.append(record["memory_id"])
            return {"choices": [{"message": {"content": '{"links":[]}'}}]}

        report = link_orphan_memories(
            self.store,
            self.config(links_enabled=True),
            provider_call=provider_call,
        )

        # Both memories are orphans (neither has edges), but only the live one
        # may be counted/evaluated.
        self.assertEqual(report["orphans_found"], 1)
        self.assertIn(active_id, seen_ids)
        self.assertNotIn(archived_id, seen_ids)


if __name__ == "__main__":
    unittest.main()
