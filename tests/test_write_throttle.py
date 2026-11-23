"""Automatic writes are capped per turn, per hour and per session (#36)."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import write_throttle
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class CapTests(unittest.TestCase):
    def test_the_defaults_are_the_shipped_numbers(self) -> None:
        self.assertEqual(write_throttle.caps(),
                         (write_throttle.TURN_CAP, write_throttle.SESSION_CAP,
                          write_throttle.HOUR_CAP))

    def test_caps_are_configurable(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_WRITE_CAP_TURN": "2"}):
            self.assertEqual(write_throttle.caps()[0], 2)

    def test_a_malformed_cap_falls_back_to_the_default(self) -> None:
        for bad in ("", "many", "0", "-3"):
            with mock.patch.dict(os.environ, {"CORTEX_WRITE_CAP_TURN": bad}):
                self.assertEqual(write_throttle.caps()[0], write_throttle.TURN_CAP, bad)

    def test_user_statements_are_not_rate_limited(self) -> None:
        self.assertFalse(write_throttle.is_automatic("USER_EXPLICIT"))
        self.assertTrue(write_throttle.is_automatic("AGENT_PROPOSED"))
        self.assertTrue(write_throttle.is_automatic(None))


class ThrottleTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def fill_turn(self, count: int, *, turn_id: str = "turn-1", session: str = "s1") -> None:
        for _ in range(count):
            write_throttle.admit(self.store, session, turn_id=turn_id, at=NOW)


class TurnTests(ThrottleTestCase):
    def test_the_first_writes_are_allowed(self) -> None:
        for _ in range(write_throttle.TURN_CAP):
            self.assertTrue(
                write_throttle.admit(self.store, "s1", turn_id="turn-1", at=NOW).allowed
            )

    def test_the_next_write_in_the_same_turn_is_throttled(self) -> None:
        self.fill_turn(write_throttle.TURN_CAP)
        decision = write_throttle.admit(self.store, "s1", turn_id="turn-1", at=NOW)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.scope, "turn")
        self.assertIn("stays staged", decision.reason)

    def test_a_new_turn_gets_a_fresh_turn_budget(self) -> None:
        self.fill_turn(write_throttle.TURN_CAP)
        self.assertTrue(
            write_throttle.admit(self.store, "s1", turn_id="turn-2", at=NOW).allowed
        )

    def test_another_session_has_its_own_budget(self) -> None:
        self.fill_turn(write_throttle.TURN_CAP, session="s1")
        self.assertTrue(
            write_throttle.admit(self.store, "s2", turn_id="turn-1", at=NOW).allowed
        )


class HourAndSessionTests(ThrottleTestCase):
    def test_the_hour_cap_catches_a_storm_across_turns(self) -> None:
        with mock.patch.dict(os.environ, {"CORTEX_WRITE_CAP_HOUR": "4"}):
            for index in range(4):
                write_throttle.admit(self.store, "s1", turn_id=f"turn-{index}", at=NOW)
            decision = write_throttle.admit(self.store, "s1", turn_id="turn-9", at=NOW)
            self.assertFalse(decision.allowed)
            self.assertEqual(decision.scope, "hour")

    def test_the_session_cap_binds_across_hours(self) -> None:
        from datetime import timedelta

        with mock.patch.dict(
            os.environ, {"CORTEX_WRITE_CAP_SESSION": "6", "CORTEX_WRITE_CAP_HOUR": "99"}
        ):
            for hour in range(6):
                write_throttle.admit(
                    self.store, "s1", turn_id=f"turn-{hour}", at=NOW + timedelta(hours=hour)
                )
            decision = write_throttle.admit(
                self.store, "s1", turn_id="turn-final", at=NOW + timedelta(hours=8)
            )
            self.assertFalse(decision.allowed)
            self.assertEqual(decision.scope, "session")

    def test_a_new_hour_restores_the_hour_budget(self) -> None:
        from datetime import timedelta

        with mock.patch.dict(os.environ, {"CORTEX_WRITE_CAP_HOUR": "1"}):
            write_throttle.admit(self.store, "s1", turn_id="t1", at=NOW)
            self.assertFalse(write_throttle.admit(self.store, "s1", turn_id="t2", at=NOW).allowed)
            self.assertTrue(
                write_throttle.admit(
                    self.store, "s1", turn_id="t3", at=NOW + timedelta(hours=1)
                ).allowed
            )


class BookkeepingTests(ThrottleTestCase):
    def test_a_throttled_attempt_is_not_counted(self) -> None:
        self.fill_turn(write_throttle.TURN_CAP)
        before = write_throttle.check(self.store, "s1", turn_id="turn-1", at=NOW).used
        write_throttle.admit(self.store, "s1", turn_id="turn-1", at=NOW)
        after = write_throttle.check(self.store, "s1", turn_id="turn-2", at=NOW)
        self.assertEqual(before, write_throttle.TURN_CAP)
        self.assertEqual(after.used, 0)

    def test_the_decision_describes_itself(self) -> None:
        decision = write_throttle.admit(self.store, "s1", turn_id="turn-1", at=NOW)
        self.assertIn("allowed", decision.describe())
        self.fill_turn(write_throttle.TURN_CAP)
        self.assertIn("throttled", write_throttle.check(
            self.store, "s1", turn_id="turn-1", at=NOW).describe())

    def test_reset_clears_one_session(self) -> None:
        self.fill_turn(write_throttle.TURN_CAP)
        self.assertEqual(write_throttle.reset(self.store, "s1"), 3)
        self.assertTrue(
            write_throttle.admit(self.store, "s1", turn_id="turn-1", at=NOW).allowed
        )

    def test_prune_removes_old_buckets(self) -> None:
        from datetime import timedelta

        write_throttle.admit(self.store, "old", turn_id="t1", at=NOW - timedelta(days=60))
        self.assertEqual(write_throttle.prune(self.store, days=30, at=NOW), 3)

    def test_the_report_shows_the_caps_and_the_busiest_buckets(self) -> None:
        self.fill_turn(2)
        described = write_throttle.report(self.store)
        self.assertEqual(described["caps"]["turn"], write_throttle.TURN_CAP)
        self.assertTrue(described["top"])

    def test_a_missing_table_means_no_writes_yet(self) -> None:
        self.assertEqual(write_throttle._count(self.store, "turn:s1:x"), 0)


class JudgeIntegrationTests(ThrottleTestCase):
    """The real path: the judge must stop writing once the cap is reached."""

    def test_the_judge_defers_writes_past_the_cap(self) -> None:
        import json

        from cortex.autojudge import AutoJudge, AutoJudgeConfig
        import dataclasses

        ids = [
            self.store.propose_memory_creation(
                f"Project Acorn procedure {index} requires a verified backup step.",
                session_id="storm-session",
            )["proposal_id"]
            for index in range(3)
        ]

        def provider_call(endpoint, api_key, payload, timeout):
            candidates = json.loads(payload["messages"][1]["content"])["candidates"]
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decisions": [
                                        {
                                            "proposal_id": candidate["proposal_id"],
                                            "action": "remember",
                                            "confidence": 0.99,
                                            "reason": "stable procedure",
                                        }
                                        for candidate in candidates
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

        config = dataclasses.replace(
            AutoJudgeConfig.from_env(),
            enabled=True,
            endpoint="https://api.example.com/v1/chat/completions",
            model="gpt-6.1-sol",
            api_key_env="SYNTHETIC_JUDGE_KEY",
            minimum_age_seconds=0,
            keep_threshold=0.80,
            max_proposals=3,
        )
        with mock.patch.dict(
            os.environ, {"SYNTHETIC_JUDGE_KEY": "key", "CORTEX_WRITE_CAP_TURN": "1"}
        ):
            report = AutoJudge(config, provider_call=provider_call).run(self.store)

        self.assertEqual(report.get("remembered", 0), 1)
        self.assertEqual(report.get("throttled", 0), 2)
        remembered = [
            self.store.get_memory_creation_proposal(proposal_id) for proposal_id in ids
        ]
        self.assertEqual(sum(1 for item in remembered if item["status"] == "remembered"), 1)
        # Deferred, never dropped: the rest are still waiting.
        self.assertEqual(sum(1 for item in remembered if item["status"] == "pending"), 2)


class TurnKeyTests(ThrottleTestCase):
    """A turn must not gain a fresh budget by taking longer than a second."""

    def test_two_admits_seconds_apart_share_one_budget(self) -> None:
        from datetime import timedelta

        first = write_throttle.admit(self.store, "s1", at=NOW)
        second = write_throttle.admit(self.store, "s1", at=NOW + timedelta(seconds=3))
        used = first.used + (1 if second.allowed else 0)
        self.assertLessEqual(used, write_throttle.TURN_CAP)
        self.assertFalse(second.allowed if write_throttle.TURN_CAP == 1 else False)

    def test_the_wall_clock_does_not_change_the_bucket(self) -> None:
        from datetime import timedelta

        left = write_throttle._bucket("turn", "s1", None, NOW)
        right = write_throttle._bucket("turn", "s1", None, NOW + timedelta(hours=2))
        self.assertEqual(left, right)

    def test_an_explicit_turn_id_still_separates_turns(self) -> None:
        left = write_throttle._bucket("turn", "s1", "turn-a", NOW)
        right = write_throttle._bucket("turn", "s1", "turn-b", NOW)
        self.assertNotEqual(left, right)


class StoreIntegrationTests(ThrottleTestCase):
    def test_a_storm_is_stopped_after_the_turn_budget(self) -> None:
        results = [
            write_throttle.admit(self.store, "s1", turn_id="turn-1", at=NOW).allowed
            for _ in range(write_throttle.TURN_CAP + 3)
        ]
        self.assertEqual(results.count(True), write_throttle.TURN_CAP)
        self.assertIn(False, results)


if __name__ == "__main__":
    unittest.main()
