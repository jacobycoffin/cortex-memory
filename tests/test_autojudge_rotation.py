"""AutoJudge selection must rotate by due-ness, not starve or re-bill.

Regression coverage for the audit's pipeline findings: the newest-first
window let recent proposals crowd out older eligible work (starvation), and a
deferred batch was re-sent identically on the next run (wasted LLM spend).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401 - loads the flat ``cortex`` package

from cortex.autojudge import AutoJudge, AutoJudgeConfig
from cortex.store import CortexStore


def _config(**kwargs: object) -> AutoJudgeConfig:
    return AutoJudgeConfig(
        **dict(
            dict(
                enabled=True,
                endpoint="http://127.0.0.1:1",
                model="synthetic",
                api_key_env="",
                minimum_age_seconds=0,
            ),
            **kwargs,
        )
    )


def _reply(data: dict) -> dict:
    return {
        "choices": [{"message": {"content": json.dumps(data)}}],
        "usage": {"total_tokens": 100},
    }


class AutoJudgeRotationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _three_proposals(self) -> list[str]:
        for number in range(3):
            self.store.propose_memory_creation(
                f"Synthetic audit server number {number} uses a durable backup schedule."
            )
        rows = self.store.list_memory_creation_proposals()
        return [str(proposal["proposal_id"]) for proposal in rows]

    def _defer_all_provider(self, calls: list[list[str]]):
        def provider(endpoint: str, api_key: str, payload: dict, timeout: float) -> dict:
            ids = [c["proposal_id"] for c in json.loads(payload["messages"][1]["content"])["candidates"]]
            calls.append(ids)
            return _reply(
                {
                    "decisions": [
                        {"proposal_id": i, "action": "defer", "confidence": 0.95, "reason": "Uncertain scope"}
                        for i in ids
                    ]
                }
            )

        return provider

    def test_oldest_eligible_proposal_is_not_starved_by_the_window(self) -> None:
        ids = self._three_proposals()
        oldest = ids[-1]
        with self.store._lock:
            self.store._conn.execute(
                "UPDATE memory_creation_proposals SET last_seen_at=? WHERE proposal_id=?",
                ("2020-01-01T00:00:00+00:00", oldest),
            )
            self.store._conn.commit()

        calls: list[list[str]] = []
        report = AutoJudge(
            _config(max_proposals=2, minimum_age_seconds=120),
            provider_call=self._defer_all_provider(calls),
        ).run(self.store)

        self.assertEqual(report["selected"], 1)
        self.assertEqual(calls, [[oldest]])
        status = self.store._conn.execute(
            "SELECT status FROM memory_creation_proposals WHERE proposal_id=?", (oldest,)
        ).fetchone()["status"]
        self.assertEqual(status, "pending")  # deferred, not lost

    def test_deferred_batch_is_not_re_billed_identically(self) -> None:
        self._three_proposals()
        calls: list[list[str]] = []
        judge = AutoJudge(_config(max_proposals=2), provider_call=self._defer_all_provider(calls))
        judge.run(self.store)
        judge.run(self.store)

        self.assertEqual(len(calls), 2)
        self.assertEqual(len(calls[0]), 2)
        # The second run must not re-offer the just-deferred batch: the
        # deferral cooldown sends the run to the remaining due candidate.
        self.assertNotEqual(calls[0], calls[1])
        self.assertTrue(set(calls[0]).isdisjoint(set(calls[1])))


if __name__ == "__main__":
    unittest.main()