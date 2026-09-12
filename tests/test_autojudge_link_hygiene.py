"""Auto-judge link hygiene: no duplicate amplification, bounded and validated.

Regression coverage for the audit's pipeline findings:
* identical links in one decision were each inserted, inflating evidence_count;
* the orphan linker ignored the configured output budget (always 8192);
* the orphan linker did not validate its provider endpoint.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests._bootstrap import ROOT  # noqa: F401 - loads the flat ``cortex`` package

from cortex.autojudge import (
    AutoJudgeConfig,
    _apply_links,
    _parse_decisions,
    link_orphan_memories,
)
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


class LinkHygieneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_duplicate_links_in_one_decision_are_not_amplified(self) -> None:
        left, _ = self.store.add_memory(
            "Synthetic Alpha stores reliable backup retention information."
        )
        right, _ = self.store.add_memory(
            "Synthetic Beta is an unrelated independent storage reference."
        )
        links = [
            {"memory_id": right, "relation": "supports", "rationale": "Synthetic hallucinated link"}
        ] * 3
        parsed = _parse_decisions(
            _reply(
                {
                    "decisions": [
                        {
                            "proposal_id": "p",
                            "action": "remember",
                            "confidence": 0.9,
                            "reason": "synthetic",
                            "links": links,
                        }
                    ]
                }
            ),
            {"p"},
        )
        created = _apply_links(
            self.store,
            new_memory_id=left,
            links=parsed[0]["links"],
            review_id="synthetic-review",
            actor="audit",
            proposal_id="p",
        )
        self.assertEqual(created, 1)
        edge = self.store._conn.execute(
            "SELECT evidence_count FROM edges WHERE src_id=? AND dst_id=?", (left, right)
        ).fetchone()
        self.assertEqual(edge["evidence_count"], 1)

    def test_orphan_linker_rejects_untrusted_endpoint_before_calling(self) -> None:
        self.store.add_memory("Synthetic orphan remains independent and has no relevant connections.")
        calls: list[dict] = []

        def empty_provider(endpoint: str, api_key: str, payload: dict, timeout: float) -> dict:
            calls.append({"endpoint": endpoint})
            return _reply({"batch_links": {}})

        with patch("cortex.autojudge.MemoryRetriever"), patch(
            "cortex.autojudge._find_related_memories", return_value=[]
        ):
            with self.assertRaises(ValueError):
                link_orphan_memories(
                    self.store,
                    _config(endpoint="http://remote.invalid", max_output_tokens=64, timeout_seconds=999),
                    provider_call=empty_provider,
                )
        self.assertEqual(calls, [])

    def test_orphan_linker_honors_output_budget_above_floor(self) -> None:
        self.store.add_memory("Synthetic orphan remains independent and has no relevant connections.")
        calls: list[dict] = []

        def empty_provider(endpoint: str, api_key: str, payload: dict, timeout: float) -> dict:
            calls.append({"max_tokens": payload["max_tokens"]})
            return _reply({"batch_links": {}})

        with patch("cortex.autojudge.MemoryRetriever"), patch(
            "cortex.autojudge._find_related_memories", return_value=[]
        ):
            link_orphan_memories(
                self.store,
                _config(endpoint="http://127.0.0.1:1", max_output_tokens=64, timeout_seconds=999),
                provider_call=empty_provider,
            )
        self.assertEqual(calls, [{"max_tokens": 256}])

        calls.clear()
        with patch("cortex.autojudge.MemoryRetriever"), patch(
            "cortex.autojudge._find_related_memories", return_value=[]
        ):
            link_orphan_memories(
                self.store,
                _config(endpoint="http://127.0.0.1:1", max_output_tokens=1800, timeout_seconds=999),
                provider_call=empty_provider,
            )
        self.assertEqual(calls, [{"max_tokens": 1800}])


if __name__ == "__main__":
    unittest.main()