"""Replay recorded live TypeSafe responses (synthetic state) to pin the answer contract.

The fixture is raw jev-1.13.0 output captured 2026-09-21 for synthetic inputs —
no real memory content. Replaying it keeps the strict Choice/Noul validation
falsifiable: if the parser drifts from what TypeSafe actually emits, these
tests fail instead of silently accepting or dropping answers.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401
from cortex import jev

_FIXTURE = Path(__file__).parent / "fixtures" / "jev_recorded_synthetic.json"


def _recorded() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


class RecordedContractTests(unittest.TestCase):
    def test_recorded_admission_decides_and_classifies(self) -> None:
        fixture = _recorded()
        response = fixture["admission"]["response"]
        candidate = fixture["admission"]["state"]["candidate"]
        decisions = jev.judge_admission_batch(
            jev.JevSettings(kind_mode="apply", decision_log=None),
            [candidate],
            call=lambda endpoint, key, payload, timeout: response,
        )
        decision = decisions[0]
        # Recorded worth_saving 0.18 sits below the semantic_strict reject gate (0.35).
        self.assertEqual(decision["action"], "reject")
        self.assertEqual(decision["path"], "semantic_strict")
        self.assertEqual(decision["model"], "jev-1.13.0")
        klass = decision["classification"]
        self.assertTrue(klass["accepted"])
        self.assertEqual(klass["kind"], "episode")
        self.assertAlmostEqual(klass["confidence"], 1.0)
        self.assertEqual(klass["question_set"], jev.KIND_QUESTION_SET_VERSION)
        self.assertEqual(decision["approved_kind"], "episode")
        self.assertEqual(decision["usage"], {"input_tokens": 962.0, "output_tokens": 209.0})

    def test_recorded_links_create_only_the_gated_related_pair(self) -> None:
        fixture = _recorded()
        response = fixture["links"]["response"]
        state = fixture["links"]["state"]
        related = [
            {"memory_id": f"related_{index}", "content": item["content"], "kind": item["kind"]}
            for index, item in enumerate(state["related"], start=1)
        ]
        candidate = {"memory_id": "candidate", "content": state["candidate"]["content"], "kind": "semantic"}
        links = jev.judge_links(
            jev.JevSettings(decision_log=None),
            candidate=candidate,
            related=related,
            call=lambda endpoint, key, payload, timeout: response,
        )
        self.assertEqual([link["memory_id"] for link in links], ["related_1"])
        self.assertEqual(links[0]["relation"], "supports")
        self.assertAlmostEqual(links[0]["link_probability"], 0.89)
        # The recorded relation confidence is 0.44: above the 0.30 default floor
        # (kept, measured on six clear-true pairs) and below a 0.60 floor, which
        # is exactly the recalibration this replay pins.
        self.assertAlmostEqual(links[0]["relation_confidence"], 0.44)
        strict = jev.judge_links(
            jev.JevSettings(decision_log=None, link_relation_threshold=0.60),
            candidate=candidate,
            related=related,
            call=lambda endpoint, key, payload, timeout: response,
        )
        self.assertEqual(strict, [])


if __name__ == "__main__":
    unittest.main()
