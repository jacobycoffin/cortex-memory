"""Memory organization uses synthetic inputs and temporary stores only."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests._bootstrap import ROOT  # noqa: F401
from tests.test_jev_engine import _config
from cortex import jev
from cortex.autojudge import AutoJudge
from cortex.store import CortexStore


def response_for(payload, *, kind="episode", confidence=0.96, worth=0.92):
    answers = {}
    for qid, question in payload["questions"].items():
        if question["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": worth}
        elif qid.endswith("memory_kind"):
            keys = question["criteria"]
            answers[qid] = {
                "type": "choice", "choice": kind, "confidence": confidence,
                "probabilities": {key: 1.0 if key == kind else 0.0 for key in keys},
            }
    return {"model": "jev-synthetic", "answers": answers, "usage": {"input_tokens": 120}}


class OrganizationTests(unittest.TestCase):
    def test_classification_is_applied_and_audited_in_admission_transaction(self):
        calls = []
        def provider(endpoint, api_key, payload, timeout):
            calls.append(payload)
            return response_for(payload)

        with tempfile.TemporaryDirectory() as tmp:
            store = CortexStore(Path(tmp) / "test.db")
            self.addCleanup(store.close)
            proposal = store.propose_memory_creation(
                "On Monday the synthetic archive server recovered after a failed disk was replaced.",
                kind="semantic",
            )
            settings = jev.JevSettings.from_env({
                "CORTEX_JEV_KIND_MODE": "apply", "CORTEX_JEV_DECISION_LOG": "",
            })
            report = AutoJudge(_config(engine="jev"), jev_call=provider, jev_settings=settings).run(store)
            original = store.get_memory_creation_proposal(proposal["proposal_id"])
            memory = store.get_memory(original["result_memory_id"])
            self.assertEqual(memory["kind"], "episode")
            self.assertEqual(report["remembered"], 1)
            self.assertEqual(len(calls), 1, "classification must share the admission request")
            original = store.get_memory_creation_proposal(proposal["proposal_id"])
            self.assertEqual(original["kind"], "semantic", "preserve the capture label for audit")
            ledger = store._conn.execute(
                "SELECT effect_json FROM operator_review_decisions WHERE proposal_id=?",
                (proposal["proposal_id"],),
            ).fetchone()
            effect = json.loads(ledger[0])
            self.assertEqual(effect["kind"], "episode")
            self.assertEqual(effect["proposed_kind"], "semantic")

    def test_batch_usage_is_counted_once_per_request(self):
        records = [{"proposal_id": str(i), "content": "Synthetic factual note.", "kind": "semantic"} for i in range(3)]
        decisions = jev.judge_admission_batch(
            jev.JevSettings(batch_size=3, concurrency=1), records,
            call=lambda endpoint, key, payload, timeout: response_for(payload),
        )
        self.assertEqual(sum(d.get("usage", {}).get("input_tokens", 0) for d in decisions), 120)

    def test_link_with_uncertain_relation_is_not_created(self):
        for confidence in (None, 0.1, float("nan"), 1.1, True):
            with self.subTest(confidence=confidence):
                def provider(endpoint, key, payload, timeout):
                    return {"answers": {
                        "l1": {"type": "noul", "noul": 0.99},
                        "r1": {"type": "choice", "choice": "extends", "confidence": confidence,
                               "probabilities": {k: 1.0 if k == "extends" else 0.0
                                                 for k in payload["questions"]["r1"]["criteria"]}},
                    }}
                links = jev.judge_links(
                    jev.JevSettings(), candidate={"content": "A synthetic server uses snapshots."},
                    related=[{"memory_id": "related", "content": "Synthetic backup policy."}],
                    call=provider,
                )
                self.assertEqual(links, [])

    def test_link_self_and_duplicate_candidates_do_not_cost_questions(self):
        calls = []
        def provider(endpoint, key, payload, timeout):
            calls.append(payload)
            return {"answers": {}}
        jev.judge_links(
            jev.JevSettings(), candidate={"memory_id": "self", "content": "Synthetic fact."},
            related=[{"memory_id": mid, "content": "Synthetic fact."} for mid in ("self", "other", "other", "")],
            call=provider,
        )
        self.assertEqual(len(calls[0]["state"]["related"]), 1)
        self.assertEqual(set(calls[0]["questions"]), {"l1", "r1"})

    def test_kind_answer_rejects_mismatched_distribution(self):
        def provider(endpoint, key, payload, timeout):
            response = response_for(payload)
            response["answers"]["memory_kind"]["probabilities"] = {"semantic": 1.0}
            return response
        decisions = jev.judge_admission_batch(
            jev.JevSettings(kind_mode="apply"),
            [{"proposal_id": "synthetic", "content": "A test event happened.", "kind": "semantic"}],
            call=provider,
        )
        self.assertNotIn("approved_kind", decisions[0])

    def test_successful_link_requests_are_included_in_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CortexStore(Path(tmp) / "test.db")
            self.addCleanup(store.close)
            related_id, _ = store.add_memory("Synthetic backup policy retains snapshots for seven days.")
            store.propose_memory_creation("Synthetic server keeps nightly backups for a week.")
            def provider(endpoint, key, payload, timeout):
                return response_for(payload)  # no links, but this call still costs tokens
            judge = AutoJudge(
                _config(engine="jev", links_enabled=True, jev_links=True),
                jev_call=provider, jev_settings=jev.JevSettings(),
            )
            with patch("cortex.autojudge._find_related_memories", return_value=[{
                "memory_id": related_id, "content": "Synthetic backup policy.", "kind": "semantic",
            }]):
                report = judge.run(store)
            self.assertEqual(report["usage"]["input_tokens"], 240)
            self.assertEqual(report["jev"]["link_calls"], 1)

    def test_shadow_unknown_and_low_confidence_keep_original_kind(self):
        for mode, kind, confidence in (("shadow", "episode", 0.96), ("apply", "unknown", 0.99),
                                       ("apply", "episode", 0.3), ("apply", "episode", float("nan"))):
            with self.subTest(mode=mode, kind=kind, confidence=confidence):
                decisions = jev.judge_admission_batch(
                    jev.JevSettings(kind_mode=mode),
                    [{"proposal_id": "synthetic", "content": "Synthetic event.", "kind": "semantic"}],
                    call=lambda e, k, p, t: response_for(p, kind=kind, confidence=confidence),
                )
                self.assertNotIn("approved_kind", decisions[0])
                self.assertEqual(decisions[0]["path"], "semantic_strict")

    def test_category_cannot_relax_admission_policy(self):
        decisions = jev.judge_admission_batch(
            jev.JevSettings(kind_mode="apply"),
            [{"proposal_id": "synthetic", "content": "Synthetic event.", "kind": "semantic"}],
            call=lambda e, k, p, t: response_for(p, worth=0.62),
        )
        self.assertEqual(decisions[0]["action"], "defer")

    def test_classification_preserves_existing_duplicate_and_stale_review(self):
        for scenario in ("duplicate", "stale"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as tmp:
                store = CortexStore(Path(tmp) / "test.db")
                try:
                    text = "On Monday the synthetic archive server recovered after disk replacement."
                    existing = None
                    if scenario == "duplicate":
                        existing, _ = store.add_memory(text, kind="operational")
                    proposal = store.propose_memory_creation(text, kind="semantic")
                    def provider(e, k, p, t):
                        if scenario == "stale":
                            store.review_memory_creation(proposal["proposal_id"], "needs_context")
                        return response_for(p)
                    AutoJudge(_config(engine="jev"), jev_call=provider,
                              jev_settings=jev.JevSettings(kind_mode="apply")).run(store)
                    if existing:
                        self.assertEqual(store.get_memory(existing)["kind"], "operational")
                    else:
                        self.assertEqual(store.stats()["memories"], 0)
                        self.assertEqual(store.get_memory_creation_proposal(proposal["proposal_id"])["status"], "needs_context")
                finally:
                    store.close()

    def test_invalid_approved_kind_is_rejected_before_any_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CortexStore(Path(tmp) / "test.db")
            try:
                proposal = store.propose_memory_creation("Synthetic policy uses nightly snapshots.")
                with self.assertRaises(ValueError):
                    store.review_memory_creation(proposal["proposal_id"], "remember", approved_kind="Episode")
                self.assertEqual(store.stats()["memories"], 0)
            finally:
                store.close()

    def test_classification_rolls_back_if_review_ledger_write_fails(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            store = CortexStore(Path(tmp) / "test.db")
            try:
                proposal = store.propose_memory_creation("Synthetic policy uses nightly snapshots.")
                store._conn.execute("CREATE TRIGGER fail_ledger BEFORE INSERT ON operator_review_decisions BEGIN SELECT RAISE(ABORT, 'synthetic'); END")
                with self.assertRaises(sqlite3.IntegrityError):
                    store.review_memory_creation(proposal["proposal_id"], "remember", approved_kind="episode")
                self.assertEqual(store.stats()["memories"], 0)
                self.assertEqual(store.get_memory_creation_proposal(proposal["proposal_id"])["status"], "pending")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
