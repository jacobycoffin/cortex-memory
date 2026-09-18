"""Jev engine integration: calibrated policy, batching, fallback, links, logging.

The Jev engine (``CORTEX_AUTO_JUDGE_ENGINE=jev``) maps typed TypeSafe answers to
Cortex's existing remember/reject/defer decisions with corpus-calibrated
thresholds. These tests stub the provider call end to end: no network, no key.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests._bootstrap import ROOT  # noqa: F401 - loads the flat ``cortex`` package

from cortex import jev
from cortex.autojudge import (
    AutoJudge,
    AutoJudgeConfig,
    AutoJudgeError,
    link_orphan_memories,
)
from cortex.store import CortexStore

_ADMISSION_QIDS = (
    "worth_saving",
    "durable",
    "standalone",
    "useful_again",
    "scope_clear",
    "curated",
    "duplicate_of_related",
)


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


def _chat_reply(data: dict) -> dict:
    return {"choices": [{"message": {"content": json.dumps(data)}}], "usage": {"total_tokens": 10}}


class _JevStub:
    """Provider-shaped stub: routes admission vs link questions by payload shape."""

    def __init__(self) -> None:
        self.ws_by_fragment: dict[str, float] = {}
        self.link_gate_by_fragment: dict[str, tuple[float, str]] = {}
        self.calls: list[dict] = []
        self.fail = False

    def __call__(self, endpoint: str, api_key: str, payload: dict, timeout: float) -> dict:
        self.calls.append(payload)
        if self.fail:
            raise jev.JevError("synthetic jev outage")
        questions = payload.get("questions") or {}
        if "l1" in questions:
            return self._links_response(payload)
        return self._admission_response(payload)

    def _admission_response(self, payload: dict) -> dict:
        state = payload["state"]
        answers: dict = {}
        if "candidate" in state:
            refs = {"": state["candidate"]}
        else:
            refs = {f"c{index}_": item for index, item in state["candidates"].items()}
        for prefix, record in refs.items():
            content = str(record.get("content") or "")
            ws = 0.5
            for fragment, value in self.ws_by_fragment.items():
                if fragment in content:
                    ws = value
                    break
            for qid in _ADMISSION_QIDS:
                value = ws if qid == "worth_saving" else 0.7
                answers[f"{prefix}{qid}"] = {"type": "noul", "noul": value}
        return {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 150}}

    def _links_response(self, payload: dict) -> dict:
        related = payload["state"].get("related") or []
        answers: dict = {}
        for index, item in enumerate(related, start=1):
            gate, relation = 0.0, "none"
            for fragment, value in self.link_gate_by_fragment.items():
                if fragment in str(item.get("content") or ""):
                    gate, relation = value
                    break
            answers[f"l{index}"] = {"type": "noul", "noul": gate}
            answers[f"r{index}"] = {
                "type": "choice",
                "choice": relation,
                "confidence": 0.9,
                "probabilities": {relation: 0.9},
            }
        return {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 90}}


class JevPolicyTests(unittest.TestCase):
    def test_map_admission_default_and_strict_paths(self) -> None:
        default = jev.DEFAULT_ADMIT_PATH
        self.assertEqual(jev.map_admission(worth_saving=0.61, boost=0.0, path=default), "remember")
        self.assertEqual(jev.map_admission(worth_saving=0.39, boost=0.0, path=default), "reject")
        self.assertEqual(jev.map_admission(worth_saving=0.50, boost=0.0, path=default), "defer")
        self.assertEqual(jev.map_admission(worth_saving=None, boost=0.0, path=default), "defer")
        strict = jev.ADMISSION_PATHS["semantic"]
        self.assertEqual(jev.map_admission(worth_saving=0.60, boost=0.0, path=strict), "defer")
        self.assertEqual(jev.map_admission(worth_saving=0.66, boost=0.0, path=strict), "remember")
        self.assertEqual(jev.map_admission(worth_saving=0.34, boost=0.0, path=strict), "reject")
        review = jev.ADMISSION_PATHS["builtin_memory"]
        self.assertEqual(jev.map_admission(worth_saving=0.99, boost=0.0, path=review), "defer")
        self.assertEqual(jev.map_admission(worth_saving=0.01, boost=0.0, path=review), "defer")

    def test_feedback_boost_tips_borderline_and_shields_from_reject(self) -> None:
        default = jev.DEFAULT_ADMIT_PATH
        self.assertEqual(jev.map_admission(worth_saving=0.55, boost=0.08, path=default), "remember")
        self.assertEqual(jev.map_admission(worth_saving=0.36, boost=0.06, path=default), "defer")

    def test_path_selection_prefers_source_type(self) -> None:
        path = jev.admission_path_for("semantic", "builtin_memory")
        self.assertEqual(path.label, "builtin_memory_review")
        path = jev.admission_path_for("semantic", "user_turn")
        self.assertEqual(path.label, "semantic_strict")
        path = jev.admission_path_for("decision", "user_turn")
        self.assertEqual(path.label, "default_p1")

    def test_settings_from_env(self) -> None:
        env = {
            "CORTEX_JEV_MODEL": "jev-1.13.0",
            "CORTEX_JEV_BATCH_SIZE": "4",
            "CORTEX_JEV_DECISION_LOG": "",
            "CORTEX_AUTO_JUDGE_CREDENTIAL_FILE": "/tmp/cortex.env",
        }
        settings = jev.JevSettings.from_env(env)
        self.assertEqual(settings.model, "jev-1.13.0")
        self.assertEqual(settings.batch_size, 4)
        self.assertIsNone(settings.decision_log)
        self.assertEqual(settings.credential_file, Path("/tmp/cortex.env"))
        default = jev.JevSettings.from_env({})
        self.assertTrue(str(default.decision_log).endswith("jev-decisions.jsonl"))

    def test_settings_validation_rejects_unbounded_values(self) -> None:
        with self.assertRaises(jev.JevError):
            jev.JevSettings(batch_size=99).validate()
        with self.assertRaises(jev.JevError):
            jev.JevSettings(link_threshold=1.5).validate()
        with self.assertRaises(jev.JevError):
            jev.JevSettings(endpoint="http://remote.invalid").validate()


class JevEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.stub = _JevStub()

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _propose(self, content: str, **kwargs: object) -> str:
        self.store.propose_memory_creation(content, **kwargs)
        rows = [
            proposal
            for proposal in self.store.list_memory_creation_proposals(status="all", limit=50)
            if proposal["content"] == content
        ]
        return str(rows[0]["proposal_id"])

    def _row(self, proposal_id: str):
        return self.store._conn.execute(
            "SELECT status, actor, decision_note FROM memory_creation_proposals WHERE proposal_id=?",
            (proposal_id,),
        ).fetchone()

    def _status(self, proposal_id: str) -> str:
        return str(self._row(proposal_id)["status"])

    def _settings(self, **kwargs: object) -> jev.JevSettings:
        defaults: dict = {"batch_size": 1, "concurrency": 2, "decision_log": None}
        defaults.update(kwargs)
        return jev.JevSettings(**defaults)

    def test_end_to_end_calibrated_actions(self) -> None:
        alpha = self._propose(
            "Synthetic Alpha records a durable backup schedule decision.", kind="operational"
        )
        beta = self._propose(
            "Synthetic Beta is a transient noisy fragment of a conversation.", kind="operational"
        )
        gamma = self._propose(
            "Synthetic Gamma sits in the uncertain middle of the band.", kind="operational"
        )
        self.stub.ws_by_fragment = {"Alpha": 0.82, "Beta": 0.18, "Gamma": 0.50}
        judge = AutoJudge(
            _config(engine="jev"), jev_call=self.stub, jev_settings=self._settings()
        )
        report = judge.run(self.store)
        self.assertEqual(report["engine"], "jev")
        self.assertEqual((report["remembered"], report["rejected"], report["deferred"]), (1, 1, 1))
        self.assertEqual(self._status(alpha), "remembered")
        self.assertEqual(self._status(beta), "rejected")
        self.assertEqual(self._status(gamma), "pending")
        self.assertEqual(report["jev"]["calls"], 3)
        self.assertEqual(report["jev"]["models"], ["jev-1.13.0"])
        self.assertIn("latency_p50_ms", report["jev"])
        row = self._row(alpha)
        self.assertEqual(row["actor"], "cortex-auto-judge:jev-1.13.0")
        self.assertIn("worth_saving 0.82", row["decision_note"])
        self.assertIn("path default_p1", row["decision_note"])

    def test_builtin_memory_defers_everything_to_review(self) -> None:
        proposal = self._propose(
            "Synthetic builtin mirror entry that would otherwise be admitted.",
            source_type="builtin_memory",
        )
        self.stub.ws_by_fragment = {"builtin": 0.95}
        judge = AutoJudge(
            _config(engine="jev"), jev_call=self.stub, jev_settings=self._settings()
        )
        report = judge.run(self.store)
        self.assertEqual(self._status(proposal), "pending")
        self.assertEqual(report["jev"]["defer_by_path"].get("builtin_memory_review"), 1)
        self.assertGreaterEqual(report["jev"]["audit_flagged"], 1)

    def test_guard_forces_needs_context_even_when_jev_remembers(self) -> None:
        proposal = self._propose("Synthetic quarantined fragment never admitted automatically.")
        with self.store._lock:
            self.store._conn.execute(
                "UPDATE memory_creation_proposals SET quarantine_reason=? WHERE proposal_id=?",
                ("secret-like content", proposal),
            )
            self.store._conn.commit()
        self.stub.ws_by_fragment = {"quarantined": 0.95}
        judge = AutoJudge(
            _config(engine="jev"), jev_call=self.stub, jev_settings=self._settings()
        )
        report = judge.run(self.store)
        self.assertEqual(self._status(proposal), "needs_context")
        self.assertEqual(report["guarded"], 1)

    def test_jev_failure_falls_back_to_the_chat_provider(self) -> None:
        proposal = self._propose("Synthetic fallback candidate for the chat provider path.")
        self.stub.fail = True
        chat_calls: list[dict] = []

        def provider(endpoint: str, api_key: str, payload: dict, timeout: float) -> dict:
            chat_calls.append(payload)
            ids = [
                item["proposal_id"]
                for item in json.loads(payload["messages"][1]["content"])["candidates"]
            ]
            return _chat_reply(
                {
                    "decisions": [
                        {"proposal_id": pid, "action": "remember", "confidence": 0.9, "reason": "fallback"}
                        for pid in ids
                    ]
                }
            )

        judge = AutoJudge(
            _config(engine="jev"),
            provider_call=provider,
            jev_call=self.stub,
            jev_settings=self._settings(),
        )
        report = judge.run(self.store)
        self.assertEqual(len(chat_calls), 1)
        self.assertEqual(report["jev"]["fallback_chunks"], 1)
        self.assertEqual(self._status(proposal), "remembered")

    def test_jev_failure_without_fallback_fails_closed(self) -> None:
        proposal = self._propose("Synthetic candidate that must stay pending on outage.")
        self.stub.fail = True
        judge = AutoJudge(
            _config(engine="jev", jev_fallback=False),
            jev_call=self.stub,
            jev_settings=self._settings(),
        )
        with self.assertRaises(AutoJudgeError):
            judge.run(self.store)
        self.assertEqual(self._status(proposal), "pending")

    def test_batched_request_shape_labels_candidates(self) -> None:
        self._propose("Synthetic first batch candidate about backups.")
        self._propose("Synthetic second batch candidate about cameras.")
        self._propose("Synthetic third trailing candidate about storage.")
        self.stub.ws_by_fragment = {"first": 0.80, "second": 0.20, "third": 0.80}
        judge = AutoJudge(
            _config(engine="jev"),
            jev_call=self.stub,
            jev_settings=self._settings(batch_size=2, concurrency=1),
        )
        judge.run(self.store)
        self.assertEqual(len(self.stub.calls), 2)
        first_questions = self.stub.calls[0]["questions"]
        self.assertIn("c1_worth_saving", first_questions)
        self.assertIn("c2_worth_saving", first_questions)
        self.assertEqual(set(self.stub.calls[0]["state"]["candidates"]), {"c1", "c2"})
        second_questions = self.stub.calls[1]["questions"]
        self.assertIn("worth_saving", second_questions)
        self.assertIn("candidate", self.stub.calls[1]["state"])

    def test_decision_log_records_answers_without_content(self) -> None:
        log_path = Path(self.tmp.name) / "jev-decisions.jsonl"
        proof = "Synthetic log-proof content string."
        self._propose(proof, kind="operational")
        self.stub.ws_by_fragment = {"log-proof": 0.82}
        judge = AutoJudge(
            _config(engine="jev"),
            jev_call=self.stub,
            jev_settings=self._settings(decision_log=log_path),
        )
        judge.run(self.store)
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        entry = json.loads(lines[0])
        self.assertEqual(entry["event"], "admission")
        self.assertEqual(entry["action"], "remember")
        self.assertEqual(entry["worth_saving"], 0.82)
        self.assertEqual(entry["question_set"], jev.QUESTION_SET_VERSION)
        self.assertEqual(entry["path"], "default_p1")
        self.assertNotIn("log-proof", json.dumps(entry))


class JevLinkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.stub = _JevStub()

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _settings(self, **kwargs: object) -> jev.JevSettings:
        defaults: dict = {"batch_size": 1, "concurrency": 2, "decision_log": None}
        defaults.update(kwargs)
        return jev.JevSettings(**defaults)

    def test_admission_links_are_created_for_remembered_candidates(self) -> None:
        related_id, _ = self.store.add_memory(
            "Synthetic related storage note about backup retention schedules."
        )
        self.store.propose_memory_creation(
            "Synthetic Delta documents the backup retention schedule policy for the lab."
        )
        self.stub.ws_by_fragment = {"Delta": 0.85}
        self.stub.link_gate_by_fragment = {"backup retention schedules": (0.80, "extends")}
        judge = AutoJudge(
            _config(engine="jev", links_enabled=True, jev_links=True),
            jev_call=self.stub,
            jev_settings=self._settings(),
        )
        with patch("cortex.autojudge.MemoryRetriever", return_value=object()), patch(
            "cortex.autojudge._find_related_memories",
            return_value=[
                {
                    "memory_id": related_id,
                    "content": "Synthetic related storage note about backup retention schedules.",
                    "kind": "semantic",
                    "score": 0.9,
                }
            ],
        ):
            report = judge.run(self.store)
        self.assertEqual(report["links_created"], 1)
        self.assertEqual(report["jev"]["link_calls"], 1)
        edge = self.store._conn.execute(
            "SELECT src_id, relation FROM edges WHERE dst_id=?", (related_id,)
        ).fetchone()
        self.assertIsNotNone(edge)
        self.assertEqual(edge["relation"], "extends")

    def test_orphan_linker_jev_engine_creates_gated_edges(self) -> None:
        related_id, _ = self.store.add_memory(
            "Synthetic established note about Frigate camera storage."
        )
        self.store.add_memory("Synthetic orphan memory about the Frigate camera storage plan.")
        self.stub.link_gate_by_fragment = {"established note": (0.80, "supports")}
        with patch("cortex.autojudge.MemoryRetriever", return_value=object()), patch(
            "cortex.autojudge._find_related_memories",
            return_value=[
                {
                    "memory_id": related_id,
                    "content": "Synthetic established note about Frigate camera storage.",
                    "kind": "semantic",
                    "score": 0.9,
                }
            ],
        ):
            report = link_orphan_memories(
                self.store,
                _config(link_engine="jev"),
                jev_call=self.stub,
                jev_settings=self._settings(),
            )
        self.assertEqual(report["link_engine"], "jev")
        self.assertEqual(report["links_created"], 1)
        edge = self.store._conn.execute(
            "SELECT relation FROM edges WHERE dst_id=?", (related_id,)
        ).fetchone()
        self.assertEqual(edge["relation"], "supports")

    def test_orphan_linker_jev_engine_below_gate_creates_nothing(self) -> None:
        related_id, _ = self.store.add_memory(
            "Synthetic established note about Plex libraries."
        )
        self.store.add_memory("Synthetic orphan memory about the Plex library plan.")
        self.stub.link_gate_by_fragment = {"established note": (0.40, "supports")}
        with patch("cortex.autojudge.MemoryRetriever", return_value=object()), patch(
            "cortex.autojudge._find_related_memories",
            return_value=[
                {
                    "memory_id": related_id,
                    "content": "Synthetic established note about Plex libraries.",
                    "kind": "semantic",
                    "score": 0.9,
                }
            ],
        ):
            report = link_orphan_memories(
                self.store,
                _config(link_engine="jev"),
                jev_call=self.stub,
                jev_settings=self._settings(),
            )
        self.assertEqual(report["links_created"], 0)
        edge = self.store._conn.execute(
            "SELECT 1 FROM edges WHERE dst_id=?", (related_id,)
        ).fetchone()
        self.assertIsNone(edge)


if __name__ == "__main__":
    unittest.main()
