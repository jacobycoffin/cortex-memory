"""What a pre-turn recall hook is promised, and what it must promise back (#144)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import preturn_contract as pc
from cortex.store import CortexStore

KNOWN = {"aaaaaaaa-1111", "bbbbbbbb-2222"}


class ValidateTests(unittest.TestCase):
    def request(self, **overrides):
        base = dict(task_id="t1", session_id="s1", query="what about the port",
                    budget_tokens=700, deadline_ms=1500)
        base.update(overrides)
        return pc.Request(**base)

    def test_a_well_formed_response_conforms(self) -> None:
        response = pc.Response(block="aaaaaaaa-1111 the port is 8100", ids=["aaaaaaaa-1111"],
                               elapsed_ms=120)
        self.assertEqual(pc.validate(self.request(), response, known_ids=KNOWN), [])

    def test_a_degraded_response_conforms(self) -> None:
        """A hook that cannot do its job still keeps the contract by saying so."""
        self.assertEqual(pc.validate(self.request(), pc.degraded("embedder down")), [])

    def test_something_that_is_not_a_response_is_a_violation(self) -> None:
        problems = pc.validate(self.request(), "a string")  # type: ignore[arg-type]
        self.assertIn("may not fail the turn", problems[0])

    def test_a_duplicate_id_is_a_violation(self) -> None:
        response = pc.Response(block="x", ids=["aaaaaaaa-1111", "aaaaaaaa-1111"])
        self.assertTrue(any("twice" in problem for problem in pc.validate(self.request(), response)))

    def test_an_unknown_id_is_a_violation(self) -> None:
        response = pc.Response(block="x", ids=["zzzzzzzz-9999"])
        problems = pc.validate(self.request(), response, known_ids=KNOWN)
        self.assertTrue(any("not memories in this store" in problem for problem in problems))

    def test_an_id_in_the_block_but_not_the_list_is_a_violation(self) -> None:
        """An id printed but not returned is an unattributable memory."""
        response = pc.Response(block="bbbbbbbb-2222 says the port is 8100", ids=["aaaaaaaa-1111"])
        problems = pc.validate(self.request(), response, known_ids=KNOWN)
        self.assertTrue(any("appear in the block but not in the id list" in problem
                            for problem in problems))

    def test_an_over_budget_block_is_a_violation(self) -> None:
        response = pc.Response(block="x" * 5000, ids=[])
        problems = pc.validate(self.request(budget_tokens=100), response)
        self.assertTrue(any("against a" in problem and "char budget" in problem
                            for problem in problems))

    def test_a_block_at_the_budget_is_fine(self) -> None:
        response = pc.Response(block="x" * 2800, ids=[])
        self.assertEqual(pc.validate(self.request(budget_tokens=700), response), [])

    def test_a_missed_deadline_must_be_reported(self) -> None:
        response = pc.Response(block="x", ids=[], elapsed_ms=9000)
        problems = pc.validate(self.request(deadline_ms=1500), response)
        self.assertTrue(any("without reporting degraded" in problem for problem in problems))

    def test_a_missed_deadline_that_is_reported_is_accepted(self) -> None:
        response = pc.Response(block="", ids=[], abstained=True, elapsed_ms=9000, degraded=True)
        self.assertEqual(pc.validate(self.request(deadline_ms=1500), response), [])

    def test_an_abstention_returning_ids_is_a_violation(self) -> None:
        response = pc.Response(block="x", ids=["aaaaaaaa-1111"], abstained=True)
        self.assertTrue(any("either abstains or it answers" in problem
                            for problem in pc.validate(self.request(), response)))

    def test_a_degraded_response_must_abstain(self) -> None:
        response = pc.Response(block="", ids=[], abstained=False, degraded=True)
        self.assertTrue(any("must say it abstained" in problem
                            for problem in pc.validate(self.request(), response)))

    def test_a_wrong_contract_version_is_a_violation(self) -> None:
        response = pc.Response(block="", ids=[], contract_version=99)
        self.assertTrue(any("contract version" in problem
                            for problem in pc.validate(self.request(), response)))


class DeterminismTests(unittest.TestCase):
    def request(self):
        return pc.Request(task_id="t", session_id="s", query="q")

    def test_the_same_request_twice_is_deterministic(self) -> None:
        described = pc.deterministic(lambda request: pc.Response(block="x", ids=["a"]), self.request())
        self.assertTrue(described["deterministic"])

    def test_a_reshuffling_hook_is_not_deterministic(self) -> None:
        counter = {"n": 0}

        def run(request):
            counter["n"] += 1
            return pc.Response(block="x", ids=[f"id{counter['n']}"])

        described = pc.deterministic(run, self.request())
        self.assertFalse(described["deterministic"])
        self.assertIn("different id lists", described["reason"])

    def test_a_raising_hook_is_reported_not_propagated(self) -> None:
        def run(request):
            raise RuntimeError("boom")

        described = pc.deterministic(run, self.request())
        self.assertFalse(described["deterministic"])
        self.assertIn("raised", described["reason"])


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def run_with(self, *, budget: int = 620, tokens: int = 500, selected: int = 5,
                 limit: int = 6, prepare_ms: float = 100.0) -> None:
        self.store.record_recall_run(
            session_id="s", query="q", mode="auto", reason="test", requested_limit=limit,
            token_budget=budget, candidate_count=1, selected_count=selected,
            estimated_tokens=tokens, prepare_ms=prepare_ms, abstained=False, task_id=None,
        )

    def test_a_conforming_pipeline_has_no_violations(self) -> None:
        self.run_with()
        self.assertEqual(pc.audit(self.store)["verdict"], "conforms")

    def test_an_over_budget_run_is_found(self) -> None:
        """A documented contract finds nothing; a checked one finds this."""
        self.run_with(budget=100, tokens=500)
        self.assertEqual(pc.audit(self.store)["over_budget"], 1)
        self.assertEqual(pc.audit(self.store)["verdict"], "violations")

    def test_an_over_limit_run_is_found(self) -> None:
        self.run_with(limit=2, selected=5)
        self.assertEqual(pc.audit(self.store)["over_limit"], 1)

    def test_an_over_deadline_run_is_counted_separately(self) -> None:
        self.run_with(prepare_ms=pc.DEFAULT_DEADLINE_MS + 500)
        described = pc.audit(self.store)
        self.assertEqual(described["over_deadline"], 1)
        self.assertEqual(described["over_budget"], 0)

    def test_an_empty_store_conforms_vacuously(self) -> None:
        described = pc.audit(self.store)
        self.assertEqual(described["runs"], 0)
        self.assertEqual(described["verdict"], "conforms")

    def test_the_contract_is_data_and_lists_its_guarantees(self) -> None:
        described = pc.contract()
        self.assertEqual(described["version"], pc.CONTRACT_VERSION)
        self.assertGreaterEqual(len(described["guarantees"]), 5)

    def test_the_render_states_the_verdict_and_the_guarantees(self) -> None:
        self.run_with()
        rendered = "\n".join(pc.render(self.store))
        self.assertIn("pre-turn contract", rendered)
        self.assertIn("may not fail the turn", rendered)

    def test_a_broken_store_conforms_vacuously(self) -> None:
        class Broken:
            _conn = None

        self.assertEqual(pc.audit(Broken())["runs"], 0)


if __name__ == "__main__":
    unittest.main()
