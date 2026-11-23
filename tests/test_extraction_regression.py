"""A frozen extraction set, so a model change has to prove itself (#47)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import extraction_regression


class FixtureTests(unittest.TestCase):
    def test_the_frozen_set_loads(self) -> None:
        cases = extraction_regression.load()
        self.assertGreaterEqual(len(cases), 8)

    def test_every_case_has_an_identifier_and_a_turn(self) -> None:
        for case in extraction_regression.load():
            self.assertTrue(case.case_id)
            self.assertTrue(case.turns)

    def test_the_fixture_carries_no_machine_paths(self) -> None:
        """Fixtures live in a repository, so they cannot carry a real path or a real key."""
        text = extraction_regression.FIXTURE.read_text(encoding="utf-8")
        self.assertNotIn("/root/", text)
        self.assertNotIn("/home/", text)
        self.assertNotIn("sk-live-", text.replace("sk-live-9f2c8b7a6d5e4c3b2a1f0e9d8c7b6a5f", ""))

    def test_the_credential_case_is_synthetic_and_scanned_for(self) -> None:
        text = extraction_regression.FIXTURE.read_text(encoding="utf-8")
        # the only secret-shaped string is the one the refusal case is about
        self.assertEqual(text.count("sk-live-"), 1)

    def test_a_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            extraction_regression.load(Path(tempfile.gettempdir()) / "no-such-fixture.jsonl")


class RunTests(unittest.TestCase):
    def fake(self, mapping):
        def reader(text: str, role: str):
            return mapping.get(text, [])
        return reader

    def case(self, case_id, text, expect=(), forbid=()) -> extraction_regression.Case:
        return extraction_regression.Case(
            case_id=case_id, turns=[{"role": "user", "text": text}],
            expect=[{"contains": item} for item in expect], forbid=list(forbid),
        )

    def test_a_met_expectation_passes(self) -> None:
        case = self.case("a", "hello", expect=["nightly channel"])
        outcome = extraction_regression.run_case(
            case, self.fake({"hello": [{"content": "we keep the nightly channel",
                                        "kind": "decision"}]})
        )
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.missing, [])

    def test_a_missing_fact_fails_and_says_which(self) -> None:
        case = self.case("a", "hello", expect=["nightly channel"])
        outcome = extraction_regression.run_case(case, self.fake({"hello": []}))
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.missing, ["nightly channel"])
        self.assertIn("missing", outcome.reason)

    def test_a_forbidden_fact_fails(self) -> None:
        case = self.case("a", "hello", forbid=["sk-live"])
        outcome = extraction_regression.run_case(
            case, self.fake({"hello": [{"content": "token sk-live-abc", "kind": "semantic"}]})
        )
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.forbidden, ["sk-live"])

    def test_the_kind_is_checked_when_stated(self) -> None:
        case = extraction_regression.Case(
            "a", [{"role": "user", "text": "hello"}],
            expect=[{"contains": "nightly channel", "kind": "preference"}],
        )
        outcome = extraction_regression.run_case(
            case, self.fake({"hello": [{"content": "nightly channel", "kind": "decision"}]})
        )
        self.assertFalse(outcome.ok)

    def test_matching_is_on_tokens_not_exact_wording(self) -> None:
        case = self.case("a", "hello", expect=["nightly channel keep"])
        outcome = extraction_regression.run_case(
            case, self.fake({"hello": [{"content": "We keep the channel nightly.",
                                        "kind": "decision"}]})
        )
        self.assertTrue(outcome.ok)

    def test_an_empty_expectation_with_a_forbid_is_a_pure_negative_case(self) -> None:
        case = self.case("a", "hello", forbid=["traffic"])
        outcome = extraction_regression.run_case(case, self.fake({"hello": []}))
        self.assertTrue(outcome.ok)

    def test_the_outcome_round_trips(self) -> None:
        case = self.case("a", "hello", expect=["x y"])
        payload = extraction_regression.run_case(case, self.fake({"hello": []})).as_dict()
        for key in ("id", "ok", "extracted", "missing", "forbidden", "reason"):
            self.assertIn(key, payload)


class SuiteTests(unittest.TestCase):
    def test_the_deterministic_reader_passes_the_frozen_set(self) -> None:
        result = extraction_regression.suite()
        self.assertEqual(result["failed"], 0, result["failures"])

    def test_the_suite_reports_every_case(self) -> None:
        result = extraction_regression.suite()
        self.assertEqual(len(result["results"]), result["cases"])

    def test_a_regressing_reader_fails_the_suite(self) -> None:
        result = extraction_regression.suite(extractor=lambda text, role: [])
        self.assertGreater(result["failed"], 0)
        # the pure-negative cases still pass: extracting nothing breaks no prohibition
        self.assertLess(result["passed"], result["cases"])

    def test_only_selects_the_named_cases(self) -> None:
        result = extraction_regression.suite(only=["pref-terse"])
        self.assertEqual(result["cases"], 1)


class BaselineTests(unittest.TestCase):
    def result(self, cases):
        return {"results": [{"id": case_id, "extracted": facts} for case_id, facts in cases]}

    def test_a_baseline_records_the_facts_per_case(self) -> None:
        recorded = extraction_regression.baseline(
            self.result([("a", [{"kind": "decision", "content": "keep nightly"}])]), model="m1"
        )
        self.assertEqual(recorded["model"], "m1")
        self.assertEqual(recorded["cases"]["a"], ["decision: keep nightly"])

    def test_an_appeared_fact_is_reported(self) -> None:
        before = extraction_regression.baseline(self.result([("a", [])]))
        after = extraction_regression.baseline(
            self.result([("a", [{"kind": "preference", "content": "no emoji"}])])
        )
        compared = extraction_regression.compare(before, after)
        self.assertTrue(compared["changed"])
        self.assertIn("a", compared["appeared"])

    def test_a_vanished_fact_is_reported(self) -> None:
        before = extraction_regression.baseline(
            self.result([("a", [{"kind": "decision", "content": "keep nightly"}])])
        )
        after = extraction_regression.baseline(self.result([("a", [])]))
        compared = extraction_regression.compare(before, after)
        self.assertIn("a", compared["vanished"])

    def test_an_unchanged_run_is_quiet(self) -> None:
        same = self.result([("a", [{"kind": "decision", "content": "keep nightly"}])])
        compared = extraction_regression.compare(
            extraction_regression.baseline(same), extraction_regression.baseline(same)
        )
        self.assertFalse(compared["changed"])

    def test_added_and_removed_cases_are_reported(self) -> None:
        before = extraction_regression.baseline(self.result([("a", [])]))
        after = extraction_regression.baseline(self.result([("b", [])]))
        compared = extraction_regression.compare(before, after)
        self.assertEqual(compared["cases_added"], ["b"])
        self.assertEqual(compared["cases_removed"], ["a"])

    def test_a_baseline_is_json_safe(self) -> None:
        recorded = extraction_regression.baseline(
            self.result([("a", [{"kind": "decision", "content": "x"}])]), model="m"
        )
        self.assertIsInstance(json.dumps(recorded), str)


if __name__ == "__main__":
    unittest.main()
