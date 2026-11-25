"""Graded components survive the trace trim (#222): a score keeps its reason."""

from __future__ import annotations

import unittest

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import trace_components


def full_dump(**overrides) -> dict:
    """A components dict shaped like the scorer's: graded signals plus adjustment bookkeeping."""
    dump = {
        "lexical": 0.82, "phrase": 0.5, "semantic": 0.71, "activation": 0.33,
        "utility": 0.4, "importance": 0.6, "confidence": 0.78, "source_reliability": 0.5,
        "currentness": 0.9, "memory_type": 0.55, "uniqueness": 0.3, "graph": 0.2,
        "neighborhood": 0.1,
        "context_boost": 0.06, "context_adaptation": 0.02, "relevance_penalty": 0.0,
        "stale_risk": 0.1, "wrong_rate": 0.0, "contradiction_risk": 0.0, "prospective_due": 0.0,
        "superseded": 0.0, "pinned": 1.0, "cold": 0.0, "archived": 0.0, "stranded": 0.0,
        "schema_example": 0.0, "specificity_request": 0.5, "operator_policy": 0.0,
        "context_gate": 1.0, "decay": 0.0,
    }
    dump.update(overrides)
    return dump


def candidate(memory_id: str = "m1", *, selected: bool = False, score: float = 0.4, **overrides) -> dict:
    return {"memory_id": memory_id, "score": score, "rank": 3, "selected": selected,
            "reason": "selected by score", "components": full_dump(**overrides)}


class CompactTests(unittest.TestCase):
    def test_the_graded_signals_are_kept(self) -> None:
        kept = trace_components.compact(candidate())["components_graded"]
        for signal in ("lexical", "semantic", "activation", "importance", "confidence", "currentness"):
            self.assertIn(signal, kept, signal)
        self.assertEqual(len(kept), 6, "six signals, so the saving is real")

    def test_the_adjustments_are_dropped_and_counted(self) -> None:
        compacted = trace_components.compact(candidate())
        self.assertNotIn("context_gate", compacted["components_graded"],
                         "a gate is a multiplier, not an explanation")
        self.assertNotIn("graph", compacted["components_graded"], "graph is a +-1 flag, not a score")
        self.assertGreaterEqual(compacted["components_dropped_count"], 10)
        self.assertTrue(compacted["components_dropped_sample"])
        self.assertNotIn("components", compacted)

    def test_the_old_marker_survives_so_existing_readers_are_unaffected(self) -> None:
        """`components_omitted` still means exactly what it meant: the full dump is not here."""
        self.assertTrue(trace_components.compact(candidate())["components_omitted"])

    def test_a_value_is_rounded_not_truncated_to_zero(self) -> None:
        kept = trace_components.compact(candidate(activation=0.00004))["components_graded"]
        self.assertEqual(kept["activation"], 0.0)
        kept = trace_components.compact(candidate(activation=0.3333333))["components_graded"]
        self.assertEqual(kept["activation"], 0.3333)

    def test_a_boolean_is_not_mistaken_for_a_score(self) -> None:
        kept = trace_components.compact(candidate(lexical=True))["components_graded"]
        self.assertNotIn("lexical", kept, "True is a flag, not a graded value")

    def test_an_absent_signal_is_absent_rather_than_zero(self) -> None:
        dump = full_dump()
        dump.pop("semantic")
        inside = {"memory_id": "m2", "score": 0.3, "components": dump}
        kept = trace_components.compact(inside)["components_graded"]
        self.assertNotIn("semantic", kept, "an unmeasured signal must not appear as 0.0")

    def test_a_candidate_with_no_components_is_left_alone(self) -> None:
        bare = {"memory_id": "m3", "score": 0.1, "components_omitted": True}
        self.assertEqual(trace_components.compact(bare), bare)

    def test_the_dump_shrinks_by_most_of_its_size(self) -> None:
        import json
        before = len(json.dumps({"memory_id": "m", "score": 0.4, "components": full_dump()}))
        after = len(json.dumps(trace_components.compact(candidate())))
        # Measured 58.4% of the full dump: the saving is real but it is NOT free, because keeping a reason
        # for every candidate costs bytes. The trade is stated in the changelog rather than rounded away.
        self.assertLess(after, before * 0.62, "the reason must be cheaper than the dump it replaces")

    # -- what a reader gets back -----------------------------------------------------

    def test_why_shows_the_graded_values(self) -> None:
        lines = trace_components.render_why(trace_components.compact(candidate()))
        joined = "\n".join(lines)
        self.assertIn("lexical", joined)
        self.assertIn("score", joined)

    def test_why_says_so_when_nothing_was_recorded(self) -> None:
        lines = trace_components.render_why({"memory_id": "m4", "score": 0.2, "components_omitted": True})
        self.assertIn("not recorded", "\n".join(lines))

    def test_coverage_counts_what_can_be_explained(self) -> None:
        rows = [
            candidate("a"),                                  # full dump
            trace_components.compact(candidate("b")),        # graded only
            {"memory_id": "c", "score": 0.1, "components_omitted": True},  # nothing
        ]
        report = trace_components.coverage(rows)
        self.assertEqual(report["candidates"], 3)
        self.assertEqual(report["full_dump"], 1)
        self.assertEqual(report["graded_only"], 1)
        self.assertEqual(report["unexplained"], 1)
        self.assertEqual(report["explainable"], 2)
        self.assertAlmostEqual(report["explainable_pct"], 66.7, places=1)

    def test_coverage_of_nothing_is_zero_not_an_error(self) -> None:
        report = trace_components.coverage([])
        self.assertEqual(report["candidates"], 0)
        self.assertEqual(report["explainable_pct"], 0.0)

    def test_summarize_keeps_the_full_dump_for_the_rows_that_have_it(self) -> None:
        kept = trace_components.summarize([candidate("a"), candidate("b")], keep_full={"a"})
        self.assertIn("components", kept[0])
        self.assertIn("components_graded", kept[1])


class TrimIntegrationTests(unittest.TestCase):
    """The seam this module exists for: the store's own trace trim."""

    def test_the_trim_now_keeps_a_reason_for_the_candidates_it_shortens(self) -> None:
        from cortex.store import _trim_trace_candidate_detail

        rows = [candidate(f"m{index}", score=index / 10.0) for index in range(20)]
        rows[19]["selected"] = True
        trimmed = _trim_trace_candidate_detail(rows)
        report = trace_components.coverage(trimmed)
        self.assertGreater(report["full_dump"], 0, "selected and top-N still get the whole dump")
        self.assertGreater(report["graded_only"], 0, "the rest keep the part that explains the score")
        self.assertEqual(report["unexplained"], 0, "no candidate is left without a reason")
        self.assertEqual(report["explainable"], report["candidates"])

    def test_a_shortened_candidate_still_says_the_dump_is_gone(self) -> None:
        from cortex.store import _trim_trace_candidate_detail

        rows = [candidate(f"m{index}", score=index / 10.0) for index in range(15)]
        trimmed = _trim_trace_candidate_detail(rows)
        shortened = [row for row in trimmed if "components" not in row]
        self.assertTrue(shortened)
        self.assertTrue(all(row.get("components_omitted") for row in shortened))


if __name__ == "__main__":
    unittest.main()
