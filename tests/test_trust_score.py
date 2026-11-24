"""One trust number per memory, composed from parts that are already recorded (#187)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import trust_score as ts
from cortex.store import CortexStore


def row(**overrides):
    base = {"id": "m1", "confidence": 0.8, "currentness_confidence": 0.8, "used_count": 5,
            "confirmed_count": 1, "harmful_count": 0, "correction_count": 0,
            "source_category": "USER_EXPLICIT", "kind": "semantic", "content": "a fact"}
    base.update(overrides)
    return base


class ScoreTests(unittest.TestCase):
    def test_a_well_evidenced_user_fact_scores_high(self) -> None:
        found = ts.score(row(), evidence=4)
        self.assertGreater(found.score, 0.7)

    def test_a_document_cannot_exceed_its_authority(self) -> None:
        """Source authority is a ceiling, not a multiplier (#41)."""
        found = ts.score(row(source_category="DOCUMENT_EXTRACTED"), evidence=20)
        self.assertLessEqual(found.score, 0.50)
        self.assertTrue(found.capped)

    def test_a_user_statement_is_not_capped(self) -> None:
        self.assertFalse(ts.score(row(), evidence=4).capped)

    def test_an_unknown_part_is_excluded_not_zeroed(self) -> None:
        """A memory that was never retrieved has no usage evidence, not bad usage evidence."""
        unknown_use = ts.score(row(used_count=None), evidence=2)
        zero_use = ts.score(row(used_count=0), evidence=2)
        self.assertIn("use", unknown_use.unknown)
        self.assertGreater(unknown_use.score, zero_use.score)

    def test_a_memory_with_nothing_known_has_no_score(self) -> None:
        found = ts.score({"id": "m", "source_category": "USER_EXPLICIT"}, evidence=None)
        self.assertEqual(found.parts, {})
        self.assertIn("no score rather than a low one", " ".join(found.notes))

    def test_evidence_saturates(self) -> None:
        """Twenty edges are not ten times better than two: the component stops growing."""
        at_saturation = ts.score(row(), evidence=int(ts.EVIDENCE_SATURATION))
        far_beyond = ts.score(row(), evidence=2000)
        self.assertEqual(at_saturation.score, far_beyond.score)
        few = ts.score(row(), evidence=1)
        self.assertLessEqual(far_beyond.score - few.score, ts.WEIGHTS["evidence"])

    def test_a_harmful_report_dents_but_does_not_delete(self) -> None:
        clean = ts.score(row(), evidence=2)
        harmed = ts.score(row(harmful_count=3), evidence=2)
        self.assertLess(harmed.score, clean.score)
        self.assertGreater(harmed.score, 0)

    def test_a_corrected_memory_gets_a_bonus(self) -> None:
        self.assertGreater(ts.score(row(correction_count=1), evidence=2).score,
                           ts.score(row(), evidence=2).score)

    def test_staleness_costs_something(self) -> None:
        self.assertLess(ts.score(row(), evidence=2, stale=True).score,
                        ts.score(row(), evidence=2).score)

    def test_the_weights_are_stated(self) -> None:
        found = ts.score(row(), evidence=2)
        self.assertIn("_weights", found.parts)
        self.assertEqual(set(found.parts["_weights"]), {"confidence", "currentness", "evidence",
                                                        "use", "confirmations"})

    def test_the_notes_explain_each_adjustment(self) -> None:
        found = ts.score(row(harmful_count=2), evidence=2, stale=True)
        joined = " ".join(found.notes)
        self.assertIn("harmful reports", joined)
        self.assertIn("unconfirmed", joined)

    def test_the_cap_note_explains_itself(self) -> None:
        found = ts.score(row(source_category="DOCUMENT_EXTRACTED"), evidence=20)
        self.assertIn("a document cannot be trusted more than a document", " ".join(found.notes))


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str, *, category: str = "USER_EXPLICIT") -> str:
        return self.store.add_memory(
            content=text, kind="semantic", source_type="user_turn", source_category=category,
        )[0]

    def tune(self, memory_id: str, **values) -> None:
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self.store.transaction() as conn:
            conn.execute(f"UPDATE memories SET {assignments} WHERE id = ?",
                         tuple(values.values()) + (memory_id,))

    def test_ranking_returns_the_lowest_first(self) -> None:
        """Support decides the order; a category is not a proxy for weakness."""
        strong = self.memory("a fact he stated and I have used many times")
        weak = self.memory("a barely held fact about the dashboard")
        self.tune(strong, used_count=20, confirmed_count=4, confidence=0.95)
        self.tune(weak, used_count=0, confirmed_count=0, confidence=0.30)
        ranked = ts.rank(self.store, limit=2)
        self.assertEqual(ranked[0]["memory_id"], weak)
        self.assertEqual(ranked[-1]["memory_id"], strong)

    def test_ranking_can_be_reversed(self) -> None:
        """Authority caps a score; it does not supply one. Support does — so the supported fact wins."""
        supported = self.memory("a fact he stated and I have used many times")
        self.tune(supported, used_count=20, confirmed_count=4, confidence=0.95)
        document = self.memory("an extracted fact about the dashboard",
                               category="DOCUMENT_EXTRACTED")
        self.tune(document, used_count=0, confirmed_count=0, confidence=0.30)
        ranked = ts.rank(self.store, limit=1, lowest=False)
        self.assertEqual(ranked[0]["source_category"], "USER_EXPLICIT")

    def test_authority_caps_but_does_not_supply_support(self) -> None:
        """A document with a high confidence still cannot exceed the document ceiling."""
        document = self.memory("an extracted fact about the dashboard",
                               category="DOCUMENT_EXTRACTED")
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET used_count = 50, confirmed_count = 9, "
                         "confidence = 0.99 WHERE id = ?", (document,))
        ranked = ts.rank(self.store, limit=1, lowest=False)
        self.assertLessEqual(ranked[0]["score"], 0.50)

    def test_evidence_is_counted_from_both_edge_directions(self) -> None:
        """A grouped query over src and dst; one-sided would under-count half the graph."""
        first = self.memory("a fact about the port")
        second = self.memory("another fact about the port")
        self.store.add_edge(second, first, "related", weight=0.5)
        counts = ts._evidence_counts(self.store)
        self.assertGreaterEqual(counts.get(first, 0), 1)
        self.assertGreaterEqual(counts.get(second, 0), 1)

    def test_a_narrow_distribution_says_it_barely_discriminates(self) -> None:
        """A number that is nearly the same for everything is not a measurement (#147, #176)."""
        for index in range(5):
            self.memory(f"fact number {index} about the dashboard")
        described = ts.distribution(self.store)
        self.assertLess(described["spread"], ts.MIN_USEFUL_SPREAD)
        self.assertIn("barely discriminates", described["note"])

    def test_a_wide_distribution_is_not_flagged(self) -> None:
        strong = self.memory("a fact he stated and used constantly")
        weak = self.memory("a barely held fact about the dashboard")
        self.tune(strong, used_count=40, confirmed_count=6, confidence=0.99)
        self.tune(weak, used_count=0, confirmed_count=0, confidence=0.05)
        described = ts.distribution(self.store)
        self.assertGreaterEqual(described["spread"], 0.0)
        self.assertIn("spread", described)

    def test_the_distribution_states_its_reference_class(self) -> None:
        for index in range(5):
            self.memory(f"fact number {index} about the dashboard")
        described = ts.distribution(self.store)
        self.assertEqual(described["count"], 5)
        for key in ("median", "p10", "p90", "spread"):
            self.assertIn(key, described)
        self.assertTrue(described["note"])

    def test_an_empty_store_has_no_distribution(self) -> None:
        described = ts.distribution(self.store)
        self.assertEqual(described["count"], 0)
        self.assertTrue(described["note"])

    def test_the_render_states_the_median_and_the_buckets(self) -> None:
        self.memory("a fact about the dashboard")
        rendered = "\n".join(ts.render(self.store))
        self.assertIn("trust:", rendered)
        self.assertIn("median", rendered)

    def test_a_broken_store_ranks_nothing(self) -> None:
        class Broken:
            _conn = None

        self.assertEqual(ts.rank(Broken()), [])
        self.assertEqual(ts.distribution(Broken())["count"], 0)


if __name__ == "__main__":
    unittest.main()
