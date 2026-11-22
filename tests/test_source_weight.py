"""A source's authority caps the confidence it can carry (#41)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import source_weight
from cortex.store import CortexStore


class AuthorityTests(unittest.TestCase):
    def test_the_ladder_orders_user_above_agent_above_web(self) -> None:
        self.assertGreater(source_weight.authority("USER_EXPLICIT"),
                           source_weight.authority("AGENT_INFERENCE"))
        self.assertGreater(source_weight.authority("AGENT_INFERENCE"),
                           source_weight.authority("WEB_CLAIM"))

    def test_a_correction_outranks_a_statement(self) -> None:
        self.assertGreater(source_weight.authority("USER_CORRECTION"),
                           source_weight.authority("USER_STATED"))

    def test_an_unknown_source_gets_the_middle_not_the_top(self) -> None:
        self.assertEqual(source_weight.authority("SOMETHING_NEW"), source_weight.AUTHORITY["UNKNOWN"])
        self.assertLess(source_weight.authority("SOMETHING_NEW"),
                        source_weight.authority("USER_EXPLICIT"))

    def test_an_empty_category_falls_back_to_the_source_type(self) -> None:
        self.assertEqual(source_weight.authority("", "web_fetch"), 0.45)

    def test_both_empty_is_unknown(self) -> None:
        self.assertEqual(source_weight.authority(None, None), source_weight.AUTHORITY["UNKNOWN"])

    def test_user_categories_are_recognised(self) -> None:
        self.assertTrue(source_weight.is_user("user_explicit"))
        self.assertFalse(source_weight.is_user("AGENT_PROPOSED"))

    def test_the_ladder_is_readable_highest_first(self) -> None:
        ladder = source_weight.ladder()
        self.assertEqual(ladder[0][1], max(value for _name, value in ladder))
        self.assertEqual(ladder[0][0], "USER_CORRECTION")


class ApplyTests(unittest.TestCase):
    def test_a_web_claim_above_its_ceiling_is_capped(self) -> None:
        weighted = source_weight.apply(0.95, "WEB_CLAIM")
        self.assertEqual(weighted.confidence, 0.45)
        self.assertTrue(weighted.capped)

    def test_a_confident_user_statement_passes_through(self) -> None:
        weighted = source_weight.apply(0.80, "USER_EXPLICIT")
        self.assertEqual(weighted.confidence, 0.80)
        self.assertFalse(weighted.capped)

    def test_the_ceiling_never_raises_confidence(self) -> None:
        weighted = source_weight.apply(0.30, "USER_CORRECTION")
        self.assertEqual(weighted.confidence, 0.30)
        self.assertFalse(weighted.capped)

    def test_an_agent_inference_cannot_exceed_its_ceiling(self) -> None:
        self.assertEqual(source_weight.confidence(0.99, "AGENT_PROPOSED"), 0.50)

    def test_values_are_clamped_to_the_unit_interval(self) -> None:
        self.assertEqual(source_weight.confidence(5.0, "USER_EXPLICIT"), 0.92)
        self.assertEqual(source_weight.confidence(-1.0, "USER_EXPLICIT"), 0.0)

    def test_a_malformed_confidence_is_treated_as_zero(self) -> None:
        self.assertEqual(source_weight.confidence("nonsense", "USER_EXPLICIT"), 0.0)

    def test_the_explanation_names_the_ceiling_and_the_stated_value(self) -> None:
        described = source_weight.apply(0.9, "WEB_CLAIM").describe()
        self.assertIn("capped from 0.900", described)
        self.assertIn("0.45 ceiling", described)

    def test_an_uncapped_explanation_does_not_pretend_otherwise(self) -> None:
        self.assertIn("below the", source_weight.apply(0.4, "WEB_CLAIM").describe())

    def test_a_source_type_carries_the_ceiling_when_no_category_does(self) -> None:
        self.assertEqual(source_weight.confidence(0.9, None, source_type="web_fetch"), 0.45)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_the_report_counts_memories_by_source(self) -> None:
        for index, category in enumerate(("USER_EXPLICIT", "USER_EXPLICIT", "AGENT_PROPOSED")):
            self.store.add_memory(
                content=f"Fact number {index} from {category}.",
                kind="semantic",
                source_type="user_turn",
                source_category=category,
            )
        described = source_weight.report(self.store)
        self.assertEqual(described["total"], 3)
        self.assertEqual(described["by_source"]["USER_EXPLICIT"], 2)
        self.assertEqual(described["ceilings"]["AGENT_PROPOSED"], 0.50)


class OverCapTests(unittest.TestCase):
    """The point of measuring: how much stored confidence is above its source's ceiling."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_a_proposal_stored_optimistically_is_measurable(self) -> None:
        self.store.add_memory(
            content="An agent thinks the port is 8100.",
            kind="semantic",
            source_type="agent_tool_proposal",
            source_category="AGENT_PROPOSED",
            confidence=0.88,
        )
        rows = self.store._conn.execute(
            "SELECT source_category, confidence FROM memories"
        ).fetchall()
        over = [
            row for row in rows
            if float(row["confidence"]) > source_weight.authority(str(row["source_category"]))
        ]
        self.assertEqual(len(over), 1)


if __name__ == "__main__":
    unittest.main()
