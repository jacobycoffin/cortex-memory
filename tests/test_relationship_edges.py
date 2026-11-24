"""Typed, evidenced relations between two entities (#62)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import aliases, relationship_edges
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class LinkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def name(self, text: str) -> None:
        """Make an entity known so proposal detection can find it."""
        aliases.add(self.store, text, text, source="manual")

    def test_a_typed_relation_with_evidence_is_recorded(self) -> None:
        made = relationship_edges.link(self.store, "jacoby", "r630", "owns",
                                       evidence="he said so", at=NOW)
        self.assertTrue(made["linked"])
        self.assertEqual(len(relationship_edges.relations_of(self.store, "jacoby")), 1)

    def test_an_untyped_relation_is_refused(self) -> None:
        made = relationship_edges.link(self.store, "jacoby", "r630", "related to",
                                       evidence="vibes")
        self.assertFalse(made["linked"])
        self.assertIn("must be one of", made["reason"])

    def test_a_relation_needs_evidence_or_a_person(self) -> None:
        made = relationship_edges.link(self.store, "jacoby", "r630", "owns")
        self.assertFalse(made["linked"])
        self.assertIn("evidence", made["reason"])

    def test_a_person_asserting_it_is_enough(self) -> None:
        made = relationship_edges.link(self.store, "jacoby", "r630", "owns", asserted_by="jacoby")
        self.assertTrue(made["linked"])

    def test_a_relation_needs_two_different_entities(self) -> None:
        self.assertFalse(relationship_edges.link(self.store, "jacoby", "jacoby", "owns",
                                                 evidence="x")["linked"])

    def test_reasserting_reinforces_instead_of_duplicating(self) -> None:
        relationship_edges.link(self.store, "jacoby", "r630", "owns", evidence="first", at=NOW)
        again = relationship_edges.link(self.store, "jacoby", "r630", "owns", evidence="second", at=NOW)
        self.assertTrue(again["reinforced"])
        relations = relationship_edges.relations_of(self.store, "jacoby")
        self.assertEqual(len(relations), 1)
        # reinforcement is capped at 1.0, so from a fresh edge the weight is already at the ceiling
        self.assertEqual(relations[0].weight, 1.0)
        self.assertIn("second", relations[0].evidence)

    def test_the_weight_is_capped(self) -> None:
        for _ in range(20):
            relationship_edges.link(self.store, "jacoby", "r630", "owns", evidence="again", at=NOW)
        self.assertLessEqual(relationship_edges.relations_of(self.store, "jacoby")[0].weight, 1.0)

    def test_a_symmetric_relation_is_stored_in_canonical_order(self) -> None:
        relationship_edges.link(self.store, "zed", "abel", "works_with", evidence="pair", at=NOW)
        relations = relationship_edges.relations_of(self.store, "zed")
        self.assertEqual(relations[0].source, "abel")
        self.assertEqual(relations[0].target, "zed")

    def test_a_symmetric_relation_cannot_appear_twice_swapped(self) -> None:
        relationship_edges.link(self.store, "zed", "abel", "works_with", evidence="pair", at=NOW)
        relationship_edges.link(self.store, "abel", "zed", "works_with", evidence="pair", at=NOW)
        self.assertEqual(len(relationship_edges.relations_of(self.store, "zed")), 1)

    def test_an_asymmetric_relation_keeps_its_direction(self) -> None:
        relationship_edges.link(self.store, "jarvis", "cortex", "manages", evidence="it does", at=NOW)
        relation = relationship_edges.relations_of(self.store, "cortex")[0]
        self.assertEqual((relation.source, relation.target), ("jarvis", "cortex"))
        self.assertIn("->", relation.triple())

    def test_a_relation_is_visible_from_both_ends(self) -> None:
        relationship_edges.link(self.store, "jacoby", "r630", "owns", evidence="x", at=NOW)
        self.assertEqual(len(relationship_edges.relations_of(self.store, "r630")), 1)

    def test_between_finds_a_direct_relation_either_way(self) -> None:
        relationship_edges.link(self.store, "jacoby", "r630", "owns", evidence="x", at=NOW)
        self.assertTrue(relationship_edges.between(self.store, "r630", "jacoby"))
        self.assertEqual(relationship_edges.between(self.store, "r630", "ct116"), [])

    def test_closing_keeps_the_row(self) -> None:
        relationship_edges.link(self.store, "jacoby", "r630", "owns", evidence="x", at=NOW)
        closed = relationship_edges.close(self.store, "jacoby", "r630", "owns", at=NOW)
        self.assertTrue(closed["closed"])
        self.assertEqual(relationship_edges.relations_of(self.store, "jacoby"), [])
        history = relationship_edges.relations_of(self.store, "jacoby", include_closed=True)
        self.assertEqual(len(history), 1)
        self.assertFalse(history[0].open)

    def test_closing_twice_changes_nothing(self) -> None:
        relationship_edges.link(self.store, "jacoby", "r630", "owns", evidence="x", at=NOW)
        relationship_edges.close(self.store, "jacoby", "r630", "owns", at=NOW)
        self.assertFalse(relationship_edges.close(self.store, "jacoby", "r630", "owns", at=NOW)["closed"])

    def test_reasserting_reopens_a_closed_relation(self) -> None:
        relationship_edges.link(self.store, "jacoby", "r630", "owns", evidence="x", at=NOW)
        relationship_edges.close(self.store, "jacoby", "r630", "owns", at=NOW)
        relationship_edges.link(self.store, "jacoby", "r630", "owns", evidence="again", at=NOW)
        self.assertEqual(len(relationship_edges.relations_of(self.store, "jacoby")), 1)

    def test_a_proposal_needs_two_known_names(self) -> None:
        self.name("jacoby")
        self.assertEqual(relationship_edges.propose(self.store, "Jacoby owns something."), [])

    def test_a_proposal_finds_the_relation_between_two_names(self) -> None:
        self.name("jacoby")
        self.name("r630")
        proposals = relationship_edges.propose(self.store, "Jacoby owns the r630 host.")
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["relation"], "owns")
        self.assertEqual(proposals[0]["source"], "jacoby")
        self.assertEqual(proposals[0]["target"], "r630")

    def test_a_proposal_writes_nothing(self) -> None:
        self.name("jacoby")
        self.name("r630")
        relationship_edges.propose(self.store, "Jacoby owns the r630 host.")
        self.assertEqual(relationship_edges.relations_of(self.store, "jacoby"), [])

    def test_a_sentence_with_no_relation_phrase_proposes_nothing(self) -> None:
        self.name("jacoby")
        self.name("r630")
        self.assertEqual(relationship_edges.propose(self.store, "Jacoby rebooted the r630 host."), [])

    def test_a_recorded_proposal_carries_the_sentence_as_its_evidence(self) -> None:
        """The sentence *is* the evidence, which is why recording it needs no separate asserter."""
        self.name("jacoby")
        self.name("r630")
        written = relationship_edges.record_proposals(self.store, "Jacoby owns the r630 host.")
        self.assertTrue(all(item["linked"] is True for item in written))
        relation = relationship_edges.relations_of(self.store, "jacoby")[0]
        self.assertIn("owns the r630", relation.evidence)

    def test_an_asserted_proposal_records_who_asserted_it(self) -> None:
        self.name("jacoby")
        self.name("r630")
        relationship_edges.record_proposals(self.store, "Jacoby owns the r630 host.",
                                            asserted_by="jacoby")
        self.assertEqual(relationship_edges.relations_of(self.store, "jacoby")[0].asserted_by,
                         "jacoby")

    def test_a_relation_describes_itself_when_open(self) -> None:
        relationship_edges.link(self.store, "jacoby", "r630", "owns", evidence="he said so", at=NOW)
        described = relationship_edges.relations_of(self.store, "jacoby")[0].describe()
        self.assertIn("[owns]", described)
        self.assertIn("he said so", described)

    def test_a_relation_describes_itself_when_closed(self) -> None:
        relationship_edges.link(self.store, "jacoby", "r630", "owns", evidence="x", at=NOW)
        relationship_edges.close(self.store, "jacoby", "r630", "owns", at=NOW)
        described = relationship_edges.relations_of(self.store, "jacoby", include_closed=True)[0]
        self.assertIn("closed", described.describe())

    def test_the_report_counts_by_relation(self) -> None:
        relationship_edges.link(self.store, "jacoby", "r630", "owns", evidence="x", at=NOW)
        relationship_edges.link(self.store, "jarvis", "cortex", "manages", evidence="y", at=NOW)
        described = relationship_edges.report(self.store)
        self.assertEqual(described["relations"], 2)
        self.assertEqual(described["by_relation"], {"owns": 1, "manages": 1})

    def test_the_round_trip(self) -> None:
        relationship_edges.link(self.store, "jacoby", "r630", "owns", evidence="x", at=NOW)
        payload = relationship_edges.relations_of(self.store, "jacoby")[0].as_dict()
        for key in ("source", "target", "relation", "evidence", "weight", "open", "triple"):
            self.assertIn(key, payload)

    def test_an_empty_store_is_quiet(self) -> None:
        self.assertEqual(relationship_edges.relations_of(self.store, "jacoby"), [])
        self.assertEqual(relationship_edges.report(self.store)["relations"], 0)
        self.assertEqual(relationship_edges.propose(self.store, "anything at all"), [])


if __name__ == "__main__":
    unittest.main()
