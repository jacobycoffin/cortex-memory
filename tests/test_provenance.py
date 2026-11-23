"""For each fact, the sequence of evidence that produced it (#51)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import provenance
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class LinkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str, **kwargs) -> str:
        kwargs.setdefault("kind", "semantic")
        kwargs.setdefault("source_type", "user_turn")
        kwargs.setdefault("source_category", "USER_EXPLICIT")
        return self.store.add_memory(content=content, **kwargs)[0]

    def test_a_link_is_recorded(self) -> None:
        memory_id = self.memory("The dashboard runs on port 8100.")
        step = provenance.link(self.store, memory_id, "verified", detail="checked by tool", at=NOW)
        self.assertIsNotNone(step)
        self.assertEqual(step["kind"], "verified")

    def test_an_unknown_step_kind_becomes_unknown(self) -> None:
        memory_id = self.memory("The dashboard runs on port 8100.")
        step = provenance.link(self.store, memory_id, "made-up", at=NOW)
        self.assertEqual(step["kind"], "unknown")

    def test_an_empty_memory_id_links_nothing(self) -> None:
        self.assertIsNone(provenance.link(self.store, "", "verified"))

    def test_linking_never_raises_into_a_write_path(self) -> None:
        broken = type("Broken", (), {"transaction": property(lambda self: 1 / 0)})()
        self.assertIsNone(provenance.link(broken, "m", "verified"))


class ChainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str, **kwargs) -> str:
        kwargs.setdefault("kind", "semantic")
        kwargs.setdefault("source_type", "user_turn")
        kwargs.setdefault("source_category", "USER_EXPLICIT")
        return self.store.add_memory(content=content, **kwargs)[0]

    def test_every_fact_has_an_origin_step(self) -> None:
        memory_id = self.memory("The dashboard runs on port 8100.")
        chain = provenance.chain(self.store, memory_id)
        self.assertEqual(chain.first["kind"], "captured")

    def test_the_origin_names_the_source(self) -> None:
        memory_id = self.memory("The dashboard runs on port 8100.")
        chain = provenance.chain(self.store, memory_id)
        self.assertIn("USER_EXPLICIT", chain.first["detail"])

    def test_a_tool_verified_fact_has_a_verified_origin(self) -> None:
        memory_id = self.memory("The dashboard runs on port 8100.",
                                source_category="TOOL_VERIFIED")
        self.assertEqual(provenance.chain(self.store, memory_id).first["kind"], "verified")

    def test_linked_steps_appear_in_time_order(self) -> None:
        memory_id = self.memory("The dashboard runs on port 8100.")
        provenance.link(self.store, memory_id, "corroborated", detail="second source",
                        at=NOW + timedelta(minutes=5))
        provenance.link(self.store, memory_id, "verified", detail="checked",
                        at=NOW + timedelta(minutes=1))
        kinds = [step["kind"] for step in provenance.chain(self.store, memory_id).steps]
        self.assertEqual(kinds, ["captured", "verified", "corroborated"])

    def test_a_correction_appears_as_corrected(self) -> None:
        from cortex import self_correction

        memory_id = self.memory("The dashboard runs on port 8100 behind nginx.")
        result = self_correction.intercept(
            self.store, "I meant 8300.", target_id=memory_id, at=NOW
        )
        chain = provenance.chain(self.store, result["memory_id"])
        self.assertTrue(chain.of("corrected"))

    def test_a_merge_appears_as_merged(self) -> None:
        from cortex import merge_review

        left = self.memory("The dashboard runs on port 8100 behind nginx.")
        right = self.memory("Nginx proxies the dashboard on port 8100.")
        proposal = merge_review.propose(self.store, left, right,
                                       merged_text="The dashboard runs on 8100 behind nginx.")
        merge_review.apply(self.store, proposal, consent=True, consented_by="jacoby", at=NOW)
        self.assertTrue(provenance.chain(self.store, left).of("merged"))

    def test_a_superseding_fact_shows_what_it_replaced(self) -> None:
        old = self.memory("The dashboard runs on port 8100.")
        new = self.memory("The dashboard runs on port 8200.")
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET supersedes_id = ? WHERE id = ?", (old, new))
        chain = provenance.chain(self.store, new)
        self.assertTrue(chain.of("superseded"))

    def test_independent_evidence_shows_as_corroboration(self) -> None:
        one = self.memory("The dashboard runs on port 8100 behind nginx.")
        two = self.memory("Nginx proxies the dashboard on port 8100.")
        self.store.add_edge(one, two, "co_observed", weight=0.5,
                            evidence_type="co_retrieval", evidence_key="k1",
                            explanation="seen together")
        chain = provenance.chain(self.store, one)
        self.assertTrue(chain.of("corroborated"))

    def test_the_chain_reads_as_a_sentence(self) -> None:
        memory_id = self.memory("The dashboard runs on port 8100.")
        provenance.link(self.store, memory_id, "verified", detail="checked", at=NOW)
        sentence = provenance.chain(self.store, memory_id).describe()
        self.assertIn("captured", sentence)
        self.assertIn("verified", sentence)
        self.assertIn("->", sentence)

    def test_a_fact_with_only_its_origin_is_still_a_chain(self) -> None:
        memory_id = self.memory("The dashboard runs on port 8100.")
        self.assertEqual(len(provenance.chain(self.store, memory_id).steps), 1)

    def test_an_unknown_memory_says_so(self) -> None:
        chain = provenance.chain(self.store, "no-such-id")
        self.assertTrue(chain.missing)
        self.assertEqual(chain.describe(), "no such memory")

    def test_the_chain_round_trips(self) -> None:
        memory_id = self.memory("The dashboard runs on port 8100.")
        payload = provenance.chain(self.store, memory_id).as_dict()
        for key in ("memory_id", "steps", "missing", "sentence"):
            self.assertIn(key, payload)

    def test_the_report_counts_steps_by_kind(self) -> None:
        memory_id = self.memory("The dashboard runs on port 8100.")
        provenance.link(self.store, memory_id, "verified", at=NOW)
        provenance.link(self.store, memory_id, "corroborated", at=NOW)
        described = provenance.report(self.store)
        self.assertEqual(described["by_kind"], {"verified": 1, "corroborated": 1})
        self.assertGreaterEqual(described["memories"], 1)

    def test_an_empty_store_is_quiet(self) -> None:
        self.assertEqual(provenance.report(self.store)["steps"], 0)


if __name__ == "__main__":
    unittest.main()
