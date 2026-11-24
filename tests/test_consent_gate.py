"""What may be remembered, decided before the write (#182)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import consent_gate as cg
from cortex.store import CortexStore


class ClassifyTests(unittest.TestCase):
    def test_a_technical_fact_is_technical(self) -> None:
        self.assertEqual(cg.classify("the dashboard listens on port 8100"), "technical")

    def test_a_credential_is_a_credential(self) -> None:
        self.assertEqual(cg.classify("the api key for the bucket is in the vault"), "credential")

    def test_work_systems_are_their_own_class(self) -> None:
        self.assertEqual(cg.classify("the work laptop joins the domain at nine"), "work_systems")

    def test_finance_is_its_own_class(self) -> None:
        self.assertEqual(cg.classify("the mortgage statement arrives on the third"), "finance")

    def test_health_is_its_own_class(self) -> None:
        self.assertEqual(cg.classify("the prescription was refilled"), "health")

    def test_the_most_protective_class_wins(self) -> None:
        """A sentence about a work credential is a credential, not a work fact."""
        self.assertEqual(cg.classify("my work password for the payroll portal"), "credential")


class DecideTests(unittest.TestCase):
    def test_a_credential_is_refused(self) -> None:
        decision = cg.decide("the token is abc123")
        self.assertEqual(decision.verdict, cg.REFUSE)
        self.assertEqual(decision.class_name, "credential")

    def test_work_systems_are_refused_outright_not_asked_about(self) -> None:
        """That boundary is his, stated once; a policy engine does not re-open it per write."""
        decision = cg.decide("my employer uses a ticket system")
        self.assertEqual(decision.verdict, cg.REFUSE)
        self.assertIn("off limits", decision.reason)

    def test_finance_defaults_to_ask(self) -> None:
        self.assertEqual(cg.decide("the mortgage is due").verdict, cg.ASK)

    def test_finance_is_stored_once_consent_is_granted(self) -> None:
        decision = cg.decide("the mortgage is due", consent={"finance": True})
        self.assertEqual(decision.verdict, cg.STORE)

    def test_consent_for_one_class_does_not_authorise_another(self) -> None:
        """Authorising technical facts never authorises finance; that is the point of classes."""
        decision = cg.decide("the prescription was refilled", consent={"finance": True})
        self.assertEqual(decision.verdict, cg.ASK)

    def test_technical_content_is_stored_by_default(self) -> None:
        self.assertEqual(cg.decide("the reverse proxy sits in front of the dashboard").verdict,
                         cg.STORE)

    def test_a_refusal_never_echoes_the_content(self) -> None:
        secret = "the token is hunter2"
        decision = cg.decide(secret)
        blob = json.dumps(decision.as_dict())
        self.assertNotIn("hunter2", blob)

    def test_the_decision_carries_a_hash_not_the_text(self) -> None:
        decision = cg.decide("a private sentence about the mortgage")
        self.assertEqual(len(decision.content_hash), 16)
        self.assertNotIn("mortgage", decision.content_hash)

    def test_every_matched_class_is_listed(self) -> None:
        decision = cg.decide("my employer payroll statement and my prescription")
        self.assertIn("work_systems", decision.classes_seen)
        self.assertIn("finance", decision.classes_seen)

    def test_the_same_content_hashes_the_same(self) -> None:
        self.assertEqual(cg.decide("a fact").content_hash, cg.decide("a fact").content_hash)

    def test_an_empty_candidate_is_still_decided(self) -> None:
        self.assertEqual(cg.decide("").verdict, cg.STORE)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_a_decision_is_logged(self) -> None:
        cg.gate(self.store, "the dashboard listens on port 8100")
        self.assertEqual(cg.summary(self.store)["decisions"], 1)

    def test_the_log_holds_a_hash_not_the_content(self) -> None:
        cg.gate(self.store, "the mortgage statement arrives on the third")
        blob = json.dumps(cg.history(self.store))
        self.assertNotIn("mortgage", blob)
        self.assertIn("finance", blob)

    def test_a_refusal_is_logged_with_its_reason(self) -> None:
        cg.gate(self.store, "the password for the router is in the vault")
        rows = cg.history(self.store)
        self.assertEqual(rows[0]["verdict"], cg.REFUSE)
        self.assertIn("refused, not scrubbed", rows[0]["reason"])

    def test_deciding_twice_updates_rather_than_duplicates(self) -> None:
        cg.gate(self.store, "the dashboard listens on port 8100")
        cg.gate(self.store, "the dashboard listens on port 8100")
        self.assertEqual(cg.summary(self.store)["decisions"], 1)

    def test_the_summary_groups_by_verdict(self) -> None:
        cg.gate(self.store, "the dashboard listens on port 8100")
        cg.gate(self.store, "the token is abc123")
        described = cg.summary(self.store)
        self.assertEqual(described["by_verdict"].get(cg.REFUSE), 1)
        self.assertEqual(described["by_verdict"].get(cg.STORE), 1)

    def test_the_summary_says_the_log_holds_hashes(self) -> None:
        cg.gate(self.store, "the dashboard listens on port 8100")
        self.assertIn("hashes, not content", cg.summary(self.store)["note"])

    def test_an_empty_log_says_nothing_writes_through_the_gate(self) -> None:
        """An empty decision log is a fact about coverage, not a clean bill of health."""
        self.assertIn("nothing writes through it yet", cg.summary(self.store)["note"])

    def test_the_classifier_is_documented_as_a_keyword_screen(self) -> None:
        self.assertIn("keyword screen", cg.classify.__doc__ or "")

    def test_the_render_lists_decisions(self) -> None:
        cg.gate(self.store, "the dashboard listens on port 8100")
        rendered = "\n".join(cg.render(self.store))
        self.assertIn("consent decisions:", rendered)
        self.assertIn(cg.STORE, rendered)

    def test_the_gate_does_not_write_a_memory(self) -> None:
        """The gate decides; the caller owns the write. A gate that also stores is not a gate."""
        cg.gate(self.store, "the dashboard listens on port 8100")
        rows = self.store._conn.execute("SELECT COUNT(*) AS count FROM memories").fetchone()
        self.assertEqual(int(rows["count"]), 0)

    def test_a_broken_store_logs_nothing_and_does_not_raise(self) -> None:
        class Broken:
            _conn = None

        self.assertEqual(cg.summary(Broken())["decisions"], 0)
        self.assertFalse(cg.record_decision(Broken(), cg.decide("a fact")))


if __name__ == "__main__":
    unittest.main()
