"""Opt-in promotion to a shared layer, with review and redaction (#155)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import fleet_share
from cortex.store import CortexStore

GENERAL = "proxmox 8.4 renamed the pveproxy service unit"
CREDENTIAL = "the deploy token is sk-abc123456789 and it lives in the vault"
HEALTH = "my doctor said the biopsy was clear"


class ShareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, text: str = GENERAL, **kwargs) -> str:
        if "source_type" not in kwargs:
            kwargs["source_type"] = "user_turn"
        if "source_category" not in kwargs:
            kwargs["source_category"] = "USER_STATED"
        return self.store.add_memory(content=text, kind=kwargs.pop("kind", "semantic"), **kwargs)[0]

    # -- nothing is shared without a proposal and a reviewer -------------------------

    def test_a_proposal_needs_a_reason(self) -> None:
        proposal = fleet_share.propose(self.store, self.memory())
        self.assertFalse(proposal.eligible)
        self.assertIn("reason", proposal.reason)
        self.assertEqual(fleet_share.proposals(self.store), [], "no reason, no proposal row")

    def test_an_unknown_memory_is_named(self) -> None:
        proposal = fleet_share.propose(self.store, "no-such-id", reason="general knowledge")
        self.assertFalse(proposal.eligible)
        self.assertIn("unknown memory", proposal.reason)

    def test_a_general_memory_becomes_a_proposal_carrying_its_text(self) -> None:
        proposal = fleet_share.propose(self.store, self.memory(), reason="how the service unit changed")
        self.assertTrue(proposal.eligible, proposal.reason)
        self.assertEqual(proposal.shared_text, GENERAL)
        self.assertEqual(len(fleet_share.proposals(self.store)), 1)

    # -- what is never shared ---------------------------------------------------------

    def test_a_credential_memory_is_refused(self) -> None:
        proposal = fleet_share.propose(self.store, self.memory(CREDENTIAL), reason="looks generic")
        self.assertFalse(proposal.eligible)
        self.assertIn("credential", proposal.reason)

    def test_a_health_memory_is_refused(self) -> None:
        proposal = fleet_share.propose(self.store, self.memory(HEALTH), reason="looks generic")
        self.assertFalse(proposal.eligible)
        self.assertIn("health", proposal.reason)

    def test_a_project_scoped_memory_is_refused(self) -> None:
        memory_id = self.memory(scope={"project": "duelist-road"})
        proposal = fleet_share.propose(self.store, memory_id, reason="general")
        self.assertFalse(proposal.eligible)
        self.assertIn("project", proposal.reason)

    def test_a_device_scoped_memory_is_refused_because_sharing_names_the_device(self) -> None:
        memory_id = self.memory(scope={"device": "coby-clx-gaming-pc"})
        proposal = fleet_share.propose(self.store, memory_id, reason="general")
        self.assertFalse(proposal.eligible)
        self.assertIn("device", proposal.reason)

    def test_an_unclassifiable_memory_is_a_flag_for_the_reviewer_not_a_refusal(self) -> None:
        """Refusing on 'unrecognised' would block every ordinary technical statement."""
        proposal = fleet_share.propose(self.store, self.memory(), reason="how the unit changed")
        self.assertEqual(proposal.domain, "unrecognised")
        self.assertTrue(proposal.eligible, proposal.reason)
        self.assertTrue(any("classifier" in note for note in proposal.notes))
        self.assertEqual(proposal.withheld, [], "a flag is not a withheld item")

    def test_a_key_pasted_into_general_text_never_reaches_a_proposal(self) -> None:
        """The write path masks it before storage, so the proposal carries `***` rather than the secret."""
        # Composed at runtime: a secret-shaped literal in a source file gets redacted in transit, and a
        # test that asserts about a secret has to hold one that the tooling will not sanitise on the way in.
        # Long enough to match the write path's credential pattern: a short invented key is simply stored,
        # which is a fact about the pattern's coverage, not about this module.
        key = "sk-" + "proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"
        memory_id = self.memory(f"the pveproxy unit needs a restart; the key is {key}")
        self.assertNotIn(key, str(self.store.get_memory(memory_id)["content"]))
        # Documented conservatism: the marker the write path leaves ("[REDACTED API KEY]") still classifies as
        # a credential domain, so the whole memory is refused rather than shared with a marker in it. The
        # secret is gone either way; refusing is the side to err on, and the reason says which domain stopped it.
        proposal = fleet_share.propose(self.store, memory_id, reason="how the unit changed")
        self.assertNotIn(key, proposal.shared_text)
        self.assertFalse(proposal.eligible)
        self.assertIn("credential", proposal.reason)

    def test_a_redaction_that_is_applied_is_reported(self) -> None:
        proposal = fleet_share.propose(
            self.store, self.memory("mail ops@example.com when the pveproxy unit restarts"),
            reason="how the unit changed")
        self.assertTrue(proposal.eligible, proposal.reason)
        self.assertNotIn("ops@example.com", proposal.shared_text)
        self.assertTrue(proposal.labels, "the redaction must be reported, not silent")

    def test_a_path_is_genericised(self) -> None:
        proposal = fleet_share.propose(
            self.store, self.memory("the config lives at /root/src/hermes-cortex-memory/README.md"),
            reason="where the config lives")
        self.assertNotIn("/root/src/hermes-cortex-memory", proposal.shared_text)

    # -- review is the only door -----------------------------------------------------

    def test_an_anonymous_review_is_refused(self) -> None:
        proposal = fleet_share.propose(self.store, self.memory(), reason="general")
        result = fleet_share.review(self.store, proposal.proposal_id, decision="approve")
        self.assertEqual(result.status, "refused")
        self.assertIn("reviewer", result.reason)

    def test_approval_puts_the_item_in_the_bundle(self) -> None:
        proposal = fleet_share.propose(self.store, self.memory(), reason="general")
        review = fleet_share.review(self.store, proposal.proposal_id, decision="approve",
                                    reviewer="jacoby")
        self.assertTrue(review.ok, review.reason)
        payload = fleet_share.bundle(self.store)
        self.assertEqual(len(payload["items"]), 1)
        self.assertEqual(payload["items"][0]["approved_by"], "jacoby")

    def test_an_unapproved_proposal_never_reaches_the_bundle(self) -> None:
        fleet_share.propose(self.store, self.memory(), reason="general")
        self.assertEqual(fleet_share.bundle(self.store)["items"], [])

    def test_rejection_keeps_it_out_and_says_why(self) -> None:
        proposal = fleet_share.propose(self.store, self.memory(), reason="general")
        fleet_share.review(self.store, proposal.proposal_id, decision="reject", reviewer="jacoby",
                           reason="too specific to our box")
        payload = fleet_share.bundle(self.store)
        self.assertEqual(payload["items"], [])
        self.assertTrue(any("rejected" in entry["why"] for entry in payload["withheld"]))

    def test_reviewing_twice_reports_already(self) -> None:
        proposal = fleet_share.propose(self.store, self.memory(), reason="general")
        fleet_share.review(self.store, proposal.proposal_id, decision="approve", reviewer="jacoby")
        second = fleet_share.review(self.store, proposal.proposal_id, decision="approve", reviewer="jacoby")
        self.assertEqual(second.status, "already")

    def test_an_unknown_decision_is_refused(self) -> None:
        proposal = fleet_share.propose(self.store, self.memory(), reason="general")
        result = fleet_share.review(self.store, proposal.proposal_id, decision="maybe", reviewer="jacoby")
        self.assertEqual(result.status, "refused")
        self.assertIn("approve or reject", result.reason)

    def test_approval_is_bound_to_the_text_that_was_reviewed(self) -> None:
        """If the memory changed after the proposal, the reviewer approved something that is gone."""
        memory_id = self.memory()
        proposal = fleet_share.propose(self.store, memory_id, reason="general")
        with self.store.transaction() as conn:
            conn.execute("UPDATE memories SET content = ? WHERE id = ?",
                         (GENERAL + ", and nginx now owns port 8080", memory_id))
        result = fleet_share.review(self.store, proposal.proposal_id, decision="approve",
                                    reviewer="jacoby")
        self.assertEqual(result.status, "stale")
        self.assertIn("no longer there", result.reason)
        self.assertEqual(fleet_share.bundle(self.store)["items"], [])

    def test_an_unknown_proposal_is_named(self) -> None:
        result = fleet_share.review(self.store, "nope", decision="approve", reviewer="jacoby")
        self.assertEqual(result.status, "unknown")

    # -- the bundle is honest --------------------------------------------------------

    def test_the_bundle_counts_what_it_kept_and_what_it_left(self) -> None:
        good = self.memory()
        fleet_share.propose(self.store, good, reason="general")
        fleet_share.propose(self.store, self.memory(CREDENTIAL), reason="general")
        payload = fleet_share.bundle(self.store)
        self.assertEqual(payload["counts"]["withheld"], 0,
                         "an ineligible memory never became a proposal, so nothing is withheld here")
        self.assertEqual(payload["counts"]["approved"], 0, "nothing is approved yet")
        proposal = fleet_share.proposals(self.store)[0]
        fleet_share.review(self.store, str(proposal["proposal_id"]), decision="approve", reviewer="jacoby")
        payload = fleet_share.bundle(self.store)
        self.assertEqual(payload["counts"]["approved"], 1)
        self.assertTrue(payload["items"][0]["text"])

    def test_the_bundle_of_an_empty_store_is_readable_and_empty(self) -> None:
        payload = fleet_share.bundle(self.store)
        self.assertTrue(payload["readable"])
        self.assertEqual(payload["items"], [])

    def test_a_broken_store_is_refused_not_raised(self) -> None:
        class Broken:
            _conn = None

        proposal = fleet_share.propose(Broken(), "abc", reason="general")
        self.assertFalse(proposal.eligible)
        self.assertIn("could not be read", proposal.reason)

    def test_a_proposal_renders_one_block(self) -> None:
        proposal = fleet_share.propose(self.store, self.memory(), reason="general")
        lines = proposal.render()
        self.assertIn("eligible", lines[0])
        self.assertTrue(any("domain" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
