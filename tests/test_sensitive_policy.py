"""What may be done with a memory in a sensitive domain, in one table (#184)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import sensitive_policy as sp
from cortex.store import CortexStore


SECRET = "the api key for the bucket is sk-live-9f3a2b7c1d4e"


class TableTests(unittest.TestCase):
    """The policy is data: every cell exists, is named, and says what it rests on."""

    def test_every_domain_covers_every_action(self) -> None:
        table = sp.policy_at()
        self.assertEqual(set(table), set(sp.DOMAINS))
        for domain, rules in table.items():
            self.assertEqual(set(rules), set(sp.ACTIONS), domain)

    def test_every_rule_is_named_and_grounded(self) -> None:
        for version in sp.POLICY_VERSIONS:
            for domain, rules in sp.policy_at(version).items():
                for action, rule in rules.items():
                    self.assertIsInstance(rule, sp.Rule)
                    self.assertTrue(domain.startswith(rule.rule_id.split(".")[0]))
                    self.assertIn(rule.decision, (sp.ALLOW, sp.ALLOW_WITH_CONSENT, sp.REFUSE))
                    self.assertTrue(rule.reason.strip(), f"{domain}/{action}")
                    self.assertTrue(rule.basis.strip(), f"{domain}/{action}")

    def test_a_rule_id_names_exactly_one_rule(self) -> None:
        """An id shared by two different rules cannot be replayed: the log would name an ambiguity.

        The same id appearing in v1, v2 and v3 is correct - it is the same rule, unchanged. What must
        not happen is one id covering two different decisions, which is why every tightening above
        carries a new revision suffix.
        """
        by_id: dict[str, sp.Rule] = {}
        for version in sp.POLICY_VERSIONS:
            for rules in sp.policy_at(version).values():
                for rule in rules.values():
                    self.assertEqual(by_id.setdefault(rule.rule_id, rule), rule, rule.rule_id)

    def test_the_current_table_has_all_three_decision_classes(self) -> None:
        """A table with no `allow` is a table that refuses everything, which is a different bug."""
        decisions = {rule.decision for rules in sp.policy_at().values() for rule in rules.values()}
        self.assertEqual(decisions, {sp.ALLOW, sp.ALLOW_WITH_CONSENT, sp.REFUSE})


class DecisionTests(unittest.TestCase):
    def test_an_ordinary_action_in_a_sensitive_domain_is_allowed(self) -> None:
        decision = sp.decide("the lease runs to March", action="recall")
        self.assertEqual(decision.decision, sp.ALLOW)
        self.assertEqual(decision.domain, "legal")
        self.assertFalse(decision.refused)

    def test_health_recall_needs_consent(self) -> None:
        decision = sp.decide("the prescription was refilled", action="recall")
        self.assertEqual(decision.decision, sp.ALLOW_WITH_CONSENT)
        self.assertEqual(decision.domain, "health")

    def test_consent_turns_allow_with_consent_into_allow_but_keeps_the_rule(self) -> None:
        decision = sp.decide("the prescription was refilled", consent={"health": True})
        self.assertEqual(decision.decision, sp.ALLOW)
        self.assertTrue(decision.consented)
        self.assertEqual(decision.rule_id, "health.recall.consent.v1")

    def test_consent_never_upgrades_a_refusal(self) -> None:
        """No caller can pass `{"credential": True}` and get a secret into a prompt."""
        decision = sp.decide("the router password is hunter2", action="recall",
                             consent={"credential": True})
        self.assertEqual(decision.decision, sp.REFUSE)

    def test_a_credential_is_refused_for_every_action(self) -> None:
        for action in sp.ACTIONS:
            decision = sp.decide(SECRET, action=action)
            self.assertEqual(decision.decision, sp.REFUSE, action)
            self.assertEqual(decision.domain, "credential")

    def test_finance_export_is_refused_after_the_second_version(self) -> None:
        decision = sp.decide("the mortgage statement arrives on the third", action="export")
        self.assertEqual(decision.decision, sp.REFUSE)
        self.assertEqual(decision.rule_id, "finance.export.refuse.v2")

    def test_finance_recall_is_only_asked_about(self) -> None:
        decision = sp.decide("the mortgage statement arrives on the third")
        self.assertEqual(decision.decision, sp.ALLOW_WITH_CONSENT)

    def test_an_unknown_action_is_refused(self) -> None:
        decision = sp.decide("the lease runs to March", action="teleport")
        self.assertEqual(decision.decision, sp.REFUSE)
        self.assertEqual(decision.rule_id, sp.UNKNOWN_ACTION_RULE.rule_id)

    def test_the_most_protective_domain_wins(self) -> None:
        decision = sp.decide("my work password for the payroll portal")
        self.assertEqual(decision.domain, "credential")

    def test_a_relationship_word_is_a_modifier_not_a_topic(self) -> None:
        """"my wife's diagnosis" is a health memory; letting the relationship pattern win would
        relabel most of the store as third-party data."""
        decision = sp.decide("my wife's diagnosis came back clear")
        self.assertEqual(decision.domain, "health")


class UnknownDomainTests(unittest.TestCase):
    def test_an_unrecognised_domain_folds_to_the_strictest_class(self) -> None:
        decision = sp.decide(domain="astrology", action="recall")
        self.assertTrue(decision.folded)
        self.assertEqual(decision.domain, sp.strictest_domain())
        self.assertEqual(decision.requested_domain, "astrology")
        self.assertEqual(decision.decision, sp.REFUSE)

    def test_an_unrecognised_domain_is_never_ordinary(self) -> None:
        """The failure mode of a new domain name arriving from a caller is unrecoverable."""
        self.assertNotEqual(sp.decide(domain="astrology").decision, sp.ALLOW)

    def test_unrecognised_text_is_not_a_licence(self) -> None:
        self.assertEqual(sp.detect_domain("xyzzy frobnicates the quux"), "unrecognised")
        self.assertEqual(sp.decide("xyzzy frobnicates the quux").decision, sp.REFUSE)

    def test_the_strictest_class_is_computed_from_the_table_not_named(self) -> None:
        """If a later version makes health the harshest class, the fold has to follow it."""
        made_up = {
            "health": {action: sp.Rule("h", sp.REFUSE, "r", "b") for action in sp.ACTIONS},
            "finance": {action: sp.Rule("f", sp.ALLOW, "r", "b") for action in sp.ACTIONS},
        }
        self.assertEqual(sp.strictest_domain(made_up), "health")


class ExplanationTests(unittest.TestCase):
    def test_a_refusal_names_the_rule_and_the_class(self) -> None:
        decision = sp.decide(SECRET, action="export")
        text = decision.explain()
        self.assertIn("refuse", text)
        self.assertIn("credential/export", text)
        self.assertIn("credential.export.refuse.v1", text)
        self.assertIn("v3", text)

    def test_a_refusal_never_echoes_the_value(self) -> None:
        decision = sp.decide(SECRET, action="export")
        blob = json.dumps(decision.as_dict()) + decision.explain() + repr(decision)
        for fragment in ("sk-live", "9f3a2b7c1d4e", "api key for the bucket"):
            self.assertNotIn(fragment, blob)

    def test_the_refusal_still_carries_a_hash_so_it_can_be_recognised(self) -> None:
        decision = sp.decide(SECRET, action="export")
        self.assertEqual(len(decision.content_hash), 16)
        self.assertIn(decision.content_hash, decision.explain())
        self.assertEqual(decision.content_hash, sp.decide(SECRET).content_hash)

    def test_a_fold_says_it_folded(self) -> None:
        self.assertIn("unrecognised", sp.decide(domain="astrology").explain())


class VersionTests(unittest.TestCase):
    def test_a_decision_records_the_version_it_was_made_under(self) -> None:
        self.assertEqual(sp.decide("the lease runs to March").policy_version, sp.CURRENT_VERSION)
        self.assertEqual(sp.decide("the lease runs to March", version=1).policy_version, 1)

    def test_a_past_table_can_be_rebuilt(self) -> None:
        self.assertEqual(sp.policy_at(1)["finance"]["export"].decision, sp.ALLOW_WITH_CONSENT)
        self.assertEqual(sp.policy_at(2)["finance"]["export"].decision, sp.REFUSE)
        self.assertEqual(sp.policy_at(3)["legal"]["derive"].decision, sp.ALLOW_WITH_CONSENT)

    def test_an_old_decision_replays_to_todays_answer(self) -> None:
        old = sp.decide(domain="finance", action="export", version=1)
        self.assertEqual(old.decision, sp.ALLOW_WITH_CONSENT)
        again = sp.replay(old)
        self.assertTrue(again.changed)
        self.assertIn("finance.export.consent.v1", again.note())
        self.assertIn("finance.export.refuse.v2", again.note())

    def test_a_replay_at_the_same_version_is_unchanged(self) -> None:
        old = sp.decide(domain="health", action="recall", version=2)
        self.assertFalse(sp.replay(old, version=2).changed)

    def test_an_unknown_version_raises_rather_than_guessing(self) -> None:
        with self.assertRaises(ValueError):
            sp.policy_at(99)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_a_decision_is_logged_with_its_rule_and_version(self) -> None:
        sp.gate(self.store, "the mortgage statement arrives on the third", action="export")
        rows = sp.history(self.store)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rule_id"], "finance.export.refuse.v2")
        self.assertEqual(int(rows[0]["policy_version"]), sp.CURRENT_VERSION)

    def test_the_log_holds_a_hash_not_the_content(self) -> None:
        sp.gate(self.store, SECRET, action="export")
        blob = json.dumps(sp.history(self.store)) + json.dumps(sp.summary(self.store))
        self.assertNotIn("sk-live", blob)
        self.assertIn("credential", blob)

    def test_deciding_twice_updates_rather_than_duplicates(self) -> None:
        sp.gate(self.store, "the lease runs to March", action="recall")
        sp.gate(self.store, "the lease runs to March", action="recall")
        self.assertEqual(sp.summary(self.store)["decisions"], 1)

    def test_the_summary_groups_by_decision_and_version(self) -> None:
        sp.gate(self.store, "the lease runs to March")
        sp.gate(self.store, SECRET, action="export")
        described = sp.summary(self.store)
        self.assertEqual(described["by_decision"][sp.ALLOW], 1)
        self.assertEqual(described["by_decision"][sp.REFUSE], 1)
        self.assertEqual(described["by_version"][sp.CURRENT_VERSION], 2)

    def test_the_gate_does_not_write_a_memory(self) -> None:
        """The policy says what may be done; the caller still owns the doing."""
        sp.gate(self.store, "the lease runs to March")
        row = self.store._conn.execute("SELECT COUNT(*) AS count FROM memories").fetchone()
        self.assertEqual(int(row["count"]), 0)

    def test_the_render_lists_decisions(self) -> None:
        sp.gate(self.store, SECRET, action="export")
        rendered = "\n".join(sp.render(self.store))
        self.assertIn("sensitive-domain decisions:", rendered)
        self.assertIn("credential.export.refuse.v1", rendered)

    def test_an_empty_log_says_nothing_asks_the_policy_yet(self) -> None:
        self.assertIn("nothing asks it yet", sp.summary(self.store)["note"])

    def test_a_broken_store_does_not_change_the_verdict(self) -> None:
        class Broken:
            _conn = None

            def transaction(self):
                raise RuntimeError("no database")

        decision = sp.gate(Broken(), SECRET, action="export")
        self.assertEqual(decision.decision, sp.REFUSE)
        self.assertFalse(sp.record_decision(Broken(), decision))
        self.assertEqual(sp.summary(Broken())["decisions"], 0)


class TallyTests(unittest.TestCase):
    def test_the_tally_counts_without_keeping_anything(self) -> None:
        counted = sp.tally([SECRET, "the lease runs to March", "the prescription was refilled"],
                           action="export")
        self.assertEqual(counted["scanned"], 3)
        self.assertEqual(counted["by_decision"][sp.REFUSE], 3)
        self.assertEqual(counted["by_domain"]["credential"], 1)
        self.assertNotIn("sk-live", json.dumps(counted))


if __name__ == "__main__":
    unittest.main()
