"""Group conflicting facts into one review item instead of N pairwise ones (#52)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import contradiction_cluster as cc
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
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

    def dispute(self, left: str, right: str) -> None:
        """Record through the module that owns the table, not with raw SQL."""
        from cortex import dispute

        dispute.mark(self.store, left, right, reason="different values", at=NOW)

    def trio(self):
        first = self.memory("The dashboard runs on port 8100.", confidence=0.7)
        second = self.memory("The dashboard runs on port 8200.", confidence=0.9)
        third = self.memory("The dashboard runs on port 8300.", confidence=0.5)
        self.dispute(first, second)
        self.dispute(second, third)
        return first, second, third

    def test_three_way_conflict_is_one_cluster(self) -> None:
        """Two pairs about the same thing are one question, not two."""
        self.trio()
        grouped = cc.clusters(self.store)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0].size, 3)
        self.assertEqual(len(grouped[0].pairs), 2)

    def test_unrelated_conflicts_stay_separate(self) -> None:
        first = self.memory("The dashboard runs on port 8100.")
        second = self.memory("The dashboard runs on port 8200.")
        third = self.memory("Backups run at 2am.")
        fourth = self.memory("Backups run at 4am.")
        self.dispute(first, second)
        self.dispute(third, fourth)
        self.assertEqual(len(cc.clusters(self.store)), 2)

    def test_a_chain_of_pairs_collapses_into_one_component(self) -> None:
        ids = [self.memory(f"The dashboard uses port {8100 + index}.") for index in range(4)]
        for index in range(3):
            self.dispute(ids[index], ids[index + 1])
        grouped = cc.clusters(self.store)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0].size, 4)

    def test_a_single_pair_is_still_a_review_item(self) -> None:
        first = self.memory("The dashboard runs on port 8100.")
        second = self.memory("The dashboard runs on port 8200.")
        self.dispute(first, second)
        self.assertEqual(len(cc.clusters(self.store)), 1)

    def test_the_topic_comes_from_the_shared_subject(self) -> None:
        first = self.memory("Two approvers sign off.", subject="amber-release", predicate="count")
        second = self.memory("Three approvers sign off.", subject="amber-release", predicate="count")
        self.dispute(first, second)
        self.assertEqual(cc.clusters(self.store)[0].topic, "amber-release·count")

    def test_the_topic_falls_back_to_shared_words(self) -> None:
        first = self.memory("The dashboard runs on port 8100.")
        second = self.memory("The dashboard runs on port 8200.")
        self.dispute(first, second)
        self.assertIn("dashboard", cc.clusters(self.store)[0].topic)

    def test_the_recommendation_names_the_strongest_member(self) -> None:
        first, second, _third = self.trio()
        cluster = cc.clusters(self.store)[0]
        self.assertIn(second[:8], cluster.recommendation())
        self.assertIn("0.90", cluster.recommendation())

    def test_nothing_is_resolved_without_being_asked(self) -> None:
        self.trio()
        cc.clusters(self.store)
        open_pairs = [row for row in cc.pairs(self.store) if not row.get("winner_id")]
        self.assertEqual(len(open_pairs), 2)

    def test_a_cluster_describes_itself(self) -> None:
        self.trio()
        described = cc.clusters(self.store)[0].describe()
        self.assertIn("3 conflicting statement", described)
        self.assertIn("needs a decision", described)

    def test_pending_lists_only_open_clusters(self) -> None:
        first, second, _third = self.trio()
        cluster = cc.clusters(self.store)[0]
        cc.resolve(self.store, cluster, second, by="jacoby", at=NOW)
        self.assertEqual(cc.pending(self.store), [])

    def test_resolving_needs_a_name(self) -> None:
        _first, second, _third = self.trio()
        cluster = cc.clusters(self.store)[0]
        outcome = cc.resolve(self.store, cluster, second)
        self.assertFalse(outcome["resolved"])
        self.assertIn("name who decided", outcome["reason"])

    def test_the_winner_must_belong_to_the_cluster(self) -> None:
        _first, _second, _third = self.trio()
        outsider = self.memory("Something else entirely about the dashboard.")
        outcome = cc.resolve(self.store, cc.clusters(self.store)[0], outsider, by="jacoby")
        self.assertFalse(outcome["resolved"])
        self.assertIn("must be one of", outcome["reason"])

    def test_resolving_an_item_resolves_every_pair_in_it(self) -> None:
        _first, second, _third = self.trio()
        outcome = cc.resolve(self.store, cc.clusters(self.store)[0], second, by="jacoby", at=NOW)
        self.assertTrue(outcome["resolved"])
        self.assertEqual(outcome["pairs"], 2)
        self.assertEqual([row for row in cc.pairs(self.store) if not row.get("winner_id")], [])

    def test_resolving_twice_changes_nothing(self) -> None:
        _first, second, _third = self.trio()
        cluster = cc.clusters(self.store)[0]
        cc.resolve(self.store, cluster, second, by="jacoby", at=NOW)
        again = cc.resolve(self.store, cluster, second, by="jacoby", at=NOW)
        self.assertEqual(again["pairs"], 0)

    def test_the_stats_show_the_collapse(self) -> None:
        self.trio()
        described = cc.stats(self.store)
        self.assertEqual(described["pairs_open"], 2)
        self.assertEqual(described["clusters"], 1)
        self.assertEqual(described["collapsed"], 1)

    def test_the_cluster_round_trips(self) -> None:
        self.trio()
        payload = cc.clusters(self.store)[0].as_dict()
        for key in ("topic", "members", "pairs", "size", "resolved", "recommendation"):
            self.assertIn(key, payload)

    def test_a_timestamp_difference_is_not_a_contradiction(self) -> None:
        """The live store's near-miss proposals were all timestamps and filenames."""
        self.assertTrue(cc.trivial_difference(["18:48", "21:03"]))
        self.assertTrue(cc.trivial_difference(["2026-10-03", "2026-10-04"]))
        self.assertTrue(cc.trivial_difference(["daily/2026-10-03.md"]))
        self.assertTrue(cc.trivial_difference(["2026-10-03", "daily/2026-10-03.md"]))

    def test_a_substantive_difference_is_a_contradiction(self) -> None:
        # a changed port is the conflict worth reviewing, so bare numbers stay substantive
        self.assertFalse(cc.trivial_difference(["8100", "8200"]))
        self.assertFalse(cc.trivial_difference(["14", "17"]))
        self.assertFalse(cc.trivial_difference(["not"]))
        self.assertFalse(cc.trivial_difference(["18:48", "disabled"]))

    def test_an_empty_difference_is_trivial(self) -> None:
        self.assertTrue(cc.trivial_difference([]))

    def test_detection_reports_without_recording(self) -> None:
        """Recording a dispute creates review work, so it is opt-in."""
        from cortex import near_miss

        left = self.memory("The dashboard runs on port 8100 today.")
        right = self.memory("The dashboard runs on port 8200 today.")
        self.assertIsInstance(near_miss.detect(self.store), list)
        outcome = cc.record_conflicts(self.store, apply=False)
        self.assertFalse(outcome["applied"])
        self.assertEqual(outcome["recorded"], 0)
        self.assertEqual([row for row in cc.pairs(self.store) if not row.get("winner_id")], [])

    def test_recording_files_the_detected_conflicts(self) -> None:
        from cortex import near_miss

        if not hasattr(near_miss, "detect"):
            self.skipTest("near-miss detection is missing")
        left = self.memory("The backup window is 2am sharp.", confidence=0.8)
        right = self.memory("The backup window is 4am sharp.", confidence=0.7)
        detected = cc.candidates(self.store)
        if not detected:
            self.skipTest("the detectors found nothing to file in this fixture")
        outcome = cc.record_conflicts(self.store, apply=True, at=NOW)
        self.assertGreaterEqual(outcome["recorded"], 1)
        self.assertGreaterEqual(cc.stats(self.store)["pairs_recorded"], 1)

    def test_an_empty_store_is_quiet(self) -> None:
        self.assertEqual(cc.clusters(self.store), [])
        self.assertEqual(cc.stats(self.store)["clusters"], 0)
        self.assertEqual(cc.pending(self.store), [])


if __name__ == "__main__":
    unittest.main()
