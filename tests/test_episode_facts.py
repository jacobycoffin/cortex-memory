"""Episode -> fact promotion (#104): repetition across sessions, citations, contradictions refuse."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import episode_facts
from cortex.store import CortexStore

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)
STATEMENT = "the proxmox box runs 8.4 and needs the pveproxy restart"


class EpisodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def episode(self, text: str, *, session: str = "s1", minutes: int = 0) -> str:
        """`episodes.id` is an INTEGER rowid, so the id is the database's to assign, not the test's."""
        stamp = (NOW + timedelta(minutes=minutes)).isoformat()
        with self.store.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO episodes (session_id, user_content, assistant_content, created_at, content_hash) "
                "VALUES (?, ?, ?, ?, ?)",
                (session, text, "ok", stamp, str(abs(hash((text, session, minutes))))),
            )
            return str(cursor.lastrowid)

    def memory_count(self) -> int:
        return int(self.store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    # -- the bar ---------------------------------------------------------------------

    def test_one_mention_is_not_a_fact(self) -> None:
        self.episode(STATEMENT, session="s1")
        cluster = episode_facts.clusters(self.store)[0]
        self.assertEqual(cluster.count, 1)
        self.assertFalse(cluster.eligible)
        self.assertEqual(episode_facts.pending(self.store), [])

    def test_repetition_inside_one_session_is_not_corroboration(self) -> None:
        for index in range(4):
            self.episode(STATEMENT, session="s1", minutes=index)
        cluster = episode_facts.clusters(self.store)[0]
        self.assertEqual(cluster.count, 4)
        self.assertEqual(len(cluster.sessions), 1)
        self.assertFalse(cluster.eligible)
        result = episode_facts.promote(self.store, cluster.key, consent=True, at=NOW)
        self.assertEqual(result.status, "refused")
        self.assertIn("one conversation", result.reason)

    def test_three_mentions_across_two_sessions_clear_the_bar(self) -> None:
        self.episode(STATEMENT, session="s1", minutes=0)
        self.episode(STATEMENT, session="s2", minutes=10)
        self.episode(STATEMENT, session="s2", minutes=20)
        cluster = episode_facts.clusters(self.store)[0]
        self.assertTrue(cluster.eligible, cluster.refusal)
        self.assertEqual(len(episode_facts.pending(self.store)), 1)

    # -- what is not a fact ----------------------------------------------------------

    def test_a_question_is_never_promoted(self) -> None:
        for index in range(3):
            self.episode("is the proxmox box on 8.4 yet?", session=f"s{index}", minutes=index)
        cluster = episode_facts.clusters(self.store)[0]
        self.assertFalse(cluster.eligible)
        self.assertIn("question", cluster.refusal)

    def test_an_instruction_to_the_assistant_is_never_promoted(self) -> None:
        """The most repeated multi-session strings in a real store are smoke tests."""
        for index in range(3):
            self.episode("Reply with exactly PONG.", session=f"s{index}", minutes=index)
        cluster = episode_facts.clusters(self.store)[0]
        self.assertFalse(cluster.eligible)
        self.assertIn("instruction", cluster.refusal)

    def test_a_system_note_is_never_promoted(self) -> None:
        for index in range(3):
            self.episode("[System note: the gateway restarted mid-turn]", session=f"s{index}", minutes=index)
        cluster = episode_facts.clusters(self.store)[0]
        self.assertFalse(cluster.eligible)
        self.assertIn("instruction", cluster.refusal)

    def test_a_string_in_a_large_share_of_sessions_is_a_template(self) -> None:
        """A person does not repeat one sentence in a fifth of everything they ever say."""
        for index in range(4):
            self.episode("the build server needs a restart before the deploy", session=f"a{index}", minutes=index)
        for index in range(8):
            self.episode(f"unrelated remark number {index} about the shelf", session=f"b{index}", minutes=index + 5)
        cluster = next(item for item in episode_facts.clusters(self.store) if item.count == 4)
        self.assertFalse(cluster.eligible)
        self.assertIn("broadcast or a template", cluster.refusal)

    def test_a_fragment_is_never_promoted(self) -> None:
        for index in range(3):
            self.episode("yes ok thanks", session=f"s{index}", minutes=index)
        cluster = episode_facts.clusters(self.store)[0]
        self.assertFalse(cluster.eligible)
        self.assertIn("words", cluster.refusal)

    # -- the fact that gets written --------------------------------------------------

    def test_promotion_writes_one_fact_with_its_citations(self) -> None:
        for index in range(3):
            self.episode(STATEMENT, session=f"s{index}", minutes=index)
        cluster = episode_facts.pending(self.store)[0]
        result = episode_facts.promote(self.store, cluster.key, consent=True, at=NOW)
        self.assertTrue(result.ok, result.reason)
        row = self.store.get_memory(result.fact_id)
        self.assertEqual(row["source_type"], "episode_promotion")
        self.assertTrue(str(row["source_ref"]).startswith("episodes:"))
        for episode_id in cluster.episode_ids[:3]:
            self.assertIn(episode_id, str(row["source_ref"]))
        self.assertGreater(float(row["confidence"]), 0.6)

    def test_the_promoted_text_is_the_most_common_spelling(self) -> None:
        """Spellings that normalise together share a cluster; the fact uses the one said most often."""
        shouted = "The Proxmox Box runs 8.4 and needs the pveproxy restart"
        self.episode(shouted, session="s1", minutes=0)
        self.episode(shouted, session="s2", minutes=1)
        self.episode(STATEMENT, session="s3", minutes=2)
        cluster = episode_facts.pending(self.store)[0]
        self.assertEqual(cluster.count, 3)
        self.assertEqual(cluster.text, shouted, "the mode, not the longest and not the last")

    def test_promotion_without_consent_writes_nothing(self) -> None:
        for index in range(3):
            self.episode(STATEMENT, session=f"s{index}", minutes=index)
        cluster = episode_facts.pending(self.store)[0]
        before = self.memory_count()
        result = episode_facts.promote(self.store, cluster.key, at=NOW)
        self.assertEqual(result.status, "refused")
        self.assertEqual(self.memory_count(), before)

    def test_promotion_is_idempotent(self) -> None:
        for index in range(3):
            self.episode(STATEMENT, session=f"s{index}", minutes=index)
        cluster = episode_facts.pending(self.store)[0]
        self.assertTrue(episode_facts.promote(self.store, cluster.key, consent=True, at=NOW).ok)
        before = self.memory_count()
        again = episode_facts.promote(self.store, cluster.key, consent=True, at=NOW)
        self.assertIn(again.status, {"already", "refused"})
        self.assertEqual(self.memory_count(), before, "no second fact")

    def test_an_unknown_cluster_is_named(self) -> None:
        result = episode_facts.promote(self.store, "nope", consent=True, at=NOW)
        self.assertEqual(result.status, "unknown")

    # -- contradictions --------------------------------------------------------------

    def test_two_statements_differing_in_a_number_block_each_other(self) -> None:
        for index in range(3):
            self.episode("the proxmox box runs 8.4 and needs the pveproxy restart", session=f"a{index}", minutes=index)
            self.episode("the proxmox box runs 8.3 and needs the pveproxy restart", session=f"b{index}", minutes=index + 10)
        everything = episode_facts.clusters(self.store)
        self.assertEqual(len(everything), 2)
        result = episode_facts.promote(self.store, everything[0].key, consent=True, at=NOW)
        self.assertEqual(result.status, "refused")
        self.assertIn("contradicts", result.reason)

    def test_a_negation_blocks_promotion_too(self) -> None:
        for index in range(3):
            self.episode("the proxmox box does need the pveproxy restart", session=f"a{index}", minutes=index)
            self.episode("the proxmox box does not need the pveproxy restart", session=f"b{index}", minutes=index + 5)
        chosen = episode_facts.clusters(self.store)[0]
        result = episode_facts.promote(self.store, chosen.key, consent=True, at=NOW)
        self.assertEqual(result.status, "refused")
        self.assertIn("contradicts", result.reason)

    def test_unrelated_statements_do_not_count_as_contradictions(self) -> None:
        for index in range(3):
            self.episode(STATEMENT, session=f"a{index}", minutes=index)
            self.episode("the walnut shelf needs a second coat of oil", session=f"b{index}", minutes=index + 5)
        cluster = next(item for item in episode_facts.clusters(self.store) if item.text == STATEMENT)
        result = episode_facts.promote(self.store, cluster.key, consent=True, at=NOW)
        self.assertTrue(result.ok, result.reason)

    # -- reporting -------------------------------------------------------------------

    def test_the_report_says_why_clusters_did_not_qualify(self) -> None:
        self.episode(STATEMENT, session="s1")
        for index in range(3):
            self.episode("the walnut shelf needs a second coat of oil", session="s9", minutes=index)
        report = episode_facts.report(self.store)
        self.assertEqual(report["clusters"], 2)
        self.assertEqual(report["eligible"], 0)
        self.assertTrue(any("seen once" in key or "one session" in key for key in report["reasons"]))

    def test_a_broken_store_reports_nothing_rather_than_raising(self) -> None:
        class Broken:
            _conn = None

        self.assertEqual(episode_facts.clusters(Broken()), [])
        result = episode_facts.promote(Broken(), "abc", consent=True, at=NOW)
        self.assertEqual(result.status, "refused")

    def test_normalisation_folds_only_case_and_punctuation(self) -> None:
        self.assertEqual(episode_facts.normalise("The Box runs 8.4!"), "the box runs 8 4")
        self.assertNotEqual(episode_facts.normalise("the box runs 8.4"), episode_facts.normalise("the box runs 8.5"))


if __name__ == "__main__":
    unittest.main()
