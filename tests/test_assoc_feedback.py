"""When the user says two things are related, the edge learns (#79)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import assoc_feedback as af
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class ParseTests(unittest.TestCase):
    def test_a_negative_sentence_is_read(self) -> None:
        self.assertEqual(af.polarity("those two aren't related"), "unrelated")

    def test_unrelated_is_read(self) -> None:
        self.assertEqual(af.polarity("the nginx note and the sourdough note are unrelated"),
                         "unrelated")

    def test_a_positive_sentence_is_read(self) -> None:
        self.assertEqual(af.polarity("those two are related"), "related")

    def test_a_silent_sentence_says_nothing(self) -> None:
        self.assertIsNone(af.polarity("I read the nginx note this morning"))

    def test_negation_wins_over_the_positive_word(self) -> None:
        self.assertEqual(af.polarity("they are not related at all"), "unrelated")

    def test_a_statement_splits_into_two_parts(self) -> None:
        parsed = af.parse("the nginx note and the sourdough note are not related")
        self.assertIsNotNone(parsed)
        left, right, sense = parsed
        self.assertIn("nginx", left)
        self.assertIn("sourdough", right)
        self.assertEqual(sense, "unrelated")

    def test_an_unparseable_sentence_returns_none(self) -> None:
        self.assertIsNone(af.parse("just thinking out loud"))


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")
        self.nginx = self.memory("Nginx proxies the dashboard on port 8100.")
        self.sourdough = self.memory("The sourdough starter is fed at 8am.")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory(self, content: str) -> str:
        return self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )[0]

    def edge_weight(self) -> float | None:
        row = self.store._conn.execute(
            "SELECT weight FROM edges WHERE (src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?)",
            (self.nginx, self.sourdough, self.sourdough, self.nginx),
        ).fetchone()
        return float(row[0]) if row else None

    def test_related_creates_the_edge(self) -> None:
        applied = af.apply(self.store, self.nginx, self.sourdough, "related", at=NOW)
        self.assertTrue(applied.ok)
        self.assertIsNotNone(self.edge_weight())

    def test_related_reinforces_an_existing_edge(self) -> None:
        self.store.add_edge(self.nginx, self.sourdough, "related", weight=0.3)
        af.apply(self.store, self.nginx, self.sourdough, "related", at=NOW)
        self.assertGreater(self.edge_weight(), 0.3)

    def test_reinforcement_is_capped(self) -> None:
        self.store.add_edge(self.nginx, self.sourdough, "related", weight=1.0)
        applied = af.apply(self.store, self.nginx, self.sourdough, "related", at=NOW)
        self.assertLessEqual(applied.weight, af.MAX_WEIGHT)

    def test_reinforcement_records_evidence(self) -> None:
        af.apply(self.store, self.nginx, self.sourdough, "related", at=NOW)
        # add_edge may store the pair in canonical order, so match either end.
        rows = self.store._conn.execute(
            "SELECT evidence_type FROM edge_evidence WHERE src_id = ? OR dst_id = ?",
            (self.nginx, self.nginx),
        ).fetchall()
        self.assertIn("user_feedback", [str(row[0]) for row in rows])

    def test_unrelated_lowers_an_existing_edge(self) -> None:
        self.store.add_edge(self.nginx, self.sourdough, "related", weight=0.6)
        af.apply(self.store, self.nginx, self.sourdough, "unrelated", at=NOW)
        self.assertLess(self.edge_weight(), 0.1)

    def test_unrelated_records_a_veto(self) -> None:
        af.apply(self.store, self.nginx, self.sourdough, "unrelated", at=NOW)
        self.assertTrue(af.veto(self.store, self.nginx, self.sourdough))
        self.assertTrue(af.veto(self.store, self.sourdough, self.nginx))

    def test_the_veto_does_not_delete_the_edge_or_its_evidence(self) -> None:
        self.store.add_edge(self.nginx, self.sourdough, "related", weight=0.6)
        af.apply(self.store, self.nginx, self.sourdough, "unrelated", at=NOW)
        self.assertIsNotNone(self.edge_weight())
        count = self.store._conn.execute("SELECT COUNT(*) FROM edge_evidence").fetchone()[0]
        self.assertGreaterEqual(int(count), 1)

    def test_unrelated_without_an_edge_is_still_recorded(self) -> None:
        applied = af.apply(self.store, self.nginx, self.sourdough, "unrelated", at=NOW)
        self.assertTrue(applied.ok)
        self.assertTrue(af.veto(self.store, self.nginx, self.sourdough))

    def test_changing_your_mind_updates_the_record(self) -> None:
        af.apply(self.store, self.nginx, self.sourdough, "unrelated", at=NOW)
        af.apply(self.store, self.nginx, self.sourdough, "related", at=NOW)
        self.assertFalse(af.veto(self.store, self.nginx, self.sourdough))

    def test_an_unknown_memory_is_refused(self) -> None:
        applied = af.apply(self.store, self.nginx, "no such memory anywhere", "related")
        self.assertFalse(applied.ok)
        self.assertIn("unknown memory", applied.reason)

    def test_the_same_memory_is_refused(self) -> None:
        self.assertFalse(af.apply(self.store, self.nginx, self.nginx, "related").ok)

    def test_an_unknown_polarity_is_refused(self) -> None:
        self.assertFalse(af.apply(self.store, self.nginx, self.sourdough, "maybe").ok)

    def test_a_statement_is_applied_end_to_end(self) -> None:
        applied = af.apply_statement(
            self.store, "Nginx proxies and sourdough starter are not related", at=NOW
        )
        self.assertTrue(applied.ok, applied.reason)
        self.assertTrue(af.veto(self.store, self.nginx, self.sourdough))

    def test_an_unreadable_statement_says_why(self) -> None:
        applied = af.apply_statement(self.store, "just thinking out loud")
        self.assertFalse(applied.ok)
        self.assertIn("could not read", applied.reason)

    def test_co_retrieval_never_reproposes_a_vetoed_pair(self) -> None:
        from cortex import co_retrieval

        with self.store.transaction() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS co_retrieval_observations (
                       left_id TEXT NOT NULL, right_id TEXT NOT NULL, session_id TEXT NOT NULL,
                       turn_id TEXT, observed_at TEXT NOT NULL,
                       PRIMARY KEY (left_id, right_id, session_id))"""
            )
            for session in ("s1", "s2", "s3"):
                conn.execute(
                    "INSERT INTO co_retrieval_observations VALUES (?, ?, ?, NULL, ?)",
                    (self.nginx, self.sourdough, session, NOW.isoformat()),
                )
        af.apply(self.store, self.nginx, self.sourdough, "unrelated", at=NOW)
        result = co_retrieval.materialize(self.store, at=NOW)
        self.assertEqual(result.get("vetoed", 0), 1)

    def test_the_report_counts_both_directions(self) -> None:
        third = self.memory("The vault lives under ~/.hermes/obsidian-vault.")
        af.apply(self.store, self.nginx, self.sourdough, "related", at=NOW)
        af.apply(self.store, self.nginx, third, "unrelated", at=NOW)
        described = af.report(self.store)
        self.assertEqual(described["related"], 1)
        self.assertEqual(described["unrelated"], 1)

    def test_the_description_is_readable_both_ways(self) -> None:
        self.assertIn("related (weight", af.apply(self.store, self.nginx, self.sourdough,
                                                   "related", at=NOW).describe())
        self.assertIn("vetoed", af.apply(self.store, self.nginx, self.sourdough,
                                         "unrelated", at=NOW).describe())
        self.assertIn("not applied", af.Applied(False, reason="x").describe())

    def test_an_empty_store_is_quiet(self) -> None:
        self.assertEqual(af.feedback(self.store), [])
        self.assertEqual(af.report(self.store)["judgements"], 0)


if __name__ == "__main__":
    unittest.main()
