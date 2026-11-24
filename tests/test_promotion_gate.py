"""A semantic claim needs independent episodes before it may exist (#105)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex import promotion_gate
from cortex.store import CortexStore

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def episode(self, text: str, *, session: str, day: int, hour: int = 9) -> int:
        stamp = f"2026-11-{day:02d}T{hour:02d}:00:00+00:00"
        with self.store.transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO episodes (session_id, user_content, assistant_content, created_at,
                                         content_hash)
                   VALUES (?, ?, 'ok', ?, ?)""",
                (session, text, stamp, f"h-{session}-{day}-{hour}-{abs(hash(text))}"),
            )
            return int(cursor.lastrowid or 0)

    def claim(self) -> str:
        return "The dashboard runs on port 8100 behind nginx"

    def test_independent_episodes_make_a_claim_ready(self) -> None:
        self.episode("the dashboard runs on port 8100 behind nginx", session="s1", day=10)
        self.episode("the dashboard runs on port 8100 behind nginx", session="s2", day=11)
        self.episode("dashboard on port 8100 behind nginx, right?", session="s3", day=12)
        verdict = promotion_gate.ready(self.store, self.claim())
        self.assertTrue(verdict.ready)
        self.assertEqual(len(verdict.support.episodes), 3)

    def test_paraphrases_in_one_session_are_one_observation(self) -> None:
        for hour in (9, 10, 11):
            self.episode("the dashboard runs on port 8100 behind nginx", session="s1", day=10,
                         hour=hour)
        verdict = promotion_gate.ready(self.store, self.claim())
        self.assertFalse(verdict.ready)
        self.assertEqual(len(verdict.support.sessions), 1)
        self.assertIn("one observation", " ".join(verdict.reasons))

    def test_one_day_is_not_enough_even_across_sessions(self) -> None:
        self.episode("the dashboard runs on port 8100 behind nginx", session="s1", day=10)
        self.episode("the dashboard runs on port 8100 behind nginx", session="s2", day=10)
        self.episode("the dashboard runs on port 8100 behind nginx", session="s3", day=10)
        verdict = promotion_gate.ready(self.store, self.claim())
        self.assertFalse(verdict.ready)
        self.assertIn("day(s)", " ".join(verdict.reasons))

    def test_unrelated_episodes_are_not_evidence(self) -> None:
        for day in (10, 11, 12):
            self.episode("the backup window moved to four in the morning", session=f"s{day}",
                         day=day)
        verdict = promotion_gate.ready(self.store, self.claim())
        self.assertFalse(verdict.ready)
        self.assertEqual(verdict.support.episodes, [])

    def test_promotion_refuses_without_evidence(self) -> None:
        made = promotion_gate.promote(self.store, self.claim())
        self.assertFalse(made["promoted"])
        self.assertIn("not ready", made["reason"])

    def test_a_refused_promotion_writes_nothing(self) -> None:
        before = self.store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        promotion_gate.promote(self.store, self.claim())
        after = self.store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        self.assertEqual(before, after)

    def test_promotion_writes_the_claim_with_citations(self) -> None:
        self.episode("the dashboard runs on port 8100 behind nginx", session="s1", day=10)
        self.episode("the dashboard runs on port 8100 behind nginx", session="s2", day=11)
        self.episode("dashboard on port 8100 behind nginx", session="s3", day=12)
        made = promotion_gate.promote(self.store, self.claim(), at=NOW)
        self.assertTrue(made["promoted"])
        self.assertEqual(len(promotion_gate.citations(self.store, made["memory_id"])), 3)

    def test_the_promoted_memory_names_its_episodes(self) -> None:
        self.episode("the dashboard runs on port 8100 behind nginx", session="s1", day=10)
        self.episode("the dashboard runs on port 8100 behind nginx", session="s2", day=11)
        self.episode("dashboard on port 8100 behind nginx", session="s3", day=12)
        made = promotion_gate.promote(self.store, self.claim(), at=NOW)
        context = self.store._conn.execute(
            "SELECT source_context FROM memories WHERE id = ?", (made["memory_id"],)
        ).fetchone()[0]
        self.assertIn("promoted from episodes", context)

    def test_a_promotion_can_be_forced_and_says_so(self) -> None:
        self.episode("the dashboard runs on port 8100 behind nginx", session="s1", day=10)
        made = promotion_gate.promote(self.store, self.claim(), allow_unready=True, at=NOW)
        self.assertTrue(made["promoted"])
        self.assertFalse(made["ready"])

    def test_the_evidence_records_the_session_and_the_day(self) -> None:
        self.episode("the dashboard runs on port 8100 behind nginx", session="s1", day=10)
        self.episode("the dashboard runs on port 8100 behind nginx", session="s2", day=11)
        self.episode("dashboard on port 8100 behind nginx", session="s3", day=12)
        made = promotion_gate.promote(self.store, self.claim(), at=NOW)
        rows = promotion_gate.citations(self.store, made["memory_id"])
        self.assertEqual({row["session_id"] for row in rows}, {"s1", "s2", "s3"})
        self.assertTrue(all(row["observed_at"] for row in rows))

    def test_a_verdict_describes_itself_both_ways(self) -> None:
        self.assertIn("not ready", promotion_gate.ready(self.store, self.claim()).describe())
        for session, day in (("s1", 10), ("s2", 11), ("s3", 12)):
            self.episode("the dashboard runs on port 8100 behind nginx", session=session, day=day)
        self.assertIn("ready", promotion_gate.ready(self.store, self.claim()).describe())

    def test_overlap_controls_how_strict_the_match_is(self) -> None:
        self.episode("the dashboard runs on port 8100 behind nginx", session="s1", day=10)
        self.episode("the dashboard runs on port 8100 behind nginx", session="s2", day=11)
        self.episode("dashboard port 8100 nginx nginx nginx", session="s3", day=12)
        loose = promotion_gate.ready(self.store, self.claim(), overlap=2)
        strict = promotion_gate.ready(self.store, self.claim(), overlap=5)
        self.assertGreaterEqual(len(loose.support.episodes), len(strict.support.episodes))

    def test_the_support_round_trips(self) -> None:
        self.episode("the dashboard runs on port 8100 behind nginx", session="s1", day=10)
        payload = promotion_gate.ready(self.store, self.claim()).as_dict()
        for key in ("ready", "statement", "support", "reasons", "sentence"):
            self.assertIn(key, payload)
        for key in ("episodes", "sessions", "days", "independent"):
            self.assertIn(key, payload["support"])

    def test_an_empty_store_is_quiet(self) -> None:
        verdict = promotion_gate.ready(self.store, self.claim())
        self.assertFalse(verdict.ready)
        self.assertEqual(promotion_gate.citations(self.store, "nothing"), [])
        described = promotion_gate.report(self.store)
        self.assertEqual(described["citations"], 0)
        self.assertEqual(described["episodes_available"], 0)

    def test_a_claim_with_no_words_is_never_ready(self) -> None:
        self.assertFalse(promotion_gate.ready(self.store, "the a of").ready)


if __name__ == "__main__":
    unittest.main()
