"""Ranking must reward use, not popularity (backlog item #3)."""

from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.store import CortexStore
from cortex.usage_weight import MAX_BOOST, PER_USE, apply_usage_boost, use_boost


@dataclass
class Fake:
    memory: dict[str, Any]
    score: float
    components: dict[str, float] = field(default_factory=dict)


class BoostMathTests(unittest.TestCase):
    def test_never_used_gets_nothing(self) -> None:
        self.assertEqual(use_boost(0), 0.0)
        self.assertEqual(use_boost(-3), 0.0)

    def test_boost_is_monotonic_but_flattens(self) -> None:
        one, two, ten = use_boost(1), use_boost(2), use_boost(10)
        self.assertGreater(two, one)
        self.assertGreater(ten, two)
        # log scaling: the tenth use adds far less than ten times the first
        self.assertLess(ten, 10 * one)

    def test_boost_is_capped(self) -> None:
        self.assertLessEqual(use_boost(10_000), MAX_BOOST)
        self.assertAlmostEqual(use_boost(10_000), MAX_BOOST)

    def test_single_use_matches_the_constant(self) -> None:
        self.assertAlmostEqual(use_boost(1), PER_USE * math.log1p(1), places=9)


class ApplyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def add(self, content: str) -> str:
        created = self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )
        return created[0] if isinstance(created, tuple) else created

    def record_use(self, memory_id: str, times: int) -> None:
        with self.store.transaction() as conn:
            for _ in range(times):
                conn.execute(
                    "INSERT INTO access_log (memory_id, event, created_at) "
                    "VALUES (?, 'used', datetime('now'))",
                    (memory_id,),
                )

    def test_used_memory_outranks_an_unused_tie(self) -> None:
        # distinct content: identical text would be deduplicated into one row
        used = self.add("The maintenance window is Sunday 02:00.")
        unused = self.add("The maintenance window is Sunday 03:00.")
        self.record_use(used, 5)
        a = Fake({"id": used}, 0.50)
        b = Fake({"id": unused}, 0.50)
        results, boosted = apply_usage_boost(self.store, [a, b])
        self.assertEqual(boosted, 1)
        by_id = {r.memory["id"]: r for r in results}
        self.assertGreater(by_id[used].score, by_id[unused].score)
        self.assertEqual(by_id[unused].score, 0.50)
        self.assertEqual(by_id[used].components["use_count"], 5)

    def test_retrieval_events_do_not_count(self) -> None:
        memory = self.add("A fact that was only ever retrieved.")
        with self.store.transaction() as conn:
            for _ in range(20):
                conn.execute(
                    "INSERT INTO access_log (memory_id, event, created_at) "
                    "VALUES (?, 'retrieved', datetime('now'))",
                    (memory,),
                )
        result = Fake({"id": memory}, 0.4)
        results, boosted = apply_usage_boost(self.store, [result])
        self.assertEqual(boosted, 0)
        self.assertEqual(result.score, 0.4)

    def test_boost_cannot_beat_a_real_relevance_gap(self) -> None:
        used = self.add("Low relevance but used often.")
        self.record_use(used, 500)
        relevant = self.add("High relevance, never used.")
        weak = Fake({"id": used}, 0.40)
        strong = Fake({"id": relevant}, 0.90)
        apply_usage_boost(self.store, [weak, strong])
        self.assertLess(weak.score, strong.score)

    def test_missing_memory_is_ignored(self) -> None:
        result = Fake({"id": "does-not-exist"}, 0.3)
        results, boosted = apply_usage_boost(self.store, [result])
        self.assertEqual((boosted, result.score), (0, 0.3))


if __name__ == "__main__":
    unittest.main()
