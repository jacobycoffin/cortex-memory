"""Elliptical follow-ups resolve against the session, not the void (item #17)."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.session_scope import (
    apply_session_boost,
    content_tokens,
    is_elliptical,
    recent_memory_ids,
)
from cortex.store import CortexStore


@dataclass
class Fake:
    memory: dict[str, Any]
    score: float
    components: dict[str, float] = field(default_factory=dict)


class EllipticalTests(unittest.TestCase):
    def test_bare_follow_ups_are_elliptical(self) -> None:
        for query in ("do that", "yeah do it", "same for the other one", "and the second one?",
                      "go ahead", "ok", "next"):
            self.assertTrue(is_elliptical(query), query)

    def test_a_query_with_a_topic_is_not_elliptical(self) -> None:
        for query in ("what did we decide about the website palette",
                      "cortex roadmap", "seerr telegram notifications",
                      "temple difficulty increments"):
            self.assertFalse(is_elliptical(query), query)

    def test_empty_query_is_not_elliptical(self) -> None:
        self.assertFalse(is_elliptical(""))
        self.assertFalse(is_elliptical("   "))

    def test_content_tokens_drop_filler(self) -> None:
        self.assertEqual(content_tokens("do that again for the other one"), [])


class BoostTests(unittest.TestCase):
    def test_a_session_memory_outranks_a_slightly_higher_stranger(self) -> None:
        familiar = Fake({"id": "known"}, 0.40)
        stranger = Fake({"id": "new"}, 0.42)
        out, count = apply_session_boost([familiar, stranger], ["known"])
        self.assertEqual(count, 1)
        ordered = sorted(out, key=lambda r: -r.score)
        self.assertEqual(ordered[0].memory["id"], "known")

    def test_no_recent_ids_is_a_no_op(self) -> None:
        result = Fake({"id": "x"}, 0.4)
        out, count = apply_session_boost([result], [])
        self.assertEqual((count, out[0].score), (0, 0.4))

    def test_the_freshest_memory_in_the_session_gets_the_most(self) -> None:
        first = Fake({"id": "older"}, 0.4)
        second = Fake({"id": "newer"}, 0.4)
        out, _ = apply_session_boost([first, second], ["newer", "older"])
        by_id = {r.memory["id"]: r for r in out}
        self.assertGreater(by_id["newer"].score, by_id["older"].score)

    def test_the_boost_cannot_beat_a_real_relevance_gap(self) -> None:
        familiar = Fake({"id": "known"}, 0.30)
        relevant = Fake({"id": "new"}, 0.70)
        out, _ = apply_session_boost([familiar, relevant], ["known"])
        self.assertLess(max(r.score for r in out if r.memory["id"] == "known"), 0.70)

    def test_input_is_not_mutated(self) -> None:
        result = Fake({"id": "known"}, 0.4)
        apply_session_boost([result], ["known"])
        self.assertEqual((result.score, result.components), (0.4, {}))


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _memory(self, content: str) -> str:
        created = self.store.add_memory(
            content=content, kind="semantic", source_type="user_turn",
            source_category="USER_EXPLICIT",
        )
        return created[0] if isinstance(created, tuple) else created

    def _touch(self, memory_id: str, session: str, when: str) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO access_log (memory_id, event, query, session_id, score, created_at) "
                "VALUES (?, 'retrieved', 'q', ?, 0.5, ?)",
                (memory_id, session, when),
            )

    def test_recent_ids_are_this_session_only_newest_first(self) -> None:
        a = self._memory("The first fact touched in this session.")
        b = self._memory("The second fact touched in this session.")
        other = self._memory("A fact touched in a different session.")
        self._touch(a, "s1", "2026-11-21 10:00:00")
        self._touch(b, "s1", "2026-11-21 10:05:00")
        self._touch(other, "other", "2026-11-21 10:09:00")
        self.assertEqual(recent_memory_ids(self.store, "s1"), [b, a])

    def test_missing_session_returns_nothing(self) -> None:
        self.assertEqual(recent_memory_ids(self.store, None), [])
        self.assertEqual(recent_memory_ids(self.store, "unknown"), [])


if __name__ == "__main__":
    unittest.main()
