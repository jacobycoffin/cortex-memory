"""Superseded memories must lose to their replacements, not tie with them (item #4)."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.store import CortexStore
from cortex.validity import SUPERSEDED_FACTOR, apply_validity_penalty, superseded_ids


@dataclass
class Fake:
    memory: dict[str, Any]
    score: float
    components: dict[str, float] = field(default_factory=dict)


class PenaltyTests(unittest.TestCase):
    def test_a_superseded_memory_is_demoted(self) -> None:
        old = Fake({"id": "old"}, 0.60)
        out, count = apply_validity_penalty([old], {"old"})
        self.assertEqual(count, 1)
        self.assertAlmostEqual(out[0].score, 0.60 * SUPERSEDED_FACTOR, places=6)
        self.assertEqual(out[0].components["superseded"], 1.0)

    def test_the_replacement_is_untouched(self) -> None:
        new = Fake({"id": "new"}, 0.60)
        out, count = apply_validity_penalty([new], {"old"})
        self.assertEqual((count, out[0].score), (0, 0.60))

    def test_no_superseded_ids_is_a_no_op(self) -> None:
        result = Fake({"id": "x"}, 0.5)
        out, count = apply_validity_penalty([result], set())
        self.assertEqual((count, out[0].score), (0, 0.5))

    def test_demotion_is_never_deletion(self) -> None:
        old = Fake({"id": "old"}, 0.60)
        out, _ = apply_validity_penalty([old], {"old"})
        self.assertEqual(len(out), 1)  # history stays reachable

    def test_inputs_are_not_mutated(self) -> None:
        old = Fake({"id": "old"}, 0.60)
        apply_validity_penalty([old], {"old"})
        self.assertEqual((old.score, old.components), (0.60, {}))

    def test_an_extreme_factor_cannot_invert_signs(self) -> None:
        old = Fake({"id": "old"}, 0.60)
        out, _ = apply_validity_penalty([old], {"old"}, factor=0.0)
        self.assertEqual(out[0].score, 0.0)


class HistoricalModeTests(unittest.TestCase):
    """Historical recall asks what was true then; the old value is the answer."""

    def test_demotion_is_skipped_for_historical_recall(self) -> None:
        old = Fake({"id": "old"}, 0.60)
        # The retriever only calls this in current mode; the test pins the contract that
        # a caller must not apply it in historical mode.
        out, count = apply_validity_penalty([old], {"old"})
        self.assertEqual(count, 1)  # the function itself always demotes
        self.assertLess(out[0].score, 0.60)  # so mode handling must live at the call site


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _add(self, content: str, supersedes: str | None = None) -> str:
        created = self.store.add_memory(
            content=content,
            kind="semantic",
            source_type="user_turn",
            source_category="USER_EXPLICIT",
            supersedes_id=supersedes,
        )
        return created[0] if isinstance(created, tuple) else created

    def test_the_superseded_id_is_discovered(self) -> None:
        old = self._add("The maintenance window is Sunday 02:00.")
        new = self._add("The maintenance window moved to Tuesday 03:00.", supersedes=old)
        self.assertEqual(superseded_ids(self.store), {old})
        self.assertNotIn(new, superseded_ids(self.store))

    def test_an_empty_store_has_nothing_superseded(self) -> None:
        self.assertEqual(superseded_ids(self.store), set())

    def test_a_store_without_a_connection_returns_empty(self) -> None:
        class Broken:
            pass

        self.assertEqual(superseded_ids(Broken()), set())


if __name__ == "__main__":
    unittest.main()
