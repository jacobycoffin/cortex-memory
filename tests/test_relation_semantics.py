from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT

from cortex.store import CortexStore


class RelationAwareActivationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _memory(self, text: str, **kwargs: object) -> str:
        return self.store.add_memory(text, **kwargs)[0]

    def test_contradiction_never_spreads_positive_activation(self) -> None:
        left = self._memory("The production gateway is blue.")
        right = self._memory("The production gateway is orange.")
        self.store.add_edge(left, right, "contradicts", weight=1.0)

        scores = self.store.association_scores({left: 1.0}, depth=2)

        self.assertNotIn(right, scores)

    def test_superseded_memory_leads_to_replacement_but_not_back(self) -> None:
        old = self._memory("The service listens on port 8000.")
        new = self._memory("The service now listens on port 8100.", supersedes_id=old)

        from_old = self.store.association_scores({old: 1.0}, depth=2)
        from_new = self.store.association_scores({new: 1.0}, depth=2)

        self.assertIn(new, from_old)
        self.assertNotIn(old, from_new)

    def test_supporting_evidence_is_reached_from_claim_only(self) -> None:
        evidence = self._memory("The deployment log records a successful blue release.")
        claim = self._memory("The blue deployment path is reliable.")
        self.store.add_edge(evidence, claim, "supports", weight=0.9)

        from_claim = self.store.association_scores({claim: 1.0}, depth=2)
        from_evidence = self.store.association_scores({evidence: 1.0}, depth=2)

        self.assertIn(evidence, from_claim)
        self.assertNotIn(claim, from_evidence)

    def test_reviewed_useful_together_relation_is_symmetric(self) -> None:
        first = self._memory("Use the release checklist before deployment.")
        second = self._memory("Check the rollback switch after deployment.")
        self.store.add_edge(first, second, "useful_together", weight=0.8)

        self.assertIn(second, self.store.association_scores({first: 1.0}, depth=1))
        self.assertIn(first, self.store.association_scores({second: 1.0}, depth=1))


if __name__ == "__main__":
    unittest.main()
