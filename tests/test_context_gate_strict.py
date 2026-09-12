"""Context gates must require genuine value matches, not substrings.

Regression coverage for the audit's retrieval finding: substring containment
admitted "nonproduction" as a match for "production", "not enabled" for
"enabled", and "v10" for "v1", so context-gated memories were recalled in
incompatible contexts.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401 - loads the flat ``cortex`` package

from cortex.retrieval import MemoryRetriever, RetrievalContext
from cortex.store import CortexStore


class ContextGateStrictnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(Path(self.tmp.name) / "cortex.db")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _scoped_memory(self, **overrides: object) -> str:
        defaults: dict[str, object] = {
            "content": "The deployment gateway is blue.",
            "kind": "operational",
            "context_mode": "context_dependent",
            "scope": {"project": "Cortex", "task_type": "deployment"},
            "entities": ["Cortex", "blue gateway"],
            "preconditions": {"environment": "production"},
            "source_context": "Verified during the Cortex production deployment.",
            "applicable_systems": ["Hermes"],
            "applicable_versions": ["0.3"],
        }
        defaults.update(overrides)
        memory_id, created = self.store.add_memory(**defaults)  # type: ignore[arg-type]
        self.assertTrue(created)
        return memory_id

    def _gate_for(self, memory_id: str, context: RetrievalContext) -> dict:
        _, diagnostics = MemoryRetriever(self.store, threshold=0.0).search_detailed(
            "Which deployment gateway is blue?",
            context=context,
        )
        decision = next(
            item for item in diagnostics.candidate_decisions if item["memory_id"] == memory_id
        )
        return decision["components"]

    def test_nonproduction_does_not_satisfy_production_precondition(self) -> None:
        memory_id = self._scoped_memory()
        components = self._gate_for(
            memory_id,
            RetrievalContext(
                active_project="Cortex",
                scope={"task_type": "deployment"},
                system_state={"environment": "nonproduction"},
            ),
        )
        self.assertEqual(components["context_gate"], 0.0)
        self.assertEqual(components["precondition_match"], 0.0)

    def test_not_enabled_does_not_satisfy_enabled_precondition(self) -> None:
        memory_id = self._scoped_memory(preconditions={"feature": "enabled"})
        components = self._gate_for(
            memory_id,
            RetrievalContext(
                active_project="Cortex",
                scope={"task_type": "deployment"},
                system_state={"feature": "not enabled"},
            ),
        )
        self.assertEqual(components["context_gate"], 0.0)

    def test_v10_does_not_satisfy_v1_version_requirement(self) -> None:
        memory_id = self._scoped_memory(applicable_versions=["v1"])
        components = self._gate_for(
            memory_id,
            RetrievalContext(
                active_project="Cortex",
                scope={"task_type": "deployment"},
                system_state={"environment": "production"},
                applicable_versions=("v10",),
            ),
        )
        self.assertEqual(components["context_gate"], 0.0)
        self.assertEqual(components["version_match"], 0.0)

    def test_exact_values_still_pass_the_gate(self) -> None:
        memory_id = self._scoped_memory()
        matched = MemoryRetriever(self.store, threshold=0.0).search(
            "Which deployment gateway is blue?",
            context=RetrievalContext(
                active_project="Cortex",
                scope={"task_type": "deployment"},
                system_state={"environment": "production"},
                applicable_systems=("Hermes",),
                applicable_versions=("0.3",),
            ),
        )
        self.assertEqual(matched[0].memory["id"], memory_id)
        self.assertEqual(matched[0].components["context_gate"], 1.0)
        self.assertEqual(matched[0].components["precondition_match"], 1.0)
        self.assertEqual(matched[0].components["version_match"], 1.0)


if __name__ == "__main__":
    unittest.main()