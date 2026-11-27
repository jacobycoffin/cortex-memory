"""Semantic-only candidates must be able to reach the top-k, on similarity.

The fusion path used to add a flat reciprocal-rank bonus to every semantic hit:
rank 0 and rank 19 differed by ~0.03, and nothing in the score reflected how
similar the vector actually was. Because the lexical term (0.20 of the score)
and the feature ``semantic`` term are computed from memory text, a candidate the
lexical pass never surfaced could not close that gap on the strength of a very
good embedding match, while a marginal neighbour received almost the same flat
credit.

These guards pin the replacement: a similarity-derived evidence curve scales the
rank bonus, and a semantic-only candidate may fill the lexical weight slot with
that evidence. Strong matches enter the top-k; weak ones still lose. Everything
runs against a temp database with a stub embedder, so no model is required.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from cortex import embeddings as embeddings_module
from cortex.retrieval import MemoryRetriever, _semantic_evidence
from cortex.store import CortexStore

try:
    import numpy  # noqa: F401 - the semantic pool ranks vectors with numpy
    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - bare environments have no numpy
    _NUMPY_AVAILABLE = False

_QUERY = "gardening soil compost fertiliser schedule"

# Decoys share the query's vocabulary but say nothing about the target.
_DECOYS = (
    "gardening soil compost notes for the front border beds",
    "compost and soil ratios recorded for the vegetable patch",
    "soil testing schedule for the compost heaps",
    "fertiliser schedule reminder for the garden soil",
    "compost turning schedule kept with the soil notes",
    "garden soil deliveries logged against the compost order",
    "soil moisture readings taken before the compost was spread",
    "fertiliser strengths written on the compost shed wall",
    "garden notes on mulching schedule and soil texture",
    "compost bin temperatures logged with the soil probe",
    "soil pH readings for the compost trial rows",
    "garden tool inventory kept beside the fertiliser shelf",
)

# The target shares no token with the query: only the embedding can find it.
_TARGET = "the tomato bed wanted extra loam and leaf mould before spring"
_NOISE = "quarterly invoice reconciliation for the workshop account"


def _unit_with_cosine(cosine: float) -> list[float]:
    """A 4-dim unit vector whose cosine with [1, 0, 0, 0] is ``cosine``."""
    return [float(cosine), math.sqrt(max(0.0, 1.0 - cosine * cosine)), 0.0, 0.0]


class _StubEmbedder:
    model_id = "stub-model"
    dim = 4

    def __init__(self, vector: list[float] | None = None, available: bool = True):
        self._vector = vector if vector is not None else [1.0, 0.0, 0.0, 0.0]
        self._available = available

    @property
    def available(self) -> bool:
        return self._available

    def embed_one(self, text: str) -> list[float]:          # noqa: ARG002
        return list(self._vector) if self._available else []

    def embed(self, texts):
        return [list(self._vector) for _ in texts] if self._available else []


@unittest.skipUnless(_NUMPY_AVAILABLE, "semantic pool requires numpy")
class SemanticLiftTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = CortexStore(str(Path(self._tmp.name) / "cortex.db"))
        self._orig_get_embedder = embeddings_module.get_embedder

        self.ids: dict[str, str] = {}
        for index, text in enumerate((*_DECOYS, _TARGET, _NOISE)):
            result = self.store.add_memory(text, state="active")
            self.ids[f"m{index}"] = result[0] if isinstance(result, tuple) else result
        self.target_id = self.ids[f"m{len(_DECOYS)}"]

    def tearDown(self):
        embeddings_module.get_embedder = self._orig_get_embedder
        self._tmp.cleanup()

    def _retrieve(self, cosine: float, *, weight: float = 10.0, limit: int = 6):
        embeddings_module.get_embedder = lambda model_dir=None: _StubEmbedder(
            vector=[1.0, 0.0, 0.0, 0.0]
        )
        self.store.set_memory_embedding(
            self.target_id, "stub-model", _unit_with_cosine(cosine)
        )
        retriever = MemoryRetriever(self.store, semantic_weight=weight)
        results, _ = retriever.search_detailed(_QUERY, limit=limit)
        return results

    # ---------------------------------------------------------------- curve

    def test_evidence_is_zero_at_or_below_the_noise_anchor(self):
        self.assertEqual(_semantic_evidence(0.10), 0.0)
        self.assertEqual(_semantic_evidence(0.45), 0.0)
        self.assertEqual(_semantic_evidence(float("nan")), 0.0)
        self.assertEqual(_semantic_evidence("not a number"), 0.0)

    def test_evidence_is_monotone_and_bounded(self):
        ladder = [_semantic_evidence(value) for value in (0.5, 0.6, 0.7, 0.8, 0.95)]
        self.assertEqual(ladder, sorted(ladder))
        self.assertLess(ladder[0], ladder[1])
        self.assertEqual(ladder[-1], 1.0)
        for value in ladder:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_configured_floor_can_only_raise_the_bar(self):
        self.assertEqual(_semantic_evidence(0.55, floor=0.9), 0.0)
        self.assertLess(
            _semantic_evidence(0.70, floor=0.65),
            _semantic_evidence(0.70),
        )

    # ------------------------------------------------------------- behaviour

    def test_strong_vector_match_reaches_the_top_k_without_lexical_overlap(self):
        results = self._retrieve(0.95)
        ids = [result.memory["id"] for result in results]
        self.assertIn(
            self.target_id,
            ids,
            "a near-identical embedding match never entered the top-k; the score "
            "is still dominated by lexical absence",
        )
        target = next(r for r in results if r.memory["id"] == self.target_id)
        self.assertGreater(target.components.get("semantic_lift", 0.0), 0.0)
        self.assertGreater(target.components.get("semantic_evidence", 0.0), 0.9)

    def test_weak_vector_match_still_loses(self):
        results = self._retrieve(0.55)
        ids = [result.memory["id"] for result in results]
        self.assertNotIn(
            self.target_id,
            ids,
            "a marginal (0.55) neighbour was promoted into the top-k",
        )

    def test_stronger_similarity_scores_strictly_higher(self):
        strong = self._retrieve(0.92)
        weak = self._retrieve(0.58)
        strong_score = next(
            (r.score for r in strong if r.memory["id"] == self.target_id), 0.0
        )
        weak_score = next(
            (r.score for r in weak if r.memory["id"] == self.target_id), 0.0
        )
        self.assertGreater(strong_score, weak_score)

    def test_lexically_found_candidates_get_no_semantic_floor(self):
        """Only candidates the lexical pass missed may fill the lexical slot."""
        results = self._retrieve(0.95, limit=6)
        for result in results:
            if result.memory["id"] == self.target_id:
                continue
            self.assertNotIn(
                "semantic_lift",
                result.components,
                "an already-retrieved candidate was lifted as if its lexical "
                "term were missing",
            )

    def test_semantic_fusion_is_still_inert_when_disabled(self):
        embeddings_module.get_embedder = lambda model_dir=None: _StubEmbedder(
            vector=[1.0, 0.0, 0.0, 0.0]
        )
        self.store.set_memory_embedding(
            self.target_id, "stub-model", _unit_with_cosine(0.95)
        )
        off = [
            result.memory["id"]
            for result in MemoryRetriever(self.store).search_detailed(_QUERY, limit=6)[0]
        ]
        self.assertNotIn(self.target_id, off)


if __name__ == "__main__":
    unittest.main()
