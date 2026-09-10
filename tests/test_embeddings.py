"""
Tests for the local ONNX embedder (cortex/embeddings.py).

The most important test here is `test_embedding_space_is_unchanged`: it hashes a
fixed sentence's embedding. If pooling, normalisation, tokenisation or the model
file changes, the hash changes and this fails — which is what we want, because a
silently different embedding space would invalidate every measurement that
justified adding this feature (fastembed agreement was cosine 1.0000).

Tests degrade to skips when the model is absent, so the suite still runs on a
machine without the model installed.
"""

from __future__ import annotations

import hashlib
import math
import tempfile
import unittest
from pathlib import Path

from embeddings import DIM, MODEL_ID, Embedder, get_embedder, pack_vector, unpack_vector

# Captured 2026-09-10 from the verified configuration (CLS pooling, L2 norm,
# onnxruntime CPU, threads=1). See plans/Cortex Semantic Fusion results.
GOLDEN_FIXTURE = "Cortex embedding regression fixture sentence."
GOLDEN_SHA256 = "1c893c2cbd542b3750bb6ca59fb07fe0353cbc7df1d484f8d3303ca60b9802c3"


def _model_available() -> bool:
    return Embedder().available


@unittest.skipUnless(_model_available(), "embedding model not installed")
class EmbedderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = get_embedder()

    def test_embedding_space_is_unchanged(self):
        """Locks the vector space. A change here invalidates prior measurements."""
        vec = self.embedder.embed_one(GOLDEN_FIXTURE)
        self.assertEqual(len(vec), DIM)
        digest = hashlib.sha256(pack_vector(vec)).hexdigest()
        self.assertEqual(
            digest,
            GOLDEN_SHA256,
            "embedding changed — pooling/normalisation/model drifted; "
            "every benchmark taken against the old space is now invalid",
        )

    def test_dimensions_and_identity(self):
        self.assertEqual(self.embedder.dim, DIM)
        self.assertEqual(self.embedder.model_id, MODEL_ID)

    def test_vectors_are_l2_normalised(self):
        for vec in self.embedder.embed(["hello world", "a second, longer sentence here"]):
            norm = math.sqrt(sum(x * x for x in vec))
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_empty_input_returns_empty(self):
        # An empty BATCH yields no vectors...
        self.assertEqual(self.embedder.embed([]), [])
        # ...but a blank STRING still yields one vector per input, so batch
        # indices always align with input indices. Callers skip blanks.
        self.assertEqual(len(self.embedder.embed([""])), 1)

    def test_batch_matches_single(self):
        """Batching must not change a vector (padding/length bugs show up here)."""
        texts = ["first probe sentence", "a much longer second probe sentence, padded differently"]
        batched = self.embedder.embed(texts)
        self.assertEqual(len(batched), 2)
        self.assertEqual(batched[0], self.embedder.embed_one(texts[0]))
        self.assertEqual(batched[1], self.embedder.embed_one(texts[1]))

    def test_session_loads_lazily(self):
        fresh = Embedder()
        self.assertIsNone(fresh._session, "model must not load at construction")
        self.assertTrue(fresh.available)


class PackingTests(unittest.TestCase):
    def test_round_trip_is_exact(self):
        # Values chosen to be exactly representable in float32.
        original = [0.5, -1.25, 3.0, 0.0, 0.125]
        self.assertEqual(unpack_vector(pack_vector(original)), original)

    def test_round_trip_within_float32_tolerance(self):
        # Arbitrary values only survive to float32 precision — that is expected.
        original = [0.1, -1e-8, 1234.5678, 1e-30]
        restored = unpack_vector(pack_vector(original))
        self.assertEqual(len(restored), len(original))
        for got, want in zip(restored, original):
            if want == 0:
                self.assertEqual(got, 0.0)
            else:
                self.assertLessEqual(abs(got - want) / abs(want), 1e-6)

    def test_pack_is_float32_little_endian(self):
        self.assertEqual(len(pack_vector([1.0] * DIM)), DIM * 4)

    def test_unpack_empty_blob(self):
        self.assertEqual(unpack_vector(b""), [])


class MissingModelTests(unittest.TestCase):
    """A missing model must never raise — the memory system keeps working."""

    def test_missing_directory_degrades_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            embedder = Embedder(model_dir=Path(tmp))
            self.assertFalse(embedder.available)
            self.assertEqual(embedder.embed(["anything"]), [])
            self.assertEqual(embedder.embed_one("anything"), [])
            self.assertIsNotNone(embedder.load_error)


if __name__ == "__main__":
    unittest.main()
