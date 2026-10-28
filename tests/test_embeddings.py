"""
Tests for the local ONNX embedder (cortex/embeddings.py).

The reference test checks a synthetic sentence against independently pooled
ONNX output from a pinned, checksummed model and tokenizer. Numeric tolerances
allow small CPU/runtime rounding differences while catching pooling, model,
normalization, and tokenizer drift. An exact float-byte digest cannot provide
that portability.

Tests degrade to skips when the model or its ONNX backend is unavailable, so the
suite still runs on a machine that cannot embed.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path

from embeddings import DIM, MODEL_ID, Embedder, get_embedder, pack_vector, unpack_vector
from scripts.download_embedding_model import MODEL_FILES, MODEL_REVISION, file_sha256

REFERENCE = json.loads((Path(__file__).with_name("fixtures") / "embedding_reference.json").read_text())
GOLDEN_FIXTURE = REFERENCE["sentence"]


def _model_available() -> bool:
    """True only when the embedder can actually produce vectors.

    File presence is not sufficient: when the ONNX backend (onnxruntime /
    tokenizers) is missing, ``Embedder.available`` still reports True while
    ``embed()`` returns []. Gating on file presence alone therefore ran the
    assertions below against empty vectors and failed with confusing
    zero-length errors instead of skipping. Probe a real embed instead.
    """
    embedder = Embedder()
    if not embedder.available:
        return False
    try:
        return len(embedder.embed_one("probe")) == DIM
    except Exception:  # noqa: BLE001 - any backend failure means "not usable"
        return False


MODEL_AVAILABLE = _model_available()
if os.environ.get("CORTEX_REQUIRE_EMBEDDINGS") == "1" and not MODEL_AVAILABLE:
    raise RuntimeError("Embedding CI requires a working local model, ONNX Runtime, tokenizers, and NumPy")


@unittest.skipUnless(MODEL_AVAILABLE, "embedding model not installed")
class EmbedderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = get_embedder()

    def test_embedding_space_matches_pinned_reference(self):
        """Check the entire vector, allowing only minor numeric rounding."""
        import numpy as np

        vec = self.embedder.embed_one(GOLDEN_FIXTURE)
        self.assertEqual(len(vec), DIM)
        np.testing.assert_allclose(
            vec, REFERENCE["vector"], rtol=1e-3, atol=5e-4,
            err_msg="embedding model, tokenizer, pooling, or normalization drifted",
        )
        # Quantized graph fusions can move individual components slightly.
        # Direction must still agree closely with the independent reference.
        expected = np.asarray(REFERENCE["vector"])
        actual = np.asarray(vec)
        cosine = float(np.dot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected)))
        self.assertGreaterEqual(cosine, 0.99999)

    def test_model_and_tokenizer_artifacts_are_pinned(self):
        self.assertEqual(REFERENCE["artifacts"], MODEL_FILES)
        self.assertEqual(REFERENCE["model_revision"], MODEL_REVISION)
        for name, digest in MODEL_FILES.items():
            with self.subTest(artifact=name):
                self.assertEqual(file_sha256(self.embedder.model_dir / name), digest)

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
