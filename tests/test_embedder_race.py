"""Embedding readiness must never expose a session without its tokenizer.

Regression coverage for the audit's retrieval finding: the lazy loader
published ``_session`` before ``_tokenizer``, so a concurrent embed could see
"ready" and crash on ``None.encode_batch``.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401 - loads the flat ``cortex`` package

from cortex.embeddings import Embedder


class EmbedderLoadRaceTests(unittest.TestCase):
    def test_session_without_tokenizer_is_treated_as_not_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            embedder = Embedder(model_dir=Path(tmp))
            # Simulate the old partial-init window: a session exists but the
            # tokenizer was not published yet.
            embedder._session = object()  # type: ignore[assignment]
            embedder._tokenizer = None

            # Must not crash; with no model files the embedder reports unloaded.
            self.assertEqual(embedder.embed(["synthetic audit text"]), [])
            self.assertIsNone(embedder._tokenizer)


if __name__ == "__main__":
    unittest.main()