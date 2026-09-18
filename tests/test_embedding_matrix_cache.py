"""Regression tests for embedding-matrix cache invalidation (2026-09-14).

``_embedding_matrix`` used to key its cache on ``(model_id, row_count)``. A
same-count vector replacement (``set_memory_embedding`` upsert, or raw SQL from
another connection) therefore served the STALE matrix: the replaced vector was
invisible to ``semantic_top_ids`` until some later row count change.

The cache is now keyed on the retrieval revision (connection-local write
triggers plus ``data_version`` for other connections). These tests pin the
observable contract: every insert, replacement, and deletion of a stored vector
is reflected on the next call, and unchanged data does not rebuild the matrix.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the package as ``cortex``)

from cortex.store import CortexStore

try:
    import numpy  # noqa: F401 - the embedding-matrix path requires numpy
    _MATRIX_AVAILABLE = True
except ImportError:  # pragma: no cover - bare environments (CI) have no numpy
    _MATRIX_AVAILABLE = False

_MATRIX_SKIP = unittest.skipUnless(
    _MATRIX_AVAILABLE, "embedding matrix requires numpy (optional dependency)"
)

MODEL = "test-model"
QUERY = [1.0, 0.0, 0.0, 0.0]


@_MATRIX_SKIP
class EmbeddingMatrixCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "cortex.db"
        self.store = CortexStore(self.db)
        self.ids = [
            self.store.add_memory("alpha vector marker one", kind="semantic")[0],
            self.store.add_memory("beta vector marker two", kind="semantic")[0],
            self.store.add_memory("gamma vector marker three", kind="semantic")[0],
        ]
        self.store.set_memory_embedding(self.ids[0], MODEL, [1.0, 0.0, 0.0, 0.0])
        self.store.set_memory_embedding(self.ids[1], MODEL, [0.0, 1.0, 0.0, 0.0])
        self.store.set_memory_embedding(self.ids[2], MODEL, [0.0, 0.0, 1.0, 0.0])

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def top(self):
        return [(str(mid), float(cos)) for mid, cos in self.store.semantic_top_ids(QUERY, MODEL, limit=5)]

    def test_replaced_vector_is_reflected(self) -> None:
        first = dict(self.top())
        self.assertAlmostEqual(first[self.ids[0]], 1.0)

        # Replace with an orthogonal vector; row count is unchanged.
        self.store.set_memory_embedding(self.ids[0], MODEL, [0.0, 0.0, 0.0, 1.0])

        second = dict(self.top())
        self.assertAlmostEqual(
            second[self.ids[0]],
            0.0,
            places=6,
            msg="a replaced embedding was not reflected — the matrix cache served stale vectors",
        )

    def test_inserted_vector_is_reflected(self) -> None:
        new_id = self.store.add_memory("delta vector marker four", kind="semantic")[0]
        self.store.set_memory_embedding(new_id, MODEL, [1.0, 0.0, 0.0, 0.0])
        ranked = dict(self.top())
        self.assertIn(new_id, ranked)
        self.assertAlmostEqual(ranked[new_id], 1.0)

    def test_deleted_vectors_are_reflected(self) -> None:
        self.assertTrue(self.top())
        self.store.delete_memory_embeddings(MODEL)
        self.assertEqual(self.top(), [])

    def test_other_connection_write_is_reflected(self) -> None:
        self.assertAlmostEqual(dict(self.top())[self.ids[0]], 1.0)
        # A raw SQLite client replaces the vector without going through the
        # store; SQLite's data_version is the only signal for that.
        external = sqlite3.connect(self.db)
        try:
            external.execute(
                "UPDATE memory_embeddings SET vector=? WHERE memory_id=? AND model_id=?",
                (_packed([0.0, 0.0, 0.0, 1.0]), self.ids[0], MODEL),
            )
            external.commit()
        finally:
            external.close()
        self.assertAlmostEqual(
            dict(self.top())[self.ids[0]],
            0.0,
            places=6,
            msg="a vector written by another connection was not reflected",
        )

    def test_unchanged_data_reuses_the_cached_matrix(self) -> None:
        self.top()  # warm
        statements: list[str] = []

        def tracer(statement: str) -> None:
            statements.append(statement)

        self.store._conn.set_trace_callback(tracer)
        try:
            self.top()
            self.top()
        finally:
            self.store._conn.set_trace_callback(None)
        rebuilds = sum(
            1 for statement in statements if "FROM memory_embeddings" in statement
        )
        self.assertEqual(
            rebuilds,
            0,
            f"unchanged data rebuilt the embedding matrix: {statements}",
        )


def _packed(vector):
    from cortex.embeddings import pack_vector

    return pack_vector(vector)


if __name__ == "__main__":
    unittest.main()
