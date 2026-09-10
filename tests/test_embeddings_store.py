"""Tests for local embedding storage on :class:`CortexStore`.

Every test runs against a throwaway SQLite file created with
``tempfile.mkdtemp()``; the live Cortex DB under ``/root/.hermes`` is never
opened here.
"""

from __future__ import annotations

import shutil
import struct
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the repo package as `cortex`)

from cortex.embeddings import DIM, pack_vector, unpack_vector
from cortex.store import CortexStore


MODEL = "BAAI/bge-small-en-v1.5"
MODEL_B = "test/other-model"


def make_vector(size: int = DIM, seed: float = 0.0) -> list[float]:
    """384 floats, each exactly representable in float32 (power-of-two scale)."""

    return [(i - size / 2 + seed) / 512.0 for i in range(size)]


class MemoryEmbeddingStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="cortex-embeddings-")
        self.db_path = Path(self.tmp) / "cortex.db"
        self.store = CortexStore(self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        shutil.rmtree(self.tmp, ignore_errors=True)


class SchemaContractTests(MemoryEmbeddingStoreTests):
    def test_memory_embeddings_table_has_required_shape(self) -> None:
        columns = self.store._conn.execute(
            "PRAGMA table_info(memory_embeddings)"
        ).fetchall()
        self.assertTrue(columns, "memory_embeddings table was not created")
        names = [row["name"] for row in columns]
        self.assertEqual(
            names,
            ["memory_id", "model_id", "dim", "vector", "created_at"],
        )
        types = {row["name"]: row["type"] for row in columns}
        self.assertEqual(types["memory_id"], "TEXT")
        self.assertEqual(types["model_id"], "TEXT")
        self.assertEqual(types["dim"], "INTEGER")
        self.assertEqual(types["vector"], "BLOB")
        self.assertEqual(types["created_at"], "TEXT")
        pk = sorted(
            (row["pk"], row["name"]) for row in columns if row["pk"]
        )
        self.assertEqual(pk, [(1, "memory_id"), (2, "model_id")])
        notnull = {row["name"] for row in columns if row["notnull"]}
        self.assertEqual(
            notnull, {"memory_id", "model_id", "dim", "vector", "created_at"}
        )


class EmbeddingRoundTripTests(MemoryEmbeddingStoreTests):
    def test_round_trip_384_float_vector(self) -> None:
        memory_id, _ = self.store.add_memory("Round trip embedding memory.")
        vector = make_vector()

        self.store.set_memory_embedding(memory_id, MODEL, vector)

        self.assertEqual(self.store.count_memory_embeddings(MODEL), 1)
        stored = self.store.get_memory_embeddings(MODEL, [memory_id])
        self.assertIn(memory_id, stored)
        read_back = stored[memory_id]
        self.assertEqual(len(read_back), DIM)
        # Values are chosen to be exactly float32-representable, so this is a
        # true exact-equality round trip, not an approximation.
        self.assertEqual(read_back, vector)
        self.assertEqual(sorted(stored.keys()), [memory_id])

        row = self.store._conn.execute(
            "SELECT dim, vector FROM memory_embeddings WHERE memory_id=? AND model_id=?",
            (memory_id, MODEL),
        ).fetchone()
        self.assertEqual(row["dim"], DIM)
        self.assertEqual(len(row["vector"]), DIM * 4)
        # Little-endian float32 bytes, exactly as pack_vector produces them.
        self.assertEqual(bytes(row["vector"]), pack_vector(vector))
        self.assertEqual(
            list(struct.unpack(f"<{DIM}f", bytes(row["vector"]))), vector
        )

    def test_get_memory_embeddings_with_empty_id_list_returns_empty_dict(self) -> None:
        memory_id, _ = self.store.add_memory("Embedding present for empty-list test.")
        self.store.set_memory_embedding(memory_id, MODEL, make_vector())

        self.assertEqual(self.store.get_memory_embeddings(MODEL, []), {})
        self.assertEqual(self.store.get_memory_embeddings(MODEL, ()), {})

    def test_get_memory_embeddings_omits_ids_without_a_row(self) -> None:
        embedded_id, _ = self.store.add_memory("This one has an embedding.")
        missing_id, _ = self.store.add_memory("This one has no embedding at all.")
        self.store.set_memory_embedding(embedded_id, MODEL, make_vector())

        result = self.store.get_memory_embeddings(MODEL, [embedded_id, missing_id])
        self.assertEqual(sorted(result.keys()), [embedded_id])
        self.assertNotIn(missing_id, result)


class MissingEmbeddingsTests(MemoryEmbeddingStoreTests):
    def test_missing_excludes_ids_that_already_have_an_embedding(self) -> None:
        first, _ = self.store.add_memory("First candidate for embedding.")
        second, _ = self.store.add_memory("Second candidate for embedding.")
        third, _ = self.store.add_memory("Third candidate for embedding.")

        self.store.set_memory_embedding(first, MODEL, make_vector())
        self.store.set_memory_embedding(second, MODEL, make_vector(seed=1.0))

        missing = self.store.memory_ids_missing_embeddings(MODEL)
        self.assertNotIn(first, missing)
        self.assertNotIn(second, missing)
        self.assertIn(third, missing)
        self.assertEqual(sorted(missing), [third])

    def test_missing_excludes_archived_memories(self) -> None:
        active_id, _ = self.store.add_memory("Active memory wanting an embedding.")
        cold_id, _ = self.store.add_memory(
            "Cold memory wanting an embedding.", state="cold"
        )
        archived_id, _ = self.store.add_memory(
            "Archived memory wanting an embedding.", state="archived"
        )
        self.assertEqual(self.store.get_memory(archived_id)["state"], "archived")

        missing = self.store.memory_ids_missing_embeddings(MODEL)
        self.assertIn(active_id, missing)
        self.assertIn(cold_id, missing)
        self.assertNotIn(archived_id, missing)
        self.assertEqual(sorted(missing), sorted([active_id, cold_id]))

    def test_missing_respects_limit_and_is_deterministic(self) -> None:
        for index in range(4):
            self.store.add_memory(f"Batch memory number {index} for limit test.")

        first_pass = self.store.memory_ids_missing_embeddings(MODEL, limit=10)
        self.assertEqual(len(first_pass), 4)
        self.assertEqual(
            self.store.memory_ids_missing_embeddings(MODEL, limit=10), first_pass
        )
        self.assertEqual(len(self.store.memory_ids_missing_embeddings(MODEL, limit=2)), 2)

    def test_missing_is_scoped_per_model(self) -> None:
        memory_id, _ = self.store.add_memory("Model-scoped missing embedding.")
        other_id, _ = self.store.add_memory("Untouched by the other model.")

        self.store.set_memory_embedding(memory_id, MODEL, make_vector())

        self.assertNotIn(memory_id, self.store.memory_ids_missing_embeddings(MODEL))
        self.assertIn(memory_id, self.store.memory_ids_missing_embeddings(MODEL_B))
        self.assertIn(other_id, self.store.memory_ids_missing_embeddings(MODEL_B))


class UpsertTests(MemoryEmbeddingStoreTests):
    def test_storing_twice_upserts_instead_of_raising(self) -> None:
        memory_id, _ = self.store.add_memory("Upsert embedding memory.")
        first = make_vector()
        second = make_vector(seed=7.0)
        self.assertNotEqual(first, second)

        self.store.set_memory_embedding(memory_id, MODEL, first)
        self.store.set_memory_embedding(memory_id, MODEL, second)  # must not raise

        self.assertEqual(self.store.count_memory_embeddings(MODEL), 1)
        self.assertEqual(
            self.store.get_memory_embeddings(MODEL, [memory_id])[memory_id], second
        )
        rows = self.store._conn.execute(
            "SELECT COUNT(*) AS total FROM memory_embeddings WHERE memory_id=? AND model_id=?",
            (memory_id, MODEL),
        ).fetchone()
        self.assertEqual(rows["total"], 1)


class CountAndDeleteTests(MemoryEmbeddingStoreTests):
    def test_count_and_delete_behave_as_specified(self) -> None:
        ids = [
            self.store.add_memory(f"Counted embedding memory {index}.")[0]
            for index in range(3)
        ]
        for index, memory_id in enumerate(ids):
            self.store.set_memory_embedding(memory_id, MODEL, make_vector(seed=index))

        self.assertEqual(self.store.count_memory_embeddings(MODEL), 3)
        self.assertEqual(self.store.count_memory_embeddings("never/used"), 0)

        deleted = self.store.delete_memory_embeddings(MODEL)
        self.assertEqual(deleted, 3)
        self.assertEqual(self.store.count_memory_embeddings(MODEL), 0)
        self.assertEqual(self.store.get_memory_embeddings(MODEL, ids), {})
        # Deleting again is a no-op, not an error.
        self.assertEqual(self.store.delete_memory_embeddings(MODEL), 0)


class MultipleModelTests(MemoryEmbeddingStoreTests):
    def test_second_model_id_is_stored_independently(self) -> None:
        memory_id, _ = self.store.add_memory("One memory, two embedding models.")
        vector_a = make_vector(seed=0.0)
        vector_b = make_vector(seed=100.0)

        self.store.set_memory_embedding(memory_id, MODEL, vector_a)
        self.store.set_memory_embedding(memory_id, MODEL_B, vector_b)

        self.assertEqual(self.store.count_memory_embeddings(MODEL), 1)
        self.assertEqual(self.store.count_memory_embeddings(MODEL_B), 1)
        self.assertEqual(
            self.store.get_memory_embeddings(MODEL, [memory_id])[memory_id], vector_a
        )
        self.assertEqual(
            self.store.get_memory_embeddings(MODEL_B, [memory_id])[memory_id], vector_b
        )
        # A model_id that was never written sees no embeddings at all.
        self.assertEqual(self.store.get_memory_embeddings("third/model", [memory_id]), {})

        # Deleting one model leaves the other intact.
        self.assertEqual(self.store.delete_memory_embeddings(MODEL), 1)
        self.assertEqual(self.store.count_memory_embeddings(MODEL), 0)
        self.assertEqual(self.store.count_memory_embeddings(MODEL_B), 1)
        self.assertEqual(
            self.store.get_memory_embeddings(MODEL_B, [memory_id])[memory_id], vector_b
        )

    def test_unpack_of_empty_blob_is_empty_vector(self) -> None:
        # Guards the helper contract the read path depends on.
        self.assertEqual(unpack_vector(b""), [])
        self.assertEqual(unpack_vector(pack_vector([])), [])


if __name__ == "__main__":
    unittest.main()
