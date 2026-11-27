"""Tests for ``scripts/backfill_short_embeddings.py``.

The script's model-bearing path needs the ONNX embedder, which is optional here,
so these tests pin the parts that must be right *before* a single vector is
written: the module loads through the shared safety helpers, it counts only
missing short-form views, its batch commit is real, and ``--dry-run`` writes
nothing.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the repo package as `cortex`)

from cortex.store import CortexStore, short_view_model_id

MODEL = "test/backfill-shortform-model"


def _load_script():
    path = ROOT / "scripts" / "backfill_short_embeddings.py"
    name = "backfill_short_embeddings_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


backfill = _load_script()


class BackfillShortEmbeddingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="cortex-backfill-short-")
        self.db_path = Path(self.tmp) / "cortex.db"
        self.store = CortexStore(self.db_path)
        self.memory_id, _ = self.store.add_memory(
            "Every memory carries a short-form vector under a view-discriminated key.",
            kind="semantic",
        )

    def tearDown(self) -> None:
        self.store.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_shares_the_safety_helpers_with_the_full_text_backfill(self) -> None:
        self.assertEqual(backfill.EXIT_OK, 0)
        self.assertEqual(backfill.EXIT_LOW_MEMORY, 2)
        self.assertEqual(backfill.memory_is_low(None, 600), True)
        self.assertEqual(backfill.parse_mem_available_mb("MemAvailable: 2048 kB"), 2)
        self.assertEqual(
            list(backfill.chunk_sequence(["a", "b", "c"], 2)), [["a", "b"], ["c"]]
        )

    def test_counts_only_missing_short_views(self) -> None:
        self.assertEqual(backfill.count_missing(self.store, MODEL), 1)

        # A full-text vector alone must not satisfy the short-form count.
        self.store.set_memory_embedding(self.memory_id, MODEL, [1.0, 0.0])
        self.assertEqual(backfill.count_missing(self.store, MODEL), 1)

        self.store.set_short_form_embedding(self.memory_id, MODEL, [0.0, 1.0])
        self.assertEqual(backfill.count_missing(self.store, MODEL), 0)
        self.assertEqual(
            self.store.count_memory_embeddings(short_view_model_id(MODEL)), 1
        )

    def test_batch_commit_makes_the_short_vector_durable(self) -> None:
        self.store.set_short_form_embedding(self.memory_id, MODEL, [0.0, 1.0])
        backfill.commit(self.store)
        self.store._conn.rollback()  # a later rollback must not lose the batch

        self.assertEqual(
            self.store.count_memory_embeddings(short_view_model_id(MODEL)), 1
        )

    def test_dry_run_reports_without_writing(self) -> None:
        self.store.set_memory_embedding(self.memory_id, MODEL, [1.0, 0.0])
        code = backfill.main(
            [
                "--db",
                str(self.db_path),
                "--dry-run",
                "--quiet",
                "--min-available-mb",
                "1",
            ]
        )
        self.assertEqual(code, backfill.EXIT_OK)
        self.assertEqual(
            self.store.count_memory_embeddings(short_view_model_id(MODEL)),
            0,
            "a dry run wrote a short-form vector",
        )

    def test_missing_database_is_an_error(self) -> None:
        code = backfill.main(
            ["--db", str(Path(self.tmp) / "absent.db"), "--dry-run", "--quiet"]
        )
        self.assertEqual(code, backfill.EXIT_ERROR)

    def test_bad_batch_size_is_an_error(self) -> None:
        code = backfill.main(
            ["--db", str(self.db_path), "--batch-size", "0", "--quiet"]
        )
        self.assertEqual(code, backfill.EXIT_ERROR)


if __name__ == "__main__":
    unittest.main()
