"""Tests for the short-form (title+summary) embedding view.

Long memories vs short queries is the motivating measurement: one vector per
memory over full content cannot place many memories inside a 200-wide cut, so
:class:`CortexStore` also stores a vector over the memory's presentation
(``display_title`` + ``display_summary``) under the view-discriminated key
``f"{model_id}#short"``. These tests pin the storage contract, the text the
second view is built from, and the merged top-id query retrieval will consume.

Every test runs against a throwaway SQLite file; the live Cortex database is
never opened here, and no test needs the ONNX embedder.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import ROOT  # noqa: F401  (loads the repo package as `cortex`)

from cortex.store import (
    SHORT_VIEW_SUFFIX,
    CortexStore,
    is_short_view_model_id,
    short_view_model_id,
)

try:  # the merged/top-id queries need numpy, exactly like semantic_top_ids
    import numpy  # noqa: F401

    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - bare CI legs have no numpy
    _NUMPY_AVAILABLE = False

_NEEDS_NUMPY = unittest.skipUnless(
    _NUMPY_AVAILABLE, "short-form top-id queries require numpy"
)

MODEL = "test/shortform-model"
OTHER_MODEL = "test/other-model"


def _unit(index: int, size: int = 4) -> list[float]:
    """One-hot-ish float32-exact vector with a 1.0 at *index*."""

    vector = [0.0] * size
    vector[index] = 1.0
    return vector


class ViewKeyTests(unittest.TestCase):
    def test_suffix_is_stable(self) -> None:
        self.assertEqual(SHORT_VIEW_SUFFIX, "#short")

    def test_short_view_key_is_discriminated(self) -> None:
        self.assertEqual(short_view_model_id(MODEL), f"{MODEL}#short")
        self.assertNotEqual(short_view_model_id(MODEL), MODEL)

    def test_short_view_key_is_idempotent(self) -> None:
        once = short_view_model_id(MODEL)
        self.assertEqual(short_view_model_id(once), once)
        self.assertTrue(is_short_view_model_id(once))

    def test_full_model_id_is_not_a_short_view(self) -> None:
        self.assertFalse(is_short_view_model_id(MODEL))
        self.assertFalse(is_short_view_model_id(""))

    def test_empty_model_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            short_view_model_id("")


class ShortFormStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="cortex-shortform-")
        self.store = CortexStore(Path(self.tmp) / "cortex.db")
        self.memory_id, _created = self.store.add_memory(
            "Cortex stores two vectors per memory so short queries can match the "
            "presentation instead of the long full-text content.",
            kind="semantic",
        )

    def tearDown(self) -> None:
        self.store.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- helpers -------------------------------------------------------
    def _write_presentation(self, title: str, summary: str, memory_id: str | None = None) -> None:
        target = memory_id or self.memory_id
        self.store._conn.execute(
            """INSERT INTO memory_presentations(
                 memory_id,display_title,display_summary,applies_when,retention_reason,
                 readability_flags_json,presentation_method,presentation_version,
                 source_digest,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(memory_id) DO UPDATE SET
                 display_title=excluded.display_title,
                 display_summary=excluded.display_summary,
                 updated_at=excluded.updated_at""",
            (
                target,
                title,
                summary,
                "always",
                "test fixture",
                "[]",
                "test",
                "1",
                "digest",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        self.store._conn.commit()

    def _embedding_rows(self) -> list[tuple[str, str]]:
        rows = self.store._conn.execute(
            "SELECT memory_id, model_id FROM memory_embeddings ORDER BY model_id"
        ).fetchall()
        return [(str(row["memory_id"]), str(row["model_id"])) for row in rows]

    # -- writes --------------------------------------------------------
    def test_short_vector_is_written_under_its_own_key(self) -> None:
        full = _unit(0)
        short = _unit(1)
        self.store.set_memory_embedding(self.memory_id, MODEL, full, short)

        stored = self.store.get_memory_embeddings(MODEL, [self.memory_id])
        stored_short = self.store.get_memory_embeddings(
            short_view_model_id(MODEL), [self.memory_id]
        )
        self.assertIn(self.memory_id, stored)
        self.assertIn(self.memory_id, stored_short)
        self.assertEqual(stored[self.memory_id], full)
        self.assertEqual(stored_short[self.memory_id], short)
        self.assertNotEqual(
            stored[self.memory_id],
            stored_short[self.memory_id],
            "the two views must not overwrite each other",
        )

    def test_full_only_write_leaves_no_short_row(self) -> None:
        self.store.set_memory_embedding(self.memory_id, MODEL, _unit(0))

        self.assertEqual(self._embedding_rows(), [(self.memory_id, MODEL)])
        self.assertIn(
            self.memory_id,
            self.store.memory_ids_missing_short_embeddings(MODEL, limit=10),
        )

    def test_short_only_write_does_not_touch_full_row(self) -> None:
        self.store.set_memory_embedding(self.memory_id, MODEL, _unit(0))
        self.store.set_short_form_embedding(self.memory_id, MODEL, _unit(2))

        stored = self.store.get_memory_embeddings(MODEL, [self.memory_id])
        stored_short = self.store.get_memory_embeddings(
            short_view_model_id(MODEL), [self.memory_id]
        )
        self.assertEqual(stored[self.memory_id], _unit(0))
        self.assertEqual(stored_short[self.memory_id], _unit(2))

    def test_short_view_is_per_model(self) -> None:
        self.store.set_memory_embedding(self.memory_id, MODEL, _unit(0), _unit(1))
        self.store.set_memory_embedding(self.memory_id, OTHER_MODEL, _unit(2), _unit(3))

        rows = dict(
            (model, vector)
            for model, vector in (
                (model_id, self.store.get_memory_embeddings(model_id, [self.memory_id]))
                for model_id in (
                    MODEL,
                    OTHER_MODEL,
                    short_view_model_id(MODEL),
                    short_view_model_id(OTHER_MODEL),
                )
            )
        )
        self.assertEqual(rows[MODEL][self.memory_id], _unit(0))
        self.assertEqual(rows[OTHER_MODEL][self.memory_id], _unit(2))
        self.assertEqual(rows[short_view_model_id(MODEL)][self.memory_id], _unit(1))
        self.assertEqual(rows[short_view_model_id(OTHER_MODEL)][self.memory_id], _unit(3))

    def test_embedding_write_joins_the_callers_transaction(self) -> None:
        """A rollback must leave no row: the write cannot commit on its own."""

        with self.assertRaises(RuntimeError):
            with self.store.transaction() as conn:
                self.store.set_memory_embedding(
                    self.memory_id, MODEL, _unit(0), _unit(1)
                )
                self.assertEqual(len(self._embedding_rows()), 2)
                raise RuntimeError("caller rolled back")

        self.assertEqual(
            self._embedding_rows(),
            [],
            "the embedding write committed inside a rolled-back transaction",
        )

    # -- short-form text ------------------------------------------------
    def test_short_form_text_uses_title_and_summary(self) -> None:
        self._write_presentation("Short view", "Title and summary are indexed together.")

        text = self.store.short_form_texts([self.memory_id])[self.memory_id]
        self.assertIn("Short view", text)
        self.assertIn("Title and summary are indexed together.", text)
        self.assertNotIn("two vectors per memory", text)

    def test_short_form_text_does_not_repeat_an_identical_summary(self) -> None:
        self._write_presentation("Just a title", "Just a title")

        text = self.store.short_form_texts([self.memory_id])[self.memory_id]
        self.assertEqual(text, "Just a title")

    def test_short_form_text_falls_back_to_content_without_presentation(self) -> None:
        self.store._conn.execute(
            "DELETE FROM memory_presentations WHERE memory_id=?", (self.memory_id,)
        )
        self.store._conn.commit()

        text = self.store.short_form_texts([self.memory_id])[self.memory_id]
        content = self.store.get_memory(self.memory_id)["content"]
        self.assertEqual(text, content)

    def test_short_form_text_falls_back_when_presentation_is_blank(self) -> None:
        self._write_presentation("", "")

        text = self.store.short_form_texts([self.memory_id])[self.memory_id]
        self.assertTrue(text.strip())
        self.assertIn("two vectors per memory", text)

    def test_short_form_texts_skips_unknown_ids(self) -> None:
        self.assertEqual(self.store.short_form_texts(["nope"]), {})
        self.assertEqual(self.store.short_form_texts([]), {})

    # -- missing-vector query ------------------------------------------
    def test_missing_short_embeddings_excludes_written_views(self) -> None:
        other, _ = self.store.add_memory("second memory for the missing view", kind="semantic")
        missing = self.store.memory_ids_missing_short_embeddings(MODEL, limit=10)
        self.assertIn(self.memory_id, missing)
        self.assertIn(other, missing)

        self.store.set_memory_embedding(self.memory_id, MODEL, _unit(0), _unit(1))
        missing = self.store.memory_ids_missing_short_embeddings(MODEL, limit=10)
        self.assertNotIn(self.memory_id, missing)
        self.assertIn(other, missing)

    # -- merged top ids -------------------------------------------------
    @_NEEDS_NUMPY
    def test_merged_query_finds_a_short_only_match(self) -> None:
        full_id = self.memory_id
        short_id, _ = self.store.add_memory(
            "A memory whose presentation says 'weekly budget export' although its "
            "stored content is long and does not use those words.",
            kind="semantic",
        )
        self._write_presentation(
            "Weekly budget export",
            "Spreadsheet of last week's spend.",
            memory_id=short_id,
        )
        self.store.set_memory_embedding(full_id, MODEL, _unit(0), _unit(3))
        self.store.set_memory_embedding(short_id, MODEL, _unit(3), _unit(0))

        query = _unit(0)
        self.assertEqual(
            [memory for memory, _ in self.store.semantic_top_ids(query, MODEL, limit=10)][0],
            full_id,
        )
        merged = dict(self.store.semantic_top_ids_merged(query, MODEL, limit=10))
        self.assertIn(full_id, merged)
        self.assertIn(short_id, merged)

    @_NEEDS_NUMPY
    def test_merged_query_dedupes_and_keeps_best_similarity(self) -> None:
        # Full view: cosine 0.6 against the query, short view: cosine 1.0.
        self.store.set_memory_embedding(
            self.memory_id, MODEL, [0.6, 0.8, 0.0, 0.0], _unit(0)
        )

        merged = self.store.semantic_top_ids_merged(_unit(0), MODEL, limit=10)
        ids = [memory for memory, _ in merged]
        self.assertEqual(
            ids.count(self.memory_id), 1, "a memory found by both views appeared twice"
        )
        self.assertAlmostEqual(dict(merged)[self.memory_id], 1.0, places=6)

    @_NEEDS_NUMPY
    def test_merged_query_is_sorted_and_bounded(self) -> None:
        first, _ = self.store.add_memory("first", kind="semantic")
        second, _ = self.store.add_memory("second", kind="semantic")
        third, _ = self.store.add_memory("third", kind="semantic")
        self.store.set_memory_embedding(first, MODEL, _unit(0), _unit(1))
        self.store.set_memory_embedding(second, MODEL, _unit(1), _unit(0))
        self.store.set_memory_embedding(third, MODEL, _unit(2), _unit(2))

        merged = self.store.semantic_top_ids_merged(_unit(0), MODEL, limit=2)
        self.assertEqual(len(merged), 2)
        scores = [score for _, score in merged]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual({memory for memory, _ in merged}, {first, second})

    @_NEEDS_NUMPY
    def test_merged_query_is_empty_without_vectors(self) -> None:
        self.assertEqual(
            self.store.semantic_top_ids_merged(_unit(0), "test/unembedded-model", limit=5),
            [],
        )
        self.assertEqual(self.store.semantic_top_ids_merged(_unit(0), MODEL, limit=0), [])

    @_NEEDS_NUMPY
    def test_short_view_replacement_is_reflected_immediately(self) -> None:
        self.store.set_memory_embedding(self.memory_id, MODEL, _unit(3), _unit(0))
        first = dict(self.store.semantic_top_ids_merged(_unit(0), MODEL, limit=5))
        self.assertAlmostEqual(first[self.memory_id], 1.0, places=6)

        # Same row count, new vector: the matrix cache must not serve the old one.
        self.store.set_memory_embedding(self.memory_id, MODEL, _unit(3), _unit(2))
        second = dict(self.store.semantic_top_ids_merged(_unit(0), MODEL, limit=5))
        self.assertAlmostEqual(second[self.memory_id], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
