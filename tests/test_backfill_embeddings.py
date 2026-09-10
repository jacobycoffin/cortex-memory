"""Unit tests for the PURE helpers in ``scripts/backfill_embeddings.py``.

These tests deliberately do not import the Cortex store, the ONNX embedder, or
any live database. They load the script module directly from its path and
exercise only the functions that make a decision without touching the model:

  * the memory-pressure check (``memory_is_low`` / ``parse_mem_available_mb``)
  * the batch-chunking helper (``chunk_sequence``) and the skip-aware page
    selector (``select_batch``)

Run: /usr/local/lib/hermes-agent/venv/bin/python -m pytest tests/test_backfill_embeddings.py -x -q
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_backfill_module():
    """Load scripts/backfill_embeddings.py without importing the Cortex package."""
    import sys

    path = _REPO_ROOT / "scripts" / "backfill_embeddings.py"
    spec = importlib.util.spec_from_file_location("backfill_embeddings_under_test", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


backfill = _load_backfill_module()


class TestParseMemAvailable(unittest.TestCase):
    def test_parses_typical_line(self) -> None:
        text = "MemTotal:       8000000 kB\nMemAvailable:    3794344 kB\nMemFree: 100 kB\n"
        # 3794344 kB // 1024 = 3705 MB
        self.assertEqual(backfill.parse_mem_available_mb(text), 3705)

    def test_picks_mem_available_not_mem_total(self) -> None:
        text = "MemTotal:       16000000 kB\nMemAvailable:     600000 kB\n"
        self.assertEqual(backfill.parse_mem_available_mb(text), 585)

    def test_missing_field_returns_none(self) -> None:
        self.assertIsNone(backfill.parse_mem_available_mb("MemTotal: 1000 kB\nMemFree: 5 kB\n"))

    def test_empty_text_returns_none(self) -> None:
        self.assertIsNone(backfill.parse_mem_available_mb(""))
        self.assertIsNone(backfill.parse_mem_available_mb(None))  # type: ignore[arg-type]

    def test_malformed_value_returns_none(self) -> None:
        self.assertIsNone(backfill.parse_mem_available_mb("MemAvailable: notanumber kB\n"))
        self.assertIsNone(backfill.parse_mem_available_mb("MemAvailable:\n"))

    def test_uses_whole_megabytes(self) -> None:
        self.assertEqual(backfill.parse_mem_available_mb("MemAvailable: 2048 kB\n"), 2)
        self.assertEqual(backfill.parse_mem_available_mb("MemAvailable: 1023 kB\n"), 0)

    def test_tolerant_of_extra_whitespace(self) -> None:
        self.assertEqual(backfill.parse_mem_available_mb("MemAvailable:\t 524288 kB\n"), 512)


class TestReadMemAvailable(unittest.TestCase):
    def test_reads_and_parses_a_real_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meminfo"
            path.write_text("MemTotal: 100 kB\nMemAvailable: 4096 kB\n", encoding="utf-8")
            self.assertEqual(backfill.read_mem_available_mb(path), 4)

    def test_missing_file_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                backfill.read_mem_available_mb(Path(tmp) / "does-not-exist")
            )


class TestMemoryPressureDecision(unittest.TestCase):
    def test_stops_when_below_threshold(self) -> None:
        self.assertTrue(backfill.memory_is_low(599, 600))

    def test_continues_when_above_threshold(self) -> None:
        self.assertFalse(backfill.memory_is_low(601, 600))

    def test_equal_to_threshold_continues(self) -> None:
        # The contract is "drops below the threshold", so equality is not low.
        self.assertFalse(backfill.memory_is_low(600, 600))

    def test_unknown_availability_is_treated_as_low(self) -> None:
        # If we cannot prove the box has headroom, we must not gamble on it.
        self.assertTrue(backfill.memory_is_low(None, 600))

    def test_zero_threshold_never_blocks_known_values(self) -> None:
        self.assertFalse(backfill.memory_is_low(0, 0))
        self.assertFalse(backfill.memory_is_low(1, 0))
        self.assertTrue(backfill.memory_is_low(None, 0))

    def test_default_threshold_is_what_the_contract_says(self) -> None:
        self.assertEqual(backfill.DEFAULT_MIN_AVAILABLE_MB, 600)
        self.assertEqual(backfill.DEFAULT_BATCH_SIZE, 64)


class TestChunkSequence(unittest.TestCase):
    def test_exact_multiple(self) -> None:
        self.assertEqual(
            list(backfill.chunk_sequence(["a", "b", "c", "d"], 2)),
            [["a", "b"], ["c", "d"]],
        )

    def test_remainder_lands_in_a_partial_last_batch(self) -> None:
        self.assertEqual(
            list(backfill.chunk_sequence(["a", "b", "c", "d", "e"], 2)),
            [["a", "b"], ["c", "d"], ["e"]],
        )

    def test_batch_larger_than_input(self) -> None:
        self.assertEqual(list(backfill.chunk_sequence(["a", "b"], 64)), [["a", "b"]])

    def test_empty_input_yields_nothing(self) -> None:
        self.assertEqual(list(backfill.chunk_sequence([], 8)), [])

    def test_batch_of_one(self) -> None:
        self.assertEqual(
            list(backfill.chunk_sequence(["x", "y"], 1)), [["x"], ["y"]]
        )

    def test_non_positive_batch_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            list(backfill.chunk_sequence(["a"], 0))
        with self.assertRaises(ValueError):
            list(backfill.chunk_sequence(["a"], -3))

    def test_yields_plain_lists_not_tuples(self) -> None:
        batches = list(backfill.chunk_sequence(["a", "b", "c"], 2))
        for batch in batches:
            self.assertIsInstance(batch, list)

    def test_covers_every_item_once(self) -> None:
        items = [str(i) for i in range(101)]
        flat = [item for batch in backfill.chunk_sequence(items, 7) for item in batch]
        self.assertEqual(flat, items)


class TestSelectBatch(unittest.TestCase):
    def test_drops_skipped_ids(self) -> None:
        self.assertEqual(
            backfill.select_batch(["a", "b", "c"], {"b"}, 3), ["a", "c"]
        )

    def test_caps_at_batch_size(self) -> None:
        self.assertEqual(
            backfill.select_batch(["a", "b", "c", "d"], set(), 2), ["a", "b"]
        )

    def test_empty_page(self) -> None:
        self.assertEqual(backfill.select_batch([], set(), 4), [])

    def test_all_skipped_yields_empty(self) -> None:
        self.assertEqual(backfill.select_batch(["a", "b"], {"a", "b"}, 4), [])

    def test_non_positive_batch_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            backfill.select_batch(["a"], set(), 0)


class TestExitCodeContract(unittest.TestCase):
    def test_exit_codes_are_distinct_and_as_documented(self) -> None:
        self.assertEqual(backfill.EXIT_OK, 0)
        self.assertEqual(backfill.EXIT_ERROR, 1)
        self.assertEqual(backfill.EXIT_LOW_MEMORY, 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
