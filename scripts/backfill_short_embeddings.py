#!/usr/bin/env python3
"""Backfill the SHORT-FORM embedding view for existing Cortex memories.

WHY THIS EXISTS
---------------
Memories are long (median ~19 tokens, p90 ~82) while real queries are short
(median ~7), so a single vector per memory over full content places many of the
memories that were actually used outside a 200-wide candidate cut. Every memory
therefore carries a second vector over its short form — the presentation's
``display_title`` + ``display_summary`` (``memory_presentations``), falling back
to the full ``content`` when no presentation exists — stored in the same
``memory_embeddings`` table under the view-discriminated key
``f"{model_id}#short"``.

New memories get both vectors on the write path. This script is the **one-off,
resumable** backfill for the corpus that predates the view; it only fills short
vectors that are missing and never rewrites full-text ones.

SAFETY CONTRACT (inherited from ``scripts/backfill_embeddings.py``)
-------------------------------------------------------------------
An earlier unbounded embedding run on the 2-core / ~8 GB box drove the kernel
OOM killer, so this tool keeps the same shape:

  * writes in SMALL batches (``--batch-size``, default 64);
  * checks ``MemAvailable`` BEFORE every batch and stops with exit 2 when it
    falls below ``--min-available-mb`` (default 600);
  * commits once per batch, so a crash or kill costs at most one batch;
  * is RESUMABLE: it re-queries memories that still lack a short vector, so a
    re-run skips finished work and continues;
  * holds only the current batch's vectors and drops them before the next.

DRY RUN
-------
``--dry-run`` reports how many memories still need a short vector and exits 0
without writing anything.

USAGE
-----
    python scripts/backfill_short_embeddings.py --dry-run
    python scripts/backfill_short_embeddings.py --batch-size 64

EXIT CODES
----------
    0  complete: no eligible memory is left without a short-form vector
    1  error (bad arguments, missing DB/model, unexpected exception)
    2  stopped early: ``MemAvailable`` fell below ``--min-available-mb``
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
from pathlib import Path
from typing import Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Pure helpers shared with the full-text backfill: memory gate, batching,
# page selection and the package bootstrap. Re-using them keeps the two tools
# from drifting apart on the safety shape.
import backfill_embeddings as _common  # noqa: E402  (path set above)

EXIT_OK = _common.EXIT_OK
EXIT_ERROR = _common.EXIT_ERROR
EXIT_LOW_MEMORY = _common.EXIT_LOW_MEMORY

DEFAULT_DB = _common.DEFAULT_DB
DEFAULT_BATCH_SIZE = _common.DEFAULT_BATCH_SIZE
DEFAULT_MIN_AVAILABLE_MB = _common.DEFAULT_MIN_AVAILABLE_MB

parse_mem_available_mb = _common.parse_mem_available_mb
read_mem_available_mb = _common.read_mem_available_mb
memory_is_low = _common.memory_is_low
chunk_sequence = _common.chunk_sequence
select_batch = _common.select_batch
_ensure_cortex_importable = _common._ensure_cortex_importable
_log = _common._log
_stop = _common._stop

# A generous ceiling for the one counting query; the values it returns are ids
# (strings), never vectors.
_COUNT_LIMIT = 5_000_000


def count_missing(store: object, model_id: str) -> int:
    """How many recallable memories still lack a short-form vector."""

    return len(store.memory_ids_missing_short_embeddings(model_id, limit=_COUNT_LIMIT))


def commit(store: object) -> None:
    """Commit the store connection at a batch boundary.

    ``set_memory_embedding`` joins the caller's transaction and does not commit
    on its own (the store's write path must stay atomic for its callers), so the
    batch boundary commit is this script's job.
    """

    conn = getattr(store, "_conn", None)
    if conn is None:
        raise RuntimeError("CortexStore has no connection to commit")
    conn.commit()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill the short-form (title+summary) embedding view for Cortex's "
            "existing memories. Fills only missing short vectors; safe by default."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", default=DEFAULT_DB, help="Cortex SQLite database.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Memories embedded per batch (each row commits).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many vectors (0 = everything missing).",
    )
    parser.add_argument(
        "--min-available-mb",
        type=int,
        default=DEFAULT_MIN_AVAILABLE_MB,
        help="Stop (exit 2) when MemAvailable drops below this many MB.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many memories need a short vector, then exit.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    parser.add_argument(
        "--model-dir", default=None, help="Override the embedding model directory."
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="ONNX intra/inter-op threads."
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if args.batch_size < 1:
        _stop(f"error: --batch-size must be >= 1 (got {args.batch_size})")
        return EXIT_ERROR
    if args.limit < 0:
        _stop(f"error: --limit must be >= 0 (got {args.limit})")
        return EXIT_ERROR

    db_path = Path(args.db).expanduser()
    if not db_path.is_file():
        _stop(f"error: database not found: {db_path}")
        return EXIT_ERROR

    quiet = bool(args.quiet)

    # Memory gate before we open the store or load the model.
    mem_mb = read_mem_available_mb()
    if memory_is_low(mem_mb, args.min_available_mb):
        _stop(
            f"STOP: MemAvailable={mem_mb} MB is below --min-available-mb="
            f"{args.min_available_mb} MB; nothing written, re-run when the box has headroom."
        )
        return EXIT_LOW_MEMORY

    _ensure_cortex_importable()
    from cortex.embeddings import MODEL_ID, Embedder  # noqa: E402 (after bootstrap)
    from cortex.store import CortexStore, short_view_model_id  # noqa: E402 (after bootstrap)

    store = CortexStore(db_path)
    model_id = MODEL_ID
    short_key = short_view_model_id(model_id)

    if args.dry_run:
        remaining = count_missing(store, model_id)
        _log(
            f"dry run: {remaining} memories need a short-form ({short_key}) vector; "
            f"nothing written.",
            quiet,
        )
        return EXIT_OK

    initial_missing = count_missing(store, model_id)
    if initial_missing == 0:
        _log(f"nothing to do: every eligible memory already has a {short_key} vector.", quiet)
        return EXIT_OK

    embedder = Embedder(model_dir=args.model_dir, threads=args.threads)
    if not getattr(embedder, "available", True):
        _stop(
            f"error: embedding model unavailable under {embedder.model_dir} "
            f"({getattr(embedder, 'load_error', 'unknown reason')})"
        )
        return EXIT_ERROR

    _log(
        f"short-form backfill start: model={short_key} missing={initial_missing} "
        f"batch_size={args.batch_size} min_available_mb={args.min_available_mb} "
        f"MemAvailable={mem_mb} MB",
        quiet,
    )

    written = 0
    batch_no = 0
    skipped: set[str] = set()

    while True:
        if args.limit > 0 and written >= args.limit:
            break

        mem_mb = read_mem_available_mb()
        if memory_is_low(mem_mb, args.min_available_mb):
            _stop(
                f"STOP after {batch_no} batch(es): MemAvailable={mem_mb} MB is below "
                f"--min-available-mb={args.min_available_mb} MB. Committed work is kept; "
                f"re-run to resume."
            )
            return EXIT_LOW_MEMORY

        fetch_limit = args.batch_size + len(skipped)
        if args.limit > 0:
            fetch_limit += max(0, args.limit - written)
        fetched = store.memory_ids_missing_short_embeddings(
            model_id, limit=max(1, fetch_limit)
        )
        ids = select_batch(fetched, skipped, args.batch_size)
        if args.limit > 0:
            ids = ids[: max(0, args.limit - written)]
        if not ids:
            break

        batch_no += 1
        texts = store.short_form_texts(ids)
        payload = [(memory_id, texts.get(memory_id, "")) for memory_id in ids]
        embeddable = [
            (memory_id, text)
            for memory_id, text in payload
            if isinstance(text, str) and text.strip()
        ]
        for memory_id, text in payload:
            if not (isinstance(text, str) and text.strip()):
                if memory_id not in skipped:
                    skipped.add(memory_id)
                    _log(
                        f"  skip {memory_id}: no short-form text (no vector written)",
                        quiet,
                    )

        vectors: list[list[float]] = []
        if embeddable:
            vectors = embedder.embed([text for _, text in embeddable])
            if len(vectors) != len(embeddable):
                _stop(
                    f"error: embedder returned {len(vectors)} vectors for "
                    f"{len(embeddable)} texts — model/runtime failure; aborting."
                )
                return EXIT_ERROR
            for (memory_id, _text), vector in zip(embeddable, vectors):
                if not vector:
                    _stop(f"error: empty vector for {memory_id}; aborting before a bad write.")
                    return EXIT_ERROR
                store.set_short_form_embedding(memory_id, model_id, vector)
                written += 1
            commit(store)

        remaining = max(0, initial_missing - written - len(skipped))
        _log(
            f"batch {batch_no}: embedded={written}/{initial_missing} remaining={remaining} "
            f"MemAvailable={mem_mb} MB",
            quiet,
        )

        del vectors
        del embeddable
        del payload
        del texts
        del ids
        vectors = embeddable = payload = texts = ids = None  # type: ignore[assignment]
        gc.collect()

    remaining = max(0, initial_missing - written - len(skipped))
    _log(
        f"done: embedded={written} skipped={len(skipped)} remaining={remaining} "
        f"MemAvailable={read_mem_available_mb()} MB",
        quiet,
    )
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        _stop("interrupted by user; committed batches are kept.")
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - surface any failure as exit 1
        _stop(f"error: {type(exc).__name__}: {exc}")
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
