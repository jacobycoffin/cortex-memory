#!/usr/bin/env python3
"""One-off, resumable, memory-bounded embedding backfill for Cortex.

WHY THIS EXISTS
---------------
Cortex is gaining a local embedding signal. Every memory that predates the
feature needs one embedding, once; new memories get theirs incrementally on the
write path. This script is the **one-off backfill** for the existing corpus.

SAFETY CONTRACT (do not weaken without re-reading this)
-------------------------------------------------------
An earlier *unbounded* embedding run on this 2-core / ~8 GB box (with ~4 GB
already held by services) drove the kernel OOM killer, which shot
``chrome-headless`` twice and restarted the Hermes gateway. The backfill must
never repeat that. Therefore this tool:

  * writes in SMALL batches (``--batch-size``, default 64);
  * checks ``MemAvailable`` in ``/proc/meminfo`` BEFORE every batch and, if it
    falls below ``--min-available-mb`` (default 600), logs a clear STOP line and
    exits 2 instead of pressing on;
  * commits after every batch, so a crash or a kill costs at most one batch;
  * is RESUMABLE: it re-queries the memories that still lack an embedding each
    iteration, so a re-run skips finished work and continues — it never
    restarts from scratch;
  * never holds embeddings for the whole corpus: only the current batch's
    vectors are alive, and they are dropped and ``gc.collect()`` runs before the
    next batch.

The embedding model also runs with a single ONNX thread (see
``cortex.embeddings``), which keeps CPU pressure off the co-resident services.

DRY RUN
-------
``--dry-run`` prints how many memories still need an embedding and exits 0
without writing any embedding. (Opening ``CortexStore`` itself runs the normal,
idempotent schema/migration path that every Cortex open performs; it writes no
embeddings.)

USAGE
-----
    python scripts/backfill_embeddings.py --dry-run
    python scripts/backfill_embeddings.py --batch-size 64 --min-available-mb 600

EXIT CODES
----------
    0  complete: no eligible memory is left without an embedding
    1  error (bad arguments, missing DB/model, unexpected exception)
    2  stopped early: ``MemAvailable`` fell below ``--min-available-mb``
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
from pathlib import Path
from typing import Iterator, Sequence

def _default_db() -> str:
    """Cortex's database, resolved from the plugin's HERMES_HOME convention."""
    import os

    home = os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")
    return str(Path(home).expanduser() / "cortex" / "cortex.db")


DEFAULT_DB = _default_db()
DEFAULT_BATCH_SIZE = 64
DEFAULT_MIN_AVAILABLE_MB = 600
MEMINFO_PATH = "/proc/meminfo"

# A generous ceiling for the one counting query; the values it returns are ids
# (strings), never vectors, and Cortex is a few thousand memories, not millions.
_COUNT_LIMIT = 5_000_000

# SQLite's host-parameter cap is 999; keep IN(...) lookups well under it.
_ID_CHUNK = 400

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_LOW_MEMORY = 2


# --------------------------------------------------------------------------
# Pure helpers (no DB, no model — unit-tested in tests/test_backfill_embeddings.py)
# --------------------------------------------------------------------------


def parse_mem_available_mb(meminfo_text: str) -> int | None:
    """Return ``MemAvailable`` in whole megabytes from ``/proc/meminfo`` text.

    Returns ``None`` when the field is absent or unparsable so the caller can
    decide how to react rather than silently reading a wrong number.
    """
    for line in (meminfo_text or "").splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) < 2:
                return None
            try:
                kb = float(parts[1])
            except (TypeError, ValueError):
                return None
            if kb < 0:
                return None
            return int(kb // 1024)
    return None


def read_mem_available_mb(path: str | Path = MEMINFO_PATH) -> int | None:
    """Read and parse ``MemAvailable`` from *path*; ``None`` if unreadable."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    return parse_mem_available_mb(text)


def memory_is_low(mem_available_mb: int | None, min_available_mb: int) -> bool:
    """True when the run must stop for memory pressure.

    Stops when available memory is **below** the threshold (equal is fine).
    Unknown availability (``None``) is treated as low: if we cannot prove the
    box has headroom, we do not gamble the gateway on it.
    """
    if mem_available_mb is None:
        return True
    return int(mem_available_mb) < int(min_available_mb)


def chunk_sequence(items: Sequence[str], batch_size: int) -> Iterator[list[str]]:
    """Yield successive lists of at most *batch_size* items.

    Raises ``ValueError`` for a non-positive ``batch_size`` rather than silently
    looping forever or returning nothing.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    items = list(items)
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def select_batch(
    fetched: Sequence[str], skipped: set[str], batch_size: int
) -> list[str]:
    """Pick up to *batch_size* ids from a fetched page, dropping *skipped*.

    ``memory_ids_missing_embeddings`` returns a deterministic order, so ids we
    deliberately skip (e.g. empty content we refuse to embed) would otherwise
    reappear at the head of every page and stall the loop. Filtering them here
    lets the caller over-fetch by ``len(skipped)`` and still make progress.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    chosen: list[str] = []
    for memory_id in fetched:
        if memory_id in skipped:
            continue
        chosen.append(memory_id)
        if len(chosen) >= batch_size:
            break
    return chosen


# --------------------------------------------------------------------------
# Import bootstrap — reuse this checkout's `cortex` package
# --------------------------------------------------------------------------


def _ensure_cortex_importable() -> Path:
    """Make THIS checkout importable as the ``cortex`` package.

    The checkout directory is ``hermes-cortex-memory`` (hyphenated), so a plain
    ``import cortex`` cannot find it. We load ``<repo>/__init__.py`` under the
    package name ``cortex`` with submodule search rooted at the checkout — the
    same trick ``tests/_bootstrap.py`` uses. This deliberately prefers the code
    under development over any installed copy of the plugin.
    """
    repo_root = Path(__file__).resolve().parents[1]
    existing = sys.modules.get("cortex")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file and Path(existing_file).resolve().parent == repo_root:
            return repo_root
    if str(repo_root.parent) not in sys.path:
        sys.path.insert(0, str(repo_root.parent))
    spec = importlib.util.spec_from_file_location(
        "cortex",
        repo_root / "__init__.py",
        submodule_search_locations=[str(repo_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load the Cortex package from {repo_root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cortex"] = module
    spec.loader.exec_module(module)
    return repo_root


# --------------------------------------------------------------------------
# Store helpers
# --------------------------------------------------------------------------


def load_memory_texts(
    store: object, memory_ids: Sequence[str], chunk_size: int = _ID_CHUNK
) -> dict[str, str]:
    """Return ``{memory_id: content}`` for *memory_ids*.

    Prefers ``CortexStore.get_memory_texts`` when present; otherwise reads
    ``memories.content`` with a chunked SELECT **on the store's own connection**
    (never a separately opened one). Falls back to the public ``get_memory``
    getter only if the connection is not reachable.
    """
    if not memory_ids:
        return {}

    getter = getattr(store, "get_memory_texts", None)
    if callable(getter):
        result = getter(list(memory_ids))
        if result is not None:
            return {str(k): (v if isinstance(v, str) else "") for k, v in result.items()}

    conn = getattr(store, "_conn", None)
    if conn is not None:
        texts: dict[str, str] = {}
        for chunk in chunk_sequence(list(memory_ids), chunk_size):
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT id, content FROM memories WHERE id IN ({placeholders})",
                tuple(chunk),
            ).fetchall()
            for row in rows:
                content = row["content"]
                texts[str(row["id"])] = content if isinstance(content, str) else ""
        return texts

    single = getattr(store, "get_memory", None)
    if callable(single):
        texts = {}
        for memory_id in memory_ids:
            row = single(memory_id)
            if row:
                content = row.get("content")
                texts[str(memory_id)] = content if isinstance(content, str) else ""
        return texts

    raise RuntimeError(
        "CortexStore exposes neither get_memory_texts, a connection, nor get_memory"
    )


def count_missing(store: object, model_id: str) -> int:
    """How many recallable memories still lack an embedding for *model_id*."""
    return len(store.memory_ids_missing_embeddings(model_id, limit=_COUNT_LIMIT))


def commit(store: object) -> None:
    """Best-effort commit of the store connection after a batch.

    ``set_memory_embedding`` already commits each row; this is belt-and-braces
    so a batch boundary is always durable even if that behaviour changes.
    """
    conn = getattr(store, "_conn", None)
    if conn is not None:
        try:
            conn.commit()
        except Exception:  # noqa: BLE001 - a failed extra commit must not abort the run
            pass


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill local embeddings for Cortex's existing memories. Safe by "
            "default: small batches, a MemAvailable gate before every batch, "
            "per-batch commits, and resumable re-runs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", default=DEFAULT_DB, help="Cortex SQLite database.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Memories per batch (kept small to bound peak memory).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Embed at most this many memories this run (0 = all).",
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
        help="Report how many memories need embedding, then exit without writing.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    parser.add_argument(
        "--model-dir", default=None, help="Override the embedding model directory."
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="ONNX intra/inter-op threads."
    )
    return parser


def _log(message: str, quiet: bool) -> None:
    if not quiet:
        print(message, flush=True)


def _stop(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


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

    # Memory gate before we even open the store / load the model.
    mem_mb = read_mem_available_mb()
    if memory_is_low(mem_mb, args.min_available_mb):
        _stop(
            f"STOP: MemAvailable={mem_mb} MB is below --min-available-mb="
            f"{args.min_available_mb} MB; nothing written, re-run when the box has headroom."
        )
        return EXIT_LOW_MEMORY

    _ensure_cortex_importable()
    from cortex.embeddings import MODEL_ID, Embedder  # noqa: E402 (after bootstrap)
    from cortex.store import CortexStore  # noqa: E402 (after bootstrap)

    store = CortexStore(db_path)
    model_id = MODEL_ID

    if args.dry_run:
        remaining = count_missing(store, model_id)
        _log(
            f"dry run: {remaining} memories need embeddings for {model_id}; nothing written.",
            quiet,
        )
        return EXIT_OK

    embedder = Embedder(model_dir=args.model_dir, threads=args.threads)
    if not getattr(embedder, "available", True):
        _stop(
            f"error: embedding model unavailable under {embedder.model_dir} "
            f"({getattr(embedder, 'load_error', 'unknown reason')})"
        )
        return EXIT_ERROR

    initial_missing = count_missing(store, model_id)
    if initial_missing == 0:
        _log(f"nothing to do: every eligible memory already has a {model_id} embedding.", quiet)
        return EXIT_OK

    _log(
        f"backfill start: model={model_id} missing={initial_missing} "
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

        # Gate on memory before every batch — this is the whole point.
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
        fetched = store.memory_ids_missing_embeddings(model_id, limit=max(1, fetch_limit))
        ids = select_batch(fetched, skipped, args.batch_size)
        if args.limit > 0:
            ids = ids[: max(0, args.limit - written)]
        if not ids:
            break

        batch_no += 1
        texts = load_memory_texts(store, ids)
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
                    _log(f"  skip {memory_id}: empty content (no embedding written)", quiet)

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
                store.set_memory_embedding(memory_id, model_id, vector)
                written += 1
            commit(store)

        remaining = max(0, initial_missing - written - len(skipped))
        _log(
            f"batch {batch_no}: embedded={written}/{initial_missing} remaining={remaining} "
            f"MemAvailable={mem_mb} MB",
            quiet,
        )

        # Drop this batch's data so memory does not accumulate across batches.
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
