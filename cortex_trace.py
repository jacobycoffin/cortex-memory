"""Opt-in phase tracing for the four expensive Cortex call paths.

One turn can run recall, write, pruning and consolidation in the same process,
so "the turn felt slow" names no phase. This module appends one JSON object per
phase to a JSONL file:

    {"ts": "...Z", "phase": "recall", "duration_ms": 12.345,
     "counts": {"candidates": 40, "returned": 6, "written": 0},
     "run_id": "...", "session_id": null, "pid": 1234}

Tracing is **off unless** ``CORTEX_TRACE=1`` (also accepts ``true``/``yes``/``on``).
The flag is read once per process; call :func:`reset` to re-read it. When it is
off, call sites pay one module-global boolean test and nothing else: no clock
read, no allocation, no file handle. ``CORTEX_TRACE_PATH`` overrides the output
file, which otherwise sits beside the database as
``$CORTEX_HOME/cortex-trace.jsonl`` (``~/.cortex/cortex-trace.jsonl`` by default).

Trace records deliberately carry no memory content and no query text: phase
name, durations, counts and identifiers only, so the file can be read by
``scripts/cortex_trace_report.py`` (p50/p95 per phase, plus every run over its
budget) without leaking what the agent remembered.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

TRACE_ENV = "CORTEX_TRACE"
TRACE_PATH_ENV = "CORTEX_TRACE_PATH"
_TRUTHY = {"1", "true", "yes", "on"}

_enabled: bool | None = None
_path: Path | None = None
_write_failed = False
_lock = threading.Lock()


def trace_enabled() -> bool:
    """Return whether tracing is on, reading the environment only once."""

    global _enabled
    if _enabled is None:
        _enabled = (os.environ.get(TRACE_ENV, "") or "").strip().lower() in _TRUTHY
    return _enabled


def trace_path() -> Path:
    """Resolve the JSONL destination (``CORTEX_TRACE_PATH`` wins)."""

    global _path
    if _path is None:
        explicit = (os.environ.get(TRACE_PATH_ENV, "") or "").strip()
        if explicit:
            _path = Path(explicit).expanduser()
        else:
            home = (os.environ.get("CORTEX_HOME", "") or "").strip()
            base = Path(home).expanduser() if home else Path.home() / ".cortex"
            _path = base / "cortex-trace.jsonl"
    return _path


def reset() -> None:
    """Forget the cached flag, path and write-failure latch (tests, reloads)."""

    global _enabled, _path, _write_failed
    _enabled = None
    _path = None
    _write_failed = False


def new_run_id() -> str:
    """Short identifier for one traced phase run."""

    return uuid.uuid4().hex[:16]


def record_phase(
    phase: str,
    duration_ms: float,
    *,
    counts: Mapping[str, Any] | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any] | None:
    """Append one JSONL line. No-op (and no allocation) unless tracing is on.

    Returns the record written, so callers and tests can assert on it.
    """

    if not trace_enabled():
        return None
    global _write_failed
    merged = {"candidates": 0, "returned": 0, "written": 0}
    # Copy only the three counters we publish, and only when given: keeps the
    # record shape stable for the report script without echoing caller data.
    if counts:
        for key in ("candidates", "returned", "written"):
            value = counts.get(key)
            if value is not None:
                merged[key] = int(value)
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "phase": str(phase),
        "duration_ms": round(max(0.0, float(duration_ms)), 3),
        "counts": merged,
        "run_id": str(run_id or new_run_id()),
        "session_id": str(session_id) if session_id else None,
        "pid": os.getpid(),
    }
    line = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
    if _write_failed:
        return record
    try:
        target = trace_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        with _lock, target.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        # Observability must never break the memory path it observes.
        _write_failed = True
    return record


def phase_start() -> float:
    """``perf_counter`` stamp for a traced phase; only call when enabled."""

    return time.perf_counter()


def elapsed_ms(started: float) -> float:
    """Milliseconds since :func:`phase_start`."""

    return (time.perf_counter() - started) * 1000.0
