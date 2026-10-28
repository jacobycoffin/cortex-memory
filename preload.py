"""Bounded, local-only recall warming with revision-checked in-memory reuse.

A warm result is fallible evidence, not a selection or an outcome. This module
never writes memories, accesses, usage batches, episodes, or training labels.
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class _Entry:
    created_at: float
    revision: tuple[int, int]
    value: Any
    speculative: bool


class RecallCache:
    """Bounded cache shared by foreground recall and its optional worker."""

    def __init__(self, *, ttl_seconds: float = 45, max_entries: int = 32):
        self.ttl_seconds = max(0.0, min(300.0, float(ttl_seconds)))
        self.max_entries = max(1, min(128, int(max_entries)))
        self._lock = threading.RLock()
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._stats = {"hits": 0, "misses": 0, "preload_hits": 0, "discarded": 0}

    @staticmethod
    def key(query: str, options: dict[str, Any], settings: tuple[Any, ...]) -> str:
        values = dict(options)
        context = values.get("context")
        if context is not None:
            values["context"] = {**context.as_record(), "goal": context.goal}
        payload = json.dumps([query, values, settings], sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def search(
        self, key: str, revision: Callable[[], tuple[int, int]],
        compute: Callable[[], Any], *, speculative: bool = False,
    ) -> Any:
        before = revision()
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if (entry and self.ttl_seconds > 0 and entry.revision == before
                    and now - entry.created_at < self.ttl_seconds):
                self._entries.move_to_end(key)
                if not speculative:
                    self._stats["hits"] += 1
                    if entry.speculative:
                        self._stats["preload_hits"] += 1
                return copy.deepcopy(entry.value)
            if not speculative:
                self._stats["misses"] += 1
            self._entries.pop(key, None)
        value = compute()
        after = revision()
        with self._lock:
            if before != after:
                self._stats["discarded"] += 1
            elif self.ttl_seconds > 0:
                self._entries[key] = _Entry(time.monotonic(), after, copy.deepcopy(value), speculative)
                self._entries.move_to_end(key)
                while len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
        return value

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {**self._stats, "entries": len(self._entries)}

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class BackgroundPreloader:
    """One lazy worker, one pending job; a newer hint replaces a queued hint.

    Foreground callers do not wait for a warming job to finish. Close drains the active
    read before the owning store closes, and cancels the pending job.
    """

    def __init__(self, warm: Callable[[str, dict[str, Any]], Any]):
        self._warm = warm
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._pending: tuple[str, dict[str, Any]] | None = None
        self._active = False
        self._closed = False
        self._stats = {"queued": 0, "completed": 0, "replaced": 0, "failed": 0}
        self._last_ms = 0.0
        self._total_ms = 0.0

    def submit(self, query: str, options: dict[str, Any]) -> bool:
        if not query.strip() or len(query) > 4096:
            return False
        for name, ceiling in (("limit", 20), ("token_budget", 4000), ("graph_depth", 2)):
            if name in options:
                try:
                    value = int(options[name])
                except (TypeError, ValueError, OverflowError):
                    return False
                if value < 0 or value > ceiling:
                    return False
        context = options.get("context")
        if context is not None and any(len(getattr(context, name)) > ceiling for name, ceiling in (
            ("scope", 64), ("system_state", 64), ("entities", 256),
            ("applicable_systems", 64), ("applicable_versions", 64),
        )):
            return False
        with self._condition:
            if self._closed:
                return False
            if self._pending is not None:
                self._stats["replaced"] += 1
            self._pending = (query, copy.deepcopy(options))
            self._stats["queued"] += 1
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="cortex-preload", daemon=True)
                self._thread.start()
            self._condition.notify_all()
        return True

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                query, options = self._pending  # type: ignore[misc]
                self._pending = None
                self._active = True
            start = time.perf_counter()
            failed = False
            try:
                self._warm(query, options)
            except Exception:
                # Speculation cannot break foreground recall. Do not log query
                # text, identifiers, or exception strings containing local data.
                failed = True
            finally:
                with self._condition:
                    self._last_ms = (time.perf_counter() - start) * 1000
                    self._total_ms += self._last_ms
                    self._stats["failed" if failed else "completed"] += 1
                    self._active = False
                    self._condition.notify_all()

    def wait_idle(self, timeout: float = 5.0) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: not self._active and self._pending is None, timeout=max(0.0, timeout)
            )

    def stats(self) -> dict[str, Any]:
        with self._condition:
            return {**self._stats, "active": self._active, "pending": self._pending is not None,
                    "last_warm_ms": round(self._last_ms, 3), "total_warm_ms": round(self._total_ms, 3)}

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join()
