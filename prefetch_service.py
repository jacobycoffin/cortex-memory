"""Bounded in-process recall executor. No usage or outcome writes in this worker."""
from __future__ import annotations

import copy
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from .preload import RecallCache
from .semantics import feature_similarity


def _is_lock_error(exc: BaseException | None) -> bool:
    """True only for SQLite busy/locked contention, never for a generic bug."""
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    message = str(exc).lower()
    return "locked" in message or "busy" in message


@dataclass
class ComputedRecall:
    value: Any
    options: dict[str, Any]
    # The revision the worker actually computed under. A speculative hint may
    # have been queued while the foreground lock held the revision unknown; the
    # worker fills it in here so the entry can still be validated later.
    revision: tuple[int, int] | None = None


@dataclass
class _Job:
    session: str
    key: str
    query: str
    options: dict[str, Any]
    revision: tuple[int, int] | None
    done: threading.Event = field(default_factory=threading.Event)
    value: Any = None
    latency_ms: float = 0
    error: BaseException | None = None
    late: bool = False


class PrefetchService:
    """One worker, bounded latest-per-session queue, exact revision/TTL cache.

    A demand miss is synchronous from the caller's perspective, but executes on
    the isolated worker so the caller can abandon its wait. No replacement
    threads are spawned when a computation hangs. The worker owns its resources
    until it exits, including after a bounded shutdown join.
    """

    def __init__(self, factory: Callable, revision: Callable, *, ttl=45, capacity=32,
                 stale_ttl=600.0, stale_min_overlap=0.2):
        self._factory, self._revision = factory, revision
        self.ttl = max(0., min(300., float(ttl)))
        self.capacity = max(1, min(128, int(capacity)))
        # Stale serving: the session's last-good result may be reused after a
        # demand timeout, but only inside this TTL and only while its revision
        # (which already ignores pure usage counters) is unchanged.
        self.stale_ttl = max(0., min(3600., float(stale_ttl)))
        self.stale_min_overlap = max(0., min(1., float(stale_min_overlap)))
        self._cv = threading.Condition()
        self._pending: OrderedDict[str, _Job] = OrderedDict()
        self._cache: OrderedDict[str, tuple[float, tuple[int, int], Any, float]] = OrderedDict()
        self._last_good: OrderedDict[str, tuple[float, tuple[int, int], str, Any, float]] = OrderedDict()
        self._active: _Job | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
        self._last_ms = 0.0
        self._total_ms = 0.0
        self._last_late_ms = 0.0
        self._stats = dict(queued=0, replaced=0, deduplicated=0, dropped=0,
                           completed=0, failed=0, hits=0, misses=0, timeouts=0,
                           stale_served=0, late_completed=0, fallback=0, lock_busy=0)

    def _enqueue(self, session, query, options, *, demand=False):
        if not query.strip() or len(query) > 4096:
            return None, False
        for name, ceiling in (("limit", 20), ("token_budget", 4000), ("graph_depth", 2)):
            try:
                if name in options and not 0 <= int(options[name]) <= ceiling:
                    return None, False
            except (TypeError, ValueError, OverflowError):
                return None, False
        context = options.get("context")
        if context is not None and any(len(getattr(context, name)) > ceiling for name, ceiling in (
            ("scope", 64), ("system_state", 64), ("entities", 256),
            ("applicable_systems", 64), ("applicable_versions", 64),
        )):
            return None, False
        # This worker-only planning flag does not change effective search
        # options. Including it would duplicate a hint and its demand request.
        key_options = {name: value for name, value in options.items() if name != "_learn_budget"}
        key = RecallCache.key(query, key_options, (session,))
        revision = self._revision()
        # A speculative hint may be queued while the foreground lock is busy, so
        # the revision can be unknown; the worker fills it in from its own read
        # connection. A demand request always carries a revision.
        if revision is None and demand:
            return None, False
        with self._cv:
            if self._closed:
                return None, False
            entry = self._cache.get(key)
            if (entry and revision is not None and entry[1] == revision
                    and time.monotonic() - entry[0] < self.ttl):
                job = _Job(session, key, query, {}, revision, value=copy.deepcopy(entry[2]), latency_ms=entry[3])
                job.done.set()
                return job, True
            if revision is not None:
                self._cache.pop(key, None)
            for job in (self._pending.get(session), self._active):
                if job and job.key == key and job.revision == revision:
                    self._stats['deduplicated'] += 1
                    if demand and self._pending.get(session) is job:
                        self._pending.move_to_end(session, last=False)
                    return job, False
            job = _Job(session, key, query, copy.deepcopy(options), revision)
            old = self._pending.pop(session, None)
            if old:
                old.done.set()
                self._stats['replaced'] += 1
            elif len(self._pending) >= self.capacity:
                _, old = self._pending.popitem(last=False)
                old.done.set()
                self._stats['dropped'] += 1
            self._pending[session] = job
            if demand:
                self._pending.move_to_end(session, last=False)
            self._stats['queued'] += 1
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name='cortex-prefetch', daemon=True)
                self._thread.start()
            self._cv.notify_all()
            return job, False

    def submit(self, session: str, query: str, options: dict) -> bool:
        job, _ = self._enqueue(session, query, options)
        return job is not None

    def recall(self, session: str, query: str, options: dict, *, timeout: float):
        start = time.monotonic()
        job, hit = self._enqueue(session, query, options, demand=True)
        if job:
            job.done.wait(max(0., timeout - (time.monotonic() - start)))
        elapsed = (time.monotonic() - start) * 1000
        status = 'hit' if hit else 'miss'
        value = job.value if job and job.done.is_set() else None
        if job and not job.done.is_set():
            status = 'timeout'
            # The worker will still finish this job and cache it (late completion).
            with self._cv:
                job.late = True
        elif value is None:
            status = 'failed'
        # A memory write while waiting makes even a completed result unusable.
        if (value is not None and job is not None and job.revision is not None
                and job.revision != self._revision()):
            value, status = None, 'invalidated'
        # A fresh result was not ready in time: reuse this session's last-good
        # entry when it is still within the stale TTL and revision-valid.
        if value is None and status == 'timeout':
            stale = self._stale(session, query)
            if stale is not None:
                value, status = stale, 'stale'
                with self._cv:
                    self._stats['stale_served'] += 1
        with self._cv:
            self._stats['hits' if hit else 'misses'] += 1
            if status == 'timeout':
                self._stats['timeouts'] += 1
        return copy.deepcopy(value), dict(
            status=status, wait_ms=elapsed, recall_ms=job.latency_ms if job else 0.,
            locked=_is_lock_error(job.error) if job else False,
        )

    def recall_stale(self, session: str, query: str):
        """Serve the session's last-good entry alone, for a last-resort fallback."""
        stale = self._stale(session, query)
        if stale is None:
            return None, None
        with self._cv:
            self._stats['stale_served'] += 1
        return copy.deepcopy(stale), dict(status='stale', wait_ms=0., recall_ms=0., locked=False)

    def _stale(self, session: str, query: str):
        if self.stale_ttl <= 0:
            return None
        with self._cv:
            entry = self._last_good.get(session)
            if entry is None:
                return None
            created_at, revision, entry_query, value, _latency = entry
            if time.monotonic() - created_at > self.stale_ttl:
                return None
        # Reuse is refused the moment any material revision changes.
        if revision is None or revision != self._revision():
            return None
        if self.stale_min_overlap > 0 and feature_similarity(query, entry_query) < self.stale_min_overlap:
            return None
        return value

    def note_fallback(self, reason: str = 'lock_busy') -> None:
        """Count a turn served by the synchronous path because the service could not run."""
        with self._cv:
            self._stats['fallback'] += 1
            if reason == 'lock_busy':
                self._stats['lock_busy'] += 1

    def _run(self):
        resource = None
        try:
            compute, resource = self._factory()
            while True:
                with self._cv:
                    self._cv.wait_for(lambda: self._closed or self._pending)
                    if self._closed:
                        return
                    _, job = self._pending.popitem(last=False)
                    self._active = job
                start = time.monotonic()
                try:
                    value = compute(job.query, job.options)
                    with self._cv:
                        if isinstance(value, ComputedRecall):
                            job.key = RecallCache.key(job.query, value.options, (job.session,))
                            if job.revision is None and value.revision is not None:
                                job.revision = value.revision
                            value = value.value
                        job.latency_ms = (time.monotonic() - start) * 1000
                        job.value = value
                        if value is not None and not self._closed and job.revision is not None:
                            if self.ttl > 0:
                                self._cache[job.key] = (time.monotonic(), job.revision, copy.deepcopy(value), job.latency_ms)
                                self._cache.move_to_end(job.key)
                                while len(self._cache) > self.capacity:
                                    self._cache.popitem(last=False)
                            if self.stale_ttl > 0:
                                self._last_good[job.session] = (
                                    time.monotonic(), job.revision, job.query, copy.deepcopy(value), job.latency_ms)
                                self._last_good.move_to_end(job.session)
                                while len(self._last_good) > self.capacity:
                                    self._last_good.popitem(last=False)
                        if job.late:
                            self._stats['late_completed'] += 1
                            self._last_late_ms = job.latency_ms
                        self._stats['completed'] += 1
                except Exception as exc:
                    # Never expose local query text or exception payloads.
                    with self._cv:
                        self._stats['failed'] += 1
                        job.error = exc
                finally:
                    with self._cv:
                        self._last_ms = (time.monotonic() - start) * 1000
                        self._total_ms += self._last_ms
                        job.done.set()
                        self._active = None
                        self._cv.notify_all()
        except BaseException:
            with self._cv:
                self._stats['failed'] += 1
                if self._active:
                    self._active.done.set()
                    self._active = None
                for job in self._pending.values():
                    job.done.set()
                self._pending.clear()
                self._cv.notify_all()
        finally:
            if resource is not None:
                resource.close()

    def wait_idle(self, timeout=5.):
        with self._cv:
            return self._cv.wait_for(lambda: not self._active and not self._pending, timeout)

    def stats(self):
        with self._cv:
            return {**self._stats, 'pending': len(self._pending), 'active': self._active is not None,
                    'entries': len(self._cache), 'last_good': len(self._last_good),
                    'last_recall_ms': self._last_ms, 'total_recall_ms': self._total_ms,
                    'last_late_ms': self._last_late_ms}

    def close(self, timeout=.2):
        with self._cv:
            self._closed = True
            for job in self._pending.values():
                job.done.set()
            self._pending.clear()
            self._cache.clear()
            self._last_good.clear()
            self._cv.notify_all()
            thread = self._thread
        if thread:
            thread.join(timeout=max(0., timeout))
