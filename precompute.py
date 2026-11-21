"""Precompute the per-turn costs that do not depend on the query (#23).

Measured on the live store, five representative queries, per stage:

    feature             32.6ms
    select              25.1ms
    prospective_merge   16.9ms   <-- a DB query per turn for "are any commitments due?"
    fts                 13.0ms
    graph               12.6ms
    scoring             10.4ms

`prospective_merge` is 17ms of every recall to answer a question whose answer changes when
somebody writes a memory — not when somebody asks. The same is true of the scoring profile
weights and the entity name list. These are *precomputable*: compute once, reuse until the
store changes.

The preloader already warms expensive query-specific work. This warms the query-independent
part, which is why it helps the foreground turn even when the guessed query was wrong.

Staleness is bounded by the store's own revision counter, not by a TTL, so a memory write
invalidates immediately — and a TTL is available for the cases where the revision is
unavailable.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

DEFAULT_TTL_SECONDS = 30.0


@dataclass
class _Slot:
    value: Any = None
    revision: Any = None
    at: float = 0.0
    valid: bool = False


@dataclass
class Precompute:
    """Revision-keyed memo for query-independent recall inputs."""

    ttl_seconds: float = DEFAULT_TTL_SECONDS
    _slots: dict[str, _Slot] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    hits: int = 0
    misses: int = 0

    def get(self, key: str, compute: Callable[[], Any], *, revision: Any = None) -> Any:
        """Return a cached value for `key`, recomputing when the revision or TTL moved."""
        now = time.monotonic()
        with self._lock:
            slot = self._slots.get(key)
            if (
                slot is not None
                and slot.valid
                and slot.revision == revision
                and (now - slot.at) <= self.ttl_seconds
            ):
                self.hits += 1
                return slot.value
        # Compute outside the lock: a slow query must not block other keys.
        value = compute()
        with self._lock:
            self._slots[key] = _Slot(value=value, revision=revision, at=now, valid=True)
            self.misses += 1
        return value

    def prime(self, key: str, value: Any, *, revision: Any = None) -> None:
        """Warm a key directly — the warm path uses this when it already has the answer."""
        with self._lock:
            self._slots[key] = _Slot(value=value, revision=revision, at=time.monotonic(), valid=True)

    def clear(self) -> None:
        with self._lock:
            self._slots.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "keys": len(self._slots)}


def revision_of(store: Any) -> Any:
    """The store's change counter, or None when it cannot be read."""
    try:
        return store.retrieval_revision()
    except Exception:  # noqa: BLE001 - caching must never break recall
        return None
