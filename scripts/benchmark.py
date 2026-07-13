from __future__ import annotations

import random
import statistics
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

try:
    from Brain.retrieval import MemoryRetriever
    from Brain.store import CortexStore
except ModuleNotFoundError:
    from cortex.retrieval import MemoryRetriever
    from cortex.store import CortexStore


def main() -> int:
    random.seed(7)
    with tempfile.TemporaryDirectory() as tmp:
        store = CortexStore(Path(tmp) / "benchmark.db")
        topics = ["hermes", "operations", "deployment", "edge server", "network", "calendar", "database", "backups"]
        for index in range(2000):
            topic = random.choice(topics)
            store.add_memory(
                f"Memory {index}: {topic} project observation with component {index % 71} and decision {index % 19}.",
                kind="semantic",
                importance=(index % 10) / 10,
            )
        retriever = MemoryRetriever(store)
        timings = []
        for index in range(200):
            query = f"What was the {random.choice(topics)} decision about component {index % 71}?"
            start = time.perf_counter()
            retriever.search(query, limit=6)
            timings.append((time.perf_counter() - start) * 1000)
        timings.sort()
        p95 = timings[int(len(timings) * 0.95) - 1]
        print(f"queries={len(timings)} memories=2000 median_ms={statistics.median(timings):.3f} p95_ms={p95:.3f}")
        store.close()
        return 0 if p95 < 50 else 1


if __name__ == "__main__":
    raise SystemExit(main())
