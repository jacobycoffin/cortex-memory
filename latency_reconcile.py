"""Which latency number is the real one? (#224)

The store recorded `prepare_ms` p50 ≈ 1,209 ms while the latency work recorded p50 ≈ 141 ms on the same
box. Both numbers were honest and they described **different configurations**, which nobody wrote down:

```
                                        p50 prepare_ms   semantic stage   select stage
embedding model unavailable (benchmark)      120-141 ms        ~0 (skipped)        ~33 ms
embedding model available (production)       1,207 ms            263 ms            275 ms
```

The mechanism: the semantic stage calls `get_embedder()`, and when the model is not importable it returns
`{}` and retrieval continues lexically. That is the documented fallback ("retrieval falls back to exactly
the previous behaviour instead of failing"), and it is also a **silent 10× measurement difference**,
because the lexical path also keeps the candidate pool small, which makes `select` cheap too.

So this module exists to stop that happening again:

- **`availability()`** reports whether an embedder can actually run here, with the model id — the flag
  every latency number needs beside it;
- **`stage_profile()`** reads the recorded stage timings and ranks them, so "where does the second go"
  is answered by data rather than by whichever stage somebody remembers;
- **`describe()`** puts the two together into one sentence that cannot be quoted without its condition.

Rule: **a latency number is not a fact until it says what configuration it measured.** The SLO in the
health screen (#169) is against production, so the benchmark has to be run where the embedder works — or
it has to say, in the same breath, that it measured the lexical fallback.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from typing import Any

WINDOW = 500
# Stages that only exist when the semantic path is live: if they are absent from a profile, the number
# describes the lexical fallback rather than production.
SEMANTIC_STAGES = ("semantic",)


def _rows(store: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in store._conn.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001
        return []


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


@dataclass
class Availability:
    available: bool = False
    model: str = ""
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"available": self.available, "model": self.model, "reason": self.reason,
                "sentence": self.describe()}

    def describe(self) -> str:
        if self.available:
            return f"semantic path live (model {self.model})"
        return (f"semantic path unavailable{' (' + self.reason + ')' if self.reason else ''}: "
                f"retrieval is running on its lexical fallback")


def availability() -> Availability:
    """Can the semantic stage actually run in this interpreter?"""
    try:
        from .embeddings import get_embedder  # noqa: PLC0415 - optional dependency by design

        embedder = get_embedder()
        return Availability(available=bool(embedder.available),
                            model=str(getattr(embedder, "model_id", "") or ""),
                            reason="" if embedder.available else "model or runtime not importable")
    except Exception as error:  # noqa: BLE001
        return Availability(available=False, model="", reason=str(error)[:120])


@dataclass
class Profile:
    runs: int = 0
    prepare_p50: float = 0.0
    prepare_p95: float = 0.0
    stages: list[dict[str, Any]] = field(default_factory=list)
    semantic_seen: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"runs": self.runs, "prepare_p50": round(self.prepare_p50, 1),
                "prepare_p95": round(self.prepare_p95, 1), "stages": self.stages,
                "semantic_seen": self.semantic_seen, "note": self.note}


def stage_profile(store: Any, *, window: int = WINDOW) -> Profile:
    """Where the time goes, from the stage timings the store already records."""
    rows = _rows(
        store,
        "SELECT prepare_ms, stage_ms_json FROM recall_runs WHERE prepare_ms IS NOT NULL "
        "ORDER BY created_at DESC, recall_id DESC LIMIT ?",
        (int(window),),
    )
    profile = Profile(runs=len(rows))
    if not rows:
        profile.note = "no timed runs recorded"
        return profile
    profile.prepare_p50 = _percentile([float(r["prepare_ms"]) for r in rows], 0.5)
    profile.prepare_p95 = _percentile([float(r["prepare_ms"]) for r in rows], 0.95)
    per_stage: dict[str, list[float]] = {}
    for row in rows:
        try:
            stages = json.loads(row["stage_ms_json"] or "{}")
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(stages, dict):
            continue
        for key, value in stages.items():
            if isinstance(value, (int, float)) and not str(key).endswith("_count"):
                per_stage.setdefault(str(key), []).append(float(value))
    ranked = sorted(per_stage.items(), key=lambda item: -statistics.median(item[1]))
    profile.stages = [
        {"stage": name, "p50": round(_percentile(values, 0.5), 1),
         "p95": round(_percentile(values, 0.95), 1), "runs": len(values)}
        for name, values in ranked
    ]
    profile.semantic_seen = any(str(item["stage"]) in SEMANTIC_STAGES for item in profile.stages)
    if not profile.semantic_seen:
        profile.note = ("no semantic stage was recorded: these runs describe the lexical fallback, "
                        "not a store with embeddings")
    return profile


def describe(store: Any, *, window: int = WINDOW, available: Availability | None = None
             ) -> dict[str, Any]:
    """One answer that cannot be quoted without its condition."""
    flag = available or availability()
    profile = stage_profile(store, window=window)
    heaviest = profile.stages[0] if profile.stages else {}
    sentence = (f"p50 {profile.prepare_p50:.0f} ms / p95 {profile.prepare_p95:.0f} ms over "
                f"{profile.runs} run(s); {flag.describe()}")
    if heaviest:
        sentence += (f"; heaviest stage {heaviest['stage']} {heaviest['p50']:.0f} ms")
    if profile.note:
        sentence += f" — {profile.note}"
    return {
        "sentence": sentence,
        "availability": flag.as_dict(),
        "profile": profile.as_dict(),
        "rule": "a latency number is not a fact until it says which configuration it measured",
    }


def compare(benchmark_ms: float, production_ms: float, *, stage: str = "semantic") -> dict[str, Any]:
    """Name the difference between two numbers, instead of averaging them into one."""
    if not benchmark_ms or not production_ms:
        return {"ratio": None, "sentence": "both numbers are needed"}
    ratio = round(production_ms / benchmark_ms, 1)
    return {
        "benchmark_ms": round(float(benchmark_ms), 1),
        "production_ms": round(float(production_ms), 1),
        "ratio": ratio,
        "sentence": (f"production is {ratio}× the benchmark; the likeliest cause is the {stage} stage, "
                     f"which returns nothing when the embedding model is not importable"),
        "check": "run the benchmark where get_embedder().available is True",
    }
