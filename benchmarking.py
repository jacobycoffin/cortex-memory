"""Bounded, versioned benchmark used by the authenticated Brain dashboard."""

from __future__ import annotations

from typing import Any, Callable

from .benchmarks.core import run_benchmark


DASHBOARD_BENCHMARK_SUITE = "cortex-retrieval-standard"
DASHBOARD_BENCHMARK_VERSION = "1"
DASHBOARD_BENCHMARK_SIZES = (100, 500, 2000)
DASHBOARD_BENCHMARK_QUERIES = 80
DASHBOARD_BENCHMARK_SEED = 7
DASHBOARD_BENCHMARK_TOP_K = 6
DASHBOARD_BENCHMARK_TOKEN_BUDGET = 700
DASHBOARD_BENCHMARK_LATENCY_TARGET_MS = 50.0

ProgressCallback = Callable[[dict[str, object]], None]


def run_dashboard_benchmark(progress_callback: ProgressCallback | None = None) -> dict[str, Any]:
    """Run the fixed local suite and return its raw report plus dashboard score.

    The suite uses a temporary synthetic database. It never reads production
    memories and never calls a model or provider endpoint.
    """

    _emit(
        progress_callback,
        phase="preparing",
        progress=4,
        message="Preparing an isolated synthetic memory database.",
    )

    def benchmark_progress(update: dict[str, Any]) -> None:
        total = max(1, int(update.get("total") or len(DASHBOARD_BENCHMARK_SIZES)))
        index = max(1, int(update.get("index") or 1))
        size = int(update.get("corpus_memories") or 0)
        if update.get("state") == "starting_scale":
            progress = 5 + round((index - 1) / total * 84)
            message = f"Building and testing the {size:,}-memory corpus."
        else:
            progress = 5 + round(index / total * 84)
            message = f"Finished {size:,} memories; moving to the next scale."
        _emit(
            progress_callback,
            phase="retrieval",
            progress=progress,
            message=message,
            corpus_memories=size,
            scale_index=index,
            scale_total=total,
        )

    report = run_benchmark(
        DASHBOARD_BENCHMARK_SIZES,
        query_count=DASHBOARD_BENCHMARK_QUERIES,
        seed=DASHBOARD_BENCHMARK_SEED,
        top_k=DASHBOARD_BENCHMARK_TOP_K,
        token_budget=DASHBOARD_BENCHMARK_TOKEN_BUDGET,
        progress_callback=benchmark_progress,
    )
    _emit(
        progress_callback,
        phase="scoring",
        progress=94,
        message="Calculating the versioned score and improvement guidance.",
    )
    summary = summarize_dashboard_benchmark(report)
    report["dashboard_summary"] = summary
    report["dashboard_suite"] = {
        "name": DASHBOARD_BENCHMARK_SUITE,
        "version": DASHBOARD_BENCHMARK_VERSION,
        "sizes": list(DASHBOARD_BENCHMARK_SIZES),
        "queries_per_scale": DASHBOARD_BENCHMARK_QUERIES,
        "seed": DASHBOARD_BENCHMARK_SEED,
        "top_k": DASHBOARD_BENCHMARK_TOP_K,
        "token_budget_approx": DASHBOARD_BENCHMARK_TOKEN_BUDGET,
    }
    _emit(
        progress_callback,
        phase="completed",
        progress=100,
        message=f"Benchmark complete. Cortex scored {summary['score']:.1f} out of 100.",
    )
    return report


def summarize_dashboard_benchmark(report: dict[str, Any]) -> dict[str, Any]:
    """Create a transparent score from the largest fixed benchmark scale."""

    results = list(report.get("results") or [])
    if not results:
        raise ValueError("benchmark report has no results")
    target = max(results, key=lambda row: int(row.get("corpus_memories") or 0))
    cortex = target.get("cortex") or {}
    default = target.get("default_built_in") or {}
    latency = cortex.get("query_latency") or {}
    context = cortex.get("context_approx_tokens") or {}
    recall = _clamp(float(cortex.get("recall_at_k") or 0.0))
    mrr = _clamp(float(cortex.get("mrr") or 0.0))
    p50_ms = max(0.0, float(latency.get("p50_ms") or 0.0))
    p95_ms = max(0.0, float(latency.get("p95_ms") or 0.0))
    context_p50 = max(0.0, float(context.get("p50") or 0.0))
    quality_score = (0.65 * recall + 0.35 * mrr) * 100.0
    speed_score = 100.0 * min(
        1.0,
        DASHBOARD_BENCHMARK_LATENCY_TARGET_MS / max(0.001, p95_ms),
    )
    score = 0.85 * quality_score + 0.15 * speed_score
    summary = {
        "score": round(score, 1),
        "quality_score": round(quality_score, 1),
        "speed_score": round(speed_score, 1),
        "recall_at_k": round(recall, 6),
        "mrr": round(mrr, 6),
        "precision_at_k": round(float(cortex.get("precision_at_k") or 0.0), 6),
        "p50_ms": round(p50_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "context_tokens_p50": round(context_p50, 1),
        "default_coverage": round(float(default.get("answer_coverage_rate") or 0.0), 6),
        "corpus_memories": int(target.get("corpus_memories") or 0),
        "queries": int(target.get("queries") or 0),
        "targets": {
            "recall_at_k": 0.98,
            "mrr": 0.95,
            "p95_ms": DASHBOARD_BENCHMARK_LATENCY_TARGET_MS,
            "context_tokens_p50": 250,
        },
        "definitions": {
            "score": (
                "85% retrieval quality and 15% local p95 retrieval speed. Retrieval quality is "
                "65% recall@6 and 35% MRR. Speed receives full credit at 50 ms p95 or faster."
            ),
            "recall_at_k": "Share of labeled questions whose correct memory appeared in the first six results.",
            "mrr": "Ranking quality. 1.0 means the correct memory was always the first result.",
            "p95_ms": "Local retrieval overhead below which 95% of measured queries completed; model time is excluded.",
            "context_tokens_p50": "Median approximate memory-context tokens returned; characters divided by four.",
        },
        "claim_boundary": (
            "This fixed synthetic suite measures retrieval quality, capacity, context size, and local retrieval "
            "overhead on this host. It does not measure model inference speed or real-user answer quality."
        ),
    }
    summary["recommendations"] = benchmark_recommendations(summary)
    return summary


def benchmark_recommendations(summary: dict[str, Any]) -> list[dict[str, str]]:
    """Translate measured components into bounded, actionable next steps."""

    recall = float(summary.get("recall_at_k") or 0.0)
    mrr = float(summary.get("mrr") or 0.0)
    p95 = float(summary.get("p95_ms") or 0.0)
    context = float(summary.get("context_tokens_p50") or 0.0)
    recommendations: list[dict[str, str]] = []
    if recall < 0.98:
        recommendations.append(
            {
                "area": "Retrieval coverage",
                "state": "improve",
                "title": "Study the missed query families first",
                "explanation": f"Recall@6 is {recall:.1%}; the target is at least 98% on this fixed suite.",
                "action": "Inspect exact versus paraphrased misses, then improve candidate generation or reranking without increasing the context budget first.",
            }
        )
    elif mrr < 0.95:
        recommendations.append(
            {
                "area": "Ranking",
                "state": "watch",
                "title": "Move correct memories closer to rank one",
                "explanation": f"Coverage is healthy, but MRR is {mrr:.3f} against the 0.95 target.",
                "action": "Tune identifier rarity, phrase overlap, and task-specific reranking while keeping recall@6 stable.",
            }
        )
    else:
        recommendations.append(
            {
                "area": "Retrieval quality",
                "state": "healthy",
                "title": "Synthetic retrieval is at the target",
                "explanation": f"Recall@6 is {recall:.1%} and MRR is {mrr:.3f} on the 2,000-memory scale.",
                "action": "Protect these gates in tests; the next meaningful quality gain should come from a frozen private real-history evaluation.",
            }
        )
    if p95 > DASHBOARD_BENCHMARK_LATENCY_TARGET_MS:
        recommendations.append(
            {
                "area": "Retrieval speed",
                "state": "improve",
                "title": "Reduce tail retrieval overhead",
                "explanation": f"p95 is {p95:.1f} ms; the fixed local target is {DASHBOARD_BENCHMARK_LATENCY_TARGET_MS:.0f} ms.",
                "action": "Profile FTS ranking, feature expansion, graph activation, SQLite WAL growth, and host load. Re-run the identical suite after each change.",
            }
        )
    else:
        recommendations.append(
            {
                "area": "Retrieval speed",
                "state": "healthy",
                "title": "Local retrieval latency is inside the gate",
                "explanation": f"p95 is {p95:.1f} ms on this host, within the 50 ms target.",
                "action": "Do not trade away recall for smaller latency changes. Use the paired live-model benchmark for any whole-agent speed claim.",
            }
        )
    if context > 250:
        recommendations.append(
            {
                "area": "Context efficiency",
                "state": "watch",
                "title": "Return less evidence without losing the answer",
                "explanation": f"Median returned context is about {context:.0f} tokens; the dashboard guardrail is 250.",
                "action": "Try more compact evidence formatting or a smaller top-k, then confirm recall@6 and MRR do not fall.",
            }
        )
    else:
        recommendations.append(
            {
                "area": "Context efficiency",
                "state": "healthy",
                "title": "The working set stays compact",
                "explanation": f"Median returned context is about {context:.0f} approximate tokens.",
                "action": "Keep context size as a guardrail rather than optimizing it alone; an empty prompt is cheap but not useful.",
            }
        )
    return recommendations


def _emit(callback: ProgressCallback | None, **payload: object) -> None:
    if callback:
        callback(dict(payload))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
