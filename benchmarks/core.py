"""Deterministic benchmark fixtures and metrics for Cortex.

The benchmark intentionally models Hermes built-in memory as a bounded,
frozen prompt snapshot. It does not pretend that built-in memory performs a
search on every turn: snapshot access is effectively constant-time. Cortex is
measured as a query-time retrieval layer, and end-to-end model timing belongs
in the separate live benchmark.
"""

from __future__ import annotations

import json
import math
import platform
import random
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..retrieval import MemoryRetriever
from ..store import CortexStore


DEFAULT_MEMORY_CHAR_LIMIT = 2200
DEFAULT_USER_CHAR_LIMIT = 1375
ENTRY_DELIMITER = "\n§\n"


@dataclass(frozen=True)
class BenchmarkFact:
    """One synthetic memory with a labeled retrieval question."""

    label: str
    project: str
    attribute: str
    value: str
    content: str
    query: str


@dataclass(frozen=True)
class LatencySummary:
    minimum_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    maximum_ms: float


_ATTRIBUTES: tuple[tuple[str, str], ...] = (
    ("notification transport", "transport protocol"),
    ("backup region", "backup location"),
    ("deployment command", "release command"),
    ("service port", "network port"),
    ("database engine", "data store"),
    ("incident owner", "on-call owner"),
    ("build runner", "CI runner"),
    ("release channel", "deployment channel"),
)

_PROJECT_WORDS = (
    "amber",
    "birch",
    "comet",
    "delta",
    "ember",
    "fjord",
    "grove",
    "harbor",
    "indigo",
    "juniper",
    "kestrel",
    "lumen",
    "mesa",
    "nova",
    "onyx",
    "prairie",
)

_VALUE_WORDS = (
    "alder",
    "beacon",
    "cinder",
    "drift",
    "elm",
    "falcon",
    "garnet",
    "hearth",
    "iris",
    "jolt",
    "kite",
    "lotus",
    "moss",
    "north",
    "orbit",
    "pine",
)


def generate_facts(size: int, *, seed: int = 7) -> list[BenchmarkFact]:
    """Create a stable synthetic corpus with exact answer labels.

    Every fact has a unique project and value token. Query wording rotates
    between exact and mild paraphrase forms while retaining an unambiguous
    project cue. This tests ranking under distractors without requiring an
    embedding model Cortex does not currently use.
    """

    if size < 1:
        raise ValueError("size must be positive")
    rng = random.Random(seed)
    facts: list[BenchmarkFact] = []
    order = list(range(size))
    rng.shuffle(order)
    for position, source_index in enumerate(order):
        attribute, paraphrase = _ATTRIBUTES[source_index % len(_ATTRIBUTES)]
        project_index = source_index // len(_ATTRIBUTES)
        project_word = _PROJECT_WORDS[project_index % len(_PROJECT_WORDS)]
        # Eight facts share a project. This creates realistic interference:
        # matching the project alone is not enough; the attribute must also be
        # resolved, and paraphrase cases are deliberately harder for FTS-only
        # retrieval.
        project = f"project-{project_word}-{project_index:05d}"
        value_word = _VALUE_WORDS[(source_index * 5 + 3) % len(_VALUE_WORDS)]
        value = f"{value_word}-{source_index:05d}"
        content = (
            f"{project}: the verified {attribute} is {value}. "
            "This is the current approved operating decision."
        )
        query_style = position % 3
        if query_style == 0:
            query = f"What is the {attribute} for {project}?"
        elif query_style == 1:
            query = f"Which {paraphrase} was approved for {project}?"
        else:
            query = f"Recall the current {attribute} decision for {project}."
        facts.append(
            BenchmarkFact(
                label=f"fact-{source_index:05d}",
                project=project,
                attribute=attribute,
                value=value,
                content=content,
                query=query,
            )
        )
    return facts


def select_queries(facts: Sequence[BenchmarkFact], count: int, *, seed: int = 7) -> list[BenchmarkFact]:
    if count < 1:
        raise ValueError("query count must be positive")
    if not facts:
        return []
    rng = random.Random(seed + len(facts) * 31)
    if count <= len(facts):
        return rng.sample(list(facts), count)
    return [facts[rng.randrange(len(facts))] for _ in range(count)]


def build_default_snapshot(
    facts: Sequence[BenchmarkFact],
    *,
    char_limit: int = DEFAULT_MEMORY_CHAR_LIMIT,
) -> tuple[str, list[BenchmarkFact]]:
    """Pack facts in observed order until Hermes's built-in limit is reached.

    Hermes's memory tool rejects a write that exceeds its configured character
    limit. The synthetic facts all have equal importance, so observed order is
    the only non-oracular retention rule available to both conditions.
    """

    if char_limit < 1:
        raise ValueError("char_limit must be positive")
    stored: list[BenchmarkFact] = []
    for fact in facts:
        candidate = ENTRY_DELIMITER.join([*(row.content for row in stored), fact.content])
        if len(candidate) > char_limit:
            continue
        stored.append(fact)
    content = ENTRY_DELIMITER.join(row.content for row in stored)
    if not content:
        return "", []
    usage = min(100, int((len(content) / char_limit) * 100))
    separator = "═" * 46
    snapshot = (
        f"{separator}\n"
        f"MEMORY (your personal notes) [{usage}% — {len(content):,}/{char_limit:,} chars]\n"
        f"{separator}\n{content}"
    )
    return snapshot, stored


def approximate_tokens(text: str) -> int:
    """Return a clearly labeled approximation, not provider billing tokens."""

    return math.ceil(len(text) / 4) if text else 0


def summarize_latencies(values_ms: Sequence[float]) -> LatencySummary:
    if not values_ms:
        return LatencySummary(0.0, 0.0, 0.0, 0.0, 0.0)
    ordered = sorted(float(value) for value in values_ms)
    return LatencySummary(
        minimum_ms=round(ordered[0], 6),
        p50_ms=round(_percentile(ordered, 0.50), 6),
        p95_ms=round(_percentile(ordered, 0.95), 6),
        p99_ms=round(_percentile(ordered, 0.99), 6),
        maximum_ms=round(ordered[-1], 6),
    )


def benchmark_scale(
    size: int,
    *,
    query_count: int = 200,
    seed: int = 7,
    top_k: int = 6,
    token_budget: int = 700,
    default_char_limit: int = DEFAULT_MEMORY_CHAR_LIMIT,
) -> dict[str, Any]:
    """Compare bounded built-in context with Cortex retrieval at one scale."""

    facts = generate_facts(size, seed=seed)
    queries = select_queries(facts, query_count, seed=seed)
    default_snapshot, default_facts = build_default_snapshot(facts, char_limit=default_char_limit)
    default_labels = {fact.label for fact in default_facts}
    default_latencies: list[float] = []
    default_context_tokens: list[int] = []
    default_hits = 0
    for fact in queries:
        start = time.perf_counter_ns()
        context = default_snapshot
        default_latencies.append((time.perf_counter_ns() - start) / 1_000_000)
        default_context_tokens.append(approximate_tokens(context))
        default_hits += int(fact.label in default_labels and fact.value in context)

    with tempfile.TemporaryDirectory(prefix="cortex-benchmark-") as tmp:
        store = CortexStore(Path(tmp) / "cortex.db")
        label_to_memory: dict[str, str] = {}
        index_start = time.perf_counter()
        for fact in facts:
            memory_id, _ = store.add_memory(
                fact.content,
                kind="decision",
                source_type="benchmark",
                source_category="TOOL_VERIFIED",
                source_ref=fact.label,
                confidence=0.95,
                currentness_confidence=0.95,
                importance=0.7,
                volatility=0.2,
                trust=0.95,
                subject=fact.project,
                predicate=fact.attribute,
                object_value=fact.value,
                extraction_method="synthetic_benchmark",
            )
            label_to_memory[fact.label] = memory_id
        index_ms = (time.perf_counter() - index_start) * 1000
        retriever = MemoryRetriever(store)

        # One untimed warm-up prevents first-query setup from dominating p95.
        retriever.search(queries[0].query, limit=top_k, token_budget=token_budget)

        cortex_latencies: list[float] = []
        cortex_context_tokens: list[int] = []
        cortex_ranks: list[int | None] = []
        cortex_precision: list[float] = []
        returned_counts: list[int] = []
        for fact in queries:
            start = time.perf_counter_ns()
            results = retriever.search(fact.query, limit=top_k, token_budget=token_budget)
            cortex_latencies.append((time.perf_counter_ns() - start) / 1_000_000)
            target_id = label_to_memory[fact.label]
            result_ids = [result.memory["id"] for result in results]
            rank = result_ids.index(target_id) + 1 if target_id in result_ids else None
            cortex_ranks.append(rank)
            returned_counts.append(len(results))
            cortex_precision.append((1.0 / len(results)) if rank and results else 0.0)
            cortex_context = "\n".join(result.memory["content"] for result in results)
            cortex_context_tokens.append(approximate_tokens(cortex_context))
        store.close()

    reciprocal_ranks = [(1.0 / rank) if rank else 0.0 for rank in cortex_ranks]
    return {
        "corpus_memories": size,
        "queries": len(queries),
        "configuration": {
            "seed": seed,
            "top_k": top_k,
            "token_budget_approx": token_budget,
            "default_memory_char_limit": default_char_limit,
        },
        "default_built_in": {
            "stored_memories": len(default_facts),
            "rejected_for_capacity": size - len(default_facts),
            "answer_coverage_rate": _ratio(default_hits, len(queries)),
            "context_approx_tokens": _distribution(default_context_tokens),
            "per_turn_snapshot_access_latency": asdict(summarize_latencies(default_latencies)),
            "notes": (
                "Frozen snapshot access only. Hermes performs no query-time search for built-in memory, "
                "so this is a capacity/context baseline rather than a retrieval-ranking baseline."
            ),
        },
        "cortex": {
            "indexed_memories": size,
            "one_time_index_ms": round(index_ms, 3),
            "recall_at_k": round(statistics.fmean(1.0 if rank else 0.0 for rank in cortex_ranks), 6),
            "mrr": round(statistics.fmean(reciprocal_ranks), 6),
            "precision_at_k": round(statistics.fmean(cortex_precision), 6),
            "returned_memories": _distribution(returned_counts),
            "context_approx_tokens": _distribution(cortex_context_tokens),
            "query_latency": asdict(summarize_latencies(cortex_latencies)),
        },
    }


def run_benchmark(
    sizes: Iterable[int],
    *,
    query_count: int = 200,
    seed: int = 7,
    top_k: int = 6,
    token_budget: int = 700,
    default_char_limit: int = DEFAULT_MEMORY_CHAR_LIMIT,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    rows = [
        benchmark_scale(
            size,
            query_count=min(query_count, size),
            seed=seed,
            top_k=top_k,
            token_budget=token_budget,
            default_char_limit=default_char_limit,
        )
        for size in sizes
    ]
    return {
        "schema_version": 1,
        "benchmark": "cortex-vs-hermes-built-in-synthetic",
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "not reported",
        },
        "scope": {
            "measures": [
                "built-in bounded-context coverage",
                "Cortex recall@k, precision@k, and MRR",
                "query-time retrieval latency",
                "approximate context size",
            ],
            "does_not_measure": [
                "LLM time to first token",
                "LLM total response latency",
                "provider-billed token counts",
                "real-user memory quality",
            ],
        },
        "results": rows,
    }


def write_report(report: dict[str, Any], output_path: Path) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return output_path, markdown_path


def render_markdown(report: dict[str, Any]) -> str:
    environment = report.get("environment") or {}
    lines = [
        "# Cortex benchmark result",
        "",
        f"Run: `{report['started_at']}` · Python {environment.get('python', 'unknown')} · "
        f"{environment.get('platform', 'platform not reported')}",
        "",
        "## What this result supports",
        "",
        "This synthetic run measures memory capacity, retrieval quality, retrieval overhead, and approximate "
        "context size. It does **not** measure LLM inference speed. Use the live paired benchmark before "
        "claiming faster time to first token or faster total responses.",
        "",
        "| Corpus | Default stored | Default coverage | Cortex recall@6 | Cortex MRR | Cortex p95 retrieval | Default context | Cortex context |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        default = row["default_built_in"]
        cortex = row["cortex"]
        lines.append(
            "| {corpus:,} | {stored:,} | {coverage:.1%} | {recall:.1%} | {mrr:.3f} | "
            "{latency:.3f} ms | {default_tokens:.0f} tokens | {cortex_tokens:.0f} tokens |".format(
                corpus=row["corpus_memories"],
                stored=default["stored_memories"],
                coverage=default["answer_coverage_rate"],
                recall=cortex["recall_at_k"],
                mrr=cortex["mrr"],
                latency=cortex["query_latency"]["p95_ms"],
                default_tokens=default["context_approx_tokens"]["p50"],
                cortex_tokens=cortex["context_approx_tokens"]["p50"],
            )
        )
    lines.extend(
        [
            "",
            "## Method in one paragraph",
            "",
            "The benchmark creates equally important synthetic operating facts, asks labeled exact and mildly "
            "paraphrased questions, and uses the same observed order for both conditions. Hermes built-in "
            "memory packs facts into its configured 2,200-character `MEMORY.md` limit and exposes that frozen "
            "snapshot on every turn. Cortex indexes the full corpus and retrieves up to six memories within its "
            "approximate 700-token budget. Index construction is reported separately and excluded from query "
            "latency. One warm-up query is excluded.",
            "",
            "## Required caveats",
            "",
            "- Synthetic results do not prove the same lift on a personal vault or real conversations.",
            "- Approximate tokens use `ceil(characters / 4)` and are not provider billing tokens.",
            "- The built-in system is a deliberately small curated snapshot, not a failed search engine.",
            "- Cortex remains additive to built-in memory in the current Hermes integration unless built-in prompt injection is disabled.",
            "- Only the paired live-model test can support an inference-speed claim.",
            "",
        ]
    )
    return "\n".join(lines)


def _distribution(values: Sequence[int | float]) -> dict[str, float]:
    if not values:
        return {"minimum": 0.0, "p50": 0.0, "p95": 0.0, "maximum": 0.0, "mean": 0.0}
    ordered = sorted(float(value) for value in values)
    return {
        "minimum": round(ordered[0], 6),
        "p50": round(_percentile(ordered, 0.50), 6),
        "p95": round(_percentile(ordered, 0.95), 6),
        "maximum": round(ordered[-1], 6),
        "mean": round(statistics.fmean(ordered), 6),
    }


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0
