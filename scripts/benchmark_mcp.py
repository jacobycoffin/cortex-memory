#!/usr/bin/env python3
"""Recall latency and hit rate through the stdio MCP server (Claude Code path).

Builds the same synthetic corpus as ``benchmark_recall.py`` (generate_facts) in a
temporary CORTEX_HOME, starts ``python -m cortex.mcp_server`` as a subprocess, and
times ``memory_search`` round trips over JSON-RPC, so the numbers include
the protocol and process boundary that Claude Code pays. Aggregates only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=2000)
    ap.add_argument("--repetitions", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model-dir", help="bge-small-en-v1.5 directory (semantic fusion); omit for lexical only")
    ap.add_argument("--python", default=sys.executable, help="Python with cortex installed")
    ap.add_argument("--output")
    args = ap.parse_args()

    from cortex.benchmarks.core import generate_facts
    from cortex.store import CortexStore

    facts = generate_facts(args.size, seed=args.seed)
    cases = [facts[(i * 47) % args.size] for i in range(args.repetitions)]
    with tempfile.TemporaryDirectory(prefix="cortex-mcp-bench-") as tmp:
        home = Path(tmp)
        store = CortexStore(home / "cortex.db")
        with store.transaction():
            for fact in facts:
                store.add_memory(fact.content, kind="decision", source_type="synthetic_benchmark",
                                 confidence=0.95, importance=0.7)
        store.close()
        env = dict(os.environ, CORTEX_HOME=str(home))
        if args.model_dir:
            env["CORTEX_MODEL_DIR"] = args.model_dir
        else:
            env["CORTEX_MODEL_DIR"] = str(home / "no-model")
        proc = subprocess.Popen([args.python, "-m", "cortex.mcp_server"], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
        ids = iter(range(1, 10**6))

        def rpc(method: str, params: dict | None = None) -> dict:
            msg = {"jsonrpc": "2.0", "id": next(ids), "method": method, "params": params or {}}
            proc.stdin.write(json.dumps(msg) + "\n")
            proc.stdin.flush()
            return json.loads(proc.stdout.readline())

        started = time.perf_counter()
        rpc("initialize", {"protocolVersion": "2025-06-18"})
        rpc("tools/call", {"name": "stats", "arguments": {}})
        # Let the background warm-up (embedder load + embedding backfill) finish.
        deadline = time.time() + 900
        while time.time() < deadline:
            st = json.loads(rpc("tools/call", {"name": "stats", "arguments": {}})["result"]["content"][0]["text"])
            if not args.model_dir or st.get("embeddings", 0) >= args.size:
                break
            time.sleep(1)
        warm_s = time.perf_counter() - started
        latencies, hits = [], 0
        for fact in cases:
            t0 = time.perf_counter()
            res = rpc("tools/call", {"name": "memory_search", "arguments": {"query": fact.query, "limit": 6}})
            latencies.append((time.perf_counter() - t0) * 1000)
            text = res["result"]["content"][0]["text"]
            hits += int(fact.content in text)
        proc.stdin.close()
        proc.wait(timeout=30)
    ordered = sorted(latencies)
    report = {
        "path": "mcp_stdio",
        "semantic": bool(args.model_dir),
        "memory_count": args.size,
        "repetitions": args.repetitions,
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[max(0, (len(ordered) * 95 + 99) // 100 - 1)], 3),
        "throughput_qps": round(1000 / statistics.mean(ordered), 1),
        "synthetic_hit_at_6": hits / len(cases),
        "startup_and_warm_s": round(warm_s, 1),
    }
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
