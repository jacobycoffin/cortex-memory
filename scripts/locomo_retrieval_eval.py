#!/usr/bin/env python3
"""
LoCoMo retrieval-only evaluation for Cortex.

Answers ONE question: **when we store a conversation, does Cortex's retriever
actually find the turns that contain the answer?**

Design constraints (see plans/Cortex Direction Scan 2026-09-10):
  * Retrieval-only. No LLM judge, no answer generation, no paid data, no
    labelling. Ground truth is LoCoMo's own shipped evidence turn IDs.
  * Per-conversation stores (the standard LoCoMo protocol): each of the 10
    conversations is an independent memory store, so retrieval never has to
    search across conversations that the question doesn't concern.
  * Baselines run in the SAME table: SQLite FTS5/BM25 and a seeded random
    ranker. A number without a baseline is not a result.
  * Nothing here touches the live cortex.db. Every store is a temp file.

Honesty rules enforced in the output:
  * metrics are named hit@k / recall@k, never "accuracy"
  * abstention (retriever returned nothing) is reported separately, never
    silently folded into the misses
  * the LoCoMo filesystem-baseline caveat prints with the table

Usage:
    PYTHONPATH=/tmp/benchpath python3 locomo_retrieval_eval.py \
        --locomo locomo10.json --out report.json [--samples 1] \
        [--limit-questions 50] [--k 10]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------- helpers

def norm(text: str) -> str:
    return " ".join(str(text or "").lower().split())


_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], 1)}


def parse_locomo_date(raw: str | None) -> str | None:
    """LoCoMo dates look like '1:56 pm on 8 May, 2023' — convert to ISO.

    The store expects an ISO-ish timestamp; passing the raw string risks a
    parse failure, so anything we cannot parse becomes None rather than a lie.
    """
    if not raw:
        return None
    text = str(raw).strip().lower()
    try:
        time_part, date_part = text.split(" on ")
        hhmm, ampm = time_part.strip().split()
        hh, mm = (int(x) for x in hhmm.split(":"))
        if ampm == "pm" and hh != 12:
            hh += 12
        elif ampm == "am" and hh == 12:
            hh = 0
        day_str, month_str, year_str = date_part.replace(",", "").split()
        month = _MONTHS.get(month_str.strip())
        if not month:
            return None
        return f"{int(year_str):04d}-{month:02d}-{int(day_str):02d}T{hh:02d}:{mm:02d}:00"
    except Exception:
        return None


def dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_ids, relevant: set, k: int) -> float:
    rels = [1 if m in relevant else 0 for m in ranked_ids[:k]]
    ideal = [1] * min(len(relevant), k)
    idcg = dcg(ideal)
    return (dcg(rels) / idcg) if idcg > 0 else 0.0


def pct(x: float) -> str:
    return f"{100.0 * x:5.1f}%"


# ------------------------------------------------------------ ingestion

def load_conversations(path: Path):
    """Yield (sample_id, turns, qa) where turns carry their dia_id + session."""
    data = json.loads(path.read_text())
    for entry in data:
        sample_id = entry.get("sample_id") or "unknown"
        conv = entry["conversation"]
        # session_N keys hold the turn list; session_N_date_time holds its date
        session_keys = sorted(
            (k for k in conv if k.startswith("session_") and isinstance(conv[k], list)),
            key=lambda k: int(k.split("_")[1]),
        )
        turns = []
        for skey in session_keys:
            session_idx = int(skey.split("_")[1])
            date = parse_locomo_date(conv.get(f"{skey}_date_time"))
            for turn in conv[skey]:
                turns.append({
                    "dia_id": turn.get("dia_id"),
                    "speaker": turn.get("speaker"),
                    "text": turn.get("text") or "",
                    "session": session_idx,
                    "date": date,
                })
        yield sample_id, turns, entry.get("qa") or []


def build_store(turns, sample_id: str):
    """Fresh Cortex store holding one memory per turn. Returns (store, retriever, map)."""
    from cortex.store import CortexStore
    from cortex.retrieval import MemoryRetriever

    tmpdir = tempfile.mkdtemp(prefix=f"locomo-{sample_id}-")
    db_path = Path(tmpdir) / "cortex.db"
    store = CortexStore(str(db_path))

    id_to_dia: dict[str, str] = {}
    dia_to_id: dict[str, str] = {}
    id_to_session: dict[str, int] = {}

    for turn in turns:
        dia = turn["dia_id"]
        if not dia:
            continue
        # Keep the turn verbatim, prefixed with its speaker — this is the unit
        # the evidence IDs point at, so the mapping stays 1:1 and honest.
        content = f"{turn['speaker']}: {turn['text']}"
        result = store.add_memory(
            content,
            kind="episode",
            source_type="conversation",
            source_ref=f"locomo:{sample_id}:{dia}",
            session_id=sample_id,
            observed_at=turn.get("date"),
            approval_state="operator_approved",
        )
        # add_memory returns (memory_id, created_flag) — a tuple, NOT a string.
        # Getting this wrong silently produces an empty ID map and a 0.0%
        # score that looks like a retriever failure.
        if isinstance(result, tuple):
            mid = result[0]
        elif isinstance(result, str):
            mid = result
        else:
            mid = getattr(result, "id", None)
        if not mid:
            continue
        id_to_dia[mid] = dia
        dia_to_id[dia] = mid
        id_to_session[mid] = turn["session"]

    retriever = MemoryRetriever(store)
    return store, retriever, id_to_dia, dia_to_id, id_to_session


# ------------------------------------------------------------- baselines

def fts_baseline(conn: sqlite3.Connection, turns, query: str, k: int):
    """Plain SQLite FTS5/BM25 over the same turn texts — the dumb baseline."""
    rows = conn.execute(
        "SELECT mem_id FROM turns_fts WHERE turns_fts MATCH ? ORDER BY bm25(turns_fts) LIMIT ?",
        (fts_query(query), k),
    ).fetchall()
    return [r[0] for r in rows]


def fts_query(text: str) -> str:
    """Turn a question into a safe FTS5 OR-query; FTS5 syntax is otherwise a trap."""
    terms = [t for t in norm(text).replace("?", " ").replace('"', " ").split() if len(t) > 1]
    terms = [t for t in terms if t.isalnum()]
    return " OR ".join(f'"{t}"' for t in terms[:24]) or '""'


def random_baseline(all_ids: list[str], rng: random.Random, k: int):
    return rng.sample(all_ids, min(k, len(all_ids)))


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo", default="locomo10.json")
    ap.add_argument("--out", default="locomo_retrieval_report.json")
    ap.add_argument("--samples", type=int, default=0, help="limit conversations (0 = all)")
    ap.add_argument("--limit-questions", type=int, default=0, help="limit QA per conversation")
    ap.add_argument("--k", type=int, default=10, help="cut-off for recall/hit metrics")
    ap.add_argument("--seed", type=int, default=20260910)
    args = ap.parse_args()

    path = Path(args.locomo)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 2

    K = args.k
    rng = random.Random(args.seed)

    # aggregated results: metric -> list of per-question values
    agg = defaultdict(list)
    agg_cat = defaultdict(lambda: defaultdict(list))
    per_conv = []
    skipped_empty_evidence = 0
    total_turns = 0

    t_start = time.time()
    conversations = list(load_conversations(path))
    if args.samples:
        conversations = conversations[: args.samples]

    for ci, (sample_id, turns, qa) in enumerate(conversations, 1):
        t0 = time.time()
        store, retriever, id_to_dia, dia_to_id, id_to_session = build_store(turns, sample_id)
        all_ids = list(id_to_dia.keys())
        total_turns += len(all_ids)

        # FTS5 baseline over the identical turn texts
        turns_by_dia = {
            t["dia_id"]: f"{t['speaker']}: {t['text']}"
            for t in turns if t.get("dia_id")
        }
        conn = store._conn
        conn.execute("DROP TABLE IF EXISTS turns_fts")
        conn.execute("CREATE VIRTUAL TABLE turns_fts USING fts5(mem_id UNINDEXED, body)")
        conn.executemany(
            "INSERT INTO turns_fts(mem_id, body) VALUES (?,?)",
            [(mid, turns_by_dia.get(dia, "")) for mid, dia in id_to_dia.items()],
        )
        conn.commit()

        questions = qa
        if args.limit_questions:
            questions = questions[: args.limit_questions]

        n_scored = 0
        for q in questions:
            evidence = [e for e in (q.get("evidence") or []) if e in dia_to_id]
            if not evidence:
                skipped_empty_evidence += 1
                continue
            relevant_ids = {dia_to_id[e] for e in evidence}
            relevant_sessions = {id_to_session[dia_to_id[e]] for e in evidence}
            qtext = q.get("question") or ""
            cat = int(q.get("category") or 0)

            # --- Cortex
            results, diag = retriever.search_detailed(qtext, limit=K)
            ranked = [r.memory["id"] for r in results]
            abstained = bool(getattr(diag, "abstained", False)) or not ranked

            # --- baselines
            bm25_ranked = fts_baseline(conn, turns, qtext, K)
            rand_ranked = random_baseline(all_ids, rng, K)

            for name, ranked_ids, is_abstain in (
                ("cortex", ranked, abstained),
                ("bm25", bm25_ranked, not bm25_ranked),
                ("random", rand_ranked, False),
            ):
                hit = 1 if relevant_ids & set(ranked_ids) else 0
                found = len(relevant_ids & set(ranked_ids))
                rec = found / len(relevant_ids)
                rr = 0.0
                for rank, mid in enumerate(ranked_ids, 1):
                    if mid in relevant_ids:
                        rr = 1.0 / rank
                        break
                sess_hit = 1 if relevant_sessions & {id_to_session[m] for m in ranked_ids if m in id_to_session} else 0
                nd = ndcg_at_k(ranked_ids, relevant_ids, K)

                for store_key in (agg, agg_cat[cat]):
                    store_key[f"{name}_hit_at_{K}"].append(hit)
                    store_key[f"{name}_recall_at_{K}"].append(rec)
                    store_key[f"{name}_mrr_at_{K}"].append(rr)
                    store_key[f"{name}_ndcg_at_{K}"].append(nd)
                    store_key[f"{name}_session_hit_at_{K}"].append(sess_hit)
                if name == "cortex":
                    agg["cortex_abstain"].append(1 if is_abstain else 0)
                    agg_cat[cat]["cortex_abstain"].append(1 if is_abstain else 0)
            n_scored += 1

        per_conv.append({
            "sample_id": sample_id,
            "memories": len(all_ids),
            "questions_scored": n_scored,
            "seconds": round(time.time() - t0, 1),
        })
        print(f"  [{ci}/{len(conversations)}] {sample_id}: {len(all_ids)} turns, "
              f"{n_scored} questions, {time.time() - t0:.0f}s", flush=True)

    elapsed = time.time() - t_start

    # ------------------------------------------------------------ report
    def mean(key, block=None):
        vals = (block or agg).get(key) or []
        return statistics.fmean(vals) if vals else float("nan")

    lines = []
    lines.append("=" * 78)
    lines.append("CORTEX RETRIEVAL EVALUATION — LoCoMo (retrieval-only)")
    lines.append("=" * 78)
    lines.append(f"conversations: {len(per_conv)}   turns stored: {total_turns}   "
                 f"questions scored: {len(agg['cortex_hit_at_%d' % K])}")
    lines.append(f"excluded for empty evidence: {skipped_empty_evidence}   "
                 f"cut-off k={K}   seed={args.seed}   wall: {elapsed:.0f}s")
    lines.append("")
    lines.append(f"{'method':<10} {'hit@'+str(K):>9} {'recall@'+str(K):>11} "
                 f"{'MRR@'+str(K):>9} {'nDCG@'+str(K):>10} {'session-hit':>12}")
    lines.append("-" * 78)
    for name in ("cortex", "bm25", "random"):
        lines.append(
            f"{name:<10} {pct(mean(f'{name}_hit_at_{K}')):>9} "
            f"{pct(mean(f'{name}_recall_at_{K}')):>11} "
            f"{mean(f'{name}_mrr_at_{K}'):>9.3f} "
            f"{mean(f'{name}_ndcg_at_{K}'):>10.3f} "
            f"{pct(mean(f'{name}_session_hit_at_{K}')):>12}"
        )
    lines.append("-" * 78)
    lines.append(f"Cortex abstained (returned nothing) on "
                 f"{pct(mean('cortex_abstain'))} of questions.")
    lines.append("Nearest-neighbour sanity floor: random ≈ "
                 f"{pct(mean(f'random_hit_at_{K}'))} — anything at or below this is noise.")

    # per-category
    catname = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
    lines.append("")
    lines.append(f"{'category':<14} {'n':>6} {'cortex hit':>11} {'bm25 hit':>10} {'cortex MRR':>11}")
    lines.append("-" * 78)
    for cat in sorted(agg_cat):
        block = agg_cat[cat]
        n = len(block.get(f"cortex_hit_at_{K}") or [])
        if not n:
            continue
        lines.append(
            f"{catname.get(cat, str(cat)):<14} {n:>6} "
            f"{pct(mean(f'cortex_hit_at_{K}', block)):>11} "
            f"{pct(mean(f'bm25_hit_at_{K}', block)):>10} "
            f"{mean(f'cortex_mrr_at_{K}', block):>11.3f}"
        )

    lines.append("")
    lines.append("CAVEATS (required reading):")
    lines.append("  * This is RETRIEVAL recall — did we find the evidence turn. It is NOT")
    lines.append("    end-to-end answer quality; no answer was generated or judged.")
    lines.append("  * LoCoMo's own published filesystem baseline scores ~74% on this data.")
    lines.append("    Compare against that, not against a vendor's end-to-end QA number.")
    lines.append("  * Never place this number in a table beside vendor scores: different")
    lines.append("    judges, backbones and top_k make those incomparable.")
    lines.append("  * This corpus is strangers' conversations, NOT the operator's vault. It")
    lines.append("    measures the retriever mechanism, not whether Cortex is useful to him.")

    report = "\n".join(lines)
    print("\n" + report)

    payload = {
        "meta": {
            "benchmark": "LoCoMo (snap-research/locomo, CC BY-NC 4.0)",
            "mode": "retrieval-only",
            "k": K,
            "seed": args.seed,
            "conversations": len(per_conv),
            "turns": total_turns,
            "questions_scored": len(agg[f"cortex_hit_at_{K}"]),
            "excluded_empty_evidence": skipped_empty_evidence,
            "wall_seconds": round(elapsed, 1),
            "caveat_filesystem_baseline": 0.74,
        },
        "aggregate": {k: (statistics.fmean(v) if v else None) for k, v in agg.items()},
        "per_category": {
            str(c): {k: (statistics.fmean(v) if v else None) for k, v in blk.items()}
            for c, blk in agg_cat.items()
        },
        "per_conversation": per_conv,
        "report_text": report,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
