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

    # semantic fusion (0.0 = OFF, the shipped default):
    PYTHONPATH=/tmp/benchpath python3 locomo_retrieval_eval.py \
        --locomo locomo10.json --out report.json --semantic-weight 10

With ``--semantic-weight`` > 0 the harness embeds every stored turn with the
local ONNX model and runs BOTH arms — fusion and weight 0 — over the SAME stores
and the SAME questions, then reports the paired per-question delta and its
standard error. Paired deltas support an SE; two separately-run averages would
not. Embeddings are generated in-run, so measuring fusion needs no backfill step
and no API key. If the embedding model is unavailable the run ABORTS rather than
quietly reporting a feature-only number as "fusion".

Measured 2026-09-11 (all 10 conversations, 5880 turns, 1977 questions, w=10):
multi-hop hit@10 56.6% -> 65.8% (+9.3, SE 2.0) and single-hop 70.2% -> 73.8%
(+3.7, SE 1.0) — both beyond 2SE, so fusion helps without a single-hop
regression. open-domain (+5.6, SE 3.3) and adversarial (+1.1, SE 1.2) are inside
noise and are reported as such.
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


# Metric order returned by score_ranking(); "abstain" is excluded from the
# paired numeric deltas because a hit/recall delta against a bool is meaningless.
_PAIRED_METRICS = ("hit", "recall", "mrr", "ndcg", "session_hit", "abstain")


def score_ranking(ranked_ids, relevant_ids: set, relevant_sessions: set,
                  id_to_session: dict, k: int, is_abstain: bool):
    """Score ONE ranking: (hit, recall, MRR, nDCG, session-hit, abstain).

    Both the fusion arm and the weight-0 arm go through this identical function,
    so a difference between them can only come from the ranking itself and never
    from two divergent copies of the metric code.
    """
    found = len(relevant_ids & set(ranked_ids))
    hit = 1 if found else 0
    recall = found / len(relevant_ids)
    rr = 0.0
    for rank, mid in enumerate(ranked_ids, 1):
        if mid in relevant_ids:
            rr = 1.0 / rank
            break
    session_hit = 1 if relevant_sessions & {
        id_to_session[m] for m in ranked_ids if m in id_to_session
    } else 0
    return (hit, recall, rr, ndcg_at_k(ranked_ids, relevant_ids, k),
            session_hit, is_abstain)


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


def build_store(turns, sample_id: str, semantic_weight: float = 0.0,
                semantic_pool: int = 20):
    """Fresh Cortex store holding one memory per turn.

    Returns ``(store, retriever, baseline_retriever, id_to_dia, dia_to_id,
    id_to_session, id_to_text)``. When fusion is enabled, ``baseline_retriever``
    is a weight-0 retriever over the **same** store, so both arms see identical
    data and the comparison is paired rather than two separate runs.
    """
    from cortex.store import CortexStore
    from cortex.retrieval import MemoryRetriever

    tmpdir = tempfile.mkdtemp(prefix=f"locomo-{sample_id}-")
    db_path = Path(tmpdir) / "cortex.db"
    store = CortexStore(str(db_path))

    id_to_dia: dict[str, str] = {}
    dia_to_id: dict[str, str] = {}
    id_to_session: dict[str, int] = {}
    id_to_text: dict[str, str] = {}

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
        id_to_text[mid] = content

    retriever = MemoryRetriever(
        store, semantic_weight=semantic_weight, semantic_pool=semantic_pool
    )
    baseline_retriever = (
        MemoryRetriever(store, semantic_weight=0.0, semantic_pool=semantic_pool)
        if semantic_weight > 0.0
        else None
    )
    return (store, retriever, baseline_retriever,
            id_to_dia, dia_to_id, id_to_session, id_to_text)


def embed_store(store, id_to_text: dict, model_dir=None, batch_size: int = 256) -> int:
    """Give every stored turn its local embedding. Returns vectors written.

    Raises ``RuntimeError`` when the embedding model is unavailable instead of
    silently continuing. A fusion run that quietly fell back to the
    feature-only path would publish a "fusion" number that never used fusion —
    the exact failure this harness exists to prevent. Fail loud, never fake.
    """
    from cortex.embeddings import MODEL_ID, get_embedder

    embedder = get_embedder(model_dir)
    if not getattr(embedder, "available", False):
        raise RuntimeError(
            "semantic fusion was requested but the local embedding model is "
            f"unavailable under {getattr(embedder, 'model_dir', '?')} "
            f"({getattr(embedder, 'load_error', 'unknown reason')}); refusing to "
            "report a fusion number produced without embeddings."
        )

    items = [(mid, text) for mid, text in id_to_text.items() if text and text.strip()]
    written = 0
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        vectors = embedder.embed([text for _, text in batch])
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"embedder returned {len(vectors)} vectors for {len(batch)} texts"
            )
        for (mid, _text), vector in zip(batch, vectors):
            if not vector:
                raise RuntimeError(f"empty embedding vector for {mid}; aborting")
            store.set_memory_embedding(mid, MODEL_ID, vector)
            written += 1
    return written


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

def build_arg_parser() -> argparse.ArgumentParser:
    """The CLI surface, extracted so the fusion defaults can be pinned by tests."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo", default="locomo10.json")
    ap.add_argument("--out", default="locomo_retrieval_report.json")
    ap.add_argument("--samples", type=int, default=0, help="limit conversations (0 = all)")
    ap.add_argument("--limit-questions", type=int, default=0, help="limit QA per conversation")
    ap.add_argument("--k", type=int, default=10, help="cut-off for recall/hit metrics")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument(
        "--semantic-weight", type=float, default=0.0,
        help="Local embedding fusion weight. 0.0 = OFF, the shipped default "
             "(reproduces the published baseline). >0 enables fusion AND runs a "
             "paired weight-0 arm over the same stores in the same pass.",
    )
    ap.add_argument(
        "--semantic-pool", type=int, default=20,
        help="How many semantic candidates fusion adds to the pool.",
    )
    ap.add_argument(
        "--embed-model-dir", default=None,
        help="Override the local embedding model directory.",
    )
    return ap


def main() -> int:
    ap = build_arg_parser()
    args = ap.parse_args()

    if args.semantic_weight < 0.0:
        print("ERROR: --semantic-weight must be >= 0", file=sys.stderr)
        return 2
    if args.semantic_pool < 1:
        print("ERROR: --semantic-pool must be >= 1", file=sys.stderr)
        return 2

    path = Path(args.locomo)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 2

    K = args.k
    rng = random.Random(args.seed)
    fusion_enabled = args.semantic_weight > 0.0

    # aggregated results: metric -> list of per-question values
    agg = defaultdict(list)
    agg_cat = defaultdict(lambda: defaultdict(list))
    # Paired per-question deltas (fusion arm minus weight-0 arm) measured over
    # the SAME stores and the SAME questions. Paired deltas carry a meaningful
    # standard error; comparing two separately-run averages would not.
    paired = defaultdict(list)
    paired_cat = defaultdict(lambda: defaultdict(list))
    per_conv = []
    skipped_empty_evidence = 0
    total_turns = 0
    embedded_total = 0

    t_start = time.time()
    conversations = list(load_conversations(path))
    if args.samples:
        conversations = conversations[: args.samples]

    for ci, (sample_id, turns, qa) in enumerate(conversations, 1):
        t0 = time.time()
        (store, retriever, baseline_retriever, id_to_dia, dia_to_id,
         id_to_session, id_to_text) = build_store(
            turns, sample_id, args.semantic_weight, args.semantic_pool
        )
        all_ids = list(id_to_dia.keys())
        total_turns += len(all_ids)
        if fusion_enabled:
            # Embeddings are generated here, per-conversation, so a fusion run
            # is self-contained: clone the repo, run one command, get the number.
            vectors_written = embed_store(store, id_to_text, args.embed_model_dir)
            embedded_total += vectors_written
            if vectors_written < len(all_ids):
                print(f"  WARNING: embedded {vectors_written}/{len(all_ids)} turns — "
                      f"turns without a vector are invisible to the semantic arm.",
                      flush=True)

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

            # --- Cortex: fusion arm when enabled, otherwise the single arm
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
                hit, rec, rr, nd, sess_hit, abst = score_ranking(
                    ranked_ids, relevant_ids, relevant_sessions, id_to_session,
                    K, is_abstain,
                )

                for store_key in (agg, agg_cat[cat]):
                    store_key[f"{name}_hit_at_{K}"].append(hit)
                    store_key[f"{name}_recall_at_{K}"].append(rec)
                    store_key[f"{name}_mrr_at_{K}"].append(rr)
                    store_key[f"{name}_ndcg_at_{K}"].append(nd)
                    store_key[f"{name}_session_hit_at_{K}"].append(sess_hit)
                if name == "cortex":
                    agg["cortex_abstain"].append(1 if abst else 0)
                    agg_cat[cat]["cortex_abstain"].append(1 if abst else 0)

            # Paired weight-0 arm: same store, same question, same scorer.
            if baseline_retriever is not None:
                b_results, b_diag = baseline_retriever.search_detailed(qtext, limit=K)
                b_ranked = [r.memory["id"] for r in b_results]
                b_abstained = bool(getattr(b_diag, "abstained", False)) or not b_ranked
                fusion_m = score_ranking(ranked, relevant_ids, relevant_sessions,
                                         id_to_session, K, abstained)
                base_m = score_ranking(b_ranked, relevant_ids, relevant_sessions,
                                       id_to_session, K, b_abstained)
                for metric, fusion_val, base_val in zip(_PAIRED_METRICS, fusion_m, base_m):
                    if metric == "abstain":
                        continue
                    diff = fusion_val - base_val
                    paired[f"{metric}_at_{K}"].append(diff)
                    paired_cat[cat][f"{metric}_at_{K}"].append(diff)
                for idx, metric in enumerate(_PAIRED_METRICS[:5]):
                    for store_key in (agg, agg_cat[cat]):
                        store_key[f"baseline_w0_{metric}_at_{K}"].append(base_m[idx])
                agg["baseline_w0_abstain"].append(1 if b_abstained else 0)
                agg_cat[cat]["baseline_w0_abstain"].append(1 if b_abstained else 0)
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
    if fusion_enabled:
        lines.append(f"semantic fusion: ON   weight={args.semantic_weight}   "
                     f"pool={args.semantic_pool}   embeddings written={embedded_total}")
        lines.append("  'cortex' is the FUSION arm; 'baseline_w0' is weight 0 over "
                     "the same stores")
    else:
        lines.append("semantic fusion: OFF (weight 0.0 — the shipped default)")
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

    # paired fusion-vs-baseline comparison
    if fusion_enabled:
        def paired_stats(block):
            """(n, mean paired delta, standard error) for the hit@K deltas."""
            diffs = block.get(f"hit_at_{K}") or []
            n = len(diffs)
            if not n:
                return 0, float("nan"), float("nan")
            m = statistics.fmean(diffs)
            se = statistics.stdev(diffs) / math.sqrt(n) if n > 1 else 0.0
            return n, m, se

        lines.append("")
        lines.append(f"PAIRED COMPARISON — fusion(w={args.semantic_weight}) minus "
                     f"baseline(w=0), same stores + same questions")
        lines.append(f"{'slice':<14} {'n':>6} {'baseline':>10} {'fusion':>9} "
                     f"{'delta':>9} {'SE':>7} {'>2SE':>6}")
        lines.append("-" * 78)
        rows = [("ALL", agg, paired)] + [
            (catname.get(c, str(c)), agg_cat[c], paired_cat[c])
            for c in sorted(paired_cat)
        ]
        for label, mblock, pblock in rows:
            n, delta, se = paired_stats(pblock)
            if not n:
                continue
            base_hit = mean(f"baseline_w0_hit_at_{K}", mblock)
            fusion_hit = mean(f"cortex_hit_at_{K}", mblock)
            if se == 0:
                verdict = "n/a"
            else:
                verdict = "yes" if abs(delta) > 2 * se else "no"
            lines.append(
                f"{label:<14} {n:>6} {pct(base_hit):>10} {pct(fusion_hit):>9} "
                f"{delta * 100:>+8.1f} {se * 100:>7.1f} {verdict:>6}"
            )
        lines.append("delta = mean of PER-QUESTION differences, so its SE is the paired")
        lines.append("SE. '>2SE' = yes means the shift is larger than twice that SE.")

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

    paired_payload: dict = {}
    if fusion_enabled:
        def _summarise(deltas):
            n = len(deltas)
            if not n:
                return {"n": 0, "mean_delta": None, "se": None}
            return {
                "n": n,
                "mean_delta": statistics.fmean(deltas),
                "se": (statistics.stdev(deltas) / math.sqrt(n)) if n > 1 else 0.0,
            }

        paired_payload["ALL"] = {k: _summarise(v) for k, v in paired.items()}
        for cat_id, block in paired_cat.items():
            paired_payload[str(cat_id)] = {k: _summarise(v) for k, v in block.items()}

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
            "semantic_fusion_enabled": fusion_enabled,
            "semantic_fusion_weight": args.semantic_weight,
            "semantic_fusion_pool": args.semantic_pool,
            "embeddings_written": embedded_total,
        },
        "paired_fusion_vs_baseline": paired_payload,
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
