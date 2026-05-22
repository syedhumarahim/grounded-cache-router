"""Stage-2 runner: RAG + byte-exact dedup + retrieval-result cache.

Compared with run_baseline.py: vector search is skipped on cache hits, prompts
are ordered prefix-friendly, and the JSONL log records cache_path / cache_hit
per query so we can compute hit-rate and latency deltas downstream.

Examples:
  python scripts/run_with_cache.py --regime exact_repeat --n 30
  python scripts/run_with_cache.py --regime paraphrase --n 30 --approx
  python scripts/run_with_cache.py --regime near_miss --n 30 --approx --threshold 0.95
  python scripts/run_with_cache.py --dataset hotpotqa --n 50 --regime exact_repeat
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragcache.cache.retrieval_cache import CachedRetriever, RetrievalCache
from ragcache.corpus import chunk_corpus
from ragcache.eval.harness import RunConfig, run_workload
from ragcache.generator import make_generator
from ragcache.pipeline import RAGPipeline
from ragcache.retriever import EmbeddingRetriever
from ragcache.toy_corpus import DOCS as TOY_DOCS, QA as TOY_QA
from ragcache.workload import build_workload


def _load_dataset(name: str, n: int):
    if name == "toy":
        return list(TOY_DOCS), list(TOY_QA)
    if name == "hotpotqa":
        from ragcache.datasets import load_hotpotqa
        return load_hotpotqa(n=n)
    if name == "mtrag":
        from ragcache.mtrag import load_mtrag
        return load_mtrag(max_conversations=max(20, n // 5))
    raise ValueError(f"unknown dataset: {name}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--regime", required=True,
                   choices=["exact_repeat", "paraphrase", "near_miss",
                            "document_drift", "long_shared_doc", "bounded_kb_cag"])
    p.add_argument("--dataset", default="toy", choices=["toy", "hotpotqa", "mtrag"])
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--approx", action="store_true",
                   help="enable approximate (embedding-similarity) retrieval cache")
    p.add_argument("--threshold", type=float, default=0.92)
    p.add_argument("--no-prefix-order", action="store_true",
                   help="disable prefix-cache-friendly chunk ordering")
    p.add_argument("--out", type=Path, default=Path("runs/cached.jsonl"))
    p.add_argument("--generator", default=None)
    args = p.parse_args()

    docs, qa = _load_dataset(args.dataset, n=max(args.n, 50))
    retriever = EmbeddingRetriever()
    retriever.build(chunk_corpus(docs))

    cache = RetrievalCache(approx=args.approx, approx_threshold=args.threshold)
    cached = CachedRetriever(retriever, cache)

    pipeline = RAGPipeline(
        cached,
        make_generator(args.generator),
        top_k=args.top_k,
        prefix_friendly_order=not args.no_prefix_order,
    )

    events = build_workload(args.regime, qa, n=args.n, seed=args.seed, docs=docs)
    cfg = RunConfig(out_path=args.out, top_k=args.top_k, max_tokens=args.max_tokens)
    records, agg = run_workload(events, pipeline, base_docs=list(docs), cfg=cfg)

    print(json.dumps({
        "regime": args.regime,
        "dataset": args.dataset,
        "n": agg.n,
        "em_mean": agg.em_mean,
        "f1_mean": agg.f1_mean,
        "ttft_p50": agg.ttft_p50,
        "latency_p50": agg.latency_p50,
        "latency_p95": agg.latency_p95,
        "tokens_in": agg.tokens_in_total,
        "tokens_out": agg.tokens_out_total,
        "hit_rate": agg.hit_rate,
        "path_mix": agg.path_mix,
        "cache_stats": {
            "lookups": cache.stats.lookups,
            "exact_hits": cache.stats.exact_hits,
            "approx_hits": cache.stats.approx_hits,
            "misses": cache.stats.misses,
            "avg_lookup_us": (cache.stats.total_lookup_s / max(1, cache.stats.lookups)) * 1e6,
        },
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
