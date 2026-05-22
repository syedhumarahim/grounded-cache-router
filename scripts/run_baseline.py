"""Stage-1 baseline: vanilla RAG + byte-exact dedup, over a chosen workload regime.

Examples:
  python scripts/run_baseline.py --regime exact_repeat --n 20
  RAGCACHE_GENERATOR=anthropic python scripts/run_baseline.py --regime paraphrase --n 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragcache.corpus import chunk_corpus
from ragcache.eval.harness import RunConfig, run_workload
from ragcache.generator import make_generator
from ragcache.pipeline import RAGPipeline
from ragcache.retriever import EmbeddingRetriever
from ragcache.toy_corpus import DOCS, QA
from ragcache.workload import build_workload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--regime", required=True,
                   choices=["exact_repeat", "paraphrase", "near_miss",
                            "document_drift", "long_shared_doc", "bounded_kb_cag"])
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs/baseline.jsonl"))
    p.add_argument("--generator", default=None,
                   help="echo | anthropic | vllm (defaults to $RAGCACHE_GENERATOR or echo)")
    args = p.parse_args()

    retriever = EmbeddingRetriever()
    retriever.build(chunk_corpus(DOCS))
    pipeline = RAGPipeline(retriever, make_generator(args.generator), top_k=args.top_k)

    events = build_workload(args.regime, QA, n=args.n, seed=args.seed, docs=DOCS)
    cfg = RunConfig(out_path=args.out, top_k=args.top_k, max_tokens=args.max_tokens)
    records, agg = run_workload(events, pipeline, base_docs=list(DOCS), cfg=cfg)

    print(json.dumps({
        "regime": args.regime,
        "n": agg.n,
        "em_mean": agg.em_mean,
        "f1_mean": agg.f1_mean,
        "ttft_p50": agg.ttft_p50,
        "ttft_p95": agg.ttft_p95,
        "latency_p50": agg.latency_p50,
        "latency_p95": agg.latency_p95,
        "tokens_in": agg.tokens_in_total,
        "tokens_out": agg.tokens_out_total,
        "path_mix": agg.path_mix,
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
