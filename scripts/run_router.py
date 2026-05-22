"""Stage-4 runner: the full evidence-validated cache router.

Examples:
  python scripts/run_router.py --regime exact_repeat --dataset toy --n 20
  python scripts/run_router.py --regime near_miss   --dataset toy --n 20
  python scripts/run_router.py --regime document_drift --dataset toy --n 20
  python scripts/run_router.py --variant naive   --regime near_miss --n 20   # disables gates
  python scripts/run_router.py --variant no-compress --regime paraphrase
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragcache.cache.answer_cache import SemanticAnswerCache
from ragcache.cache.retrieval_cache import RetrievalCache
from ragcache.cache.router import CacheRouter
from ragcache.cache.validator import EvidenceValidator, ValidatorConfig
from ragcache.compression import QueryConditionedCompressor
from ragcache.corpus import chunk_corpus
from ragcache.eval.router_harness import run_router_workload
from ragcache.generator import make_generator
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


def _make_validator(variant: str) -> ValidatorConfig:
    """Variants used in ablations.

    full         -- all 4 gates (paper's proposed system).
    naive        -- semantic cache only; all gates off (stage-3 baseline).
    no-version   -- skip version gate.
    no-evidence  -- skip evidence-IoU gate (loose query-only match).
    no-support   -- skip lexical support gate.
    """
    if variant == "full":
        return ValidatorConfig()
    if variant == "naive":
        return ValidatorConfig(
            query_sim_threshold=0.85, evidence_iou_threshold=0.0,
            require_version_match=False, support_mode="off",
        )
    if variant == "no-version":
        return ValidatorConfig(require_version_match=False)
    if variant == "no-evidence":
        return ValidatorConfig(evidence_iou_threshold=0.0)
    if variant == "no-support":
        return ValidatorConfig(support_mode="off")
    raise ValueError(f"unknown variant: {variant}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--regime", required=True,
                   choices=["exact_repeat", "paraphrase", "near_miss",
                            "document_drift", "long_shared_doc", "bounded_kb_cag"])
    p.add_argument("--dataset", default="toy", choices=["toy", "hotpotqa", "mtrag"])
    p.add_argument("--variant", default="full",
                   choices=["full", "naive", "no-version", "no-evidence", "no-support"])
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-ratio", type=float, default=0.6,
                   help="compressor keep ratio; 1.0 disables compression")
    p.add_argument("--generator", default=None)
    p.add_argument("--out", type=Path, default=Path("runs/router.jsonl"))
    args = p.parse_args()

    docs, qa = _load_dataset(args.dataset, n=max(args.n, 50))
    retriever = EmbeddingRetriever()
    retriever.build(chunk_corpus(docs))

    compressor = None
    if args.keep_ratio < 0.999:
        compressor = QueryConditionedCompressor(
            encode=lambda texts: retriever._encode(texts),
            keep_ratio=args.keep_ratio,
        )

    router = CacheRouter(
        retriever=retriever,
        generator=make_generator(args.generator),
        top_k=args.top_k,
        retrieval_cache=RetrievalCache(approx=True),
        answer_cache=SemanticAnswerCache(threshold=0.90),
        validator=EvidenceValidator(_make_validator(args.variant)),
        compressor=compressor,
    )

    events = build_workload(args.regime, qa, n=args.n, seed=args.seed, docs=docs)
    records, agg = run_router_workload(events, router, base_docs=list(docs),
                                       out_path=args.out, max_tokens=args.max_tokens)

    print(json.dumps({
        "regime": args.regime,
        "dataset": args.dataset,
        "variant": args.variant,
        "keep_ratio": args.keep_ratio,
        "n": agg.n,
        "em_mean": agg.em_mean,
        "f1_mean": agg.f1_mean,
        "ttft_p50": agg.ttft_p50,
        "latency_p50": agg.latency_p50,
        "latency_p95": agg.latency_p95,
        "tokens_in": agg.tokens_in_total,
        "tokens_out": agg.tokens_out_total,
        "hit_rate": agg.hit_rate,
        "false_hit_rate": agg.false_hit_rate,
        "stale_hit_rate": agg.stale_hit_rate,
        "unsupported_rate": agg.unsupported_rate,
        "path_mix": agg.path_mix,
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
