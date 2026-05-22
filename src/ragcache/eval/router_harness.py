"""Harness for the stage-4 CacheRouter.

Mirrors `eval.harness.run_workload` but uses RouterResult and populates the
cache-safety counters (false_hit / stale_hit / unsupported) so the paper's
Table 2 (cache safety) can be computed directly from JSONL.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..cache.router import CacheRouter, RouterResult
from ..cache.validator import lexical_support
from ..corpus import Document, chunk_corpus
from ..signature import EvidenceSignature
from ..workload import QueryEvent
from .metrics import Aggregate, QueryRecord, aggregate, contains_gold, exact_match, token_f1


def _record_from_router(ev: QueryEvent, r: RouterResult,
                        gold_f1_threshold: float = 0.5,
                        stale_support_threshold: float = 0.6) -> QueryRecord:
    em = exact_match(r.answer, ev.gold_answer) if ev.gold_answer else None
    f1 = token_f1(r.answer, ev.gold_answer) if ev.gold_answer else None
    # Substring containment: an extractive answer that returns a whole
    # sentence containing the short gold factoid is "correct" for the
    # purpose of false-hit auditing.
    contained = contains_gold(r.answer, ev.gold_answer) if ev.gold_answer else None

    is_answer_hit = r.decision.path == "answer_cache"
    is_retrieval_hit = r.decision.retrieval_hit_kind in ("exact", "approx")

    # Cache-safety counters (only meaningful on answer-cache hits).
    cache_false_hit = False
    cache_stale_hit = False
    cache_unsupported = False
    if is_answer_hit:
        # A hit is a false hit when neither the F1 nor the substring check
        # confirms the cached answer matches gold.
        if (ev.gold_answer is not None and f1 is not None
                and f1 < gold_f1_threshold and not contained):
            cache_false_hit = True
        # If validator skipped the version gate, an undetected stale hit is
        # possible -- compare signatures stored in gate_report.
        rep = r.decision.gate_report
        if rep is not None and not rep.versions_ok:
            cache_stale_hit = True
        # Lexical support is recomputed cheaply here even when support gate
        # was off, so the counter is always meaningful for safety analysis.
        # (No fresh chunks accessible here -- skip if validator was off.)
        if rep is not None and rep.support_score is not None and rep.support_score < stale_support_threshold:
            cache_unsupported = True

    path = "answer_cache" if is_answer_hit else (
        "retrieval_cache" if is_retrieval_hit else "generate"
    )

    return QueryRecord(
        qid=ev.qid,
        regime=ev.regime,
        question=ev.text,
        answer=r.answer,
        gold_answer=ev.gold_answer,
        em=em, f1=f1,
        ttft_s=(r.gen.ttft_s if r.gen else 0.0),
        retrieval_latency_s=r.retrieval_latency_s,
        total_latency_s=r.total_latency_s,
        prompt_tokens=(r.gen.prompt_tokens if r.gen else 0),
        completion_tokens=(r.gen.completion_tokens if r.gen else 0),
        backend=(r.gen.backend if r.gen else "answer_cache"),
        cache_path=path,
        cache_hit=is_answer_hit or is_retrieval_hit,
        cache_false_hit=cache_false_hit,
        cache_stale_hit=cache_stale_hit,
        cache_unsupported=cache_unsupported,
        dedup_dropped=r.dedup.duplicates_dropped,
        dedup_bytes_saved=r.dedup.bytes_saved,
    )


def run_router_workload(
    events: Iterable[QueryEvent],
    router: CacheRouter,
    base_docs: list[Document],
    out_path: Path,
    max_tokens: int = 256,
) -> tuple[list[QueryRecord], Aggregate]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[QueryRecord] = []
    current_key = "|".join(d.doc_id + d.version for d in base_docs)

    with out_path.open("w") as fh:
        for ev in events:
            if ev.corpus_snapshot is not None:
                key = "|".join(d.doc_id + d.version for d in ev.corpus_snapshot)
                if key != current_key:
                    router.build(chunk_corpus(ev.corpus_snapshot))
                    current_key = key
            else:
                base_key = "|".join(d.doc_id + d.version for d in base_docs)
                if current_key != base_key:
                    router.build(chunk_corpus(base_docs))
                    current_key = base_key

            r = router.run(ev.text, max_tokens=max_tokens)
            rec = _record_from_router(ev, r)
            records.append(rec)
            fh.write(json.dumps(rec.to_json()) + "\n")
            fh.flush()
    return records, aggregate(records)
