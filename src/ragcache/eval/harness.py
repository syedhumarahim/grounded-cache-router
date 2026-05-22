"""Run a workload through a (possibly cache-routed) pipeline and log JSONL."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..corpus import Document, chunk_corpus
from ..pipeline import RAGPipeline
from ..retriever import EmbeddingRetriever
from ..workload import QueryEvent
from .metrics import Aggregate, QueryRecord, aggregate, exact_match, token_f1


@dataclass
class RunConfig:
    out_path: Path
    top_k: int = 5
    max_tokens: int = 256


def _rebuild_index_for_snapshot(
    retriever: EmbeddingRetriever, snapshot: tuple[Document, ...]
) -> None:
    retriever.build(chunk_corpus(snapshot))


def run_workload(
    events: Iterable[QueryEvent],
    pipeline: RAGPipeline,
    base_docs: list[Document],
    cfg: RunConfig,
) -> tuple[list[QueryRecord], Aggregate]:
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[QueryRecord] = []
    # Assume the caller already built the index with base_docs.
    current_snapshot_key: str = "|".join(d.doc_id + d.version for d in base_docs)

    with cfg.out_path.open("w") as fh:
        for ev in events:
            # Snapshot swap for document_drift regime.
            if ev.corpus_snapshot is not None:
                key = "|".join(d.doc_id + d.version for d in ev.corpus_snapshot)
                if key != current_snapshot_key:
                    _rebuild_index_for_snapshot(pipeline.retriever, ev.corpus_snapshot)
                    current_snapshot_key = key
            else:
                base_key = "|".join(d.doc_id + d.version for d in base_docs)
                if current_snapshot_key != base_key:
                    _rebuild_index_for_snapshot(pipeline.retriever, tuple(base_docs))
                    current_snapshot_key = base_key

            result = pipeline.run(ev.text, max_tokens=cfg.max_tokens)
            em = exact_match(result.answer, ev.gold_answer) if ev.gold_answer else None
            f1 = token_f1(result.answer, ev.gold_answer) if ev.gold_answer else None
            rec = QueryRecord(
                qid=ev.qid,
                regime=ev.regime,
                question=ev.text,
                answer=result.answer,
                gold_answer=ev.gold_answer,
                em=em,
                f1=f1,
                ttft_s=result.gen.ttft_s,
                retrieval_latency_s=result.retrieval_latency_s,
                total_latency_s=result.total_latency_s,
                prompt_tokens=result.gen.prompt_tokens,
                completion_tokens=result.gen.completion_tokens,
                backend=result.gen.backend,
                dedup_dropped=result.dedup.duplicates_dropped,
                dedup_bytes_saved=result.dedup.bytes_saved,
                cache_path=(
                    "retrieval_cache" if result.retrieval_cache_hit in ("exact", "approx")
                    else "generate"
                ),
                cache_hit=result.retrieval_cache_hit in ("exact", "approx"),
            )
            records.append(rec)
            fh.write(json.dumps(rec.to_json()) + "\n")
            fh.flush()
    return records, aggregate(records)
