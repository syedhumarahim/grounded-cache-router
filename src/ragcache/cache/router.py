"""Stage 4: the cache router (paper's core system).

Routes each query through, in order:

  1. retrieval-result cache lookup     (cheap, deterministic)
  2. semantic answer-cache lookup       (cosine query similarity)
  3. evidence-validation gate           (Jaccard + version + support)
       if pass -> return cached answer  (cache_path=answer_cache)
       else    -> fall through
  4. compression fallback              (query-conditioned)
  5. generate                          (LLM call)

Every per-query record tracks the gate outcomes so post-hoc analysis can
attribute every saved generation, every false-hit avoided, and every
unnecessary recomputation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ..compression import CompressionStats, QueryConditionedCompressor
from ..corpus import Chunk, dedup_chunks, DedupStats
from ..generator import Generator, GenResult
from ..pipeline import PROMPT_TEMPLATE, _stable_order
from ..retriever import EmbeddingRetriever, Retrieved
from ..signature import EvidenceSignature
from .answer_cache import SemanticAnswerCache
from .retrieval_cache import CachedRetriever, RetrievalCache
from .validator import EvidenceValidator, GateReport, ValidatorConfig


@dataclass
class RouterDecision:
    path: str                                 # answer_cache | retrieval_cache | generate
    answer_cache_hit_sim: float | None = None
    gate_report: GateReport | None = None
    retrieval_hit_kind: str | None = None     # exact | approx | miss
    compression_ratio: float | None = None


@dataclass
class RouterResult:
    question: str
    answer: str
    signature: EvidenceSignature
    gen: GenResult | None                     # None on answer_cache hit
    decision: RouterDecision
    dedup: DedupStats
    compression: CompressionStats | None
    retrieval_latency_s: float
    total_latency_s: float


class CacheRouter:
    def __init__(
        self,
        retriever: EmbeddingRetriever,
        generator: Generator,
        top_k: int = 5,
        retrieval_cache: RetrievalCache | None = None,
        answer_cache: SemanticAnswerCache | None = None,
        validator: EvidenceValidator | None = None,
        compressor: QueryConditionedCompressor | None = None,
        enable_compression: bool = True,
    ):
        self.retriever = retriever
        self.cached_retriever = CachedRetriever(
            retriever, retrieval_cache or RetrievalCache(approx=True),
        )
        self.generator = generator
        self.top_k = top_k
        self.answer_cache = answer_cache or SemanticAnswerCache()
        self.validator = validator or EvidenceValidator(ValidatorConfig())
        self.compressor = compressor
        self.enable_compression = enable_compression and (compressor is not None)

    # convenience for harness snapshot swaps
    def build(self, chunks) -> None:
        # Rebuild the retrieval index for the new corpus snapshot, but
        # deliberately keep the answer cache live: that is exactly the
        # state under which G3 (version match) and G2 (evidence overlap)
        # earn their keep on document-drift traffic.
        self.cached_retriever.build(chunks)

    def _embed(self, q: str):
        return self.retriever._encode([q])[0]

    def _format_context(self, chunks: list[Chunk]) -> str:
        return "\n\n".join(f"[{c.chunk_id} v{c.version}]\n{c.text}" for c in chunks)

    def run(self, question: str, max_tokens: int = 256) -> RouterResult:
        t_total = time.perf_counter()

        # 1. retrieval (with retrieval-result cache).
        cr = self.cached_retriever.search(question, k=self.top_k)
        retrieved = cr.retrieved
        retrieval_latency = cr.retrieval_latency_s + cr.lookup_latency_s

        # 2. dedup + signature + prefix-friendly order.
        deduped, dstats = dedup_chunks(retrieved.chunks)
        keep_ids = {c.chunk_id for c in deduped}
        kept_scores = [s for c, s in zip(retrieved.chunks, retrieved.scores) if c.chunk_id in keep_ids]
        deduped, kept_scores = _stable_order(deduped, kept_scores)
        fresh_sig = EvidenceSignature.from_chunks(deduped, kept_scores)

        # 3. answer-cache lookup + validation.
        q_emb = self._embed(question)
        hit = self.answer_cache.lookup(q_emb)
        if hit is not None:
            report = self.validator.validate(
                query_sim=hit.similarity,
                cached_sig=hit.entry.signature,
                cached_answer=hit.entry.answer,
                fresh_sig=fresh_sig,
                fresh_chunks=deduped,
            )
            if report.all_passed:
                return RouterResult(
                    question=question,
                    answer=hit.entry.answer,
                    signature=fresh_sig,
                    gen=None,
                    decision=RouterDecision(
                        path="answer_cache",
                        answer_cache_hit_sim=hit.similarity,
                        gate_report=report,
                        retrieval_hit_kind=cr.hit_kind,
                    ),
                    dedup=dstats,
                    compression=None,
                    retrieval_latency_s=retrieval_latency,
                    total_latency_s=time.perf_counter() - t_total,
                )
            # else fall through; the report is attached to the final decision.
            rejected_report = report
        else:
            rejected_report = None

        # 4. optional compression.
        comp_stats: CompressionStats | None = None
        ctx_chunks = deduped
        if self.enable_compression and self.compressor is not None and ctx_chunks:
            ctx_chunks, comp_stats = self.compressor.compress(question, deduped)

        # 5. generate.
        prompt = PROMPT_TEMPLATE.format(
            context=self._format_context(ctx_chunks), question=question
        )
        gen = self.generator.generate(prompt, max_tokens=max_tokens)

        # 6. insert into answer cache.
        self.answer_cache.insert(question, q_emb, gen.text, fresh_sig)

        return RouterResult(
            question=question,
            answer=gen.text,
            signature=fresh_sig,
            gen=gen,
            decision=RouterDecision(
                path="generate",
                answer_cache_hit_sim=hit.similarity if hit else None,
                gate_report=rejected_report,
                retrieval_hit_kind=cr.hit_kind,
                compression_ratio=comp_stats.ratio if comp_stats else None,
            ),
            dedup=dstats,
            compression=comp_stats,
            retrieval_latency_s=retrieval_latency,
            total_latency_s=time.perf_counter() - t_total,
        )
