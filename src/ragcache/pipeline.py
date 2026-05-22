"""End-to-end RAG: retrieve -> dedup -> order -> prompt -> generate.

The retriever is duck-typed: anything with a `search(query, k)` method that
returns either a `Retrieved` or a `CachedRetrievalResult` works. This lets the
same pipeline drive vanilla retrieval (stage 1) and cached retrieval (stage 2)
without branching at the call site.

Chunk ordering for prefix-cache friendliness (vLLM APC / SGLang RadixAttention):
context blocks are sorted by (doc_id, chunk_id) so repeated queries against
the same documents produce identical prompt prefixes — the precondition for
APC to actually reuse KV blocks.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .corpus import Chunk, dedup_chunks, DedupStats
from .generator import Generator, GenResult
from .retriever import Retrieved
from .signature import EvidenceSignature

PROMPT_TEMPLATE = """You are answering using ONLY the provided context. If the context is insufficient, say "I don't know."

# Context
{context}

# Question
{question}

# Answer
"""


@dataclass
class RAGResult:
    question: str
    answer: str
    retrieved: Retrieved
    dedup: DedupStats
    signature: EvidenceSignature
    gen: GenResult
    retrieval_latency_s: float
    retrieval_cache_hit: str | None   # exact | approx | miss | None (uncached)
    total_latency_s: float


def _stable_order(chunks: list[Chunk], scores: list[float]) -> tuple[list[Chunk], list[float]]:
    """Sort by (doc_id, chunk_id) so identical evidence sets yield identical prompts."""
    paired = sorted(zip(chunks, scores), key=lambda cs: (cs[0].doc_id, cs[0].chunk_id))
    if not paired:
        return [], []
    cs, ss = zip(*paired)
    return list(cs), list(ss)


class RAGPipeline:
    def __init__(
        self,
        retriever: Any,
        generator: Generator,
        top_k: int = 5,
        prefix_friendly_order: bool = True,
    ):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.prefix_friendly_order = prefix_friendly_order

    def _format_context(self, chunks: list[Chunk]) -> str:
        return "\n\n".join(f"[{c.chunk_id} v{c.version}]\n{c.text}" for c in chunks)

    def _retrieve(self, question: str) -> tuple[Retrieved, float, str | None]:
        t = time.perf_counter()
        out = self.retriever.search(question, k=self.top_k)
        # CachedRetrievalResult duck-type
        if hasattr(out, "retrieved") and hasattr(out, "hit_kind"):
            return out.retrieved, out.retrieval_latency_s + out.lookup_latency_s, out.hit_kind
        return out, time.perf_counter() - t, None

    def run(self, question: str, max_tokens: int = 256) -> RAGResult:
        t_total = time.perf_counter()
        retrieved, retrieval_latency, hit_kind = self._retrieve(question)

        deduped, dstats = dedup_chunks(retrieved.chunks)
        keep_ids = {c.chunk_id for c in deduped}
        kept_scores = [s for c, s in zip(retrieved.chunks, retrieved.scores) if c.chunk_id in keep_ids]
        if self.prefix_friendly_order:
            deduped, kept_scores = _stable_order(deduped, kept_scores)
        sig = EvidenceSignature.from_chunks(deduped, kept_scores)

        prompt = PROMPT_TEMPLATE.format(
            context=self._format_context(deduped), question=question
        )
        gen = self.generator.generate(prompt, max_tokens=max_tokens)
        return RAGResult(
            question=question,
            answer=gen.text,
            retrieved=retrieved,
            dedup=dstats,
            signature=sig,
            gen=gen,
            retrieval_latency_s=retrieval_latency,
            retrieval_cache_hit=hit_kind,
            total_latency_s=time.perf_counter() - t_total,
        )
