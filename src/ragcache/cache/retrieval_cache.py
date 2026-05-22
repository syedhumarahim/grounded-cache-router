"""Stage 2: retrieval-result cache.

Two cache modes coexist:

  * exact   : key = normalized query string. O(1) lookup, zero false-hit risk
              for a given embedding model.
  * approx  : key = quantized query embedding. Reuses the retrieved chunk set
              when a new query is cosine-close (>= threshold) to a recent one.

Both modes return the *retrieved chunks* (and their scores) — the generator
still runs. The semantic answer cache (stage 3) and evidence-validated router
(stage 4) sit one layer above this.

On hit we still emit an EvidenceSignature constructed from the cached chunks,
so downstream stages see a uniform interface.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..corpus import Chunk
from ..retriever import EmbeddingRetriever, Retrieved


def _norm_query(q: str) -> str:
    return " ".join(q.lower().strip().split())


@dataclass
class CacheStats:
    lookups: int = 0
    exact_hits: int = 0
    approx_hits: int = 0
    misses: int = 0
    total_lookup_s: float = 0.0


class RetrievalCache:
    def __init__(
        self,
        max_entries: int = 1024,
        approx: bool = True,
        approx_threshold: float = 0.92,
    ):
        self.max_entries = max_entries
        self.approx = approx
        self.threshold = approx_threshold
        # exact map: norm_query -> (chunks, scores)
        self._exact: "OrderedDict[str, tuple[list[Chunk], list[float]]]" = OrderedDict()
        # approx index: list of (embedding, chunks, scores). Linear scan; fine
        # for paper-scale workloads, swap for FAISS later if needed.
        self._approx: list[tuple[np.ndarray, list[Chunk], list[float]]] = []
        self.stats = CacheStats()

    def clear(self) -> None:
        """Drop all cached entries; keep cumulative stats."""
        self._exact.clear()
        self._approx.clear()

    def _evict(self) -> None:
        while len(self._exact) > self.max_entries:
            self._exact.popitem(last=False)
        while len(self._approx) > self.max_entries:
            self._approx.pop(0)

    def lookup(
        self, query: str, embed_fn: Optional[Callable[[str], np.ndarray]] = None
    ) -> tuple[Optional[Retrieved], str]:
        """Returns (Retrieved | None, hit_kind in {exact, approx, miss})."""
        t0 = time.perf_counter()
        self.stats.lookups += 1
        nq = _norm_query(query)
        if nq in self._exact:
            chunks, scores = self._exact[nq]
            self._exact.move_to_end(nq)
            self.stats.exact_hits += 1
            self.stats.total_lookup_s += time.perf_counter() - t0
            return Retrieved(chunks=list(chunks), scores=list(scores)), "exact"

        if self.approx and embed_fn is not None and self._approx:
            qv = embed_fn(query)
            # cosine on already-normalized vectors == dot product
            mat = np.stack([e for e, _, _ in self._approx])
            sims = mat @ qv
            best = int(np.argmax(sims))
            if float(sims[best]) >= self.threshold:
                _, chunks, scores = self._approx[best]
                self.stats.approx_hits += 1
                self.stats.total_lookup_s += time.perf_counter() - t0
                return Retrieved(chunks=list(chunks), scores=list(scores)), "approx"

        self.stats.misses += 1
        self.stats.total_lookup_s += time.perf_counter() - t0
        return None, "miss"

    def insert(
        self,
        query: str,
        retrieved: Retrieved,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    ) -> None:
        nq = _norm_query(query)
        self._exact[nq] = (list(retrieved.chunks), list(retrieved.scores))
        if self.approx and embed_fn is not None:
            self._approx.append((embed_fn(query), list(retrieved.chunks), list(retrieved.scores)))
        self._evict()


# ---------- pipeline wrapper ----------

@dataclass
class CachedRetrievalResult:
    retrieved: Retrieved
    hit_kind: str            # exact | approx | miss
    lookup_latency_s: float
    retrieval_latency_s: float  # 0.0 on hit


class CachedRetriever:
    """Wraps an EmbeddingRetriever with a RetrievalCache."""

    def __init__(self, retriever: EmbeddingRetriever, cache: RetrievalCache):
        self.retriever = retriever
        self.cache = cache

    def build(self, chunks) -> None:
        """Rebuild the underlying index and drop the retrieval cache.

        Cache entries are tied to a corpus snapshot; on a snapshot swap (e.g.
        the document_drift regime) the cached chunk objects no longer match
        the new index, so we invalidate to avoid stale-hit contamination.
        """
        self.retriever.build(chunks)
        self.cache.clear()

    def _embed(self, q: str) -> np.ndarray:
        return self.retriever._encode([q])[0]

    def search(self, query: str, k: int = 5) -> CachedRetrievalResult:
        t0 = time.perf_counter()
        hit, kind = self.cache.lookup(query, embed_fn=self._embed if self.cache.approx else None)
        lookup_lat = time.perf_counter() - t0
        if hit is not None:
            # Trim or pad to requested k (cache may have stored a different k).
            return CachedRetrievalResult(
                retrieved=Retrieved(chunks=hit.chunks[:k], scores=hit.scores[:k]),
                hit_kind=kind,
                lookup_latency_s=lookup_lat,
                retrieval_latency_s=0.0,
            )
        t1 = time.perf_counter()
        retrieved = self.retriever.search(query, k=k)
        ret_lat = time.perf_counter() - t1
        self.cache.insert(query, retrieved, embed_fn=self._embed if self.cache.approx else None)
        return CachedRetrievalResult(
            retrieved=retrieved,
            hit_kind="miss",
            lookup_latency_s=lookup_lat,
            retrieval_latency_s=ret_lat,
        )
