"""Stage 3: semantic answer cache (the "naive" baseline our paper compares against).

This is intentionally the strawman: it stores answers keyed on the query
embedding and returns a cached answer whenever a new query is cosine-close to
a prior one, with no checks on the evidence the cached answer was based on.

The literature has shown this fails in three predictable ways, and these are
the failures our stage-4 validator targets:

  1. near-miss queries (similar wording, different evidence required)
  2. document drift (same query, evidence has changed)
  3. multi-turn context inversion (same surface query, different referent)

Stage 4 (`router.py`) reuses the same storage but adds gates before returning.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..signature import EvidenceSignature


@dataclass
class AnswerCacheEntry:
    query: str
    query_emb: np.ndarray            # L2-normalized
    answer: str
    signature: EvidenceSignature
    created_at: float = field(default_factory=time.time)
    ttl_s: float | None = None       # optional freshness window


@dataclass
class AnswerCacheStats:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    expired: int = 0


@dataclass
class AnswerCacheHit:
    entry: AnswerCacheEntry
    similarity: float


class SemanticAnswerCache:
    def __init__(self, threshold: float = 0.93, max_entries: int = 4096,
                 default_ttl_s: float | None = None):
        self.threshold = threshold
        self.max_entries = max_entries
        self.default_ttl_s = default_ttl_s
        self._entries: list[AnswerCacheEntry] = []
        self.stats = AnswerCacheStats()

    def clear(self) -> None:
        self._entries.clear()

    def insert(
        self,
        query: str,
        query_emb: np.ndarray,
        answer: str,
        signature: EvidenceSignature,
        ttl_s: float | None = None,
    ) -> None:
        self._entries.append(AnswerCacheEntry(
            query=query, query_emb=query_emb, answer=answer,
            signature=signature, ttl_s=ttl_s if ttl_s is not None else self.default_ttl_s,
        ))
        # FIFO eviction; an LRU policy is a swap.
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)

    def _expired(self, e: AnswerCacheEntry, now: float) -> bool:
        return e.ttl_s is not None and (now - e.created_at) > e.ttl_s

    def lookup(self, query_emb: np.ndarray) -> Optional[AnswerCacheHit]:
        """Return the most-similar non-expired entry above threshold, else None."""
        self.stats.lookups += 1
        if not self._entries:
            self.stats.misses += 1
            return None
        now = time.time()
        live = [e for e in self._entries if not self._expired(e, now)]
        self.stats.expired += len(self._entries) - len(live)
        # purge expired in place
        if len(live) != len(self._entries):
            self._entries = live
        if not live:
            self.stats.misses += 1
            return None
        mat = np.stack([e.query_emb for e in live])
        sims = mat @ query_emb
        best = int(np.argmax(sims))
        sim = float(sims[best])
        if sim >= self.threshold:
            self.stats.hits += 1
            return AnswerCacheHit(entry=live[best], similarity=sim)
        self.stats.misses += 1
        return None
