"""Stage-4 compression fallback.

Practical baseline along the lines of Provence / PISCO / LongLLMLingua:
query-conditioned sentence selection within each retrieved chunk. We score
each sentence by cosine similarity between its embedding and the query
embedding, then keep the top sentences up to a token budget. Cheap, no extra
training; serves as the "compression baseline" in our ablations.

Future drop-ins (OSCAR, SeleCom, REFRAG) implement the same `compress()`
signature.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .corpus import Chunk

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class CompressionStats:
    sentences_in: int = 0
    sentences_out: int = 0
    chars_in: int = 0
    chars_out: int = 0

    @property
    def ratio(self) -> float:
        return (self.chars_out / self.chars_in) if self.chars_in else 1.0


@dataclass
class QueryConditionedCompressor:
    """Encoder is any callable[[list[str]], np.ndarray of shape (N, D)] with
    L2-normalized rows. We use the retriever's encoder in practice."""

    encode: Callable[[list[str]], np.ndarray]
    keep_ratio: float = 0.5
    min_sentences_per_chunk: int = 1
    char_budget: int | None = None   # if set, overrides keep_ratio globally

    def compress(self, query: str, chunks: list[Chunk]) -> tuple[list[Chunk], CompressionStats]:
        stats = CompressionStats()
        if not chunks:
            return [], stats

        # 1. split each chunk into sentences, track origin.
        sentences: list[str] = []
        origin: list[int] = []
        for i, c in enumerate(chunks):
            sents = [s.strip() for s in _SENT_SPLIT.split(c.text) if s.strip()]
            sentences.extend(sents)
            origin.extend([i] * len(sents))
            stats.sentences_in += len(sents)
            stats.chars_in += len(c.text)
        if not sentences:
            return chunks, stats

        # 2. score all sentences vs query in one batched encode call.
        q = self.encode([query])[0]
        S = self.encode(sentences)
        scores = S @ q          # cosine on normalized vectors

        # 3. global budget: either char budget or keep_ratio.
        order = np.argsort(-scores)
        if self.char_budget is not None:
            chosen: set[int] = set()
            running = 0
            for idx in order:
                running += len(sentences[idx])
                if running > self.char_budget:
                    break
                chosen.add(int(idx))
        else:
            keep_n = max(1, int(round(len(sentences) * self.keep_ratio)))
            chosen = set(int(i) for i in order[:keep_n])

        # 4. enforce min-per-chunk: always keep the highest-scored sentence
        # in each chunk to avoid emptying a chunk entirely.
        for ci in range(len(chunks)):
            in_chunk = [i for i, o in enumerate(origin) if o == ci]
            kept = [i for i in in_chunk if i in chosen]
            if len(kept) < self.min_sentences_per_chunk and in_chunk:
                in_chunk.sort(key=lambda i: -scores[i])
                for i in in_chunk[: self.min_sentences_per_chunk - len(kept)]:
                    chosen.add(i)

        # 5. rebuild chunks preserving original sentence order within each.
        out: list[Chunk] = []
        for ci, c in enumerate(chunks):
            keep_idx = sorted(i for i, o in enumerate(origin) if o == ci and i in chosen)
            if not keep_idx:
                continue
            new_text = " ".join(sentences[i] for i in keep_idx)
            stats.sentences_out += len(keep_idx)
            stats.chars_out += len(new_text)
            out.append(Chunk.make(c.doc_id, int(c.chunk_id.split("::")[-1]),
                                  new_text, version=c.version))
        return out, stats
