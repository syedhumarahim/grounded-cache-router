"""Embedding retriever over FAISS (CPU). Mac-friendly today, swappable later."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .corpus import Chunk


@dataclass
class Retrieved:
    chunks: list[Chunk]
    scores: list[float]


class EmbeddingRetriever:
    """sentence-transformers + FAISS inner-product (cosine on normalized vectors).

    The model and faiss imports are deferred so `import ragcache` stays cheap
    and tests that don't need retrieval don't pay the cost.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # noqa: WPS433

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._index = None
        self._chunks: list[Chunk] = []

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        emb = self.model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return emb.astype("float32")

    def build(self, chunks: Sequence[Chunk]) -> None:
        import faiss  # noqa: WPS433

        self._chunks = list(chunks)
        emb = self._encode([c.text for c in self._chunks])
        index = faiss.IndexFlatIP(self.dim)
        index.add(emb)
        self._index = index

    def search(self, query: str, k: int = 5) -> Retrieved:
        if self._index is None:
            raise RuntimeError("Retriever index not built. Call build() first.")
        q = self._encode([query])
        scores, idx = self._index.search(q, k)
        chunks = [self._chunks[i] for i in idx[0] if i >= 0]
        return Retrieved(chunks=chunks, scores=[float(s) for s in scores[0][: len(chunks)]])
