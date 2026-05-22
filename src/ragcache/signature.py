"""EvidenceSignature: the object the stage-4 validator compares against.

Intentionally lightweight so it serializes cheaply into a cache key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .corpus import Chunk


@dataclass(frozen=True)
class EvidenceSignature:
    doc_ids: tuple[str, ...]
    chunk_ids: tuple[str, ...]
    chunk_hashes: tuple[str, ...]
    versions: tuple[str, ...]
    scores: tuple[float, ...]

    @staticmethod
    def from_chunks(chunks: Iterable[Chunk], scores: Iterable[float]) -> "EvidenceSignature":
        chunks = list(chunks)
        scores = list(scores)
        return EvidenceSignature(
            doc_ids=tuple(c.doc_id for c in chunks),
            chunk_ids=tuple(c.chunk_id for c in chunks),
            chunk_hashes=tuple(c.chunk_hash for c in chunks),
            versions=tuple(c.version for c in chunks),
            scores=tuple(float(s) for s in scores),
        )

    def jaccard(self, other: "EvidenceSignature") -> float:
        a, b = set(self.chunk_hashes), set(other.chunk_hashes)
        if not a and not b:
            return 1.0
        return len(a & b) / max(1, len(a | b))

    def versions_match(self, other: "EvidenceSignature") -> bool:
        """All shared chunk_ids must carry the same version string."""
        amap = dict(zip(self.chunk_ids, self.versions))
        bmap = dict(zip(other.chunk_ids, other.versions))
        shared = set(amap) & set(bmap)
        return all(amap[k] == bmap[k] for k in shared)
