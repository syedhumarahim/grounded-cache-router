"""Corpus + chunking + byte-exact deduplication.

A `Chunk` is the unit of retrieval and the unit of cache evidence. Each chunk
carries a stable byte-level hash (used for dedup and for the EvidenceSignature
in stage 4) and an optional version tag (used by the freshness gate).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str          # f"{doc_id}::{ordinal}"
    text: str
    chunk_hash: str        # sha1 of text (normalized)
    version: str = "v1"    # bumped when the source doc is edited

    @staticmethod
    def make(doc_id: str, ordinal: int, text: str, version: str = "v1") -> "Chunk":
        norm = text.strip()
        return Chunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}::{ordinal}",
            text=norm,
            chunk_hash=_sha1(norm),
            version=version,
        )


@dataclass
class Document:
    doc_id: str
    text: str
    version: str = "v1"


def chunk_document(doc: Document, target_chars: int = 600, overlap: int = 80) -> list[Chunk]:
    """Greedy sentence-aware character chunker. Deterministic for a given input."""
    sents = re.split(r"(?<=[.!?])\s+", doc.text.strip())
    chunks: list[Chunk] = []
    buf: list[str] = []
    cur = 0
    ordinal = 0
    for s in sents:
        if not s:
            continue
        if cur + len(s) > target_chars and buf:
            text = " ".join(buf).strip()
            chunks.append(Chunk.make(doc.doc_id, ordinal, text, doc.version))
            ordinal += 1
            # overlap: keep the tail of the previous chunk
            if overlap > 0 and len(text) > overlap:
                tail = text[-overlap:]
                buf = [tail, s]
                cur = len(tail) + len(s)
            else:
                buf = [s]
                cur = len(s)
        else:
            buf.append(s)
            cur += len(s) + 1
    if buf:
        text = " ".join(buf).strip()
        if text:
            chunks.append(Chunk.make(doc.doc_id, ordinal, text, doc.version))
    return chunks


def chunk_corpus(docs: Iterable[Document], **kw) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, **kw))
    return out


# ---------- deterministic deduplication ----------

@dataclass
class DedupStats:
    seen: int = 0
    kept: int = 0
    duplicates_dropped: int = 0
    bytes_saved: int = 0


def dedup_chunks(chunks: Iterable[Chunk]) -> tuple[list[Chunk], DedupStats]:
    """Byte-exact dedup over chunk_hash. Preserves first-seen order."""
    stats = DedupStats()
    seen: set[str] = set()
    out: list[Chunk] = []
    for c in chunks:
        stats.seen += 1
        if c.chunk_hash in seen:
            stats.duplicates_dropped += 1
            stats.bytes_saved += len(c.text)
            continue
        seen.add(c.chunk_hash)
        out.append(c)
        stats.kept += 1
    return out, stats


def dedup_text_blocks(blocks: Iterable[str]) -> tuple[list[str], DedupStats]:
    """Dedup arbitrary prompt-scaffold strings by sha1. Used for system/user templates."""
    stats = DedupStats()
    seen: set[str] = set()
    out: list[str] = []
    for b in blocks:
        stats.seen += 1
        h = _sha1(b.strip())
        if h in seen:
            stats.duplicates_dropped += 1
            stats.bytes_saved += len(b)
            continue
        seen.add(h)
        out.append(b)
        stats.kept += 1
    return out, stats
