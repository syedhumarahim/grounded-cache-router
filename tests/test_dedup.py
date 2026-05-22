from ragcache.corpus import Chunk, Document, chunk_document, dedup_chunks


def test_chunking_is_deterministic():
    d = Document(doc_id="d1", text="Sentence one. Sentence two. Sentence three. " * 10)
    a = chunk_document(d, target_chars=80, overlap=10)
    b = chunk_document(d, target_chars=80, overlap=10)
    assert [c.chunk_hash for c in a] == [c.chunk_hash for c in b]
    assert all(c.doc_id == "d1" for c in a)


def test_dedup_drops_byte_exact():
    c1 = Chunk.make("d", 0, "alpha beta")
    c2 = Chunk.make("d", 1, "alpha beta")   # same text -> same hash
    c3 = Chunk.make("d", 2, "gamma")
    kept, stats = dedup_chunks([c1, c2, c3])
    assert len(kept) == 2
    assert stats.duplicates_dropped == 1
    assert stats.kept == 2
