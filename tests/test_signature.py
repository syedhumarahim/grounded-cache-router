from ragcache.corpus import Chunk
from ragcache.signature import EvidenceSignature


def test_jaccard_and_versions():
    a = [Chunk.make("d1", 0, "x"), Chunk.make("d1", 1, "y")]
    b = [Chunk.make("d1", 0, "x"), Chunk.make("d2", 0, "z")]
    sa = EvidenceSignature.from_chunks(a, [0.9, 0.8])
    sb = EvidenceSignature.from_chunks(b, [0.9, 0.7])
    # overlap on text "x" hash; jaccard = 1/3
    j = sa.jaccard(sb)
    assert 0.3 < j < 0.4
    assert sa.versions_match(sb) is True


def test_versions_mismatch_detected():
    a = [Chunk.make("d1", 0, "x", version="v1")]
    b = [Chunk.make("d1", 0, "x", version="v2")]
    sa = EvidenceSignature.from_chunks(a, [1.0])
    sb = EvidenceSignature.from_chunks(b, [1.0])
    assert sa.versions_match(sb) is False
