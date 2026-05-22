import numpy as np

from ragcache.cache.retrieval_cache import RetrievalCache
from ragcache.corpus import Chunk
from ragcache.retriever import Retrieved


def _make_retrieved():
    chunks = [Chunk.make("d1", 0, "alpha"), Chunk.make("d2", 0, "beta")]
    return Retrieved(chunks=chunks, scores=[0.9, 0.8])


def test_exact_hit_and_miss():
    cache = RetrievalCache(approx=False)
    hit, kind = cache.lookup("what is alpha?")
    assert hit is None and kind == "miss"
    cache.insert("what is alpha?", _make_retrieved())
    hit, kind = cache.lookup("WHAT  is  alpha?")  # normalization
    assert hit is not None and kind == "exact"
    assert [c.chunk_id for c in hit.chunks] == ["d1::0", "d2::0"]


def test_approx_hit():
    cache = RetrievalCache(approx=True, approx_threshold=0.9)
    v_inserted = np.array([1.0, 0.0, 0.0], dtype="float32")
    cache.insert("alpha question", _make_retrieved(), embed_fn=lambda _q: v_inserted)
    # near-duplicate embedding -> cosine ~0.995
    near = np.array([0.995, 0.0998, 0.0], dtype="float32")
    near = near / np.linalg.norm(near)
    hit, kind = cache.lookup("approximately alpha", embed_fn=lambda _q: near)
    assert kind == "approx" and hit is not None


def test_approx_miss_below_threshold():
    cache = RetrievalCache(approx=True, approx_threshold=0.99)
    v = np.array([1.0, 0.0], dtype="float32")
    cache.insert("q1", _make_retrieved(), embed_fn=lambda _q: v)
    far = np.array([0.5, 0.866], dtype="float32")  # cos ~0.5
    hit, kind = cache.lookup("q-far", embed_fn=lambda _q: far)
    assert kind == "miss" and hit is None
