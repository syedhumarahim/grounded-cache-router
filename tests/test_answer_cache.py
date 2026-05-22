import time
import numpy as np

from ragcache.cache.answer_cache import SemanticAnswerCache
from ragcache.corpus import Chunk
from ragcache.signature import EvidenceSignature


def _sig():
    return EvidenceSignature.from_chunks([Chunk.make("d", 0, "x")], [1.0])


def _n(v):
    v = np.asarray(v, dtype="float32"); return v / np.linalg.norm(v)


def test_hit_above_threshold():
    c = SemanticAnswerCache(threshold=0.9)
    c.insert("q1", _n([1, 0]), "ans1", _sig())
    hit = c.lookup(_n([0.99, 0.14]))
    assert hit is not None and hit.entry.answer == "ans1"


def test_miss_below_threshold():
    c = SemanticAnswerCache(threshold=0.99)
    c.insert("q1", _n([1, 0]), "ans1", _sig())
    assert c.lookup(_n([0.6, 0.8])) is None


def test_ttl_expiry():
    c = SemanticAnswerCache(threshold=0.5, default_ttl_s=0.01)
    c.insert("q", _n([1, 0]), "ans", _sig())
    time.sleep(0.02)
    assert c.lookup(_n([1, 0])) is None
    assert c.stats.expired >= 1
