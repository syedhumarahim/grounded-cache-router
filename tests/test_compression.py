import numpy as np

from ragcache.compression import QueryConditionedCompressor
from ragcache.corpus import Chunk


def _det_encoder(texts):
    """Cheap deterministic encoder: hash characters into a 16-d vector."""
    out = np.zeros((len(texts), 16), dtype="float32")
    for i, t in enumerate(texts):
        for ch in t.lower():
            out[i, ord(ch) % 16] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def test_compression_keeps_min_one_per_chunk_and_shrinks():
    chunks = [
        Chunk.make("d", 0, "The Moon orbits Earth. Mars has two moons. Sun is a star."),
        Chunk.make("d", 1, "Olympus Mons is on Mars. Phobos and Deimos are moons of Mars."),
    ]
    c = QueryConditionedCompressor(encode=_det_encoder, keep_ratio=0.4)
    out, stats = c.compress("Tell me about Mars and its moons", chunks)
    assert len(out) == 2
    assert stats.sentences_out >= 2 and stats.sentences_out < stats.sentences_in
    assert stats.ratio < 1.0
