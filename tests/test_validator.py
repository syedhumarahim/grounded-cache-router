from ragcache.cache.validator import EvidenceValidator, ValidatorConfig, lexical_support
from ragcache.corpus import Chunk
from ragcache.signature import EvidenceSignature


def _sig(chunks):
    return EvidenceSignature.from_chunks(chunks, [1.0] * len(chunks))


def test_lexical_support_full_overlap():
    chunks = [Chunk.make("d", 0, "The capital of France is Paris.")]
    assert lexical_support("Paris is the capital", chunks) == 1.0


def test_lexical_support_no_overlap():
    chunks = [Chunk.make("d", 0, "The Moon is Earth's satellite.")]
    assert lexical_support("Photosynthesis converts sunlight to glucose.", chunks) < 0.2


def test_query_sim_gate_rejects():
    v = EvidenceValidator(ValidatorConfig(query_sim_threshold=0.95))
    cs = [Chunk.make("d", 0, "alpha beta")]
    rep = v.validate(query_sim=0.80, cached_sig=_sig(cs), cached_answer="alpha",
                     fresh_sig=_sig(cs), fresh_chunks=cs)
    assert not rep.all_passed and "query_sim" in rep.reason_rejected


def test_evidence_iou_gate_rejects_near_miss():
    cached = [Chunk.make("d1", 0, "alpha")]
    fresh = [Chunk.make("d2", 0, "beta")]
    v = EvidenceValidator(ValidatorConfig(evidence_iou_threshold=0.5, support_mode="off"))
    rep = v.validate(query_sim=0.99, cached_sig=_sig(cached), cached_answer="alpha",
                     fresh_sig=_sig(fresh), fresh_chunks=fresh)
    assert not rep.all_passed and "evidence_iou" in rep.reason_rejected


def test_version_gate_rejects_drift():
    cached = [Chunk.make("d1", 0, "alpha beta", version="v1")]
    fresh = [Chunk.make("d1", 0, "alpha beta", version="v2")]
    v = EvidenceValidator(ValidatorConfig(support_mode="off"))
    rep = v.validate(query_sim=0.99, cached_sig=_sig(cached), cached_answer="alpha",
                     fresh_sig=_sig(fresh), fresh_chunks=fresh)
    assert not rep.all_passed and rep.reason_rejected == "version_mismatch"


def test_all_pass():
    cs = [Chunk.make("d", 0, "Paris is the capital of France.")]
    v = EvidenceValidator(ValidatorConfig(evidence_iou_threshold=0.5,
                                          lexical_support_threshold=0.5))
    rep = v.validate(query_sim=0.99, cached_sig=_sig(cs), cached_answer="Paris capital France",
                     fresh_sig=_sig(cs), fresh_chunks=cs)
    assert rep.all_passed
