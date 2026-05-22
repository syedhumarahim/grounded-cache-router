from ragcache.toy_corpus import DOCS, QA
from ragcache.workload import build_workload


def test_all_regimes_produce_events():
    for regime in ["exact_repeat", "paraphrase", "near_miss",
                   "long_shared_doc", "bounded_kb_cag"]:
        evs = build_workload(regime, QA, n=5, seed=0)
        assert len(evs) == 5
        assert all(e.regime == regime for e in evs)


def test_document_drift_primes_then_replays():
    evs = build_workload("document_drift", QA, n=6, seed=0, docs=list(DOCS))
    assert len(evs) == 6
    # First half: priming on base corpus (no snapshot).
    assert all(e.corpus_snapshot is None for e in evs[:3])
    # Second half: replay under one shared drifted snapshot with mutated text.
    drift_events = evs[3:]
    snap = drift_events[0].corpus_snapshot
    assert all(e.corpus_snapshot is snap for e in drift_events), \
        "drift events must share one snapshot"
    assert all(d.version.endswith(".drift") for d in snap)
    # Drift events re-issue priming queries (same qid stem).
    primed = {e.qid.split('#')[0] for e in evs[:3]}
    assert {e.qid.split('#')[0] for e in drift_events}.issubset(primed)
