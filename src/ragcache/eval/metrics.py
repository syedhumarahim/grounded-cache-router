"""Per-query and aggregate metrics.

Cache-specific counters (hit / false-hit / stale-hit / unsupported) are kept
here even though stages 2-4 are what populate them — having a single struct
means the harness format never changes.
"""
from __future__ import annotations

import re
import statistics
import string
from collections import Counter
from dataclasses import asdict, dataclass, field


# ---------- text normalization (SQuAD-style EM/F1) ----------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s: str) -> str:
    s = s.lower()
    s = s.translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def contains_gold(pred: str, gold: str) -> float:
    """1.0 if normalized gold is a substring of normalized pred. Robust to
    verbose extractive answers (e.g. returning a whole sentence containing
    the short factoid)."""
    p, g = normalize(pred), normalize(gold)
    return float(bool(g) and g in p)


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec = same / len(p)
    rec = same / len(g)
    return 2 * prec * rec / (prec + rec)


# ---------- per-query record ----------

@dataclass
class QueryRecord:
    qid: str
    regime: str
    question: str
    answer: str
    gold_answer: str | None
    em: float | None
    f1: float | None
    ttft_s: float
    retrieval_latency_s: float
    total_latency_s: float
    prompt_tokens: int
    completion_tokens: int
    backend: str
    # cache counters (stages 2-4 will set these)
    cache_path: str = "generate"        # generate | retrieval_cache | answer_cache | dedup_only
    cache_hit: bool = False
    cache_false_hit: bool = False       # hit returned, gold disagreed
    cache_stale_hit: bool = False       # hit returned, evidence versions drifted
    cache_unsupported: bool = False     # hit returned, answer not entailed by current evidence
    dedup_dropped: int = 0
    dedup_bytes_saved: int = 0

    def to_json(self) -> dict:
        return asdict(self)


# ---------- aggregation ----------

@dataclass
class Aggregate:
    n: int
    em_mean: float | None
    f1_mean: float | None
    ttft_p50: float
    ttft_p95: float
    latency_p50: float
    latency_p95: float
    tokens_in_total: int
    tokens_out_total: int
    hit_rate: float                  # any cache_hit (answer or retrieval)  / N
    answer_hit_rate: float           # answer_cache hits                    / N
    false_hit_rate: float            # false hits / answer_cache hits
    unsafe_served_rate: float        # false hits / N  (operator-facing)
    stale_hit_rate: float
    unsupported_rate: float
    path_mix: dict[str, int] = field(default_factory=dict)


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return xs[k]


def aggregate(records: list[QueryRecord]) -> Aggregate:
    if not records:
        return Aggregate(0, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, {})
    em_vals = [r.em for r in records if r.em is not None]
    f1_vals = [r.f1 for r in records if r.f1 is not None]
    ttfts = [r.ttft_s for r in records]
    lats = [r.total_latency_s for r in records]
    hits = sum(r.cache_hit for r in records)
    # Only answer_cache paths can ever produce a false/stale/unsupported hit;
    # retrieval_cache merely skips vector search and still regenerates. We
    # therefore normalize the safety counters by answer-cache hits, and also
    # report `unsafe_served_rate = false_hits / N` as the operator-facing
    # "fraction of all queries that got a wrong cached answer".
    answer_hits = sum(1 for r in records if r.cache_path == "answer_cache")
    fh_count = sum(r.cache_false_hit for r in records)
    return Aggregate(
        n=len(records),
        em_mean=statistics.mean(em_vals) if em_vals else None,
        f1_mean=statistics.mean(f1_vals) if f1_vals else None,
        ttft_p50=_pct(ttfts, 0.50),
        ttft_p95=_pct(ttfts, 0.95),
        latency_p50=_pct(lats, 0.50),
        latency_p95=_pct(lats, 0.95),
        tokens_in_total=sum(r.prompt_tokens for r in records),
        tokens_out_total=sum(r.completion_tokens for r in records),
        hit_rate=hits / len(records),
        answer_hit_rate=answer_hits / len(records),
        false_hit_rate=fh_count / max(1, answer_hits),
        unsafe_served_rate=fh_count / len(records),
        stale_hit_rate=sum(r.cache_stale_hit for r in records) / max(1, answer_hits),
        unsupported_rate=sum(r.cache_unsupported for r in records) / max(1, answer_hits),
        path_mix=dict(Counter(r.cache_path for r in records)),
    )
