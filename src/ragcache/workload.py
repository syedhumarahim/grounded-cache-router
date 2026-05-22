"""Synthetic workload generation for the six traffic regimes.

The regimes are the experimental backbone of the paper: they let us measure
when a cache hit is *correct* (exact_repeat, paraphrase), when a cached hit
would be *wrong* (near_miss, document_drift), and when retrieval itself may
be skippable (bounded_kb_cag). For stage 1 we only need the request stream;
the cache itself arrives in stages 2-4.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal

from .corpus import Document

Regime = Literal[
    "exact_repeat",
    "paraphrase",
    "near_miss",
    "document_drift",
    "long_shared_doc",
    "bounded_kb_cag",
]


@dataclass
class QueryEvent:
    qid: str
    text: str
    gold_answer: str | None = None
    gold_doc_ids: tuple[str, ...] = ()
    regime: Regime = "exact_repeat"
    # If non-None, the corpus should be replaced with this snapshot before serving.
    corpus_snapshot: tuple[Document, ...] | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class QAItem:
    qid: str
    question: str
    paraphrases: tuple[str, ...] = ()
    near_misses: tuple[str, ...] = ()      # similar wording, different evidence needed
    gold_answer: str | None = None
    gold_doc_ids: tuple[str, ...] = ()


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def gen_exact_repeat(items: list[QAItem], n: int, rng: random.Random) -> list[QueryEvent]:
    pool = [QueryEvent(qid=i.qid, text=i.question, gold_answer=i.gold_answer,
                       gold_doc_ids=i.gold_doc_ids, regime="exact_repeat") for i in items]
    out = []
    for k in range(n):
        e = rng.choice(pool)
        out.append(QueryEvent(qid=f"{e.qid}#r{k}", text=e.text, gold_answer=e.gold_answer,
                              gold_doc_ids=e.gold_doc_ids, regime="exact_repeat"))
    return out


def gen_paraphrase(items: list[QAItem], n: int, rng: random.Random) -> list[QueryEvent]:
    cands = [i for i in items if i.paraphrases]
    out = []
    for k in range(n):
        i = rng.choice(cands)
        text = rng.choice(i.paraphrases)
        out.append(QueryEvent(qid=f"{i.qid}#p{k}", text=text, gold_answer=i.gold_answer,
                              gold_doc_ids=i.gold_doc_ids, regime="paraphrase"))
    return out


def gen_near_miss(items: list[QAItem], n: int, rng: random.Random) -> list[QueryEvent]:
    """Lexically similar to a prior query but requires DIFFERENT evidence.

    Cache *must reject* these. The gold_doc_ids deliberately differ from the
    superficially-similar item.
    """
    cands = [i for i in items if i.near_misses]
    out = []
    for k in range(n):
        i = rng.choice(cands)
        text = rng.choice(i.near_misses)
        out.append(QueryEvent(qid=f"{i.qid}#nm{k}", text=text,
                              gold_answer=None,
                              gold_doc_ids=(),  # different evidence; corpus-specific gold lives elsewhere
                              regime="near_miss",
                              meta={"prior_qid": i.qid}))
    return out


_NUM_RE = re.compile(r"\b(\d[\d,]*)(\.\d+)?\b")


def _mutate_numbers(text: str, rng: random.Random) -> str:
    """Replace each number in `text` with a different number of similar shape.
    Deterministic given `rng`. Used by document_drift to make sure the
    answer-bearing tokens actually change between snapshots.
    """
    def repl(m):
        whole, frac = m.group(1), m.group(2) or ""
        digits = whole.replace(",", "")
        if len(digits) <= 1:
            new = str((int(digits) + rng.randint(1, 8)) % 10)
        else:
            # bump by 5-20%
            v = int(digits)
            delta = max(1, int(v * (0.05 + rng.random() * 0.15)))
            new_v = v + delta * rng.choice([-1, 1])
            new = f"{abs(new_v):,}" if "," in whole else str(abs(new_v))
        return new + frac
    return _NUM_RE.sub(repl, text)


def gen_document_drift(
    items: list[QAItem],
    docs: list[Document],
    n: int,
    rng: random.Random,
) -> list[QueryEvent]:
    """Half priming on the base corpus, half replay on a *single shared*
    drifted snapshot whose numeric tokens have been mutated. Sharing the
    snapshot keeps retriever rebuilds to one for the whole regime
    (otherwise every event would rebuild FAISS), and pairing each drift
    event with a prior priming event under the original corpus is what
    lets G3 (version match) actually fire.
    """
    elig = [x for x in items if x.gold_doc_ids]
    if not elig:
        return []

    drifted = []
    for d in docs:
        text = _mutate_numbers(d.text, rng)
        if text == d.text:
            text = d.text + " [drifted]"
        drifted.append(Document(doc_id=d.doc_id, text=text,
                                version=d.version + ".drift"))
    drifted_snap = tuple(drifted)

    half = max(1, n // 2)
    out: list[QueryEvent] = []
    # Phase 1: prime the answer cache on the base corpus.
    priming = [rng.choice(elig) for _ in range(half)]
    for k, i in enumerate(priming):
        out.append(QueryEvent(qid=f"{i.qid}#pr{k}", text=i.question,
                              gold_answer=i.gold_answer,
                              gold_doc_ids=i.gold_doc_ids,
                              regime="document_drift",
                              corpus_snapshot=None))
    # Phase 2: replay primed queries under the drifted snapshot.
    for k, i in enumerate(priming[: n - half]):
        out.append(QueryEvent(qid=f"{i.qid}#dd{k}", text=i.question,
                              gold_answer=i.gold_answer,
                              gold_doc_ids=i.gold_doc_ids,
                              regime="document_drift",
                              corpus_snapshot=drifted_snap,
                              meta={"prior_qid": i.qid, "phase": "drift"}))
    return out


def gen_long_shared_doc(items: list[QAItem], n: int, rng: random.Random,
                       doc_id: str | None = None) -> list[QueryEvent]:
    """Many questions hitting the same document — the prefix/dedup-favorable regime."""
    if doc_id is None:
        pool = [i for i in items if i.gold_doc_ids]
    else:
        pool = [i for i in items if doc_id in i.gold_doc_ids]
    if not pool:
        return []
    out = []
    for k in range(n):
        i = rng.choice(pool)
        out.append(QueryEvent(qid=f"{i.qid}#ls{k}", text=i.question,
                              gold_answer=i.gold_answer,
                              gold_doc_ids=i.gold_doc_ids,
                              regime="long_shared_doc"))
    return out


def gen_bounded_kb_cag(items: list[QAItem], n: int, rng: random.Random) -> list[QueryEvent]:
    """All material assumed to fit in context; just emit queries tagged for the CAG path."""
    out = []
    for k in range(n):
        i = rng.choice(items)
        out.append(QueryEvent(qid=f"{i.qid}#cag{k}", text=i.question,
                              gold_answer=i.gold_answer,
                              gold_doc_ids=i.gold_doc_ids,
                              regime="bounded_kb_cag"))
    return out


REGIME_FNS: dict[Regime, Callable] = {
    "exact_repeat": gen_exact_repeat,
    "paraphrase": gen_paraphrase,
    "near_miss": gen_near_miss,
    "long_shared_doc": gen_long_shared_doc,
    "bounded_kb_cag": gen_bounded_kb_cag,
}


def build_workload(
    regime: Regime,
    items: list[QAItem],
    n: int,
    seed: int = 0,
    docs: list[Document] | None = None,
) -> list[QueryEvent]:
    rng = _rng(seed)
    if regime == "document_drift":
        if docs is None:
            raise ValueError("document_drift regime requires `docs`")
        return gen_document_drift(items, docs, n, rng)
    return REGIME_FNS[regime](items, n, rng)
