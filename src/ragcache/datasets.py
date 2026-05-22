"""Real-dataset adapters. Stage 1.5 wires HotpotQA; later stages can add
RAGBench, mtRAG, RAGTruth alongside without changing the harness.

HotpotQA is convenient because each example ships its own small set of context
paragraphs (the "distractor" setting), so we get a per-example mini-corpus
without setting up a wikipedia index. For multi-hop sensitivity later we just
re-use the full corpus across all questions.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

from .corpus import Document
from .workload import QAItem


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60]


def _stable_id(prefix: str, *parts: str) -> str:
    h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}"


def _paraphrases_for(q: str) -> tuple[str, ...]:
    """Cheap deterministic paraphrases. Good enough as a workload signal for
    semantic-cache hit-rate analysis; replace with an LLM-generated set if we
    want stronger paraphrase coverage."""
    q = q.strip().rstrip("?")
    cands = [
        f"Can you tell me: {q}?",
        f"{q} -- what is the answer?",
        f"I'd like to know, {q.lower()}?",
    ]
    return tuple(dict.fromkeys(cands))  # de-dup preserve-order


def load_hotpotqa(
    split: str = "validation",
    n: int | None = 100,
    name: str = "distractor",
    seed: int = 0,
) -> tuple[list[Document], list[QAItem]]:
    """Returns (docs, qa_items). Each HotpotQA example contributes its context
    paragraphs as Documents and one QAItem with a near-miss drawn from another
    example whose question is lexically similar but evidence is disjoint.
    """
    try:
        from datasets import load_dataset  # noqa: WPS433
    except ImportError as e:
        raise RuntimeError(
            "load_hotpotqa needs `datasets`. Install with: pip install datasets"
        ) from e

    ds = load_dataset("hotpot_qa", name, split=split, trust_remote_code=True)
    if n is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    docs: dict[str, Document] = {}
    items: list[QAItem] = []

    for ex in ds:
        # Build documents from the context titles+sentences.
        titles = ex["context"]["title"]
        sent_lists = ex["context"]["sentences"]
        per_ex_doc_ids: list[str] = []
        for title, sents in zip(titles, sent_lists):
            doc_id = _slug(title) or _stable_id("doc", title)
            text = " ".join(s.strip() for s in sents if s.strip())
            if not text:
                continue
            # Keep the first occurrence; HotpotQA contexts overlap heavily.
            docs.setdefault(doc_id, Document(doc_id=doc_id, text=text))
            per_ex_doc_ids.append(doc_id)

        # Gold supporting docs.
        sup_titles = ex["supporting_facts"]["title"]
        gold_ids = tuple(dict.fromkeys(_slug(t) for t in sup_titles))

        qid = _stable_id("q", ex["_id"] if "_id" in ex else ex["question"])
        items.append(
            QAItem(
                qid=qid,
                question=ex["question"].strip(),
                paraphrases=_paraphrases_for(ex["question"]),
                near_misses=(),  # filled below
                gold_answer=ex["answer"].strip() if ex.get("answer") else None,
                gold_doc_ids=gold_ids,
            )
        )

    # Near-miss assignment: for each item, find the lexically-closest other
    # question whose gold_doc_ids are disjoint. Uses a cheap token-overlap
    # metric; expensive enough at n=100 to leave as O(n^2).
    items = _assign_near_misses(items)
    return list(docs.values()), items


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 2}


def _assign_near_misses(items: list[QAItem]) -> list[QAItem]:
    out: list[QAItem] = []
    tok = [_tokens(i.question) for i in items]
    for i, item in enumerate(items):
        best_j, best_score = -1, -1.0
        for j, other in enumerate(items):
            if i == j:
                continue
            if set(item.gold_doc_ids) & set(other.gold_doc_ids):
                continue  # overlapping evidence -> not a true near-miss
            inter = len(tok[i] & tok[j])
            union = len(tok[i] | tok[j]) or 1
            score = inter / union
            if score > best_score:
                best_score, best_j = score, j
        nm = (items[best_j].question,) if best_j >= 0 and best_score > 0.15 else ()
        out.append(
            QAItem(
                qid=item.qid,
                question=item.question,
                paraphrases=item.paraphrases,
                near_misses=nm,
                gold_answer=item.gold_answer,
                gold_doc_ids=item.gold_doc_ids,
            )
        )
    return out
