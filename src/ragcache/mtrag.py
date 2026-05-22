"""mtRAG (IBM Multi-Turn RAG Benchmark) adapter.

mtRAG is the canonical multi-turn corpus for cache-safety analysis: same
surface question can refer to different referents across turns. We treat
each (user turn, next agent turn) as a QAItem and use the same-conversation
turns as a pool for the near-miss regime (since same-conversation later
turns are lexically related but reference different evidence).

Setup: clone IBM/mt-rag-benchmark and pass `--mtrag-path` (or set
$MTRAG_PATH) to its root. The repo is ~150 MB.

   git clone https://github.com/IBM/mt-rag-benchmark.git ~/data/mt-rag-benchmark
   export MTRAG_PATH=~/data/mt-rag-benchmark
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from .corpus import Document
from .workload import QAItem


def _slug(s: str, n: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:n]
    return s or "doc"


def _resolve_path(path: Optional[str | Path]) -> Path:
    p = Path(path) if path else Path(os.environ.get("MTRAG_PATH", ""))
    if not p or not p.exists():
        raise FileNotFoundError(
            "mtRAG benchmark not found. Clone and point MTRAG_PATH at it:\n"
            "  git clone https://github.com/IBM/mt-rag-benchmark.git\n"
            "  export MTRAG_PATH=$(pwd)/mt-rag-benchmark"
        )
    return p


def load_mtrag(
    path: Optional[str | Path] = None,
    split: str = "human",
    max_conversations: int = 30,
    seed: int = 0,
) -> tuple[list[Document], list[QAItem]]:
    """Load a slice of mtRAG.

    Args:
      split: "human" (110 high-quality dialogues) or "synthetic" (larger).
      max_conversations: cap on number of conversations loaded.

    Returns (docs, items). Documents are collected from all contexts across
    selected conversations; items are one per user turn that has an agent
    response with contexts. Near-misses are assigned within-conversation:
    later turns that share content tokens but reference different docs.
    """
    root = _resolve_path(path)
    conv_path = root / f"mtrag-{split}" / "conversations" / "conversations.json"
    if not conv_path.exists():
        raise FileNotFoundError(f"missing {conv_path}")

    with conv_path.open() as f:
        conversations = json.load(f)

    import random
    rng = random.Random(seed)
    rng.shuffle(conversations)
    conversations = conversations[:max_conversations]

    docs: dict[str, Document] = {}
    items_by_conv: list[list[QAItem]] = []

    for ci, conv in enumerate(conversations):
        msgs = conv.get("messages", [])
        conv_items: list[QAItem] = []
        # Walk in user/agent pairs.
        for j in range(len(msgs) - 1):
            user = msgs[j]
            agent = msgs[j + 1]
            if user.get("speaker") != "user" or agent.get("speaker") != "agent":
                continue
            contexts = agent.get("contexts") or []
            if not contexts:
                continue
            gold_ids = []
            for c in contexts:
                did = _slug(c.get("document_id") or c.get("title") or "")
                if not did:
                    continue
                if did not in docs:
                    docs[did] = Document(doc_id=did,
                                         text=c.get("text", "").strip(),
                                         version="v1")
                gold_ids.append(did)
            gold_ids = list(dict.fromkeys(gold_ids))  # de-dup, keep order
            question = user.get("text", "").strip()
            answer = (agent.get("text") or agent.get("original_text") or "").strip()
            if not question or not answer or not gold_ids:
                continue
            qid = f"mt_c{ci}_t{j}"
            conv_items.append(QAItem(
                qid=qid,
                question=question,
                paraphrases=_cheap_paraphrases(question),
                near_misses=(),  # filled below
                gold_answer=answer,
                gold_doc_ids=tuple(gold_ids),
            ))
        items_by_conv.append(conv_items)

    items = _assign_within_conv_near_misses(items_by_conv)
    return list(docs.values()), items


def _cheap_paraphrases(q: str) -> tuple[str, ...]:
    q = q.strip().rstrip("?")
    return (
        f"Can you tell me: {q}?",
        f"I'd like to know, {q.lower()}?",
        f"{q} -- what's the answer?",
    )


def _content_tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) >= 3}


def _assign_within_conv_near_misses(items_by_conv: list[list[QAItem]]) -> list[QAItem]:
    """For each item, find the highest-token-overlap *other* item in the
    same conversation whose gold_doc_ids are disjoint. This captures the
    multi-turn referent-shift failure mode: same surface tokens, different
    grounding."""
    out: list[QAItem] = []
    for conv_items in items_by_conv:
        tok = [_content_tokens(i.question) for i in conv_items]
        for i, item in enumerate(conv_items):
            best_j, best_s = -1, -1.0
            for j, other in enumerate(conv_items):
                if i == j:
                    continue
                if set(item.gold_doc_ids) & set(other.gold_doc_ids):
                    continue
                inter = len(tok[i] & tok[j])
                if inter == 0:
                    continue
                union = len(tok[i] | tok[j]) or 1
                s = inter / union
                if s > best_s:
                    best_s, best_j = s, j
            nm = (conv_items[best_j].question,) if best_j >= 0 and best_s > 0.1 else ()
            out.append(QAItem(
                qid=item.qid,
                question=item.question,
                paraphrases=item.paraphrases,
                near_misses=nm,
                gold_answer=item.gold_answer,
                gold_doc_ids=item.gold_doc_ids,
            ))
    return out
