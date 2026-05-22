"""Stage 4: the evidence validator -- the paper's core contribution.

Given (cached query, cached answer, cached EvidenceSignature, fresh query,
fresh EvidenceSignature, fresh retrieved chunks), decide whether reusing the
cached answer is *safe*. Four gates:

  G1  query_sim      -- cached query is semantically close to fresh query
  G2  evidence_iou   -- cached evidence overlaps strongly with fresh evidence
  G3  version_match  -- shared chunks carry the same version tag
  G4  support_check  -- cached answer is still supported by fresh evidence

G4 has two backends:
  * lexical   : claim tokens are covered by the fresh-evidence token bag
                (cheap, deterministic, no LLM calls; the default).
  * judge     : a Generator is asked yes/no whether the fresh evidence
                supports the cached answer. Optional; costs one extra call.

A query that fails any enabled gate is sent to the generation path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

from ..corpus import Chunk
from ..generator import Generator
from ..signature import EvidenceSignature

SupportMode = Literal["off", "lexical", "judge"]


@dataclass
class GateReport:
    query_sim_ok: bool
    evidence_iou_ok: bool
    versions_ok: bool
    support_ok: bool
    query_sim: float
    evidence_iou: float
    support_score: float | None       # None when support gate is off
    reason_rejected: str | None       # e.g. "evidence_iou=0.12<0.6"

    @property
    def all_passed(self) -> bool:
        return (
            self.query_sim_ok
            and self.evidence_iou_ok
            and self.versions_ok
            and self.support_ok
        )


@dataclass
class ValidatorConfig:
    query_sim_threshold: float = 0.93
    evidence_iou_threshold: float = 0.6
    require_version_match: bool = True
    support_mode: SupportMode = "lexical"
    lexical_support_threshold: float = 0.6
    judge_max_tokens: int = 8


_STOP = {
    "the", "a", "an", "of", "in", "on", "and", "or", "to", "is", "are",
    "was", "were", "be", "been", "being", "for", "with", "by", "at", "as",
    "this", "that", "these", "those", "it", "its", "from", "into", "than",
    "but", "not", "no", "do", "does", "did", "have", "has", "had", "i",
    "you", "we", "they", "he", "she", "him", "her", "them", "us", "me",
    "my", "your", "their", "his", "our",
}


def _content_tokens(s: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", s.lower())
    return {t for t in toks if len(t) >= 3 and t not in _STOP}


def lexical_support(cached_answer: str, fresh_chunks: list[Chunk]) -> float:
    """Fraction of content tokens in the cached answer that also appear in
    the fresh retrieved evidence. Cheap proxy for entailment."""
    a = _content_tokens(cached_answer)
    if not a:
        return 1.0
    e: set[str] = set()
    for c in fresh_chunks:
        e |= _content_tokens(c.text)
    return len(a & e) / len(a)


_JUDGE_PROMPT = """Given the EVIDENCE below, is the ANSWER fully supported by the evidence?
Reply with exactly one word: YES or NO.

EVIDENCE:
{evidence}

ANSWER:
{answer}

Reply:"""


def judge_support(
    cached_answer: str,
    fresh_chunks: list[Chunk],
    generator: Generator,
    max_tokens: int = 8,
) -> float:
    """Returns 1.0 if judge answers YES, else 0.0. Costs one LLM call."""
    evidence = "\n\n".join(c.text for c in fresh_chunks)
    out = generator.generate(
        _JUDGE_PROMPT.format(evidence=evidence, answer=cached_answer),
        max_tokens=max_tokens, temperature=0.0,
    )
    return 1.0 if out.text.strip().lower().startswith("yes") else 0.0


class EvidenceValidator:
    def __init__(self, cfg: ValidatorConfig | None = None,
                 judge_generator: Generator | None = None):
        self.cfg = cfg or ValidatorConfig()
        self.judge_generator = judge_generator
        if self.cfg.support_mode == "judge" and judge_generator is None:
            raise ValueError("support_mode='judge' requires judge_generator")

    def validate(
        self,
        query_sim: float,
        cached_sig: EvidenceSignature,
        cached_answer: str,
        fresh_sig: EvidenceSignature,
        fresh_chunks: list[Chunk],
    ) -> GateReport:
        cfg = self.cfg
        g1 = query_sim >= cfg.query_sim_threshold
        iou = cached_sig.jaccard(fresh_sig)
        g2 = iou >= cfg.evidence_iou_threshold
        g3 = (not cfg.require_version_match) or cached_sig.versions_match(fresh_sig)

        support_score: float | None
        if cfg.support_mode == "off":
            support_score, g4 = None, True
        elif cfg.support_mode == "lexical":
            support_score = lexical_support(cached_answer, fresh_chunks)
            g4 = support_score >= cfg.lexical_support_threshold
        else:  # judge
            support_score = judge_support(
                cached_answer, fresh_chunks, self.judge_generator,
                max_tokens=cfg.judge_max_tokens,
            )
            g4 = support_score >= 0.5

        reason = None
        if not g1:
            reason = f"query_sim={query_sim:.2f}<{cfg.query_sim_threshold}"
        elif not g2:
            reason = f"evidence_iou={iou:.2f}<{cfg.evidence_iou_threshold}"
        elif not g3:
            reason = "version_mismatch"
        elif not g4:
            reason = f"support={support_score:.2f}<{cfg.lexical_support_threshold}"

        return GateReport(
            query_sim_ok=g1, evidence_iou_ok=g2, versions_ok=g3, support_ok=g4,
            query_sim=query_sim, evidence_iou=iou, support_score=support_score,
            reason_rejected=reason,
        )
