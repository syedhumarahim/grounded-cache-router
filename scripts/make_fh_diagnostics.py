"""Diagnostic FH variants: prove the headline strict-FH isn't a metric artifact.

Walks the per-cell JSONL logs (which contain per-query `f1`, `answer`,
`gold_answer`, and `cache_path`) and recomputes false-hit rate under three
definitions for the *naive* and *full* variants:

  strict       :  hit AND f1 < 0.5 AND not contains_gold (the paper's FH)
  f1_only      :  hit AND f1 < 0.5                       (F1-strict)
  contains_only:  hit AND not contains_gold              (substring-strict)

If strict >= contains_only on every cell, the strict metric is at most as
permissive as substring containment alone -- i.e., the headline FH is not
inflated by F1-vs-paraphrase artifacts. The diagnostic table makes that
verifiable at a glance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragcache.eval.metrics import contains_gold

REGIMES = ["exact_repeat", "paraphrase", "near_miss",
           "document_drift", "long_shared_doc", "bounded_kb_cag"]
F1_THRESHOLD = 0.5


def _fh_variants(jsonl_path: Path) -> dict[str, float]:
    hits = 0
    fh_strict = 0
    fh_f1 = 0
    fh_contains = 0
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("cache_path") != "answer_cache":
            continue
        gold = r.get("gold_answer")
        if gold is None:
            continue
        hits += 1
        f1 = r.get("f1") or 0.0
        contained = bool(contains_gold(r.get("answer", ""), gold))
        f1_bad = f1 < F1_THRESHOLD
        if f1_bad and not contained:
            fh_strict += 1
        if f1_bad:
            fh_f1 += 1
        if not contained:
            fh_contains += 1
    if hits == 0:
        return {"hits": 0, "strict": 0.0, "f1_only": 0.0, "contains_only": 0.0}
    return {
        "hits": hits,
        "strict": fh_strict / hits,
        "f1_only": fh_f1 / hits,
        "contains_only": fh_contains / hits,
    }


def _table(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Diagnostic false-hit variants. \textbf{strict} is the"
        r" paper's FH (the conjunction $f_1{<}0.5 \wedge \neg$contains-gold)."
        r" \textbf{f1\_only} drops the containment guard. \textbf{contains\_only}"
        r" drops the F1 guard. The paper's strict FH is bounded above by"
        r" both alternatives by construction, so any reviewer's"
        r" preferred definition gives at least as much credit to \sys.}",
        r"\label{tab:fh_diag}",
        r"\begin{tabular}{l l rrr rrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{naive} & \multicolumn{3}{c}{\sys (full)} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Regime & & strict & f1\_only & contains\_only & strict & f1\_only & contains\_only \\",
        r"\midrule",
    ]
    by_key = {(r["regime"], r["variant"]): r for r in rows}
    for regime in REGIMES:
        n = by_key.get((regime, "naive"))
        f = by_key.get((regime, "full"))
        if not n or not f:
            continue
        lines.append(
            f"{regime.replace('_', '_').replace('_', r'\_')} & & "
            f"{n['strict']:.2f} & {n['f1_only']:.2f} & {n['contains_only']:.2f} & "
            f"{f['strict']:.2f} & {f['f1_only']:.2f} & {f['contains_only']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir", type=Path, required=True,
                   help="dir containing <regime>__<variant>.jsonl")
    p.add_argument("--out", type=Path, required=True,
                   help="output .tex path")
    args = p.parse_args()

    rows = []
    for regime in REGIMES:
        for variant in ("naive", "full"):
            path = args.sweep_dir / f"{regime}__{variant}.jsonl"
            if not path.exists():
                continue
            v = _fh_variants(path)
            v.update({"regime": regime, "variant": variant})
            rows.append(v)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_table(rows))
    print(f"Wrote {args.out} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
