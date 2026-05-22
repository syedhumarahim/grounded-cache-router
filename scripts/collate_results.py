"""Turn runs/sweeps/summary.json into paper/tables/*.tex.

Produces:
  - main_table.tex     hit_rate, false_hit, stale_hit, latency_p50, tokens_in
                       for variant=full across all regimes vs naive.
  - ablation.tex       per-gate ablation on regimes where each gate matters.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _fmt(v):
    if v is None:
        return "--"
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"
    return str(v)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dataset", default="",
                   help="dataset label for table captions (e.g. 'HotpotQA')")
    args = p.parse_args()
    ds = f" ({args.dataset})" if args.dataset else ""

    rows = json.loads(args.inp.read_text())
    by = defaultdict(dict)
    for r in rows:
        by[r["regime"]][r["variant"]] = r
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- main_table.tex ----
    # Primary metric is unsafe_served_rate = false_hits / N (operator-facing:
    # "fraction of all queries that got a wrong cached answer"). We also
    # report answer_hit_rate (the fraction of queries served from the answer
    # cache) and conditional false-hit-rate FH = false_hits / answer_hits.
    out = []
    out.append(r"\begin{table}[t]")
    out.append(r"\centering\small")
    out.append(rf"\caption{{Naive semantic answer cache vs.\ \sys (full evidence "
               rf"validation) across regimes{ds}. \textbf{{aHR}} is the answer-cache "
               r"hit rate (fraction of queries served from cache); \textbf{USR} "
               r"is the unsafe-served-rate (fraction of \emph{all} queries that "
               r"received a wrong cached answer, our primary safety metric); "
               r"\textbf{FH}$=$USR/aHR is the conditional false-hit rate among "
               r"served cached answers.}")
    out.append(r"\label{tab:main}")
    out.append(r"\begin{tabular}{l rrr rrr}")
    out.append(r"\toprule")
    out.append(r" & \multicolumn{3}{c}{Naive} & \multicolumn{3}{c}{\sys (full)} \\")
    out.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
    out.append(r"Regime & aHR & USR & FH & aHR & USR & FH \\")
    out.append(r"\midrule")
    for regime, variants in by.items():
        n, f = variants.get("naive", {}), variants.get("full", {})
        out.append(
            f"{regime.replace('_', r'\_')} & "
            f"{_fmt(n.get('answer_hit_rate'))} & {_fmt(n.get('unsafe_served_rate'))} & {_fmt(n.get('false_hit_rate'))} & "
            f"{_fmt(f.get('answer_hit_rate'))} & {_fmt(f.get('unsafe_served_rate'))} & {_fmt(f.get('false_hit_rate'))} \\\\"
        )
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    (args.out / "main_table.tex").write_text("\n".join(out) + "\n")

    # ---- ablation.tex ----
    abl_variants = ["naive", "no-version", "no-evidence", "no-support", "full"]
    out = []
    out.append(r"\begin{table}[t]")
    out.append(r"\centering\small")
    out.append(rf"\caption{{Per-gate ablation: hit-rate vs.\ false-hit-rate across regimes{ds}.}}")
    out.append(r"\label{tab:ablation}")
    out.append(r"\begin{tabular}{l " + "rr " * len(abl_variants) + r"}")
    out.append(r"\toprule")
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{v}}}" for v in abl_variants)
    out.append(f" Regime & {head} \\\\")
    out.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}")
    out.append(" & " + " & ".join(["HR & FH"] * len(abl_variants)) + r" \\")
    out.append(r"\midrule")
    for regime, variants in by.items():
        cells = []
        for v in abl_variants:
            r = variants.get(v, {})
            cells.append(_fmt(r.get("hit_rate")))
            cells.append(_fmt(r.get("false_hit_rate")))
        out.append(regime.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    (args.out / "ablation.tex").write_text("\n".join(out) + "\n")

    print(f"Wrote {args.out / 'main_table.tex'} and {args.out / 'ablation.tex'}")


if __name__ == "__main__":
    main()
