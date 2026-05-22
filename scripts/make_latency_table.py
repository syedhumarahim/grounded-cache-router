"""Latency table that compares against the right baseline: no-cache RAG.

The headline "full is 1.9x slower than naive" is misleading -- naive is
artificially fast because it serves cached answers indiscriminately
(including the unsafe ones). The honest baseline is "no cache at all, every
query goes through retrieve+generate." We synthesize that baseline from
existing per-cell JSONL data without spending any GPU time: every variant
contains records whose cache_path is exactly 'generate', and those records
are unbiased samples of true no-cache end-to-end latency.

Produces paper/tables_<ds>/latency_vs_nocache.tex.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

REGIMES = ["exact_repeat", "paraphrase", "near_miss",
           "document_drift", "long_shared_doc", "bounded_kb_cag"]
VARIANTS = ["naive", "no-support", "full"]


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return xs[k]


def _generate_latency_p50(sweep_dir: Path) -> float:
    """p50 of total_latency_s across every record (any cell, any variant)
    whose cache_path == 'generate'. This is the empirical no-cache baseline:
    the latency of a query that goes through retrieve + generate end-to-end.
    """
    lats: list[float] = []
    for f in sweep_dir.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("cache_path") == "generate":
                lats.append(float(r["total_latency_s"]))
    return _pct(lats, 0.50), len(lats)


def _variant_p50(rows: list[dict], variant: str) -> float:
    """Mean across regimes of latency_p50 for the given variant."""
    xs = [r["latency_p50"] for r in rows if r["variant"] == variant]
    return mean(xs) if xs else 0.0


def _table(rows: list[dict], no_cache_p50: float, no_cache_n: int, ds: str = "") -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        rf"\caption{{End-to-end latency vs.\ the honest baseline{ds}. The"
        r" \emph{no-cache} row is the empirical p50 latency of all"
        r" \texttt{generate}-path records observed across every sweep cell"
        rf" ($n={no_cache_n}$ records); it is the latency of a RAG query"
        r" that hits no cache at all. The other rows are the per-variant"
        r" p50 latency averaged across the six regimes. Speedup is computed"
        r" against the no-cache baseline. Naive caching is fastest but,"
        r" per Table~\ref{tab:main}, unsafe; \sys{} retains a meaningful"
        r" speedup while delivering USR$\to$0.}",
        r"\label{tab:latency_vs_nocache}",
        r"\begin{tabular}{l rrr}",
        r"\toprule",
        r"Variant & Latency$_{p50}$ (s) & Speedup vs.\ no-cache & USR \\",
        r"\midrule",
    ]
    # USR row average for context
    usr_by_v = {v: mean(r["unsafe_served_rate"] for r in rows if r["variant"] == v)
                for v in VARIANTS}
    lines.append(f"no-cache (always generate) & {no_cache_p50:.3f} & 1.00$\\times$ & 0.00 \\\\")
    for v in VARIANTS:
        lat = _variant_p50(rows, v)
        speedup = no_cache_p50 / lat if lat > 0 else 0.0
        lines.append(
            f"\\texttt{{{v}}} & {lat:.3f} & {speedup:.2f}$\\times$ & {usr_by_v[v]:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dataset", default="")
    args = p.parse_args()
    ds = f" ({args.dataset})" if args.dataset else ""

    rows = json.loads(args.summary.read_text())
    no_cache_p50, n_gen = _generate_latency_p50(args.sweep_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_table(rows, no_cache_p50, n_gen, ds))
    print(f"Wrote {args.out}  (no-cache p50 = {no_cache_p50:.3f}s over {n_gen} generate records)")


if __name__ == "__main__":
    main()
