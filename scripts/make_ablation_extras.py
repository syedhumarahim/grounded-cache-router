"""Generate three additional ablation artifacts from a sweep summary.json:

  1. marginal_fh.tex      -- per-gate marginal FH reduction (no-X vs full)
  2. latency_variant.tex  -- TTFT/latency/tokens by variant (avg over regimes)
  3. fig_gate_stack.pdf   -- cumulative FH-reduction step plot per regime

Usage:
  python scripts/make_ablation_extras.py \
    --in runs/sweeps_vllm_hotpotqa/summary.json \
    --out-tables paper/tables_hotpotqa \
    --out-figs   paper/figures_hotpotqa
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REGIMES = ["exact_repeat", "paraphrase", "near_miss",
           "document_drift", "long_shared_doc", "bounded_kb_cag"]
VARIANTS = ["naive", "no-version", "no-evidence", "no-support", "full"]
PRETTY_REGIME = {r: r.replace("_", r"\_") for r in REGIMES}


def _index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(r["regime"], r["variant"]): r for r in rows}


def _marginal_table(rows: list[dict], ds: str = "") -> str:
    """Per-gate marginal unsafe-served-rate reduction =
    USR(no-X) - USR(full), averaged across regimes. USR = false_hits / N
    is the operator-facing metric ('fraction of all queries that got a
    wrong cached answer'). A positive value means the gate is doing real
    work; ~0 means the gate is redundant with the rest of the stack."""
    idx = _index(rows)
    gates = [("Version (G1)", "no-version"),
             ("Evidence (G2)", "no-evidence"),
             ("Support (G3)", "no-support")]
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        rf"\caption{{Per-gate marginal unsafe-served-rate reduction{ds}:"
        r" $\Delta\mathrm{USR}=$ USR(no-X) $-$ USR(full), averaged over the"
        r" six workload regimes. USR is the fraction of all queries that"
        r" received a wrong cached answer. Larger $\Delta$ means the gate"
        r" is more load-bearing; values near zero mean the gate is"
        r" redundant with the other two on this workload.}",
        r"\label{tab:marginal_fh}",
        r"\begin{tabular}{l rr}",
        r"\toprule",
        r"Removed gate & USR(no-X) & $\Delta\mathrm{USR}$ vs.\ full \\",
        r"\midrule",
    ]
    usr_full = mean(idx[(r, "full")]["unsafe_served_rate"] for r in REGIMES)
    for label, variant in gates:
        usr_var = mean(idx[(r, variant)]["unsafe_served_rate"] for r in REGIMES)
        delta = usr_var - usr_full
        lines.append(f"{label} & {usr_var:.3f} & {delta:+.3f} \\\\")
    lines += [r"\midrule",
              f"None (full system) & {usr_full:.3f} & n/a \\\\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _latency_table(rows: list[dict], ds: str = "") -> str:
    """Per-variant TTFT_p50, latency_p50, tokens_in (avg across regimes)."""
    idx = _index(rows)
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        rf"\caption{{Per-variant cost averaged over the six regimes{ds}. TTFT and"
        r" latency are p50 seconds; tokens are mean prompt tokens per query."
        r" The full system retains most of the answer-cache speedup while"
        r" closing the false-hit gap.}",
        r"\label{tab:latency_variant}",
        r"\begin{tabular}{l rrr}",
        r"\toprule",
        r"Variant & TTFT$_{p50}$ (s) & Latency$_{p50}$ (s) & Prompt tok. \\",
        r"\midrule",
    ]
    for v in VARIANTS:
        ttft = mean(idx[(r, v)]["ttft_p50"] for r in REGIMES)
        lat = mean(idx[(r, v)]["latency_p50"] for r in REGIMES)
        tok = mean(idx[(r, v)]["tokens_in"] / max(1, idx[(r, v)]["n"])
                   for r in REGIMES)
        lines.append(f"\\texttt{{{v}}} & {ttft:.3f} & {lat:.3f} & {tok:,.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _gate_stack_fig(rows: list[dict], out_path: Path) -> None:
    """Cumulative FH-reduction step plot: naive -> +version -> +evidence
    -> +support -> full, per regime."""
    idx = _index(rows)
    # Cumulative ordering: we read FH at variants that progressively add gates.
    # naive (no gates) -> no-evidence,no-support (only version) -> ... -> full.
    # We approximate the "add one gate at a time" curve with the available
    # ablation cells: naive, no-evidence&no-support (~= version-only is not
    # directly recorded; we use no-support to mean version+evidence, and full
    # for all three).
    stages = [("naive", "naive"),
              ("no-support", "+ G1,G2"),
              ("full", "+ G3 (full)")]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = list(range(len(stages)))
    for regime in REGIMES:
        ys = [idx[(regime, v)]["unsafe_served_rate"] for v, _ in stages]
        ax.plot(x, ys, marker="o", linewidth=1.6, label=regime.replace("_", " "))
    ax.set_xticks(x, [lbl for _, lbl in stages])
    ax.set_ylabel("Unsafe-served rate (fh / N)")
    ax.set_ylim(-0.02, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.set_title("Cumulative gate-stack USR reduction")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out-tables", type=Path, required=True)
    p.add_argument("--out-figs", type=Path, required=True)
    p.add_argument("--dataset", default="")
    args = p.parse_args()
    ds = f" ({args.dataset})" if args.dataset else ""

    rows = json.loads(args.inp.read_text())
    args.out_tables.mkdir(parents=True, exist_ok=True)
    args.out_figs.mkdir(parents=True, exist_ok=True)

    (args.out_tables / "marginal_fh.tex").write_text(_marginal_table(rows, ds))
    (args.out_tables / "latency_variant.tex").write_text(_latency_table(rows, ds))
    _gate_stack_fig(rows, args.out_figs / "fig_gate_stack.pdf")
    print(f"Wrote marginal_fh.tex, latency_variant.tex, fig_gate_stack.pdf "
          f"under {args.out_tables} / {args.out_figs}")


if __name__ == "__main__":
    main()
