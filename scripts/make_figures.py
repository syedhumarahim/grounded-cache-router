"""Generate paper/figures/*.pdf from runs/sweeps/summary.json.

Three figures:
  fig_hr_fh.pdf     grouped bar: hit-rate vs false-hit by regime, naive vs full
  fig_path_mix.pdf  stacked bar: % of queries served from generate / retrieval_cache / answer_cache
  fig_safety.pdf    scatter: x=hit-rate, y=false-hit-rate, one point per variant per regime
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REGIME_ORDER = ["exact_repeat", "paraphrase", "near_miss",
                "document_drift", "long_shared_doc", "bounded_kb_cag"]
VARIANT_ORDER = ["naive", "no-version", "no-evidence", "no-support", "full"]


def _load(path: Path) -> dict[tuple[str, str], dict]:
    rows = json.loads(path.read_text())
    return {(r["regime"], r["variant"]): r for r in rows}


def fig_hr_fh(d, out: Path) -> None:
    regimes = REGIME_ORDER
    x = np.arange(len(regimes))
    w = 0.18
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    for i, (variant, color) in enumerate([("naive", "#bbbbbb"), ("full", "#2a6df4")]):
        ahr = [d.get((r, variant), {}).get("answer_hit_rate", 0) or 0 for r in regimes]
        usr = [d.get((r, variant), {}).get("unsafe_served_rate", 0) or 0 for r in regimes]
        ax.bar(x + (i - 0.5) * w * 2, ahr, w * 2, label=f"{variant} aHR", color=color, alpha=0.7)
        ax.bar(x + (i - 0.5) * w * 2, usr, w * 2, label=f"{variant} USR",
               color=color, hatch="///", edgecolor="black", linewidth=0.5,
               alpha=0.95, bottom=0)
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", "\n") for r in regimes], fontsize=9)
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", ncol=2, fontsize=8, frameon=False)
    ax.set_title("Answer-cache hit rate (solid) and unsafe-served rate (hatched) per regime")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_path_mix(d, out: Path) -> None:
    regimes = REGIME_ORDER
    paths = ["answer_cache", "retrieval_cache", "generate"]
    colors = {"answer_cache": "#2a6df4", "retrieval_cache": "#8fbcff", "generate": "#dddddd"}
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    x = np.arange(len(regimes))
    bottoms = np.zeros(len(regimes))
    for p in paths:
        vals = []
        for r in regimes:
            row = d.get((r, "full"), {})
            mix = row.get("path_mix", {}) or {}
            n = sum(mix.values()) or 1
            vals.append(mix.get(p, 0) / n)
        vals = np.array(vals)
        ax.bar(x, vals, bottom=bottoms, color=colors[p], label=p, edgecolor="white")
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", "\n") for r in regimes], fontsize=9)
    ax.set_ylabel("share of queries")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", ncol=3, fontsize=8, frameon=False)
    ax.set_title("Path mix under GroundedCache (full)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_safety(d, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    markers = {"naive": "o", "no-version": "^", "no-evidence": "s",
               "no-support": "D", "full": "*"}
    for variant in VARIANT_ORDER:
        xs, ys, labels = [], [], []
        for regime in REGIME_ORDER:
            row = d.get((regime, variant), {})
            xs.append(row.get("answer_hit_rate", 0) or 0)
            ys.append(row.get("unsafe_served_rate", 0) or 0)
            labels.append(regime)
        ax.scatter(xs, ys, marker=markers[variant], s=80 if variant == "full" else 40,
                   label=variant, alpha=0.85)
    ax.set_xlabel("answer-cache hit rate (higher = more reuse)")
    ax.set_ylabel("unsafe-served rate (lower = safer)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title("Safety/speedup Pareto: per regime, per variant")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    d = _load(args.inp)
    fig_hr_fh(d, args.out_dir / "fig_hr_fh.pdf")
    fig_path_mix(d, args.out_dir / "fig_path_mix.pdf")
    fig_safety(d, args.out_dir / "fig_safety.pdf")
    print(f"Wrote 3 figures to {args.out_dir}")


if __name__ == "__main__":
    main()
