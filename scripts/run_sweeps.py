"""Run the paper's full sweep: every (regime x variant) combination.

Produces one JSONL per cell under runs/sweeps/ plus a single
runs/sweeps/summary.json that collate_results.py turns into LaTeX tables.

Usage:
  python scripts/run_sweeps.py --dataset toy --n 30
  python scripts/run_sweeps.py --dataset hotpotqa --n 100 --generator echo
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REGIMES = [
    "exact_repeat", "paraphrase", "near_miss",
    "document_drift", "long_shared_doc", "bounded_kb_cag",
]
VARIANTS = ["naive", "no-version", "no-evidence", "no-support", "full"]


def _aggregate_from_jsonl(path: Path) -> dict:
    """Recompute a sweep-summary row from a per-cell JSONL log.
    Mirrors the fields run_router.py prints on stdout.
    """
    from ragcache.eval.metrics import QueryRecord, aggregate  # local import
    recs: list[QueryRecord] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        # Drop unknown keys defensively.
        allowed = {f for f in QueryRecord.__dataclass_fields__}
        recs.append(QueryRecord(**{k: v for k, v in d.items() if k in allowed}))
    agg = aggregate(recs)
    return {
        "n": agg.n, "em_mean": agg.em_mean, "f1_mean": agg.f1_mean,
        "ttft_p50": agg.ttft_p50, "latency_p50": agg.latency_p50,
        "latency_p95": agg.latency_p95,
        "tokens_in": agg.tokens_in_total, "tokens_out": agg.tokens_out_total,
        "hit_rate": agg.hit_rate,
        "answer_hit_rate": agg.answer_hit_rate,
        "false_hit_rate": agg.false_hit_rate,
        "unsafe_served_rate": agg.unsafe_served_rate,
        "stale_hit_rate": agg.stale_hit_rate, "unsupported_rate": agg.unsupported_rate,
        "path_mix": agg.path_mix, "out": str(path),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="toy")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--keep-ratio", type=float, default=0.6)
    p.add_argument("--generator", default=None)
    p.add_argument("--out-dir", type=Path, default=Path("runs/sweeps"))
    p.add_argument("--regimes", nargs="+", default=REGIMES)
    p.add_argument("--variants", nargs="+", default=VARIANTS)
    p.add_argument("--resume", action="store_true",
                   help="skip cells whose JSONL already exists (re-aggregates summary)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    for regime in args.regimes:
        for variant in args.variants:
            out_path = args.out_dir / f"{regime}__{variant}.jsonl"
            if args.resume and out_path.exists() and out_path.stat().st_size > 0:
                # Recompute the summary row from the existing JSONL so the
                # final summary.json stays complete even when we skip cells.
                print(f"\n>>> {regime} / {variant}  [skipped, resume]", flush=True)
                row = _aggregate_from_jsonl(out_path)
                row["regime"] = regime
                row["variant"] = variant
                summary.append(row)
                continue
            cmd = [
                sys.executable, "scripts/run_router.py",
                "--regime", regime, "--variant", variant,
                "--dataset", args.dataset, "--n", str(args.n),
                "--top-k", str(args.top_k), "--keep-ratio", str(args.keep_ratio),
                "--out", str(out_path),
            ]
            if args.generator:
                cmd += ["--generator", args.generator]
            print(f"\n>>> {regime} / {variant}", flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(proc.stderr, file=sys.stderr)
                raise SystemExit(proc.returncode)
            # Pull the JSON summary printed by run_router.
            # It is the last JSON object in stdout.
            try:
                last = proc.stdout.rstrip().rsplit("\n{", 1)
                payload = "{" + last[-1] if len(last) > 1 else proc.stdout
                row = json.loads(payload.strip())
            except Exception:
                row = {"raw_stdout": proc.stdout}
            row["regime"] = regime
            row["variant"] = variant
            summary.append(row)

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {len(summary)} rows to {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
