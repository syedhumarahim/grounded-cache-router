# Reproducing the paper's results

A single `make all` reproduces every number and figure in the paper, using a
deterministic extractive backend so no API key or GPU is required.

## Quickstart

```bash
make install         # one-time: venv + pip install
make test            # 19 unit tests should pass
make all             # sweep -> tables -> figures -> paper/main.pdf
```

`make all` invokes:

1. `scripts/run_sweeps.py` -- runs 6 regimes x 5 router variants on HotpotQA
   (n=120 by default) and writes JSONL per cell plus
   `runs/sweeps_hotpot/summary.json`.
2. `scripts/collate_results.py` -- summary -> `paper/tables/main_table.tex`
   and `paper/tables/ablation.tex`.
3. `scripts/results_text.py` -- summary -> `paper/tables/results_text.tex`
   (the auto-generated headline-numbers paragraph in the Results section).
4. `scripts/make_figures.py` -- summary ->
   `paper/figures/{fig_hr_fh,fig_path_mix,fig_safety}.pdf`.
5. `latexmk` -- compile `paper/main.pdf`.

## Knobs

| Variable | Default | Effect |
|---|---|---|
| `DATASET` | `hotpotqa` | `toy` for the smoke control |
| `N` | `120` | number of queries per (regime, variant) cell |
| `TOPK` | `3` | top-k retrieval |
| `GEN` | `extractive` | `anthropic` or `vllm` for real-LLM runs |

Example: real-LLM sweep on a single GPU running vLLM:

```bash
export RAGCACHE_GENERATOR=vllm
export VLLM_BASE_URL=https://your-runpod-host
export VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
make sweep tables figures paper N=200
```

## What each variant is

| Variant | Gates active |
|---|---|
| `naive` | semantic similarity only |
| `no-version` | G1 + G2 + G4 (skip version match) |
| `no-evidence` | G1 + G3 + G4 (skip evidence-IoU) |
| `no-support` | G1 + G2 + G3 (skip support) |
| `full` | **all four (the paper's proposed system)** |
