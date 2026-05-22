#!/usr/bin/env bash
# Drives both the HotpotQA and mtRAG sweeps against whatever generator
# is active (extractive | vllm | anthropic), then regenerates the paper
# tables and figures from each sweep into paper/tables_<dataset>/ etc.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

N="${N:-200}"
TOPK="${TOPK:-3}"
GEN="${RAGCACHE_GENERATOR:-${GEN:-extractive}}"
export RAGCACHE_GENERATOR="$GEN"
# Datasets that need extra config:
export MTRAG_PATH="${MTRAG_PATH:-/tmp/mt-rag-benchmark}"
if [ ! -d "$MTRAG_PATH" ]; then
  echo "mtRAG not found at $MTRAG_PATH -- cloning..."
  git clone --depth 1 https://github.com/IBM/mt-rag-benchmark.git "$MTRAG_PATH"
fi

for ds in hotpotqa mtrag; do
  out="runs/sweeps_${GEN}_${ds}"
  echo ">>> Sweep: dataset=$ds  gen=$GEN  N=$N  out=$out"
  python scripts/run_sweeps.py --dataset "$ds" --n "$N" --top-k "$TOPK" --out-dir "$out" --resume
  case "$ds" in
    hotpotqa) prose_name=HotpotQA ;;
    mtrag)    prose_name=mtRAG ;;
    *)        prose_name="$ds" ;;
  esac
  python scripts/collate_results.py --in "$out/summary.json" --out "paper/tables_${ds}" --dataset "$prose_name"
  python scripts/results_text.py    --in "$out/summary.json" --out "paper/tables_${ds}/results_text.tex" --dataset "$prose_name"
  python scripts/make_figures.py    --in "$out/summary.json" --out-dir "paper/figures_${ds}"
  python scripts/make_ablation_extras.py \
      --in "$out/summary.json" \
      --out-tables "paper/tables_${ds}" \
      --out-figs   "paper/figures_${ds}" \
      --dataset "$prose_name"
  python scripts/make_latency_table.py \
      --sweep-dir "$out" \
      --summary   "$out/summary.json" \
      --out       "paper/tables_${ds}/latency_vs_nocache.tex" \
      --dataset "$prose_name"
done

echo "Done. Per-dataset artifacts under paper/tables_{hotpotqa,mtrag} and paper/figures_{hotpotqa,mtrag}."
