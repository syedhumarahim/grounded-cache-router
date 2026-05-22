# RunPod vLLM runbook

Step-by-step for getting the paper's "real-LLM" numbers on a single GPU.

## 1. Spin up a pod

In the RunPod dashboard:

| | |
|---|---|
| Template | **vLLM Latest** (community template, OpenAI-compatible server) |
| GPU | 1 × A100 80GB or 1 × H100 80GB (or 1 × A6000 48GB for the 8B model) |
| Disk | 80 GB volume |
| Expose HTTP port | **8000** |
| Env: `MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` |
| Env: `MAX_MODEL_LEN` | `8192` |
| Env: `ENABLE_PREFIX_CACHING` | `true`  ← this is what makes vLLM APC active |
| Env: `HUGGING_FACE_HUB_TOKEN` | your HF token (gated model) |

If you prefer to start vLLM by hand, the equivalent command on the pod's
shell is:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enable-prefix-caching \
  --max-model-len 8192 \
  --host 0.0.0.0 --port 8000
```

When the pod is running, RunPod gives you a public proxy URL like
`https://abc123-8000.proxy.runpod.net`.

## 2. Configure your local env

```bash
export VLLM_BASE_URL=https://abc123-8000.proxy.runpod.net
export VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
# Optional if your endpoint requires a key (default is "EMPTY"):
# export VLLM_API_KEY=...
```

## 3. Sanity check

```bash
python scripts/check_vllm.py
```

You should see model IDs and a TTFT around 100–500 ms with a real
completion. If TTFT is multiple seconds the pod is probably cold or
under-spec.

## 4. Run the sweep

```bash
make sweep GEN=vllm DATASET=hotpotqa N=200 SWEEP_DIR=runs/sweeps_vllm
make tables figures SWEEP_DIR=runs/sweeps_vllm
```

This launches 30 cells (6 regimes × 5 variants). Expect ~30–60 min total
depending on GPU. The Makefile writes tables/figures into `paper/` so the
LaTeX recompiles unchanged.

## 5. Multi-dataset sweep

The Makefile supports re-running with a different dataset by changing
`DATASET=` and `SWEEP_DIR=`. To get the multi-turn referent-shift result
the paper discusses, run an mtRAG sweep too:

```bash
git clone https://github.com/IBM/mt-rag-benchmark.git ~/data/mt-rag-benchmark
export MTRAG_PATH=~/data/mt-rag-benchmark
make sweep GEN=vllm DATASET=mtrag N=150 SWEEP_DIR=runs/sweeps_mtrag
make tables figures SWEEP_DIR=runs/sweeps_mtrag
```

## 6. Cost guidance

The full HotpotQA n=200 sweep is ~6000 generations of ≤256 tokens. At
typical RunPod pricing for one A100 80GB ($1.50–$2/hr) the sweep is
roughly $1–$2 of compute. Tear the pod down after.

## What changes vs. the extractive baseline

- **TTFT becomes a real number.** The extractive backend reports
  near-zero TTFT (it's just a sentence picker). With vLLM you get real
  millisecond-scale TTFT and the prefix-cache-friendly chunk ordering
  starts to pay off measurably.
- **False-hit rate on benign regimes drops sharply.** The extractive
  backend's verbose answers fail SQuAD-style F1 against short HotpotQA
  gold strings even when correct, inflating FH on benign regimes. A real
  instruction-tuned LLM produces concise answers that match gold cleanly.
- **The document-drift result becomes more dramatic.** With real LLM
  answers, naive cache reuse will surface stale information directly,
  making \sys's version-gate rejection more impactful.

Once the sweep is in, regenerate the paper:

```bash
make tables figures SWEEP_DIR=runs/sweeps_vllm
# upload to Overleaf or compile locally
```
