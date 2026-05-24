# Evidence-Validated Cache Routing for RAG

**Paper:** [Grounded Cache Routing for Retrieval-Augmented Generation: When Is It Safe to Reuse an Answer?](cache_llm.pdf)

Modern RAG systems cache answers to save cost and latency, but a cosine-similar query can map to a different correct answer when the underlying evidence has changed. GroundedCache is a four-gate evidence-validated cache router that admits a cached answer only when reuse is provably safe.

## Key Idea

Instead of asking *how to reuse faster*, we ask *when reuse is safe*. A cached answer is served only when four cheap gates simultaneously pass:

| Gate | Check |
|------|-------|
| G1 — Query Similarity | Cosine similarity between new and cached query exceeds threshold |
| G2 — Evidence Overlap | Jaccard overlap of retrieved chunk IDs with cached evidence signature |
| G3 — Version Validity | Source document versions/hashes match between cached and current evidence |
| G4 — Lexical Support | Cached answer tokens are covered by the freshly retrieved evidence |

If any gate fails, the router falls back to query-conditioned compression and full generation.

## Results

Evaluated across **HotpotQA** (multi-hop) and **mtRAG** (multi-turn) with **12,000 real-LLM generations** (Qwen2.5-7B-Instruct on vLLM with Automatic Prefix Caching):

- **HotpotQA:** Drives unsafe-served rate (USR) to **0.0%** on every regime (vs. 15-35% under naive caching)
- **mtRAG document-drift:** Reduces USR from 51.5% to 1.5% (34x reduction)
- **Latency:** End-to-end p50 stays within 1.04-1.07x of a no-cache RAG baseline

## Six-Regime Workload

We provide a deterministic workload synthesizer that stress-tests cache *safety*, not just hit rate:

1. **exact_repeat** — identical query, same evidence
2. **paraphrase** — rephrased query, same evidence
3. **near_miss** — similar query, different gold answer
4. **document_drift** — same query, mutated evidence (numbers changed)
5. **long_shared_doc** — queries sharing a long document but needing different sections
6. **bounded_kb_cag** — cache-augmented generation over a fixed knowledge base

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run a quick smoke test with the extractive (deterministic) backend
python scripts/run_baseline.py --regime exact_repeat --n 20

# Run the full router with all four gates
python scripts/run_router.py --regime document_drift --variant full --n 50
```

For real-LLM evaluation, point at a vLLM endpoint:

```bash
export RAGCACHE_GENERATOR=vllm
export VLLM_BASE_URL=http://<your-vllm-endpoint>:8000/v1
python scripts/run_sweeps.py --dataset hotpotqa --n 200 --out-dir runs/sweep
```

## Repository Layout

```
src/ragcache/
  corpus.py          # Chunking + byte-exact dedup
  retriever.py       # Embedding + FAISS index
  generator.py       # Pluggable LLM backends (vllm / extractive / echo)
  pipeline.py        # Retrieve -> dedup -> prompt -> generate
  workload.py        # Six-regime traffic synthesis
  signature.py       # EvidenceSignature (doc_ids, chunk_hashes, versions)
  compression.py     # Query-conditioned context compression
  cache/
    answer_cache.py   # Semantic answer cache (cosine similarity)
    retrieval_cache.py# Retrieval-result cache
    validator.py      # Four-gate evidence validator
    router.py         # Full cache router orchestrator
  eval/
    metrics.py        # USR, aHR, FH, latency, token metrics
    harness.py        # Baseline evaluation harness
    router_harness.py # Router evaluation harness
scripts/
  run_sweeps.py       # Full sweep across regimes x variants
  run_paper_sweeps.sh # Reproduce all paper numbers
  collate_results.py  # Generate LaTeX tables from sweep results
  make_figures.py     # Generate paper figures
tests/                # Unit tests for all components
```

## Reproducing Paper Results

See [REPRO.md](REPRO.md) for full instructions. The short version:

```bash
# With a vLLM endpoint running Qwen2.5-7B-Instruct:
export RAGCACHE_GENERATOR=vllm
export VLLM_BASE_URL=http://<endpoint>:8000/v1
bash scripts/run_paper_sweeps.sh
```

## Citation

```bibtex
@article{shah2026groundedcache,
  title={Grounded Cache Routing for Retrieval-Augmented Generation: When Is It Safe to Reuse an Answer?},
  author={Shah, Syed Huma},
  year={2026}
}
```

## License

CC BY 4.0
