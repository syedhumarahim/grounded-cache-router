# rag-cache-router

Evidence-validated cache routing for Retrieval-Augmented Generation.

**Working title (arXiv):** *Grounded Cache Routing for Retrieval-Augmented Generation*.

**Thesis.** Reuse a cached answer only when (1) the query is semantically close to a prior query, (2) the newly retrieved evidence overlaps strongly with the cached answer's evidence, (3) document versions/timestamps are still valid, and (4) the cached answer is still entailed by current evidence. Prefix/KV caching is treated as an orthogonal serving primitive, not the contribution.

## Stage map

| Stage | Scope | Status |
| --- | --- | --- |
| 1 | Baseline RAG + byte-exact dedup + eval harness + workload synthesis | in progress |
| 2 | Prefix caching (vLLM APC) + retrieval-result cache | todo |
| 3 | Naive semantic answer cache (baseline) | todo |
| 4 | Evidence validation + freshness gating + compression fallback (**paper contribution**) | todo |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[api,dev]"
python scripts/run_baseline.py --regime exact_repeat --n 20
```

Set `ANTHROPIC_API_KEY` for the API generator, or point `RAGCACHE_GENERATOR=vllm` + `VLLM_BASE_URL` at a RunPod vLLM endpoint.

## Layout

```
src/ragcache/
  corpus.py        chunking + byte-exact dedup
  retriever.py     embedding + FAISS index
  generator.py     pluggable LLM clients (api / vllm / echo)
  pipeline.py      retrieve -> dedup -> prompt -> generate
  workload.py      6 traffic regimes
  signature.py     EvidenceSignature (doc_ids, chunk_hashes, versions)
  cache/           (stages 2-4)
  eval/
    metrics.py
    harness.py
scripts/run_baseline.py
tests/
```
