"""Sanity-check a vLLM (OpenAI-compatible) endpoint before running a sweep.

Verifies the endpoint is reachable, the model is loaded, and a real
streamed completion comes back with non-zero TTFT. Run this once after
spinning up your RunPod pod and before launching `make sweep GEN=vllm`.

Usage:
  export VLLM_BASE_URL=https://<pod>.proxy.runpod.net
  export VLLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
  python scripts/check_vllm.py
"""
from __future__ import annotations

import os
import sys
import time

import httpx


def main() -> int:
    base = os.environ.get("VLLM_BASE_URL")
    model = os.environ.get("VLLM_MODEL")
    if not base or not model:
        print("ERROR: set VLLM_BASE_URL and VLLM_MODEL first.", file=sys.stderr)
        return 2
    base = base.rstrip("/")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

    # 1. /v1/models reachability
    print(f"--> GET {base}/v1/models")
    r = httpx.get(f"{base}/v1/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    r.raise_for_status()
    ids = [m["id"] for m in r.json().get("data", [])]
    print(f"    models exposed: {ids}")
    if model not in ids:
        print(f"WARNING: requested model {model!r} not in served list.")

    # 2. streamed completion -- measure TTFT
    prompt = "Q: What is the capital of France?\nA:"
    print(f"--> POST {base}/v1/completions  (stream)")
    t0 = time.perf_counter()
    ttft = None
    text = []
    with httpx.Client(timeout=120) as c, c.stream(
        "POST", f"{base}/v1/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "prompt": prompt,
              "max_tokens": 32, "temperature": 0.0, "stream": True},
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload.strip() == "[DONE]":
                break
            import json as _json
            tok = _json.loads(payload)["choices"][0].get("text", "")
            if tok:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text.append(tok)
    latency = time.perf_counter() - t0
    print(f"    TTFT: {ttft*1000:.0f} ms  latency: {latency*1000:.0f} ms")
    print(f"    completion: {''.join(text)!r}")
    print("\nOK. Ready for: make sweep GEN=vllm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
