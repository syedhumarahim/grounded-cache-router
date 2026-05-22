"""Pluggable generator interface.

Three concrete backends:
  - `echo`      : deterministic, dependency-free, used by unit tests
  - `anthropic` : Claude via the official SDK (works on a laptop)
  - `vllm`      : OpenAI-compatible vLLM HTTP endpoint (e.g. on RunPod)

All backends return a `GenResult` with TTFT, total latency, and a token-count
estimate so the eval harness has a uniform shape.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def _ntok(s: str) -> int:
        return len(_ENC.encode(s))
except Exception:  # pragma: no cover
    def _ntok(s: str) -> int:
        return max(1, len(s) // 4)


@dataclass
class GenResult:
    text: str
    ttft_s: float
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    backend: str


class Generator(ABC):
    name: str = "abstract"

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> GenResult:
        ...


class ExtractiveGenerator(Generator):
    """Deterministic, free, reproducible 'LLM'.

    Parses the prompt's Context block, scores each sentence by content-token
    overlap with the Question, and returns the top sentence(s). Because the
    answer is literally drawn from the retrieved evidence, the lexical support
    score in the stage-4 validator is high on benign reuse and drops sharply
    when evidence changes -- so every validator gate exercises meaningfully.

    Used as the paper's default backend for fully-reproducible runs that need
    no API key. Swap in `anthropic` or `vllm` for production quality numbers.
    """
    name = "extractive"

    _STOP = {
        "the", "a", "an", "of", "in", "on", "and", "or", "to", "is", "are",
        "was", "were", "be", "been", "for", "with", "by", "at", "as", "this",
        "that", "these", "those", "it", "its", "from", "into", "than", "but",
        "not", "no", "do", "does", "did", "have", "has", "had", "what",
        "which", "who", "whom", "whose", "when", "where", "why", "how",
    }

    def __init__(self, top_sentences: int = 1):
        self.top_sentences = top_sentences

    def _tokens(self, s: str) -> list[str]:
        import re
        return [t for t in re.findall(r"[a-z0-9]+", s.lower())
                if len(t) >= 3 and t not in self._STOP]

    def _parse(self, prompt: str) -> tuple[str, str]:
        ctx, _, rest = prompt.partition("# Context")
        ctx, _, q_block = rest.partition("# Question")
        q, _, _ = q_block.partition("# Answer")
        return ctx.strip(), q.strip()

    def generate(self, prompt, max_tokens=256, temperature=0.0):
        import re
        t0 = time.perf_counter()
        ctx, question = self._parse(prompt)
        # Strip the "[chunk_id vN]" headers before sentence splitting.
        ctx_clean = re.sub(r"\[[^\]]+\]\n?", " ", ctx)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ctx_clean) if s.strip()]
        if not sentences:
            text = "I don't know."
        else:
            q_toks = set(self._tokens(question))
            scored = sorted(
                ((len(set(self._tokens(s)) & q_toks), -i, s)
                 for i, s in enumerate(sentences)),
                reverse=True,
            )
            best = [s for score, _, s in scored[: self.top_sentences] if score > 0]
            text = " ".join(best) if best else "I don't know."
        ttft = time.perf_counter() - t0
        return GenResult(
            text=text,
            ttft_s=ttft,
            latency_s=time.perf_counter() - t0,
            prompt_tokens=_ntok(prompt),
            completion_tokens=_ntok(text),
            backend=self.name,
        )


class EchoGenerator(Generator):
    """Returns a deterministic answer based on the prompt tail. Zero-dep."""
    name = "echo"

    def generate(self, prompt, max_tokens=256, temperature=0.0):
        t0 = time.perf_counter()
        # Pretend first token takes 30ms, then stream the rest instantly.
        time.sleep(0.0)
        ttft = time.perf_counter() - t0
        text = f"[echo] {prompt.strip().splitlines()[-1][:200]}"
        return GenResult(
            text=text,
            ttft_s=ttft,
            latency_s=time.perf_counter() - t0,
            prompt_tokens=_ntok(prompt),
            completion_tokens=_ntok(text),
            backend=self.name,
        )


class AnthropicGenerator(Generator):
    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        from anthropic import Anthropic  # noqa: WPS433
        self.client = Anthropic()
        self.model = model

    def generate(self, prompt, max_tokens=256, temperature=0.0):
        t0 = time.perf_counter()
        ttft = None
        text_parts: list[str] = []
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for chunk in stream.text_stream:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text_parts.append(chunk)
        text = "".join(text_parts)
        latency = time.perf_counter() - t0
        return GenResult(
            text=text,
            ttft_s=ttft or latency,
            latency_s=latency,
            prompt_tokens=_ntok(prompt),
            completion_tokens=_ntok(text),
            backend=self.name,
        )


class VLLMGenerator(Generator):
    """OpenAI-compatible vLLM server (e.g. RunPod)."""
    name = "vllm"

    def __init__(self, model: str, base_url: str | None = None, api_key: str = "EMPTY",
                 timeout_s: float = 600.0, max_retries: int = 3):
        import httpx  # noqa: WPS433
        self.model = model
        self.base_url = (base_url or os.environ["VLLM_BASE_URL"]).rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self._http = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=30.0))

    def generate(self, prompt, max_tokens=256, temperature=0.0):
        import httpx  # noqa: WPS433
        import json as _json
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            ttft = None
            text_parts: list[str] = []
            url = f"{self.base_url}/v1/completions"
            try:
                with self._http.stream(
                    "POST", url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model, "prompt": prompt,
                        "max_tokens": max_tokens, "temperature": temperature,
                        "stream": True,
                    },
                ) as resp:
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        payload = line[len("data: "):]
                        if payload.strip() == "[DONE]":
                            break
                        obj = _json.loads(payload)
                        tok = obj["choices"][0].get("text", "")
                        if tok:
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            text_parts.append(tok)
                text = "".join(text_parts)
                latency = time.perf_counter() - t0
                return GenResult(
                    text=text, ttft_s=ttft or latency, latency_s=latency,
                    prompt_tokens=_ntok(prompt), completion_tokens=_ntok(text),
                    backend=self.name,
                )
            except (httpx.ReadTimeout, httpx.RemoteProtocolError,
                    httpx.ConnectError, httpx.ReadError) as e:
                last_exc = e
                # Exponential backoff; vLLM is usually back within seconds.
                time.sleep(2 ** attempt)
        raise RuntimeError(f"vLLM generate failed after {self.max_retries} retries") from last_exc


def make_generator(kind: str | None = None) -> Generator:
    """Factory honoring RAGCACHE_GENERATOR env if `kind` is None."""
    kind = (kind or os.environ.get("RAGCACHE_GENERATOR", "extractive")).lower()
    if kind == "echo":
        return EchoGenerator()
    if kind == "extractive":
        return ExtractiveGenerator()
    if kind == "anthropic":
        return AnthropicGenerator(model=os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"))
    if kind == "vllm":
        return VLLMGenerator(model=os.environ["VLLM_MODEL"])
    raise ValueError(f"unknown generator kind: {kind}")
