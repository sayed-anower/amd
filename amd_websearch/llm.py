"""llama.cpp client.

Talks to a `llama-server` over its OpenAI-compatible `/v1/chat/completions`
endpoint using the standard library only. Supports JSON-schema constrained
decoding, which is what keeps a 0.5B model emitting a parseable
`{"res": ..., "sources": [...]}` object on the first try.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import LlmConfig


class LlmError(RuntimeError):
    """Raised when the llama.cpp server cannot be reached or returns garbage."""


@dataclass
class LlmResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LlamaClient:
    """Minimal, dependency-free client for llama.cpp's OpenAI-compatible API."""

    def __init__(self, config: LlmConfig | None = None) -> None:
        self.config = config or LlmConfig()

    # ------------------------------------------------------------------ utils
    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                data = json.loads(raw)
                data["_latency_ms"] = int((time.perf_counter() - started) * 1000)
                return data
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.config.retries:
                    time.sleep(0.4 * (attempt + 1))
        raise LlmError(f"llama.cpp request to {url} failed: {last_error}")

    def health(self) -> bool:
        """True when the server answers /health."""
        try:
            url = self.config.base_url.rstrip("/").removesuffix("/v1") + "/health"
            with urllib.request.urlopen(url, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    # ------------------------------------------------------------------- chat
    def chat(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int = 256,
        temperature: float | None = None,
        stop: Sequence[str] | None = None,
        json_schema: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> LlmResponse:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
            "max_tokens": int(max_tokens),
            "temperature": self.config.temperature if temperature is None else temperature,
            "top_k": 1,
            "cache_prompt": self.config.cache_prompt,
            "stream": False,
        }
        if stop:
            payload["stop"] = list(stop)
        if seed is not None:
            payload["seed"] = int(seed)
        if json_schema is not None:
            # llama.cpp: schema-constrained JSON decoding -> always parseable.
            payload["response_format"] = {"type": "json_object", "schema": json_schema}

        data = self._post(self.config.chat_url, payload)
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(f"unexpected llama.cpp response shape: {str(data)[:300]}") from exc

        usage = data.get("usage") or {}
        return LlmResponse(
            text=text.strip(),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=int(data.get("_latency_ms") or 0),
            raw=data,
        )

    # -------------------------------------------------------------- embeddings
    def embed(self, texts: Sequence[str], model: str | None = None) -> list[list[float]]:
        """Embed texts with a llama.cpp-hosted embedding model (used by the
        self-hosted AMD docs provider so it needs zero external services)."""
        payload = {
            "model": model or self.config.model,
            "input": list(texts),
        }
        data = self._post(self.config.embeddings_url, payload)
        try:
            return [item["embedding"] for item in data["data"]]
        except (KeyError, TypeError) as exc:
            raise LlmError("unexpected embeddings response shape") from exc
