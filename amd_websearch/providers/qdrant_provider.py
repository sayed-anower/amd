"""Self-hosted Qdrant provider (the AMD-internal replacement for Tavily).

Search provider is pluggable. For AMD internal use, replace Tavily with
self-hosted Qdrant vector DB of AMD docs - zero external cost, zero
dependency, 100% runs on MI300X.

Implementation notes
--------------------
* Talks to Qdrant over plain HTTP (`/collections/{c}/points/search`), so no
  `qdrant-client` install is needed -> zero dependency.
* Query vectors come from any OpenAI-compatible `/v1/embeddings` endpoint.
  Point it at a llama.cpp server holding an embedding GGUF (e.g.
  `nomic-embed-text-v1.5`) or at vLLM/TEI running on MI300X -> no external
  API, no cost.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from ..llm import LlmError
from .base import SearchProvider, SearchResult, strip_html

TEXT_KEYS = ("text", "content", "document", "chunk", "body", "page_content")
URL_KEYS = ("url", "source", "link", "source_url", "href", "uri")
TITLE_KEYS = ("title", "name", "heading", "doc_title")


def _pick(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


class QdrantDocsProvider(SearchProvider):
    name = "qdrant"

    def __init__(self, config, llm=None) -> None:
        self.url = config.qdrant_url.rstrip("/")
        self.collection = config.qdrant_collection
        self.api_key = config.qdrant_api_key
        self.top_k = int(getattr(config, "qdrant_top_k", 6))
        self.timeout = float(getattr(config, "request_timeout", 30.0))
        self.embed_url = getattr(config, "qdrant_embed_url", "")
        self.embed_model = getattr(config, "qdrant_embed_model", "")
        self.llm = llm
        self._hf_embedder = None
        if not self.embed_url and getattr(config, "qdrant_embedding_model", ""):
            self._hf_model_id = config.qdrant_embedding_model
        else:
            self._hf_model_id = ""

    # ------------------------------------------------------------- embeddings
    def _embed_query(self, query: str) -> list[float]:
        if self.embed_url:
            payload = {"model": self.embed_model, "input": query}
            req = urllib.request.Request(
                self.embed_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            return list(data["data"][0]["embedding"])

        if self._hf_model_id:
            if self._hf_embedder is None:
                try:
                    from sentence_transformers import SentenceTransformer  # type: ignore
                except ImportError as exc:  # pragma: no cover - optional dependency
                    raise RuntimeError(
                        "QDRANT_EMBEDDING_MODEL needs `pip install sentence-transformers`, "
                        "or set QDRANT_EMBED_URL to an OpenAI-compatible /v1/embeddings endpoint."
                    ) from exc
                self._hf_embedder = SentenceTransformer(self._hf_model_id)
            vec = self._hf_embedder.encode(query)
            return [float(x) for x in vec]

        if self.llm is not None:
            try:
                return self.llm.embed([query])[0]
            except LlmError:
                pass

        raise RuntimeError(
            "No embedding backend configured. Set QDRANT_EMBED_URL "
            "(e.g. http://127.0.0.1:8099/v1/embeddings) or QDRANT_EMBEDDING_MODEL."
        )

    # ----------------------------------------------------------------- search
    def _search(self, vector: list[float], limit: int) -> list[dict[str, Any]]:
        url = f"{self.url}/collections/{self.collection}/points/search"
        payload = {"vector": vector, "limit": limit, "with_payload": True, "with_vector": False}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        return data.get("result", []) or []

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        query = (query or "").strip()
        if not query:
            return []
        vector = self._embed_query(query)
        hits = self._search(vector, min(int(max_results), max(1, self.top_k)))
        out: list[SearchResult] = []
        for hit in hits:
            payload = hit.get("payload") or {}
            text = strip_html(_pick(payload, TEXT_KEYS))
            if not text:
                continue
            out.append(
                SearchResult(
                    title=_pick(payload, TITLE_KEYS) or _pick(payload, URL_KEYS) or "document",
                    url=_pick(payload, URL_KEYS),
                    snippet=text,
                    score=float(hit.get("score") or 0.0),
                    provider=self.name,
                    raw={"id": hit.get("id")},
                )
            )
        return out
