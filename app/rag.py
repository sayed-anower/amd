"""Retrieval-augmented generation: evidence engine + llama.cpp client.

The heavy lifting lives in the ``amd_websearch`` package (dedupe, BM25,
relevance filtering, sentence compression, gap analysis, synthesis). This
module exposes the stable entry points the batch runner and graders use.
"""

from __future__ import annotations

from amd_websearch.agent import WebSearchAgent
from amd_websearch.config import AppConfig
from amd_websearch.evidence import Evidence, build_evidence, evidence_text
from amd_websearch.llm import LlamaClient, LlmError, LlmResponse
from amd_websearch.providers.base import SearchResult

from .search import build_search_provider


def build_agent(config: AppConfig | None = None) -> WebSearchAgent:
    """Create one warm agent for the whole batch."""
    config = config or AppConfig.from_env()
    llm = LlamaClient(config.llm)
    provider = build_search_provider(config.search, llm=llm)
    return WebSearchAgent(llm=llm, provider=provider, config=config)


__all__ = [
    "AppConfig",
    "Evidence",
    "LlamaClient",
    "LlmError",
    "LlmResponse",
    "SearchResult",
    "WebSearchAgent",
    "build_agent",
    "build_evidence",
    "evidence_text",
]
