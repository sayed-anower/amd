"""Pluggable search providers.

The agent only ever talks to a `SearchProvider`, so the backend can be
swapped without touching the reasoning code:

    tavily   -> hosted web search (used for the public demo)
    qdrant   -> self-hosted AMD docs vector DB (AMD internal, zero external cost)
    local    -> offline JSONL corpus, BM25 (demo mode / tests / no API key)
"""

from .base import SearchProvider, SearchResult
from .local import LocalCorpusProvider
from .qdrant_provider import QdrantDocsProvider
from .tavily import TavilyProvider

__all__ = [
    "SearchProvider",
    "SearchResult",
    "TavilyProvider",
    "QdrantDocsProvider",
    "LocalCorpusProvider",
    "build_provider",
]


def build_provider(config, llm=None):
    """Factory: build the configured provider, falling back if it is unusable."""
    name = (config.provider or "tavily").lower()

    if name == "tavily":
        if config.api_key:
            return TavilyProvider(config)
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Export it, or run with "
            "SEARCH_PROVIDER=local for the offline demo corpus."
        )

    if name == "qdrant":
        if not config.qdrant_url:
            raise RuntimeError("QDRANT_URL is required for the qdrant provider.")
        return QdrantDocsProvider(config, llm=llm)

    if name == "local":
        return LocalCorpusProvider(config)

    raise RuntimeError(f"unknown search provider: {name!r}")
