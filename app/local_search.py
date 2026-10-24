"""Offline local-JSONL search fallback (works with ``--network none``)."""

from __future__ import annotations

from amd_websearch.config import SearchConfig
from amd_websearch.providers.local import LocalCorpusProvider

from .search import resolve_corpus_path


class LocalJSONLSearch(LocalCorpusProvider):
    """Thin, explicit wrapper around the bundled BM25 corpus provider."""

    name = "local"


def build_local_provider(config: SearchConfig) -> LocalJSONLSearch:
    config.provider = "local"
    config.local_corpus_path = resolve_corpus_path(config)
    return LocalJSONLSearch(config)


__all__ = ["LocalJSONLSearch", "build_local_provider"]
