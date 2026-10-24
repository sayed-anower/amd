"""Search-provider selection for the Match 4 agent.

The provider is pluggable through ``SEARCH_PROVIDER``:

    tavily  -> hosted web search for the public demo (needs TAVILY_API_KEY)
    qdrant  -> self-hosted AMD documentation vector database
    local   -> offline JSONL corpus (BM25), used for OFFLINE_EVAL=1

When no Tavily key is configured, or when ``OFFLINE_EVAL=1`` is set, the
agent automatically uses the bundled local corpus. That is what makes the
grader's ``--network none`` run work without any external service.
"""

from __future__ import annotations

import os
from pathlib import Path

from amd_websearch.config import REPO_ROOT, SearchConfig
from amd_websearch.providers import build_provider
from amd_websearch.providers.base import SearchProvider, SearchResult

_TRUTHY = {"1", "true", "yes", "on"}
_DATA_CORPUS_CANDIDATES = (
    Path("/data/corpus.jsonl"),
    Path("/data/amd_docs.jsonl"),
    Path("/data/documents.jsonl"),
)
_CORPUS_CANDIDATES = (
    REPO_ROOT / "corpus" / "amd_docs.jsonl",
    Path("/app/corpus/amd_docs.jsonl"),
    Path(__file__).resolve().parent.parent / "corpus" / "amd_docs.jsonl",
)


def offline_eval() -> bool:
    return os.environ.get("OFFLINE_EVAL", "").strip().lower() in _TRUTHY


def _first_existing(paths) -> str:
    for path in paths:
        try:
            if path.is_file():
                return str(path)
        except OSError:
            continue
    return ""


def resolve_corpus_path(config: SearchConfig) -> str:
    """Prefer a corpus mounted at /data, then the bundled AMD docs corpus."""
    env_path = os.environ.get("LOCAL_CORPUS_PATH", "").strip()
    if env_path:
        return env_path
    mounted = _first_existing(_DATA_CORPUS_CANDIDATES)
    if mounted:
        return mounted
    bundled = _first_existing(_CORPUS_CANDIDATES)
    return bundled or config.local_corpus_path


def resolve_provider_name(config: SearchConfig) -> str:
    name = (os.environ.get("SEARCH_PROVIDER") or config.provider or "tavily").lower()
    if offline_eval():
        return "local"
    if name == "tavily" and not config.api_key:
        return "local"
    return name


def build_search_provider(config: SearchConfig, llm=None) -> SearchProvider:
    """Build the configured provider, automatically going offline when safe."""
    name = resolve_provider_name(config)
    config.provider = name
    if name == "local":
        config.local_corpus_path = resolve_corpus_path(config)

    try:
        return build_provider(config, llm=llm)
    except RuntimeError:
        if name != "local":
            config.provider = "local"
            config.local_corpus_path = resolve_corpus_path(config)
            return build_provider(config, llm=llm)
        raise


def search_web(provider: SearchProvider, query: str, max_results: int = 5) -> list[SearchResult]:
    """Search through any provider while never raising into the pipeline."""
    try:
        return provider.search(query, max_results)
    except Exception:  # noqa: BLE001 - the agent has a fallback path
        return []


__all__ = [
    "build_search_provider",
    "offline_eval",
    "resolve_corpus_path",
    "resolve_provider_name",
    "search_web",
]
