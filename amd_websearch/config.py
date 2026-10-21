"""Configuration for the web retrieval agent.

Everything is environment driven so the same code runs on a laptop (CPU
llama.cpp) and on MI300X (ROCm llama.cpp) without edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def load_dotenv(path: str | os.PathLike[str] | None = None) -> None:
    """Minimal .env loader (stdlib only, no python-dotenv dependency)."""

    dotenv_path = Path(path) if path else REPO_ROOT / ".env"
    if not dotenv_path.is_file():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class LlmConfig:
    base_url: str = "http://127.0.0.1:8099/v1"
    model: str = "qwen2.5-0.5b-instruct"
    api_key: str = ""
    timeout: float = 120.0
    retries: int = 2
    temperature: float = 0.0
    cache_prompt: bool = True

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    @property
    def embeddings_url(self) -> str:
        return self.base_url.rstrip("/") + "/embeddings"


@dataclass
class SearchConfig:
    provider: str = "tavily"
    api_key: str = ""
    max_results_per_query: int = 5
    search_depth: str = "basic"          # basic | advanced
    max_sub_queries: int = 2
    max_rounds: int = 2
    request_timeout: float = 30.0
    max_parallel_searches: int = 4
    include_raw_content: bool = False
    include_answer: bool = False
    qdrant_url: str = ""
    qdrant_collection: str = ""
    qdrant_api_key: str = ""
    qdrant_top_k: int = 6
    qdrant_embedding_model: str = ""    # optional: HF model id for sentence-transformers
    qdrant_embed_url: str = ""          # optional: OpenAI-compatible /v1/embeddings endpoint
    qdrant_embed_model: str = ""
    local_corpus_path: str = str(REPO_ROOT / "corpus" / "amd_docs.jsonl")


@dataclass
class BudgetConfig:
    """Everything here trades answer quality against token burn."""

    max_snippets: int = 6
    evidence_char_budget: int = 3200
    snippet_char_limit: int = 420
    max_response_words: int = 60
    planner_max_tokens: int = 64
    answer_max_tokens: int = 200
    gap_max_tokens: int = 40
    ground_coverage_threshold: float = 0.55
    dedupe_jaccard: float = 0.85


@dataclass
class AppConfig:
    llm: LlmConfig = field(default_factory=LlmConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    system_prompt_path: str = str(REPO_ROOT / "system-prompt.txt")
    planner_prompt_path: str = str(REPO_ROOT / "planner-prompt.txt")
    provider_fallback: str = "local"    # used when the primary provider is unusable
    debug: bool = False

    @classmethod
    def from_env(cls, dotenv: bool = True) -> "AppConfig":
        if dotenv:
            load_dotenv()
        return cls(
            llm=LlmConfig(
                base_url=_env("LLM_BASE_URL", "http://127.0.0.1:8099/v1"),
                model=_env("LLM_MODEL", "qwen2.5-0.5b-instruct"),
                api_key=_env("LLM_API_KEY", ""),
                timeout=_env_float("LLM_TIMEOUT", 120.0),
                retries=_env_int("LLM_RETRIES", 2),
                temperature=_env_float("LLM_TEMPERATURE", 0.0),
                cache_prompt=_env_bool("LLM_CACHE_PROMPT", True),
            ),
            search=SearchConfig(
                provider=_env("SEARCH_PROVIDER", "tavily"),
                api_key=_env("TAVILY_API_KEY", ""),
                max_results_per_query=_env_int("TAVILY_MAX_RESULTS", 5),
                search_depth=_env("TAVILY_SEARCH_DEPTH", "basic"),
                max_sub_queries=_env_int("MAX_SUB_QUERIES", 2),
                max_rounds=_env_int("MAX_ROUNDS", 2),
                request_timeout=_env_float("SEARCH_TIMEOUT", 30.0),
                max_parallel_searches=_env_int("MAX_PARALLEL_SEARCHES", 4),
                include_raw_content=_env_bool("TAVILY_INCLUDE_RAW", False),
                include_answer=_env_bool("TAVILY_INCLUDE_ANSWER", False),
                qdrant_url=_env("QDRANT_URL", ""),
                qdrant_collection=_env("QDRANT_COLLECTION", ""),
                qdrant_api_key=_env("QDRANT_API_KEY", ""),
                qdrant_top_k=_env_int("QDRANT_TOP_K", 6),
                qdrant_embedding_model=_env("QDRANT_EMBEDDING_MODEL", ""),
                qdrant_embed_url=_env("QDRANT_EMBED_URL", ""),
                qdrant_embed_model=_env("QDRANT_EMBED_MODEL", ""),
                local_corpus_path=_env("LOCAL_CORPUS_PATH", str(REPO_ROOT / "corpus" / "amd_docs.jsonl")),
            ),
            budget=BudgetConfig(
                max_snippets=_env_int("MAX_SNIPPETS", 6),
                evidence_char_budget=_env_int("EVIDENCE_CHAR_BUDGET", 3200),
                snippet_char_limit=_env_int("SNIPPET_CHAR_LIMIT", 420),
                max_response_words=_env_int("MAX_RESPONSE_WORDS", 60),
                planner_max_tokens=_env_int("PLANNER_MAX_TOKENS", 64),
                answer_max_tokens=_env_int("ANSWER_MAX_TOKENS", 200),
                gap_max_tokens=_env_int("GAP_MAX_TOKENS", 40),
                ground_coverage_threshold=_env_float("GROUND_COVERAGE", 0.55),
            ),
            system_prompt_path=_env("SYSTEM_PROMPT_PATH", str(REPO_ROOT / "system-prompt.txt")),
            planner_prompt_path=_env("PLANNER_PROMPT_PATH", str(REPO_ROOT / "planner-prompt.txt")),
            provider_fallback=_env("PROVIDER_FALLBACK", "local"),
            debug=_env_bool("DEBUG", False),
        )
