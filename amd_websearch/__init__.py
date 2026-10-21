"""AMD AI League - Match 4: Intelligent Web Information Retrieval.

A fast, token-frugal web retrieval agent:

    query -> (llama.cpp planner) -> Tavily (pluggable) -> evidence selection
          -> (llama.cpp Qwen2.5-0.5B-Instruct) -> {"res": "...", "sources": [...]}

The only hard requirement at runtime is a llama.cpp server holding
Qwen2.5-0.5B-Instruct. The search provider is an interface, so it can be
swapped for a self-hosted AMD docs index without touching the agent.
"""

from .agent import WebSearchAgent
from .config import AppConfig
from .llm import LlamaClient

__all__ = ["WebSearchAgent", "AppConfig", "LlamaClient"]
__version__ = "1.0.0"
