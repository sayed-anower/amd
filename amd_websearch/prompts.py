"""Prompt loading.

The system prompt lives in `system-prompt.txt` (repo root) so it can be
tuned without touching Python. If the file is missing we fall back to a
built-in copy so the agent never hard-fails on a missing file.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_SYSTEM_PROMPT = """You are a precise web information retrieval agent.
You receive a QUESTION and numbered EVIDENCE retrieved from the web.
Answer ONLY from the EVIDENCE. Never use outside knowledge and never guess.
Copy names, numbers, dates and versions exactly as written in the EVIDENCE.
"sources" must contain only URLs copied verbatim from the EVIDENCE.
"res" must be ONE short paragraph of at most 45 words, plain text, no markdown.
Output exactly one JSON object: {"res": "<short answer>", "sources": ["<url>"]}
"""

DEFAULT_PLANNER_PROMPT = """You are the search planner of a web retrieval agent.
Given a user question, output 1 or 2 web search queries in the "queries" array.
Each query has 2 to 12 concrete keywords. No numbering, no explanation.
"""

PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["queries"],
}

GAP_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
    },
    "required": ["query"],
}


def load_prompt(path: str | Path, fallback: str = "") -> str:
    p = Path(path)
    try:
        if p.is_file():
            text = p.read_text(encoding="utf-8").strip()
            if text:
                return text
    except OSError:
        pass
    return fallback
