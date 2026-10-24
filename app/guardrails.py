"""Output guardrails: JSON repair, grounding, source validation, word cap.

Re-exported from the audited implementation plus a final contract check used
by the batch runner, so every line written to ``answers.jsonl`` has exactly
``id``, ``res`` and ``sources``.
"""

from __future__ import annotations

from typing import Any

from amd_websearch.guardrails import (
    ANSWER_SCHEMA,
    build_answer,
    extract_json_object,
    extractive_summary,
    focus_answer,
    ground_sentences,
    truncate_words,
)

MAX_WORDS = 45


def enforce_contract(payload: dict[str, Any], max_words: int = MAX_WORDS) -> dict[str, Any]:
    """Return a valid ``{"res", "sources"}`` object, no exceptions."""
    res = payload.get("res") if isinstance(payload, dict) else ""
    if not isinstance(res, str):
        res = str(res or "")
    sources = payload.get("sources") if isinstance(payload, dict) else []
    if isinstance(sources, str):
        sources = [sources]
    cleaned_sources: list[str] = []
    if isinstance(sources, (list, tuple)):
        for source in sources:
            if not isinstance(source, str):
                continue
            source = source.strip()
            if source.lower().startswith(("http://", "https://")) and source not in cleaned_sources:
                cleaned_sources.append(source)
    res = truncate_words(" ".join(res.split()), max_words).strip()
    if not res:
        res = "No answer could be grounded in the retrieved web sources."
    return {"res": res, "sources": cleaned_sources[:3]}


__all__ = [
    "ANSWER_SCHEMA",
    "MAX_WORDS",
    "build_answer",
    "enforce_contract",
    "extract_json_object",
    "extractive_summary",
    "focus_answer",
    "ground_sentences",
    "truncate_words",
]
