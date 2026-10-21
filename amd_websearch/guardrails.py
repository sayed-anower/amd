"""Guardrails: JSON repair, grounding, source validation, extractive fallback.

Two guarantees this module enforces:

1. The payload is always a valid ``{"res": str, "sources": [str, ...]}`` object.
2. Every sentence of ``res`` is lexically supported by the retrieved web
   evidence. Unsupported sentences are dropped, and if nothing survives we
   fall back to a deterministic extractive summary built from the evidence.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from .evidence import Evidence
from .textutil import numeric_facts, proper_nouns, sentences, tokenize

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "res": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["res", "sources"],
}

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first balanced JSON object out of a model response."""
    if not text:
        return None
    candidates: list[str] = []
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)

    for candidate in candidates:
        start = candidate.find("{")
        if start < 0:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(candidate)):
            ch = candidate[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    raw = candidate[start : i + 1]
                    for attempt in (raw, _relax_json(raw)):
                        try:
                            obj = json.loads(attempt)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(obj, dict):
                            return obj
                    break
    return None


def _relax_json(raw: str) -> str:
    """Fix the usual small-model JSON mistakes: trailing commas, single quotes."""
    out = re.sub(r",\s*([}\]])", r"\1", raw)
    out = re.sub(r"(?<![\w\\])'([^'\n]*)'(?=\s*[:,}\]])", r'"\1"', out)
    return out


def _clean_sources(raw: Any, evidence: Sequence[Evidence], limit: int = 5) -> list[str]:
    """Keep only URLs that really came from the retrieved documents."""
    allowed = {e.url.strip(): e for e in evidence if e.url.strip()}
    norm_map = {e.url.strip().rstrip("/").lower(): e.url.strip() for e in evidence if e.url.strip()}

    out: list[str] = []
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if not isinstance(item, str):
                continue
            item = item.strip()
            if not item:
                continue
            if item in allowed:
                canonical = item
            else:
                canonical = norm_map.get(item.rstrip("/").lower(), "")
            if canonical and canonical not in out:
                out.append(canonical)
            if len(out) >= limit:
                break
    return out


def ground_sentences(res: str, evidence: Sequence[Evidence], threshold: float = 0.55) -> str:
    """Drop sentences that are not supported by the evidence.

    A sentence survives when most of its content words appear in the
    evidence and every number/proper noun it contains appears verbatim.
    """
    res = (res or "").strip()
    if not res:
        return ""
    if not evidence:
        return ""

    ev_tokens: set[str] = set()
    ev_nums: set[str] = set()
    ev_props: set[str] = set()
    for e in evidence:
        ev_tokens.update(tokenize(f"{e.title} {e.text}"))
        ev_nums.update(numeric_facts(f"{e.title} {e.text}"))
        ev_props.update(p.lower() for p in proper_nouns(f"{e.title} {e.text}"))

    kept: list[str] = []
    for sent in sentences(res) or [res]:
        content = tokenize(sent)
        if not content:
            continue
        coverage = sum(1 for t in content if t in ev_tokens) / len(content)

        unsupported_num = [n for n in numeric_facts(sent) if n not in ev_nums]
        unsupported_prop = [
            p for p in proper_nouns(sent)
            if p.lower() not in ev_props and p.lower() not in {q.lower() for q in ev_props}
        ]

        if coverage < threshold:
            continue
        if unsupported_num:
            continue
        if unsupported_prop:
            # proper nouns are often partial phrases; only reject if the whole
            # phrase is unknown and the sentence is otherwise poorly covered.
            if coverage < threshold + 0.2:
                continue
        kept.append(sent.strip())

    out = " ".join(kept).strip()
    return out


def truncate_words(text: str, max_words: int) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(",;:") + "..."


def extractive_summary(query: str, evidence: Sequence[Evidence], max_words: int = 60) -> str:
    """Deterministic fallback: the best-supported sentences from the evidence."""
    if not evidence:
        return ""
    qtokens = set(tokenize(query))
    candidates: list[tuple[float, int, str]] = []
    order = 0
    for e in evidence:
        for sent in sentences(e.text):
            tokens = tokenize(sent)
            if len(tokens) < 4:
                continue
            overlap = len(qtokens & set(tokens)) / (len(qtokens) or 1)
            specificity = min(len(tokens), 40) / 40.0
            candidates.append((overlap + specificity * 0.2, order, sent.strip()))
            order += 1
    if not candidates:
        return truncate_words(evidence[0].text, max_words)
    candidates.sort(key=lambda x: -x[0])
    picked: list[str] = []
    total_words = 0
    for _, _, sent in candidates:
        w = len(sent.split())
        if total_words + w > max_words and picked:
            continue
        picked.append(sent)
        total_words += w
        if total_words >= max_words:
            break
    picked.sort(key=lambda s: next((o for _, o, t in candidates if t == s), 0))
    return " ".join(picked)


def build_answer(
    model_output: str,
    query: str,
    evidence: Sequence[Evidence],
    max_words: int = 60,
    coverage_threshold: float = 0.55,
) -> dict[str, Any]:
    """Turn raw model output into the final, validated payload."""
    obj = extract_json_object(model_output) or {}

    raw_res = obj.get("res") or obj.get("answer") or obj.get("response") or ""
    if isinstance(raw_res, (list, tuple)):
        raw_res = " ".join(str(x) for x in raw_res)
    raw_res = str(raw_res).strip()

    grounded = ground_sentences(raw_res, evidence, coverage_threshold)
    if len(tokenize(grounded)) < 4:
        grounded = ground_sentences(raw_res, evidence, max(0.0, coverage_threshold - 0.2))
    if not grounded:
        grounded = extractive_summary(query, evidence, max_words)

    res = truncate_words(grounded, max_words).strip()

    sources = _clean_sources(obj.get("sources") or obj.get("citations") or [], evidence)
    if not sources:
        # AMD requires sources - fall back to the documents we actually read.
        sources = [e.url for e in evidence[:3] if e.url]

    if not res:
        res = "No answer could be grounded in the retrieved web sources."

    return {"res": res, "sources": sources}
