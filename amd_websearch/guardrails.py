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
_COMMON_START = {
    "the", "this", "these", "those", "it", "a", "an", "in", "on", "for",
    "however", "additionally", "also", "there", "they", "we", "you", "its",
    "their", "our", "according", "based",
}
_MARKDOWN_RE = re.compile(r"[*_`#>]+")


def _plain(text: str) -> str:
    """Remove the markdown tokens a small model still emits despite the prompt."""
    text = _MARKDOWN_RE.sub("", text or "")
    text = re.sub(r"^\s*(?:[-•]|\d+[.)])\s*", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip()



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


def _clean_sources(raw: Any, evidence: Sequence[Evidence], limit: int = 3) -> list[str]:
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
            if not item.lower().startswith(("http://", "https://")):
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
            if p.lower() not in ev_props and p.lower() not in _COMMON_START
        ]

        if coverage < threshold:
            continue
        if unsupported_num or unsupported_prop:
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


_NUMERIC_INTENT = (
    "how many", "how much", "version", "latest", "release", "date", "when",
    "price", "cost", "size", "count", "number", "percentage", "bandwidth",
    "memory", "capacity", "speed", "year", "how long", "how old",
)


def _numeric_intent(query: str) -> bool:
    low = f" {(query or '').lower()} "
    return any(marker in low for marker in _NUMERIC_INTENT)


def focus_answer(answer: str, query: str) -> str:
    """Keep quantity/version answers on the facts the question asked for.

    A 0.5B model tends to append everything it read. When the question asks
    for a number, version or date, sentences without such a fact are noise.
    A leading subject sentence is kept only when it is short and relevant.
    """
    sents = sentences(answer) or ([answer] if answer else [])
    if len(sents) <= 1 or not _numeric_intent(query):
        return " ".join(s.strip() for s in sents)

    q = set(tokenize(query))
    generic = {
        "how", "much", "many", "version", "latest", "releas", "date", "when",
        "price", "cost", "size", "count", "number", "percentag", "bandwidth",
        "memori", "capac", "speed", "year", "long", "old",
    }
    specific = q - generic
    numeric = [
        s for s in sents
        if numeric_facts(s) and (not specific or (q & set(tokenize(s))))
    ]
    if not numeric:
        return " ".join(s.strip() for s in sents)

    first = sents[0]
    keep_first = (
        first not in numeric
        and len(first.split()) <= 14
        and bool(q & set(tokenize(first)))
    )
    kept = ([first] if keep_first else []) + numeric
    seen: set[str] = set()
    ordered: list[str] = []
    for sent in sents:
        if sent in kept and sent not in seen:
            ordered.append(sent.strip())
            seen.add(sent)
    return " ".join(ordered)


def _rank_sources(
    sources: Sequence[str], res: str, evidence: Sequence[Evidence], limit: int
) -> list[str]:
    """Order/filter sources by how well they support the final answer.

    Numbers and versions are weighted heavily, so a source that contains the
    exact specification quoted in the answer outranks a page that only shares
    a product name.
    """
    if not sources:
        return []
    answer_tokens = set(tokenize(res))
    answer_nums = numeric_facts(res)
    by_url = {e.url.strip(): e for e in evidence if e.url.strip()}

    # If the model cited a vague page but another retrieved page contains the
    # exact specification in the answer, make that page a candidate too.
    candidates = list(sources)
    if answer_nums:
        for e in evidence:
            url = e.url.strip()
            if url and url not in candidates and (answer_nums & numeric_facts(f"{e.title} {e.text}")):
                candidates.append(url)

    scored: list[tuple[float, int, int, str]] = []
    for i, url in enumerate(candidates):
        e = by_url.get(url.strip())
        if e is None:
            continue
        doc = f"{e.title} {e.text}"
        doc_tokens = set(tokenize(doc))
        overlap = len(answer_tokens & doc_tokens) / (len(answer_tokens) or 1)
        num_hits = len(answer_nums & numeric_facts(doc))
        num_score = (num_hits / len(answer_nums)) if answer_nums else 0.0
        scored.append((overlap + 2.0 * num_score, num_hits, i, url))
    if not scored:
        return list(sources[:limit])
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))

    numeric_supported = [item for item in scored if item[1] > 0]
    pool = numeric_supported if (answer_nums and numeric_supported) else scored
    kept = [url for _, _, _, url in pool][:limit]
    return kept or [url for _, _, _, url in scored[:limit]]


def build_answer(
    model_output: str,
    query: str,
    evidence: Sequence[Evidence],
    max_words: int = 45,
    coverage_threshold: float = 0.6,
    max_sources: int = 3,
) -> dict[str, Any]:
    """Turn raw model output into the final, validated payload."""
    obj = extract_json_object(model_output) or {}

    raw_res = obj.get("res") or obj.get("answer") or obj.get("response") or ""
    if isinstance(raw_res, (list, tuple)):
        raw_res = " ".join(str(x) for x in raw_res)
    raw_res = _plain(str(raw_res))

    grounded = ground_sentences(raw_res, evidence, coverage_threshold)
    if len(tokenize(grounded)) < 4:
        grounded = ground_sentences(raw_res, evidence, max(0.0, coverage_threshold - 0.2))
    if not grounded:
        grounded = extractive_summary(query, evidence, max_words)
    grounded = focus_answer(grounded, query)

    res = truncate_words(grounded, max_words).strip()

    sources = _clean_sources(
        obj.get("sources") or obj.get("citations") or [], evidence, limit=max_sources
    )
    if not sources:
        # AMD requires sources - fall back to the documents we actually read.
        sources = [e.url for e in evidence[:max_sources] if e.url]
    sources = _rank_sources(sources, res, evidence, max_sources)

    if not res:
        res = "No answer could be grounded in the retrieved web sources."

    return {"res": res, "sources": sources}
