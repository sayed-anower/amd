"""Evidence selection: dedupe, rank, compress.

The model only sees what survives this step, so this is where token burn is
actually controlled. We keep the highest-signal sentences from the highest
-signal documents and throw the rest away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .providers.base import SearchResult
from .textutil import BM25, jaccard_char, sentences, tokenize


@dataclass
class Evidence:
    index: int
    title: str
    url: str
    text: str
    score: float = 0.0

    def as_card(self) -> str:
        """Compact rendering used inside the prompt."""
        title = self.title.strip() or self.url
        return f"[{self.index}] {title}\nURL: {self.url}\n{self.text}"


def dedupe(results: Sequence[SearchResult], threshold: float = 0.85) -> list[SearchResult]:
    """Drop duplicate URLs and near-duplicate snippets (syndicated articles)."""
    seen_urls: set[str] = set()
    kept: list[SearchResult] = []
    for res in results:
        key = res.key()
        if key and key in seen_urls:
            continue
        duplicate = False
        for other in kept:
            if other.url == res.url:
                duplicate = True
                break
            if res.snippet and other.snippet and jaccard_char(res.snippet, other.snippet) >= threshold:
                duplicate = True
                break
        if duplicate:
            continue
        if key:
            seen_urls.add(key)
        kept.append(res)
    return kept


def compress_snippet(text: str, query_tokens: Sequence[str], limit: int) -> str:
    """Keep the query-relevant sentences of a snippet within `limit` chars."""
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text

    sents = sentences(text)
    if not sents:
        sents = [text]

    scored: list[tuple[float, int, str]] = []
    for i, sent in enumerate(sents):
        tokens = set(tokenize(sent))
        overlap = sum(1 for t in query_tokens if t in tokens)
        # position bonus: lead sentences usually carry the definition
        position_bonus = max(0.0, 0.35 - 0.02 * i)
        scored.append((overlap + position_bonus, i, sent))

    scored.sort(key=lambda x: (-x[0], x[1]))

    chosen: list[str] = []
    total = 0
    for _, i, sent in scored:
        if total + len(sent) + 1 > limit and chosen:
            continue
        chosen.append(sent)
        total += len(sent) + 1
        if total >= limit:
            break

    chosen.sort(key=lambda s: sents.index(s))
    out = " ".join(chosen).strip()
    if len(out) > limit:
        out = out[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "..."
    return out


def build_evidence(
    query: str,
    results: Sequence[SearchResult],
    max_snippets: int = 4,
    char_budget: int = 1500,
    snippet_char_limit: int = 360,
    dedupe_threshold: float = 0.85,
    min_query_coverage: float = 0.2,
) -> list[Evidence]:
    """Rank, compress and budget the retrieved results."""
    results = dedupe(list(results), threshold=dedupe_threshold)
    if not results:
        return []

    qtokens = tokenize(query)
    qset = set(qtokens)
    corpus = [tokenize(f"{r.title} {r.snippet}") for r in results]
    bm25 = BM25(corpus)
    scores = bm25.scores(qtokens)

    max_provider = max((r.score for r in results), default=0.0)
    ranked: list[tuple[float, float, int, SearchResult]] = []
    for i, res in enumerate(results):
        provider_norm = (res.score / max_provider) if max_provider else 0.0
        coverage = (len(qset & set(corpus[i])) / len(qset)) if qset else 1.0
        combined = scores[i] + 0.8 * provider_norm
        ranked.append((combined, coverage, i, res))
    ranked.sort(key=lambda x: -x[0])

    # Drop documents that share almost nothing with the question. This keeps
    # unrelated facts out of the prompt (quality *and* prefill speed).
    relevant = [item for item in ranked if item[1] >= min_query_coverage]
    if not relevant:
        relevant = ranked

    evidence: list[Evidence] = []
    used = 0
    for combined, _, _, res in relevant:
        if len(evidence) >= max_snippets:
            break
        remaining = char_budget - used
        if remaining <= 120:
            break
        limit = min(snippet_char_limit, remaining)
        text = compress_snippet(res.snippet, qtokens, limit)
        if not text:
            continue
        title = (res.title or res.url or "").strip()
        overhead = len(title) + len(res.url) + 24
        if used + len(text) + overhead > char_budget:
            text = compress_snippet(res.snippet, qtokens, max(60, remaining - overhead))
            if not text:
                continue
        used += len(text) + len(title) + len(res.url) + 24
        evidence.append(
            Evidence(
                index=len(evidence) + 1,
                title=title,
                url=res.url,
                text=text,
                score=float(combined),
            )
        )
    return evidence


def evidence_text(evidence: Iterable[Evidence]) -> str:
    return "\n\n".join(e.text for e in evidence)
