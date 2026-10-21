"""Offline corpus provider (BM25 over a local JSONL file).

Two jobs:
  1. demo / CI mode when no Tavily key is available;
  2. reference implementation of the "self-hosted AMD docs" provider that the
     Qdrant provider replaces in production.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..textutil import BM25, tokenize
from .base import SearchProvider, SearchResult, strip_html


class LocalCorpusProvider(SearchProvider):
    name = "local"

    def __init__(self, config) -> None:
        self.corpus_path = Path(getattr(config, "local_corpus_path", "") or "")
        self._docs: list[dict[str, Any]] = []
        self._bm25: BM25 | None = None
        self._load()

    def _load(self) -> None:
        if not self.corpus_path.is_file():
            return
        docs: list[dict[str, Any]] = []
        for line in self.corpus_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            docs.append(
                {
                    "title": strip_html(item.get("title") or ""),
                    "url": item.get("url") or "",
                    "text": strip_html(item.get("text") or item.get("content") or ""),
                    "keywords": item.get("keywords") or [],
                }
            )
        self._docs = docs
        if docs:
            corpus_tokens = [
                tokenize(f"{d['title']} {' '.join(d['keywords'])} {d['text']}") for d in docs
            ]
            self._bm25 = BM25(corpus_tokens)

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        query = (query or "").strip()
        if not query or not self._docs or self._bm25 is None:
            return []
        qtokens = tokenize(query)
        scores = self._bm25.scores(qtokens)
        ranked = sorted(range(len(self._docs)), key=lambda i: scores[i], reverse=True)
        out: list[SearchResult] = []
        for idx in ranked[: max(1, max_results)]:
            if scores[idx] <= 0.0:
                continue
            doc = self._docs[idx]
            out.append(
                SearchResult(
                    title=doc["title"] or doc["url"],
                    url=doc["url"],
                    snippet=doc["text"],
                    score=float(scores[idx]),
                    provider=self.name,
                )
            )
        return out
