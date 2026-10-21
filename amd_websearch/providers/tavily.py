"""Tavily hosted web search provider.

Only stdlib `urllib` is used, so the agent adds zero pip dependencies.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .base import SearchProvider, SearchResult, strip_html

TAVILY_ENDPOINT = "https://api.tavily.com/search"


class TavilyProvider(SearchProvider):
    name = "tavily"

    def __init__(self, config) -> None:
        self.api_key = config.api_key
        self.timeout = getattr(config, "request_timeout", 30.0)
        self.search_depth = getattr(config, "search_depth", "basic")
        self.include_raw_content = getattr(config, "include_raw_content", False)
        self.include_answer = getattr(config, "include_answer", False)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(TAVILY_ENDPOINT, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        query = (query or "").strip()
        if not query:
            return []
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max(1, min(int(max_results), 20)),
            "search_depth": self.search_depth,
            "topic": "general",
            "include_answer": bool(self.include_answer),
            "include_raw_content": bool(self.include_raw_content),
            "include_images": False,
        }
        try:
            data = self._post(payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            raise RuntimeError(f"Tavily HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Tavily unreachable: {exc.reason}") from exc

        results: list[SearchResult] = []
        for item in data.get("results", []) or []:
            url = (item.get("url") or "").strip()
            content = strip_html(item.get("content") or item.get("raw_content") or "")
            if not url or not content:
                continue
            results.append(
                SearchResult(
                    title=strip_html(item.get("title") or url)[:180],
                    url=url,
                    snippet=content,
                    score=float(item.get("score") or 0.0),
                    provider=self.name,
                    raw={k: item.get(k) for k in ("title", "url", "published_date") if item.get(k)},
                )
            )
        return results
