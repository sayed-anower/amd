"""Search provider interface + result model."""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    """One retrieved document/snippet."""

    title: str
    url: str
    snippet: str
    score: float = 0.0
    provider: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        return normalize_url(self.url)


_WS = re.compile(r"\s+")
_TAG = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    if not text:
        return ""
    text = _TAG.sub(" ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return _WS.sub(" ", text).strip()


def normalize_url(url: str) -> str:
    """Collapse URLs to a comparable key (scheme/www/trailing-slash insensitive)."""
    if not url:
        return ""
    url = url.strip().lower()
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^www\.", "", url)
    return url.rstrip("/")


class SearchProvider(abc.ABC):
    """Every backend implements this, so the agent stays backend agnostic."""

    name: str = "base"

    @abc.abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Return ranked results for `query`. Must never raise for empty input."""

    def close(self) -> None:  # pragma: no cover - providers may not need cleanup
        return None
