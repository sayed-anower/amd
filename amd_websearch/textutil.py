"""Text utilities: tokenization, BM25 scoring, sentence handling.

Pure stdlib on purpose - the whole agent runs with zero pip dependencies.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to", "in", "on", "at",
    "by", "for", "with", "about", "into", "from", "up", "down", "out", "over", "under", "is",
    "are", "was", "were", "be", "been", "being", "do", "does", "did", "doing", "have", "has",
    "had", "i", "you", "he", "she", "it", "we", "they", "them", "his", "her", "its", "our",
    "your", "my", "me", "us", "this", "that", "these", "those", "as", "so", "than", "too",
    "very", "can", "will", "just", "should", "now", "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "not", "only", "own", "same", "s", "t", "don", "ll", "re", "ve",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-_/+#][a-z0-9]+)*")
_SENT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")
_WORD_NUM_RE = re.compile(r"\b\d[\d,.]*\b")
_PROPER_RE = re.compile(r"\b[A-Z][A-Za-z0-9]{2,}(?:\s+[A-Z][A-Za-z0-9]{2,})*\b")


def stem(word: str) -> str:
    """Very light suffix stripping - enough for BM25 without a stemmer dep."""
    w = word
    for suffix in ("ations", "ation", "ingly", "edly", "ing", "ies", "ied", "es", "ed", "ly", "s"):
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            w = w[: -len(suffix)]
            break
    if w.endswith("e") and len(w) > 4:
        w = w[:-1]
    return w


def tokenize(text: str, keep_stopwords: bool = False, do_stem: bool = True) -> list[str]:
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    out: list[str] = []
    for tok in tokens:
        if not keep_stopwords and tok in STOPWORDS:
            continue
        out.append(stem(tok) if do_stem else tok)
    return out


def content_tokens(text: str) -> list[str]:
    """Tokens that carry meaning - used by the grounding check."""
    return tokenize(text)


def sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = [p.strip() for p in _SENT_RE.split(text) if p and p.strip()]
    return [p for p in parts if len(p) > 1]


def numeric_facts(text: str) -> set[str]:
    """Numbers/versions - these must be copied verbatim from the evidence."""
    return {m.group(0).strip(".,") for m in _WORD_NUM_RE.finditer(text or "")}


def proper_nouns(text: str) -> set[str]:
    return {m.group(0).strip() for m in _PROPER_RE.finditer(text or "")}


class BM25:
    """Small BM25 implementation (k1=1.5, b=0.75)."""

    def __init__(self, documents: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = [list(doc) for doc in documents]
        self.n = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / self.n) if self.n else 0.0
        self.tf: list[Counter[str]] = [Counter(d) for d in self.docs]
        df: Counter[str] = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        self.idf: dict[str, float] = {
            term: math.log(1.0 + (self.n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def scores(self, query_tokens: Iterable[str]) -> list[float]:
        q = list(query_tokens)
        out: list[float] = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i]) or 1
            score = 0.0
            for term in q:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += idf * (freq * (self.k1 + 1)) / (denom or 1)
            out.append(score)
        return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def jaccard_char(a: str, b: str) -> float:
    """Cheap near-duplicate check for snippets (character 3-grams)."""
    def grams(text: str) -> set[str]:
        text = re.sub(r"\s+", " ", text.lower())
        return {text[i : i + 3] for i in range(max(0, len(text) - 2))}

    return jaccard(grams(a), grams(b))
