"""The retrieval agent.

    query
      -> planner (llama.cpp / Qwen2.5-0.5B-Instruct)  : search queries
      -> search provider (Tavily / Qdrant / local)      : raw web results
      -> evidence selection                            : dedupe + rank + compress
      -> gap analysis (optional second round)          : one follow-up query
      -> synthesizer (llama.cpp)                       : {"res", "sources"}
      -> guardrails                                    : JSON repair + grounding
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from .config import AppConfig
from .evidence import Evidence, build_evidence
from .guardrails import ANSWER_SCHEMA, build_answer, extract_json_object
from .llm import LlamaClient, LlmError, LlmResponse
from .prompts import GAP_SCHEMA, PLANNER_SCHEMA, load_prompt
from .providers import build_provider
from .providers.base import SearchResult
from .textutil import tokenize

_LINE_CLEAN_RE = re.compile(r"^[\s\-\*\d\.\)\#\(\[\"']+|[\"'\[\]]+$")
_COMPLEX_MARKERS = (
    " and ",
    " vs ",
    " versus ",
    " compare ",
    " difference ",
    " both ",
    " also ",
    " plus ",
    " or ",
)


class WebSearchAgent:
    """Fast, token-frugal web retrieval agent."""

    def __init__(
        self,
        llm: LlamaClient | None = None,
        provider=None,
        config: AppConfig | None = None,
        system_prompt: str | None = None,
        planner_prompt: str | None = None,
    ) -> None:
        self.config = config or AppConfig.from_env()
        self.llm = llm or LlamaClient(self.config.llm)
        self.provider = provider if provider is not None else build_provider(self.config.search, self.llm)
        self.system_prompt = system_prompt or load_prompt(
            self.config.system_prompt_path, ""
        )
        self.planner_prompt = planner_prompt or load_prompt(self.config.planner_prompt_path, "")
        self.last_trace: dict[str, Any] = {}

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _clean_query(text: str) -> str:
        text = _LINE_CLEAN_RE.sub("", (text or "").strip())
        text = re.sub(r"\s+", " ", text).strip()
        return text[:120]

    def _llm_json(self, messages: list[dict[str, str]], schema: dict[str, Any], max_tokens: int) -> dict[str, Any]:
        resp = self.llm.chat(messages, max_tokens=max_tokens, json_schema=schema)
        obj = extract_json_object(resp.text)
        return obj if isinstance(obj, dict) else {}

    # ---------------------------------------------------------------- planner
    def _needs_planner(self, query: str) -> bool:
        """Skip the planner LLM call for simple lookups; it is the slowest
        optional step and a direct query usually works better anyway."""
        mode = (self.config.search.planner_mode or "auto").lower()
        if mode == "off":
            return False
        if mode == "always":
            return True
        words = query.split()
        if len(words) > self.config.search.planner_short_query_words:
            return True
        if query.count("?") > 1 or "\n" in query:
            return True
        low = f" {query.lower()} "
        return any(marker in low for marker in _COMPLEX_MARKERS)

    def plan_queries(self, query: str) -> list[str]:
        """Ask the model which web queries to run (JSON-schema constrained)."""
        if not self._needs_planner(query):
            cleaned = self._clean_query(query)
            return [cleaned] if cleaned else []

        messages = [
            {"role": "system", "content": self.planner_prompt},
            {"role": "user", "content": f"QUESTION: {query}\nQUERIES:"},
        ]
        try:
            obj = self._llm_json(messages, PLANNER_SCHEMA, self.config.budget.planner_max_tokens)
        except LlmError:
            obj = {}

        queries: list[str] = []
        for raw in obj.get("queries") or []:
            if not isinstance(raw, str):
                continue
            cleaned = self._clean_query(raw)
            if len(cleaned) >= 3 and cleaned.lower() not in {q.lower() for q in queries}:
                queries.append(cleaned)
            if len(queries) >= self.config.search.max_sub_queries:
                break

        if not queries:
            queries = [self._clean_query(query)] or ["web search"]
        return queries

    def gap_query(self, query: str, evidence: Sequence[Evidence]) -> str | None:
        """Ask for one follow-up query when the evidence looks incomplete."""
        if self.config.search.max_rounds <= 1:
            return None
        if self._evidence_coverage(query, evidence) >= self.config.budget.gap_min_coverage:
            return None
        digest = "\n".join(
            f"[{e.index}] {e.title} | {e.url} | {e.text[:110]}" for e in evidence[:4]
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the gap analyst of a web retrieval agent. You see the web "
                    "results found so far. If they do not contain the answer, output one "
                    "extra search query to find the missing information. If they already "
                    "contain the answer, output an empty string. Output only the JSON "
                    'object {"query": "<query or empty>"}.'
                ),
            },
            {"role": "user", "content": f"QUESTION: {query}\nFOUND SO FAR:\n{digest}\nQUERY:"},
        ]
        try:
            obj = self._llm_json(messages, GAP_SCHEMA, self.config.budget.gap_max_tokens)
        except LlmError:
            return None
        gap = self._clean_query(str(obj.get("query") or ""))
        if not gap or gap.lower() in {"none", "no", "null", "nothing"}:
            return None
        return gap

    # ----------------------------------------------------------------- search
    @staticmethod
    def _evidence_coverage(query: str, evidence: Sequence[Evidence]) -> float:
        """Fraction of query keywords already present in the evidence."""
        q = set(tokenize(query))
        if not q:
            return 1.0
        found: set[str] = set()
        for e in evidence:
            found.update(tokenize(f"{e.title} {e.text}"))
        return len(q & found) / len(q)

    def search_many(self, queries: Sequence[str]) -> tuple[list[SearchResult], list[str]]:
        """Run the queries in parallel; returns (results, errors)."""
        queries = [q for q in queries if q]
        if not queries:
            return [], []

        results: list[SearchResult] = []
        errors: list[str] = []

        def _run(q: str) -> tuple[str, list[SearchResult] | None, str | None]:
            try:
                return q, self.provider.search(q, self.config.search.max_results_per_query), None
            except Exception as exc:  # noqa: BLE001 - provider failures must not kill the run
                return q, None, f"{type(exc).__name__}: {exc}"

        workers = max(1, min(self.config.search.max_parallel_searches, len(queries)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _, res, err in pool.map(_run, queries):
                if err:
                    errors.append(err)
                elif res:
                    results.extend(res)

        if not results:
            results, fallback_errors = self._fallback_search(queries)
            errors.extend(fallback_errors)
        return results, errors

    def _fallback_search(self, queries: Sequence[str]) -> tuple[list[SearchResult], list[str]]:
        """Try the configured fallback provider when the main one fails."""
        fallback_name = (self.config.provider_fallback or "").lower()
        if not fallback_name or fallback_name == getattr(self.provider, "name", "").lower():
            return [], []
        try:
            provider = build_provider(self._fallback_config(fallback_name), self.llm)
            out: list[SearchResult] = []
            for q in queries:
                try:
                    out.extend(provider.search(q, self.config.search.max_results_per_query))
                except Exception:  # noqa: BLE001
                    continue
            return out, [f"fallback provider '{fallback_name}' used"]
        except Exception as exc:  # noqa: BLE001
            return [], [f"fallback failed: {exc}"]

    def _fallback_config(self, name: str):
        import copy

        cfg = copy.deepcopy(self.config.search)
        cfg.provider = name
        return cfg

    # ------------------------------------------------------------- synthesizer
    def synthesize(self, query: str, evidence: Sequence[Evidence]) -> LlmResponse:
        cards = "\n\n".join(e.as_card() for e in evidence)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"QUESTION: {query}\n\n"
                    f"EVIDENCE:\n{cards}\n\n"
                    "Return the JSON object now."
                ),
            },
        ]
        return self.llm.chat(
            messages,
            max_tokens=self.config.budget.answer_max_tokens,
            json_schema=ANSWER_SCHEMA,
        )

    # -------------------------------------------------------------------- run
    def run(self, query: str) -> dict[str, Any]:
        """Full pipeline. Returns the final `{"res", "sources"}` payload."""
        started = time.perf_counter()
        query = (query or "").strip()
        trace: dict[str, Any] = {"query": query, "provider": getattr(self.provider, "name", "?")}

        if not query:
            return {
                "res": "Please provide a question to look up on the web.",
                "sources": [],
            }

        # 1. plan
        t0 = time.perf_counter()
        queries = self.plan_queries(query)
        trace["queries"] = queries
        trace["plan_ms"] = int((time.perf_counter() - t0) * 1000)

        # 2. search
        t0 = time.perf_counter()
        results, errors = self.search_many(queries)
        trace["raw_results"] = len(results)
        trace["search_ms"] = int((time.perf_counter() - t0) * 1000)
        if errors:
            trace["errors"] = errors

        # 3. evidence
        t0 = time.perf_counter()
        evidence = build_evidence(
            query,
            results,
            max_snippets=self.config.budget.max_snippets,
            char_budget=self.config.budget.evidence_char_budget,
            snippet_char_limit=self.config.budget.snippet_char_limit,
            dedupe_threshold=self.config.budget.dedupe_jaccard,
            min_query_coverage=self.config.budget.min_result_coverage,
        )
        trace["evidence_ms"] = int((time.perf_counter() - t0) * 1000)
        trace["evidence"] = [
            {"i": e.index, "title": e.title, "url": e.url, "chars": len(e.text)} for e in evidence
        ]

        # 4. optional second round
        if evidence and self.config.search.max_rounds > 1:
            gap = self.gap_query(query, evidence)
            trace["gap_query"] = gap
            if gap and gap.lower() not in {q.lower() for q in queries}:
                more, more_errors = self.search_many([gap])
                if more:
                    merged = list(results) + more
                    evidence = build_evidence(
                        query,
                        merged,
                        max_snippets=self.config.budget.max_snippets,
                        char_budget=self.config.budget.evidence_char_budget,
                        snippet_char_limit=self.config.budget.snippet_char_limit,
                        dedupe_threshold=self.config.budget.dedupe_jaccard,
                        min_query_coverage=self.config.budget.min_result_coverage,
                    )
                    trace["raw_results"] = len(merged)
                if more_errors:
                    trace.setdefault("errors", []).extend(more_errors)

        # 5. synthesize
        t0 = time.perf_counter()
        if evidence:
            try:
                resp = self.synthesize(query, evidence)
                trace["llm_prompt_tokens"] = resp.prompt_tokens
                trace["llm_completion_tokens"] = resp.completion_tokens
                trace["llm_ms"] = resp.latency_ms
                raw_output = resp.text
            except LlmError as exc:
                trace.setdefault("errors", []).append(f"llm: {exc}")
                raw_output = ""
        else:
            raw_output = ""
        trace["synth_ms"] = int((time.perf_counter() - t0) * 1000)

        # 6. guardrails
        payload = build_answer(
            raw_output,
            query,
            evidence,
            max_words=self.config.budget.max_response_words,
            coverage_threshold=self.config.budget.ground_coverage_threshold,
            max_sources=self.config.budget.max_sources,
        )

        if not evidence:
            payload = {
                "res": "No web sources were retrieved for this question, so there is nothing to ground an answer in.",
                "sources": [],
            }

        trace["total_ms"] = int((time.perf_counter() - started) * 1000)
        self.last_trace = trace
        return payload

    # convenience ---------------------------------------------------------
    def warmup(self) -> bool:
        return self.llm.health()
