"""Offline tests: no network, no llama.cpp server required."""

from __future__ import annotations

import json
import unittest

from amd_websearch.agent import WebSearchAgent
from amd_websearch.config import AppConfig, BudgetConfig, LlmConfig, SearchConfig
from amd_websearch.evidence import build_evidence
from amd_websearch.guardrails import build_answer
from amd_websearch.llm import LlmResponse
from amd_websearch.providers.base import SearchProvider, SearchResult
from amd_websearch.providers.tavily import TavilyProvider

SYSTEM = "Answer from evidence. Return JSON {res, sources}."
PLANNER = "Write search queries. Return JSON {queries}."


class FakeProvider(SearchProvider):
    name = "fake"

    def __init__(self, results):
        self.results = results
        self.queries = []

    def search(self, query, max_results=5):
        self.queries.append(query)
        return list(self.results)


class BrokenLlm:
    def chat(self, *args, **kwargs):
        from amd_websearch.llm import LlmError

        raise LlmError("offline")

    def health(self):
        return False


class FakeLlm:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, max_tokens=96, temperature=None, stop=None, json_schema=None, seed=None):
        self.calls.append({"messages": messages, "schema": json_schema})
        if json_schema and "queries" in json_schema.get("properties", {}):
            text = self._next('{"queries": ["direct query"]}')
        elif json_schema and "query" in json_schema.get("properties", {}):
            text = self._next('{"query": ""}')
        else:
            text = self._next('{"res": "No answer found.", "sources": []}')
        return LlmResponse(text=text, prompt_tokens=100, completion_tokens=20, latency_ms=1)

    def _next(self, default):
        return self.replies.pop(0) if self.replies else default

    def health(self):
        return True


def make_agent(replies, results, max_rounds=1, **overrides):
    config = AppConfig(
        llm=LlmConfig(),
        search=SearchConfig(provider="fake", max_rounds=max_rounds),
        budget=BudgetConfig(),
    )
    for key, value in overrides.items():
        setattr(config.budget, key, value)
    provider = FakeProvider(results)
    llm = FakeLlm(replies)
    agent = WebSearchAgent(
        llm=llm,
        provider=provider,
        config=config,
        system_prompt=SYSTEM,
        planner_prompt=PLANNER,
    )
    return agent, provider, llm


def doc(title, url, snippet, score=1.0):
    return SearchResult(title=title, url=url, snippet=snippet, score=score, provider="fake")


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.results = [
            doc(
                "AMD Instinct MI300X specifications",
                "https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html",
                "The AMD Instinct MI300X accelerator is built on CDNA 3. It has 192 GB of HBM3 memory "
                "with a memory bandwidth of 5.3 TB/s and 1307 TFLOPS of FP16 compute.",
            ),
            doc(
                "ROCm overview",
                "https://rocm.docs.amd.com/en/latest/about.html",
                "ROCm is a platform for running applications on AMD GPUs, including HIP and profiling tools.",
                score=0.4,
            ),
        ]

    def test_short_query_skips_planner_and_gap(self):
        agent, provider, llm = make_agent(
            ['{"res": "MI300X has 192 GB of HBM3 memory.", "sources": ["https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html"]}'],
            self.results,
            max_rounds=2,
        )
        payload = agent.run("MI300X HBM3 memory")
        self.assertEqual(provider.queries, ["MI300X HBM3 memory"])
        self.assertEqual(len(llm.calls), 1)
        self.assertIn("192 GB", payload["res"])
        self.assertEqual(
            payload["sources"],
            ["https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html"],
        )
        self.assertLessEqual(len(payload["res"].split()), 45)

    def test_invented_url_is_replaced_with_real_evidence(self):
        agent, _, _ = make_agent(
            ['{"res": "MI300X has 192 GB of HBM3 memory.", "sources": ["https://evil.example/fake"]}'],
            self.results,
        )
        payload = agent.run("MI300X HBM3 memory")
        self.assertEqual(
            payload["sources"],
            ["https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html"],
        )

    def test_hallucinated_number_is_dropped(self):
        agent, _, _ = make_agent(
            ['{"res": "The AMD Instinct MI300X has 256 GB of HBM3 memory.", "sources": []}'],
            self.results,
        )
        payload = agent.run("How much HBM3 memory does the MI300X have?")
        self.assertNotIn("256", payload["res"])
        self.assertTrue(payload["sources"])

    def test_numeric_intent_drops_off_topic_sentences(self):
        results = [
            doc(
                "MI300X specifications",
                "https://a.example/mi300x",
                "The AMD Instinct MI300X is built on CDNA 3. It has 192 GB of HBM3 memory. "
                "AMD Developer Cloud provides access to MI300X GPUs. ROCm 6.3 supports Ubuntu 22.04.",
            )
        ]
        agent, _, _ = make_agent(
            ['{"res": "The AMD Instinct MI300X is built on CDNA 3. It has 192 GB of HBM3 memory. AMD Developer Cloud provides access to MI300X GPUs. ROCm 6.3 supports Ubuntu 22.04.", "sources": []}'],
            results,
        )
        payload = agent.run("How much HBM3 memory does the MI300X have?")
        self.assertIn("192 GB", payload["res"])
        self.assertNotIn("Developer Cloud", payload["res"])
        self.assertNotIn("Ubuntu", payload["res"])

    def test_hallucinated_name_is_dropped(self):
        agent, _, _ = make_agent(
            ['{"res": "The MI400X has 192 GB of HBM3 memory.", "sources": []}'],
            self.results,
        )
        payload = agent.run("How much HBM3 memory does the MI300X have?")
        self.assertNotIn("MI400X", payload["res"])
        self.assertTrue(payload["sources"])

    def test_markdown_is_stripped(self):
        agent, _, _ = make_agent(
            ['{"res": "**MI300X** has `192 GB` of HBM3 memory.", "sources": []}'],
            self.results,
        )
        payload = agent.run("MI300X HBM3 memory")
        self.assertNotIn("*", payload["res"])
        self.assertNotIn("`", payload["res"])

    def test_complex_query_uses_planner(self):
        agent, provider, llm = make_agent(
            [
                '{"queries": ["MI300X HBM3 memory", "AMD MI300X bandwidth"]}',
                '{"res": "MI300X has 192 GB of HBM3 memory.", "sources": []}',
            ],
            self.results,
        )
        payload = agent.run("Compare MI300X HBM3 memory and bandwidth with the previous generation")
        self.assertEqual(len(provider.queries), 2)
        self.assertIn("192 GB", payload["res"])
        self.assertTrue(payload["sources"])

    def test_extractive_fallback_when_llm_is_down(self):
        results = [
            doc(
                "MI300X specifications",
                "https://a.example/mi300x",
                "The AMD Instinct MI300X has 192 GB of HBM3 memory and 5.3 TB/s bandwidth.",
            )
        ]
        agent, _, _ = make_agent([], results)
        agent.llm = BrokenLlm()
        payload = agent.run("MI300X HBM3 memory")
        self.assertIn("192 GB", payload["res"])
        self.assertEqual(payload["sources"], ["https://a.example/mi300x"])

    def test_empty_search_returns_no_sources(self):
        agent, _, _ = make_agent([], [])
        agent.config.provider_fallback = ""
        payload = agent.run("What is the latest ROCm version?")
        self.assertEqual(payload["sources"], [])
        self.assertIn("No web sources", payload["res"])


class EvidenceTests(unittest.TestCase):
    def test_irrelevant_documents_are_filtered(self):
        results = [
            doc("MI300X memory", "https://a.example/mi300x", "MI300X has 192 GB of HBM3 memory."),
            doc("Cooking guide", "https://b.example/cooking", "Boil water and add pasta for ten minutes."),
        ]
        evidence = build_evidence("MI300X HBM3 memory", results, max_snippets=4, char_budget=1500)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].url, "https://a.example/mi300x")

    def test_char_budget_is_respected(self):
        many = [
            doc(f"MI300X memory {i}", f"https://a.example/{i}", "MI300X HBM3 memory " * 40)
            for i in range(8)
        ]
        evidence = build_evidence("MI300X HBM3 memory", many, max_snippets=6, char_budget=700, snippet_char_limit=300)
        total = sum(len(e.text) + len(e.title) + len(e.url) + 24 for e in evidence)
        self.assertLessEqual(total, 700)


class GuardrailTests(unittest.TestCase):
    def test_sources_are_limited_and_verbatim(self):
        evidence = build_evidence(
            "MI300X memory",
            [doc(f"doc {i}", f"https://a.example/{i}", "MI300X has 192 GB HBM3 memory.") for i in range(5)],
            max_snippets=5,
        )
        payload = build_answer(
            json.dumps({"res": "MI300X has 192 GB HBM3 memory.", "sources": [e.url for e in evidence]}),
            "MI300X memory",
            evidence,
            max_sources=2,
        )
        self.assertLessEqual(len(payload["sources"]), 2)
        self.assertTrue(all(s in {e.url for e in evidence} for s in payload["sources"]))


    def test_sources_are_ranked_by_support_for_the_answer(self):
        results = [
            doc("MI300X memory", "https://a.example/mi300x", "MI300X has 192 GB of HBM3 memory."),
            doc("ROCm Linux support", "https://b.example/rocm", "ROCm supports Linux on AMD GPUs."),
        ]
        evidence = build_evidence("MI300X HBM3 memory", results)
        payload = build_answer(
            json.dumps({
                "res": "MI300X has 192 GB of HBM3 memory.",
                "sources": ["https://b.example/rocm", "https://a.example/mi300x"],
            }),
            "MI300X HBM3 memory",
            evidence,
        )
        self.assertEqual(payload["sources"][0], "https://a.example/mi300x")

    def test_wrong_model_source_is_replaced_by_the_page_with_the_fact(self):
        results = [
            doc("MI300X memory", "https://a.example/mi300x", "MI300X has 192 GB of HBM3 memory."),
            doc("ROCm Linux support", "https://b.example/rocm", "ROCm supports Linux on AMD GPUs."),
        ]
        evidence = build_evidence("MI300X HBM3 memory", results)
        payload = build_answer(
            json.dumps({
                "res": "MI300X has 192 GB of HBM3 memory.",
                "sources": ["https://b.example/rocm"],
            }),
            "MI300X HBM3 memory",
            evidence,
        )
        self.assertEqual(payload["sources"], ["https://a.example/mi300x"])


class TavilyParsingTests(unittest.TestCase):
    def test_response_is_parsed_into_results(self):
        config = type("C", (), {
            "api_key": "test",
            "request_timeout": 5,
            "search_depth": "basic",
            "include_raw_content": False,
            "include_answer": False,
        })()
        provider = TavilyProvider(config)
        provider._post = lambda payload: {
            "results": [
                {
                    "title": "AMD MI300X",
                    "url": "https://www.amd.com/mi300x",
                    "content": "<b>192 GB</b> of HBM3 memory.",
                    "score": 0.9,
                },
                {"title": "no url", "url": "", "content": "ignored"},
            ]
        }
        results = provider.search("MI300X memory")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://www.amd.com/mi300x")
        self.assertEqual(results[0].snippet, "192 GB of HBM3 memory.")


if __name__ == "__main__":
    unittest.main()
