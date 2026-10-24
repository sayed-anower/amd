#!/usr/bin/env python3
"""Match 4 batch runner.

Reads questions from ``/data/questions.jsonl`` and writes one answer per line
to ``/output/answers.jsonl``:

    {"id": ..., "res": "... <=45 words ...", "sources": ["https://..."]}

Only the standard library is used. The llama.cpp server is expected to be
warm on ``LLM_BASE_URL`` (run.sh starts it before this process); if it is not
reachable the guardrails still produce a grounded extractive answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.guardrails import MAX_WORDS, enforce_contract  # noqa: E402
from app.rag import AppConfig, build_agent  # noqa: E402

DEFAULT_INPUT = os.environ.get("QUESTIONS_FILE", "/data/questions.jsonl")
DEFAULT_OUTPUT = os.environ.get("ANSWERS_FILE", "/output/answers.jsonl")


def _log(message: str) -> None:
    print(f"[amd-agent] {message}", file=sys.stderr, flush=True)


def _read_questions(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            _log(f"skipping malformed JSON on line {line_number}")
            continue
        if isinstance(item, str):
            item = {"question": item}
        if isinstance(item, dict):
            items.append(item)
    return items


def _question_text(item: dict[str, Any], index: int) -> str:
    for key in ("question", "query", "prompt", "q", "text", "input"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    _log(f"question {index} has no question/query field; skipping")
    return ""


def _question_id(item: dict[str, Any], index: int) -> Any:
    for key in ("id", "qid", "question_id", "uid"):
        if key in item and item[key] is not None:
            return item[key]
    return index


def _answer_one(agent, question: str) -> dict[str, Any]:
    try:
        payload = agent.run(question)
    except Exception as exc:  # noqa: BLE001 - one bad question must not stop the batch
        _log(f"agent error: {type(exc).__name__}: {exc}")
        payload = {"res": "No answer could be grounded in the retrieved web sources.", "sources": []}
    return enforce_contract(payload, MAX_WORDS)


def _iter_outputs(items: Iterable[dict[str, Any]], agent) -> Iterable[dict[str, Any]]:
    for index, item in enumerate(items):
        question = _question_text(item, index)
        if not question:
            continue
        started = time.perf_counter()
        answer = _answer_one(agent, question)
        elapsed = time.perf_counter() - started
        _log(f"q{index} {elapsed:.1f}s words={len(answer['res'].split())} sources={len(answer['sources'])}")
        yield {"id": _question_id(item, index), **answer}


def _wait_for_llm(agent, timeout: float) -> None:
    deadline = time.time() + max(0.0, timeout)
    while time.time() < deadline:
        if agent.warmup():
            return
        time.sleep(1.0)
    _log("llama.cpp server is not healthy; using extractive fallback")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AMD Match 4 web-search batch runner")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="questions JSONL (default /data/questions.jsonl)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="answers JSONL (default /output/answers.jsonl)")
    parser.add_argument("--limit", type=int, default=0, help="only answer the first N questions")
    parser.add_argument("--wait-llm", type=float, default=float(os.environ.get("LLM_STARTUP_TIMEOUT", "60")))
    parser.add_argument("--trace", action="store_true", help="log per-query pipeline traces")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        _log(f"input file not found: {input_path}")
        return 2

    items = _read_questions(input_path)
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        _log("no questions to answer")
        return 0

    config = AppConfig.from_env()
    if config.budget.max_response_words > MAX_WORDS:
        config.budget.max_response_words = MAX_WORDS
    agent = build_agent(config)
    _wait_for_llm(agent, args.wait_llm)
    _log(f"provider={getattr(agent.provider, 'name', '?')} model={config.llm.model} questions={len(items)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for output in _iter_outputs(items, agent):
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            if args.trace and agent.last_trace:
                _log(json.dumps(agent.last_trace, ensure_ascii=False))

    _log(f"wrote {written} answers to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
