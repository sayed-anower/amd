"""CLI entry point.

    python -m amd_websearch "what is the latest ROCm release?"
    python -m amd_websearch --pretty --trace "AMD MI300X memory bandwidth"
    python -m amd_websearch --serve --port 8000
    python -m amd_websearch --batch queries.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import WebSearchAgent
from .config import AppConfig
from .llm import LlamaClient
from .providers import build_provider


def _build_agent(args: argparse.Namespace) -> WebSearchAgent:
    config = AppConfig.from_env()
    if args.provider:
        config.search.provider = args.provider
    if args.base_url:
        config.llm.base_url = args.base_url
    if args.model:
        config.llm.model = args.model
    if args.max_rounds is not None:
        config.search.max_rounds = args.max_rounds
    if args.max_snippets is not None:
        config.budget.max_snippets = args.max_snippets
    if args.max_words is not None:
        config.budget.max_response_words = args.max_words

    llm = LlamaClient(config.llm)
    try:
        provider = build_provider(config.search, llm)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        print(
            "[hint] export TAVILY_API_KEY=tv-...  or run with --provider local "
            "(offline demo corpus)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return WebSearchAgent(llm=llm, provider=provider, config=config)


def _emit(payload: dict, pretty: bool) -> None:
    print(json.dumps(payload, indent=2 if pretty else None, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="amd-websearch",
        description="Intelligent Web Information Retrieval agent (llama.cpp + Qwen2.5-0.5B-Instruct).",
    )
    parser.add_argument("query", nargs="*", help="the question to research")
    parser.add_argument("--provider", help="tavily | qdrant | local")
    parser.add_argument("--base-url", help="llama.cpp server base url (default http://127.0.0.1:8099/v1)")
    parser.add_argument("--model", help="model name reported to llama.cpp")
    parser.add_argument("--max-rounds", type=int, help="max search rounds (1 disables the gap round)")
    parser.add_argument("--max-snippets", type=int, help="max evidence snippets")
    parser.add_argument("--max-words", type=int, help="max words in the answer")
    parser.add_argument("--pretty", action="store_true", help="pretty-print the JSON")
    parser.add_argument("--trace", action="store_true", help="print diagnostics to stderr")
    parser.add_argument("--batch", help="file with one query per line")
    parser.add_argument("--interactive", action="store_true", help="repl mode")
    parser.add_argument("--serve", action="store_true", help="run the HTTP API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if args.serve:
        from .server import run_server

        config = AppConfig.from_env()
        if args.provider:
            config.search.provider = args.provider
        if args.base_url:
            config.llm.base_url = args.base_url
        run_server(config, host=args.host, port=args.port)
        return 0

    agent = _build_agent(args)

    if args.batch:
        path = Path(args.batch)
        if not path.is_file():
            print(f"[error] no such file: {path}", file=sys.stderr)
            return 2
        for line in path.read_text(encoding="utf-8").splitlines():
            query = line.strip()
            if not query or query.startswith("#"):
                continue
            payload = agent.run(query)
            print(json.dumps({"query": query, **payload}, ensure_ascii=False))
        return 0

    if args.interactive:
        print("[amd-websearch] type a question, or 'exit' to quit.", flush=True)
        while True:
            try:
                query = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not query:
                continue
            if query.lower() in {"exit", "quit", ":q"}:
                return 0
            payload = agent.run(query)
            _emit(payload, args.pretty)
            print()
        return 0

    query = " ".join(args.query).strip()
    if not query:
        parser.print_help()
        return 1

    payload = agent.run(query)
    _emit(payload, args.pretty)
    if args.trace:
        print(json.dumps(agent.last_trace, indent=2, ensure_ascii=False), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
