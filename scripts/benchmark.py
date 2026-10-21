#!/usr/bin/env python3
"""Run a small query set and write latency/token traces inside the repo.

    python scripts/benchmark.py --provider local
    python scripts/benchmark.py --provider tavily --trace-file trace.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amd_websearch.agent import WebSearchAgent  # noqa: E402
from amd_websearch.config import AppConfig  # noqa: E402

DEFAULT_QUERIES = [
    "How much HBM3 memory does AMD MI300X have?",
    "What is the latest ROCm release version?",
    "Which operating systems does ROCm support?",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", nargs="*", help="queries to run")
    parser.add_argument("--provider", help="tavily | qdrant | local")
    parser.add_argument("--trace-file", default="", help="write JSONL traces here (kept inside the repo)")
    parser.add_argument("--max-rounds", type=int, default=None)
    args = parser.parse_args()

    config = AppConfig.from_env()
    if args.provider:
        config.search.provider = args.provider
    if args.max_rounds is not None:
        config.search.max_rounds = args.max_rounds

    agent = WebSearchAgent(config=config)
    queries = args.queries or DEFAULT_QUERIES

    trace_path = Path(args.trace_file) if args.trace_file else None
    if trace_path and not trace_path.is_absolute():
        trace_path = ROOT / trace_path

    rows = []
    for query in queries:
        payload = agent.run(query)
        trace = {"query": query, **agent.last_trace, "payload": payload}
        rows.append(trace)
        print(
            f"{trace.get('total_ms', 0):>6} ms | "
            f"{trace.get('llm_prompt_tokens', 0):>4} in / "
            f"{trace.get('llm_completion_tokens', 0):>3} out | "
            f"{payload['res']}"
        )

    if trace_path:
        with trace_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[benchmark] trace written to {trace_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
