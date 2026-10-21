"""HTTP API for the web retrieval agent (stdlib only).

Endpoints
    GET  /health   -> {"status": "ok", "llm": bool, "provider": "tavily"}
    POST /search   -> {"query": "..."}  =>  {"res": "...", "sources": [...]}
    GET  /         -> service info

The agent (and therefore the model) is kept warm between requests, so the
first token comes out in ~a second instead of paying model load time.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .agent import WebSearchAgent
from .config import AppConfig

_LOCK = threading.Lock()  # llama.cpp handles concurrency, but keep traces sane


class Handler(BaseHTTPRequestHandler):
    server_version = "AMDWebSearch/1.0"
    agent: WebSearchAgent | None = None

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter default logging
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/health", ""):
            agent = self.agent
            healthy = bool(agent and agent.warmup())
            self._send(
                200 if healthy else 503,
                {
                    "status": "ok" if healthy else "degraded",
                    "llm": healthy,
                    "provider": getattr(agent.provider, "name", "?") if agent else "?",
                    "model": agent.llm.config.model if agent else "?",
                },
            )
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/search":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
        except (ValueError, json.JSONDecodeError):
            self._send(400, {"error": "invalid json body"})
            return

        query = str(data.get("query") or data.get("q") or "").strip()
        if not query:
            self._send(400, {"error": "missing 'query' field"})
            return

        agent = self.agent
        if agent is None:
            self._send(503, {"error": "agent unavailable"})
            return

        with _LOCK:
            try:
                payload = agent.run(query)
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
        self._send(200, payload)


def serve(agent: WebSearchAgent, host: str = "0.0.0.0", port: int = 8000, verbose: bool = False):
    handler = type("BoundHandler", (Handler,), {"agent": agent})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.verbose = verbose  # type: ignore[attr-defined]
    return httpd


def run_server(config: AppConfig | None = None, host: str = "0.0.0.0", port: int = 8000, verbose: bool = False) -> None:
    config = config or AppConfig.from_env()
    agent = WebSearchAgent(config=config)
    httpd = serve(agent, host=host, port=port, verbose=verbose)
    print(f"[amd-websearch] API on http://{host}:{port}  (provider={agent.provider.name})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[amd-websearch] shutting down", flush=True)
    finally:
        httpd.server_close()
