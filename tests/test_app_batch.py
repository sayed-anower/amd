"""Batch-runner contract tests: no model, no network."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app import main as app_main


class FakeAgent:
    provider = type("P", (), {"name": "local"})()

    def __init__(self, payload):
        self.payload = payload
        self.last_trace = {}

    def run(self, query):
        return dict(self.payload)

    def warmup(self):
        return True


class BatchRunnerTests(unittest.TestCase):
    def _run(self, payload, questions):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "questions.jsonl"
            dst = Path(tmp) / "answers.jsonl"
            src.write_text("\n".join(json.dumps(q) for q in questions) + "\n", encoding="utf-8")
            original = app_main.build_agent
            app_main.build_agent = lambda config: FakeAgent(payload)
            try:
                code = app_main.main(["--input", str(src), "--output", str(dst), "--wait-llm", "0"])
            finally:
                app_main.build_agent = original
            self.assertEqual(code, 0)
            return [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines()]

    def test_ids_are_preserved_and_word_cap_is_hard(self):
        long_answer = " ".join(f"word{i}" for i in range(80))
        payload = {
            "res": long_answer,
            "sources": ["https://a.example/x", "ftp://bad.example/y", "https://a.example/x"],
        }
        out = self._run(payload, [{"id": "q-1", "question": "a?"}, {"id": 2, "query": "b?"}])
        self.assertEqual([row["id"] for row in out], ["q-1", 2])
        for row in out:
            self.assertEqual(set(row), {"id", "res", "sources"})
            self.assertLessEqual(len(row["res"].split()), 45)
            self.assertEqual(row["sources"], ["https://a.example/x"])

    def test_missing_question_field_does_not_emit_broken_line(self):
        out = self._run({"res": "ok", "sources": []}, [{"id": 1}, {"id": 2, "question": "real?"}])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["id"], 2)


if __name__ == "__main__":
    unittest.main()
