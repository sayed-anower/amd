# AMD Web Search Agent - Match 4

Intelligent Web Information Retrieval for the **AMD AI League - Match 4**.

The agent takes a natural-language question, finds the right web pages, reads the
retrieved text with a **local Qwen2.5-0.5B-Instruct** model running on
**llama.cpp**, and returns a short, grounded JSON answer:

```json
{"res": "The AMD Instinct MI300X has 192 GB of HBM3 memory with 5.3 TB/s bandwidth.", "sources": ["https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html"]}
```

`res` is capped at **45 words** and `sources` always contains real URLs from the
retrieved pages, because sources are mandatory for Match 4. Everything is
standard-library Python (no `pip install`) and runs on CPU or on AMD ROCm.

## Why this design

- **Short answers, low token burn.** Evidence is deduplicated, relevance-filtered,
  compressed to ~1k characters, and the model is limited to ~96 output tokens.
- **No invented facts.** The model only sees retrieved evidence. A grounding
  guardrail drops sentences that contain numbers or names not present in that
  evidence, and an extractive fallback takes over if the model output is weak.
- **No invented links.** Sources are validated against the exact URLs in the
  evidence; hallucinated URLs are discarded and replaced with the pages actually
  read.
- **Fast by default.** For simple lookups the planner model call is skipped, the
  second search round is skipped when the evidence already covers the question,
  and one llama.cpp server is kept warm between requests.

> Search provider is pluggable. For AMD internal use, replace Tavily with self-hosted Qdrant vector DB of AMD docs - zero external cost, zero dependency, 100% runs on MI300X.

## Pipeline

```text
question
   |
   v
[planner: Qwen2.5-0.5B] --(skipped for simple lookups)--> 1-2 web queries
   |
   v
[search provider]  Tavily (demo) | Qdrant AMD docs (internal) | local JSONL (offline)
   |
   v
[evidence selection]  dedupe -> relevance filter -> BM25 rank -> sentence compression
   |
   v
[optional gap round]  only when evidence coverage is below 0.6
   |
   v
[synthesizer: Qwen2.5-0.5B]  JSON-schema constrained decoding -> {"res", "sources"}
   |
   v
[guardrails]  JSON repair -> grounding -> source validation -> 45-word cap
```

## Quick start (Google Colab / local)

```bash
cd /content/amd

# 1. Configure search (Tavily demo key)
cp .env.example .env
# edit .env and set TAVILY_API_KEY=tvly-...

# 2. Start llama.cpp with the bundled Qwen2.5-0.5B-Instruct GGUF
bash scripts/start_llama.sh

# 3. Ask a question
python -m amd_websearch --pretty "How much HBM3 memory does AMD MI300X have?"
```

No Tavily key? The demo corpus works offline:

```bash
python -m amd_websearch --provider local --pretty "What is the latest ROCm release version?"
```

`start_llama.sh` writes `llama-server.log` and `llama-server.pid` **inside this
folder**. Nothing is written to `/tmp`.

## HTTP API

```bash
bash scripts/start_api.sh

curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Which operating systems does ROCm support?"}'
```

Response:

```json
{
  "res": "ROCm supports Linux operating systems and AMD Radeon and Instinct GPUs.",
  "sources": ["https://rocm.docs.amd.com/en/latest/about.html"]
}
```

`server.log` and `server.pid` stay inside the project directory. Stop both
services with `bash scripts/stop.sh`.

## CLI

```bash
python -m amd_websearch "QUESTION"                 # JSON output
python -m amd_websearch --pretty --trace "QUESTION" # pretty JSON + stderr trace
python -m amd_websearch --interactive               # REPL
python -m amd_websearch --batch queries.txt         # one query per line
python -m amd_websearch --provider local "QUESTION" # offline corpus
```

## Configuration

All settings are environment variables, documented in `.env.example`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SEARCH_PROVIDER` | `tavily` | `tavily`, `qdrant`, or `local` |
| `TAVILY_API_KEY` | - | Tavily demo key |
| `LLM_BASE_URL` | `http://127.0.0.1:8099/v1` | llama.cpp OpenAI-compatible endpoint |
| `PLANNER_MODE` | `auto` | `auto` skips the planner for short lookups, `always`, `off` |
| `MAX_SNIPPETS` | `3` | evidence documents shown to the model |
| `EVIDENCE_CHAR_BUDGET` | `1000` | hard cap on evidence characters |
| `SNIPPET_CHAR_LIMIT` | `280` | characters kept per document |
| `MAX_RESPONSE_WORDS` | `45` | answer length cap |
| `MAX_SOURCES` | `3` | URLs returned |
| `ANSWER_MAX_TOKENS` | `80` | output-token cap |
| `MAX_ROUNDS` | `2` | search rounds; round two is coverage-gated |
| `GAP_MIN_COVERAGE` | `0.6` | skip the gap round above this keyword coverage |

Speed/quality examples:

```bash
# Fastest: no planner, one search round
PLANNER_MODE=off MAX_ROUNDS=1 python -m amd_websearch "QUESTION"

# Best recall: force planning and a second round
PLANNER_MODE=always MAX_ROUNDS=2 MAX_SNIPPETS=6 EVIDENCE_CHAR_BUDGET=2200 \
  python -m amd_websearch "QUESTION"
```

## Search providers

The agent only talks to the `SearchProvider` interface (`amd_websearch/providers/base.py`),
so swapping backends does not touch the reasoning code.

| Provider | Use case |
| --- | --- |
| `tavily` | Live web search for this public demo |
| `qdrant` | Self-hosted AMD docs vector DB over HTTP, no client dependency |
| `local` | Offline JSONL corpus with BM25, used for tests and demos |

### AMD internal / MI300X deployment

The Qdrant provider needs no external service beyond your own infrastructure:

```bash
SEARCH_PROVIDER=qdrant \
QDRANT_URL=http://127.0.0.1:6333 \
QDRANT_COLLECTION=amd_docs \
QDRANT_EMBED_URL=http://127.0.0.1:8099/v1/embeddings \
QDRANT_EMBED_MODEL=nomic-embed-text \
python -m amd_websearch "QUESTION"
```

Run llama.cpp with ROCm on MI300X (`-ngl 99`) and point `LLM_BASE_URL` at it.
`corpus/amd_docs.jsonl` shows the document shape expected by the local provider
and can be used to seed the Qdrant collection.

## Logs and traces

Everything remains in `/content/amd/`:

| File | Written by |
| --- | --- |
| `llama-server.log`, `llama-server.pid` | `scripts/start_llama.sh` |
| `server.log`, `server.pid` | `scripts/start_api.sh` |
| `trace.txt` | `python scripts/benchmark.py --trace-file trace.txt` |

## Measured footprint

On the free Colab **2-vCPU CPU runtime** with the bundled Q4_K_M model, the
defaults use roughly **480-530 prompt tokens and 80 output tokens** per query
and finish in about **6-18 seconds** (simple lookup vs. planned query). On an
MI300X with the llama.cpp ROCm build this prefill/generation cost is negligible,
and the token caps keep cost low regardless of hardware.

## Tests and benchmark

```bash
python -m unittest discover -s tests -v
python scripts/benchmark.py --provider local --trace-file trace.txt
```

The unit tests need no network and no llama.cpp server. They cover JSON repair,
grounding, hallucinated numbers/URLs, evidence filtering, budgets, planner
gating and Tavily response parsing.

## Output contract

- Always a single JSON object with exactly `res` and `sources`.
- `res`: plain text, at most 45 words, grounded only in retrieved evidence.
- `sources`: 1-3 HTTP(S) URLs copied verbatim from the evidence.
- If nothing can be grounded, the agent says so briefly and still returns the
  URLs it actually read (or an empty list when no source was retrieved).
