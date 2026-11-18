# Dar Global & Wasalt Assistant

Agentic RAG chatbot over public data from **darglobal.co.uk** and **wasalt.sa**, powered by a free OpenRouter model, containerised with Docker.

## Architecture

```
scraper/ (offline)            data/kb/                       app/ (runtime, FastAPI)
 browser crawl ──► raw jsonl ──► chunks.jsonl (BM25)  ──►  Agent loop ──► OpenRouter (free model chain)
 build_kb.py                     listings.jsonl (structured)   │  tools: search_knowledge / search_listings / get_page
                                 meta.json                     └─ streamed NDJSON events ──► chat UI (tool trace + [n] citations)
```

* **Retrieval** - dependency-free BM25 (title-boosted, max 2 chunks per page) for passages, plus a structured filter over Wasalt listings (city, type, price, bedrooms).
* **Agent** - every turn is pre-seeded with auto-retrieved context (keeps small free models grounded), then the model may call tools for up to `MAX_STEPS` rounds. Answers cite sources as `[n]`.
* **Model fallback** - `MODEL_CHAIN` (comma-separated) is tried in order on any error/empty reply; default is free models with tool-calling (`google/gemma-4-31b-it:free`, ...).
* **Guardrails** - answers only from the knowledge base, snapshot-date disclosure, no financial advice, per-IP rate limit, key only via env var.

## Run

```bash
docker build -t darglobal-wasalt-bot .
docker run -p 7860:7860 -e OPENROUTER_API_KEY=sk-or-... darglobal-wasalt-bot
# open http://localhost:7860   (health: /api/health)
```

Env vars: `OPENROUTER_API_KEY` (required), `MODEL_CHAIN`, `MAX_STEPS` (default 4), `RATE_LIMIT_PER_10MIN` (default 30).

## Data collection

Both sites sit behind bot protection (Cloudflare / Imperva), so plain HTTP clients get 403 or a CAPTCHA. Pages were fetched from a normal browser session (same-origin `fetch`, ~0.6 s between requests) after a human passed the Imperva check; robots.txt rules and the sitemap were respected. Run `python scraper/build_kb.py` to rebuild `data/kb/` from `data/raw/`.

Limitations: Wasalt is a sampled snapshot (category pages + ~160 listings), not the whole marketplace; data may be stale.
