"""Planner-executor agent optimised for latency (target: first token < 2 s, full answer ~2-5 s).

Per turn:
  1. Plan      - deterministic intent router (code, ~1 ms): knowledge question vs. structured listing search,
                 extracting city / property type / price / bedrooms filters.
  2. Execute   - retrieval tools run locally in the same step (BM25 passages + listing filter + optional page read).
  3. Synthesize- ONE streamed LLM call grounded on the tool results; tokens are forwarded to the UI immediately.
Fallbacks: ordered free-model chain (fast models first, per-model first-token timeout), LRU answer cache.
"""
from __future__ import annotations
import json, os, re, time
from collections import OrderedDict
from typing import AsyncIterator
import httpx
from .kb import KnowledgeBase

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_CHAIN = [m.strip() for m in os.getenv(
    "MODEL_CHAIN",
    "nvidia/nemotron-3.5-lightning:free,nvidia/nemotron-3-ultra-550b-a55b:free,inclusionai/ling-3.2-flash:free,"
    "google/gemma-4-31b-it:free,openrouter/free",
).split(",") if m.strip()]
FIRST_TOKEN_TIMEOUT = float(os.getenv("FIRST_TOKEN_TIMEOUT", "8"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "700"))
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "200"))

SYSTEM = """You are the Dar Global & Wasalt Knowledge Assistant for two real-estate brands:
- Dar Global (darglobal.co.uk): luxury branded residences and developments worldwide, part of Dar Al Arkan.
- Wasalt (wasalt.sa): Saudi Arabia's property marketplace (sale/rent listings, projects, auctions, market guides).

Rules:
1. Use ONLY the numbered context provided. If the answer is not there, say so briefly and point to the relevant website or contact channel. Never invent prices, dates, yields or availability.
2. Cite sources inline as [1], [2] using the context numbers. Be concise: a direct answer first, then short bullets if useful. Prices in SAR unless stated.
3. Listings are a dated snapshot and may be stale; tell users to verify on the live site. No financial or investment advice.
4. Answer in the user's language (English or Arabic)."""

CITIES = {"riyadh": "Riyadh", "jeddah": "Jeddah", "makkah": "Makkah", "mecca": "Makkah", "madinah": "Madinah",
          "medina": "Madinah", "dammam": "Dammam", "khobar": "Khobar", "taif": "Taif", "abha": "Abha",
          "khamis": "Khamis", "tabuk": "Tabuk", "buraydah": "Buraydah"}
PTYPES = {"apartment": "apartment", "flat": "apartment", "villa": "villa", "land": "land", "plot": "land",
          "building": "building", "floor": "floor", "townhouse": "townhouse", "shop": "shop", "office": "office"}


def _money(s: str, unit: str | None) -> float:
    v = float(s.strip(".,").replace(",", ""))
    u = (unit or "").lower()
    if u in ("k", "thousand"): v *= 1e3
    elif u in ("m", "mn", "million"): v *= 1e6
    return v


def plan(question: str) -> dict:
    """Deterministic router. Returns {'intent': 'listings'|'knowledge', 'filters': {...}}."""
    q = question.lower()
    f: dict = {}
    for k, v in CITIES.items():
        if re.search(rf"\b{k}", q): f["city"] = v; break
    for k, v in PTYPES.items():
        if re.search(rf"\b{k}s?\b", q): f["property_type"] = v; break
    f["listing_type"] = "rent" if re.search(r"\b(rent|rental|lease)\b", q) else "sale"
    m = re.search(r"(?:under|below|less than|up to|max(?:imum)?|within|<)\s*(?:sar\s*)?([\d.,]+)\s*(k|m|mn|million|thousand)?", q)
    if m:
        try: f["max_price"] = _money(m.group(1), m.group(2))
        except ValueError: pass
    m = re.search(r"(?:over|above|more than|at least|min(?:imum)?|from|>)\s*(?:sar\s*)?([\d.,]+)\s*(k|m|mn|million|thousand)?", q)
    if m and "max_price" not in f:
        try: f["min_price"] = _money(m.group(1), m.group(2))
        except ValueError: pass
    m = re.search(r"(\d+)\s*[- ]?(?:bed|br\b|bedroom)", q)
    if m: f["min_bedrooms"] = int(m.group(1))
    wants_listings = bool(re.search(r"\b(show|find|list|looking for|search|available|cheapest|budget|options)\b", q)
                          and ("property_type" in f or "max_price" in f or "city" in f)) \
        or bool(re.search(r"\b(under|below|within)\b", q) and "property_type" in f)
    return {"intent": "listings" if wants_listings else "knowledge", "filters": f}


class Agent:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.cache: OrderedDict[str, dict] = OrderedDict()

    @staticmethod
    def _src(sources, url, title) -> int:
        for i, s in enumerate(sources, 1):
            if s["url"] == url:
                return i
        sources.append({"url": url, "title": title})
        return len(sources)

    def build_context(self, question: str, p: dict, sources: list[dict], steps: list[dict]) -> str:
        parts = []
        if p["intent"] == "listings":
            res, total = self.kb.search_listings(limit=6, **p["filters"])
            steps.append({"name": "search_listings", "detail": json.dumps(p["filters"]), "result": f"{total} match(es)"})
            if res:
                rows = [f"Wasalt listing snapshot - {total} matching, cheapest {len(res)} shown:"]
                for l in res:
                    n = self._src(sources, l["url"], l["title"])
                    rows.append(f"[{n}] {l['title']} | {l['listing_type']} | {l['price']:,.0f} SAR | "
                                f"{l.get('bedrooms') or 'n/a'} bed | {l.get('area_sqm') or 'n/a'} sqm | {l['location']}")
                parts.append("\n".join(rows))
            else:
                parts.append("Wasalt listing snapshot: NO listings match these filters (the snapshot is only a sample of the marketplace).")
        site = "wasalt" if p["intent"] == "listings" else None
        hits = self.kb.search(question, k=4, site=site)
        steps.append({"name": "search_knowledge", "detail": question[:80], "result": f"{len(hits)} passage(s)"})
        for h in hits:
            n = self._src(sources, h["url"], h["title"])
            parts.append(f"[{n}] ({h['site']}) {h['title']} - {h['url']}\n{h['text'][:650]}")
        return "\n\n".join(parts) or "No relevant context found."

    async def _stream_model(self, client, model, messages) -> AsyncIterator[str]:
        body = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": MAX_TOKENS,
                "stream": True, "reasoning": {"enabled": False}}
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "HTTP-Referer": os.getenv("APP_URL", "https://darglobal-wasalt-assistant.onrender.com"),
                   "X-Title": "DarGlobal-Wasalt Assistant"}
        async with client.stream("POST", OPENROUTER_URL, json=body, headers=headers,
                                 timeout=httpx.Timeout(30, connect=5, read=FIRST_TOKEN_TIMEOUT)) as r:
            if r.status_code == 400 and "reasoning" in (await r.aread()).decode("utf-8", "ignore").lower():
                raise RuntimeError("reasoning-mandatory")
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line.endswith("[DONE]"):
                    continue
                try:
                    d = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if d.get("error"):
                    raise RuntimeError(str(d["error"])[:120])
                tok = ((d.get("choices") or [{}])[0].get("delta") or {}).get("content")
                if tok:
                    yield tok

    async def stream(self, question: str, history: list[dict]) -> AsyncIterator[dict]:
        t0 = time.time()
        if not self.api_key:
            yield {"type": "error", "message": "Service is not configured (missing API key)."}
            return
        key = question.strip().lower()
        if not history and key in self.cache:
            c = self.cache[key]; self.cache.move_to_end(key)
            yield {"type": "answer", **c, "cached": True, "latency_s": round(time.time() - t0, 2)}
            return
        sources: list[dict] = []; steps: list[dict] = []
        p = plan(question)
        steps.append({"name": "plan", "detail": p["intent"], "result": json.dumps(p["filters"]) if p["intent"] == "listings" else ""})
        context = self.build_context(question, p, sources, steps)
        yield {"type": "steps", "steps": steps, "sources": sources, "intent": p["intent"]}
        snap = self.kb.meta.get("scraped_at", "unknown")
        messages = [{"role": "system", "content": SYSTEM + f"\nData snapshot date: {snap}."}]
        for h in history[-6:]:
            if h.get("role") in ("user", "assistant") and isinstance(h.get("content"), str):
                messages.append({"role": h["role"], "content": h["content"][:1500]})
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

        errors = []
        async with httpx.AsyncClient() as client:
            for model in MODEL_CHAIN:
                text, first = "", None
                try:
                    async for tok in self._stream_model(client, model, messages):
                        if first is None:
                            first = time.time() - t0
                        text += tok
                        yield {"type": "token", "text": tok}
                    if text.strip():
                        out = {"text": text.strip(), "model": model, "sources": sources, "steps": steps,
                               "first_token_s": round(first or 0, 2)}
                        if not history:
                            self.cache[key] = out
                            while len(self.cache) > CACHE_SIZE: self.cache.popitem(last=False)
                        yield {"type": "answer", **out, "latency_s": round(time.time() - t0, 2)}
                        return
                    errors.append(f"{model}: empty")
                except Exception as e:
                    errors.append(f"{model}: {type(e).__name__} {e}")
                    if text:  # partial output already shown: finish with what we have rather than restart
                        yield {"type": "answer", "text": text.strip(), "model": model, "sources": sources, "steps": steps,
                               "latency_s": round(time.time() - t0, 2)}
                        return
                    yield {"type": "reset"}
        yield {"type": "error", "message": "The language model providers are busy right now. Please retry in a moment."}
