"""Agentic chat loop: tool-calling over OpenRouter free models with automatic model fallback.

Flow per user turn:
  1. Pre-retrieval  - BM25 hits for the question are injected as context (keeps weak models grounded).
  2. Tool loop      - the model may call search_knowledge / search_listings / get_page (max MAX_STEPS).
  3. Answer         - grounded answer with [n] citations; every source is returned to the UI.
"""
from __future__ import annotations
import json, os, time
from typing import AsyncIterator
import httpx
from .kb import KnowledgeBase

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_CHAIN = [m.strip() for m in os.getenv(
    "MODEL_CHAIN",
    "google/gemma-4-31b-it:free,nvidia/nemotron-3-ultra-550b-a55b:free,inclusionai/ling-3.2-flash:free,openrouter/free",
).split(",") if m.strip()]
MAX_STEPS = int(os.getenv("MAX_STEPS", "4"))

SYSTEM = """You are the Dar Global & Wasalt Assistant, an AI concierge for two Saudi/Gulf real-estate brands:
- Dar Global (darglobal.co.uk): luxury branded residences and developments worldwide (partners such as Trump, Aston Martin, Elie Saab, Missoni, Marriott), part of Dar Al Arkan.
- Wasalt (wasalt.sa): Saudi Arabia's property marketplace for sale/rent listings, projects, auctions and market guides.

Rules:
1. Ground every factual claim in the provided context or tool results. If it is not there, say you don't have that information in your knowledge base and suggest contacting the company. Never invent prices, dates, yields or availability.
2. Use the tools when the context is insufficient: search_knowledge for company/project/market info, search_listings for concrete Wasalt listings (filter by city, type, price, bedrooms), get_page to read a full page.
3. Cite sources inline as [1], [2] matching the numbered sources you were given. Keep answers concise (short paragraphs or bullets). Prices are in SAR unless stated.
4. Listings are a point-in-time snapshot (see data date) and may be stale; tell users to verify on the live site. You do not give financial or investment advice.
5. Reply in the user's language (English or Arabic)."""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_knowledge",
        "description": "Full-text search over scraped Dar Global and Wasalt pages (projects, locations, partners, press, blog, market guides, FAQs, policies).",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search keywords"},
            "site": {"type": "string", "enum": ["darglobal", "wasalt", "any"], "description": "Restrict to one brand"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_listings",
        "description": "Filter the structured snapshot of Wasalt property listings (sale/rent). Returns cheapest first.",
        "parameters": {"type": "object", "properties": {
            "listing_type": {"type": "string", "enum": ["sale", "rent"]},
            "property_type": {"type": "string", "description": "e.g. apartment, villa, land, building, floor"},
            "city": {"type": "string", "description": "e.g. Riyadh, Jeddah, Dammam"},
            "district": {"type": "string"},
            "min_price": {"type": "number"}, "max_price": {"type": "number"},
            "min_bedrooms": {"type": "integer"},
            "limit": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "get_page",
        "description": "Return the full scraped text of one page by its URL (taken from a previous search result).",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
]


class Agent:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")

    # numbered citations shared by pre-retrieval and tools
    @staticmethod
    def _src(sources: list[dict], url: str, title: str) -> int:
        for i, s in enumerate(sources, 1):
            if s["url"] == url:
                return i
        sources.append({"url": url, "title": title})
        return len(sources)

    def _fmt_chunks(self, hits, sources, max_chars=900):
        lines = []
        for h in hits:
            n = self._src(sources, h["url"], h["title"])
            lines.append(f"[{n}] ({h['site']}) {h['title']} - {h['url']}\n{h['text'][:max_chars]}")
        return "\n\n".join(lines) or "No results."

    def run_tool(self, name: str, args: dict, sources: list[dict]) -> str:
        try:
            if name == "search_knowledge":
                hits = self.kb.search(args.get("query", ""), k=5, site=args.get("site"))
                return self._fmt_chunks(hits, sources)
            if name == "search_listings":
                limit = max(1, min(int(args.get("limit") or 6), 10))
                res, total = self.kb.search_listings(
                    args.get("listing_type"), args.get("property_type"), args.get("city"), args.get("district"),
                    args.get("min_price"), args.get("max_price"), args.get("min_bedrooms"), limit)
                if not res:
                    return "No matching listings in the snapshot (it only holds a sample of Wasalt listings)."
                out = [f"{total} matching listings in snapshot; showing {len(res)} cheapest:"]
                for l in res:
                    n = self._src(sources, l["url"], l["title"])
                    out.append(f"[{n}] {l['title']} | {l.get('listing_type')} | {l.get('price')} SAR | "
                               f"{l.get('bedrooms')} bed | {l.get('area_sqm')} sqm | {l.get('location')}")
                return "\n".join(out)
            if name == "get_page":
                p = self.kb.get_page(args.get("url", ""))
                if not p:
                    return "Page not found in knowledge base."
                n = self._src(sources, p["url"], p["title"])
                return f"[{n}] {p['title']} - {p['url']}\n{p['text']}"
        except Exception as e:  # tool errors must never crash the loop
            return f"Tool error: {e}"
        return f"Unknown tool {name}"

    async def _complete(self, client: httpx.AsyncClient, messages, use_tools=True):
        """Try each model in the chain until one answers."""
        last_err = "no models tried"
        for model in MODEL_CHAIN:
            body = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 1200}
            if use_tools:
                body["tools"] = TOOLS
            try:
                r = await client.post(OPENROUTER_URL, json=body, headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": os.getenv("APP_URL", "https://huggingface.co/spaces"),
                    "X-Title": "DarGlobal-Wasalt Assistant"}, timeout=75)
                if r.status_code != 200:
                    last_err = f"{model}: HTTP {r.status_code} {r.text[:200]}"
                    continue
                data = r.json()
                if "error" in data or not data.get("choices"):
                    last_err = f"{model}: {str(data.get('error', data))[:200]}"
                    continue
                msg = data["choices"][0]["message"]
                if not (msg.get("content") or msg.get("tool_calls")):
                    last_err = f"{model}: empty response"
                    continue
                return model, msg
            except Exception as e:
                last_err = f"{model}: {type(e).__name__} {e}"
        raise RuntimeError(last_err)

    async def stream(self, question: str, history: list[dict]) -> AsyncIterator[dict]:
        """Yield events: {type: status|tool|answer|error, ...}."""
        if not self.api_key:
            yield {"type": "error", "message": "Server is missing OPENROUTER_API_KEY."}
            return
        sources: list[dict] = []
        pre = self.kb.search(question, k=4)
        context = self._fmt_chunks(pre, sources)
        snap = self.kb.meta.get("scraped_at", "unknown")
        yield {"type": "status", "message": f"Retrieved {len(pre)} passages"}
        messages = [{"role": "system", "content": SYSTEM + f"\nData snapshot date: {snap}."}]
        for h in history[-8:]:
            if h.get("role") in ("user", "assistant") and isinstance(h.get("content"), str):
                messages.append({"role": h["role"], "content": h["content"][:3000]})
        messages.append({"role": "user", "content": f"{question}\n\n--- Auto-retrieved context ---\n{context}"})
        t0 = time.time()
        async with httpx.AsyncClient() as client:
            try:
                for step in range(MAX_STEPS + 1):
                    model, msg = await self._complete(client, messages, use_tools=step < MAX_STEPS)
                    calls = msg.get("tool_calls") or []
                    if not calls:
                        yield {"type": "answer", "text": (msg.get("content") or "").strip(), "model": model,
                               "sources": sources, "latency_s": round(time.time() - t0, 1), "steps": step}
                        return
                    messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
                    for c in calls:
                        fn = c["function"]
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        yield {"type": "tool", "name": fn["name"], "args": args}
                        result = self.run_tool(fn["name"], args, sources)
                        messages.append({"role": "tool", "tool_call_id": c["id"], "content": result[:6000]})
            except Exception as e:
                yield {"type": "error", "message": f"Model providers unavailable right now ({e}). Please retry in a minute."}
