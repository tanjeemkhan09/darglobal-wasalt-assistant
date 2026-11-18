import json, os, time
from collections import defaultdict, deque
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .agent import Agent, MODEL_CHAIN
from .kb import KnowledgeBase

app = FastAPI(title="Dar Global & Wasalt Assistant")
kb = KnowledgeBase()
agent = Agent(kb)
STATIC = Path(__file__).parent / "static"

RATE = int(os.getenv("RATE_LIMIT_PER_10MIN", "30"))
_hits: dict[str, deque] = defaultdict(deque)


def limited(ip: str) -> bool:
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > 600:
        q.popleft()
    if len(q) >= RATE:
        return True
    q.append(now)
    return False


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[dict] = []


@app.get("/api/health")
def health():
    return {"status": "ok", "chunks": len(kb.chunks), "listings": len(kb.listings), "models": MODEL_CHAIN,
            "key_configured": bool(agent.api_key), "snapshot": kb.meta}


@app.get("/api/warmup")
def warmup():
    """Cheap endpoint the UI pings on load so a sleeping host wakes before the first question."""
    return {"ok": True}


@app.post("/api/chat")
async def chat(body: ChatIn, request: Request):
    ip = (request.headers.get("x-forwarded-for") or request.client.host).split(",")[0].strip()
    if limited(ip):
        return JSONResponse({"error": "Rate limit reached, please wait a few minutes."}, status_code=429)

    async def gen():
        async for ev in agent.stream(body.message, body.history):
            yield json.dumps(ev, ensure_ascii=False) + "\n"
    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
