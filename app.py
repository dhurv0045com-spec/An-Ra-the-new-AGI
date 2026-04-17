from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from generate import detect_repetition, generate, generate_with_trace, get_model_info
from full_system_connector import build_capability_graph

START_TIME = time.time()
SESSION_DIR = Path("/content/drive/MyDrive/AnRa/sessions/")
SESSION_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS: Dict[str, Deque[Dict[str, str]]] = defaultdict(lambda: deque(maxlen=40))
LOGGER = logging.getLogger("anra.api")
logging.basicConfig(level=logging.INFO)


# ======================================================================================
# Adapter
# ======================================================================================
class ModelAdapter:
    def __init__(self) -> None:
        self.info: Dict[str, Any] = {}

    def load(self) -> None:
        self.info = get_model_info()

    def run(self, prompt: str, strategy: str = "nucleus", **params: Any) -> str:
        # SWAP POINT: replace only this line to redirect to a new backend model runtime.
        return generate(prompt, strategy=strategy, **params)

    def run_trace(self, prompt: str, strategy: str = "nucleus", **params: Any) -> Dict[str, Any]:
        trace = generate_with_trace(prompt, strategy=strategy, **params)
        rep = detect_repetition(trace.generated_text)
        return {"text": trace.generated_text, "elapsed_ms": trace.elapsed_ms, "trace": trace.summary(), "repetition": rep}


ADAPTER = ModelAdapter()
SYSTEM_GRAPH: Dict[str, Any] = {}


# ======================================================================================
# Session store helpers
# ======================================================================================

def _session_file(session_id: str) -> Path:
    return SESSION_DIR / f"{session_id}.json"


def _load_all_sessions_from_disk() -> None:
    for file in SESSION_DIR.glob("*.json"):
        try:
            sid = file.stem
            payload = json.loads(file.read_text(encoding="utf-8"))
            dq = deque(payload, maxlen=40)
            SESSIONS[sid] = dq
        except Exception as exc:
            LOGGER.warning("Failed to preload session file %s: %s", file, exc)


def _load_session(session_id: str) -> Deque[Dict[str, str]]:
    if session_id in SESSIONS:
        return SESSIONS[session_id]
    file = _session_file(session_id)
    if file.exists():
        data = json.loads(file.read_text(encoding="utf-8"))
        SESSIONS[session_id].extend(data)
    return SESSIONS[session_id]


def _save_session(session_id: str) -> None:
    _session_file(session_id).write_text(
        json.dumps(list(SESSIONS[session_id]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_context(session_id: str, message: str, history_pairs: int = 20) -> str:
    history = list(_load_session(session_id))[-(history_pairs * 2) :]
    chunks: List[str] = []
    i = 0
    while i < len(history) - 1:
        if history[i]["role"] == "user" and history[i + 1]["role"] == "assistant":
            chunks.append(f"H: {history[i]['content']}\nANRA: {history[i + 1]['content']}\n")
            i += 2
        else:
            i += 1
    chunks.append(f"H: {message}\nANRA:")
    return "".join(chunks)


def _turn_count(history: Deque[Dict[str, str]]) -> int:
    return sum(1 for x in history if x.get("role") == "assistant")


# ======================================================================================
# FastAPI lifecycle
# ======================================================================================
@asynccontextmanager
async def lifespan(_: FastAPI):
    ADAPTER.load()
    _load_all_sessions_from_disk()
    global SYSTEM_GRAPH
    SYSTEM_GRAPH = build_capability_graph(Path(__file__).resolve().parent)
    LOGGER.info("Loaded model, preloaded %d sessions, indexed %d files", len(SESSIONS), SYSTEM_GRAPH.get("file_count", 0))
    yield


app = FastAPI(title="An-Ra API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    dt = (time.perf_counter() - t0) * 1000
    LOGGER.info("%s %s %s %.2fms", request.method, request.url.path, response.status_code, dt)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(_: Request, exc: Exception):
    LOGGER.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"error": "internal_error", "message": str(exc)})


# ======================================================================================
# Request/response schemas
# ======================================================================================
class GenerateRequest(BaseModel):
    prompt: str
    strategy: str = "nucleus"
    params: Dict[str, Any] = Field(default_factory=dict)


class GenerateResponse(BaseModel):
    response: str
    strategy: str
    tokens_generated: int
    time_ms: float
    repetition: Dict[str, Any]
    diagnostics: Dict[str, Any]


class ChatRequest(BaseModel):
    session_id: str
    message: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    turn: int
    history: List[Dict[str, str]]
    repetition: Dict[str, Any]


class ResetRequest(BaseModel):
    session_id: str


class ResetResponse(BaseModel):
    cleared: bool
    session_id: str


class HealthResponse(BaseModel):
    status: str
    model: str
    checkpoint: str
    device: str
    vocab_size: int
    uptime_seconds: float
    sessions_active: int


# ======================================================================================
# Endpoints
# ======================================================================================
@app.post("/generate", response_model=GenerateResponse)
async def generate_route(body: GenerateRequest):
    t0 = time.perf_counter()
    payload = ADAPTER.run_trace(body.prompt, strategy=body.strategy, **body.params)
    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "response": payload["text"],
        "strategy": body.strategy,
        "tokens_generated": len(payload["text"]),
        "time_ms": elapsed,
        "repetition": payload["repetition"],
        "diagnostics": payload["trace"],
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_route(body: ChatRequest):
    history = _load_session(body.session_id)
    prompt = _build_context(body.session_id, body.message)
    run_params = dict(body.params)
    strategy = run_params.pop("strategy", "nucleus")
    payload = ADAPTER.run_trace(prompt, strategy=strategy, **run_params)

    history.append({"role": "user", "content": body.message})
    history.append({"role": "assistant", "content": payload["text"]})
    _save_session(body.session_id)

    return {
        "reply": payload["text"],
        "session_id": body.session_id,
        "turn": _turn_count(history),
        "history": list(history),
        "repetition": payload["repetition"],
    }


@app.get("/health", response_model=HealthResponse)
async def health_route():
    info = ADAPTER.info or get_model_info()
    return {
        "status": "ok",
        "model": "An-Ra",
        "checkpoint": str(info.get("checkpoint", "unknown")),
        "device": str(info.get("device", "unknown")),
        "vocab_size": int(info.get("vocab_size", -1)),
        "uptime_seconds": time.time() - START_TIME,
        "sessions_active": len(SESSIONS),
    }


@app.post("/reset", response_model=ResetResponse)
async def reset_route(body: ResetRequest):
    SESSIONS.pop(body.session_id, None)
    file = _session_file(body.session_id)
    if file.exists():
        file.unlink()
    return {"cleared": True, "session_id": body.session_id}


@app.get("/strategies")
async def strategies_route():
    return {
        "available": ["greedy", "temperature", "topk", "nucleus", "beam", "contrastive"],
        "default": "nucleus",
    }




@app.get("/system-map")
async def system_map_route():
    return SYSTEM_GRAPH or build_capability_graph(Path(__file__).resolve().parent)


@app.get("/phase-health")
async def phase_health_route():
    graph = SYSTEM_GRAPH or build_capability_graph(Path(__file__).resolve().parent)
    return {
        "status": "ok",
        "capabilities": graph.get("capabilities", {}),
        "phase_snapshots": graph.get("phase_snapshots", []),
    }


# ======================================================================================
# Built-in API test
# ======================================================================================
async def test_api(base_url: str = "http://127.0.0.1:8000") -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        r_health = await client.get("/health")
        assert r_health.status_code == 200 and "status" in r_health.json()

        r_generate = await client.post(
            "/generate",
            json={"prompt": "H: hi\nANRA:", "strategy": "nucleus", "params": {"max_new_tokens": 30}},
        )
        j_generate = r_generate.json()
        assert r_generate.status_code == 200
        assert "response" in j_generate and "time_ms" in j_generate and "diagnostics" in j_generate

        r_chat = await client.post(
            "/chat",
            json={"session_id": "api_test", "message": "hello", "params": {"max_new_tokens": 20}},
        )
        j_chat = r_chat.json()
        assert r_chat.status_code == 200
        assert "reply" in j_chat and "history" in j_chat and "turn" in j_chat

        r_reset = await client.post("/reset", json={"session_id": "api_test"})
        assert r_reset.status_code == 200 and r_reset.json().get("cleared") is True

        r_strat = await client.get("/strategies")
        assert r_strat.status_code == 200 and "available" in r_strat.json()

        r_map = await client.get("/system-map")
        assert r_map.status_code == 200 and "file_count" in r_map.json()

        r_phase = await client.get("/phase-health")
        assert r_phase.status_code == 200 and "capabilities" in r_phase.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        import asyncio
        import threading
        import urllib.request

        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

        t = threading.Thread(target=run_server, daemon=True)
        t.start()
        for _ in range(40):
            try:
                urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1)
                break
            except Exception:
                time.sleep(0.25)
        asyncio.run(test_api())
        print("API READY")
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)
