from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from generate import (
    GenerationConfig, generate, generate_stream, generate_traced, get_model_info,
    load_ghost_state, save_ghost_state
)
from full_system_connector import build_capability_graph
from optimize_context_window import ContextWindowOptimizer

START_TIME = time.time()
SESSION_DIR = Path("/content/drive/MyDrive/AnRa/sessions/")
SESSION_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS: Dict[str, Deque[Dict[str, str]]] = defaultdict(lambda: deque(maxlen=40))
SESSION_META: Dict[str, Dict[str, Any]] = {}
RATE_LIMIT_STORE: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=10))
LOGGER = logging.getLogger("anra.api")
logging.basicConfig(level=logging.INFO)


class ModelAdapter:
    def __init__(self) -> None:
        self.info: Dict[str, Any] = {}

    def load(self) -> None:
        self.info = get_model_info()

    def run(self, prompt: str, strategy: str = "nucleus", **params: Any) -> str:
        # SWAP POINT: replace only this line to redirect to a new backend model runtime.
        return generate(prompt, strategy=strategy, **params)


ADAPTER = ModelAdapter()
SYSTEM_GRAPH: Dict[str, Any] = {}
_ctx_optimizer = ContextWindowOptimizer()


# Memory system bridge (45J)
try:
    import sys as _sys
    _sys.path.insert(0, str((Path(__file__).resolve().parent / "phase2" / "memory (45J)")))
    from memory_manager import MemoryManager

    class _MemoryBridge:
        def __init__(self):
            self._mm = MemoryManager(data_dir="/content/drive/MyDrive/AnRa/memory_db", user_id="anra")
            self.semantic = self

        def search(self, query: str, top_k: int = 3):
            rows = self._mm.retrieve(query, limit=top_k, type="semantic")
            return rows

    MEMORY_SYSTEM = _MemoryBridge()
except Exception as _mem_exc:
    LOGGER.warning("Memory bridge unavailable: %s", _mem_exc)
    MEMORY_SYSTEM = None


def format_memory_context(memory_results: List[Dict[str, Any]]) -> str:
    lines = ["[Retrieved Memory Context]"]
    for i, item in enumerate(memory_results, start=1):
        lines.append(f"{i}. {item.get('summary', '')}")
        if item.get('content'):
            lines.append(f"   detail: {item.get('content')[:240]}")
    return "\n".join(lines)


# Memory system bridge (45J)
try:
    import sys as _sys
    _sys.path.insert(0, str((Path(__file__).resolve().parent / "phase2" / "memory (45J)")))
    from memory_manager import MemoryManager

    class _MemoryBridge:
        def __init__(self):
            self._mm = MemoryManager(data_dir="/content/drive/MyDrive/AnRa/memory_db", user_id="anra")
            self.semantic = self

        def search(self, query: str, top_k: int = 3):
            rows = self._mm.retrieve(query, limit=top_k, type="semantic")
            return rows

    MEMORY_SYSTEM = _MemoryBridge()
except Exception as _mem_exc:
    LOGGER.warning("Memory bridge unavailable: %s", _mem_exc)
    MEMORY_SYSTEM = None


def format_memory_context(memory_results: List[Dict[str, Any]]) -> str:
    lines = ["[Retrieved Memory Context]"]
    for i, item in enumerate(memory_results, start=1):
        lines.append(f"{i}. {item.get('summary', '')}")
        if item.get('content'):
            lines.append(f"   detail: {item.get('content')[:240]}")
    return "\n".join(lines)


def _session_file(session_id: str) -> Path:
    return SESSION_DIR / f"{session_id}.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_meta(session_id: str) -> None:
    if session_id not in SESSION_META:
        SESSION_META[session_id] = {
            "created_at": _now_iso(),
            "last_active": _now_iso(),
            "total_turns": 0,
            "strategy_used": "nucleus",
        }


def _load_all_sessions_from_disk() -> None:
    for file in SESSION_DIR.glob("*.json"):
        try:
            sid = file.stem
            payload = json.loads(file.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                history = payload
                meta = {"created_at": _now_iso(), "last_active": _now_iso(), "total_turns": 0, "strategy_used": "nucleus"}
            else:
                history = payload.get("history", [])
                meta = payload.get("metadata", {})
            SESSIONS[sid] = deque(history, maxlen=40)
            _ensure_meta(sid)
            SESSION_META[sid].update(meta)
        except Exception as exc:
            LOGGER.warning("Failed to preload session file %s: %s", file, exc)


def _load_session(session_id: str) -> Deque[Dict[str, str]]:
    _ensure_meta(session_id)
    if session_id in SESSIONS:
        return SESSIONS[session_id]
    file = _session_file(session_id)
    if file.exists():
        payload = json.loads(file.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            SESSIONS[session_id].extend(payload)
        else:
            SESSIONS[session_id].extend(payload.get("history", []))
            SESSION_META[session_id].update(payload.get("metadata", {}))
    return SESSIONS[session_id]


def _save_session(session_id: str) -> None:
    payload = {"history": list(SESSIONS[session_id]), "metadata": SESSION_META.get(session_id, {})}
    _session_file(session_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _serialize_context_from_turns(turns: List[Dict[str, str]], message: str) -> str:
    context_parts: List[str] = []
    i = 0
    while i < len(turns) - 1:
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
            segment = f"H: {turns[i]['content']}\nANRA: {turns[i + 1]['content']}\n"
            assert "\n\n" not in segment
            context_parts.append(segment)
            i += 2
        else:
            i += 1
    final = f"H: {message}\nANRA:"
    return "".join(context_parts) + final


def _build_context(session_id: str, message: str) -> tuple[str, int, bool]:
    history = list(_load_session(session_id))[-40:]
    truncated = False
    while True:
        context = _serialize_context_from_turns(history, message)
        if len(context) <= 1024 or len(history) <= 1:
            turns_included = sum(1 for x in history if x.get("role") == "assistant")
            return context, turns_included, truncated
        history = history[2:]
        truncated = True


def _turn_count(history: Deque[Dict[str, str]]) -> int:
    return sum(1 for x in history if x.get("role") == "assistant")


def _rate_limit_or_429(session_id: str, request_id: str):
    now = time.time()
    window = RATE_LIMIT_STORE[session_id]
    window.append(now)
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) > 10:
        retry_after = max(1, int(60 - (now - window[0])))
        return JSONResponse(
            status_code=429,
            content={"error": "rate_limit", "request_id": request_id, "retry_after_seconds": retry_after},
        )
    return None


@asynccontextmanager
async def lifespan(_: FastAPI):
    ADAPTER.load()
    _load_all_sessions_from_disk()
    global SYSTEM_GRAPH
    SYSTEM_GRAPH = build_capability_graph(Path(__file__).resolve().parent)
    LOGGER.info("Loaded model, preloaded %d sessions, indexed %d files", len(SESSIONS), SYSTEM_GRAPH.get("file_count", 0))
    yield


app = FastAPI(title="An-Ra API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    dt = (time.perf_counter() - t0) * 1000
    LOGGER.info("[req_id=%s] %s %s %s %.2fms", request_id, request.method, request.url.path, response.status_code, dt)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    LOGGER.exception("[req_id=%s] Unhandled error\n%s", getattr(request.state, "request_id", "unknown"), traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "request_id": getattr(request.state, "request_id", "unknown"),
            "message": "An internal error occurred.",
        },
    )


class GenerateRequest(BaseModel):
    prompt: str
    strategy: str = "nucleus"
    session_id: str = "generate_default"
    params: Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    session_id: str


@app.post("/generate")
async def generate_route(body: GenerateRequest, request: Request):
    limited = _rate_limit_or_429(body.session_id, request.state.request_id)
    if limited:
        return limited

    cfg = GenerationConfig(strategy=body.strategy)
    for k, v in body.params.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    load_ghost_state(body.session_id)
    trace = generate_traced(body.prompt, cfg, session_id=body.session_id)
    save_ghost_state(body.session_id)
    entropy_avg = sum(trace.entropy_curve) / max(len(trace.entropy_curve), 1)
    max_prob_avg = sum(trace.max_prob_curve) / max(len(trace.max_prob_curve), 1)

    return {
        "response": trace.output,
        "strategy": trace.strategy,
        "tokens_generated": trace.tokens_generated,
        "time_ms": trace.time_ms,
        "trace": {
            "entropy_avg": entropy_avg,
            "max_prob_avg": max_prob_avg,
            "repeated_ngrams": trace.repeated_ngrams_detected,
            "stopped_by": trace.stopped_by,
        },
    }


@app.post("/chat")
async def chat_route(body: ChatRequest, request: Request):
    limited = _rate_limit_or_429(body.session_id, request.state.request_id)
    if limited:
        return limited

    history = _load_session(body.session_id)

    memory_results = []
    if MEMORY_SYSTEM is not None:
        try:
            memory_results = MEMORY_SYSTEM.semantic.search(query=body.message, top_k=3)
        except Exception as mem_exc:
            LOGGER.warning("Memory query failed for session %s: %s", body.session_id, mem_exc)

    session_pairs = []
    turns = list(history)
    i = 0
    while i < len(turns) - 1:
        if turns[i].get("role") == "user" and turns[i + 1].get("role") == "assistant":
            session_pairs.append((turns[i].get("content", ""), turns[i + 1].get("content", "")))
            i += 2
        else:
            i += 1

    ctx_result = _ctx_optimizer.build_optimized_context(
        session_history=session_pairs,
        memory_results=memory_results,
        current_message=body.message
    )
    context = ctx_result["context"]

    memory_results = []
    if MEMORY_SYSTEM is not None:
        try:
            memory_results = MEMORY_SYSTEM.semantic.search(query=body.message, top_k=3)
        except Exception as mem_exc:
            LOGGER.warning("Memory query failed for session %s: %s", body.session_id, mem_exc)
    if memory_results:
        context_prefix = format_memory_context(memory_results)
        context = context_prefix + "\n" + context

    strategy = body.params.get("strategy", "nucleus")
    cfg = GenerationConfig(strategy=strategy)
    for k, v in body.params.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    run_params = {k: v for k, v in cfg.__dict__.items() if k != 'strategy'}
    load_ghost_state(body.session_id)
    reply = ADAPTER.run(context, strategy=cfg.strategy, **run_params)
    save_ghost_state(body.session_id)

    history.append({"role": "user", "content": body.message})
    history.append({"role": "assistant", "content": reply})
    _ensure_meta(body.session_id)
    SESSION_META[body.session_id]["last_active"] = _now_iso()
    SESSION_META[body.session_id]["total_turns"] = _turn_count(history)
    SESSION_META[body.session_id]["strategy_used"] = cfg.strategy
    _save_session(body.session_id)

    return {
        "reply": reply,
        "session_id": body.session_id,
        "turn": _turn_count(history),
        "history": list(history),
        "context_length": ctx_result["context_length"],
        "turns_included": ctx_result["turns_included"],
        "context_truncated": ctx_result["context_truncated"],
        "memory_truncated": ctx_result["memory_truncated"],
    }


@app.get("/stream")
async def stream_route(session_id: str, message: str, strategy: str = "nucleus"):
    history = _load_session(session_id)
    context, _, _ = _build_context(session_id, message)
    cfg = GenerationConfig(strategy=strategy)

    def event_gen():
        assembled = ""
        try:
            for ch in generate_stream(context, cfg):
                assembled += ch
                yield f"data: {ch}\n\n"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": assembled})
            _ensure_meta(session_id)
            SESSION_META[session_id]["last_active"] = _now_iso()
            SESSION_META[session_id]["total_turns"] = _turn_count(history)
            SESSION_META[session_id]["strategy_used"] = strategy
            _save_session(session_id)
            yield "data: [DONE]\n\n"
        except GeneratorExit:
            return

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/sessions")
async def sessions_route():
    return {
        "active_sessions": len(SESSIONS),
        "session_ids": list(SESSIONS.keys()),
        "total_turns": {sid: meta.get("total_turns", 0) for sid, meta in SESSION_META.items()},
        "metadata": SESSION_META,
    }


@app.get("/health")
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


@app.post("/reset")
async def reset_route(body: ResetRequest):
    SESSIONS.pop(body.session_id, None)
    SESSION_META.pop(body.session_id, None)
    file = _session_file(body.session_id)
    if file.exists():
        file.unlink()
    return {"cleared": True, "session_id": body.session_id}


@app.get("/strategies")
async def strategies_route():
    return {
        "greedy": {"description": "Deterministic argmax decoding", "params": {}},
        "temperature": {"description": "Temperature sampling", "params": {"temperature": 0.8}},
        "topk": {"description": "Top-k sampling", "params": {"top_k": 40}},
        "nucleus": {"description": "Top-p nucleus sampling", "params": {"top_p": 0.92}},
        "beam": {"description": "Beam search", "params": {"beam_width": 4}},
        "contrastive": {"description": "Contrastive or nucleus fallback", "params": {"top_p": 0.92}},
    }


@app.get("/debug/context/{session_id}")
async def debug_context_route(session_id: str, message: str = "debug"):
    context, _, _ = _build_context(session_id, message)
    return {"context": context}


@app.get("/system-map")
async def system_map_route():
    return SYSTEM_GRAPH or build_capability_graph(Path(__file__).resolve().parent)


@app.get("/phase-health")
async def phase_health_route():
    graph = SYSTEM_GRAPH or build_capability_graph(Path(__file__).resolve().parent)
    return {"status": "ok", "capabilities": graph.get("capabilities", {}), "phase_snapshots": graph.get("phase_snapshots", [])}


async def test_api(base_url: str = "http://127.0.0.1:8000") -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        assert (await client.get("/health")).status_code == 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run An-Ra API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("app:app", host=args.host, port=args.port, reload=False)
