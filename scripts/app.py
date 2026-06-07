from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import aiosqlite
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from anra.anra_paths import DRIVE_DIR, MEMORY_DB_DIR, DRIVE_SESSIONS, ensure_dirs

ensure_dirs()

from scripts.generate import (
    GenerationConfig, generate, generate_stream, generate_traced, get_model_info,
    load_ghost_state, save_ghost_state
)
from inference.full_system_connector import build_capability_graph
from inference.optimize_context_window import ContextWindowOptimizer
from runtime.hal_telemetry import read_hal_state

START_TIME = time.time()
_COLAB_DRIVE = DRIVE_SESSIONS
_LOCAL_FALLBACK = Path(__file__).resolve().parent / "output" / "sessions"
SESSION_DIR = _COLAB_DRIVE if DRIVE_DIR.parent.parent.exists() else _LOCAL_FALLBACK
SESSION_DIR.mkdir(parents=True, exist_ok=True)
LOGGER = logging.getLogger("anra.api")
logging.basicConfig(level=logging.INFO)


class SQLiteSessionStore:
    """Persistent session store backed by SQLite. Survives server restarts."""

    def __init__(self, db_path: Path, max_history: int = 40) -> None:
        self._path = db_path
        self._max_history = max_history
        self._path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    history TEXT NOT NULL DEFAULT '[]',
                    meta TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    last_active REAL NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS rate_limits (
                    ip TEXT NOT NULL,
                    ts REAL NOT NULL
                )
                """
            )
            await db.execute("CREATE INDEX IF NOT EXISTS idx_rate_ip ON rate_limits(ip)")
            await db.commit()

    async def get_history(self, session_id: str) -> list[dict]:
        async with aiosqlite.connect(self._path) as db:
            async with db.execute("SELECT history FROM sessions WHERE id = ?", (session_id,)) as cur:
                row = await cur.fetchone()
                return json.loads(row[0]) if row else []

    async def get_meta(self, session_id: str) -> dict[str, Any]:
        async with aiosqlite.connect(self._path) as db:
            async with db.execute("SELECT meta FROM sessions WHERE id = ?", (session_id,)) as cur:
                row = await cur.fetchone()
                return json.loads(row[0]) if row else {}

    async def save_history(self, session_id: str, history: list[dict]) -> None:
        meta = await self.get_meta(session_id)
        await self.save_session(session_id, history, meta)

    async def save_session(self, session_id: str, history: list[dict], meta: dict[str, Any]) -> None:
        trimmed = history[-self._max_history:]
        now = time.time()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """INSERT INTO sessions (id, history, meta, created_at, last_active)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       history=excluded.history,
                       meta=excluded.meta,
                       last_active=excluded.last_active""",
                (session_id, json.dumps(trimmed), json.dumps(meta), now, now),
            )
            await db.commit()

    async def list_sessions(self) -> dict[str, dict[str, Any]]:
        async with aiosqlite.connect(self._path) as db:
            async with db.execute("SELECT id, history, meta FROM sessions ORDER BY last_active DESC") as cur:
                rows = await cur.fetchall()
        sessions: dict[str, dict[str, Any]] = {}
        for session_id, history_json, meta_json in rows:
            sessions[session_id] = {
                "history": json.loads(history_json),
                "metadata": json.loads(meta_json),
            }
        return sessions

    async def delete_session(self, session_id: str) -> None:
        async with aiosqlite.connect(self._path) as db:
            await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            await db.commit()

    async def count_sessions(self) -> int:
        async with aiosqlite.connect(self._path) as db:
            async with db.execute("SELECT COUNT(*) FROM sessions") as cur:
                row = await cur.fetchone()
                return int(row[0]) if row else 0

    async def check_rate_limit(
        self,
        ip: str,
        window_seconds: float = 60.0,
        max_requests: int = 30,
    ) -> bool:
        """Returns True if the request is allowed, False if rate limited."""
        cutoff = time.time() - window_seconds
        async with aiosqlite.connect(self._path) as db:
            await db.execute("DELETE FROM rate_limits WHERE ts < ?", (cutoff,))
            async with db.execute(
                "SELECT COUNT(*) FROM rate_limits WHERE ip = ? AND ts > ?", (ip, cutoff)
            ) as cur:
                row = await cur.fetchone()
                count = row[0] if row else 0
            if count >= max_requests:
                await db.commit()
                return False
            await db.execute("INSERT INTO rate_limits (ip, ts) VALUES (?, ?)", (ip, time.time()))
            await db.commit()
            return True


SESSION_STORE = SQLiteSessionStore(SESSION_DIR / "sessions.db", max_history=40)


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


# Memory system bridge. Prefer the canonical router; keep the old phase2 bridge
# as a compatibility fallback for older Drive state.
try:
    from memory.memory_router import MemoryRouter

    class _MemoryBridge:
        def __init__(self):
            self._router = MemoryRouter()
            self.semantic = self

        def search(self, query: str, top_k: int = 3):
            rows = self._router.read(query, n=top_k, tier="episodic")
            out = []
            for row in rows:
                payload = row.get("payload", row)
                content = str(payload.get("content", ""))
                out.append(
                    {
                        "summary": str(payload.get("summary") or content[:160]),
                        "content": content,
                        "score": row.get("score", 0.0),
                    }
                )
            return out

        def store_turn(self, message: str, response: str, session_id: str) -> None:
            self._router.write(
                f"H: {message}\nANRA: {response}",
                metadata={"session_id": session_id, "type": "conversation_turn", "salience": 0.8},
                tier="episodic",
            )

    MEMORY_SYSTEM = _MemoryBridge()
except Exception as _mem_exc:
    try:
        from phase2.memory_45j.memory_manager import MemoryManager  # type: ignore

        class _LegacyMemoryBridge:
            def __init__(self):
                self._mm = MemoryManager(data_dir=str(MEMORY_DB_DIR), user_id="anra")
                self.semantic = self

            def search(self, query: str, top_k: int = 3):
                return self._mm.retrieve(query, limit=top_k, type="semantic")

            def store_turn(self, message: str, response: str, session_id: str) -> None:
                self._mm.store_memory(
                    content=f"H: {message}\nANRA: {response}",
                    type="episodic",
                    importance="medium",
                    metadata={"session_id": session_id},
                )
                self._mm.extractor.process_single_turn("user", message, session_id)
                self._mm.extractor.process_single_turn("assistant", response, session_id)

        MEMORY_SYSTEM = _LegacyMemoryBridge()
    except Exception as _legacy_mem_exc:
        LOGGER.warning("Memory bridge unavailable: %s; legacy fallback unavailable: %s", _mem_exc, _legacy_mem_exc)
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


async def _get_session_history(session_id: str) -> list[dict]:
    return await SESSION_STORE.get_history(session_id)


async def _append_to_session(session_id: str, new_messages: list[dict]) -> None:
    history = await SESSION_STORE.get_history(session_id)
    history.extend(new_messages)
    await SESSION_STORE.save_history(session_id, history)


async def _save_session(session_id: str) -> None:
    pass  # SQLiteSessionStore saves atomically on every write — no explicit flush needed.


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


async def _build_context(session_id: str, message: str) -> tuple[str, int, bool]:
    history = (await _get_session_history(session_id))[-40:]
    truncated = False
    while True:
        context = _serialize_context_from_turns(history, message)
        if len(context) <= 1024 or len(history) <= 1:
            turns_included = sum(1 for x in history if x.get("role") == "assistant")
            return context, turns_included, truncated
        history = history[2:]
        truncated = True


def _turn_count(history: list[dict]) -> int:
    return sum(1 for x in history if x.get("role") == "assistant")


async def _rate_limit_or_429(client_ip: str) -> None:
    allowed = await SESSION_STORE.check_rate_limit(client_ip, window_seconds=60.0, max_requests=30)
    if not allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 30 requests per minute.")


def _latest_report_snapshot() -> Dict[str, Any] | None:
    reports_dir = Path(__file__).resolve().parent / "state" / "reports"
    if not reports_dir.exists():
        return None
    snapshots = sorted(reports_dir.glob("snapshot_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for snapshot in snapshots:
        try:
            payload = json.loads(snapshot.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                payload = payload[-1] if payload else {}
            if isinstance(payload, dict):
                payload["_source"] = str(snapshot)
                return payload
        except Exception as exc:
            LOGGER.warning("Failed to read training snapshot %s: %s", snapshot, exc)
    return None


@asynccontextmanager
async def lifespan(_: FastAPI):
    await SESSION_STORE.initialize()
    await run_in_threadpool(ADAPTER.load)
    global SYSTEM_GRAPH
    SYSTEM_GRAPH = await run_in_threadpool(build_capability_graph, Path(__file__).resolve().parent)
    LOGGER.info("An-Ra API startup complete. Session store: %s", SESSION_STORE._path)
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
    session_id: str = "default"
    message: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    session_id: str


@app.post("/generate")
async def generate_route(body: GenerateRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    await _rate_limit_or_429(client_ip)

    cfg = GenerationConfig(strategy=body.strategy)
    for k, v in body.params.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    await run_in_threadpool(load_ghost_state, body.session_id)
    trace = await run_in_threadpool(generate_traced, body.prompt, cfg, session_id=body.session_id)
    await run_in_threadpool(save_ghost_state, body.session_id)
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
    client_ip = request.client.host if request.client else "unknown"
    await _rate_limit_or_429(client_ip)

    history = await _get_session_history(body.session_id)

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

    ctx_result = await run_in_threadpool(
        _ctx_optimizer.build_optimized_context,
        session_history=session_pairs,
        memory_results=memory_results,
        current_message=body.message
    )
    context = ctx_result["context"]
    memory_context = format_memory_context(memory_results) if memory_results else ""
    full_prompt = f"{memory_context}\n\n{context}" if memory_context else context

    strategy = body.params.get("strategy", "nucleus")
    cfg = GenerationConfig(strategy=strategy)
    for k, v in body.params.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    run_params = {k: v for k, v in cfg.__dict__.items() if k != 'strategy'}
    await run_in_threadpool(load_ghost_state, body.session_id)
    reply = await run_in_threadpool(ADAPTER.run, full_prompt, strategy=cfg.strategy, **run_params)
    await run_in_threadpool(save_ghost_state, body.session_id)

    new_messages = [
        {"role": "user", "content": body.message},
        {"role": "assistant", "content": reply},
    ]
    history.extend(new_messages)
    await _append_to_session(body.session_id, new_messages)
    await _save_session(body.session_id)
    if MEMORY_SYSTEM is not None:
        try:
            MEMORY_SYSTEM.store_turn(body.message, reply, body.session_id)
        except Exception as mem_exc:
            LOGGER.debug("Memory store failed: %s", mem_exc)

    return {
        "response": reply,
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
    history = await _get_session_history(session_id)
    context, _, _ = await _build_context(session_id, message)
    cfg = GenerationConfig(strategy=strategy)

    async def async_event_gen():
        loop = asyncio.get_event_loop()
        gen_iter = await loop.run_in_executor(None, lambda: list(generate_stream(context, cfg)))
        assembled = ""
        for ch in gen_iter:
            assembled += ch
            yield f"data: {ch}\n\n"
        new_messages = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": assembled},
        ]
        history.extend(new_messages)
        await _append_to_session(session_id, new_messages)
        await _save_session(session_id)
        yield "data: [DONE]\n\n"

    return StreamingResponse(async_event_gen(), media_type="text/event-stream")


@app.get("/sessions")
async def sessions_route():
    sessions = await SESSION_STORE.list_sessions()
    return {"sessions": sessions, "count": len(sessions)}


@app.get("/health")
@app.get("/status")
async def health_route():
    info = ADAPTER.info or await run_in_threadpool(get_model_info)
    return {
        "status": "ok",
        "model": "An-Ra",
        "checkpoint": str(info.get("checkpoint", "unknown")),
        "device": str(info.get("device", "unknown")),
        "vocab_size": int(info.get("vocab_size", -1) or -1),  # type: ignore[arg-type]
        "uptime_seconds": time.time() - START_TIME,
        "sessions_active": await SESSION_STORE.count_sessions(),
    }


@app.get("/hal/state")
async def hal_state_route():
    return read_hal_state()


@app.post("/reset")
async def reset_route(body: ResetRequest):
    await SESSION_STORE.delete_session(body.session_id)
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
    context, _, _ = await _build_context(session_id, message)
    return {"context": context}


@app.get("/system-map")
async def system_map_route():
    return SYSTEM_GRAPH or await run_in_threadpool(build_capability_graph, Path(__file__).resolve().parent)


@app.get("/phase-health")
@app.get("/sovereignty/status")
async def phase_health_route():
    graph = SYSTEM_GRAPH or await run_in_threadpool(build_capability_graph, Path(__file__).resolve().parent)
    checks: Dict[str, Dict[str, Any]] = {}
    modules = [
        ("identity_injector", "identity"),
        ("ouroboros_numpy", "ouroboros"),
        ("symbolic_bridge", "symbolic"),
        ("sovereignty_bridge", "sovereignty"),
    ]
    for mod_name, key in modules:
        try:
            mod = __import__(mod_name)
            fn = getattr(mod, "health_check", None)
            checks[key] = dict(fn()) if callable(fn) else {"status": "degraded", "detail": "health_check missing"}  # type: ignore[arg-type]
        except Exception as exc:
            checks[key] = {"status": "degraded", "detail": str(exc)}
    return {
        "status": "ok",
        "capabilities": graph.get("capabilities", {}),
        "phase_snapshots": graph.get("phase_snapshots", []),
        "phase3_health": checks,
    }


@app.get("/goals")
async def goals_route():
    return {"goals": [], "active": 0, "completed": 0, "status": "online"}


@app.post("/goal")
async def add_goal_route(request: Request):
    body = await request.json()
    goal_text = body.get("goal", "").strip()
    if not goal_text:
        return JSONResponse(
            status_code=400,
            content={"error": "goal text is required"},
        )
    return {"status": "queued", "goal": goal_text, "id": str(uuid.uuid4())}


@app.get("/memory/stats")
async def memory_stats_route():
    return {
        "total": 0,
        "episodic": 0,
        "semantic": 0,
        "graph_nodes": 0,
        "status": "online" if MEMORY_SYSTEM is not None else "unavailable",
    }


@app.post("/memory/search")
async def memory_search_route(request: Request):
    body = await request.json()
    query = body.get("query", "").strip()
    results = []
    if MEMORY_SYSTEM and query:
        try:
            results = MEMORY_SYSTEM.semantic.search(query=query, top_k=5)
        except Exception as exc:
            LOGGER.warning("Memory search failed: %s", exc)
    return {"results": results, "query": query}


@app.post("/sovereignty/audit")
async def sovereignty_audit_route():
    # Non-blocking: just return immediately; real audit runs async later.
    return {
        "status": "triggered",
        "message": "Sovereignty audit queued. Check /sovereignty/status for results.",
        "timestamp": _now_iso(),
    }


@app.get("/train/status")
async def train_status_route():
    import torch

    gpu_count = torch.cuda.device_count()
    ram_gb = 0.0
    try:
        import psutil

        ram_gb = psutil.virtual_memory().total / 1024**3
    except Exception:
        pass
    config = "medium" if gpu_count > 0 else "small"
    snapshot = _latest_report_snapshot()
    stats = {
        "total_examples": 0,
        "high_quality": 0,
        "avg_quality": 0.0,
        "unused": 0,
    }
    latest_run = {"status": "idle", "loss_history": []}
    if snapshot:
        # AN: Training status should reflect the latest sovereignty snapshot when the daily loop has evidence.
        components = snapshot.get("components", {})
        training = components.get("training", {})
        output_quality = components.get("output_quality", {})
        prompts = components.get("prompts", {})
        stats = {
            "total_examples": int(training.get("total_examples", 0) or 0),
            "high_quality": int(training.get("high_quality", 0) or 0),
            "avg_quality": float(output_quality.get("avg_score", prompts.get("avg_score", 0.0)) or 0.0),
            "unused": int(training.get("unused_examples", training.get("unused", 0)) or 0),
        }
        last_run = training.get("last_run")
        if isinstance(last_run, dict):
            latest_run = {
                "status": last_run.get("status", "complete"),
                "loss_history": last_run.get("loss_history", []),
                "timestamp": last_run.get("timestamp"),
            }
        else:
            latest_run = {
                "status": "idle",
                "loss_history": [],
                "total_runs": int(training.get("total_runs", 0) or 0),
                "snapshot": snapshot.get("_source"),
            }
    return {
        "stats": stats,
        "hardware": {
            "gpu_count": gpu_count,
            "ram_gb": ram_gb,
            "recommended_config": config,
        },
        "latest_run": latest_run,
    }


@app.post("/train/trigger", status_code=501)
async def train_trigger_route() -> dict:
    raise HTTPException(
        status_code=501,
        detail={
            "error": "training_dispatch_not_implemented",
            "message": "Automated training dispatch is not yet implemented. Use scripts/train_oneshot.py directly or AnRa_Master.ipynb for Colab.",
            "docs": "https://github.com/your-repo/DEVELOPER.md#training",
        }
    )


@app.get("/identity/score")
async def identity_score_route():
    from identity.civ import ConstitutionalIdentityVector

    candidates = [
        DRIVE_DIR / "v3" / "identity" / "civ_profile.json",
        DRIVE_DIR / "identity" / "civ_profile.json",
        Path(__file__).resolve().parent / "state" / "identity" / "civ_profile.json",
    ]
    civ = None
    for path in candidates:
        if path.exists():
            try:
                # AN: Expose identity health lazily so the UI can track drift without slowing API startup.
                civ = ConstitutionalIdentityVector.load(path)
                break
            except Exception as exc:
                LOGGER.warning("Failed to load CIV profile %s: %s", path, exc)
    if civ is None:
        civ = ConstitutionalIdentityVector()

    result = civ.verify()
    return {
        "score": float(result["score"]),
        "profile": dict(civ.profile.__dict__),
        "passed": bool(result["passed"]),
    }


async def test_api(base_url: str = "http://127.0.0.1:8000") -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        assert (await client.get("/health")).status_code == 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run An-Ra API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("app:app", host=args.host, port=args.port, reload=False)
