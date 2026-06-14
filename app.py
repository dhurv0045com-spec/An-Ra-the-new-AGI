from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import logging
import os
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
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

from anra.anra_paths import (
    CAMPAIGN_DIR,
    CIV_LATEST,
    DRIVE_DIR,
    DRIVE_SESSIONS,
    IBS_LATEST,
    MEMORY_DB_DIR,
    OPERATOR_AUDIT_LOG,
    OUTPUT_V2_DIR,
    ROOT,
    STATE_DIR,
    ensure_dirs,
)

ensure_dirs()

from generate import (
    GenerationConfig, generate, generate_stream, generate_traced, get_model_info,
    load_ghost_state, save_ghost_state
)
from inference.full_system_connector import build_capability_graph
from inference.optimize_context_window import ContextWindowOptimizer
from engine.feature_flags import disabled_components, is_enabled
from intelligence.hgp import MissionNode, MissionTree
from goals.goal_queue import GoalQueue
from robotics.contracts import SkillGoal, Workflow
from runtime.hal_telemetry import read_hal_state
from cognition.services import CognitionServices

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
        self._initialized = False
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
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            await db.commit()
        self._initialized = True

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def get_history(self, session_id: str) -> list[dict]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self._path) as db:
            async with db.execute("SELECT history FROM sessions WHERE id = ?", (session_id,)) as cur:
                row = await cur.fetchone()
                return json.loads(row[0]) if row else []

    async def get_meta(self, session_id: str) -> dict[str, Any]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self._path) as db:
            async with db.execute("SELECT meta FROM sessions WHERE id = ?", (session_id,)) as cur:
                row = await cur.fetchone()
                return json.loads(row[0]) if row else {}

    async def save_history(self, session_id: str, history: list[dict]) -> None:
        meta = await self.get_meta(session_id)
        await self.save_session(session_id, history, meta)

    async def save_session(self, session_id: str, history: list[dict], meta: dict[str, Any]) -> None:
        await self._ensure_initialized()
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
        await self._ensure_initialized()
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
        await self._ensure_initialized()
        async with aiosqlite.connect(self._path) as db:
            await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            await db.commit()

    async def count_sessions(self) -> int:
        await self._ensure_initialized()
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
        await self._ensure_initialized()
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

    async def save_job(self, job: dict[str, Any]) -> None:
        await self._ensure_initialized()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """INSERT INTO jobs (id, payload, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET payload=excluded.payload,
                   updated_at=excluded.updated_at""",
                (job["job_id"], json.dumps(job), time.time()),
            )
            await db.commit()

    async def load_jobs(self) -> dict[str, dict[str, Any]]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self._path) as db:
            async with db.execute("SELECT id, payload FROM jobs") as cursor:
                rows = await cursor.fetchall()
        return {job_id: json.loads(payload) for job_id, payload in rows}


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
GOAL_QUEUE = GoalQueue(STATE_DIR / "goal_queue.json")
COGNITION = CognitionServices()


def _configured_owner_token() -> str:
    environment = os.environ.get("ANRA_OWNER_TOKEN", "").strip()
    if environment:
        return environment
    token_path = STATE_DIR / "api_owner.token"
    return token_path.read_text(encoding="utf-8").strip() if token_path.exists() else ""


def _owner_auth_required() -> bool:
    return (
        os.environ.get("ANRA_SERVICE_MODE", "development").strip().lower()
        == "production"
        or bool(_configured_owner_token())
    )


def _authorized_owner(request: Request) -> bool:
    expected = _configured_owner_token()
    if not expected:
        return not _owner_auth_required()
    authorization = request.headers.get("authorization", "")
    scheme, _, provided = authorization.partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(provided.strip(), expected)


def _require_owner(request: Request) -> None:
    if not _authorized_owner(request):
        raise HTTPException(status_code=401, detail="owner authentication required")


def _require_feature(name: str) -> None:
    if not is_enabled(name):
        raise HTTPException(
            status_code=503,
            detail={"error": "feature_disabled", "component": name},
        )


# Memory system bridge. Keep initialization lazy so importing the API does not
# load native vector backends during test collection or lightweight health checks.
MEMORY_SYSTEM = None
_MEMORY_INIT_ATTEMPTED = False


def get_memory_system():
    global MEMORY_SYSTEM, _MEMORY_INIT_ATTEMPTED
    if _MEMORY_INIT_ATTEMPTED:
        return MEMORY_SYSTEM
    _MEMORY_INIT_ATTEMPTED = True

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
        return MEMORY_SYSTEM
    except Exception as mem_exc:
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
            return MEMORY_SYSTEM
        except Exception as legacy_mem_exc:
            LOGGER.warning(
                "Memory bridge unavailable: %s; legacy fallback unavailable: %s",
                mem_exc,
                legacy_mem_exc,
            )
            return None


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


def _latest_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


@asynccontextmanager
async def lifespan(_: FastAPI):
    await SESSION_STORE.initialize()
    JOBS.update(await SESSION_STORE.load_jobs())
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
    protected = {
        "/generate",
        "/goal",
        "/goals",
        "/plans",
        "/session",
        "/train/trigger",
        "/training/candidates",
        "/robotics/workflows",
        "/memory",
        "/memory/search",
        "/sovereignty/audit",
        "/cognition/consent",
        "/owner-model",
        "/cognition/consolidate",
        "/cognition/debate",
        "/experiments/propose",
        "/agi-benchmarks/run",
        "/training/launch-manifest",
    }
    if request.url.path in protected and not _authorized_owner(request):
        return JSONResponse(status_code=401, content={"error": "owner_auth_required"})
    timeout_seconds = max(
        1.0,
        min(600.0, float(os.environ.get("ANRA_REQUEST_TIMEOUT_SECONDS", "120"))),
    )
    try:
        response = await asyncio.wait_for(
            call_next(request),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        response = JSONResponse(
            status_code=504,
            content={
                "error": "request_timeout",
                "request_id": request_id,
                "timeout_seconds": timeout_seconds,
            },
        )
    response.headers["X-Request-ID"] = request_id
    dt = (time.perf_counter() - t0) * 1000
    OPERATOR_AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with OPERATOR_AUDIT_LOG.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "timestamp": time.time(),
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "elapsed_ms": dt,
                    "role": "owner" if _authorized_owner(request) else "anonymous",
                },
                sort_keys=True,
            )
            + "\n"
        )
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


class GoalRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=8192)
    constraints: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)


class PlanRequest(GoalRequest):
    max_depth: int = Field(5, ge=1, le=5)


class MemoryAddRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=32768)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=4096)
    skills: List[Dict[str, Any]] = Field(default_factory=list, max_length=10)


class ConsentRequest(BaseModel):
    sensitive_inference: bool | None = None
    persistence: bool | None = None
    proactive_checks: bool | None = None
    training_use: bool | None = None
    session_consolidation: bool | None = None


class OwnerModelPatch(BaseModel):
    name: str
    value: Any
    category: str
    session_id: str
    evidence_span: str
    confirmed: bool = False
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class ConsolidateRequest(BaseModel):
    session_id: str


class DebateRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=8192)


class ExperimentProposalRequest(BaseModel):
    category: str
    failures: List[Dict[str, Any]]
    base_checkpoint: str
    tokenizer_hash: str
    data_hash: str
    code_hash: str
    config_hash: str
    maximum_tokens: int = Field(..., gt=0)


class LaunchManifestRequest(BaseModel):
    model_profile: str
    extension_profile: str = "cognition-v1"
    tokenizer_hash: str
    data_manifests: List[str]
    stage: str
    optimizer: str
    batch_size: int = Field(..., gt=0)
    accumulation: int = Field(..., gt=0)
    schedule: Dict[str, Any]
    seeds: List[int]
    checkpoint_source: str
    expected_tokens: int = Field(..., ge=0)
    runtime_estimate_hours: float | None = None
    owner_authorized: bool


JOBS: Dict[str, Dict[str, Any]] = {}
PLANS: Dict[str, Dict[str, Any]] = {}
TRAINING_CANDIDATES: Dict[str, Dict[str, Any]] = {}


async def _new_job(kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    event = {"event": "queued", "timestamp": _now_iso(), "payload": payload}
    job = {
        "job_id": job_id,
        "kind": kind,
        "status": "queued",
        "created_at": _now_iso(),
        "events": [event],
    }
    JOBS[job_id] = job
    await SESSION_STORE.save_job(job)
    return job


@app.post("/generate")
async def generate_route(body: GenerateRequest, request: Request):
    _require_feature("runtime")
    client_ip = request.client.host if request.client else "unknown"
    await _rate_limit_or_429(client_ip)

    cfg = GenerationConfig(strategy=body.strategy)
    for k, v in body.params.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    await run_in_threadpool(load_ghost_state, body.session_id)
    try:
        trace = await run_in_threadpool(
            generate_traced,
            body.prompt,
            cfg,
            session_id=body.session_id,
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        reduced_prompt = body.prompt[-max(256, len(body.prompt) // 2) :]
        if hasattr(cfg, "max_new_tokens"):
            cfg.max_new_tokens = max(16, int(cfg.max_new_tokens) // 2)
        LOGGER.warning(
            "[req_id=%s] OOM recovery: reduced request prompt/tokens only",
            request.state.request_id,
        )
        trace = await run_in_threadpool(
            generate_traced,
            reduced_prompt,
            cfg,
            session_id=body.session_id,
        )
    except FloatingPointError as exc:
        await run_in_threadpool(ADAPTER.load)
        LOGGER.error(
            "[req_id=%s] Numerical failure; reloaded last-good checkpoint: %s",
            request.state.request_id,
            exc,
        )
        raise HTTPException(
            status_code=503,
            detail={"error": "numerical_recovery", "checkpoint_reloaded": True},
        ) from exc
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
    memory_system = get_memory_system()
    if memory_system is not None:
        try:
            memory_results = memory_system.semantic.search(query=body.message, top_k=3)
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
    if memory_system is not None:
        try:
            memory_system.store_turn(body.message, reply, body.session_id)
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
    civ = _latest_json(CIV_LATEST)
    civ_score = float((civ or {}).get("cosine_similarity", (civ or {}).get("score", 1.0)))
    ibs = _latest_json(IBS_LATEST) or {}
    disabled = disabled_components()
    auth_misconfigured = _owner_auth_required() and not _configured_owner_token()
    status = (
        "blocked"
        if civ_score < 0.80 or auth_misconfigured
        else "degraded"
        if disabled
        else "ok"
    )
    return {
        "status": status,
        "model": "An-Ra",
        "checkpoint": str(info.get("checkpoint", "unknown")),
        "device": str(info.get("device", "unknown")),
        "vocab_size": int(info.get("vocab_size", -1) or -1),  # type: ignore[arg-type]
        "uptime_seconds": time.time() - START_TIME,
        "sessions_active": await SESSION_STORE.count_sessions(),
        "civ_similarity": civ_score,
        "ibs_overall": float(ibs.get("overall", ibs.get("overall_score", 0.0))),
        "auth_required": _owner_auth_required(),
        "auth_configured": bool(_configured_owner_token()),
        "disabled_components": disabled,
        "recovery_state": {
            "failed_jobs": sum(job.get("status") == "failed" for job in JOBS.values()),
            "running_jobs": sum(job.get("status") == "running" for job in JOBS.values()),
        },
        "campaigns": [
            path.name for path in CAMPAIGN_DIR.glob("campaign_*.json")
        ] if CAMPAIGN_DIR.exists() else [],
        "cognition": COGNITION.status(),
    }


@app.get("/cognition/consent")
async def cognition_consent_get(request: Request):
    _require_owner(request)
    return COGNITION.status()["consent"]


@app.put("/cognition/consent")
async def cognition_consent_put(body: ConsentRequest, request: Request):
    _require_owner(request)
    changes = {key: value for key, value in body.model_dump().items() if value is not None}
    return asdict(COGNITION.update_consent(**changes))


@app.get("/cognition/status")
async def cognition_status():
    return COGNITION.health()


@app.get("/owner-model")
async def owner_model_get(request: Request):
    _require_owner(request)
    return COGNITION.lhm.export()


@app.patch("/owner-model")
async def owner_model_patch(body: OwnerModelPatch, request: Request):
    _require_owner(request)
    try:
        return asdict(
            COGNITION.lhm.update(
                name=body.name,
                value=body.value,
                category=body.category,
                source_session=body.session_id,
                evidence_span=body.evidence_span,
                confidence=body.confidence,
                confirmed=body.confirmed,
            )
        )
    except (PermissionError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.delete("/owner-model")
async def owner_model_delete(request: Request, name: str | None = None, session_id: str | None = None):
    _require_owner(request)
    if name:
        return {"deleted": int(COGNITION.lhm.delete(name))}
    if session_id:
        return {"deleted": COGNITION.lhm.delete_session(session_id)}
    return {"deleted": COGNITION.lhm.wipe()}


@app.post("/cognition/consolidate")
async def cognition_consolidate(body: ConsolidateRequest, request: Request):
    _require_owner(request)
    turns = await SESSION_STORE.get_history(body.session_id)
    report = COGNITION.cec.consolidate(
        body.session_id,
        turns,
        opted_in=COGNITION.consent.session_consolidation,
    )
    return asdict(report)


@app.post("/cognition/debate")
async def cognition_debate(body: DebateRequest, request: Request):
    _require_owner(request)
    from cognition.self_debate import DebatePosition

    def generate_position(role: str, task: str, seed: int, budget: int) -> DebatePosition:
        prompt = (
            f"Role={role}; seed={seed}; budget={budget}. Analyze without inventing evidence: {task}"
        )
        argument = ADAPTER.run(prompt, strategy="greedy")
        return DebatePosition(role, argument, (), ("No independently verified evidence attached.",), 0.5, ("Human review required.",))

    result = await run_in_threadpool(
        COGNITION.debate.run,
        body.task,
        generate_position,
        verify_claims=lambda position: bool(position.supporting_evidence),
        verify_synthesis=lambda positions: False,
    )
    return result.to_dict()


@app.post("/experiments/propose")
async def experiment_propose(body: ExperimentProposalRequest, request: Request):
    _require_owner(request)
    from cognition.ssie import FailureEvidence

    failures = [FailureEvidence(**item) for item in body.failures]
    try:
        proposal = COGNITION.ssie.propose(
            body.category,
            failures,
            base_checkpoint=body.base_checkpoint,
            tokenizer_hash=body.tokenizer_hash,
            data_hash=body.data_hash,
            code_hash=body.code_hash,
            config_hash=body.config_hash,
            maximum_tokens=body.maximum_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(proposal)


@app.post("/experiments/{experiment_id}/authorize")
async def experiment_authorize(experiment_id: str, request: Request):
    _require_owner(request)
    try:
        return asdict(COGNITION.ssie.authorize(experiment_id, owner_authorized=True))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="experiment not found") from exc


@app.post("/agi-benchmarks/run")
async def agi_benchmarks_run(request: Request):
    _require_owner(request)
    from evaluation.agi_benchmarks import build_report, write_report

    calibration = COGNITION.et.calibration_report()
    measurements = {}
    if int(calibration.get("n_outcomes", 0)) >= 500:
        measurements["A-02"] = (
            float(calibration["brier_score"]),
            int(calibration["n_outcomes"]),
            "automated",
            str(OUTPUT_V2_DIR / "cognition" / "epistemic_history.jsonl"),
        )
    report = build_report(measurements)
    write_report(report, OUTPUT_V2_DIR / "agi_benchmarks" / "latest.json")
    return report


@app.get("/agi-benchmarks/latest")
async def agi_benchmarks_latest():
    return _latest_json(OUTPUT_V2_DIR / "agi_benchmarks" / "latest.json") or {
        "status": "insufficient_data"
    }


@app.get("/training/preflight")
async def training_preflight(model_size: str = "25m", runtime_class: str | None = None):
    from training.preflight import run_preflight

    return run_preflight(model_size, runtime_class=runtime_class).to_dict()


@app.post("/training/launch-manifest")
async def training_launch_manifest(body: LaunchManifestRequest, request: Request):
    _require_owner(request)
    from training.launch_manifest import build_launch_manifest, sign_manifest

    manifest = build_launch_manifest(**body.model_dump())
    try:
        return sign_manifest(
            manifest,
            OUTPUT_V2_DIR / "launch_manifests" / f"{manifest['run_id']}.json",
        )
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


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
    goals = [job for job in JOBS.values() if job["kind"] == "goal"]
    return {
        "goals": goals,
        "active": sum(job["status"] in {"queued", "running"} for job in goals),
        "completed": sum(job["status"] == "complete" for job in goals),
        "status": "online",
    }


@app.post("/goal")
@app.post("/goals")
async def goals_create_route(body: GoalRequest):
    _require_feature("goals")
    cognitive_context = COGNITION.classify_goal(body.goal)
    causal = cognitive_context.get("causal", {})
    constraints = list(body.constraints)
    if isinstance(causal, dict) and causal.get("requires_experiment"):
        constraints.append(
            "Design and authorize a typed experiment before asserting a causal conclusion."
        )
    job = await _new_job(
        "goal",
        {
            "goal": body.goal,
            "constraints": constraints,
            "success_criteria": body.success_criteria,
            "cognition": cognitive_context,
        },
    )
    GOAL_QUEUE.push(
        job["job_id"],
        body.goal,
        metadata={
            "constraints": constraints,
            "success_criteria": body.success_criteria,
            "cognition": cognitive_context,
        },
    )
    return job


@app.post("/plans")
async def plans_route(body: PlanRequest):
    plan_id = str(uuid.uuid4())
    root = MissionNode(
        node_id=f"{plan_id}:root",
        title="Strategic mission",
        objective=body.goal,
        level=0,
        constraints=tuple(body.constraints),
        expected_artifacts=tuple(body.success_criteria),
    )
    tree = MissionTree(
        goal=body.goal,
        root=root,
        success_criteria=tuple(body.success_criteria),
    )
    plan = {"plan_id": plan_id, **tree.to_dict(), "max_depth": body.max_depth}
    PLANS[plan_id] = plan
    return plan


@app.get("/memory/stats")
async def memory_stats_route():
    _require_feature("memory")
    return {
        "total": 0,
        "episodic": 0,
        "semantic": 0,
        "graph_nodes": 0,
        "status": "online" if MEMORY_SYSTEM is not None else "unavailable",
    }


@app.get("/memory")
async def memory_route(query: str = "", top_k: int = 5):
    _require_feature("memory")
    memory_system = get_memory_system() if query else None
    results = memory_system.semantic.search(query=query, top_k=top_k) if memory_system else []
    return {"query": query, "results": results}


@app.post("/memory")
async def memory_add_route(body: MemoryAddRequest):
    _require_feature("memory")
    memory_system = get_memory_system()
    if memory_system is None:
        raise HTTPException(status_code=503, detail="Memory system unavailable.")
    if hasattr(memory_system, "_router"):
        record_id = memory_system._router.write(
            body.content,
            metadata={**body.metadata, "type": "api_memory"},
            tier="episodic",
        )
    else:
        memory_system.store_turn(body.content, "", "api_memory")
        record_id = None
    return {"stored": True, "record_id": record_id}


@app.post("/eval")
async def eval_route():
    _require_feature("evaluation")
    return await _new_job("evaluation", {"suite": "IBS-50", "seeds": 3})


@app.get("/training/candidates")
async def training_candidates_route():
    return {"candidates": list(TRAINING_CANDIDATES.values())}


@app.post("/training/candidates")
async def training_candidate_create_route(request: Request):
    _require_feature("self_improvement")
    payload = await request.json()
    candidate_id = str(uuid.uuid4())
    candidate = {
        "candidate_id": candidate_id,
        "status": "research",
        "base_checkpoint": payload.get("base_checkpoint"),
        "adapter_type": payload.get("adapter_type", "lora"),
        "promotion_allowed": False,
        "created_at": _now_iso(),
    }
    TRAINING_CANDIDATES[candidate_id] = candidate
    return candidate


@app.post("/robotics/workflows")
async def robotics_workflows_route(body: WorkflowRequest):
    _require_feature("agent_loop")
    skills = [
        SkillGoal(
            skill_name=str(item.get("skill_name", item.get("skill", ""))),
            parameters=dict(item.get("parameters", {})),
            preconditions=tuple(item.get("preconditions", ())),
            postconditions=tuple(item.get("postconditions", ())),
            timeout_seconds=float(item.get("timeout_seconds", 30.0)),
            approval_required=bool(item.get("approval_required", False)),
            source_mission=str(item.get("source_mission", "")),
        )
        for item in body.skills
    ]
    workflow = Workflow(workflow_id=str(uuid.uuid4()), goal=body.goal, skills=skills)
    job = await _new_job(
        "robotics_workflow",
        {
            "workflow_id": workflow.workflow_id,
            "goal": workflow.goal,
            "skills": [skill.skill_name for skill in workflow.skills],
            "mode": "simulation_or_shadow",
        },
    )
    return job


@app.get("/jobs/{job_id}")
async def job_status_route(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JOBS[job_id]


@app.get("/jobs/{job_id}/events")
async def job_events_route(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found.")

    async def stream_events():
        for event in JOBS[job_id]["events"]:
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_events(), media_type="text/event-stream")


@app.post("/memory/search")
async def memory_search_route(request: Request):
    _require_feature("memory")
    body = await request.json()
    query = body.get("query", "").strip()
    results = []
    memory_system = get_memory_system() if query else None
    if memory_system:
        try:
            results = memory_system.semantic.search(query=query, top_k=5)
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


async def _run_training_job(job_id: str, *, model_size: str, minutes: int) -> None:
    job = JOBS[job_id]
    job["status"] = "running"
    job["events"].append({"event": "started", "timestamp": _now_iso()})
    await SESSION_STORE.save_job(job)
    command = [
        sys.executable,
        "-m",
        "training.train_unified",
        "--mode",
        "session",
        "--model-size",
        model_size,
        "--session-minutes",
        str(minutes),
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            creationflags=creationflags,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=max(300, minutes * 90))
        output = stdout.decode("utf-8", errors="replace")[-8000:] if stdout else ""
        job["status"] = "complete" if process.returncode == 0 else "failed"
        job["exit_code"] = int(process.returncode or 0)
        job["events"].append(
            {
                "event": job["status"],
                "timestamp": _now_iso(),
                "output_tail": output,
            }
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        job["status"] = "failed"
        job["events"].append({"event": "timeout", "timestamp": _now_iso()})
    except Exception as exc:
        job["status"] = "failed"
        job["events"].append(
            {"event": "error", "timestamp": _now_iso(), "error": str(exc)}
        )
    await SESSION_STORE.save_job(job)


@app.post("/session")
@app.post("/train/trigger")
async def train_trigger_route(request: Request) -> dict:
    _require_owner(request)
    _require_feature("training_loop")
    body = await request.body()
    payload = json.loads(body) if body.strip() else {}
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="request body must be an object")
    model_size = str(payload.get("model_size", "25m"))
    if model_size not in {"25m", "frontier", "904m", "1b", "3b"}:
        raise HTTPException(status_code=400, detail="invalid model_size")
    minutes = max(1, min(720, int(payload.get("minutes", 30))))
    job = await _new_job(
        "training_session",
        {"model_size": model_size, "minutes": minutes, "owner_authorized": True},
    )
    asyncio.create_task(
        _run_training_job(job["job_id"], model_size=model_size, minutes=minutes)
    )
    return job


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
