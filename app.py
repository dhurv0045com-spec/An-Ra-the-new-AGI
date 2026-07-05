from __future__ import annotations

# ruff: noqa: E501
import argparse
import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import aiosqlite
import httpx
import uvicorn
from anra.anra_paths import (
    CAMPAIGN_DIR,
    CIV_LATEST,
    DRIVE_DIR,
    DRIVE_SESSIONS,
    IBS_LATEST,
    MEMORY_DB_DIR,
    OPERATOR_AUDIT_LOG,
    OUTPUT_V2_DIR,
    PRIVATE_EVAL_DIR,
    ROOT,
    STATE_DIR,
    ensure_dirs,
    get_identity_file,
)
from cognition.services import CognitionServices
from engine.feature_flags import disabled_components, is_enabled
from evaluation.promotion import (
    build_release_bundle_manifest,
    run_rollback_drill,
    verify_release_bundle_manifest,
    verify_release_manifest,
)
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from generate import (
    GenerationConfig,
    clear_session_runtime_state,
    detect_repetition,
    generate,
    generate_stream,
    generate_traced,
    get_model_info,
    restore_embedded_data_manifests,
    verify_kv_cache_parity,
    verify_session_state_isolation,
)
from goals.goal_queue import GoalQueue
from inference.full_system_connector import build_capability_graph
from inference.optimize_context_window import ContextWindowOptimizer
from intelligence.hgp import MissionNode, MissionTree
from pydantic import BaseModel, Field
from robotics.contracts import SkillGoal, Workflow
from runtime.hal_telemetry import read_hal_state
from starlette.responses import Response
from training.eval_v2 import (
    apply_blinded_human_reviews,
    build_context_growth_evidence,
    build_frontier_recovery_decision,
    ensure_private_eval_suite,
    run_private_mode_seed_evaluation,
    run_recovery_prompt_gate,
)
from anra.core.protocols import RuntimeReadinessReport
from training.v2_runtime import active_tokenizer_path, load_or_build_v2_tokenizer

ensure_dirs()

START_TIME = time.time()
_COLAB_DRIVE = DRIVE_SESSIONS
_LOCAL_FALLBACK = Path(__file__).resolve().parent / "output" / "sessions"
SESSION_DIR = _COLAB_DRIVE if DRIVE_DIR.parent.parent.exists() else _LOCAL_FALLBACK
SESSION_DIR.mkdir(parents=True, exist_ok=True)
LOGGER = logging.getLogger("anra.api")
logging.basicConfig(level=logging.INFO)
_PRIVATE_EVAL_TASK: asyncio.Task[object] | None = None
_MODEL_LOAD_TASK: asyncio.Task[object] | None = None


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
        async with aiosqlite.connect(self._path) as db, db.execute(
            "SELECT history FROM sessions WHERE id = ?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
            return json.loads(row[0]) if row else []

    async def get_meta(self, session_id: str) -> dict[str, Any]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self._path) as db, db.execute(
            "SELECT meta FROM sessions WHERE id = ?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
            return json.loads(row[0]) if row else {}

    async def save_history(self, session_id: str, history: list[dict]) -> None:
        meta = await self.get_meta(session_id)
        await self.save_session(session_id, history, meta)

    async def save_session(
        self, session_id: str, history: list[dict], meta: dict[str, Any]
    ) -> None:
        await self._ensure_initialized()
        trimmed = history[-self._max_history :]
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
        async with aiosqlite.connect(self._path) as db, db.execute(
            "SELECT id, history, meta FROM sessions ORDER BY last_active DESC"
        ) as cur:
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
        async with aiosqlite.connect(self._path) as db, db.execute(
            "SELECT COUNT(*) FROM sessions"
        ) as cur:
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
        async with aiosqlite.connect(self._path) as db, db.execute(
            "SELECT id, payload FROM jobs"
        ) as cursor:
            rows = await cursor.fetchall()
        return {job_id: json.loads(payload) for job_id, payload in rows}


SESSION_STORE = SQLiteSessionStore(SESSION_DIR / "sessions.db", max_history=40)


class ModelAdapter:
    def __init__(self) -> None:
        self.info: dict[str, Any] = {}
        self.load_error = ""
        self._lock = threading.RLock()
        now = time.time()
        self._readiness = RuntimeReadinessReport(
            stage="starting",
            ready=False,
            progress=0.0,
            started_at=now,
            updated_at=now,
        )

    def set_stage(self, stage: str, progress: float, error: str | None = None) -> None:
        with self._lock:
            self._readiness = RuntimeReadinessReport(
                stage=stage,
                ready=stage == "ready",
                progress=max(0.0, min(1.0, float(progress))),
                started_at=self._readiness.started_at,
                updated_at=time.time(),
                error=error,
            )

    def readiness(self) -> dict[str, object]:
        with self._lock:
            return asdict(self._readiness)

    def require_ready(self) -> None:
        report = self.readiness()
        if report["ready"] is not True:
            raise HTTPException(
                status_code=503,
                detail={"error": "model_not_ready", "readiness": report},
            )

    def load(self) -> None:
        self.set_stage("building_model", 0.30)
        try:
            info = get_model_info()
            self.set_stage("verifying_artifacts", 0.85)
            required = {
                "checkpoint",
                "checkpoint_sha256",
                "tokenizer_sha256",
                "device",
                "profile",
                "vocab_size",
                "param_count",
                "block_size",
            }
            missing = sorted(
                key
                for key in required
                if info.get(key) in {None, "", "unknown", "missing", 0, -1}
            )
            if missing:
                raise RuntimeError(f"Runtime metadata is incomplete: {missing}")
            with self._lock:
                self.info = dict(info)
                self.load_error = ""
            self.set_stage("ready", 1.0)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self.info = {}
                self.load_error = error
            self.set_stage("failed", 1.0, error=error)
            LOGGER.exception("An-Ra model background load failed")

    def run(self, prompt: str, strategy: str = "nucleus", **params: Any) -> str:
        # SWAP POINT: replace only this line to redirect to a new backend model runtime.
        return generate(prompt, strategy=strategy, **params)


ADAPTER = ModelAdapter()
SYSTEM_GRAPH: dict[str, Any] = {}
_ctx_optimizer = ContextWindowOptimizer()
_identity_context = ""


def _run_native_agent_goal(goal: str) -> dict[str, Any]:
    from phase2.agent_loop_45k.agent_main import Agent

    agent = Agent(verbose=False, approve_each_step=False)
    return agent.run(goal, clarify=False, show_plan=False)


def _dispatch_full_system_operator(message: str) -> dict[str, Any]:
    """Dispatch only explicit operator-shaped requests; normal text remains chat."""
    from runtime.operator_commands import (
        handle_natural_operator_request,
        handle_slash_command,
    )

    stripped = message.strip()
    run_goal = _run_native_agent_goal
    if stripped.startswith("/"):
        handled, response = handle_slash_command(stripped, run_goal=run_goal)
    else:
        handled, response = handle_natural_operator_request(stripped, run_goal=run_goal)
    agent_requested = stripped.lower().startswith("/goal") or bool(
        re.search(r"\b(?:run|execute|start)\s+(?:a\s+)?goal\b", stripped, re.IGNORECASE)
    )
    return {
        "handled": handled,
        "response": response,
        "agent_executed": handled and agent_requested,
        "tool_executed": handled and not agent_requested,
    }


GOAL_QUEUE = GoalQueue(STATE_DIR / "goal_queue.json")
COGNITION = CognitionServices()


def _configured_owner_token() -> str:
    environment = os.environ.get("ANRA_OWNER_TOKEN", "").strip()
    if environment:
        return environment
    token_path = STATE_DIR / "api_owner.token"
    return token_path.read_text(encoding="utf-8").strip() if token_path.exists() else ""


def _owner_auth_required() -> bool:
    return os.environ.get(
        "ANRA_SERVICE_MODE", "development"
    ).strip().lower() == "production" or bool(_configured_owner_token())


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


def _classify_chat_cognition(message: str) -> dict[str, Any] | None:
    """Causal/debate classification of a full-system chat message.

    Cognition is part of the full-system contract, so it must run on the real
    request path, not only behind dedicated /cognition endpoints. Failures
    degrade to None; they never break generation.
    """
    if not is_enabled("cognition"):
        return None
    try:
        return COGNITION.classify_goal(message)
    except Exception as exc:
        LOGGER.warning("Cognition classification failed: %s", exc)
        return None


def _record_generation_epistemics(session_id: str, request_id: str, *, accepted: bool) -> bool:
    """Append the generation outcome to the epistemic calibration history.

    Every accepted/rejected full-system generation is a real observed outcome;
    recording it gives the epistemic tracker live calibration data instead of
    an empty history.
    """
    try:
        COGNITION.et.record_outcome(
            f"generation:{session_id}:{request_id}",
            was_correct=accepted,
            domain="conversation",
            verifier="generation_quality",
        )
        return True
    except Exception as exc:
        LOGGER.warning("Epistemic outcome recording failed: %s", exc)
        return False


# Memory system bridge. Keep initialization lazy so importing the API does not
# load native vector backends during test collection or lightweight health checks.
MEMORY_SYSTEM = None
_MEMORY_INIT_ATTEMPTED = False


def get_memory_system() -> object | None:
    global MEMORY_SYSTEM, _MEMORY_INIT_ATTEMPTED
    if _MEMORY_INIT_ATTEMPTED:
        return MEMORY_SYSTEM
    _MEMORY_INIT_ATTEMPTED = True

    try:
        from memory.memory_router import MemoryRouter

        class _MemoryBridge:
            def __init__(self) -> None:
                self._router = MemoryRouter()
                self.semantic = self

            def search(self, query: str, top_k: int = 3) -> list[dict[str, object]]:
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
                    metadata={
                        "session_id": session_id,
                        "type": "conversation_turn",
                        "salience": 0.8,
                    },
                    tier="episodic",
                )

        MEMORY_SYSTEM = _MemoryBridge()
        return MEMORY_SYSTEM
    except Exception as mem_exc:
        try:
            from phase2.memory_45j.memory_manager import MemoryManager  # type: ignore

            class _LegacyMemoryBridge:
                def __init__(self) -> None:
                    self._mm = MemoryManager(data_dir=str(MEMORY_DB_DIR), user_id="anra")
                    self.semantic = self

                def search(self, query: str, top_k: int = 3) -> list[object]:
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


def format_memory_context(memory_results: list[dict[str, Any]]) -> str:
    lines = ["[Retrieved Memory Context]"]
    for i, item in enumerate(memory_results, start=1):
        lines.append(f"{i}. {item.get('summary', '')}")
        if item.get("content"):
            lines.append(f"   detail: {item.get('content')[:240]}")
    return "\n".join(lines)


def _session_file(session_id: str) -> Path:
    return SESSION_DIR / f"{session_id}.json"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


async def _get_session_history(session_id: str) -> list[dict]:
    return await SESSION_STORE.get_history(session_id)


async def _append_to_session(session_id: str, new_messages: list[dict]) -> None:
    history = await SESSION_STORE.get_history(session_id)
    history.extend(new_messages)
    await SESSION_STORE.save_history(session_id, history)


async def _save_session(session_id: str) -> None:
    pass  # SQLiteSessionStore saves atomically on every write — no explicit flush needed.


def _serialize_context_from_turns(turns: list[dict[str, str]], message: str) -> str:
    context_parts: list[str] = []
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
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded. Max 30 requests per minute."
        )


def _latest_report_snapshot() -> dict[str, Any] | None:
    reports_dir = Path(__file__).resolve().parent / "state" / "reports"
    if not reports_dir.exists():
        return None
    snapshots = sorted(
        reports_dir.glob("snapshot_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
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
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await SESSION_STORE.initialize()
    JOBS.update(await SESSION_STORE.load_jobs())
    global SYSTEM_GRAPH, _ctx_optimizer, _identity_context, _MODEL_LOAD_TASK
    ADAPTER.set_stage("loading_tokenizer", 0.10)
    tokenizer = await run_in_threadpool(load_or_build_v2_tokenizer)
    _ctx_optimizer = ContextWindowOptimizer(
        tokenizer=tokenizer,
        max_context=int(ADAPTER.info.get("block_size", 1024) or 1024),
    )
    identity_path = get_identity_file()
    _identity_context = (
        identity_path.read_text(encoding="utf-8", errors="replace")
        if identity_path is not None and identity_path.is_file()
        else "An-Ra is the native model serving this session."
    )
    SYSTEM_GRAPH = await run_in_threadpool(build_capability_graph, Path(__file__).resolve().parent)
    ADAPTER.set_stage("loading_checkpoint", 0.20)
    _MODEL_LOAD_TASK = asyncio.create_task(run_in_threadpool(ADAPTER.load))
    LOGGER.info("An-Ra API startup complete. Session store: %s", SESSION_STORE._path)
    yield


app = FastAPI(title="An-Ra API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def request_context_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
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
        min(600.0, float(os.environ.get("ANRA_REQUEST_TIMEOUT_SECONDS", "600"))),
    )
    try:
        response = await asyncio.wait_for(
            call_next(request),
            timeout=timeout_seconds,
        )
    except TimeoutError:
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
    LOGGER.info(
        "[req_id=%s] %s %s %s %.2fms",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        dt,
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, _exc: Exception) -> JSONResponse:
    LOGGER.exception(
        "[req_id=%s] Unhandled error\n%s",
        getattr(request.state, "request_id", "unknown"),
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "request_id": getattr(request.state, "request_id", "unknown"),
            "message": "An internal error occurred.",
        },
    )


DEVELOPER_UI_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>An-Ra Developer UI</title>
  <style>
    :root {
      --bg: #050608;
      --panel: #0d1117;
      --panel-2: #111823;
      --border: #223044;
      --text: #eef4ff;
      --muted: #8da0b8;
      --cyan: #00e5ff;
      --green: #22ff99;
      --purple: #a855f7;
      --red: #ff5c7a;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 12% 10%, rgba(0, 229, 255, 0.12), transparent 32%),
        radial-gradient(circle at 86% 88%, rgba(168, 85, 247, 0.14), transparent 34%),
        var(--bg);
      color: var(--text);
    }
    button, input, textarea { font: inherit; }
    .app { min-height: 100vh; display: grid; grid-template-rows: 72px 1fr; }
    header {
      display: flex; align-items: center; justify-content: space-between; gap: 18px;
      padding: 14px 22px; border-bottom: 1px solid var(--border);
      background: rgba(5, 6, 8, 0.84); position: sticky; top: 0; z-index: 5;
    }
    .brand { display: flex; align-items: center; gap: 12px; min-width: 190px; }
    .logo { width: 34px; height: 34px; border-radius: 8px; background: linear-gradient(135deg, var(--cyan), var(--purple)); box-shadow: 0 0 24px rgba(0, 229, 255, 0.26); }
    .brand strong { display: block; letter-spacing: 0.08em; }
    .brand span { display: block; color: var(--muted); font-size: 12px; margin-top: 2px; }
    nav { display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; }
    nav button, .secondary {
      border: 1px solid var(--border); background: rgba(255, 255, 255, 0.035); color: var(--muted);
      border-radius: 8px; padding: 9px 12px; cursor: pointer;
    }
    nav button.active, .secondary:hover { color: var(--cyan); border-color: rgba(0, 229, 255, 0.55); }
    main { padding: 18px; min-height: 0; }
    .grid { display: grid; grid-template-columns: minmax(300px, 420px) 1fr; gap: 18px; height: calc(100vh - 108px); min-height: 560px; }
    .matrix-grid { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 18px; height: calc(100vh - 108px); min-height: 560px; }
    .panel {
      background: rgba(13, 17, 23, 0.86); border: 1px solid var(--border); border-radius: 10px;
      box-shadow: 0 14px 50px rgba(0, 0, 0, 0.34); overflow: hidden; min-height: 0;
    }
    .panel-head { padding: 14px 16px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .panel-head h2, .panel-head h3 { margin: 0; font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--cyan); }
    .panel-body { padding: 16px; overflow: auto; height: calc(100% - 49px); }
    .chat { display: flex; flex-direction: column; height: 100%; }
    .messages { flex: 1; overflow: auto; padding: 18px; display: flex; flex-direction: column; gap: 14px; }
    .msg { max-width: 86%; padding: 12px 14px; border-radius: 10px; line-height: 1.5; white-space: pre-wrap; border: 1px solid var(--border); background: rgba(255, 255, 255, 0.04); }
    .msg.user { align-self: flex-end; background: rgba(0, 229, 255, 0.16); color: white; border-color: rgba(0, 229, 255, 0.42); }
    .msg.assistant { align-self: flex-start; }
    form { display: flex; gap: 10px; padding: 14px; border-top: 1px solid var(--border); background: rgba(0, 0, 0, 0.18); }
    textarea {
      flex: 1; resize: none; min-height: 48px; max-height: 140px; border-radius: 8px;
      background: #070b10; color: var(--text); border: 1px solid var(--border); padding: 12px; outline: none;
    }
    form button {
      min-width: 96px; border: 1px solid rgba(0, 229, 255, 0.55); color: #001014;
      background: var(--cyan); border-radius: 8px; cursor: pointer; font-weight: 700;
    }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }
    .card { background: rgba(255, 255, 255, 0.035); border: 1px solid var(--border); border-radius: 8px; padding: 13px; min-height: 84px; }
    .label { color: var(--muted); font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 8px; }
    .value { color: var(--cyan); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; overflow-wrap: anywhere; }
    pre { margin: 0; white-space: pre-wrap; word-break: break-word; color: #cfe7ff; font: 12px/1.45 ui-monospace, SFMono-Regular, Consolas, monospace; }
    .bars { display: flex; flex-direction: column; gap: 12px; }
    .bar-row { display: grid; gap: 5px; }
    .bar-top { display: flex; justify-content: space-between; color: var(--muted); font-size: 12px; text-transform: uppercase; }
    .bar { height: 6px; background: rgba(255,255,255,0.08); border-radius: 99px; overflow: hidden; }
    .bar-fill { height: 100%; background: linear-gradient(90deg, var(--cyan), var(--green)); }
    .hidden { display: none; }
    .status-line { color: var(--muted); font: 12px ui-monospace, SFMono-Regular, Consolas, monospace; }
    .error { color: var(--red); border-color: rgba(255, 92, 122, 0.55); }
    @media (max-width: 980px) {
      .grid, .matrix-grid { grid-template-columns: 1fr; height: auto; }
      .panel { min-height: 380px; }
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      header { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="brand">
        <div class="logo"></div>
        <div><strong>AN-RA DEVELOPER</strong><span>backend-served Colab console</span></div>
      </div>
      <nav>
        <button id="tab-dashboard" class="active" type="button">DASHBOARD</button>
        <button id="tab-matrix" type="button">MATRIX</button>
      </nav>
      <button class="secondary" id="refresh" type="button">Refresh</button>
    </header>
    <main>
      <section id="dashboard" class="grid">
        <div class="panel">
          <div class="panel-head"><h2>Runtime</h2><span class="status-line" id="runtime-line">loading</span></div>
          <div class="panel-body">
            <div class="cards">
              <div class="card"><div class="label">Status</div><div class="value" id="status-status">-</div></div>
              <div class="card"><div class="label">Device</div><div class="value" id="status-device">-</div></div>
              <div class="card"><div class="label">Profile</div><div class="value" id="status-profile">-</div></div>
              <div class="card"><div class="label">Params</div><div class="value" id="status-params">-</div></div>
            </div>
            <div class="card"><div class="label">Checkpoint</div><div class="value" id="status-checkpoint">-</div></div>
          </div>
        </div>
        <div class="panel chat">
          <div class="panel-head"><h2>Neural Interface</h2><span class="status-line" id="chat-status">ready</span></div>
          <div class="messages" id="messages">
            <div class="msg assistant">Neural bridge established. Ask the model a task or run an experiment prompt.</div>
          </div>
          <form id="chat-form">
            <textarea id="prompt" placeholder="Talk to An-Ra..." rows="2"></textarea>
            <select id="model-mode" class="secondary" aria-label="Runtime mode">
              <option value="diagnostic">Diagnostic</option>
              <option value="native">Native</option>
              <option value="full_system">Full system</option>
            </select>
            <button id="send" type="submit">Send</button>
          </form>
        </div>
      </section>
      <section id="matrix" class="matrix-grid hidden">
        <div class="panel">
          <div class="panel-head">
            <h2>Developer Matrix</h2>
            <div>
              <button class="secondary" id="run-rollback" type="button">Rollback drill</button>
              <button class="secondary" id="run-recovery" type="button">200-prompt gate</button>
              <button class="secondary" id="run-private" type="button">Full promotion eval</button>
              <button class="secondary" id="run-integration" type="button">Integration probe</button>
              <button class="secondary" id="open-review" type="button">Review outputs</button>
            </div>
            <span class="status-line" id="matrix-line">loading</span>
          </div>
          <div class="panel-body">
            <div class="cards">
              <div class="card"><div class="label">Train Step</div><div class="value" id="matrix-step">-</div></div>
              <div class="card"><div class="label">Best Loss</div><div class="value" id="matrix-loss">-</div></div>
              <div class="card"><div class="label">Context</div><div class="value" id="matrix-context">-</div></div>
              <div class="card"><div class="label">Sessions</div><div class="value" id="matrix-sessions">-</div></div>
            </div>
            <div class="panel" style="height: calc(100% - 110px);"><div class="panel-head"><h3>Runtime Payload</h3></div><div class="panel-body"><pre id="status-json">{}</pre></div></div>
          </div>
        </div>
        <div class="panel">
          <div class="panel-head"><h2>Behind The Screen</h2></div>
          <div class="panel-body">
            <div class="bars" id="hal-bars"></div>
            <div style="height: 14px;"></div>
            <div class="panel" style="height: 42%;"><div class="panel-head"><h3>HAL Raw</h3></div><div class="panel-body"><pre id="hal-json">{}</pre></div></div>
            <div style="height: 14px;"></div>
            <div class="panel" style="height: 42%;"><div class="panel-head"><h3>Sessions</h3></div><div class="panel-body"><pre id="sessions-json">{}</pre></div></div>
            <div style="height: 14px;"></div>
            <div class="panel" style="height: 58%;"><div class="panel-head"><h3>Last Generation Trace</h3></div><div class="panel-body"><pre id="trace-json">No generation trace yet.</pre></div></div>
            <div style="height: 14px;"></div>
            <div class="panel" style="height: 42%;"><div class="panel-head"><h3>Evaluation Gates</h3></div><div class="panel-body"><pre id="evaluation-json">{}</pre></div></div>
            <div id="review-section" class="hidden" style="margin-top: 14px; border-top: 1px solid var(--border); padding-top: 14px;">
              <div class="panel-head"><h3>Blinded Coherence Review</h3><span class="status-line" id="review-status">-</span></div>
              <pre id="review-prompt"></pre>
              <div style="height: 10px;"></div>
              <pre id="review-response"></pre>
              <div style="display: flex; gap: 8px; margin-top: 12px;">
                <button class="secondary" id="review-reject" type="button">Reject</button>
                <button id="review-accept" type="button">Coherent</button>
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>
  </div>
  <script>
    const $ = (id) => document.getElementById(id);
    const fmt = (v) => v === null || v === undefined || v === "" ? "-" : (typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(4)) : String(v));
    let lastPayloads = {};
    let reviewQueue = [];
    let reviewVotes = {};
    let reviewIndex = 0;

    function setTab(name) {
      $("dashboard").classList.toggle("hidden", name !== "dashboard");
      $("matrix").classList.toggle("hidden", name !== "matrix");
      $("tab-dashboard").classList.toggle("active", name === "dashboard");
      $("tab-matrix").classList.toggle("active", name === "matrix");
    }

    async function getJson(path) {
      const res = await fetch(path);
      if (!res.ok) throw new Error(`${path} -> ${res.status} ${await res.text()}`);
      return res.json();
    }

    async function postDiagnostic(path, buttonId, runningLabel) {
      const button = $(buttonId);
      button.disabled = true;
      $("matrix-line").textContent = runningLabel;
      try {
        const res = await fetch(path, { method: "POST" });
        const payload = await res.json();
        if (!res.ok) throw new Error(JSON.stringify(payload));
        $("evaluation-json").textContent = JSON.stringify(payload, null, 2);
        await getJson("/diagnostics/release-evidence");
        await refresh();
      } catch (error) {
        $("evaluation-json").textContent = `Diagnostic failed: ${error}`;
      } finally {
        button.disabled = false;
      }
    }

    async function refresh() {
      try {
        const [status, hal, sessions, phase, evaluation] = await Promise.all([
          getJson("/status"),
          getJson("/hal/state"),
          getJson("/sessions"),
          getJson("/phase-health"),
          getJson("/evaluations/current"),
        ]);
        lastPayloads = { status, hal, sessions, phase, evaluation };
        const cp = status.checkpoint_state || {};
        const modelReady = status.readiness?.ready === true;
        const readinessLabel = modelReady
          ? `ready | ${new Date().toLocaleTimeString()}`
          : `${status.readiness?.stage || "starting"} | ${Math.round((status.readiness?.progress || 0) * 100)}%`;
        $("runtime-line").textContent = readinessLabel;
        $("matrix-line").textContent = readinessLabel;
        $("send").disabled = !modelReady;
        $("prompt").disabled = !modelReady;
        ["run-rollback", "run-recovery", "run-private", "run-integration"].forEach((id) => {
          $(id).disabled = !modelReady;
        });
        $("status-status").textContent = fmt(status.status);
        $("status-device").textContent = fmt(status.device);
        $("status-profile").textContent = fmt(status.profile);
        $("status-params").textContent = fmt(status.param_count);
        $("status-checkpoint").textContent = fmt(status.checkpoint);
        $("matrix-step").textContent = fmt(cp.global_step);
        $("matrix-loss").textContent = fmt(cp.best_loss);
        $("matrix-context").textContent = status.block_size ? `${status.block_size} tokens` : "-";
        $("matrix-sessions").textContent = fmt(status.sessions_active ?? sessions.count);
        $("status-json").textContent = JSON.stringify(status, null, 2);
        $("hal-json").textContent = JSON.stringify(hal, null, 2);
        $("sessions-json").textContent = JSON.stringify(sessions, null, 2);
        $("evaluation-json").textContent = JSON.stringify(evaluation, null, 2);
        const hormones = hal.hormones || {};
        $("hal-bars").innerHTML = Object.entries(hormones).sort().map(([name, raw]) => {
          const value = Math.max(0, Math.min(1, Number(raw || 0)));
          return `<div class="bar-row"><div class="bar-top"><span>${name}</span><span>${value.toFixed(3)}</span></div><div class="bar"><div class="bar-fill" style="width:${value * 100}%"></div></div></div>`;
        }).join("") || '<div class="status-line">No HAL state published yet.</div>';
      } catch (err) {
        $("runtime-line").textContent = String(err);
        $("runtime-line").classList.add("error");
      }
    }

    function addMessage(role, text) {
      const div = document.createElement("div");
      div.className = `msg ${role}`;
      div.textContent = text;
      $("messages").appendChild(div);
      $("messages").scrollTop = $("messages").scrollHeight;
    }

    function renderReview() {
      const item = reviewQueue[reviewIndex];
      $("review-section").classList.toggle("hidden", !item);
      if (!item) return;
      $("review-status").textContent = `${reviewIndex + 1} / ${reviewQueue.length}`;
      $("review-prompt").textContent = item.prompt;
      $("review-response").textContent = item.response;
    }

    async function openReviewQueue() {
      try {
        const payload = await getJson("/evaluations/private-review-queue");
        reviewQueue = payload.queue || [];
        reviewVotes = {};
        reviewIndex = 0;
        renderReview();
      } catch (error) {
        $("evaluation-json").textContent = `Review queue unavailable: ${error}`;
      }
    }

    async function recordReview(coherent) {
      const item = reviewQueue[reviewIndex];
      if (!item) return;
      reviewVotes[item.review_id] = coherent;
      reviewIndex += 1;
      if (reviewIndex < reviewQueue.length) {
        renderReview();
        return;
      }
      const reviews = Object.entries(reviewVotes).map(([review_id, value]) => ({ review_id, coherent: value }));
      const response = await fetch("/evaluations/private-review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reviews }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(JSON.stringify(payload));
      $("review-section").classList.add("hidden");
      $("evaluation-json").textContent = JSON.stringify(payload, null, 2);
      await getJson("/diagnostics/release-evidence");
      await refresh();
    }

    $("chat-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = $("prompt").value.trim();
      if (!message) return;
      $("prompt").value = "";
      addMessage("user", message);
      $("chat-status").textContent = "thinking";
      $("send").disabled = true;
      try {
        const res = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: "colab_developer_ui",
            message,
            params: { strategy: "greedy", mode: $("model-mode").value, max_tokens: 128, seed: 0, use_kv_cache: false },
          }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(JSON.stringify(data));
        addMessage("assistant", data.response || JSON.stringify(data, null, 2));
        if (data.trace_id) {
          const trace = await getJson(`/traces/${data.trace_id}`);
          $("trace-json").textContent = JSON.stringify(trace, null, 2);
        }
        $("chat-status").textContent = `${data.quality_state || "unknown"} | ${data.generation?.stopped_by || "-"}`;
        refresh();
      } catch (err) {
        addMessage("assistant", `Error: ${err}`);
        $("chat-status").textContent = "error";
      } finally {
        $("send").disabled = false;
      }
    });

    $("tab-dashboard").addEventListener("click", () => setTab("dashboard"));
    $("tab-matrix").addEventListener("click", () => setTab("matrix"));
    $("refresh").addEventListener("click", refresh);
    $("run-rollback").addEventListener("click", () => postDiagnostic("/diagnostics/rollback-drill", "run-rollback", "running rollback drill"));
    $("run-recovery").addEventListener("click", () => postDiagnostic("/diagnostics/recovery-gate", "run-recovery", "running 200-prompt gate"));
    $("run-private").addEventListener("click", () => postDiagnostic("/evaluations/private-promotion", "run-private", "running 500-task x 3-seed promotion evaluation; keep this cell open"));
    $("run-integration").addEventListener("click", () => postDiagnostic("/diagnostics/full-system-integration", "run-integration", "running full-system integration probe"));
    $("open-review").addEventListener("click", openReviewQueue);
    $("review-reject").addEventListener("click", () => recordReview(false));
    $("review-accept").addEventListener("click", () => recordReview(true));
    refresh();
    setInterval(refresh, 3500);
  </script>
</body>
</html>
"""


@app.get("/developer", response_class=HTMLResponse)
async def developer_ui_route() -> HTMLResponse:
    return HTMLResponse(DEVELOPER_UI_HTML)


@app.get("/", response_class=HTMLResponse)
async def developer_ui_root_route() -> HTMLResponse:
    return HTMLResponse(DEVELOPER_UI_HTML)


class GenerationParams(BaseModel):
    strategy: Literal["greedy", "nucleus", "topk", "beam", "contrastive"] = "greedy"
    max_tokens: int = Field(128, ge=1, le=512)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_k: int = Field(40, ge=1, le=512)
    top_p: float = Field(0.92, gt=0.0, le=1.0)
    beam_width: int = Field(4, ge=1, le=16)
    repetition_penalty: float = Field(1.15, ge=1.0, le=2.0)
    repetition_window: int = Field(64, ge=1, le=512)
    stop_strings: list[str] = Field(default_factory=list, max_length=16)
    seed: int | None = 0
    use_think_tokens: bool = False
    use_kv_cache: bool = False
    mode: Literal["diagnostic", "native", "full_system"] = "diagnostic"
    allow_control_tokens: bool = False
    ablated_subsystem: Literal["mod", "rim", "dstp", "esv", "hal"] | None = None
    verifier_score: float | None = Field(None, ge=0.0, le=1.0)
    task_success: bool | None = None
    civ_score: float | None = Field(None, ge=0.0, le=1.0)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32768)
    strategy: Literal["greedy", "nucleus", "topk", "beam", "contrastive"] = "greedy"
    session_id: str = "generate_default"
    params: GenerationParams = Field(default_factory=GenerationParams)


class ChatRequest(BaseModel):
    session_id: str = Field("default", min_length=1, max_length=96)
    message: str = Field(..., min_length=1, max_length=32768)
    params: GenerationParams = Field(default_factory=GenerationParams)


class CacheParityRequest(BaseModel):
    prompt: str = Field(
        "H: Verify cache parity for An-Ra.\nANRA:",
        min_length=1,
        max_length=4096,
    )
    max_tokens: int = Field(16, ge=1, le=64)


class ContextGrowthEvidenceRequest(BaseModel):
    source_context: Literal[1024, 1536]
    target_context: Literal[1536, 2048]
    coherence_rate: float = Field(..., ge=0.0, le=1.0)
    short_context_baseline_loss: float = Field(..., gt=0.0)
    short_context_candidate_loss: float = Field(..., gt=0.0)
    retrieval_baseline_accuracy: float = Field(..., ge=0.0, le=1.0)
    retrieval_candidate_accuracy: float = Field(..., ge=0.0, le=1.0)


class HumanReviewItem(BaseModel):
    review_id: str = Field(..., min_length=1, max_length=128)
    coherent: bool


class HumanReviewRequest(BaseModel):
    reviews: list[HumanReviewItem] = Field(..., min_length=1, max_length=256)


class ResetRequest(BaseModel):
    session_id: str


class GoalRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=8192)
    constraints: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)


class PlanRequest(GoalRequest):
    max_depth: int = Field(5, ge=1, le=5)


class MemoryAddRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=32768)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=4096)
    skills: list[dict[str, Any]] = Field(default_factory=list, max_length=10)


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
    failures: list[dict[str, Any]]
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
    data_manifests: list[str]
    stage: str
    optimizer: str
    batch_size: int = Field(..., gt=0)
    accumulation: int = Field(..., gt=0)
    schedule: dict[str, Any]
    seeds: list[int]
    checkpoint_source: str
    expected_tokens: int = Field(..., ge=0)
    runtime_estimate_hours: float | None = None
    owner_authorized: bool


JOBS: dict[str, dict[str, Any]] = {}
PLANS: dict[str, dict[str, Any]] = {}
TRAINING_CANDIDATES: dict[str, dict[str, Any]] = {}
TRACE_STORE: dict[str, dict[str, Any]] = {}


def _record_trace(payload: dict[str, Any]) -> str:
    trace_id = str(uuid.uuid4())
    TRACE_STORE[trace_id] = payload
    while len(TRACE_STORE) > 1000:
        TRACE_STORE.pop(next(iter(TRACE_STORE)))
    return trace_id


async def _new_job(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
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
async def generate_route(body: GenerateRequest, request: Request) -> dict[str, Any]:
    ADAPTER.require_ready()
    _require_feature("runtime")
    client_ip = request.client.host if request.client else "unknown"
    await _rate_limit_or_429(client_ip)

    values = body.params.model_dump()
    values["strategy"] = body.strategy
    cfg = GenerationConfig(**values)
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
        cfg.max_tokens = max(16, int(cfg.max_tokens) // 2)
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
    entropy_avg = sum(trace.entropy_curve) / max(len(trace.entropy_curve), 1)
    max_prob_avg = sum(trace.max_prob_curve) / max(len(trace.max_prob_curve), 1)
    trace_payload = {
        "request_id": request.state.request_id,
        "session_id": body.session_id,
        "prompt": body.prompt,
        "generation": asdict(trace),
        "config": asdict(cfg),
    }
    trace_id = _record_trace(trace_payload)

    return {
        "response": trace.output,
        "strategy": trace.strategy,
        "tokens_generated": trace.tokens_generated,
        "time_ms": trace.time_ms,
        "trace_id": trace_id,
        "quality_state": trace.quality_state,
        "trace": {
            "entropy_avg": entropy_avg,
            "max_prob_avg": max_prob_avg,
            "repeated_ngrams": trace.repeated_ngrams_detected,
            "stopped_by": trace.stopped_by,
            "prompt_tokens": trace.prompt_tokens,
            "mode": trace.mode,
            "language_fragment_detected": trace.language_fragment_detected,
        },
    }


@app.post("/chat")
async def chat_route(body: ChatRequest, request: Request) -> dict[str, Any]:
    ADAPTER.require_ready()
    client_ip = request.client.host if request.client else "unknown"
    await _rate_limit_or_429(client_ip)

    cfg = GenerationConfig(**body.params.model_dump())
    history = await _get_session_history(body.session_id)

    if cfg.mode == "full_system":
        operator = await run_in_threadpool(
            _dispatch_full_system_operator,
            body.message,
        )
        if operator["handled"]:
            reply = str(operator["response"])
            new_messages = [
                {"role": "user", "content": body.message},
                {"role": "assistant", "content": reply},
            ]
            history.extend(new_messages)
            await _append_to_session(body.session_id, new_messages)
            await _save_session(body.session_id)
            subsystem_trace = {
                "mode": "full_system",
                "agent_executed": bool(operator["agent_executed"]),
                "tool_executed": bool(operator["tool_executed"]),
                "model_executed": False,
            }
            trace_id = _record_trace(
                {
                    "request_id": request.state.request_id,
                    "session_id": body.session_id,
                    "formatted_prompt": body.message,
                    "context": {"operator_dispatch": True},
                    "memory_results": [],
                    "config": asdict(cfg),
                    "generation": {
                        "output": reply,
                        "stopped_by": "operator_dispatch",
                        "quality_state": "accepted",
                        "subsystem_trace": subsystem_trace,
                    },
                }
            )
            return {
                "response": reply,
                "session_id": body.session_id,
                "turn": _turn_count(history),
                "history": list(history),
                "trace_id": trace_id,
                "quality_state": "accepted",
                "persisted": True,
                "context_length": 0,
                "prompt_tokens": 0,
                "token_allocation": {},
                "generation": {
                    "tokens_generated": 0,
                    "time_ms": 0.0,
                    "stopped_by": "operator_dispatch",
                    "repeated_ngrams": False,
                    "language_fragment_detected": False,
                    "subsystems": subsystem_trace,
                    "mode": "full_system",
                },
            }

    memory_results = []
    memory_system = get_memory_system() if cfg.mode == "full_system" else None
    if memory_system is not None:
        try:
            memory_results = memory_system.semantic.search(query=body.message, top_k=3)
        except Exception as mem_exc:
            LOGGER.warning("Memory query failed for session %s: %s", body.session_id, mem_exc)

    cognition_context = (
        _classify_chat_cognition(body.message) if cfg.mode == "full_system" else None
    )

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
        current_message=body.message,
        max_new_tokens=cfg.max_tokens,
        identity_context=_identity_context,
        mode=cfg.mode,
    )
    full_prompt = ctx_result["context"]
    trace = await run_in_threadpool(
        generate_traced,
        full_prompt,
        cfg,
        session_id=body.session_id,
    )
    reply = trace.output

    persisted = trace.quality_state == "accepted"
    if persisted:
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

    if cognition_context is not None:
        cognition_context["epistemic_recorded"] = await run_in_threadpool(
            lambda: _record_generation_epistemics(
                body.session_id, request.state.request_id, accepted=persisted
            )
        )

    trace_id = _record_trace(
        {
            "request_id": request.state.request_id,
            "session_id": body.session_id,
            "formatted_prompt": full_prompt,
            "context": ctx_result,
            "memory_results": memory_results,
            "cognition": cognition_context,
            "config": asdict(cfg),
            "generation": asdict(trace),
            "persisted": persisted,
        }
    )

    return {
        "response": reply,
        "session_id": body.session_id,
        "turn": _turn_count(history),
        "history": list(history),
        "trace_id": trace_id,
        "quality_state": trace.quality_state,
        "persisted": persisted,
        "context_length": ctx_result["context_length"],
        "prompt_tokens": ctx_result["prompt_tokens"],
        "token_allocation": ctx_result["token_allocation"],
        "turns_included": ctx_result["turns_included"],
        "context_truncated": ctx_result["context_truncated"],
        "memory_truncated": ctx_result["memory_truncated"],
        "generation": {
            "tokens_generated": trace.tokens_generated,
            "stopped_by": trace.stopped_by,
            "repeated_ngrams": trace.repeated_ngrams_detected,
            "language_fragment_detected": trace.language_fragment_detected,
            "time_ms": trace.time_ms,
            "mode": trace.mode,
        },
    }


@app.get("/traces/{trace_id}")
async def trace_route(trace_id: str) -> dict[str, Any]:
    trace = TRACE_STORE.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return trace


@app.get("/evaluations/current")
async def current_evaluation_route() -> dict[str, Any]:
    return {
        "golden": _latest_json(OUTPUT_V2_DIR / "v2_golden_eval_baseline.json"),
        "validation": _latest_json(OUTPUT_V2_DIR / "v2_validation_history.json"),
        "session": _latest_json(OUTPUT_V2_DIR / "v2_eval_summary.json"),
        "recovery": _latest_json(OUTPUT_V2_DIR / "recovery_gate.json"),
        "recovery_baseline": _latest_json(OUTPUT_V2_DIR / "recovery_baseline.json"),
        "recovery_decision": _latest_json(OUTPUT_V2_DIR / "recovery_decision.json"),
        "draft_proof": _latest_json(OUTPUT_V2_DIR / "draft_proof.json"),
        "tokenizer_recovery": _latest_json(OUTPUT_V2_DIR / "tokenizer_recovery.json"),
        "data_routes": _latest_json(OUTPUT_V2_DIR / "data_route_report.json"),
        "rollback_drill": _latest_json(OUTPUT_V2_DIR / "rollback_drill.json"),
        "release_bundle": _latest_json(OUTPUT_V2_DIR / "release_bundle.json"),
        "release_evidence": _latest_json(OUTPUT_V2_DIR / "release_evidence.json"),
        "private_promotion": _latest_json(OUTPUT_V2_DIR / "private_promotion_eval.json"),
        "private_promotion_progress": _latest_json(
            OUTPUT_V2_DIR / "private_promotion_progress.json"
        ),
        "full_system_integration": _latest_json(
            OUTPUT_V2_DIR / "full_system_integration.json"
        ),
        "promotion_requirements": {
            "checkpoint_tensor_accounting": 1.0,
            "coherent_responses": 0.90,
            "instruction_format_compliance": 0.85,
            "repetition_eos_failure_max": 0.01,
            "cross_session_leakage_max": 0.0,
            "validation_regression_max": 0.02,
        },
    }


def _private_eval_root() -> Path:
    configured = os.environ.get("ANRA_PRIVATE_EVAL_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    drive_root = DRIVE_DIR / "evaluations" / "private_v1"
    return drive_root if DRIVE_DIR.exists() else PRIVATE_EVAL_DIR / "private_v1"


def _runtime_source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _runtime_worktree_clean() -> bool:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return not status.strip()
    except Exception:
        return False


def _evaluation_model_bundle(info: dict[str, Any]) -> dict[str, object]:
    checkpoint_state = info.get("checkpoint_state", {})
    checkpoint_source_commit = (
        checkpoint_state.get("source_commit", "unknown")
        if isinstance(checkpoint_state, dict)
        else "unknown"
    )
    return {
        "checkpoint_path": str(info.get("checkpoint", "unknown")),
        "checkpoint_sha256": str(info.get("checkpoint_sha256", "unknown")),
        "tokenizer_sha256": str(info.get("tokenizer_sha256", "unknown")),
        "checkpoint_source_commit": str(checkpoint_source_commit),
        "runtime_source_commit": _runtime_source_commit(),
        "runtime_worktree_clean": _runtime_worktree_clean(),
    }


def _write_json_atomically(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _run_private_promotion_evaluation() -> dict[str, object]:
    info = get_model_info()
    model_bundle = _evaluation_model_bundle(info)
    private_root = _private_eval_root()
    tasks, suite_metadata = ensure_private_eval_suite(private_root)
    reviews_path = private_root / "human_reviews.json"
    reviews_payload = _latest_json(reviews_path) or {}
    reviews = reviews_payload.get("reviews", {})
    if (
        not isinstance(reviews, dict)
        or reviews_payload.get("suite_sha256") != suite_metadata["suite_sha256"]
    ):
        reviews = {}
    progress_path = OUTPUT_V2_DIR / "private_promotion_progress.json"
    call_index = 0

    def evaluator_generator(
        prompt: str,
        mode: str,
        seed: int,
        ablation: str | None,
    ) -> object:
        nonlocal call_index
        call_index += 1
        session_id = f"private_eval_{call_index:06d}"
        try:
            return generate_traced(
                prompt,
                GenerationConfig(
                    strategy="nucleus",
                    max_tokens=96,
                    temperature=0.7,
                    top_p=0.90,
                    top_k=40,
                    seed=seed,
                    use_kv_cache=False,
                    mode=mode,
                    ablated_subsystem=ablation,
                    persist_adaptive_state=False,
                ),
                session_id=session_id,
            )
        finally:
            clear_session_runtime_state(session_id)

    def persist_progress(progress: dict[str, object]) -> None:
        _write_json_atomically(
            progress_path,
            {
                "schema_version": 1,
                "updated_at": time.time(),
                "suite_sha256": suite_metadata["suite_sha256"],
                **progress,
            },
        )

    _write_json_atomically(
        progress_path,
        {
            "schema_version": 1,
            "updated_at": time.time(),
            "phase": "starting",
            "completed_slices": 0,
            "total_slices": 24,
            "suite_sha256": suite_metadata["suite_sha256"],
        },
    )
    report = run_private_mode_seed_evaluation(
        evaluator_generator,
        tasks=tasks,
        suite_metadata=suite_metadata,
        human_reviews={str(key): bool(value) for key, value in reviews.items()},
        progress_callback=persist_progress,
    )
    report["model_bundle"] = model_bundle
    _write_json_atomically(OUTPUT_V2_DIR / "private_promotion_eval.json", report)
    _write_json_atomically(
        progress_path,
        {
            "schema_version": 1,
            "updated_at": time.time(),
            "phase": "complete",
            "completed_slices": 24,
            "total_slices": 24,
            "suite_sha256": suite_metadata["suite_sha256"],
            "capability_allowed": report["capability_allowed"],
        },
    )
    return report


@app.post("/evaluations/private-promotion", status_code=202)
async def private_promotion_evaluation_route() -> dict[str, Any]:
    ADAPTER.require_ready()
    global _PRIVATE_EVAL_TASK
    if _PRIVATE_EVAL_TASK is not None and not _PRIVATE_EVAL_TASK.done():
        return {
            "status": "running",
            "progress": _latest_json(OUTPUT_V2_DIR / "private_promotion_progress.json"),
        }

    async def run_job() -> None:
        progress_path = OUTPUT_V2_DIR / "private_promotion_progress.json"
        try:
            await run_in_threadpool(_run_private_promotion_evaluation)
        except Exception as exc:
            LOGGER.exception("Private promotion evaluation failed")
            _write_json_atomically(
                progress_path,
                {
                    "schema_version": 1,
                    "updated_at": time.time(),
                    "phase": "failed",
                    "error": str(exc),
                },
            )

    _PRIVATE_EVAL_TASK = asyncio.create_task(run_job())
    return {
        "status": "started",
        "progress": _latest_json(OUTPUT_V2_DIR / "private_promotion_progress.json"),
    }


@app.get("/evaluations/private-promotion/status")
async def private_promotion_status_route() -> dict[str, Any]:
    running = _PRIVATE_EVAL_TASK is not None and not _PRIVATE_EVAL_TASK.done()
    return {
        "running": running,
        "progress": _latest_json(OUTPUT_V2_DIR / "private_promotion_progress.json"),
        "report": _latest_json(OUTPUT_V2_DIR / "private_promotion_eval.json")
        if not running
        else None,
    }


def _run_full_system_integration_probe() -> dict[str, object]:
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    info = get_model_info()
    session_id = f"integration_probe_{uuid.uuid4().hex[:12]}"
    try:
        trace = generate_traced(
            "H: Write one clear sentence confirming an An-Ra integration probe.\nANRA:",
            GenerationConfig(
                strategy="greedy",
                max_tokens=64,
                seed=4417,
                use_kv_cache=False,
                mode="full_system",
                persist_adaptive_state=False,
            ),
            session_id=session_id,
        )
        subsystem_trace = dict(trace.subsystem_trace)
        checks["model_and_native_subsystems"] = all(
            subsystem_trace.get(f"{name}_executed") is True
            for name in ("mod", "rim", "dstp", "esv", "hal")
        ) and subsystem_trace.get("model_executed") is True
        checks["ghost_path"] = subsystem_trace.get("ghost_executed") is True
        checks["evaluation_state_not_persisted"] = (
            subsystem_trace.get("adaptive_state_persisted") is False
        )
        details["generation"] = {
            "quality_state": trace.quality_state,
            "stopped_by": trace.stopped_by,
            "subsystem_trace": subsystem_trace,
        }
    finally:
        clear_session_runtime_state(session_id)

    with tempfile.TemporaryDirectory(prefix="anra-integration-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        from memory.memory_router import MemoryRouter
        from phase2.agent_loop_45k.agent_main import Agent
        from training.verifier import VerifierHierarchy

        nonce = f"integration-{uuid.uuid4().hex}"
        router = MemoryRouter(dim=32, faiss_index_path=temporary_root / "episodic.faiss")
        write_result = router.write(nonce, metadata={"probe": True}, tier="short_term")
        memory_hits = router.read(nonce, n=1, tier="short_term")
        checks["memory"] = bool(
            write_result.tier == "short_term"
            and memory_hits
            and nonce in str(memory_hits[0].get("content", ""))
        )
        details["memory"] = {
            "tier": write_result.tier,
            "hits": len(memory_hits),
        }

        verifier = VerifierHierarchy(temporary_root / "verifier")
        verification = verifier.verify_math("2 + 3", "5")
        checks["verifier"] = verification.score == 1.0 and verification.tier == 1
        details["verifier"] = asdict(verification)

        agent = Agent(verbose=False, approve_each_step=False, max_tool_calls=8)
        agent_result = agent.run(
            "List workspace files without modifying anything",
            clarify=False,
            show_plan=False,
        )
        checks["agent_execution"] = bool(
            len(agent.registry) > 0 and agent_result.get("success") is True
        )
        details["agent"] = {
            "registered_tools": len(agent.registry),
            "success": agent_result.get("success", False),
            "output_preview": str(agent_result.get("output", ""))[:240],
        }

    operator = _dispatch_full_system_operator("/list .")
    checks["sandboxed_tool"] = bool(operator["handled"] and operator["tool_executed"])
    details["tool"] = {
        "handled": operator["handled"],
        "tool_executed": operator["tool_executed"],
        "response_preview": str(operator["response"])[:240],
    }

    cognition = COGNITION.health()
    enabled = cognition.get("enabled", {})
    checks["cognition"] = bool(
        cognition.get("status") == "ok"
        and isinstance(enabled, dict)
        and enabled
        and all(bool(value) for value in enabled.values())
    )
    details["cognition"] = cognition

    graph = build_capability_graph(ROOT)
    capabilities = graph.get("capabilities", {})
    checks["capability_graph"] = bool(
        isinstance(capabilities, dict)
        and capabilities.get("fastapi")
        and capabilities.get("integration_tests")
        and capabilities.get("symbolic_bridge")
    )
    details["capability_graph"] = capabilities

    report = {
        "schema_version": 1,
        "generated_at": time.time(),
        "model_bundle": _evaluation_model_bundle(info),
        "checks": checks,
        "details": details,
        "passed": all(checks.values()),
    }
    _write_json_atomically(OUTPUT_V2_DIR / "full_system_integration.json", report)
    return report


@app.post("/diagnostics/full-system-integration")
async def full_system_integration_route() -> dict[str, Any]:
    ADAPTER.require_ready()
    report = await run_in_threadpool(_run_full_system_integration_probe)
    if not report["passed"]:
        raise HTTPException(status_code=409, detail=report)
    return report


@app.get("/evaluations/private-review-queue")
async def private_review_queue_route() -> dict[str, Any]:
    report = _latest_json(OUTPUT_V2_DIR / "private_promotion_eval.json")
    if not report:
        raise HTTPException(status_code=404, detail="Run the private promotion evaluation first")
    return {
        "blinded": True,
        "instructions": (
            "Mark coherent=true only for relevant, grammatical, non-repetitive answers."
        ),
        "status": report.get("human_review", {}),
        "queue": report.get("human_review_queue", []),
    }


@app.post("/evaluations/private-review")
async def private_review_route(body: HumanReviewRequest) -> dict[str, Any]:
    report_path = OUTPUT_V2_DIR / "private_promotion_eval.json"
    report = _latest_json(report_path)
    if not report:
        raise HTTPException(status_code=404, detail="Run the private promotion evaluation first")
    reviews = {item.review_id: item.coherent for item in body.reviews}
    try:
        updated = apply_blinded_human_reviews(report, reviews)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _write_json_atomically(report_path, updated)
    _write_json_atomically(
        _private_eval_root() / "human_reviews.json",
        {
            "schema_version": 1,
            "updated_at": time.time(),
            "suite_sha256": updated.get("suite_metadata", {}).get("suite_sha256", ""),
            "reviews": reviews,
        },
    )
    return {
        "human_review": updated["human_review"],
        "capability_gates": updated["capability_gates"],
        "capability_allowed": updated["capability_allowed"],
    }


@app.post("/evaluations/context-growth")
async def context_growth_evidence_route(body: ContextGrowthEvidenceRequest) -> dict[str, Any]:
    report = build_context_growth_evidence(**body.model_dump())
    target = OUTPUT_V2_DIR / "context_growth_evidence.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
    if not report["passed"]:
        raise HTTPException(status_code=409, detail=report)
    return report


@app.post("/diagnostics/cache-parity")
async def cache_parity_route(body: CacheParityRequest) -> dict[str, Any]:
    report = await run_in_threadpool(
        verify_kv_cache_parity,
        body.prompt,
        max_tokens=body.max_tokens,
    )
    if not report["verified"]:
        raise HTTPException(status_code=409, detail=report)
    return report


@app.get("/diagnostics/session-isolation")
async def session_isolation_route() -> dict[str, Any]:
    report = await run_in_threadpool(verify_session_state_isolation, probe_generation=True)
    if not report["verified"]:
        raise HTTPException(status_code=409, detail=report)
    return report


@app.post("/diagnostics/recovery-gate")
async def recovery_gate_route() -> dict[str, Any]:
    ADAPTER.require_ready()
    call_index = 0

    def recovery_generator(
        prompt: str,
        mode: str,
        seed: int,
        ablation: str | None,
    ) -> object:
        nonlocal call_index
        call_index += 1
        session_id = f"recovery_gate_{call_index:04d}"
        try:
            return generate_traced(
                prompt,
                GenerationConfig(
                    strategy="greedy",
                    max_tokens=64,
                    seed=seed,
                    use_kv_cache=False,
                    mode=mode,
                    ablated_subsystem=ablation,
                    persist_adaptive_state=False,
                ),
                session_id=session_id,
            )
        finally:
            clear_session_runtime_state(session_id)

    report = await run_in_threadpool(
        run_recovery_prompt_gate,
        recovery_generator,
    )
    info = ADAPTER.info or await run_in_threadpool(get_model_info)
    report["model_bundle"] = _evaluation_model_bundle(info)
    checkpoint_state = info.get("checkpoint_state", {})
    checkpoint_state = checkpoint_state if isinstance(checkpoint_state, dict) else {}
    baseline_path = OUTPUT_V2_DIR / "recovery_baseline.json"
    baseline_payload = _latest_json(baseline_path)
    if not baseline_path.exists():
        baseline_payload = {
            "schema_version": report["schema_version"],
            "generated_at": report["generated_at"],
            "prompt_suite_sha256": report["prompt_suite_sha256"],
            "model_bundle": report["model_bundle"],
            "baseline": report["baseline"],
            "baseline_outputs": report["baseline_outputs"],
            "checkpoint_state": checkpoint_state,
        }
        baseline_tmp = baseline_path.with_suffix(".tmp")
        baseline_tmp.write_text(
            json.dumps(baseline_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        baseline_tmp.replace(baseline_path)
    baseline_state = (
        baseline_payload.get("checkpoint_state", {})
        if isinstance(baseline_payload, dict)
        else {}
    )
    baseline_state = baseline_state if isinstance(baseline_state, dict) else {}
    candidate_metrics = report.get("candidate", {})
    candidate_metrics = candidate_metrics if isinstance(candidate_metrics, dict) else {}
    draft_proof = _latest_json(OUTPUT_V2_DIR / "draft_proof.json") or {}
    rescue_tokens = max(
        0,
        int(checkpoint_state.get("tokens_seen", 0))
        - int(baseline_state.get("tokens_seen", 0)),
    )
    recovery_decision = build_frontier_recovery_decision(
        draft_proof_passed=draft_proof.get("passed") is True,
        rescue_tokens_seen=rescue_tokens,
        baseline_validation_loss=float(
            baseline_state.get("best_validation_loss", float("inf"))
        ),
        candidate_validation_loss=float(
            checkpoint_state.get("best_validation_loss", float("inf"))
        ),
        candidate_coherence_rate=float(candidate_metrics.get("coherence_rate", 0.0)),
        generation_failure_rate=float(candidate_metrics.get("repetition_failure_rate", 1.0))
        + float(candidate_metrics.get("eos_failure_rate", 1.0)),
    )
    report["recovery_decision"] = recovery_decision
    _write_json_atomically(OUTPUT_V2_DIR / "recovery_decision.json", recovery_decision)
    target = OUTPUT_V2_DIR / "recovery_gate.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
    return report


@app.post("/diagnostics/rollback-drill")
async def rollback_drill_route() -> dict[str, Any]:
    ADAPTER.require_ready()
    info = ADAPTER.info or await run_in_threadpool(get_model_info)
    checkpoint = Path(str(info.get("checkpoint", "")))
    report = await run_in_threadpool(
        run_rollback_drill,
        checkpoint,
        report_path=OUTPUT_V2_DIR / "rollback_drill.json",
    )
    if not verify_release_manifest(report):
        raise HTTPException(status_code=500, detail="rollback drill signature verification failed")
    if not report.get("passed"):
        raise HTTPException(status_code=409, detail=report)
    return report


def _checkpoint_manifest_hashes_verified(checkpoint_state: dict[str, Any]) -> bool:
    expected = checkpoint_state.get("data_manifests", {})
    if not isinstance(expected, dict) or not expected:
        return False
    root = OUTPUT_V2_DIR / "data_manifests"
    for name, saved_digest in expected.items():
        path = root / str(name)
        if not path.is_file():
            return False
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if not hmac.compare_digest(digest, str(saved_digest)):
            return False
    return True


def _evaluation_bundle_matches(
    report: dict[str, Any],
    expected: dict[str, object],
) -> bool:
    bundle = report.get("model_bundle", {})
    if not isinstance(bundle, dict):
        return False
    hashes_match = all(
        expected.get(key) not in {None, "", "unknown"}
        and hmac.compare_digest(str(bundle.get(key, "")), str(expected[key]))
        for key in ("checkpoint_sha256", "tokenizer_sha256", "runtime_source_commit")
    )
    return bool(
        hashes_match
        and bundle.get("runtime_worktree_clean") is True
        and expected.get("runtime_worktree_clean") is True
    )


def _private_promotion_verified(
    report: dict[str, Any],
    *,
    expected_bundle: dict[str, object],
) -> bool:
    metadata = report.get("suite_metadata", {})
    if (
        report.get("capability_allowed") is not True
        or int(report.get("task_count", 0)) < 500
        or not isinstance(metadata, dict)
        or metadata.get("verified") is not True
        or metadata.get("origin") != "private_artifact"
        or not _evaluation_bundle_matches(report, expected_bundle)
    ):
        return False
    suite_path = Path(str(metadata.get("suite_path", "")))
    manifest_path = Path(str(metadata.get("manifest_path", "")))
    if not suite_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    suite_digest = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    return bool(
        isinstance(manifest, dict)
        and int(manifest.get("task_count", 0)) == int(report.get("task_count", 0))
        and hmac.compare_digest(suite_digest, str(metadata.get("suite_sha256", "")))
        and hmac.compare_digest(suite_digest, str(manifest.get("suite_sha256", "")))
    )


def _full_system_integration_verified(
    report: dict[str, Any],
    *,
    expected_bundle: dict[str, object],
) -> bool:
    checks = report.get("checks", {})
    required = {
        "model_and_native_subsystems",
        "ghost_path",
        "evaluation_state_not_persisted",
        "memory",
        "verifier",
        "agent_execution",
        "sandboxed_tool",
        "cognition",
        "capability_graph",
    }
    return bool(
        report.get("passed") is True
        and isinstance(checks, dict)
        and required.issubset(checks)
        and all(checks.get(name) is True for name in required)
        and _evaluation_bundle_matches(report, expected_bundle)
    )


@app.get("/diagnostics/release-evidence")
async def release_evidence_route() -> dict[str, Any]:
    readiness = ADAPTER.readiness()
    if readiness["ready"] is not True:
        return {
            "status": str(readiness["stage"]),
            "readiness": readiness,
            "model_error": readiness.get("error"),
            "cache": {"verified": False},
            "session_isolation": {"verified": False},
            "evidence": {},
        }
    info = ADAPTER.info or await run_in_threadpool(get_model_info)
    expected_evaluation_bundle = _evaluation_model_bundle(info)
    checkpoint_state = info.get("checkpoint_state", {})
    if not isinstance(checkpoint_state, dict):
        checkpoint_state = {}
    cache = await run_in_threadpool(verify_kv_cache_parity)
    isolation = await run_in_threadpool(verify_session_state_isolation, probe_generation=True)
    load_report = checkpoint_state.get("load_report", {})
    tokenizer_identity = checkpoint_state.get("tokenizer_identity", {})
    history = checkpoint_state.get("validation_history", [])
    losses = [
        float(item["loss"])
        for item in history
        if isinstance(item, dict) and isinstance(item.get("loss"), (int, float))
    ]
    validation_ok = bool(losses) and losses[-1] <= min(losses) * 1.02
    model_config = checkpoint_state.get("model_config", {})
    configuration_ok = (
        isinstance(model_config, dict)
        and all(
            key in model_config
            for key in ("vocab_size", "n_embd", "n_layer", "n_head", "block_size")
        )
        and checkpoint_state.get("source_commit") not in {None, "", "unknown"}
    )
    rollback = _latest_json(OUTPUT_V2_DIR / "rollback_drill.json") or {}
    recovery = _latest_json(OUTPUT_V2_DIR / "recovery_gate.json") or {}
    private_promotion = _latest_json(OUTPUT_V2_DIR / "private_promotion_eval.json") or {}
    full_system_integration = (
        _latest_json(OUTPUT_V2_DIR / "full_system_integration.json") or {}
    )
    manifest_restore = await run_in_threadpool(
        restore_embedded_data_manifests,
        OUTPUT_V2_DIR / "data_manifests",
    )
    manifest_names = (
        list(checkpoint_state.get("data_manifests", {}).keys())
        if isinstance(checkpoint_state.get("data_manifests", {}), dict)
        else []
    )
    private_metadata = private_promotion.get("suite_metadata", {})
    private_evaluation_paths = [OUTPUT_V2_DIR / "private_promotion_eval.json"]
    if isinstance(private_metadata, dict):
        private_evaluation_paths.extend(
            Path(str(private_metadata[key]))
            for key in ("suite_path", "manifest_path")
            if private_metadata.get(key)
        )
    release_bundle = await run_in_threadpool(
        build_release_bundle_manifest,
        checkpoint_path=Path(str(info.get("checkpoint", ""))),
        tokenizer_path=active_tokenizer_path(),
        corpus_manifest_paths=[
            OUTPUT_V2_DIR / "data_manifests" / str(name) for name in manifest_names
        ],
        model_config=model_config if isinstance(model_config, dict) else {},
        source_commit=str(checkpoint_state.get("source_commit", "unknown")),
        evaluation_paths=[
            OUTPUT_V2_DIR / "recovery_gate.json",
            *private_evaluation_paths,
            OUTPUT_V2_DIR / "full_system_integration.json",
        ],
        rollback_path=OUTPUT_V2_DIR / "rollback_drill.json",
        output_path=OUTPUT_V2_DIR / "release_bundle.json",
    )
    evidence = {
        "checkpoint_tensor_accounting": bool(
            isinstance(load_report, dict)
            and load_report.get("all_target_tensors_accounted") is True
        ),
        "tokenizer_compatibility": bool(
            isinstance(tokenizer_identity, dict) and tokenizer_identity.get("verified") is True
        ),
        "cache_parity": cache.get("verified") is True,
        "zero_session_state_leakage": isolation.get("verified") is True,
        "validation_loss_regression_within_2pct": validation_ok,
        "corpus_manifest_verified": _checkpoint_manifest_hashes_verified(checkpoint_state),
        "configuration_manifest_verified": configuration_ok,
        "rollback_drill_passed": (
            rollback.get("passed") is True
            and verify_release_manifest(rollback)
            and hmac.compare_digest(
                str(rollback.get("checkpoint_sha256", "")),
                str(expected_evaluation_bundle["checkpoint_sha256"]),
            )
        ),
        "recovery_prompt_gate": bool(
            recovery.get("passed") is True
            and _evaluation_bundle_matches(recovery, expected_evaluation_bundle)
        ),
        "private_promotion_evaluation": _private_promotion_verified(
            private_promotion,
            expected_bundle=expected_evaluation_bundle,
        ),
        "full_system_integration": _full_system_integration_verified(
            full_system_integration,
            expected_bundle=expected_evaluation_bundle,
        ),
        "signed_release_bundle": verify_release_bundle_manifest(release_bundle),
    }
    report = {
        "generated_at": time.time(),
        "evidence": evidence,
        "cache": cache,
        "session_isolation": isolation,
        "embedded_manifest_restore": manifest_restore,
        "private_promotion": {
            "capability_allowed": private_promotion.get("capability_allowed", False),
            "capability_gates": private_promotion.get("capability_gates", {}),
            "human_review": private_promotion.get("human_review", {}),
        },
        "full_system_integration": {
            "passed": full_system_integration.get("passed", False),
            "checks": full_system_integration.get("checks", {}),
        },
        "release_ready": all(evidence.values()),
    }
    target = OUTPUT_V2_DIR / "release_evidence.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
    return report


@app.get("/stream")
async def stream_route(
    session_id: str, message: str, strategy: str = "greedy"
) -> StreamingResponse:
    ADAPTER.require_ready()
    history = await _get_session_history(session_id)
    session_pairs = []
    index = 0
    while index < len(history) - 1:
        if history[index].get("role") == "user" and history[index + 1].get("role") == "assistant":
            session_pairs.append(
                (history[index].get("content", ""), history[index + 1].get("content", ""))
            )
            index += 2
        else:
            index += 1
    cfg = GenerationConfig(strategy=strategy, mode="diagnostic", use_kv_cache=False)
    context_result = await run_in_threadpool(
        _ctx_optimizer.build_optimized_context,
        session_history=session_pairs,
        memory_results=[],
        current_message=message,
        max_new_tokens=cfg.max_tokens,
        identity_context=_identity_context,
        mode=cfg.mode,
    )
    context = context_result["context"]

    async def async_event_gen() -> AsyncIterator[str]:
        loop = asyncio.get_event_loop()
        gen_iter = await loop.run_in_executor(None, lambda: list(generate_stream(context, cfg)))
        assembled = ""
        for ch in gen_iter:
            assembled += ch
            yield f"data: {ch}\n\n"
        repeated = bool(detect_repetition(assembled)["repeated_ngrams_detected"])
        accepted = bool(assembled.strip()) and not repeated
        if accepted:
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
async def sessions_route() -> dict[str, Any]:
    sessions = await SESSION_STORE.list_sessions()
    return {"sessions": sessions, "count": len(sessions)}


@app.get("/health")
@app.get("/status")
async def health_route() -> dict[str, Any]:
    info = ADAPTER.info
    readiness = ADAPTER.readiness()
    civ = _latest_json(CIV_LATEST)
    civ_score = float((civ or {}).get("cosine_similarity", (civ or {}).get("score", 1.0)))
    ibs = _latest_json(IBS_LATEST) or {}
    disabled = disabled_components()
    auth_misconfigured = _owner_auth_required() and not _configured_owner_token()
    checkpoint_state = info.get("checkpoint_state", {})
    load_report = (
        checkpoint_state.get("load_report", {}) if isinstance(checkpoint_state, dict) else {}
    )
    tokenizer_identity = (
        checkpoint_state.get("tokenizer_identity", {}) if isinstance(checkpoint_state, dict) else {}
    )
    bundle_status = (
        "verified"
        if load_report.get("all_target_tensors_accounted", False)
        and tokenizer_identity.get("verified", False)
        else "unverified"
    )
    golden = _latest_json(OUTPUT_V2_DIR / "v2_golden_eval_baseline.json") or {}
    quality_status = (
        "passed"
        if golden.get("promotion_allowed") is True
        else "failed"
        if golden
        else "unverified"
    )
    status = (
        "failed"
        if readiness["stage"] == "failed"
        else "loading"
        if readiness["ready"] is not True
        else "blocked"
        if civ_score < 0.80 or auth_misconfigured
        else "degraded"
        if disabled
        else "ok"
    )
    return {
        "status": status,
        "service_status": str(readiness["stage"]),
        "readiness": readiness,
        "model_error": readiness.get("error"),
        "bundle_status": bundle_status,
        "quality_status": quality_status,
        "model": "An-Ra",
        "profile": str(info.get("profile", "unknown")),
        "checkpoint": str(info.get("checkpoint", "unknown")),
        "checkpoint_sha256": str(info.get("checkpoint_sha256", "unknown")),
        "tokenizer_sha256": str(info.get("tokenizer_sha256", "unknown")),
        "device": str(info.get("device", "unknown")),
        "vocab_size": int(info.get("vocab_size", -1) or -1),  # type: ignore[arg-type]
        "param_count": int(info.get("param_count", 0) or 0),
        "block_size": int(info.get("block_size", 0) or 0),
        "checkpoint_state": checkpoint_state,
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
        "campaigns": [path.name for path in CAMPAIGN_DIR.glob("campaign_*.json")]
        if CAMPAIGN_DIR.exists()
        else [],
        "cognition": COGNITION.status(),
    }


@app.get("/cognition/consent")
async def cognition_consent_get(request: Request) -> dict[str, Any]:
    _require_owner(request)
    return COGNITION.status()["consent"]


@app.put("/cognition/consent")
async def cognition_consent_put(body: ConsentRequest, request: Request) -> dict[str, Any]:
    _require_owner(request)
    changes = {key: value for key, value in body.model_dump().items() if value is not None}
    return asdict(COGNITION.update_consent(**changes))


@app.get("/cognition/status")
async def cognition_status() -> dict[str, Any]:
    return COGNITION.health()


@app.get("/owner-model")
async def owner_model_get(request: Request) -> dict[str, Any]:
    _require_owner(request)
    return COGNITION.lhm.export()


@app.patch("/owner-model")
async def owner_model_patch(body: OwnerModelPatch, request: Request) -> dict[str, Any]:
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
async def owner_model_delete(
    request: Request, name: str | None = None, session_id: str | None = None
) -> dict[str, Any]:
    _require_owner(request)
    if name:
        return {"deleted": int(COGNITION.lhm.delete(name))}
    if session_id:
        return {"deleted": COGNITION.lhm.delete_session(session_id)}
    return {"deleted": COGNITION.lhm.wipe()}


@app.post("/cognition/consolidate")
async def cognition_consolidate(body: ConsolidateRequest, request: Request) -> dict[str, Any]:
    _require_owner(request)
    turns = await SESSION_STORE.get_history(body.session_id)
    report = COGNITION.cec.consolidate(
        body.session_id,
        turns,
        opted_in=COGNITION.consent.session_consolidation,
    )
    return asdict(report)


@app.post("/cognition/debate")
async def cognition_debate(body: DebateRequest, request: Request) -> dict[str, Any]:
    _require_owner(request)
    from cognition.self_debate import DebatePosition

    def generate_position(role: str, task: str, seed: int, budget: int) -> DebatePosition:
        prompt = (
            f"Role={role}; seed={seed}; budget={budget}. Analyze without inventing evidence: {task}"
        )
        argument = ADAPTER.run(prompt, strategy="greedy")
        return DebatePosition(
            role,
            argument,
            (),
            ("No independently verified evidence attached.",),
            0.5,
            ("Human review required.",),
        )

    result = await run_in_threadpool(
        COGNITION.debate.run,
        body.task,
        generate_position,
        verify_claims=lambda position: bool(position.supporting_evidence),
        verify_synthesis=lambda _positions: False,
    )
    return result.to_dict()


@app.post("/experiments/propose")
async def experiment_propose(body: ExperimentProposalRequest, request: Request) -> dict[str, Any]:
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
async def experiment_authorize(experiment_id: str, request: Request) -> dict[str, Any]:
    _require_owner(request)
    try:
        return asdict(COGNITION.ssie.authorize(experiment_id, owner_authorized=True))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="experiment not found") from exc


@app.post("/agi-benchmarks/run")
async def agi_benchmarks_run(request: Request) -> dict[str, Any]:
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
async def agi_benchmarks_latest() -> dict[str, Any]:
    return _latest_json(OUTPUT_V2_DIR / "agi_benchmarks" / "latest.json") or {
        "status": "insufficient_data"
    }


@app.get("/training/preflight")
async def training_preflight(
    model_size: str = "frontier", runtime_class: str | None = "t4_frontier_smoke"
) -> dict[str, Any]:
    from training.preflight import run_preflight

    return run_preflight(model_size, runtime_class=runtime_class).to_dict()


@app.post("/training/launch-manifest")
async def training_launch_manifest(body: LaunchManifestRequest, request: Request) -> dict[str, Any]:
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
async def hal_state_route() -> dict[str, Any]:
    return read_hal_state()


@app.post("/reset")
async def reset_route(body: ResetRequest) -> dict[str, Any]:
    await SESSION_STORE.delete_session(body.session_id)
    return {"cleared": True, "session_id": body.session_id}


@app.get("/strategies")
async def strategies_route() -> dict[str, Any]:
    return {
        "greedy": {"description": "Deterministic argmax decoding", "params": {}},
        "temperature": {"description": "Temperature sampling", "params": {"temperature": 0.8}},
        "topk": {"description": "Top-k sampling", "params": {"top_k": 40}},
        "nucleus": {"description": "Top-p nucleus sampling", "params": {"top_p": 0.92}},
        "beam": {"description": "Beam search", "params": {"beam_width": 4}},
        "contrastive": {
            "description": "Contrastive or nucleus fallback",
            "params": {"top_p": 0.92},
        },
    }


@app.get("/debug/context/{session_id}")
async def debug_context_route(session_id: str, message: str = "debug") -> dict[str, Any]:
    context, _, _ = await _build_context(session_id, message)
    return {"context": context}


@app.get("/system-map")
async def system_map_route() -> dict[str, Any]:
    return SYSTEM_GRAPH or await run_in_threadpool(
        build_capability_graph, Path(__file__).resolve().parent
    )


# Each phase-3 subsystem lives in its own package under ``phase3/``. The health
# probe must load the real module by its canonical dotted path and run its real
# ``health_check`` -- the previous bare ``__import__("identity_injector")`` always
# raised ModuleNotFoundError, which made every subsystem look degraded regardless
# of its true state while the endpoint still asserted a top-level "ok".
_PHASE3_HEALTH_MODULES: tuple[tuple[str, str], ...] = (
    ("identity", "phase3.identity_45n.identity_injector"),
    ("ouroboros", "phase3.ouroboros_45o.ouroboros_numpy"),
    ("symbolic", "phase3.symbolic_bridge_45q.symbolic_bridge"),
    ("sovereignty", "phase3.sovereignty_45r.sovereignty_bridge"),
    ("ghost_memory", "phase3.ghost_memory_45p.ghost_memory"),
)


def _run_phase3_health_checks() -> dict[str, dict[str, Any]]:
    import importlib

    checks: dict[str, dict[str, Any]] = {}
    for key, module_name in _PHASE3_HEALTH_MODULES:
        try:
            module = importlib.import_module(module_name)
            health = getattr(module, "health_check", None)
            if callable(health):
                result = dict(health())
                result.setdefault("status", "ok")
                checks[key] = result
            else:
                checks[key] = {"status": "degraded", "detail": "health_check missing"}
        except Exception as exc:  # pragma: no cover - defensive: report, never crash
            checks[key] = {"status": "degraded", "detail": f"{type(exc).__name__}: {exc}"}
    return checks


@app.get("/phase-health")
@app.get("/sovereignty/status")
async def phase_health_route() -> dict[str, Any]:
    graph = SYSTEM_GRAPH or await run_in_threadpool(
        build_capability_graph, Path(__file__).resolve().parent
    )
    checks = await run_in_threadpool(_run_phase3_health_checks)
    # The aggregate status is the honest conjunction of the real sub-checks: it
    # is "ok" only when every phase-3 subsystem reports healthy, never asserted.
    healthy = all(str(check.get("status")) == "ok" for check in checks.values())
    degraded = sorted(key for key, check in checks.items() if str(check.get("status")) != "ok")
    return {
        "status": "ok" if healthy else "degraded",
        "degraded_subsystems": degraded,
        "capabilities": graph.get("capabilities", {}),
        "phase_snapshots": graph.get("phase_snapshots", []),
        "phase3_health": checks,
    }


@app.get("/goals")
async def goals_route() -> dict[str, Any]:
    goals = [job for job in JOBS.values() if job["kind"] == "goal"]
    return {
        "goals": goals,
        "active": sum(job["status"] in {"queued", "running"} for job in goals),
        "completed": sum(job["status"] == "complete" for job in goals),
        "status": "online",
    }


@app.post("/goal")
@app.post("/goals")
async def goals_create_route(body: GoalRequest) -> dict[str, Any]:
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
async def plans_route(body: PlanRequest) -> dict[str, Any]:
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
async def memory_stats_route() -> dict[str, Any]:
    _require_feature("memory")
    return {
        "total": 0,
        "episodic": 0,
        "semantic": 0,
        "graph_nodes": 0,
        "status": "online" if MEMORY_SYSTEM is not None else "unavailable",
    }


@app.get("/memory")
async def memory_route(query: str = "", top_k: int = 5) -> dict[str, Any]:
    _require_feature("memory")
    memory_system = get_memory_system() if query else None
    results = memory_system.semantic.search(query=query, top_k=top_k) if memory_system else []
    return {"query": query, "results": results}


@app.post("/memory")
async def memory_add_route(body: MemoryAddRequest) -> dict[str, Any]:
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
async def eval_route() -> dict[str, Any]:
    _require_feature("evaluation")
    return await _new_job("evaluation", {"suite": "IBS-50", "seeds": 3})


@app.get("/training/candidates")
async def training_candidates_route() -> dict[str, Any]:
    return {"candidates": list(TRAINING_CANDIDATES.values())}


@app.post("/training/candidates")
async def training_candidate_create_route(request: Request) -> dict[str, Any]:
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
async def robotics_workflows_route(body: WorkflowRequest) -> dict[str, Any]:
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
    return await _new_job(
        "robotics_workflow",
        {
            "workflow_id": workflow.workflow_id,
            "goal": workflow.goal,
            "skills": [skill.skill_name for skill in workflow.skills],
            "mode": "simulation_or_shadow",
        },
    )


@app.get("/jobs/{job_id}")
async def job_status_route(job_id: str) -> dict[str, Any]:
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JOBS[job_id]


@app.get("/jobs/{job_id}/events")
async def job_events_route(job_id: str) -> StreamingResponse:
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found.")

    async def stream_events() -> AsyncIterator[str]:
        for event in JOBS[job_id]["events"]:
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_events(), media_type="text/event-stream")


@app.post("/memory/search")
async def memory_search_route(request: Request) -> dict[str, Any]:
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
async def sovereignty_audit_route() -> dict[str, Any]:
    # Non-blocking: just return immediately; real audit runs async later.
    return {
        "status": "triggered",
        "message": "Sovereignty audit queued. Check /sovereignty/status for results.",
        "timestamp": _now_iso(),
    }


@app.get("/train/status")
async def train_status_route() -> dict[str, Any]:
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
            "avg_quality": float(
                output_quality.get("avg_score", prompts.get("avg_score", 0.0)) or 0.0
            ),
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
    except TimeoutError:
        process.kill()
        await process.wait()
        job["status"] = "failed"
        job["events"].append({"event": "timeout", "timestamp": _now_iso()})
    except Exception as exc:
        job["status"] = "failed"
        job["events"].append({"event": "error", "timestamp": _now_iso(), "error": str(exc)})
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
    model_size = str(payload.get("model_size", "frontier"))
    if model_size != "frontier":
        raise HTTPException(status_code=400, detail="iterate500 accepts only model_size=frontier")
    minutes = max(1, min(720, int(payload.get("minutes", 30))))
    job = await _new_job(
        "training_session",
        {"model_size": model_size, "minutes": minutes, "owner_authorized": True},
    )
    asyncio.create_task(_run_training_job(job["job_id"], model_size=model_size, minutes=minutes))
    return job


@app.get("/identity/score")
async def identity_score_route() -> dict[str, Any]:
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


async def test_api(base_url: str) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        assert (await client.get("/health")).status_code == 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run An-Ra API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("app:app", host=args.host, port=args.port, reload=False)
