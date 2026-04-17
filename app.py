"""
app.py — An-Ra Sovereign AI System · Production REST API
=========================================================
Architecture: FastAPI (async-ready, auto-docs, typed, streaming-upgradable)
Session store: disk-backed JSON (survives Colab restarts within session dir)
Model: loads ONCE on startup via lifespan context
Adapter: swap generate.py in ONE place — ModelAdapter.run()
"""

import json
import logging
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("an-ra.api")

# ─── Constants ────────────────────────────────────────────────────────────────

MAX_EXCHANGES       = 20          # rolling window cap (user+assistant pairs)
CONTEXT_CHAR_LIMIT  = 4096        # max chars fed into generate() per call
SESSION_DIR         = Path("./sessions")  # disk persistence directory
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# ─── Model Adapter ────────────────────────────────────────────────────────────
# THIS IS THE ONLY PLACE YOU TOUCH WHEN generate.py CHANGES.
# Everything else in this file is adapter-agnostic.

class ModelAdapter:
    """
    Thin wrapper over generate.py.
    Decouples the rest of the API from the generation interface.
    """

    def __init__(self):
        self.model       = None
        self.vocab_size  = None
        self.device      = None
        self.loaded      = False
        self.load_error  = None

    def load(self):
        """Load the model once. Called at startup."""
        try:
            # ── SWAP POINT ──────────────────────────────────────────────────
            # If your generate.py exposes a model object to load explicitly,
            # do it here. Example:
            #   from generate import load_model, VOCAB_SIZE, DEVICE
            #   self.model = load_model()
            #   self.vocab_size = VOCAB_SIZE
            #   self.device = DEVICE
            #
            # For now we import the function and probe what we can.
            from generate import generate as _generate  # noqa: F401

            # Attempt to pull model metadata from generate module
            import generate as _gen_module
            self.vocab_size = getattr(_gen_module, "VOCAB_SIZE",
                              getattr(_gen_module, "vocab_size", "unknown"))
            self.device     = getattr(_gen_module, "DEVICE",
                              getattr(_gen_module, "device", "cpu"))
            self.loaded     = True
            log.info(f"An-Ra loaded | vocab={self.vocab_size} | device={self.device}")
            # ── END SWAP POINT ───────────────────────────────────────────────

        except Exception as exc:
            self.load_error = str(exc)
            log.error(f"Model load failed: {exc}")
            # We allow startup to succeed so /health can report the error.

    def run(
        self,
        prompt:   str,
        strategy: str = "greedy",
        **kwargs: Any,
    ) -> Tuple[str, float]:
        """
        Call generate() and return (response_text, elapsed_seconds).
        Raises HTTPException if model isn't loaded.
        """
        if not self.loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model not loaded: {self.load_error}",
            )

        # ── SWAP POINT ──────────────────────────────────────────────────────
        # Adjust this call to match your actual generate() signature.
        # Current assumption: generate(prompt, strategy, **kwargs) → str
        from generate import generate  # local import — already cached by Python

        t0 = time.perf_counter()
        try:
            text = generate(prompt, strategy, **kwargs)
        except Exception as exc:
            log.exception("generate() raised an exception")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Generation failed: {exc}",
            )
        elapsed = time.perf_counter() - t0
        # ── END SWAP POINT ───────────────────────────────────────────────────

        return text, elapsed


# Singleton model adapter
adapter = ModelAdapter()


# ─── Session Store ────────────────────────────────────────────────────────────

class SessionStore:
    """
    In-memory dict backed by per-session JSON files on disk.
    Format: { session_id: deque([{role, content, ts}, ...]) }
    Each file: sessions/<session_id>.json
    """

    def __init__(self, session_dir: Path):
        self._dir: Path = session_dir
        self._cache: Dict[str, Deque[Dict]] = {}

    # ── private helpers ───────────────────────────────────────────────────────

    def _path(self, sid: str) -> Path:
        return self._dir / f"{sid}.json"

    def _load_from_disk(self, sid: str) -> Deque[Dict]:
        p = self._path(sid)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                dq = deque(data, maxlen=MAX_EXCHANGES * 2)
                log.info(f"Session {sid} restored from disk ({len(dq)} messages)")
                return dq
            except Exception as exc:
                log.warning(f"Could not load session {sid} from disk: {exc}")
        return deque(maxlen=MAX_EXCHANGES * 2)

    def _save_to_disk(self, sid: str) -> None:
        p = self._path(sid)
        try:
            p.write_text(
                json.dumps(list(self._cache[sid]), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning(f"Could not persist session {sid}: {exc}")

    # ── public API ────────────────────────────────────────────────────────────

    def get(self, sid: str) -> Deque[Dict]:
        if sid not in self._cache:
            self._cache[sid] = self._load_from_disk(sid)
        return self._cache[sid]

    def append(self, sid: str, role: str, content: str) -> None:
        history = self.get(sid)
        history.append({
            "role":    role,
            "content": content,
            "ts":      datetime.now(timezone.utc).isoformat(),
        })
        self._save_to_disk(sid)

    def reset(self, sid: str) -> None:
        self._cache[sid] = deque(maxlen=MAX_EXCHANGES * 2)
        p = self._path(sid)
        if p.exists():
            p.unlink()
        log.info(f"Session {sid} reset")

    def history_as_list(self, sid: str) -> List[Dict]:
        return list(self.get(sid))

    def build_context(self, sid: str, new_message: str) -> str:
        """
        Build the rolling context string fed to generate().
        Format:
            User: <msg>
            An-Ra: <msg>
            ...
            User: <new_message>
        Truncated from the LEFT to stay within CONTEXT_CHAR_LIMIT.
        """
        lines = []
        for entry in self.get(sid):
            prefix = "User" if entry["role"] == "user" else "An-Ra"
            lines.append(f"{prefix}: {entry['content']}")
        lines.append(f"User: {new_message}")

        context = "\n".join(lines)

        # Trim from the left if over limit, keep the most recent exchanges
        if len(context) > CONTEXT_CHAR_LIMIT:
            context = context[-CONTEXT_CHAR_LIMIT:]
            # Snap to the first complete "User:" boundary after trim
            idx = context.find("\nUser:")
            if idx != -1:
                context = context[idx + 1:]

        return context


store = SessionStore(SESSION_DIR)

# ─── Startup / Shutdown ───────────────────────────────────────────────────────

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("An-Ra API starting — loading model...")
    adapter.load()
    log.info("An-Ra API ready.")
    yield
    log.info("An-Ra API shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="An-Ra Sovereign AI · REST API",
    version="2.0.0",
    description="Production API for the An-Ra transformer language model.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten to specific origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request Logging Middleware ───────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    log.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} | {elapsed_ms:.1f}ms"
    )
    return response


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt:   str            = Field(...,         min_length=1, description="Raw prompt text")
    strategy: str            = Field("greedy",                  description="Generation strategy")
    params:   Dict[str, Any] = Field(default_factory=dict,      description="Extra kwargs passed to generate()")

class GenerateResponse(BaseModel):
    response:      str
    strategy:      str
    prompt_chars:  int
    response_chars: int
    elapsed_sec:   float
    rating:        Optional[float] = None   # client can POST back a rating

class ChatRequest(BaseModel):
    session_id:   Optional[str]  = Field(None,  description="Omit to auto-create a new session")
    user_message: str            = Field(...,   min_length=1)
    strategy:     str            = Field("greedy")
    params:       Dict[str, Any] = Field(default_factory=dict)

class ChatResponse(BaseModel):
    session_id:   str
    response:     str
    history:      List[Dict]
    strategy:     str
    elapsed_sec:  float
    context_chars: int
    rating:       Optional[float] = None

class ResetRequest(BaseModel):
    session_id: str

class HealthResponse(BaseModel):
    status:      str
    model_loaded: bool
    vocab_size:  Any
    device:      Any
    uptime_sec:  float
    version:     str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    """
    Returns model status, vocab size, device, uptime.
    Always returns 200 — check model_loaded for readiness.
    """
    return HealthResponse(
        status      = "ok" if adapter.loaded else "degraded",
        model_loaded= adapter.loaded,
        vocab_size  = adapter.vocab_size,
        device      = adapter.device,
        uptime_sec  = round(time.time() - START_TIME, 2),
        version     = "2.0.0",
    )


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_endpoint(req: GenerateRequest):
    """
    Single-shot, stateless generation.
    No session, no history — just prompt → response.
    """
    text, elapsed = adapter.run(req.prompt, req.strategy, **req.params)

    return GenerateResponse(
        response       = text,
        strategy       = req.strategy,
        prompt_chars   = len(req.prompt),
        response_chars = len(text),
        elapsed_sec    = round(elapsed, 4),
    )


@app.post("/chat", response_model=ChatResponse, tags=["Conversation"])
async def chat_endpoint(req: ChatRequest):
    """
    Stateful conversation endpoint.

    Session lifecycle:
    - Omit session_id → new UUID session created, persisted to disk
    - Provide session_id → history loaded from disk (or memory cache)
    - History capped at last MAX_EXCHANGES (20) user+assistant pairs
    - Context built as rolling char window, trimmed from left at 4096 chars

    Why left-trim instead of truncate from right?
    Because the most recent exchange must always survive intact.
    Older context fades out naturally — same as a human conversation.
    """
    sid = req.session_id or str(uuid.uuid4())

    # Build context string from history + new message
    context = store.build_context(sid, req.user_message)
    context_chars = len(context)

    # Run generation
    response_text, elapsed = adapter.run(context, req.strategy, **req.params)

    # Persist both sides of the exchange
    store.append(sid, "user",      req.user_message)
    store.append(sid, "assistant", response_text)

    return ChatResponse(
        session_id    = sid,
        response      = response_text,
        history       = store.history_as_list(sid),
        strategy      = req.strategy,
        elapsed_sec   = round(elapsed, 4),
        context_chars = context_chars,
    )


@app.post("/reset", tags=["Session"])
async def reset_endpoint(req: ResetRequest):
    """
    Clears a session's conversation history from memory and disk.
    Returns 404 if the session never existed.
    """
    p = SESSION_DIR / f"{req.session_id}.json"
    exists_on_disk = p.exists()
    exists_in_mem  = req.session_id in store._cache

    if not exists_on_disk and not exists_in_mem:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{req.session_id}' not found.",
        )

    store.reset(req.session_id)
    return JSONResponse({"status": "ok", "session_id": req.session_id, "cleared": True})


@app.post("/rate", tags=["Feedback"])
async def rate_response(
    session_id: str,
    message_index: int,
    rating: float = Field(..., ge=1.0, le=5.0),
):
    """
    Attach a rating (1–5) to a specific message in a session's history.
    Persists back to disk. Designed for future RLHF data collection.
    """
    history = store.history_as_list(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found or empty.")
    if message_index < 0 or message_index >= len(history):
        raise HTTPException(status_code=400, detail=f"message_index out of range (0–{len(history)-1}).")

    # Write rating into the message entry
    store.get(session_id)[message_index]["rating"] = rating
    store._save_to_disk(session_id)

    return JSONResponse({
        "status":        "rated",
        "session_id":    session_id,
        "message_index": message_index,
        "rating":        rating,
    })


# ─── Global Exception Handler ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception(f"Unhandled exception on {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error.", "type": type(exc).__name__},
    )


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,       # set True during local dev only
        log_level="info",
    )
