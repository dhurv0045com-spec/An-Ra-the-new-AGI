# ruff: noqa: E501
"""Focused local app for the protected V4 SFT checkpoint.

This is intentionally separate from the broad developer service.  It owns one
local, verified SFT checkpoint, loads only its model weights into CUDA, keeps
conversation state in memory, and exposes a compact manual evaluation surface.
The companion desktop controller owns this server process and terminates it
when its window is closed.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import torch
from anra.sft_conversation import SFT_PROMPT_SCHEMA, render_chat_prompt
from evaluation.sft_behavior_gate import check_smoke_response
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from generate import GenerationConfig, generate_traced, get_model_info, unload_runtime
from pydantic import BaseModel, Field

from runtime.local_checkpoint import LocalSFTCheckpoint, resolve_local_sft_checkpoint

_HEARTBEAT_TIMEOUT_SECONDS = max(
    15, int(os.environ.get("ANRA_PROTOTYPE_IDLE_SHUTDOWN_SECONDS", "45"))
)
_MAX_HISTORY_TURNS = 8
_SMOKE_PROMPTS: tuple[tuple[str, str], ...] = (
    ("instruction_following", "Give two concise steps for organizing a small project."),
    ("dialogue", "Respond warmly to a person who says they had a difficult day."),
    ("code", "Write a Python function that returns the larger of two numbers."),
    ("mathematics", "What is 17 plus 28? Show the arithmetic briefly."),
    ("decomposition", "Break preparing a healthy breakfast into three steps."),
    ("tool_contracts", "Show a minimal JSON object describing a successful tool result."),
    ("uncertainty", "How should you answer when you do not have enough evidence?"),
    ("correction", "Rewrite this sentence clearly: The results was not consistent."),
)


class GenerationControls(BaseModel):
    strategy: Literal["greedy", "nucleus", "topk"] = "nucleus"
    max_tokens: int = Field(64, ge=1, le=160)
    temperature: float = Field(0.7, ge=0.05, le=2.0)
    top_k: int = Field(40, ge=1, le=128)
    top_p: float = Field(0.92, ge=0.05, le=1.0)
    repetition_penalty: float = Field(1.15, ge=1.0, le=2.0)
    seed: int | None = Field(0, ge=0, le=2_147_483_647)
    mode: Literal["diagnostic", "native"] = "diagnostic"

    def generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            strategy=self.strategy,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            seed=self.seed,
            mode=self.mode,
            # This prototype is an evaluation and conversation surface.  It
            # does not mutate runtime-adaptive state or start agents/tools.
            persist_adaptive_state=False,
        )


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=12_000)
    session_id: str = Field("local", min_length=1, max_length=64)
    controls: GenerationControls = Field(default_factory=GenerationControls)


class EvaluationRequest(BaseModel):
    controls: GenerationControls = Field(
        default_factory=lambda: GenerationControls(max_tokens=40, strategy="greedy")
    )
    categories: list[str] | None = Field(default=None, max_length=len(_SMOKE_PROMPTS))


class PrototypeRuntime:
    """Thread-safe owner for the one resident prototype model."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stage = "starting"
        self._error = ""
        self._started_at = time.time()
        self._checkpoint: LocalSFTCheckpoint | None = None
        self._info: dict[str, Any] = {}
        self._last_heartbeat: float | None = None
        self._seen_ui = False
        self._shutdown_requested = False
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._last_evaluation: dict[str, Any] | None = None

    def load(self) -> None:
        with self._lock:
            self._stage = "resolving_checkpoint"
            self._error = ""
        try:
            checkpoint = resolve_local_sft_checkpoint()
            os.environ["ANRA_CHECKPOINT_PATH"] = str(checkpoint.path)
            with self._lock:
                self._checkpoint = checkpoint
                self._stage = "loading_model_on_gpu"
            # ``generate`` opens the full-resume archive on CPU and copies only
            # model weights to CUDA. Optimizer state never becomes GPU-resident.
            info = get_model_info()
            loaded = str(info.get("checkpoint", ""))
            if loaded != str(checkpoint.path):
                raise RuntimeError(
                    "runtime selected a different checkpoint than the protected SFT source"
                )
            checkpoint_state = dict(info.get("checkpoint_state", {}))
            sft_metadata = checkpoint_state.get("sft", {})
            if (
                checkpoint_state.get("data_profile") != "sft-v4"
                or checkpoint_state.get("training_data_layout") != "assistant_only_sft_v1"
                or not isinstance(sft_metadata, dict)
                or sft_metadata.get("stage") != "sft"
            ):
                raise RuntimeError(
                    "the selected file is not a signed V4 assistant-only SFT checkpoint"
                )
            if str(info.get("device", "")).lower().startswith("cuda") is False:
                raise RuntimeError("CUDA is unavailable; the local SFT prototype requires your GPU")
            with self._lock:
                self._info = dict(info)
                self._stage = "ready"
        except Exception as error:
            unload_runtime()
            with self._lock:
                self._stage = "failed"
                self._error = f"{type(error).__name__}: {error}"

    def unload(self, *, reason: str) -> None:
        unload_runtime()
        with self._lock:
            self._stage = "unloaded"
            self._info = {}
            self._sessions.clear()
            self._error = ""
            if reason in {"idle_page_closed", "operator_stop"}:
                self._shutdown_requested = True

    def heartbeat(self) -> None:
        with self._lock:
            self._seen_ui = True
            self._last_heartbeat = time.monotonic()

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def conversation_prompt(self, session_id: str, message: str) -> str:
        with self._lock:
            history = list(self._sessions.get(session_id, ())[-(2 * _MAX_HISTORY_TURNS) :])
        return render_chat_prompt(history, message)

    def add_turn(self, session_id: str, message: str, response: str) -> int:
        with self._lock:
            history = self._sessions.setdefault(session_id, [])
            history.extend(
                [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response},
                ]
            )
            del history[: max(0, len(history) - 2 * _MAX_HISTORY_TURNS)]
            return len(history) // 2

    def status(self) -> dict[str, Any]:
        with self._lock:
            last_heartbeat_age = (
                None
                if self._last_heartbeat is None
                else round(max(0.0, time.monotonic() - self._last_heartbeat), 1)
            )
            info = dict(self._info)
            checkpoint = self._checkpoint
            return {
                "stage": self._stage,
                "ready": self._stage == "ready",
                "error": self._error or None,
                "started_at": self._started_at,
                "checkpoint": str(checkpoint.path) if checkpoint else None,
                "checkpoint_source": checkpoint.source if checkpoint else None,
                "checkpoint_sha256": info.get("checkpoint_sha256"),
                "model": {
                    "profile": info.get("profile"),
                    "parameters": info.get("param_count"),
                    "parameter_breakdown": info.get("parameter_breakdown"),
                    "vocabulary": info.get("vocab_size"),
                    "context": info.get("block_size"),
                    "training_step": dict(info.get("checkpoint_state", {})).get("global_step"),
                },
                "gpu": _gpu_snapshot(),
                "sessions": len(self._sessions),
                "last_heartbeat_age_seconds": last_heartbeat_age,
                "shutdown_requested": self._shutdown_requested,
                "idle_timeout_seconds": _HEARTBEAT_TIMEOUT_SECONDS,
            }


def _gpu_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    index = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(index)
    return {
        "available": True,
        "name": torch.cuda.get_device_name(index),
        "allocated_bytes": torch.cuda.memory_allocated(index),
        "reserved_bytes": torch.cuda.memory_reserved(index),
        "free_bytes": free,
        "total_bytes": total,
    }


def _trace_summary(trace: object) -> dict[str, Any]:
    return {
        "tokens_generated": int(getattr(trace, "tokens_generated", 0)),
        "prompt_tokens": int(getattr(trace, "prompt_tokens", 0)),
        "latency_ms": round(float(getattr(trace, "time_ms", 0.0)), 1),
        "stopped_by": str(getattr(trace, "stopped_by", "unknown")),
        "quality_state": str(getattr(trace, "quality_state", "unknown")),
        "repetition_detected": bool(getattr(trace, "repeated_ngrams_detected", False)),
        "fragment_detected": bool(getattr(trace, "language_fragment_detected", False)),
        "mean_entropy": round(
            sum(getattr(trace, "entropy_curve", []))
            / max(1, len(getattr(trace, "entropy_curve", []))),
            4,
        ),
    }


async def _idle_watch(runtime: PrototypeRuntime) -> None:
    while True:
        await asyncio.sleep(3)
        status = runtime.status()
        if (
            status["ready"]
            and status["last_heartbeat_age_seconds"] is not None
            and float(status["last_heartbeat_age_seconds"])
            > _HEARTBEAT_TIMEOUT_SECONDS
        ):
            await run_in_threadpool(runtime.unload, reason="idle_page_closed")
            return


def create_app() -> FastAPI:
    runtime = PrototypeRuntime()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = runtime
        app.state.loader = asyncio.create_task(asyncio.to_thread(runtime.load))
        app.state.idle_watch = asyncio.create_task(_idle_watch(runtime))
        try:
            yield
        finally:
            app.state.loader.cancel()
            app.state.idle_watch.cancel()
            await run_in_threadpool(runtime.unload, reason="server_closed")

    app = FastAPI(title="An-Ra V4 SFT Prototype", version="1.0", lifespan=lifespan)

    def ready_runtime() -> PrototypeRuntime:
        current: PrototypeRuntime = app.state.runtime
        if current.status()["ready"] is not True:
            raise HTTPException(status_code=503, detail=current.status())
        return current

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(PROTOTYPE_HTML)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return runtime.status()

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return runtime.status()

    @app.post("/api/heartbeat")
    async def heartbeat() -> dict[str, Any]:
        runtime.heartbeat()
        return {"accepted": True, "idle_timeout_seconds": _HEARTBEAT_TIMEOUT_SECONDS}

    @app.post("/api/session/{session_id}/clear")
    async def clear_session(session_id: str) -> dict[str, Any]:
        runtime.clear_session(session_id)
        return {"cleared": True, "session_id": session_id}

    @app.post("/api/runtime/unload")
    async def unload() -> dict[str, Any]:
        await run_in_threadpool(runtime.unload, reason="operator_stop")
        return runtime.status()

    @app.post("/api/chat")
    async def chat(body: ChatRequest) -> dict[str, Any]:
        current = ready_runtime()
        prompt = current.conversation_prompt(body.session_id, body.message)
        trace = await run_in_threadpool(
            generate_traced,
            prompt,
            body.controls.generation_config(),
            session_id=f"prototype-{body.session_id}",
        )
        response = str(trace.output)
        turn = current.add_turn(body.session_id, body.message, response)
        return {
            "response": response,
            "turn": turn,
            "trace": _trace_summary(trace),
            "prompt_format": SFT_PROMPT_SCHEMA,
        }

    @app.post("/api/evaluations/run")
    async def run_evaluation(body: EvaluationRequest) -> dict[str, Any]:
        current = ready_runtime()
        requested = set(body.categories or [category for category, _ in _SMOKE_PROMPTS])
        unknown = requested - {category for category, _ in _SMOKE_PROMPTS}
        if unknown:
            raise HTTPException(status_code=422, detail={"unknown_categories": sorted(unknown)})
        rows: list[dict[str, Any]] = []
        for category, prompt in _SMOKE_PROMPTS:
            if category not in requested:
                continue
            trace = await run_in_threadpool(
                generate_traced,
                render_chat_prompt([], prompt),
                body.controls.generation_config(),
                session_id=f"prototype-evaluation-{category}",
            )
            rows.append(
                {
                    "category": category,
                    "prompt": prompt,
                    "response": str(trace.output),
                    "trace": _trace_summary(trace),
                }
            )
            passed, requirement = check_smoke_response(category, str(trace.output))
            rows[-1]["behavior_pass"] = passed
            rows[-1]["requirement"] = requirement
        report = {
            "run_id": str(uuid.uuid4()),
            "checkpoint_sha256": current.status().get("checkpoint_sha256"),
            "controls": body.controls.model_dump(),
            "rows": rows,
            "operational_pass": bool(rows)
            and all(bool(row["behavior_pass"]) for row in rows),
            "note": (
                "This is an operational smoke result, not a claim of correctness "
                "or general intelligence."
            ),
        }
        runtime._last_evaluation = report
        return report

    @app.get("/api/evaluations/latest")
    async def latest_evaluation() -> dict[str, Any]:
        return runtime._last_evaluation or {"available": False}

    return app


app = create_app()


PROTOTYPE_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>An-Ra V4 SFT Prototype</title>
  <style>
    :root { color-scheme: dark; --bg:#0a0d12; --panel:#111720; --soft:#171f2b; --line:#2b394d; --text:#e8eef7; --muted:#9aa9bd; --accent:#65dcff; --good:#77d69a; --bad:#ff8290; }
    * { box-sizing:border-box; } body { margin:0; color:var(--text); background:var(--bg); font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif; }
    header { display:flex; align-items:center; gap:14px; padding:14px 20px; border-bottom:1px solid var(--line); background:#0e131c; position:sticky; top:0; z-index:2; }
    h1 { font-size:16px; letter-spacing:.06em; margin:0; } h2 { font-size:14px; margin:0 0 12px; } h3 { font-size:12px; margin:0 0 7px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; }
    .status { margin-left:auto; color:var(--accent); font:12px ui-monospace,Consolas,monospace; } button,select,input,textarea { font:inherit; } button { color:var(--text); background:var(--soft); border:1px solid var(--line); border-radius:7px; padding:8px 11px; cursor:pointer; } button:hover:not(:disabled) { border-color:var(--accent); } button.primary { background:var(--accent); color:#061019; font-weight:700; } button.danger { color:var(--bad); } button:disabled { opacity:.45; cursor:not-allowed; }
    main { max-width:1400px; margin:0 auto; padding:18px; } .tabs { display:flex; gap:8px; margin-bottom:14px; } .tabs button.active { border-color:var(--accent); color:var(--accent); } .view { display:none; } .view.active { display:block; }
    .chat-layout { display:grid; grid-template-columns:minmax(0,1fr) 330px; gap:16px; min-height:640px; } .panel { border:1px solid var(--line); background:var(--panel); border-radius:9px; padding:16px; } .messages { min-height:470px; max-height:62vh; overflow:auto; display:flex; flex-direction:column; gap:12px; padding:2px; } .message { max-width:88%; padding:10px 12px; white-space:pre-wrap; border-radius:8px; background:var(--soft); border:1px solid var(--line); } .message.user { align-self:flex-end; border-color:#2f7186; background:#102833; } .message.assistant { align-self:flex-start; }
    form { display:grid; grid-template-columns:1fr auto; gap:10px; margin-top:14px; } textarea { resize:vertical; min-height:68px; color:var(--text); background:#0c1119; border:1px solid var(--line); border-radius:7px; padding:10px; } .controls { display:grid; grid-template-columns:1fr 1fr; gap:10px; } label { display:grid; gap:5px; color:var(--muted); font-size:12px; } input,select { width:100%; color:var(--text); background:#0c1119; border:1px solid var(--line); border-radius:6px; padding:7px; } .controls-wide { display:flex; gap:8px; margin-top:14px; flex-wrap:wrap; } pre { margin:0; overflow:auto; white-space:pre-wrap; word-break:break-word; color:#c9d7e9; font:12px/1.4 ui-monospace,Consolas,monospace; } .metric { padding:9px 0; border-bottom:1px solid var(--line); } .metric:last-child { border-bottom:0; } .metric b { display:block; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; } .metric span { overflow-wrap:anywhere; } .notice { color:var(--muted); margin:0 0 12px; } table { width:100%; border-collapse:collapse; } th,td { padding:10px; vertical-align:top; text-align:left; border-bottom:1px solid var(--line); } th { color:var(--muted); font-size:11px; text-transform:uppercase; } .good { color:var(--good); } .bad { color:var(--bad); } .developer-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; } @media (max-width:850px){ header { flex-wrap:wrap; } .status { margin-left:0; width:100%; } .chat-layout,.developer-grid { grid-template-columns:1fr; } .messages { min-height:340px; } }
  </style>
</head>
<body>
  <header><h1>AN-RA V4 SFT PROTOTYPE</h1><span class="status" id="status">Starting local model…</span><button class="danger" id="stop">Stop app &amp; free GPU</button></header>
  <main>
    <div class="tabs"><button class="active" data-view="chat">Chat</button><button data-view="evaluation">Evaluation</button><button data-view="developer">Developer</button></div>
    <section id="chat" class="view active"><div class="chat-layout"><div class="panel"><h2>Conversation</h2><div class="messages" id="messages"><div class="message assistant">The SFT checkpoint is loading to your GPU. This prototype keeps its conversation only in this local session.</div></div><form id="chat-form"><textarea id="prompt" placeholder="Talk to An-Ra…" disabled></textarea><button class="primary" id="send" disabled>Send</button></form></div><aside class="panel"><h2>Generation controls</h2><div class="controls"><label>Mode<select id="mode"><option value="diagnostic">Diagnostic</option><option value="native">Native</option></select></label><label>Strategy<select id="strategy"><option value="nucleus">Nucleus</option><option value="greedy">Greedy</option><option value="topk">Top-k</option></select></label><label>Max tokens<input id="max-tokens" type="number" min="1" max="160" value="64"></label><label>Temperature<input id="temperature" type="number" min="0.05" max="2" step="0.05" value="0.7"></label><label>Top-p<input id="top-p" type="number" min="0.05" max="1" step="0.01" value="0.92"></label><label>Seed<input id="seed" type="number" min="0" value="0"></label></div><div class="controls-wide"><button id="clear" type="button">Clear chat</button></div><h3 style="margin-top:18px">Last trace</h3><pre id="trace">No generation yet.</pre></aside></div></section>
    <section id="evaluation" class="view"><div class="panel"><h2>SFT operational evaluation</h2><p class="notice">Runs the eight fixed SFT categories on the loaded checkpoint. Results show generation behavior, not proof of factual correctness.</p><div class="controls-wide"><button class="primary" id="run-eval">Run eight-prompt smoke</button><span class="status" id="eval-status"></span></div><div id="eval-results" style="margin-top:16px"></div></div></section>
    <section id="developer" class="view"><div class="developer-grid"><div class="panel"><h2>Runtime evidence</h2><div id="runtime"></div></div><div class="panel"><h2>Developer controls</h2><p class="notice">The launcher owns this server. Closing its desktop window stops the process; closing this browser page stops it after the heartbeat timeout.</p><div class="controls-wide"><button id="release" class="danger">Unload model from GPU</button><button id="refresh">Refresh status</button></div><h3 style="margin-top:18px">Raw status</h3><pre id="raw-status">{}</pre></div></div></section>
  </main>
  <script>
    const $ = id => document.getElementById(id); const sessionId = 'prototype-' + crypto.randomUUID(); let ready = false;
    const controls = () => ({ strategy:$('strategy').value, max_tokens:Number($('max-tokens').value), temperature:Number($('temperature').value), top_p:Number($('top-p').value), top_k:40, repetition_penalty:1.15, seed:$('seed').value === '' ? null : Number($('seed').value), mode:$('mode').value });
    async function api(path, options={}) { const r = await fetch(path, options); const body = await r.json(); if(!r.ok) throw new Error(body.detail ? JSON.stringify(body.detail) : JSON.stringify(body)); return body; }
    function append(role,text){ const e=document.createElement('div'); e.className='message '+role; e.textContent=text; $('messages').append(e); $('messages').scrollTop=$('messages').scrollHeight; }
    function bytes(v){ return typeof v==='number' ? (v/1024/1024/1024).toFixed(2)+' GB' : '–'; }
    async function refresh(){ try { const s=await api('/api/status'); ready=!!s.ready; $('status').textContent=ready ? `Ready · ${s.gpu.name} · ${bytes(s.gpu.allocated_bytes)} GPU` : `${s.stage}${s.error ? ' · '+s.error : ''}`; $('send').disabled=!ready; $('prompt').disabled=!ready; const m=s.model||{}; $('runtime').innerHTML=[['Checkpoint',s.checkpoint],['Checkpoint SHA-256',s.checkpoint_sha256],['Model parameters',m.parameters?.toLocaleString()],['Tokenizer vocabulary',m.vocabulary],['Context length',m.context],['Training step',m.training_step],['GPU',s.gpu.name],['GPU allocated',bytes(s.gpu.allocated_bytes)],['GPU free',bytes(s.gpu.free_bytes)],['Heartbeat timeout',s.idle_timeout_seconds+' seconds']].map(([k,v])=>`<div class="metric"><b>${k}</b><span>${v ?? '–'}</span></div>`).join(''); $('raw-status').textContent=JSON.stringify(s,null,2); if(s.shutdown_requested){ $('status').textContent='Stopping and freeing GPU…'; } } catch(e){ $('status').textContent='Status unavailable: '+e; } }
    async function beat(){ try { await api('/api/heartbeat',{method:'POST'}); } catch(_){} }
    $('chat-form').addEventListener('submit',async e=>{e.preventDefault(); const message=$('prompt').value.trim(); if(!message||!ready)return; $('prompt').value=''; append('user',message); $('send').disabled=true; try { const r=await api('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message,session_id:sessionId,controls:controls()})}); append('assistant',r.response||'(empty response)'); $('trace').textContent=JSON.stringify(r.trace,null,2); } catch(err){ append('assistant','Generation error: '+err); } finally { await refresh(); }});
    $('clear').onclick=async()=>{await api('/api/session/'+sessionId+'/clear',{method:'POST'}); $('messages').innerHTML=''; $('trace').textContent='Conversation cleared.';};
    $('run-eval').onclick=async()=>{const b=$('run-eval'); b.disabled=true; $('eval-status').textContent='Running on the local GPU…'; try { const r=await api('/api/evaluations/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({controls:{...controls(),max_tokens:Math.min(64,controls().max_tokens)}})}); $('eval-status').textContent=r.operational_pass?'Behavior gate passed':'Behavior gate failed — keep this checkpoint in research'; $('eval-results').innerHTML=`<table><thead><tr><th>Category</th><th>Gate</th><th>Response</th><th>Trace</th></tr></thead><tbody>${r.rows.map(x=>`<tr><td>${x.category}</td><td class="${x.behavior_pass?'good':'bad'}">${x.behavior_pass?'PASS':'FAIL'}<br><small>${escapeHtml(x.requirement)}</small></td><td>${escapeHtml(x.response)}</td><td><pre>${escapeHtml(JSON.stringify(x.trace,null,2))}</pre></td></tr>`).join('')}</tbody></table>`; }catch(err){$('eval-status').textContent='Evaluation error: '+err;}finally{b.disabled=false;await refresh();}};
    function escapeHtml(value){const e=document.createElement('div');e.textContent=value;return e.innerHTML;}
    $('release').onclick=async()=>{await api('/api/runtime/unload',{method:'POST'});await refresh();}; $('stop').onclick=async()=>{try{await api('/api/runtime/unload',{method:'POST'});}finally{$('status').textContent='Stopping app and freeing GPU…';}}; $('refresh').onclick=refresh;
    document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-view]').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('.view').forEach(x=>x.classList.toggle('active',x.id===b.dataset.view));});
    beat(); refresh(); setInterval(beat,5000); setInterval(refresh,3000);
  </script>
</body>
</html>
"""
