"""
server.py
FastAPI Backend for An-Ra Web Interface
=========================================
Wraps the Phase 3 MasterSystem into a robust REST and WebSocket backend
accessible by the Vite React frontend.
"""

from contextlib import asynccontextmanager
import sys
import os
import asyncio
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# ── Dynamic Path Resolution ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
PHASE2_45M_DIR = PROJECT_ROOT / "phase2" / "master_system (45M)"

# Use importlib to load MasterSystem from the directory with spaces/parentheses
import importlib.util
system_py_path = PHASE2_45M_DIR / "system.py"
spec = importlib.util.spec_from_file_location("master_system", str(system_py_path))
master_system_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(master_system_module)
MasterSystem = master_system_module.MasterSystem

# ── Global App State ──────────────────────────────────────────────────────────
system: Optional[MasterSystem] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global system
    print("[An-Ra] Initializing MasterSystem backend...")
    try:
        system = MasterSystem()
        # Non-blocking start if possible, or wrap in thread if it blocks significantly
        await asyncio.to_thread(system.start)
        print("[An-Ra] [OK] MasterSystem active")
    except Exception as e:
        print(f"[An-Ra] [CRITICAL] Backend Initialization Failed: {e}")
        import traceback
        traceback.print_exc()
        system = None
    
    yield
    
    if system:
        print("[An-Ra] Shutting down MasterSystem...")
        await asyncio.to_thread(system.stop)

# ── API Setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="An-Ra Web Interface", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to Vite dev server origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Models ────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

class GoalRequest(BaseModel):
    title: str
    description: str
    priority: str = "medium"

class MemorySearchRequest(BaseModel):
    query: str
    limit: int = 10

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    """Returns the full system status, identical to CLI dashboard."""
    if not system:
        raise HTTPException(status_code=503, detail="System not active")
    return system.status()

@app.post("/api/chat")
async def process_chat(req: ChatRequest):
    """Processes a message through the Phase 3 pipeline (45Q>45N>45P>45O)."""
    if not system:
        raise HTTPException(status_code=503, detail="System not active")
    
    # Run chat synchronously but without blocking the async event loop
    response = await asyncio.to_thread(system.chat, req.message)
    return {"message": req.message, "response": response}

@app.post("/api/goal")
async def run_goal(req: GoalRequest):
    """Spins up the agent loop for the specified goal."""
    if not system:
        raise HTTPException(status_code=503, detail="System not active")
    
    # The run_goal blocks; in a real app this should be enqueued
    result = await asyncio.to_thread(system.run_goal, req.title + ": " + req.description)
    return {"success": result.get("success"), "output": result.get("output", "Done.")}

@app.get("/api/goals")
async def active_goals():
    if not system: return []
    active = system.goals.db.list_goals("active")
    return [{"id": g.goal_id, "title": g.title, "progress": g.progress_pct} for g in active]

# ── Training Endpoints ───────────────────────────────────────────────────────

@app.get("/api/train/status")
async def get_train_status():
    """Returns metrics from Continuous Learning and Distributed Trainer."""
    if not system: raise HTTPException(status_code=503, detail="System not active")
    
    stats = system.continuous_learn.run_stats()
    hw = system.trainer.hardware_profile()
    
    # Get last run history for charting
    runs = system.trainer.db.list_runs()
    latest_run = runs[0] if runs else None
    
    return {
        "stats": stats,
        "hardware": hw,
        "latest_run": {
            "status": latest_run.status if latest_run else "idle",
            "loss_history": latest_run.metrics_history if latest_run else [],
            "id": latest_run.run_id if latest_run else None
        } if latest_run else None
    }

@app.post("/api/train/trigger")
async def trigger_training_run():
    """Manually triggers the weekly self-training run on gathered data."""
    if not system: raise HTTPException(status_code=503, detail="System not active")
    
    # Start in background via MasterSystem's existing method
    result = await asyncio.to_thread(system._self_training_run)
    return {"status": "started", "message": result}

# ── Memory Endpoints ─────────────────────────────────────────────────────────

@app.get("/api/memory/stats")
async def get_memory_stats():
    if not system: 
        raise HTTPException(status_code=503, detail="System not active")
    
    stats = {
        "memory_45j": {"total_nodes": 0, "total_episodes": 0},
        "knowledge_base": {"total_entries": 0},
        "ghost_memory": {"active": False, "compression_ratio": 0.0}
    }
    
    if system.memory:
        try:
            m_stats = await asyncio.to_thread(system.memory.stats)
            store_stats = m_stats.get("memory", {})
            by_type = store_stats.get("by_type", {})
            
            stats["memory_45j"]["total_nodes"] = m_stats.get("graph", {}).get("nodes", by_type.get("semantic", 0))
            stats["memory_45j"]["total_episodes"] = by_type.get("episodic", 0)
        except Exception as e:
            print(f"Error fetching memory stats: {e}")

    if hasattr(system, 'knowledge_base') and system.knowledge_base:
        try:
            kb_stats = await asyncio.to_thread(system.knowledge_base.stats)
            stats["knowledge_base"]["total_entries"] = kb_stats.get("total_entries", 0)
        except Exception as e:
            print(f"Error fetching knowledge base stats: {e}")
            
    if system.ghost_memory:
        try:
            gm_status = await asyncio.to_thread(system.ghost_memory.status)
            stats["ghost_memory"]["active"] = True
            stats["ghost_memory"].update(gm_status)
        except Exception:
            pass
            
    return stats

@app.post("/api/memory/search")
async def search_memory(req: MemorySearchRequest):
    """Allows searching semantic and episodic memory via web interface."""
    if not system or not system.memory:
        raise HTTPException(status_code=503, detail="Memory sub-system not active")
    
    results = await asyncio.to_thread(system.memory.retrieve, req.query, limit=req.limit)
    return {"query": req.query, "results": results}

# ── Sovereignty Endpoints ────────────────────────────────────────────────────

@app.get("/api/sovereignty/status")
async def get_sovereignty_status():
    """Returns current audit health and last report summary."""
    if not system or not system.sovereignty:
        return {"enabled": False, "status": "Not initialized"}
    
    status_data = system.sovereignty.status()
    report = system.sovereignty.get_nightly_report()
    
    return {
        "enabled": status_data.get("enabled", False),
        "status": "online" if status_data.get("available") else "initializing",
        "last_audit": "Recent", 
        "report": report
    }

@app.post("/api/sovereignty/audit")
async def trigger_audit():
    """Triggers a manual code and safety audit."""
    if not system or not system.sovereignty:
        raise HTTPException(status_code=503, detail="Sovereignty daemon not active")
    
    # Manual audit trigger
    success = await asyncio.to_thread(system.sovereignty.trigger_pipeline)
    return {"status": "audit_triggered" if success else "failed", "message": "Manual sovereignty audit has been queued." if success else "Daemon busy."}

@app.get("/api/briefing")
async def get_briefing():
    """Returns morning briefing string."""
    briefing = await asyncio.to_thread(system.morning_briefing)
    return {"text": briefing}

# ── serve React UI ───────────────────────────────────────────────────────────
app.mount("/assets", StaticFiles(directory=str(PROJECT_ROOT / "ui" / "assets")), name="assets")

@app.get("/")
@app.get("/{catchall:path}")
async def serve_ui():
    """Serves the Vite-built React frontend."""
    return FileResponse(str(PROJECT_ROOT / "ui" / "index.html"))

if __name__ == "__main__":
    import uvicorn
    # Start the server
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
