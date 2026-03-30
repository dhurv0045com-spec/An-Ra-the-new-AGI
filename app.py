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
import zipfile
import asyncio
from pathlib import Path

# ── Dynamic Hugging Face Deployment Hack ──────────────────────────────────────
# If we are on HF and missing the source directories, extract them from the zip!
if not os.path.exists("phase2") and os.path.exists("anra_code.zip"):
    print("[HF Deploy] Extracting anra_code.zip...")
    with zipfile.ZipFile("anra_code.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    print("[HF Deploy] Extraction complete!")

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# ── Dynamic Path Resolution to load An-Ra Core ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
PHASE2_45M   = PROJECT_ROOT / "phase2" / "master_system (45M)"

# Add paths to sys.path so we can import MasterSystem
if str(PHASE2_45M) not in sys.path:
    sys.path.insert(0, str(PHASE2_45M))

for p3 in ["identity (45N)", "ouroboros (45O)", "ghost_memory (45P)", "symbolic_bridge (45Q)", "sovereignty (45R)"]:
    p = str(PROJECT_ROOT / "phase3" / p3)
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(str(PHASE2_45M))
from system import MasterSystem

# ── Global App State ──────────────────────────────────────────────────────────
system: MasterSystem = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global system
    print("[An-Ra] Initializing MasterSystem backend...")
    system = MasterSystem()
    system.start()
    yield
    print("[An-Ra] Shutting down MasterSystem...")
    system.stop()

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

@app.post("/api/train")
async def trigger_training():
    """Mock endpoint to trigger the high-scale training pipeline."""
    # In a real environment, this would call subprocess to start train_identity_scale.py
    # and return a Job ID.
    return {
        "status": "training_triggered",
        "message": "High-scale LoRA training triggered. Awaiting GPU cluster connection..."
    }

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
