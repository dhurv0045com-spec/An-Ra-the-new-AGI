from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT

CORE_DIR = ROOT / "core"
TRAINING_DIR = ROOT / "training"
INFERENCE_DIR = ROOT / "inference"
TOKENIZER_DIR = ROOT / "tokenizer"
CONFIG_DIR = ROOT / "config"
SCRIPTS_DIR = ROOT / "scripts"
TESTS_DIR = ROOT / "tests"
TRAINING_DATA_DIR = ROOT / "training_data"
OUTPUT_DIR = ROOT / "output"
STATE_DIR = ROOT / "state"
WORKSPACE_DIR = ROOT / "workspace"

PHASE2_DIR = ROOT / "phase2"
FINE_TUNING_DIR = PHASE2_DIR / "fine_tuning (45I)"
MEMORY_DIR = PHASE2_DIR / "memory (45J)"
AGENT_LOOP_DIR = PHASE2_DIR / "agent_loop (45k)"
SELF_IMPROVEMENT_DIR = PHASE2_DIR / "self_improvement (45l)"
MASTER_SYSTEM_DIR = PHASE2_DIR / "master_system (45M)"

PHASE3_DIR = ROOT / "phase3"
IDENTITY_DIR = PHASE3_DIR / "identity (45N)"
OUROBOROS_DIR = PHASE3_DIR / "ouroboros (45O)"
GHOST_MEMORY_DIR = PHASE3_DIR / "ghost_memory (45P)"
SYMBOLIC_BRIDGE_DIR = PHASE3_DIR / "symbolic_bridge (45Q)"
SOVEREIGNTY_DIR = PHASE3_DIR / "sovereignty (45R)"

DRIVE_DIR = Path("/content/drive/MyDrive/AnRa")
DRIVE_CHECKPOINTS = DRIVE_DIR / "checkpoints"
DRIVE_IDENTITY = DRIVE_DIR / "identity"
DRIVE_LOGS = DRIVE_DIR / "logs"
DRIVE_MEMORY = DRIVE_DIR / "memory_db"
DRIVE_SESSIONS = DRIVE_DIR / "sessions"
DRIVE_V2_DIR = DRIVE_DIR / "v2"
DRIVE_V2_CHECKPOINTS = DRIVE_V2_DIR / "checkpoints"
DRIVE_V3_DIR = DRIVE_DIR / "v3"

# Block 2 additions
DRIVE_MANIFEST = DRIVE_DIR / "manifest.json"
DRIVE_AUDIT_LOG = DRIVE_LOGS / "audit.log"
DRIVE_GHOST_DB = DRIVE_MEMORY / "ghost_memory.sqlite3"
DRIVE_FAISS_INDEX = DRIVE_MEMORY / "episodic.faiss"
DRIVE_GRAPH_NODES = DRIVE_MEMORY / "graph_nodes.json"
DRIVE_GRAPH_EDGES = DRIVE_MEMORY / "graph_edges.json"
MEMORY_DIR_LOCAL = ROOT / "memory"

MEMORY_DB_DIR = DRIVE_MEMORY
TEACHER_REASONING_V2_FILE = TRAINING_DATA_DIR / "teacher_reasoning_v2.jsonl"
SYMBOLIC_REASONING_V2_FILE = TRAINING_DATA_DIR / "symbolic_reasoning_v2.jsonl"

DATASET_CANONICAL = TRAINING_DATA_DIR / "anra_training.txt"
DATASET = DATASET_CANONICAL
DATASET_LEGACY = TRAINING_DATA_DIR / "anra_dataset_v6_1.txt"
DATASET_DRIVE_LEGACY = DRIVE_DIR / "anra_dataset_v6_1.txt"
GHOST_DB_LOCAL = MEMORY_DIR_LOCAL / "ghost_memory.sqlite3"
FAISS_INDEX_LOCAL = MEMORY_DIR_LOCAL / "episodic.faiss"
REGRET_STATE = DRIVE_V3_DIR / "training" / "regret_state.json"

OUTPUT_V2_DIR = ROOT / "output" / "v2"
V2_BRAIN_CHECKPOINT = ROOT / "anra_v2_brain.pt"
V2_IDENTITY_CHECKPOINT = ROOT / "anra_v2_identity.pt"
V2_OUROBOROS_CHECKPOINT = ROOT / "anra_v2_ouroboros.pt"
V3_TOKENIZER_FILE = TOKENIZER_DIR / "tokenizer_v3.json"
V2_TOKENIZER_FILE = V3_TOKENIZER_FILE

REQUIRED_DIRS = [
    STATE_DIR,
    OUTPUT_DIR / "checkpoints",
    OUTPUT_DIR / "logs",
    OUTPUT_V2_DIR,
    TRAINING_DATA_DIR,
    ROOT / "checkpoints",
    ROOT / "history",
    MEMORY_DIR_LOCAL,
    WORKSPACE_DIR / "git_workspace",
]


def inject_all_paths() -> None:
    paths = [
        ROOT, CORE_DIR, TRAINING_DIR, INFERENCE_DIR, CONFIG_DIR, SCRIPTS_DIR,
        FINE_TUNING_DIR, MEMORY_DIR, AGENT_LOOP_DIR, SELF_IMPROVEMENT_DIR,
        MASTER_SYSTEM_DIR, IDENTITY_DIR, OUROBOROS_DIR, GHOST_MEMORY_DIR,
        SYMBOLIC_BRIDGE_DIR, SOVEREIGNTY_DIR,
    ]
    for p in reversed(paths):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def ensure_dirs() -> None:
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    # Ensure canonical dataset exists.
    if not DATASET_CANONICAL.exists() and DATASET_DRIVE_LEGACY.exists():
        try:
            import shutil
            shutil.copy2(DATASET_DRIVE_LEGACY, DATASET_CANONICAL)
        except Exception:
            pass


def get_dataset_file() -> Path:
    candidates = [
        DATASET_CANONICAL,
        DATASET,
        TRAINING_DATA_DIR / "anra_dataset_v6_1.txt",
        DATASET_LEGACY,
        DATASET_DRIVE_LEGACY,
        DRIVE_V2_DIR / "anra_dataset_v6_1.txt",
        ROOT / "anra_dataset_v6_1.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return DATASET


def get_tokenizer_file() -> Path:
    for c in [TOKENIZER_DIR / "tokenizer.pkl", ROOT / "tokenizer.pkl"]:
        if c.exists():
            return c
    return TOKENIZER_DIR / "tokenizer.pkl"


def get_identity_file() -> Path:
    candidates = [
        DRIVE_IDENTITY / "anra_identity_combined.txt",
        DRIVE_IDENTITY / "anra_identity_v4_fluent.txt",
        IDENTITY_DIR / "anra_identity_combined.txt",
        IDENTITY_DIR / "anra_identity_v4_fluent.txt",
        ROOT / "anra_identity_combined.txt",
        ROOT / "anra_identity_v4_fluent.txt",
    ]
    for c in candidates:
        if c.exists():
            return c

    # broad scan across drive as final fallback
    if DRIVE_DIR.exists():
        patterns = ("*identity*.txt", "*identity*.md")
        for pattern in patterns:
            for match in DRIVE_DIR.rglob(pattern):
                if match.is_file():
                    return match
    return IDENTITY_DIR / "anra_identity_combined.txt"


def get_checkpoint() -> Path | None:
    for c in [ROOT / "anra_brain_identity.pt", DRIVE_CHECKPOINTS / "anra_brain_identity.pt", ROOT / "anra_brain.pt", DRIVE_CHECKPOINTS / "anra_brain.pt"]:
        if c.exists():
            return c
    return None

def get_optimization_config() -> Path:
    for c in [CONFIG_DIR / "optimization_config.json", ROOT / "AnRa" / "optimization_config.json"]:
        if c.exists():
            return c.resolve()
    return (CONFIG_DIR / "optimization_config.json").resolve()


def get_v2_tokenizer_file() -> Path:
    candidates = [V3_TOKENIZER_FILE, V2_TOKENIZER_FILE, ROOT / "tokenizer_v2.json", ROOT / "tokenizer_v3.json", DRIVE_V2_DIR / "tokenizer_v3.json", DRIVE_V2_DIR / "tokenizer_v2.json"]
    for c in candidates:
        if c.exists():
            return c
    return V2_TOKENIZER_FILE


def get_v2_checkpoint(kind: str = "brain") -> Path:
    mapping = {
        "brain": [V2_BRAIN_CHECKPOINT, DRIVE_V2_CHECKPOINTS / V2_BRAIN_CHECKPOINT.name],
        "identity": [V2_IDENTITY_CHECKPOINT, DRIVE_V2_CHECKPOINTS / V2_IDENTITY_CHECKPOINT.name],
        "ouroboros": [V2_OUROBOROS_CHECKPOINT, DRIVE_V2_CHECKPOINTS / V2_OUROBOROS_CHECKPOINT.name],
    }
    candidates = mapping.get(kind, mapping["brain"])
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def get_v2_checkpoints_dir() -> Path:
    return DRIVE_V2_CHECKPOINTS


class PathRegistry:
    ROOT = ROOT; PROJECT_ROOT = PROJECT_ROOT; CORE_DIR = CORE_DIR; TRAINING_DIR = TRAINING_DIR
    INFERENCE_DIR = INFERENCE_DIR; TOKENIZER_DIR = TOKENIZER_DIR; CONFIG_DIR = CONFIG_DIR; SCRIPTS_DIR = SCRIPTS_DIR
    TESTS_DIR = TESTS_DIR; TRAINING_DATA_DIR = TRAINING_DATA_DIR; OUTPUT_DIR = OUTPUT_DIR; STATE_DIR = STATE_DIR
    WORKSPACE_DIR = WORKSPACE_DIR; PHASE2_DIR = PHASE2_DIR; FINE_TUNING_DIR = FINE_TUNING_DIR; MEMORY_DIR = MEMORY_DIR
    AGENT_LOOP_DIR = AGENT_LOOP_DIR; SELF_IMPROVEMENT_DIR = SELF_IMPROVEMENT_DIR; MASTER_SYSTEM_DIR = MASTER_SYSTEM_DIR
    PHASE3_DIR = PHASE3_DIR; IDENTITY_DIR = IDENTITY_DIR; OUROBOROS_DIR = OUROBOROS_DIR; GHOST_MEMORY_DIR = GHOST_MEMORY_DIR
    SYMBOLIC_BRIDGE_DIR = SYMBOLIC_BRIDGE_DIR; SOVEREIGNTY_DIR = SOVEREIGNTY_DIR; DRIVE_DIR = DRIVE_DIR
    DRIVE_CHECKPOINTS = DRIVE_CHECKPOINTS; DRIVE_IDENTITY = DRIVE_IDENTITY; DRIVE_LOGS = DRIVE_LOGS; DRIVE_MEMORY = DRIVE_MEMORY
    DRIVE_SESSIONS = DRIVE_SESSIONS; DRIVE_V2_DIR = DRIVE_V2_DIR; DRIVE_V2_CHECKPOINTS = DRIVE_V2_CHECKPOINTS
    DRIVE_V3_DIR = DRIVE_V3_DIR
    DRIVE_MANIFEST = DRIVE_MANIFEST; DRIVE_AUDIT_LOG = DRIVE_AUDIT_LOG; DRIVE_GHOST_DB = DRIVE_GHOST_DB
    DRIVE_FAISS_INDEX = DRIVE_FAISS_INDEX; DRIVE_GRAPH_NODES = DRIVE_GRAPH_NODES; DRIVE_GRAPH_EDGES = DRIVE_GRAPH_EDGES
    MEMORY_DB_DIR = MEMORY_DB_DIR; TEACHER_REASONING_V2_FILE = TEACHER_REASONING_V2_FILE; SYMBOLIC_REASONING_V2_FILE = SYMBOLIC_REASONING_V2_FILE
    DATASET = DATASET; DATASET_CANONICAL = DATASET_CANONICAL; DATASET_LEGACY = DATASET_LEGACY; DATASET_DRIVE_LEGACY = DATASET_DRIVE_LEGACY; OUTPUT_V2_DIR = OUTPUT_V2_DIR
    GHOST_DB_LOCAL = GHOST_DB_LOCAL; FAISS_INDEX_LOCAL = FAISS_INDEX_LOCAL; REGRET_STATE = REGRET_STATE
    V2_BRAIN_CHECKPOINT = V2_BRAIN_CHECKPOINT; V2_IDENTITY_CHECKPOINT = V2_IDENTITY_CHECKPOINT; V2_OUROBOROS_CHECKPOINT = V2_OUROBOROS_CHECKPOINT
    V2_TOKENIZER_FILE = V2_TOKENIZER_FILE; V3_TOKENIZER_FILE = V3_TOKENIZER_FILE
