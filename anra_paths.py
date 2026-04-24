from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CORE_DIR = ROOT / "core"
TRAINING_DIR = ROOT / "training"
INFERENCE_DIR = ROOT / "inference"
TOKENIZER_DIR = ROOT / "tokenizer"
CONFIG_DIR = ROOT / "config"
SCRIPTS_DIR = ROOT / "scripts"
TESTS_DIR = ROOT / "tests"
TRAINING_DATA_DIR = ROOT / "training_data"

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

REQUIRED_DIRS = [
    ROOT / "state",
    ROOT / "output" / "checkpoints",
    ROOT / "output" / "logs",
    ROOT / "training_data",
    ROOT / "checkpoints",
    ROOT / "history",
]


def inject_all_paths() -> None:
    paths = [
        ROOT,
        FINE_TUNING_DIR,
        MEMORY_DIR,
        AGENT_LOOP_DIR,
        SELF_IMPROVEMENT_DIR,
        MASTER_SYSTEM_DIR,
        IDENTITY_DIR,
        OUROBOROS_DIR,
        GHOST_MEMORY_DIR,
        SYMBOLIC_BRIDGE_DIR,
        SOVEREIGNTY_DIR,
    ]
    for p in reversed(paths):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def ensure_dirs() -> None:
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def get_dataset_file() -> Path:
    """Find the primary training dataset."""
    candidates = [
        TRAINING_DATA_DIR / "anra_dataset_v6_1.txt",
        ROOT / "anra_dataset_v6_1.txt",
        DRIVE_DIR / "anra_dataset_v6_1.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return TRAINING_DATA_DIR / "anra_dataset_v6_1.txt"


def get_tokenizer_file() -> Path:
    """Find the tokenizer pickle."""
    candidates = [
        TOKENIZER_DIR / "tokenizer.pkl",
        ROOT / "tokenizer.pkl",
        DRIVE_DIR / "tokenizer.pkl",
    ]
    for c in candidates:
        if c.exists():
            return c
    return TOKENIZER_DIR / "tokenizer.pkl"


def get_identity_file() -> Path:
    candidates = [
        IDENTITY_DIR / "anra_identity_combined.txt",
        ROOT / "anra_identity_combined.txt",
        DRIVE_IDENTITY / "anra_identity_combined.txt",
        IDENTITY_DIR / "anra_identity_v4_fluent.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return IDENTITY_DIR / "anra_identity_combined.txt"


def get_checkpoint() -> Path | None:
    candidates = [
        ROOT / "anra_brain_identity.pt",
        DRIVE_CHECKPOINTS / "anra_brain_identity.pt",
        ROOT / "anra_brain.pt",
        DRIVE_CHECKPOINTS / "anra_brain.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def get_optimization_config() -> Path:
    """Find optimization config at an absolute path."""
    candidates = [
        CONFIG_DIR / "optimization_config.json",
        ROOT / "AnRa" / "optimization_config.json",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return (CONFIG_DIR / "optimization_config.json").resolve()
