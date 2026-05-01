from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Legacy alias used by older notebooks and Gemini-generated Colab cells.
# Keep this exported forever to avoid `ImportError: cannot import name 'PROJECT_ROOT'`.
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
DRIVE_TOKENIZER_DIR = DRIVE_DIR / "tokenizer"
DRIVE_TOKENIZER_V3_DIR = DRIVE_TOKENIZER_DIR / "v3"

OUTPUT_V2_DIR = ROOT / "output" / "v2"
V2_BRAIN_CHECKPOINT = ROOT / "anra_v2_brain.pt"
V2_IDENTITY_CHECKPOINT = ROOT / "anra_v2_identity.pt"
V2_OUROBOROS_CHECKPOINT = ROOT / "anra_v2_ouroboros.pt"
V2_TOKENIZER_FILE = TOKENIZER_DIR / "tokenizer_v2.json"
V3_TOKENIZER_FILE = TOKENIZER_DIR / "tokenizer_v3.json"
V3_TOKENIZER_META_FILE = TOKENIZER_DIR / "tokenizer_v3.json.meta.json"
DATASET_PRIMARY = TRAINING_DATA_DIR / "anra_dataset_v6_1.txt"
DATASET_FALLBACK = DRIVE_DIR / "anra_dataset_v6_1.txt"
TEACHER_REASONING_V2_FILE = TRAINING_DATA_DIR / "teacher_reasoning_v2.jsonl"
MEMORY_DB_DIR = DRIVE_DIR / "memory_db"
GOALS_FILE = STATE_DIR / "goals.json"
REPLAY_BUFFER_FILE = STATE_DIR / "replay_buffer.jsonl"
STATE_DB_FILE = STATE_DIR / "anra_state.db"
AUDIT_REPORT_FILE = OUTPUT_V2_DIR / "audit_report.json"

REQUIRED_DIRS = [
    STATE_DIR,
    OUTPUT_DIR / "checkpoints",
    OUTPUT_DIR / "logs",
    OUTPUT_V2_DIR,
    TRAINING_DATA_DIR,
    ROOT / "checkpoints",
    ROOT / "history",
]


def inject_all_paths() -> None:
    paths = [
        ROOT,
        CORE_DIR,
        TRAINING_DIR,
        INFERENCE_DIR,
        CONFIG_DIR,
        SCRIPTS_DIR,
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
        DATASET_PRIMARY,
        DATASET_FALLBACK,
    ]
    for c in candidates:
        if c.exists():
            return c
    return DATASET_PRIMARY


def get_tokenizer_file() -> Path:
    """Find the tokenizer pickle."""
    candidates = [
        TOKENIZER_DIR / "tokenizer.pkl",
        ROOT / "tokenizer.pkl",
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


def get_v2_tokenizer_file() -> Path:
    candidates = [
        V2_TOKENIZER_FILE,
        ROOT / "tokenizer_v3.json",
        DRIVE_V2_DIR / "tokenizer_v3.json",
    ]
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
    """Backward-compatible helper kept for legacy notebook code."""
    return DRIVE_V2_CHECKPOINTS


# ── Backward-compatibility: PathRegistry class ────────────────────────────────
# Some legacy notebooks import `from anra_paths import PathRegistry`.
# This class wraps all module-level constants so those imports succeed.

class PathRegistry:
    """Compatibility shim — exposes every path constant as a class attribute."""

    ROOT = ROOT
    PROJECT_ROOT = PROJECT_ROOT
    CORE_DIR = CORE_DIR
    TRAINING_DIR = TRAINING_DIR
    INFERENCE_DIR = INFERENCE_DIR
    TOKENIZER_DIR = TOKENIZER_DIR
    CONFIG_DIR = CONFIG_DIR
    SCRIPTS_DIR = SCRIPTS_DIR
    TESTS_DIR = TESTS_DIR
    TRAINING_DATA_DIR = TRAINING_DATA_DIR
    OUTPUT_DIR = OUTPUT_DIR
    STATE_DIR = STATE_DIR
    WORKSPACE_DIR = WORKSPACE_DIR

    PHASE2_DIR = PHASE2_DIR
    FINE_TUNING_DIR = FINE_TUNING_DIR
    MEMORY_DIR = MEMORY_DIR
    AGENT_LOOP_DIR = AGENT_LOOP_DIR
    SELF_IMPROVEMENT_DIR = SELF_IMPROVEMENT_DIR
    MASTER_SYSTEM_DIR = MASTER_SYSTEM_DIR

    PHASE3_DIR = PHASE3_DIR
    IDENTITY_DIR = IDENTITY_DIR
    OUROBOROS_DIR = OUROBOROS_DIR
    GHOST_MEMORY_DIR = GHOST_MEMORY_DIR
    SYMBOLIC_BRIDGE_DIR = SYMBOLIC_BRIDGE_DIR
    SOVEREIGNTY_DIR = SOVEREIGNTY_DIR

    DRIVE_DIR = DRIVE_DIR
    DRIVE_CHECKPOINTS = DRIVE_CHECKPOINTS
    DRIVE_IDENTITY = DRIVE_IDENTITY
    DRIVE_LOGS = DRIVE_LOGS
    DRIVE_MEMORY = DRIVE_MEMORY
    DRIVE_SESSIONS = DRIVE_SESSIONS
    DRIVE_V2_DIR = DRIVE_V2_DIR
    DRIVE_V2_CHECKPOINTS = DRIVE_V2_CHECKPOINTS
    DRIVE_TOKENIZER_DIR = DRIVE_TOKENIZER_DIR
    DRIVE_TOKENIZER_V3_DIR = DRIVE_TOKENIZER_V3_DIR
    OUTPUT_V2_DIR = OUTPUT_V2_DIR
    V2_BRAIN_CHECKPOINT = V2_BRAIN_CHECKPOINT
    V2_IDENTITY_CHECKPOINT = V2_IDENTITY_CHECKPOINT
    V2_OUROBOROS_CHECKPOINT = V2_OUROBOROS_CHECKPOINT
    V2_TOKENIZER_FILE = V2_TOKENIZER_FILE
    V3_TOKENIZER_FILE = V3_TOKENIZER_FILE
    V3_TOKENIZER_META_FILE = V3_TOKENIZER_META_FILE
    DATASET_PRIMARY = DATASET_PRIMARY
    DATASET_FALLBACK = DATASET_FALLBACK
    TEACHER_REASONING_V2_FILE = TEACHER_REASONING_V2_FILE
    MEMORY_DB_DIR = MEMORY_DB_DIR
    GOALS_FILE = GOALS_FILE
    REPLAY_BUFFER_FILE = REPLAY_BUFFER_FILE
    STATE_DB_FILE = STATE_DB_FILE
    AUDIT_REPORT_FILE = AUDIT_REPORT_FILE

    @staticmethod
    def inject_all_paths() -> None:
        inject_all_paths()

    @staticmethod
    def ensure_dirs() -> None:
        ensure_dirs()

    @staticmethod
    def get_dataset_file() -> Path:
        return get_dataset_file()

    @staticmethod
    def get_tokenizer_file() -> Path:
        return get_tokenizer_file()

    @staticmethod
    def get_identity_file() -> Path:
        return get_identity_file()

    @staticmethod
    def get_checkpoint() -> Path | None:
        return get_checkpoint()

    @staticmethod
    def get_optimization_config() -> Path:
        return get_optimization_config()

    @staticmethod
    def get_v2_tokenizer_file() -> Path:
        return get_v2_tokenizer_file()

    @staticmethod
    def get_v2_checkpoint(kind: str = "brain") -> Path:
        return get_v2_checkpoint(kind)

    @staticmethod
    def get_v2_checkpoints_dir() -> Path:
        return get_v2_checkpoints_dir()
