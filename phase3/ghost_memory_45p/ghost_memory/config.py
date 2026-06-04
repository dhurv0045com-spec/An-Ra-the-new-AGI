"""
45P Ghost State Memory — configuration.

Centralizes tunable defaults (quantization, retrieval, decay, storage paths)
so integrations and tests can override a single GhostConfig object without
hunting through the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Defaults requested by the 45P specification
QUANTIZE_BITS: int = 3
# Key half uses extra bits on the unit sphere for stable cosine retrieval (3–8).
KEY_QUANTIZE_BITS: int = 5
TOP_K: int = 5
DECAY_HALF_LIFE_DAYS: float = 30.0
MAX_MEMORIES: int = 10_000
SIMILARITY_THRESH: float = 0.65

EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384


@dataclass
class GhostConfig:
    """
    Tunable parameters for Ghost State Memory.

    Attributes mirror module-level constants but allow per-instance overrides
    for tests and multi-tenant deployments.
    """

    quantize_bits: int = QUANTIZE_BITS
    key_quantize_bits: int = KEY_QUANTIZE_BITS
    top_k: int = TOP_K
    decay_half_life_days: float = DECAY_HALF_LIFE_DAYS
    max_memories: int = MAX_MEMORIES
    similarity_thresh: float = SIMILARITY_THRESH
    embedding_model: str = EMBEDDING_MODEL
    embedding_dim: int = EMBEDDING_DIM
    storage_dir: Path = field(default_factory=lambda: Path.home() / ".ghost_memory")
    db_filename: str = "memories.sqlite"
    vectors_filename: str = "compressed_vectors.npy"
    index_filename: str = "vector_index.json"

    def db_path(self) -> Path:
        """Absolute path to the SQLite database file."""
        return self.storage_dir / self.db_filename

    def vectors_path(self) -> Path:
        """Absolute path to the numpy archive of compressed rows."""
        return self.storage_dir / self.vectors_filename

    def index_path(self) -> Path:
        """JSON sidecar for variable-length row offsets (if used)."""
        return self.storage_dir / self.index_filename


def default_config(storage_dir: Optional[Path] = None) -> GhostConfig:
    """
    Build a GhostConfig with optional storage directory override.

    Why: callers often want an isolated directory per session or test without
    mutating global defaults.
    """
    if storage_dir is None:
        return GhostConfig()
    p = Path(storage_dir)
    return GhostConfig(storage_dir=p.resolve())
