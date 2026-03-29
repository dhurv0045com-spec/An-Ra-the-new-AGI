"""
45P Ghost State Memory — capture, quantize, and persist conversation turns.

Thread-safe SQLite metadata plus a parallel numpy object archive of compressed
embedding blobs. Embeddings default to sentence-transformers/all-MiniLM-L6-v2
but callers may inject a mock embedder for tests and offline demos.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import GhostConfig, default_config
from .injector import build_prompt, format_ghost_block
from .quantizer import compress_vector
from .retriever import retrieve_memories

EmbedFn = Callable[[str], np.ndarray]


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the memories table if it does not exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            vector_idx INTEGER NOT NULL
        )
        """
    )
    conn.commit()


class MemoryStore:
    """
    Persistent memory: each turn is embedded, compressed, and indexed.

    Why: separates hot in-memory vectors from durable SQLite rows so retrieval
    can scan blobs without reloading raw chat history text at scale.
    """

    def __init__(
        self,
        config: Optional[GhostConfig] = None,
        embedder: Optional[EmbedFn] = None,
    ) -> None:
        """
        Open or create storage under ``config.storage_dir`` and load vector archive.

        ``embedder`` overrides MiniLM for deterministic tests when provided.
        """
        self._config = config or default_config()
        self._embedder: Optional[EmbedFn] = embedder
        self._model = None
        self._lock = threading.RLock()
        self._vectors: List[bytes] = []
        self._storage_dir = Path(self._config.storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_vectors_file()
        self._init_db()

    def _init_db(self) -> None:
        """Open SQLite and ensure schema exists."""
        with self._lock:
            conn = sqlite3.connect(self._config.db_path(), check_same_thread=False)
            try:
                _ensure_schema(conn)
            finally:
                conn.close()

    def _load_vectors_file(self) -> None:
        """Load compressed blobs from numpy object archive if present."""
        path = self._config.vectors_path()
        if not path.is_file():
            self._vectors = []
            return
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray):
            self._vectors = [bytes(x) for x in arr.flat]
        else:
            self._vectors = []

    def _save_vectors_file(self) -> None:
        """Persist compressed blobs to numpy .npy (object array)."""
        path = self._config.vectors_path()
        np.save(path, np.array(self._vectors, dtype=object), allow_pickle=True)

    def _get_sentence_model(self):
        """Lazy-load the sentence-transformers model once per process."""
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._config.embedding_model)
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """
        Produce a dense embedding vector for `text`.

        Uses the injected embedder when provided; otherwise loads
        SentenceTransformer and returns float32 numpy of shape (embedding_dim,).
        """
        if self._embedder is not None:
            v = self._embedder(text)
            return np.asarray(v, dtype=np.float32).ravel()
        model = self._get_sentence_model()
        v = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        out = np.asarray(v, dtype=np.float32).ravel()
        if out.shape[0] != self._config.embedding_dim:
            raise ValueError(
                f"Embedding dim {out.shape[0]} != config {self._config.embedding_dim}"
            )
        return out

    def add_turn(self, role: str, text: str) -> int:
        """
        Embed and store a single conversation line (user or assistant).

        Returns the new SQLite row id. Prunes oldest low-decay rows if over cap.
        """
        clean_role = role.strip() or "user"
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("text must be non-empty")
        with self._lock:
            vec = self.embed_text(clean_text)
            blob = compress_vector(
                vec,
                bits=self._config.quantize_bits,
                key_bits=self._config.key_quantize_bits,
            )
            idx = len(self._vectors)
            self._vectors.append(blob)
            self._save_vectors_file()
            conn = sqlite3.connect(self._config.db_path(), check_same_thread=False)
            try:
                cur = conn.execute(
                    "INSERT INTO memories (created_at, role, text, vector_idx) VALUES (?, ?, ?, ?)",
                    (time.time(), clean_role, clean_text, idx),
                )
                conn.commit()
                new_id = int(cur.lastrowid)
            finally:
                conn.close()
            self._maybe_prune()
            return new_id

    def _maybe_prune(self) -> None:
        """If row count exceeds max_memories, drop lowest decay-weighted rows."""
        conn = sqlite3.connect(self._config.db_path(), check_same_thread=False)
        try:
            cur = conn.execute("SELECT COUNT(*) FROM memories")
            (n,) = cur.fetchone()
            if n <= self._config.max_memories:
                return
            cur = conn.execute(
                "SELECT id, created_at, role, text, vector_idx FROM memories ORDER BY id ASC"
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        from . import decay as decay_mod

        scored = []
        now = time.time()
        for mid, created_at, role, text, vidx in rows:
            d = decay_mod.apply_decay(
                created_at, reference_time=now, half_life_days=self._config.decay_half_life_days
            )
            scored.append((d, mid, created_at, role, text, vidx))
        scored.sort(key=lambda x: x[0])
        n_drop = len(scored) - self._config.max_memories
        to_remove = set(t[1] for t in scored[:n_drop])
        if not to_remove:
            return
        kept = [t for t in scored if t[1] not in to_remove]
        new_vectors: List[bytes] = []
        with self._lock:
            conn = sqlite3.connect(self._config.db_path(), check_same_thread=False)
            try:
                conn.execute("DELETE FROM memories")
                for _, mid, created_at, role, text, vidx in kept:
                    new_idx = len(new_vectors)
                    new_vectors.append(self._vectors[int(vidx)])
                    conn.execute(
                        "INSERT INTO memories (id, created_at, role, text, vector_idx) VALUES (?, ?, ?, ?, ?)",
                        (mid, created_at, role, text, new_idx),
                    )
                conn.commit()
            finally:
                conn.close()
            self._vectors = new_vectors
            self._save_vectors_file()

    def iter_retrieval_rows(self) -> List[Tuple[int, float, str, str, bytes]]:
        """
        Return all memories as tuples for the retriever.

        Format: (id, created_at, role, text, compressed_blob).
        """
        with self._lock:
            conn = sqlite3.connect(self._config.db_path(), check_same_thread=False)
            try:
                cur = conn.execute(
                    "SELECT id, created_at, role, text, vector_idx FROM memories ORDER BY id ASC"
                )
                sql_rows = cur.fetchall()
            finally:
                conn.close()
        out: List[Tuple[int, float, str, str, bytes]] = []
        for mid, created_at, role, text, vidx in sql_rows:
            blob = self._vectors[int(vidx)]
            out.append((int(mid), float(created_at), str(role), str(text), blob))
        return out


class GhostMemory:
    """
    High-level façade: add turns, retrieve, and build Ghost Context prompts.

    Integrators typically hold one instance per user/session directory.
    """

    def __init__(
        self,
        config: Optional[GhostConfig] = None,
        embedder: Optional[EmbedFn] = None,
    ) -> None:
        """
        Construct a façade around :class:`MemoryStore` with the same options.

        ``embedder`` is optional; when omitted, ``SentenceTransformer`` is loaded
        on first embedding call.
        """
        self._config = config or default_config()
        self._store = MemoryStore(config=self._config, embedder=embedder)

    @property
    def config(self) -> GhostConfig:
        """Active :class:`GhostConfig` (same object as the underlying store)."""
        return self._store._config

    def add_turn(self, role: str, text: str) -> int:
        """Record a conversation line with compressed embedding."""
        return self._store.add_turn(role, text)

    def retrieve(self, query_text: str) -> List[Dict[str, object]]:
        """Return ranked memory dicts for injection."""
        rows = self._store.iter_retrieval_rows()
        return retrieve_memories(
            query_text,
            rows,
            self._store.embed_text,
            self._config,
        )

    def build_ghost_prompt(self, user_message: str) -> str:
        """
        Retrieve relevant memories for `user_message` and prepend Ghost Context.

        Returns the full string to send to the model (ghost block + user text).
        """
        snippets = self.retrieve(user_message)
        block = format_ghost_block(snippets)
        return build_prompt(block, user_message)

    def memory_store(self) -> MemoryStore:
        """Expose the underlying store for advanced integrations."""
        return self._store
