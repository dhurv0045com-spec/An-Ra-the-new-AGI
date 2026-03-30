"""
memory/vector_memory.py — Vector Memory with Semantic Search

Stores all memories as dense vector embeddings.
Retrieves semantically similar memories at query time.
Enables the model to find relevant past interactions
even when exact keywords don't match.

No external dependencies. Pure NumPy TF-IDF embeddings
with cosine similarity. Drop-in upgrade path to real
embeddings (sentence-transformers) when available.

Architecture:
    Text → Embed → Store in matrix M (N × D)
    Query → Embed → Cosine similarity with all rows → Top-K
"""

import numpy as np
import json, os, pickle, hashlib, time, re, math
import sqlite3, threading
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime


STATE_DIR  = Path("state")
MEMORY_DIR = Path("state/vector_memory")
STATE_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_DIR.mkdir(parents=True, exist_ok=True)


# ── Embedding engine ───────────────────────────────────────────────────────────

class TFIDFEmbedder:
    """
    TF-IDF document embedder using a fixed vocabulary.
    Produces dense-ish vectors by hashing tokens to fixed dimensions.

    Upgrade path:
        Replace with SentenceTransformer('all-MiniLM-L6-v2') for real semantic embeddings.
        Interface is identical — just swap this class.
    """

    DIM = 512   # embedding dimension

    def __init__(self):
        self._idf:   Optional[np.ndarray] = None
        self._vocab: Optional[Dict[str, int]] = None
        self._fitted = False

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-z]{2,}\b', text.lower())

    def _hash_embed(self, text: str) -> np.ndarray:
        """
        Hash-based embedding when IDF is not fitted.
        Deterministic: same text → same vector.
        Not semantic but gives stable, unique representations.
        """
        tokens = self._tokenize(text)
        vec    = np.zeros(self.DIM, dtype=np.float32)
        for token in tokens:
            h   = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % self.DIM
            # Bigram features for slightly better discrimination
            vec[idx]              += 1.0
            vec[(idx + 1) % self.DIM] += 0.5
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)

    def fit(self, texts: List[str]):
        """
        Fit IDF weights on a corpus.
        Call once when you have enough documents.
        """
        from collections import Counter
        vocab_set = set()
        for t in texts:
            vocab_set.update(self._tokenize(t))
        self._vocab = {w: i for i, w in enumerate(sorted(vocab_set))}
        V = len(self._vocab)

        # Compute document frequency
        df = np.zeros(V, dtype=np.float32)
        for t in texts:
            present = set(self._tokenize(t))
            for w in present:
                if w in self._vocab:
                    df[self._vocab[w]] += 1

        self._idf    = np.log((len(texts) + 1) / (df + 1)) + 1.0  # smooth IDF
        self._fitted = True

    def embed(self, text: str) -> np.ndarray:
        """
        Embed text to a unit-norm vector of dimension DIM.
        """
        if not self._fitted or self._vocab is None:
            return self._hash_embed(text)

        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.DIM, dtype=np.float32)

        # TF
        from collections import Counter
        tf   = Counter(tokens)
        V    = len(self._vocab)
        vec  = np.zeros(V, dtype=np.float32)
        for w, count in tf.items():
            if w in self._vocab:
                vec[self._vocab[w]] = math.log(1 + count) * self._idf[self._vocab[w]]

        # Project to DIM via random projection (Johnson-Lindenstrauss)
        if not hasattr(self, '_proj') or self._proj.shape[1] != V:
            rng       = np.random.RandomState(42)
            self._proj = rng.randn(self.DIM, V).astype(np.float32) / math.sqrt(self.DIM)

        vec = self._proj @ vec
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts. Returns (N, DIM) matrix."""
        return np.stack([self.embed(t) for t in texts])


# ── Memory Entry ───────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    mem_id:     str
    timestamp:  str
    text:       str
    category:   str          # "interaction" / "task" / "knowledge" / "goal" / "fact"
    importance: float = 0.5  # 0-1, higher = retrieved more often
    access_count: int = 0
    last_accessed: Optional[str] = None
    metadata:   Dict[str, Any] = field(default_factory=dict)
    # Vector stored separately in matrix


# ── Vector Store ───────────────────────────────────────────────────────────────

class VectorStore:
    """
    Persistent vector memory.
    Stores embeddings in a NumPy matrix + entries in SQLite.
    Supports semantic search via cosine similarity.
    """

    MATRIX_PATH  = MEMORY_DIR / "vectors.npy"
    ID_MAP_PATH  = MEMORY_DIR / "id_map.json"
    EMBEDDER_PATH = MEMORY_DIR / "embedder.pkl"
    DB_PATH      = MEMORY_DIR / "memory.db"

    def __init__(self, embedder: Optional[TFIDFEmbedder] = None):
        self.embedder   = embedder or TFIDFEmbedder()
        self._matrix:   Optional[np.ndarray] = None   # (N, DIM)
        self._id_map:   List[str] = []                 # index → mem_id
        self._lock      = threading.Lock()
        self._conn      = sqlite3.connect(str(self.DB_PATH), check_same_thread=False)
        self._init_db()
        self._load()

    def _init_db(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                mem_id TEXT PRIMARY KEY, data TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_cat ON memories(mem_id);
        """)
        self._conn.commit()

    def _load(self):
        """Load persisted matrix and id map from disk."""
        if self.MATRIX_PATH.exists() and self.ID_MAP_PATH.exists():
            try:
                self._matrix = np.load(str(self.MATRIX_PATH))
                with open(self.ID_MAP_PATH) as f:
                    self._id_map = json.load(f)
            except Exception:
                self._matrix  = None
                self._id_map   = []
        if self.EMBEDDER_PATH.exists():
            try:
                with open(self.EMBEDDER_PATH, "rb") as f:
                    self.embedder = pickle.load(f)
            except Exception:
                pass

    def _save(self):
        """Persist matrix and id map to disk."""
        if self._matrix is not None:
            np.save(str(self.MATRIX_PATH), self._matrix)
        with open(self.ID_MAP_PATH, "w") as f:
            json.dump(self._id_map, f)

    def _save_embedder(self):
        with open(self.EMBEDDER_PATH, "wb") as f:
            pickle.dump(self.embedder, f)

    def add(self, text: str, category: str = "interaction",
            importance: float = 0.5, metadata: dict = None) -> MemoryEntry:
        """
        Add a new memory. Embed the text and store in the matrix.
        Returns the MemoryEntry.
        """
        mem_id = hashlib.sha256((text + str(time.time())).encode()).hexdigest()[:16]
        entry  = MemoryEntry(
            mem_id       = mem_id,
            timestamp    = datetime.utcnow().isoformat(),
            text         = text[:2000],
            category     = category,
            importance   = importance,
            metadata     = metadata or {},
        )

        # Embed
        vec = self.embedder.embed(text).astype(np.float32)

        with self._lock:
            # Append to matrix
            if self._matrix is None:
                self._matrix = vec[np.newaxis, :]
            else:
                self._matrix = np.vstack([self._matrix, vec[np.newaxis, :]])
            self._id_map.append(mem_id)

            # Save to DB
            self._conn.execute(
                "INSERT OR REPLACE INTO memories VALUES (?,?)",
                (mem_id, json.dumps(asdict(entry)))
            )
            self._conn.commit()
            self._save()

        return entry

    def search(self, query: str, top_k: int = 5,
               category: Optional[str] = None,
               min_importance: float = 0.0) -> List[Tuple[float, MemoryEntry]]:
        """
        Semantic search: find the top_k most similar memories to query.

        Args:
            query:          text to search for
            top_k:          number of results to return
            category:       filter by category
            min_importance: minimum importance score

        Returns:
            List of (similarity_score, MemoryEntry) sorted by score descending
        """
        if self._matrix is None or len(self._id_map) == 0:
            return []

        q_vec = self.embedder.embed(query).astype(np.float32)

        # Cosine similarity = dot product (vectors are unit norm)
        similarities = self._matrix @ q_vec                    # (N,)

        # Boost by importance
        importances = np.array([
            self._get_importance(mid) for mid in self._id_map
        ], dtype=np.float32)
        scores = 0.7 * similarities + 0.3 * importances

        # Sort descending
        sorted_idx = np.argsort(scores)[::-1]

        results = []
        for idx in sorted_idx:
            if len(results) >= top_k:
                break
            mem_id = self._id_map[idx]
            entry  = self._get_entry(mem_id)
            if entry is None:
                continue
            if category and entry.category != category:
                continue
            if entry.importance < min_importance:
                continue
            # Update access count
            entry.access_count  += 1
            entry.last_accessed  = datetime.utcnow().isoformat()
            self._update_entry(entry)
            results.append((float(scores[idx]), entry))

        return results

    def _get_importance(self, mem_id: str) -> float:
        entry = self._get_entry(mem_id)
        return entry.importance if entry else 0.5

    def _get_entry(self, mem_id: str) -> Optional[MemoryEntry]:
        row = self._conn.execute(
            "SELECT data FROM memories WHERE mem_id=?", (mem_id,)).fetchone()
        if row:
            return MemoryEntry(**json.loads(row[0]))
        return None

    def _update_entry(self, entry: MemoryEntry):
        self._conn.execute(
            "UPDATE memories SET data=? WHERE mem_id=?",
            (json.dumps(asdict(entry)), entry.mem_id)
        )
        self._conn.commit()

    def all_entries(self, category: Optional[str] = None) -> List[MemoryEntry]:
        rows = self._conn.execute("SELECT data FROM memories").fetchall()
        entries = [MemoryEntry(**json.loads(r[0])) for r in rows]
        if category:
            entries = [e for e in entries if e.category == category]
        return entries

    def forget(self, mem_id: str) -> bool:
        """Remove a memory by ID."""
        with self._lock:
            if mem_id not in self._id_map:
                return False
            idx = self._id_map.index(mem_id)
            self._matrix = np.delete(self._matrix, idx, axis=0)
            self._id_map.pop(idx)
            self._conn.execute("DELETE FROM memories WHERE mem_id=?", (mem_id,))
            self._conn.commit()
            self._save()
        return True

    def refit_embedder(self):
        """Refit TF-IDF on all stored memories for better search quality."""
        entries = self.all_entries()
        if len(entries) < 10:
            return
        texts = [e.text for e in entries]
        self.embedder.fit(texts)
        self._save_embedder()
        # Recompute all embeddings
        new_matrix = np.zeros((len(entries), TFIDFEmbedder.DIM), dtype=np.float32)
        for i, entry in enumerate(entries):
            new_matrix[i] = self.embedder.embed(entry.text)
        with self._lock:
            self._matrix = new_matrix
            self._id_map = [e.mem_id for e in entries]
            self._save()

    def stats(self) -> dict:
        entries = self.all_entries()
        n = len(entries)
        cats = {}
        for e in entries:
            cats[e.category] = cats.get(e.category, 0) + 1
        return {
            "total_memories":  n,
            "matrix_shape":    list(self._matrix.shape) if self._matrix is not None else None,
            "categories":      cats,
            "avg_importance":  sum(e.importance for e in entries) / max(n, 1),
            "embedding_dim":   TFIDFEmbedder.DIM,
        }

    def inject_context(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant memories and format them as context to prepend to a prompt.
        This is how memory augments model responses.
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return ""
        lines = ["[Relevant context from memory:]"]
        for score, entry in results:
            ts = entry.timestamp[:10]
            lines.append(f"  [{ts}] ({entry.category}) {entry.text[:200]}")
        return "\n".join(lines)


# ── Memory-Augmented Generation ────────────────────────────────────────────────

class MemoryAugmentedModel:
    """
    Wraps a TransformerLM with vector memory retrieval.
    Before each generation, retrieves relevant memories and prepends as context.
    """

    def __init__(self, model, tokenizer, vector_store: VectorStore):
        self.model        = model
        self.tokenizer    = tokenizer
        self.vector_store = vector_store

    def generate(self, prompt: str, max_new: int = 100,
                 temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9,
                 use_memory: bool = True) -> Tuple[str, List[Tuple[float, MemoryEntry]]]:
        """
        Generate text with optional memory augmentation.

        Returns:
            (generated_text, retrieved_memories)
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from myai_v2 import generate as base_generate

        retrieved = []
        augmented_prompt = prompt

        if use_memory:
            retrieved = self.vector_store.search(prompt, top_k=3)
            context   = self.vector_store.inject_context(prompt, top_k=3)
            if context:
                augmented_prompt = context + "\n\n" + prompt

        output = base_generate(
            self.model, self.tokenizer, augmented_prompt,
            max_new=max_new, temperature=temperature,
            top_k=top_k, top_p=top_p,
        )

        # Store this interaction as a new memory
        self.vector_store.add(
            text       = f"Q: {prompt[:200]}\nA: {output[:300]}",
            category   = "interaction",
            importance = 0.5,
        )

        return output, retrieved
