"""
memory/store.py — Step 1: Memory Store Architecture
Core schema, types, and persistence layer.
SQLite backend: reliable, zero-dependency, survives restarts.
"""

import sqlite3
import json
import uuid
import time
import hashlib
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path


# ─────────────────────────────────────────────
# Memory types
# ─────────────────────────────────────────────

class MemoryType(str, Enum):
    EPISODIC = "episodic"    # specific past conversations
    SEMANTIC  = "semantic"   # extracted facts and knowledge
    WORKING   = "working"    # current session, volatile


class ImportanceLevel(str, Enum):
    CRITICAL = "critical"   # never expires (5.0)
    HIGH     = "high"       # long lived (4.0)
    MEDIUM   = "medium"     # standard (3.0)
    LOW      = "low"        # expires first (2.0)
    EPHEMERAL = "ephemeral" # session only (1.0)

IMPORTANCE_SCORES = {
    ImportanceLevel.CRITICAL:  5.0,
    ImportanceLevel.HIGH:      4.0,
    ImportanceLevel.MEDIUM:    3.0,
    ImportanceLevel.LOW:       2.0,
    ImportanceLevel.EPHEMERAL: 1.0,
}


@dataclass
class Memory:
    """A single unit of memory."""
    id:           str
    type:         MemoryType
    content:      str                       # raw text
    summary:      str                       # short summary (1 line)
    metadata:     Dict[str, Any]            # flexible structured data
    importance:   float                     # 0.0 – 5.0
    created_at:   float                     # unix timestamp
    accessed_at:  float                     # last retrieval
    access_count: int                       # how many times retrieved
    expires_at:   Optional[float]           # None = never
    user_id:      str                       # owner
    session_id:   Optional[str]             # originating session
    embedding:    Optional[List[float]]     # vector (stored separately)
    tags:         List[str]                 # topic tags

    @classmethod
    def create(cls, content: str, type: MemoryType,
               summary: str = "",
               metadata: Optional[Dict] = None,
               importance: float = 3.0,
               user_id: str = "default",
               session_id: Optional[str] = None,
               tags: Optional[List[str]] = None,
               ttl_days: Optional[float] = None) -> "Memory":
        now = time.time()
        expires = now + ttl_days * 86400 if ttl_days else None
        return cls(
            id=str(uuid.uuid4()),
            type=type,
            content=content,
            summary=summary or content[:120],
            metadata=metadata or {},
            importance=importance,
            created_at=now,
            accessed_at=now,
            access_count=0,
            expires_at=expires,
            user_id=user_id,
            session_id=session_id,
            embedding=None,
            tags=tags or [],
        )

    def touch(self):
        self.accessed_at = time.time()
        self.access_count += 1

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Memory":
        d = dict(d)
        d["type"] = MemoryType(d["type"])
        if "embedding" in d and isinstance(d["embedding"], str):
            d["embedding"] = json.loads(d["embedding"])
        if "metadata" in d and isinstance(d["metadata"], str):
            d["metadata"] = json.loads(d["metadata"])
        if "tags" in d and isinstance(d["tags"], str):
            d["tags"] = json.loads(d["tags"])
        return cls(**d)


# ─────────────────────────────────────────────
# SQLite schema + persistence
# ─────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,
    type         TEXT NOT NULL,
    content      TEXT NOT NULL,
    summary      TEXT NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}',
    importance   REAL NOT NULL DEFAULT 3.0,
    created_at   REAL NOT NULL,
    accessed_at  REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    expires_at   REAL,
    user_id      TEXT NOT NULL DEFAULT 'default',
    session_id   TEXT,
    tags         TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS embeddings (
    memory_id    TEXT PRIMARY KEY,
    vector       BLOB NOT NULL,
    model_name   TEXT NOT NULL,
    dim          INTEGER NOT NULL,
    created_at   REAL NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS deletion_log (
    id           TEXT PRIMARY KEY,
    memory_id    TEXT NOT NULL,
    reason       TEXT,
    deleted_at   REAL NOT NULL,
    user_id      TEXT NOT NULL,
    content_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    started_at   REAL NOT NULL,
    ended_at     REAL,
    summary      TEXT,
    message_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_memories_type       ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_user       ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_memories_created    ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_accessed   ON memories(accessed_at DESC);
"""


class MemoryStore:
    """
    Core persistence layer. Wraps SQLite.
    All higher-level memory systems (episodic, semantic, working)
    go through this interface.
    """

    def __init__(self, db_path: str = "memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ── CRUD ──────────────────────────────────

    def save(self, memory: Memory) -> Memory:
        self._conn.execute("""
            INSERT OR REPLACE INTO memories
            (id, type, content, summary, metadata, importance,
             created_at, accessed_at, access_count, expires_at,
             user_id, session_id, tags)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            memory.id, memory.type.value, memory.content,
            memory.summary, json.dumps(memory.metadata),
            memory.importance, memory.created_at, memory.accessed_at,
            memory.access_count, memory.expires_at,
            memory.user_id, memory.session_id,
            json.dumps(memory.tags),
        ))
        self._conn.commit()
        return memory

    def get(self, memory_id: str) -> Optional[Memory]:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_memory(row) if row else None

    def update(self, memory: Memory):
        self.save(memory)

    def delete(self, memory_id: str, reason: str = "explicit",
               user_id: str = "default"):
        mem = self.get(memory_id)
        if mem:
            content_hash = hashlib.sha256(mem.content.encode()).hexdigest()
            self._conn.execute("""
                INSERT INTO deletion_log (id, memory_id, reason, deleted_at, user_id, content_hash)
                VALUES (?,?,?,?,?,?)
            """, (str(uuid.uuid4()), memory_id, reason,
                  time.time(), user_id, content_hash))
            self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._conn.commit()

    def list(self, user_id: str = "default",
             type: Optional[MemoryType] = None,
             limit: int = 100,
             offset: int = 0,
             order_by: str = "importance DESC, accessed_at DESC") -> List[Memory]:
        q = "SELECT * FROM memories WHERE user_id = ?"
        params: list = [user_id]
        if type:
            q += " AND type = ?"
            params.append(type.value)
        q += f" ORDER BY {order_by} LIMIT ? OFFSET ?"
        params += [limit, offset]
        rows = self._conn.execute(q, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def count(self, user_id: str = "default",
              type: Optional[MemoryType] = None) -> int:
        q = "SELECT COUNT(*) FROM memories WHERE user_id = ?"
        params: list = [user_id]
        if type:
            q += " AND type = ?"
            params.append(type.value)
        return self._conn.execute(q, params).fetchone()[0]

    # ── Embeddings ────────────────────────────

    def save_embedding(self, memory_id: str, vector: list,
                       model_name: str = "tfidf"):
        import struct
        blob = struct.pack(f"{len(vector)}f", *vector)
        self._conn.execute("""
            INSERT OR REPLACE INTO embeddings
            (memory_id, vector, model_name, dim, created_at)
            VALUES (?,?,?,?,?)
        """, (memory_id, blob, model_name, len(vector), time.time()))
        self._conn.commit()

    def load_embedding(self, memory_id: str) -> Optional[List[float]]:
        import struct
        row = self._conn.execute(
            "SELECT vector, dim FROM embeddings WHERE memory_id = ?",
            (memory_id,)
        ).fetchone()
        if not row:
            return None
        return list(struct.unpack(f"{row['dim']}f", row["vector"]))

    def load_all_embeddings(self, user_id: str = "default",
                             type: Optional[MemoryType] = None):
        """Returns {memory_id: vector} for all memories with embeddings."""
        import struct
        q = """
            SELECT e.memory_id, e.vector, e.dim
            FROM embeddings e
            JOIN memories m ON e.memory_id = m.id
            WHERE m.user_id = ?
        """
        params: list = [user_id]
        if type:
            q += " AND m.type = ?"
            params.append(type.value)
        rows = self._conn.execute(q, params).fetchall()
        result = {}
        for row in rows:
            vec = list(struct.unpack(f"{row['dim']}f", row["vector"]))
            result[row["memory_id"]] = vec
        return result

    # ── Sessions ──────────────────────────────

    def start_session(self, user_id: str = "default") -> str:
        sid = str(uuid.uuid4())
        self._conn.execute("""
            INSERT INTO sessions (id, user_id, started_at)
            VALUES (?,?,?)
        """, (sid, user_id, time.time()))
        self._conn.commit()
        return sid

    def end_session(self, session_id: str, summary: str = ""):
        self._conn.execute("""
            UPDATE sessions SET ended_at = ?, summary = ?
            WHERE id = ?
        """, (time.time(), summary, session_id))
        self._conn.commit()

    # ── Bulk operations ───────────────────────

    def purge_expired(self) -> int:
        now = time.time()
        rows = self._conn.execute(
            "SELECT id FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,)
        ).fetchall()
        for row in rows:
            self.delete(row["id"], reason="expired")
        return len(rows)

    def wipe(self, user_id: str = "default", confirm: bool = False):
        if not confirm:
            raise ValueError("Pass confirm=True to wipe all memories")
        ids = self._conn.execute(
            "SELECT id FROM memories WHERE user_id = ?", (user_id,)
        ).fetchall()
        for row in ids:
            self.delete(row["id"], reason="wipe", user_id=user_id)

    def keyword_search(self, query: str, user_id: str = "default",
                        type: Optional[MemoryType] = None,
                        limit: int = 20) -> List[Memory]:
        words = query.lower().split()
        if not words:
            return []
        q = "SELECT * FROM memories WHERE user_id = ?"
        params: list = [user_id]
        if type:
            q += " AND type = ?"
            params.append(type.value)
        q += " AND ("
        q += " OR ".join(
            "(LOWER(content) LIKE ? OR LOWER(summary) LIKE ? OR LOWER(tags) LIKE ?)"
            for _ in words
        )
        q += ")"
        for w in words:
            params += [f"%{w}%", f"%{w}%", f"%{w}%"]
        q += f" ORDER BY importance DESC, accessed_at DESC LIMIT {limit}"
        rows = self._conn.execute(q, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def stats(self, user_id: str = "default") -> dict:
        total = self.count(user_id)
        by_type = {}
        for t in MemoryType:
            by_type[t.value] = self.count(user_id, t)
        oldest = self._conn.execute(
            "SELECT MIN(created_at) FROM memories WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        return {
            "total": total,
            "by_type": by_type,
            "oldest_memory_days": (time.time() - oldest) / 86400 if oldest else 0,
            "db_path": str(self.db_path),
        }

    # ── Internal ──────────────────────────────

    def _row_to_memory(self, row) -> Memory:
        d = dict(row)
        d["type"] = MemoryType(d["type"])
        d["metadata"] = json.loads(d["metadata"])
        d["tags"] = json.loads(d["tags"])
        d["embedding"] = None  # loaded separately on demand
        return Memory(**d)

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
