from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

from anra_paths import DRIVE_FAISS_INDEX, DRIVE_GHOST_DB


def _numpy():
    import numpy as np

    return np


@dataclass
class MemoryWriteResult:
    tier: str
    record_id: str


class MemoryRouter:
    """Unified interface over memory tiers.

    Tiers: episodic (faiss), ghost metadata, short-term cache, and graph placeholders.
    """

    def __init__(
        self,
        dim: int = 256,
        faiss_index_path: str | Path | None = None,
        esv=None,
    ) -> None:
        self.dim = int(dim)
        self.esv = esv
        self.short_term: list[dict] = []
        self.graph: dict[str, list[str]] = {}
        self.ghost_db_path = Path(DRIVE_GHOST_DB)
        idx_path = Path(faiss_index_path) if faiss_index_path is not None else Path(DRIVE_FAISS_INDEX)
        from memory.faiss_store import FAISSEpisodicStore

        self.episodic = FAISSEpisodicStore(index_path=idx_path, dim=self.dim)
        self.episodic.load()

    def _hash_embed(self, text: str) -> np.ndarray:
        np = _numpy()
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = text.lower().split()
        for tok in tokens:
            h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        return vec

    def write(self, content: str, *, metadata: dict | None = None, tier: str = "episodic") -> MemoryWriteResult:
        metadata = metadata or {}
        record_id = hashlib.sha1(f"{content}|{metadata}".encode("utf-8")).hexdigest()[:16]

        if tier == "episodic" and self.esv is not None:
            threshold_fn = getattr(self.esv, "memory_write_threshold", None)
            if callable(threshold_fn):
                threshold = float(threshold_fn())
                salience = metadata.get("salience", metadata.get("importance"))
                if salience is not None and float(salience) < threshold:
                    metadata = {**metadata, "esv_threshold": threshold, "routed_from": "episodic"}
                    self.short_term.append({"record_id": record_id, "content": content, "metadata": metadata})
                    self.short_term = self.short_term[-256:]
                    return MemoryWriteResult(tier="short_term", record_id=record_id)

        if tier == "short_term":
            self.short_term.append({"record_id": record_id, "content": content, "metadata": metadata})
            self.short_term = self.short_term[-256:]
            return MemoryWriteResult(tier=tier, record_id=record_id)

        if tier == "graph":
            src = str(metadata.get("src", "root"))
            dst = str(metadata.get("dst", content[:64]))
            self.graph.setdefault(src, []).append(dst)
            return MemoryWriteResult(tier=tier, record_id=record_id)

        if tier == "ghost":
            self.ghost_db_path.parent.mkdir(parents=True, exist_ok=True)
            row = {"record_id": record_id, "content": content, "metadata": metadata}
            with self.ghost_db_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            return MemoryWriteResult(tier=tier, record_id=record_id)

        vec = self._hash_embed(content)
        payload = {"content": content, **metadata}
        self.episodic.add(record_id, vec, payload)
        self.episodic.save()
        return MemoryWriteResult(tier="episodic", record_id=record_id)

    def read(self, query: str | np.ndarray, n: int = 8, *, tier: str = "episodic") -> list[dict]:
        if tier == "short_term":
            q = str(query).lower()
            hits = [x for x in reversed(self.short_term) if q in x.get("content", "").lower()]
            return hits[:n]

        if tier == "graph":
            key = str(query)
            return [{"src": key, "dst": dst} for dst in self.graph.get(key, [])[:n]]

        if tier == "ghost":
            if not self.ghost_db_path.exists():
                return []
            q = str(query).lower()
            rows = []
            for line in self.ghost_db_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    if q in str(row.get("content", "")).lower():
                        rows.append(row)
                except Exception:
                    continue
            return rows[-n:]

        np = _numpy()
        qvec = query if isinstance(query, np.ndarray) else self._hash_embed(str(query))
        return self.episodic.search(qvec, k=n)
