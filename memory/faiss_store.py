from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


@dataclass
class EpisodicRecord:
    record_id: str
    vector: np.ndarray
    payload: dict


class FAISSEpisodicStore:
    def __init__(self, index_path: str | Path, dim: int) -> None:
        self.index_path = Path(index_path)
        self.dim = int(dim)
        self._records: list[EpisodicRecord] = []
        self._faiss = None
        self._index = None
        self._load_backend()

    def _load_backend(self) -> None:
        try:
            import faiss  # type: ignore

            self._faiss = faiss
            self._index = faiss.IndexFlatIP(self.dim)
        except Exception:
            self._faiss = None
            self._index = None

    def _norm(self, v: np.ndarray) -> np.ndarray:
        vv = np.asarray(v, dtype=np.float32).reshape(-1)
        if vv.shape[0] != self.dim:
            raise ValueError(f"Embedding dim mismatch: expected {self.dim}, got {vv.shape[0]}")
        n = np.linalg.norm(vv)
        return vv if n == 0 else vv / n

    def add(self, record_id: str, vector: np.ndarray, payload: dict) -> None:
        vec = self._norm(vector)
        self._records.append(EpisodicRecord(record_id=record_id, vector=vec, payload=dict(payload)))
        if self._index is not None:
            self._index.add(vec.reshape(1, -1))

    def search(self, query: np.ndarray, k: int = 8) -> list[dict]:
        if not self._records:
            return []
        q = self._norm(query)
        k = max(1, min(int(k), len(self._records)))

        if self._index is not None:
            sims, idxs = self._index.search(q.reshape(1, -1), k)
            out = []
            for score, idx in zip(sims[0], idxs[0]):
                if idx < 0:
                    continue
                r = self._records[int(idx)]
                out.append({"record_id": r.record_id, "score": float(score), "payload": r.payload})
            return out

        mat = np.stack([r.vector for r in self._records], axis=0)
        sims = mat @ q
        order = np.argsort(-sims)[:k]
        return [
            {"record_id": self._records[i].record_id, "score": float(sims[i]), "payload": self._records[i].payload}
            for i in order
        ]

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "dim": self.dim,
            "records": [
                {"record_id": r.record_id, "vector": r.vector.tolist(), "payload": r.payload}
                for r in self._records
            ],
        }
        self.index_path.with_suffix(self.index_path.suffix + ".json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        if self._faiss is not None and self._index is not None:
            self._faiss.write_index(self._index, str(self.index_path))

    def load(self) -> None:
        meta_path = self.index_path.with_suffix(self.index_path.suffix + ".json")
        if not meta_path.exists():
            return
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        self.dim = int(data.get("dim", self.dim))
        self._records = [
            EpisodicRecord(
                record_id=str(r["record_id"]),
                vector=self._norm(np.array(r["vector"], dtype=np.float32)),
                payload=dict(r.get("payload", {})),
            )
            for r in data.get("records", [])
        ]
        if self._faiss is not None and self.index_path.exists():
            try:
                self._index = self._faiss.read_index(str(self.index_path))
            except Exception:
                self._index = self._faiss.IndexFlatIP(self.dim)
                if self._records:
                    self._index.add(np.stack([r.vector for r in self._records], axis=0))
