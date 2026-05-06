from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

from anra_paths import DRIVE_FAISS_INDEX, DRIVE_GHOST_DB

try:
    from identity.hal import HALModule
except Exception:
    HALModule = None


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
        hal: HALModule | None = None,
        embedding_model=None,
        embedding_tokenizer=None,
        embedding_fn=None,
    ) -> None:
        self.dim = int(dim)
        self.esv = esv
        self.hal = hal
        self.embedding_model = embedding_model
        self.embedding_tokenizer = embedding_tokenizer
        self.embedding_fn = embedding_fn
        self.short_term: list[dict] = []
        self.graph: dict[str, list[str]] = {}
        self.ghost_db_path = Path(DRIVE_GHOST_DB)
        idx_path = Path(faiss_index_path) if faiss_index_path is not None else Path(DRIVE_FAISS_INDEX)
        from memory.faiss_store import FAISSEpisodicStore

        self.episodic = FAISSEpisodicStore(index_path=idx_path, dim=self.dim)
        self.episodic.load()

    def _fit_dim(self, vector) -> np.ndarray:
        np = _numpy()
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] == self.dim:
            return vec
        if vec.shape[0] > self.dim:
            return vec[: self.dim]
        out = np.zeros(self.dim, dtype=np.float32)
        out[: vec.shape[0]] = vec
        return out

    def _pool_model_output(self, output, attention_mask=None):
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None and isinstance(output, (tuple, list)) and output:
            hidden = output[0]
        if hidden is None:
            return output

        try:
            import torch

            if attention_mask is not None:
                mask = attention_mask.to(hidden.device).unsqueeze(-1).float()
                return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            return hidden.mean(dim=1)
        except Exception:
            return hidden

    def _local_semantic_projection(self, text: str) -> np.ndarray:
        np = _numpy()
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = [tok for tok in text.lower().split() if tok]
        if not tokens:
            return vec
        for pos, tok in enumerate(tokens):
            features = {tok, tok[:4], tok[-4:]}
            features.update(tok[i : i + 3] for i in range(max(1, len(tok) - 2)))
            weight = 1.0 / (1.0 + pos * 0.01)
            for feature in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest, "little") % self.dim
                sign = 1.0 if digest[0] & 1 else -1.0
                vec[idx] += sign * weight
        return vec

    def _semantic_embed(self, text: str) -> np.ndarray:
        np = _numpy()

        if callable(self.embedding_fn):
            return self._fit_dim(self.embedding_fn(text))

        model = self.embedding_model
        if model is not None:
            encode = getattr(model, "encode", None)
            if callable(encode):
                return self._fit_dim(encode(text))

            tokenizer = self.embedding_tokenizer
            if tokenizer is not None:
                try:
                    import torch

                    with torch.no_grad():
                        if callable(tokenizer):
                            batch = tokenizer(text, return_tensors="pt", truncation=True)
                            output = model(**batch)
                            pooled = self._pool_model_output(output, batch.get("attention_mask"))
                        else:
                            ids = tokenizer.encode(text)
                            x = torch.tensor([ids], dtype=torch.long)
                            output = model(x)
                            pooled = self._pool_model_output(output)
                    return self._fit_dim(pooled.detach().cpu().numpy())
                except Exception:
                    pass

        return self._fit_dim(np.tanh(self._local_semantic_projection(text)))

    def write(self, content: str, *, metadata: dict | None = None, tier: str = "episodic") -> MemoryWriteResult:
        metadata = metadata or {}
        record_id = hashlib.sha1(f"{content}|{metadata}".encode("utf-8")).hexdigest()[:16]

        # Threat patterns bypass all thresholds — always write
        is_threat = (metadata or {}).get("kind") == "threat_pattern"

        if tier == "episodic" and not is_threat:
            if self.hal is not None:
                threshold = self.hal.memory_threshold()
                salience = float((metadata or {}).get("salience",
                               (metadata or {}).get("importance", 0.5)))
                if salience < threshold:
                    metadata = {**(metadata or {}),
                                "hal_threshold": threshold,
                                "routed_from": "episodic"}
                    self.short_term.append({"record_id": record_id,
                                            "content": content,
                                            "metadata": metadata})
                    self.short_term = self.short_term[-256:]
                    return MemoryWriteResult(tier="short_term",
                                            record_id=record_id)
            elif self.esv is not None:
                # fallback to original ESV logic unchanged
                threshold_fn = getattr(self.esv, "memory_write_threshold", None)
                if callable(threshold_fn):
                    threshold = float(threshold_fn())
                    salience = (metadata or {}).get("salience",
                              (metadata or {}).get("importance"))
                    if salience is not None and float(salience) < threshold:
                        metadata = {**(metadata or {}),
                                    "esv_threshold": threshold,
                                    "routed_from": "episodic"}
                        self.short_term.append({"record_id": record_id,
                                                "content": content,
                                                "metadata": metadata})
                        self.short_term = self.short_term[-256:]
                        return MemoryWriteResult(tier="short_term",
                                                record_id=record_id)

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

        vec = self._semantic_embed(content)
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
        qvec = query if isinstance(query, np.ndarray) else self._semantic_embed(str(query))
        return self.episodic.search(qvec, k=n)
