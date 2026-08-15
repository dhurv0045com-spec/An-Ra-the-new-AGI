from __future__ import annotations

import hashlib
import json

# Legacy model/tokenizer adapters intentionally accept heterogeneous objects;
# NumPy stays lazy so lightweight runtime imports do not load native backends.
# ruff: noqa: ANN001, ANN202, E501, F821
from dataclasses import dataclass
from pathlib import Path

from anra.anra_paths import DRIVE_FAISS_INDEX, DRIVE_MEMORY_JOURNAL
from engine.metric_bus import instrument
from engine.telemetry import trace
from retrieval.adapters import BM25RetrieverAdapter, VectorRetrieverAdapter
from retrieval.hybrid import HybridRetriever
from retrieval.protocols import RetrievalQuery

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

    Tiers: episodic (FAISS), durable journal, short-term cache, and graph placeholders.
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
        self.journal_path = Path(DRIVE_MEMORY_JOURNAL)
        idx_path = Path(faiss_index_path) if faiss_index_path is not None else Path(DRIVE_FAISS_INDEX)
        from anra.memory.bm25 import BM25MemoryTier

        from memory.faiss_store import FAISSEpisodicStore

        self.episodic = FAISSEpisodicStore(index_path=idx_path, dim=self.dim)
        self.episodic.load()
        self.bm25 = BM25MemoryTier()
        self.retrieval = HybridRetriever(
            (
                VectorRetrieverAdapter(self.episodic, self._semantic_embed),
                BM25RetrieverAdapter(self.bm25),
            )
        )

    @staticmethod
    def _record_memory_event(
        *,
        kind: str,
        trace_id: str | None,
        inputs: dict[str, object],
        output: object,
        metadata: dict[str, object] | None = None,
    ) -> None:
        try:
            from runtime.experience_ledger import record_experience

            record_experience(
                trace_id=trace_id,
                kind=kind,
                inputs=inputs,
                output=output,
                gate_record={"allowed": True, "gate": "memory_policy"},
                source="memory.router",
                metadata=metadata or {},
            )
        except Exception:
            pass

    def _finish_write(
        self,
        result: MemoryWriteResult,
        *,
        content: str,
        requested_tier: str,
        trace_id: str | None,
        metadata: dict[str, object],
    ) -> MemoryWriteResult:
        self._record_memory_event(
            kind="memory_write",
            trace_id=trace_id,
            inputs={
                "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                "requested_tier": requested_tier,
            },
            output={"record_id": result.record_id, "tier": result.tier},
            metadata={
                "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                "session_id": metadata.get("session_id"),
                "memory_type": metadata.get("type", metadata.get("kind")),
            },
        )
        return result

    def _finish_read(
        self,
        rows: list[dict],
        *,
        query: object,
        tier: str,
        trace_id: str | None,
    ) -> list[dict]:
        self._record_memory_event(
            kind="memory_recall",
            trace_id=trace_id,
            inputs={"query": str(query), "tier": tier},
            output={
                "record_ids": [
                    str(row.get("record_id", row.get("id", ""))) for row in rows
                ],
                "hit_count": len(rows),
            },
        )
        return rows

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

    @trace("memory_router", "write")
    @instrument("memory")
    def write(
        self,
        content: str,
        *,
        metadata: dict | None = None,
        tier: str = "episodic",
        trace_id: str | None = None,
    ) -> MemoryWriteResult:
        metadata = metadata or {}
        trace_id = trace_id or str(metadata.get("trace_id") or "") or None
        record_id = hashlib.sha1(f"{content}|{metadata}".encode()).hexdigest()[:16]

        def finish(result: MemoryWriteResult) -> MemoryWriteResult:
            return self._finish_write(
                result,
                content=content,
                requested_tier=tier,
                trace_id=trace_id,
                metadata=metadata,
            )

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
                    return finish(MemoryWriteResult(tier="short_term", record_id=record_id))
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
                        return finish(
                            MemoryWriteResult(tier="short_term", record_id=record_id)
                        )

        if tier == "short_term":
            self.short_term.append({"record_id": record_id, "content": content, "metadata": metadata})
            self.short_term = self.short_term[-256:]
            return finish(MemoryWriteResult(tier=tier, record_id=record_id))

        if tier == "graph":
            src = str(metadata.get("src", "root"))
            dst = str(metadata.get("dst", content[:64]))
            self.graph.setdefault(src, []).append(dst)
            return finish(MemoryWriteResult(tier=tier, record_id=record_id))

        if tier == "journal":
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            row = {"record_id": record_id, "content": content, "metadata": metadata}
            with self.journal_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            return finish(MemoryWriteResult(tier=tier, record_id=record_id))

        vec = self._semantic_embed(content)
        payload = {"content": content, **metadata}
        self.episodic.add(record_id, vec, payload)
        self.bm25.write(content, {**metadata, "canonical_id": record_id})
        self.episodic.save()
        return finish(MemoryWriteResult(tier="episodic", record_id=record_id))

    @trace("memory_router", "read")
    @instrument("memory")
    def read(
        self,
        query: str | np.ndarray,
        n: int = 8,
        *,
        tier: str = "episodic",
        trace_id: str | None = None,
    ) -> list[dict]:
        def finish(rows: list[dict]) -> list[dict]:
            return self._finish_read(rows, query=query, tier=tier, trace_id=trace_id)

        if tier == "short_term":
            q = str(query).lower()
            hits = [x for x in reversed(self.short_term) if q in x.get("content", "").lower()]
            return finish(hits[:n])

        if tier == "graph":
            key = str(query)
            return finish([{"src": key, "dst": dst} for dst in self.graph.get(key, [])[:n]])

        if tier == "journal":
            if not self.journal_path.exists():
                return finish([])
            q = str(query).lower()
            rows = []
            for line in self.journal_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    if q in str(row.get("content", "")).lower():
                        rows.append(row)
                except Exception:
                    continue
            return finish(rows[-n:])

        if tier == "bm25":
            return finish([
                {
                    "record_id": record.metadata.get("canonical_id", record.id),
                    "score": float(record.score),
                    "payload": {"content": record.text, **record.metadata},
                    "retriever": "bm25",
                }
                for record in self.bm25.read(str(query), n=n)
            ])

        if tier == "hybrid":
            np = _numpy()
            vector = query if isinstance(query, np.ndarray) else None
            text = "" if vector is not None else str(query)
            hits = self.retrieval.search(
                RetrievalQuery(text=text, vector=vector, limit=n, trace_id=trace_id)
            )
            return finish([
                {
                    "record_id": hit.id,
                    "score": hit.score,
                    "payload": {"content": hit.text, **hit.metadata},
                    "retrievers": [item.retriever for item in hit.provenance],
                    "provenance": [
                        {
                            "tier": item.retriever,
                            "rank": item.rank,
                            "raw_score": item.raw_score,
                            "weight": item.weight,
                            "source_id": hit.id,
                        }
                        for item in hit.provenance
                    ],
                }
                for hit in hits
            ])

        np = _numpy()
        qvec = query if isinstance(query, np.ndarray) else self._semantic_embed(str(query))
        return finish(self.episodic.search(qvec, k=n))

    @trace("memory_router", "forget")
    @instrument("memory")
    def forget(
        self,
        record_id: str,
        *,
        tier: str = "episodic",
        trace_id: str | None = None,
    ) -> bool:
        deleted = False

        if tier == "short_term":
            original_len = len(self.short_term)
            self.short_term = [
                row for row in self.short_term if str(row.get("record_id", "")) != record_id
            ]
            deleted = len(self.short_term) != original_len
        elif tier == "episodic":
            deleted = self.episodic.delete(record_id)
            deleted = self.bm25.delete_canonical(record_id) or deleted
        elif tier == "journal":
            if self.journal_path.exists():
                kept: list[str] = []
                for line in self.journal_path.read_text(encoding="utf-8").splitlines():
                    try:
                        row = json.loads(line)
                    except Exception:
                        kept.append(line)
                        continue
                    if str(row.get("record_id", "")) == record_id:
                        deleted = True
                    else:
                        kept.append(line)
                if deleted:
                    self.journal_path.write_text(
                        ("\n".join(kept) + "\n") if kept else "",
                        encoding="utf-8",
                    )
        elif tier == "graph":
            deleted = False

        self._record_memory_event(
            kind="memory_forget",
            trace_id=trace_id,
            inputs={"record_id": record_id, "tier": tier},
            output={"deleted": deleted},
        )
        return deleted

    @trace("memory_router", "edit")
    @instrument("memory")
    def edit(
        self,
        record_id: str,
        content: str,
        *,
        metadata: dict | None = None,
        tier: str = "episodic",
        trace_id: str | None = None,
    ) -> MemoryWriteResult | None:
        metadata = metadata or {}
        deleted = self.forget(record_id, tier=tier, trace_id=trace_id)
        if not deleted:
            self._record_memory_event(
                kind="memory_edit",
                trace_id=trace_id,
                inputs={
                    "record_id": record_id,
                    "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                    "tier": tier,
                },
                output={"updated": False, "replacement_record_id": None},
                metadata={
                    "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                    "reason": "record_not_found",
                },
            )
            return None

        replacement = self.write(
            content,
            metadata={**metadata, "replaces_record_id": record_id},
            tier=tier,
            trace_id=trace_id,
        )
        self._record_memory_event(
            kind="memory_edit",
            trace_id=trace_id,
            inputs={
                "record_id": record_id,
                "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                "tier": tier,
            },
            output={"updated": True, "replacement_record_id": replacement.record_id},
            metadata={"content_hash": hashlib.sha256(content.encode()).hexdigest()},
        )
        return replacement
