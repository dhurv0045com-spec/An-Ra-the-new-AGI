"""Replay pipeline for verifier-approved self-improvement examples."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import random
from typing import Iterable


@dataclass
class ReplayRecord:
    prompt: str
    target: str
    source: str
    score: float = 1.0
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ReplayBatch:
    records: list[ReplayRecord]

    def texts(self) -> list[str]:
        return [f"{r.prompt}\n{r.target}" for r in self.records]


class ReplayPipeline:
    """Small durable buffer for STaR, RLVR, and verifier replay examples."""

    def __init__(self, max_size: int = 8192, path: str | Path | None = None) -> None:
        self.max_size = int(max_size)
        self.path = Path(path) if path is not None else None
        self.records: list[ReplayRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def add(
        self,
        prompt: str,
        target: str,
        *,
        source: str,
        score: float = 1.0,
        weight: float = 1.0,
        metadata: dict | None = None,
    ) -> ReplayRecord:
        record = ReplayRecord(
            prompt=str(prompt),
            target=str(target),
            source=str(source),
            score=float(score),
            weight=float(weight),
            metadata=dict(metadata or {}),
        )
        self.records.append(record)
        if len(self.records) > self.max_size:
            self.records = self.records[-self.max_size :]
        return record

    def extend(self, records: Iterable[ReplayRecord | dict]) -> None:
        for record in records:
            if isinstance(record, ReplayRecord):
                self.add(
                    record.prompt,
                    record.target,
                    source=record.source,
                    score=record.score,
                    weight=record.weight,
                    metadata=record.metadata,
                )
            else:
                self.add(
                    record.get("prompt", ""),
                    record.get("target", record.get("chain", record.get("response", ""))),
                    source=record.get("source", "replay"),
                    score=record.get("score", 1.0),
                    weight=record.get("weight", 1.0),
                    metadata=record.get("metadata", {}),
                )

    def add_star_examples(self, examples: Iterable[object]) -> int:
        count = 0
        for ex in examples:
            self.add(
                getattr(ex, "prompt", ""),
                getattr(ex, "chain", getattr(ex, "rationale", "")),
                source=f"star:{getattr(ex, 'source', 'direct')}",
                score=getattr(ex, "score", 1.0),
                weight=getattr(ex, "weight", 1.0),
                metadata={"answer": getattr(ex, "answer", "")},
            )
            count += 1
        return count

    def add_rlvr_step(self, step, min_reward: float = 0.0) -> int:
        count = 0
        prompt = getattr(getattr(step, "task", None), "prompt", "")
        task_id = getattr(getattr(step, "task", None), "task_id", "")
        for i, (completion, reward) in enumerate(zip(step.completions, step.rewards)):
            if float(reward) < min_reward:
                continue
            self.add(
                prompt,
                completion,
                source="rlvr",
                score=float(reward),
                weight=max(0.0, float(reward)),
                metadata={"task_id": task_id, "completion_index": i},
            )
            count += 1
        return count

    def sample(
        self,
        batch_size: int,
        *,
        min_score: float | None = None,
        seed: int | None = None,
    ) -> ReplayBatch:
        pool = self.records
        if min_score is not None:
            pool = [r for r in pool if r.score >= float(min_score)]
        if not pool:
            return ReplayBatch([])

        rng = random.Random(seed)
        k = min(int(batch_size), len(pool))
        weights = [max(1e-6, r.weight) for r in pool]
        return ReplayBatch(rng.choices(pool, weights=weights, k=k))

    def save(self, path: str | Path | None = None) -> Path:
        out = Path(path) if path is not None else self.path
        if out is None:
            raise ValueError("ReplayPipeline.save requires a path.")
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = [json.dumps(asdict(record), ensure_ascii=True) for record in self.records]
        out.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path, max_size: int = 8192) -> "ReplayPipeline":
        pipe = cls(max_size=max_size, path=path)
        p = Path(path)
        if not p.exists():
            return pipe
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            pipe.add(
                data.get("prompt", ""),
                data.get("target", ""),
                source=data.get("source", "replay"),
                score=data.get("score", 1.0),
                weight=data.get("weight", 1.0),
                metadata=data.get("metadata", {}),
            )
        return pipe

    def finetune(
        self,
        model,
        tokenizer,
        optimizer,
        *,
        n_steps: int = 50,
        batch_size: int = 1,
        max_length: int | None = None,
    ) -> list[float]:
        """Language-model replay fine-tuning over sampled prompt-target text."""
        if not self.records:
            return []

        import torch
        import torch.nn.functional as F

        model.train()
        device = next(model.parameters()).device
        block = max_length or getattr(model, "block_size", 512)
        losses: list[float] = []

        for step_idx in range(n_steps):
            batch = self.sample(batch_size, seed=step_idx).records
            step_losses = []
            for record in batch:
                ids = tokenizer.encode(f"{record.prompt}\n{record.target}")[-block:]
                if len(ids) < 2:
                    continue
                x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
                y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                step_losses.append(loss * record.weight)

            if not step_losses:
                continue
            loss = torch.stack(step_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        return losses
