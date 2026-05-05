from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any


@dataclass
class CIVProfile:
    truthfulness: float = 0.8
    safety: float = 0.9
    autonomy: float = 0.7
    coherence: float = 0.8


class ConstitutionalIdentityVector:
    def __init__(self, profile: CIVProfile | None = None) -> None:
        self.profile = profile or CIVProfile()

    def update(self, evidence: dict[str, float], alpha: float = 0.05) -> None:
        alpha = max(0.0, min(1.0, float(alpha)))
        for k, v in evidence.items():
            if hasattr(self.profile, k):
                current = float(getattr(self.profile, k))
                # AN: CIV should slowly learn from scored data instead of staying a static gate.
                updated = current * (1 - alpha) + float(v) * alpha
                setattr(self.profile, k, max(0.0, min(1.0, updated)))

    def score(self, evidence: dict[str, float] | None = None) -> float:
        evidence = evidence or {}
        self.update(evidence)
        vals = asdict(self.profile)
        for k, v in evidence.items():
            if k in vals:
                vals[k] = max(0.0, min(1.0, 0.7 * vals[k] + 0.3 * float(v)))
        return sum(vals.values()) / len(vals)

    def verify(self, min_score: float = 0.7, evidence: dict[str, float] | None = None) -> dict:
        s = self.score(evidence)
        return {"score": s, "passed": s >= min_score}

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self.profile), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ConstitutionalIdentityVector":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(CIVProfile(**data))


class CIVGuard:
    """Representation-level identity drift check for compact evals."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        identity_path: str | Path,
        layer_idx: int = -1,
        threshold: float = 0.92,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.identity_path = Path(identity_path)
        self.layer_idx = layer_idx
        self.threshold = float(threshold)
        self.baseline = None

    def _identity_prompts(self) -> list[str]:
        if not self.identity_path.exists():
            return ["H: Who are you?"]
        prompts = [line.strip() for line in self.identity_path.read_text(encoding="utf-8").splitlines()]
        return [line for line in prompts if line] or ["H: Who are you?"]

    def _prompt_vector(self, prompt: str):
        import torch

        ids = self.tokenizer.encode(prompt)
        block_size = int(getattr(self.model, "block_size", len(ids)) or len(ids))
        ids = ids[-block_size:] or [1]
        device = next(self.model.parameters()).device
        x = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            # AN: CIVGuard tracks identity in hidden space so promotion can catch drift before text style changes.
            if hasattr(self.model, "emb") and hasattr(self.model, "blocks"):
                h = self.model.emb(x)
                blocks = list(self.model.blocks)
                target = self.layer_idx if self.layer_idx >= 0 else len(blocks) + self.layer_idx
                target = max(0, min(target, len(blocks) - 1))
                for i, block in enumerate(blocks):
                    h = block(h)
                    if isinstance(h, tuple):
                        h = h[0]
                    if i == target:
                        return h.mean(dim=(0, 1)).detach().cpu()

            out = self.model(x)
            if isinstance(out, tuple):
                out = out[0]
            return out.mean(dim=(0, 1)).detach().cpu()

    def current_vector(self):
        import torch

        vectors = [self._prompt_vector(prompt) for prompt in self._identity_prompts()]
        return torch.stack(vectors).mean(dim=0)

    def compute_baseline(self):
        self.baseline = self.current_vector()
        return self.baseline

    def verify(self) -> tuple[float, bool]:
        import torch

        if self.baseline is None:
            self.compute_baseline()
        current = self.current_vector()
        score = torch.nn.functional.cosine_similarity(self.baseline, current, dim=0).item()
        return score, score >= self.threshold
