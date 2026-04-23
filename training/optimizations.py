import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
import re
from typing import Tuple


class AdaptiveScheduler:
    def __init__(self, base_lr, warmup_steps, total_steps):
        self.base_lr = base_lr
        self.initial_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.loss_history = []
        self.lr_adjustments = 0

    def get_lr(self, step, current_loss=None):
        if step < self.warmup_steps:
            return self.base_lr * (step / max(self.warmup_steps, 1))

        if current_loss is not None:
            self.loss_history.append(current_loss)
            if len(self.loss_history) > 10:
                variance = np.var(self.loss_history[-10:])
                if variance < 0.0001 and self.lr_adjustments < 3:
                    self.base_lr *= 0.7
                    self.lr_adjustments += 1

        denom = max(self.total_steps - self.warmup_steps, 1)
        progress = (step - self.warmup_steps) / denom
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + np.cos(np.pi * progress))
        return self.initial_lr * (0.1 + 0.9 * cosine)


class MultiScaleHardSampleDetector:
    def __init__(self):
        self.patterns = {
            'math': r'\d+\s*[-+*/=^%<>]+\s*\d+',
            'logic': r'\b(if|then|therefore|prove|implies)\b',
            'code': r'\b(def|for|while|class|import)\b',
            'question': r'\b(why|how|explain|analyze)\b',
            'recursive': r'\b(recursive|loop|iterate)\b',
        }

    def detect(self, batch_text: str) -> Tuple[bool, int]:
        samples = [s.strip() for s in batch_text.split('\n') if s.strip()]
        if not samples:
            return False, 0

        scores = {p: 0 for p in self.patterns}
        for sample in samples:
            for pname, pat in self.patterns.items():
                if re.search(pat, sample, re.IGNORECASE):
                    scores[pname] += 1

        total = sum(scores.values())
        ratio = total / (len(samples) * len(self.patterns))

        if scores['recursive'] > 0:
            difficulty = 3 if ratio > 0.40 else 2
        elif ratio > 0.50:
            difficulty = 3
        elif ratio > 0.30:
            difficulty = 2
        elif ratio > 0.15:
            difficulty = 1
        else:
            difficulty = 0

        return difficulty > 0, difficulty


class GradientCheckpointedOuroboros(nn.Module):
    def __init__(self, model, passes=3):
        super().__init__()
        self.model = model
        self.passes = passes
        self.entropy_profile = []

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run N passes and return logits/loss tuple."""
        total_loss = torch.tensor(0.0, device=x.device)
        self.entropy_profile = []
        logits = None
        weights = [1.0, 0.8, 1.2]

        for pass_idx in range(self.passes):
            if pass_idx > 0 and self.training:
                from typing import cast
                logits, loss = cast(Tuple[torch.Tensor, torch.Tensor], checkpoint.checkpoint(
                    self._forward_model, x, targets, use_reentrant=False
                ))
            else:
                logits, loss = self.model(x, targets)

            if loss is not None:
                total_loss = total_loss + loss * weights[min(pass_idx, 2)]

            probs = torch.softmax(logits[:, -1, :], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(-1).mean().item()
            self.entropy_profile.append(float(entropy))

        avg_loss = total_loss / max(self.passes, 1)
        from typing import cast
        return cast(torch.Tensor, logits), cast(torch.Tensor, avg_loss)

    def get_config(self):
        return {"passes": self.passes, "entropy_profile": list(self.entropy_profile)}

    def _forward_model(self, x, targets):
        return self.model(x, targets)
