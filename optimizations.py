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

    def forward(self, batch_input, num_passes=None):
        if num_passes is None:
            num_passes = self.passes

        self.entropy_profile = []
        running_logits = None

        for pass_idx in range(num_passes):
            pass_tensor = torch.tensor(pass_idx, device=batch_input.device, dtype=torch.long)
            logits = checkpoint.checkpoint(
                self._compute_pass,
                batch_input,
                pass_tensor,
                use_reentrant=False
            )

            if running_logits is None:
                running_logits = logits.clone()
            else:
                alpha = 1.0 / (pass_idx + 1)
                running_logits = (1.0 - alpha) * running_logits + alpha * logits

        return running_logits

    def get_config(self):
        return {"passes": self.passes, "entropy_profile": list(self.entropy_profile)}

    def _compute_pass(self, batch_input, pass_idx):
        pass_idx_int = int(pass_idx.item()) if torch.is_tensor(pass_idx) else int(pass_idx)
        logits, _ = self.model(batch_input)
        temperature = 1.0 + 0.2 * pass_idx_int
        probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
        entropy = -torch.sum(probs * torch.log(torch.clamp(probs, 1e-9, 1.0)), dim=-1)
        self.entropy_profile.append(float(entropy.mean().item()))
        return logits
