"""Reference Connector deliberation wrapper providing bounded thought policy.

This layer sits outside the Core Executor, owning sampling, candidate generation,
repetition penalties, and scoring heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .errors import ContextOverflowError
from .executor import CoreExecutor
from .generate import generate
from .model import AnRaCore
from .tokenizer import V4Tokenizer


def _has_invalid_terminal(response: str) -> bool:
    """Reject the only known malformed text sentinel, not every response.

    The former ``endswith((\"\", \"<unk>\"))`` check always matched because every
    string ends with the empty string.  This helper keeps the compatibility
    heuristic narrow and independently testable.
    """

    return response.rstrip().endswith("<unk>")


@dataclass(frozen=True, slots=True)
class ThoughtPolicy:
    """Bounded inference policy; it never grants tools or mutates the core model."""

    mode: str = "direct"
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 0.92
    candidates: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {"direct", "deliberate"}:
            raise ValueError("mode must be direct or deliberate")
        if not 1 <= self.max_new_tokens <= 512:
            raise ValueError("max_new_tokens must be between 1 and 512")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if not 1 <= self.candidates <= 4:
            raise ValueError("candidates must be between 1 and 4")
        if self.mode == "direct" and self.candidates != 1:
            raise ValueError("direct mode uses exactly one candidate")


@dataclass(frozen=True, slots=True)
class Thought:
    text: str
    mode: str
    self_likelihood: float
    candidates_considered: int
    prompt_tokens: int
    generated_tokens: int
    checkpoint_step: int | None


class Brain:
    """Reference Connector interface providing bounded thought generation over Core."""

    def __init__(
        self,
        model_or_executor: AnRaCore | CoreExecutor,
        tokenizer: V4Tokenizer,
        *,
        checkpoint_step: int | None = None,
    ) -> None:
        if isinstance(model_or_executor, CoreExecutor):
            self.executor = model_or_executor
            self.model = self.executor.model
        else:
            self.model = model_or_executor.eval()
            self.executor = CoreExecutor(self.model, tokenizer=tokenizer)
        self.tokenizer = tokenizer
        self.checkpoint_step = checkpoint_step

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        tokenizer_path: str | Path,
        *,
        device: str = "cpu",
    ) -> Brain:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        executor = CoreExecutor.from_checkpoint(
            checkpoint,
            tokenizer_path=tokenizer_path,
            device=device,
        )
        tokenizer = executor.tokenizer
        if tokenizer is None:  # Defensive: strict from_checkpoint always binds one.
            raise RuntimeError("Core executor did not bind a tokenizer")
        identity = executor.checkpoint_identity
        return cls(executor, tokenizer, checkpoint_step=identity.global_step)

    def describe(self) -> dict[str, object]:
        return {
            "layer": "connector_reference",
            "core": self.executor.describe(),
            "role": "bounded thought policy",
            "checkpoint_step": self.checkpoint_step,
        }

    @torch.inference_mode()
    def _score(self, prompt_ids: list[int], response: str) -> float:
        response_ids = self.tokenizer.encode(response)
        if not response_ids:
            return float("-inf")
        prefix = [self.tokenizer.bos_token_id, *prompt_ids]
        combined = prefix + response_ids
        if len(combined) > self.model.config.block_size:
            raise ContextOverflowError(
                "Candidate scoring exceeds the Core context capacity",
                details={"tokens": len(combined), "capacity": self.model.config.block_size},
            )
        prefix_length = max(1, len(combined) - len(response_ids))
        token_ids = torch.tensor(
            [combined], dtype=torch.long, device=self.executor.device
        )
        logits = self.model(token_ids[:, :-1])
        targets = token_ids[:, 1:]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        selected = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        start = max(0, prefix_length - 1)
        likelihood = float(selected[:, start:].mean().item())
        normalized = [token for token in response.lower().split() if token]
        if len(normalized) >= 6 and len(set(normalized)) / len(normalized) < 0.45:
            likelihood -= 2.0
        if _has_invalid_terminal(response):
            likelihood -= 1.0
        return likelihood

    def think(self, prompt: str, policy: ThoughtPolicy = ThoughtPolicy()) -> Thought:
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            raise ValueError("prompt produced no tokens")
        count = policy.candidates if policy.mode == "deliberate" else 1
        candidates: list[tuple[float, str]] = []
        for index in range(count):
            text = generate(
                self.executor,
                self.tokenizer,
                prompt,
                max_new_tokens=policy.max_new_tokens,
                temperature=policy.temperature,
                top_p=policy.top_p,
                seed=policy.seed + index,
            )
            candidates.append((self._score(prompt_ids, text), text))
        score, response = max(candidates, key=lambda item: item[0])
        self_likelihood = (
            0.0
            if score == float("-inf")
            else max(0.0, min(1.0, torch.exp(torch.tensor(score)).item()))
        )
        return Thought(
            text=response,
            mode=policy.mode,
            self_likelihood=float(self_likelihood),
            candidates_considered=count,
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(self.tokenizer.encode(response)),
            checkpoint_step=self.checkpoint_step,
        )
