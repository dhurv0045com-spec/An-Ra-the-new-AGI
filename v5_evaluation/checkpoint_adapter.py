"""Checkpoint-backed V5 evaluation adapter: the real raw-Core model path.

Loads a checkpoint's model payload into a freshly constructed, read-only V5
core bound to the exact frozen tokenizer, and exposes the three adapter
calls: candidate-suffix scoring (summed suffix log-probability, shared prefix
tokenization, uniform EOS), greedy free generation (EOS-or-cap stop), and
constrained generation.  It accepts no evaluator truth, contains no
task-family logic, and never mutates the checkpoint or the loaded weights.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from typing import Any

from v5_contracts.model_spec import ModelSpec
from v5_model.core import initialize, packed_layout
from v5_tokenizer.adapter import FrozenTokenizer


ADAPTER_SCHEMA = "anra-v5-checkpoint-adapter/v1"
SCORING_RULE = "summed candidate-suffix log-probability, shared prefix, uniform EOS"
DECODING_RULE = "greedy; temperature 0; stop on EOS or cap"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


@dataclass(frozen=True, slots=True)
class AdapterIdentity:
    schema: str
    checkpoint_sha256: str
    model_payload_sha256: str
    model_spec_sha256: str
    tokenizer_artifact_sha256: str
    scoring_rule: str
    decoding_rule: str

    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json({
            "schema": self.schema,
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_payload_sha256": self.model_payload_sha256,
            "model_spec_sha256": self.model_spec_sha256,
            "tokenizer_artifact_sha256": self.tokenizer_artifact_sha256,
            "scoring_rule": self.scoring_rule,
            "decoding_rule": self.decoding_rule,
        })).hexdigest()


class CheckpointBackedV5Adapter:
    """Immutable raw-Core adapter over one identified checkpoint."""

    def __init__(
        self,
        *,
        checkpoint_sha256: str,
        model_payload: bytes,
        model_spec: ModelSpec,
        tokenizer: FrozenTokenizer,
        seed: int = 0,
        device: Any | None = None,
        torch_module: Any = None,
    ) -> None:
        if torch_module is None:
            import torch as torch_module
        if len(checkpoint_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in checkpoint_sha256
        ):
            raise ValueError("checkpoint identity must be a lowercase SHA-256")
        self.torch = torch_module
        self.spec = model_spec
        self.tokenizer = tokenizer
        self.device = device
        specials = tokenizer.identity.special_token_ids
        self.bos_id = int(specials["bos"])
        self.eos_id = int(specials["eos"])
        self.pad_id = int(specials["pad"])
        model = initialize(model_spec, seed=seed, torch_module=torch_module)
        state_dict = torch_module.load(
            io.BytesIO(model_payload), map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict)
        if device is not None:
            model = model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model
        self.identity = AdapterIdentity(
            schema=ADAPTER_SCHEMA,
            checkpoint_sha256=checkpoint_sha256,
            model_payload_sha256=hashlib.sha256(model_payload).hexdigest(),
            model_spec_sha256=model_spec.sha256(),
            tokenizer_artifact_sha256=tokenizer.identity.artifact_sha256,
            scoring_rule=SCORING_RULE,
            decoding_rule=DECODING_RULE,
        )

    # -- shared tensor plumbing -------------------------------------------
    def _logits(self, token_ids: list[int]) -> Any:
        torch = self.torch
        if not 1 < len(token_ids) <= self.spec.context_length:
            raise ValueError("tokenized input must fill [2, context_length]")
        tokens = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        # one visible sequence: a single segment under the canonical packed
        # layout, giving exactly the causal + per-segment semantics of training
        segment_ids = torch.zeros_like(tokens, dtype=torch.int32)
        positions, mask = packed_layout(segment_ids, torch_module=torch)
        mask = mask.to(tokens.device)
        with torch.no_grad():
            logits = self.model(tokens, positions, mask)
        return logits[0].float()

    def _suffix_ids(self, prefix_text: str, candidate: str) -> tuple[list[int], list[int]]:
        """Tokenize with a verified prefix property; fail closed on drift."""

        prefix_ids = self.tokenizer.encode(prefix_text)
        full_ids = self.tokenizer.encode(prefix_text + candidate)
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                "candidate tokenization does not extend the prefix; scoring rule would "
                "contaminate the prompt likelihood"
            )
        return prefix_ids, full_ids[len(prefix_ids):]

    # -- the three adapter calls -------------------------------------------
    def score_candidates(self, context: str, query: str, candidates: list[str]) -> list[float]:
        """Sum candidate-suffix token log-probabilities only."""

        if not candidates:
            raise ValueError("candidate sets cannot be empty")
        prefix_text = context + query
        scores: list[float] = []
        for candidate in candidates:
            prefix_ids, suffix_ids = self._suffix_ids(prefix_text, candidate)
            suffix_ids = [*suffix_ids, self.eos_id]
            token_ids = [self.bos_id, *prefix_ids, *suffix_ids]
            if len(token_ids) > self.spec.context_length:
                raise ValueError("candidate context exceeds the native context window")
            logits = self._logits(token_ids[:-1])
            log_probs = self.torch.log_softmax(logits, dim=-1)
            targets = token_ids[1:]
            first_suffix_target = len(prefix_ids)  # target index of the first suffix token
            score = 0.0
            for index in range(first_suffix_target, len(targets)):
                score += float(log_probs[index, targets[index]].item())
            scores.append(score)
        return scores

    def generate_free(self, prompt: str, max_new_tokens: int = 64) -> str:
        """Greedy free generation; stop on EOS or the token cap."""

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        prompt_ids = self.tokenizer.encode(prompt)
        token_ids = [self.bos_id, *prompt_ids]
        generated: list[int] = []
        for _ in range(max_new_tokens):
            if len(token_ids) >= self.spec.context_length:
                break
            logits = self._logits(token_ids)
            next_id = int(self.torch.argmax(logits[-1]).item())
            if next_id == self.eos_id:
                break
            token_ids.append(next_id)
            generated.append(next_id)
        return self.tokenizer.decode(generated)

    def generate_constrained(self, prompt: str, candidates: list[str]) -> str:
        """Greedy constrained generation over scored candidates."""

        scores = self.score_candidates(prompt, "", candidates)
        best = max(range(len(candidates)), key=lambda index: scores[index])
        return candidates[best]


__all__ = [
    "ADAPTER_SCHEMA",
    "AdapterIdentity",
    "CheckpointBackedV5Adapter",
    "SCORING_RULE",
]
