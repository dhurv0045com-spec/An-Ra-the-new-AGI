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
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Any

from v5_contracts.model_spec import ModelSpec
from v5_model.core import initialize, packed_layout
from v5_evaluation.adapter import GenerationResult
from v5_tokenizer.adapter import FrozenTokenizer


ADAPTER_SCHEMA = "anra-v5-checkpoint-adapter/v1"
SCORING_RULE = "summed candidate-suffix log-probability, shared prefix, uniform EOS"
DECODING_RULE = "greedy; temperature 0; stop on EOS, token cap, or context limit; record stop cause"
# Stable mechanism identity for protocol binding. This names HOW scores are
# computed; it never selects a production decision policy (which stays NULL).
SCORING_CONTRACT_ID = "anra-v5-scoring-contract/summed-suffix-logprob-v1"
SCORING_CONTRACT_SHA256 = hashlib.sha256(SCORING_CONTRACT_ID.encode("utf-8")).hexdigest()
ADAPTER_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
CALLER_ASSERTED_CHECKPOINT_BINDING = "caller-asserted-model-payload/v1"
CHECKPOINT_STORE_BINDING = "v5-content-addressed-checkpoint-store/v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


@dataclass(frozen=True, slots=True)
class AdapterIdentity:
    schema: str
    checkpoint_sha256: str
    model_payload_sha256: str
    parameter_sha256: str
    model_spec_sha256: str
    tokenizer_artifact_sha256: str
    scoring_rule: str
    decoding_rule: str
    implementation_sha256: str
    checkpoint_binding_schema: str = CALLER_ASSERTED_CHECKPOINT_BINDING
    source_tree_sha256: str | None = None

    def sha256(self) -> str:
        value = {
            "schema": self.schema,
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_payload_sha256": self.model_payload_sha256,
            "parameter_sha256": self.parameter_sha256,
            "model_spec_sha256": self.model_spec_sha256,
            "tokenizer_artifact_sha256": self.tokenizer_artifact_sha256,
            "scoring_rule": self.scoring_rule,
            "decoding_rule": self.decoding_rule,
            "implementation_sha256": self.implementation_sha256,
            "checkpoint_binding_schema": self.checkpoint_binding_schema,
        }
        if self.source_tree_sha256 is not None:
            value["source_tree_sha256"] = self.source_tree_sha256
        return hashlib.sha256(_canonical_json(value)).hexdigest()


class CheckpointBackedV5Adapter:
    """Immutable raw-Core adapter over one identified checkpoint."""

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False) and name not in {"last_generation_result"}:
            raise AttributeError(f"checkpoint adapter field is immutable: {name}")
        object.__setattr__(self, name, value)

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
        self.last_generation_result: GenerationResult | None = None
        self._verified_checkpoint_sha256: str | None = None
        self._verified_training_state_sha256: str | None = None
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
        self.scoring_contract_id = SCORING_CONTRACT_ID
        self.scoring_contract_sha256 = SCORING_CONTRACT_SHA256
        parameter_digest = hashlib.sha256()
        for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
            data = parameter.detach().to("cpu", dtype=torch_module.float32).contiguous().numpy()
            parameter_digest.update(name.encode("utf-8") + b"\0")
            parameter_digest.update(data.tobytes())
        self.identity = AdapterIdentity(
            schema=ADAPTER_SCHEMA,
            checkpoint_sha256=checkpoint_sha256,
            model_payload_sha256=hashlib.sha256(model_payload).hexdigest(),
            parameter_sha256=parameter_digest.hexdigest(),
            model_spec_sha256=model_spec.sha256(),
            tokenizer_artifact_sha256=tokenizer.identity.artifact_sha256,
            scoring_rule=SCORING_RULE,
            decoding_rule=DECODING_RULE,
            implementation_sha256=ADAPTER_IMPLEMENTATION_SHA256,
        )
        self._sealed = True

    @classmethod
    def from_checkpoint_store(
        cls,
        *,
        checkpoint_store: Any,
        checkpoint_sha256: str,
        model_spec: ModelSpec,
        tokenizer: FrozenTokenizer,
        seed: int = 0,
        device: Any | None = None,
        torch_module: Any = None,
    ) -> "CheckpointBackedV5Adapter":
        """Load a model only after verifying its complete checkpoint transaction."""

        state, payloads = checkpoint_store.restore(checkpoint_sha256)
        identities = state.identities
        tokenizer.identity.assert_valid()
        if identities.model_spec_sha256 != model_spec.sha256():
            raise ValueError("checkpoint training state names a different ModelSpec")
        if identities.tokenizer_sha256 != tokenizer.identity.artifact_sha256:
            raise ValueError("checkpoint training state names a different tokenizer artifact")
        model_payload = payloads.get("model.bin")
        if not isinstance(model_payload, bytes):
            raise ValueError("verified checkpoint does not contain model.bin")
        adapter = cls(
            checkpoint_sha256=checkpoint_sha256,
            model_payload=model_payload,
            model_spec=model_spec,
            tokenizer=tokenizer,
            seed=seed,
            device=device,
            torch_module=torch_module,
        )
        object.__setattr__(
            adapter,
            "identity",
            replace(
                adapter.identity,
                checkpoint_binding_schema=CHECKPOINT_STORE_BINDING,
                source_tree_sha256=identities.source_tree_sha256,
            ),
        )
        object.__setattr__(adapter, "_verified_checkpoint_sha256", checkpoint_sha256)
        object.__setattr__(adapter, "_verified_training_state_sha256", state.sha256())
        return adapter

    def parameter_sha256(self) -> str:
        """Recompute the live model identity to catch post-load weight mutation."""

        parameter_digest = hashlib.sha256()
        for name, parameter in sorted(self.model.named_parameters(), key=lambda item: item[0]):
            data = parameter.detach().to("cpu", dtype=self.torch.float32).contiguous().numpy()
            parameter_digest.update(name.encode("utf-8") + b"\0")
            parameter_digest.update(data.tobytes())
        return parameter_digest.hexdigest()

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

    def generate_free_with_status(
        self, prompt: str, max_new_tokens: int = 64
    ) -> GenerationResult:
        """Greedy free generation with EOS/cap status bound to the output."""

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        prompt_ids = self.tokenizer.encode(prompt)
        token_ids = [self.bos_id, *prompt_ids]
        generated: list[int] = []
        stop_reason = "token_cap"
        for _ in range(max_new_tokens):
            if len(token_ids) >= self.spec.context_length:
                stop_reason = "context_limit"
                break
            logits = self._logits(token_ids)
            next_id = int(self.torch.argmax(logits[-1]).item())
            if next_id == self.eos_id:
                stop_reason = "eos"
                break
            token_ids.append(next_id)
            generated.append(next_id)
        result = GenerationResult(
            text=self.tokenizer.decode(generated),
            terminated_eos=stop_reason == "eos",
            generated_tokens=len(generated),
            stop_reason=stop_reason,
        )
        result.assert_valid()
        self.last_generation_result = result
        return result

    def generate_free(self, prompt: str, max_new_tokens: int = 64) -> str:
        """Compatibility wrapper; evidence-producing callers use status form."""

        return self.generate_free_with_status(prompt, max_new_tokens).text

    def generate_constrained(self, prompt: str, candidates: list[str]) -> str:
        """Greedy constrained generation over scored candidates."""

        scores = self.score_candidates(prompt, "", candidates)
        best = max(range(len(candidates)), key=lambda index: scores[index])
        return candidates[best]


# Keep direct references to the inference implementation captured when this
# module is imported. Phase-One qualification checks the references and
# rejects instance-shadowed methods/subclasses before any generation runs.
_CANONICAL_INFERENCE_METHODS = {
    name: getattr(CheckpointBackedV5Adapter, name)
    for name in (
        "_logits",
        "_suffix_ids",
        "parameter_sha256",
        "score_candidates",
        "generate_free_with_status",
        "generate_free",
        "generate_constrained",
    )
}


def assert_canonical_checkpoint_adapter(adapter: Any) -> None:
    """Reject adapter subclassing and instance-level inference overrides.

    This is an evaluation integrity guard, not a Python security sandbox. It
    prevents a caller from attaching an oracle method to an otherwise genuine
    checkpoint adapter and then passing its descriptive identity to Signac.
    """

    if type(adapter) is not CheckpointBackedV5Adapter:
        raise ValueError("Phase-One requires the canonical checkpoint-backed V5 adapter type")
    instance_attributes = vars(adapter)
    for name, implementation in _CANONICAL_INFERENCE_METHODS.items():
        if name in instance_attributes or getattr(type(adapter), name, None) is not implementation:
            raise ValueError(f"checkpoint adapter inference method is overridden: {name}")
    identity = getattr(adapter, "identity", None)
    if getattr(identity, "implementation_sha256", None) != ADAPTER_IMPLEMENTATION_SHA256:
        raise ValueError("checkpoint adapter identity does not bind the canonical implementation")
    if (
        getattr(identity, "checkpoint_binding_schema", None) != CHECKPOINT_STORE_BINDING
        or getattr(adapter, "_verified_checkpoint_sha256", None) != identity.checkpoint_sha256
        or not getattr(adapter, "_verified_training_state_sha256", None)
    ):
        raise ValueError("Phase-One requires a checkpoint-store-verified model and training-state binding")
    if type(adapter.tokenizer) is not FrozenTokenizer:
        raise ValueError("checkpoint adapter must use the frozen tokenizer facade")
    identity_tokenizer = adapter.tokenizer.identity
    identity_tokenizer.assert_valid()
    if identity_tokenizer.artifact_sha256 != identity.tokenizer_artifact_sha256:
        raise ValueError("checkpoint adapter tokenizer identity changed after loading")
    if adapter.parameter_sha256() != identity.parameter_sha256:
        raise ValueError("checkpoint adapter model parameters changed after identity was recorded")


__all__ = [
    "ADAPTER_SCHEMA",
    "ADAPTER_IMPLEMENTATION_SHA256",
    "CALLER_ASSERTED_CHECKPOINT_BINDING",
    "CHECKPOINT_STORE_BINDING",
    "assert_canonical_checkpoint_adapter",
    "AdapterIdentity",
    "CheckpointBackedV5Adapter",
    "GenerationResult",
    "SCORING_CONTRACT_ID",
    "SCORING_CONTRACT_SHA256",
    "SCORING_RULE",
    "assert_canonical_checkpoint_adapter",
]
