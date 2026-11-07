"""Versioned tensor and greedy-decoding boundary for the existing ELENDRIL core.

The adapter owns no parameters, optimizer, task labels or truth. Categorical
logits include normalized log probabilities from associative decoders. No output
symbols are masked or repaired. An identity records compatibility, not quality.
"""

from dataclasses import asdict, dataclass
import re
from typing import Any, Sequence

from .codec import digest


@dataclass(frozen=True, slots=True)
class BackendIdentity:
    architecture_sha256: str
    tokenizer_sha256: str
    vocabulary_size: int
    context_length: int
    checkpoint_sha256: str | None = None
    schema: str = "elendril-neural-backend-v1"

    def __post_init__(self):
        if self.schema != "elendril-neural-backend-v1":
            raise ValueError("unsupported backend schema")
        hashes = (self.architecture_sha256, self.tokenizer_sha256)
        if self.checkpoint_sha256 is not None:
            hashes += (self.checkpoint_sha256,)
        if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes):
            raise ValueError("backend identities require full SHA-256 values")
        if any(type(value) is not int or value <= 0 for value in (self.vocabulary_size, self.context_length)):
            raise ValueError("backend dimensions must be positive integers")

    @property
    def identity(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True, slots=True)
class Generation:
    tokens: tuple[int, ...]
    token_probabilities: tuple[float, ...]
    ended_eos: bool
    forward_calls: int
    evaluated_positions: int
    stop_reason: str


class NeuralBackend:
    """Validate packed tensors and expose model-origin categorical predictions.

    Layouts are rebuilt from segment IDs, preventing arbitrary masks from
    silently exposing future tokens. Padding must use the declared PAD ID.
    Exact resume/migration remains the checkpoint writer's responsibility.
    """

    def __init__(self, model: Any, identity: BackendIdentity, *, pad_id: int = 0,
                 bos_id: int = 1, eos_id: int = 2):
        base = getattr(model, "core", model)
        config = base.config
        if config.vocabulary_size != identity.vocabulary_size or config.context_length != identity.context_length:
            raise ValueError("backend dimensions disagree with model configuration")
        specials = (pad_id, bos_id, eos_id)
        if len(set(specials)) != 3 or any(type(value) is not int or not 0 <= value < identity.vocabulary_size for value in specials):
            raise ValueError("PAD/BOS/EOS must be distinct vocabulary IDs")
        self.model, self.binding = model, identity
        self.pad_id, self.bos_id, self.eos_id = specials

    @property
    def device(self):
        return next(self.model.parameters()).device

    def parameter_inventory(self) -> dict:
        rows = {name: {"shape": list(parameter.shape), "dtype": str(parameter.dtype),
                       "numel": parameter.numel(), "trainable": parameter.requires_grad}
                for name, parameter in self.model.named_parameters()}
        return {"schema": "elendril-parameter-inventory-v1", "parameters": rows,
                "total": sum(row["numel"] for row in rows.values()), "sha256": digest(rows)}

    def logits(self, token_ids, segment_ids, *, activation_checkpointing: bool = False):
        import torch
        from v5_model.core import packed_layout

        if (token_ids.ndim != 2 or not token_ids.shape[0] or not 0 < token_ids.shape[1] <= self.binding.context_length
                or token_ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError("tokens must be nonempty rank-two integers within native context")
        if segment_ids.shape != token_ids.shape or segment_ids.device != token_ids.device or token_ids.device != self.device:
            raise ValueError("tokens, segments and model must share shape/device")
        if ((token_ids < 0) | (token_ids >= self.binding.vocabulary_size)).any().item():
            raise ValueError("token outside bound vocabulary")
        if ((segment_ids < 0) != (token_ids == self.pad_id)).any().item() or not (segment_ids >= 0).any(dim=1).all().item():
            raise ValueError("padding must agree with segments and every row needs content")
        positions, mask = packed_layout(segment_ids, torch_module=torch)
        if type(activation_checkpointing) is not bool:
            raise ValueError("activation_checkpointing must be boolean")
        if activation_checkpointing:
            # Associative wrappers do not expose the activation-checkpoint flag.
            if hasattr(self.model, "core"):
                raise ValueError("activation checkpointing is not supported by this wrapper")
            result = self.model(token_ids, positions, mask, use_activation_checkpointing=True)
        else:
            result = self.model(token_ids, positions, mask)
        if result.shape != (*token_ids.shape, self.binding.vocabulary_size):
            raise ValueError("model violated categorical-logit shape contract")
        if not torch.isfinite(result).all().item():
            raise FloatingPointError("nonfinite backend logits")
        return result

    def _prompt(self, ids: Sequence[int]) -> tuple[int, ...]:
        ids = tuple(ids)
        if (not ids or ids[0] != self.bos_id or len(ids) > self.binding.context_length
                or any(type(token) is not int or not 0 <= token < self.binding.vocabulary_size or token == self.pad_id for token in ids)):
            raise ValueError("prompt requires BOS and bounded non-PAD vocabulary IDs")
        return ids

    def next_distribution(self, ids: Sequence[int]):
        """Return one unmasked categorical distribution, without a KV cache.

        This inference-only call evaluates len(ids) positions. Callers making
        costed action choices must charge that amount before invoking it.
        """
        import torch

        ids = self._prompt(ids)
        modes = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            with torch.inference_mode():
                tokens = torch.tensor([ids], device=self.device, dtype=torch.int64)
                return self.logits(tokens, torch.zeros_like(tokens))[0, -1].softmax(-1)
        finally:
            for module, mode in modes:
                module.training = mode

    def generate(self, prompts: Sequence[Sequence[int]], *, max_new_tokens: int) -> tuple[Generation, ...]:
        """Greedy complete outputs; no context truncation or forced EOS."""
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= self.binding.context_length or not prompts:
            raise ValueError("generation needs prompts and a positive bounded token limit")
        prompts = tuple(self._prompt(ids) for ids in prompts)
        if any(len(ids) + max_new_tokens - 1 > self.binding.context_length for ids in prompts):
            raise ValueError("generation exceeds native context")
        outputs = []
        for prompt in prompts:
            ids, tokens, probabilities, positions = list(prompt), [], [], 0
            stop_reason = "length"
            for _ in range(max_new_tokens):
                scores = self.next_distribution(ids)
                token = int(scores.argmax())
                tokens.append(token)
                probabilities.append(float(scores[token]))
                positions += len(ids)
                if token == self.eos_id:
                    stop_reason = "eos"
                    break
                if token in (self.pad_id, self.bos_id):
                    # Preserve the emitted illegal completion rather than
                    # feeding a padding/control token back as public content.
                    stop_reason = "invalid-special"
                    break
                ids.append(token)
            outputs.append(Generation(tuple(tokens), tuple(probabilities), tokens[-1] == self.eos_id,
                                      len(tokens), positions, stop_reason))
        return tuple(outputs)


def set_attention_backend(model: Any, backend: str) -> int:
    """Select an explicit execution candidate; state_dict and weights stay intact.

    ``repeat-sdpa`` is the unchanged reference and explicit fallback.
    ``native-gqa`` asks PyTorch SDPA to manage GQA. Unsupported kernels fail;
    no OOM, numerical error or backend exception is hidden by auto-fallback.
    """
    if backend not in ("repeat-sdpa", "native-gqa"):
        raise ValueError("unknown attention execution backend")
    layers = [module for module in model.modules() if hasattr(module, "execution_backend")]
    if not layers:
        raise ValueError("model exposes no compatible attention layers")
    for layer in layers:
        layer.execution_backend = backend
    return len(layers)
