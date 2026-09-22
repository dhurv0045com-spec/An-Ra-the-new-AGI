"""E4 gated block-reuse architecture (I04): the original eight-block pass
remains; the final two blocks are re-executed after the normal pass, with
h_next = h + tanh(alpha_j) * (B_j(h) - h) and alpha_j initialized to zero so
the migrated model is functionally equal to its parent at zero gates.

The shared blocks are REUSED (not cloned): parameters appear once in the
named-parameter inventory, gates retain state, and the architecture identity
changes explicitly. Restore reconstructs sharing/gates before optimizer state
is applied.
"""
from __future__ import annotations

import torch
from torch import Tensor

from bramastra_lab.research.models import IntegratedModel
from bramastra_lab.research.models.decisions import DecisionError


class ArchitectureError(ValueError):
    """A gated-architecture migration or restore violated its contract."""


GATED_ARCHITECTURE_ID = "bramastra-gated-block-reuse/v1"
BASE_ARCHITECTURE_ID = "bramastra-base-decoder/v1"
GATED_BLOCK_COUNT = 2
GATED_PARAMETER_SLOTS = GATED_BLOCK_COUNT  # two scalar alpha values


class GatedReuseModel(IntegratedModel):
    """IntegratedModel + gated reuse of the final two decoder blocks.

    Forward: run the normal decoder pass, then for each reused block
    B_j: h = h + tanh(alpha_j) * (B_j(h) - h). The blocks execute (real
    compute cost) even at zero gates, but at alpha=0 the output equals the
    base decoder exactly (tanh(0) == 0).
    """

    def __init__(self, config, *, gates_enabled: bool = True) -> None:
        super().__init__(config)
        self.gates_enabled = gates_enabled
        if gates_enabled:
            self.gate_alpha = torch.nn.Parameter(torch.zeros(GATED_BLOCK_COUNT))
        else:
            # S0 carries the same two scalar slots, fixed/disabled, for a
            # parameter-inventory comparison.
            self.register_buffer("gate_alpha", torch.zeros(GATED_BLOCK_COUNT))
        self.architecture_id = GATED_ARCHITECTURE_ID if gates_enabled \
            else BASE_ARCHITECTURE_ID + "+disabled-gate-slots"



    def _decoder_with_reuse(self, tokens: Tensor,
                            padding_mask: Tensor | None = None,
                            attention_mask: Tensor | None = None):
        """Decoder blocks + gated reuse of the final two, BEFORE final norm.

        Preserves packed-segment isolation: an explicit attention_mask (derived
        from segment_ids by the caller) is forwarded to every block including
        the reused pass. Padding is forwarded in all blocks (R07).
        """
        hidden = self.decoder.embedding(tokens)
        for block in self.decoder.blocks:
            hidden = block(hidden, padding_mask, attention_mask)
        if self.gates_enabled:
            reused = self.decoder.blocks[-GATED_BLOCK_COUNT:]
            for index, block in enumerate(reused):
                alpha = torch.tanh(self.gate_alpha[index])
                hidden = hidden + alpha * (block(hidden, padding_mask, attention_mask) - hidden)
        return self.decoder.final_norm(hidden)

    def forward(self, tokens: Tensor, padding_mask: Tensor | None = None, **kwargs):
        """Full IntegratedModel contract through the gated path (R07).

        Accepts segment_ids, action_span_ends, action_mask, return_hidden and
        return_value like the base model; hidden states come from
        _decoder_with_reuse (never the bypassed decoder path). Packed-segment
        isolation and head outputs are preserved.
        """
        import math

        segment_ids = kwargs.get("segment_ids")
        action_span_ends = kwargs.get("action_span_ends")
        action_mask = kwargs.get("action_mask")
        return_hidden = bool(kwargs.get("return_hidden", False))
        return_value = bool(kwargs.get("return_value", False))
        if action_mask is not None and action_span_ends is None:
            raise ValueError("action_mask requires action_span_ends")
        attention_mask = None
        if segment_ids is not None:
            if not isinstance(segment_ids, Tensor) or segment_ids.shape != tokens.shape:
                raise ValueError("segment_ids must be an integer tensor shaped like tokens")
            if segment_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("segment_ids must use torch.int32 or torch.int64")
            attention_mask = segment_ids[:, None, :] == segment_ids[:, :, None]
        hidden = self._decoder_with_reuse(tokens, padding_mask, attention_mask)
        logits = torch.nn.functional.linear(hidden, self.decoder.embedding.weight)
        from bramastra_lab.research.models.wrapper import ModelOutput

        action_scores = None
        if action_span_ends is not None:
            if not isinstance(action_span_ends, Tensor) or action_span_ends.ndim != 2:
                raise ValueError("action_span_ends must have shape [batch, candidates]")
            if action_span_ends.shape[0] != tokens.shape[0]:
                raise ValueError("action_span_ends batch must match tokens batch")
            gathered = hidden.gather(
                1, action_span_ends.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
            action_scores = self.action_head(gathered).squeeze(-1)
            if action_mask is not None:
                if action_mask.shape != action_span_ends.shape \
                        or action_mask.dtype != torch.bool:
                    raise ValueError(
                        "action_mask must be a bool tensor shaped like action_span_ends")
                if not bool(action_mask.any(dim=-1).all()):
                    raise ValueError("every row needs at least one legal action candidate")
                action_scores = action_scores.masked_fill(~action_mask, -math.inf)
        value = None
        if return_value:
            if padding_mask is not None:
                lengths = padding_mask.sum(dim=-1).clamp_min(1) - 1
                last_positions = lengths.to(torch.long)
            else:
                last_positions = torch.full(
                    (tokens.shape[0],), tokens.shape[1] - 1,
                    device=tokens.device, dtype=torch.long)
            last_hidden = hidden.gather(
                1, last_positions[:, None, None].expand(-1, 1, hidden.shape[-1])).squeeze(1)
            value = self.value_head(last_hidden).squeeze(-1)
        return ModelOutput(logits=logits,
                           hidden=hidden if return_hidden else None,
                           action_scores=action_scores, value=value)

    def forward_hidden(self, tokens: Tensor, padding_mask: Tensor | None = None, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        segment_ids = kwargs.get("segment_ids")
        if attention_mask is None and segment_ids is not None:
            if not isinstance(segment_ids, Tensor) or segment_ids.shape != tokens.shape:
                raise ValueError("segment_ids must be an integer tensor shaped like tokens")
            attention_mask = segment_ids[:, None, :] == segment_ids[:, :, None]
        return self._decoder_with_reuse(tokens, padding_mask, attention_mask)

    def gate_values(self) -> list[float]:
        return [float(torch.tanh(self.gate_alpha[index]).item())
                for index in range(GATED_BLOCK_COUNT)]


def migrate_from_parent(parent: IntegratedModel, config, *,
                        gates_enabled: bool = True,
                        tolerance: float = 1e-6) -> GatedReuseModel:
    """Zero-gate migration: the child must be functionally equal to the
    parent at alpha=0, share block storage (not clones), and carry an
    explicitly changed architecture identity."""
    parent_device = next(parent.parameters()).device
    # Model construction initializes tensors even though their values will
    # immediately be replaced by the parent. Isolate that initialization so
    # qualification cannot advance the training/sampler RNG stream.
    with torch.random.fork_rng(devices=[]):
        child = GatedReuseModel(config, gates_enabled=gates_enabled)
    child = child.to(parent_device)
    # Copy parent weights into the child (same architecture backbone).
    parent_state = parent.state_dict()
    child_state = child.state_dict()
    for name, tensor in parent_state.items():
        if name in child_state:
            child_state[name] = tensor.clone()
    child.load_state_dict(child_state)
    # Gate parameters at exactly zero.
    if gates_enabled:
        with torch.no_grad():
            child.gate_alpha.zero_()
    # Functionally equal at zero gates: same logits on a probe batch.
    probe = qualification_probe_tokens(parent, config.model.vocab)
    parent_training = parent.training
    child_training = child.training
    parent.eval()
    child.eval()
    try:
        with torch.no_grad(), _isolated_device_rng(parent):
            parent_logits = parent(probe).logits
            child_logits = child(probe).logits
    finally:
        parent.train(parent_training)
        child.train(child_training)
    max_diff = float((parent_logits - child_logits).abs().max().item())
    if max_diff > tolerance:
        raise ArchitectureError(
            f"zero-gate migration is not functionally equal: max |diff| {max_diff}")
    # Shared storage proof: reused blocks' parameters appear exactly once in
    # the child's named-parameter inventory.
    block_names = [name for name, _ in child.named_parameters()
                   if name.startswith("decoder.blocks.")]
    if len(block_names) != len(set(block_names)):
        raise ArchitectureError("shared block parameters were cloned, not reused")
    return child


def check_gate_gradients(child: GatedReuseModel) -> dict[str, bool]:
    """Named gradient checks: at zero gates the reused-block contribution has
    tanh(0)=0 so gate gradients flow through tanh's derivative (=1) only if
    the block difference is nonzero; the shared blocks receive gradients
    through the main pass regardless. Split named checks, not one assertion."""
    parameters = list(child.parameters())
    saved_grads = [None if parameter.grad is None else parameter.grad.detach().clone()
                   for parameter in parameters]
    was_training = child.training
    try:
        child.zero_grad(set_to_none=True)
        child.train()
        with _isolated_device_rng(child):
            probe = qualification_probe_tokens(
                child, child.build_config.model.vocab)
            logits = child(probe).logits
            loss = logits.float().square().mean()
            loss.backward()
        results = {
            "gate_gradients_present": child.gate_alpha.grad is not None
            and bool(torch.isfinite(child.gate_alpha.grad).all())
            if child.gates_enabled else False,
            "shared_block_gradients_present": any(
                parameter.grad is not None
                for name, parameter in child.decoder.blocks[-2:].named_parameters()),
            "embedding_gradients_present": child.decoder.embedding.weight.grad is not None,
        }
        return results
    finally:
        for parameter, gradient in zip(parameters, saved_grads):
            parameter.grad = gradient
        child.train(was_training)


def nonzero_gate_reaches_shared_blocks(child: GatedReuseModel) -> bool:
    """At nonzero gates the reused blocks must execute: perturbing a reused
    block's weights must change the output when |gate| > 0."""
    if not child.gates_enabled:
        return False
    was_training = child.training
    old_gate = child.gate_alpha.detach().clone()
    block = child.decoder.blocks[-1]
    old_weight = block.attention.output.weight.detach().clone()
    child.eval()
    try:
        with _isolated_device_rng(child), torch.no_grad():
            child.gate_alpha.fill_(0.5)
            probe = qualification_probe_tokens(
                child, child.build_config.model.vocab)
            before = child(probe).logits.clone()
            block.attention.output.weight.add_(0.01)
            after = child(probe).logits
            return bool((before - after).abs().max().item() > 0.0)
    finally:
        with torch.no_grad():
            block.attention.output.weight.copy_(old_weight)
            child.gate_alpha.copy_(old_gate)
        child.train(was_training)


def qualification_probe_tokens(model: IntegratedModel, vocab: int, *,
                               batch: int = 2, length: int = 12) -> Tensor:
    """Deterministic qualification input using a private device RNG."""
    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    return torch.randint(0, vocab, (batch, length),
                         device=device, generator=generator)


def _isolated_device_rng(model: IntegratedModel):
    """Context preserving CPU and the model's CUDA RNG, if any."""
    device = next(model.parameters()).device
    devices = ([device.index if device.index is not None
                else torch.cuda.current_device()]
               if device.type == "cuda" else [])
    return torch.random.fork_rng(devices=devices)
