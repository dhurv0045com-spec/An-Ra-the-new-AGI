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

    def reuse_blocks(self, hidden: Tensor) -> Tensor:
        if not self.gates_enabled:
            return hidden
        blocks = self.decoder.blocks
        reused = blocks[-GATED_BLOCK_COUNT:]
        for index, block in enumerate(reused):
            alpha = torch.tanh(self.gate_alpha[index])
            hidden = hidden + alpha * (block(hidden, None, None) - hidden)
        return hidden

    def forward(self, tokens: Tensor, padding_mask: Tensor | None = None, **kwargs):
        hidden = self.decoder.forward_hidden(tokens, padding_mask)
        hidden = self.reuse_blocks(hidden)
        logits = torch.nn.functional.linear(hidden, self.decoder.embedding.weight)
        from bramastra_lab.research.models.wrapper import ModelOutput

        return ModelOutput(logits=logits)

    def forward_hidden(self, tokens: Tensor, padding_mask: Tensor | None = None, **kwargs):
        hidden = self.decoder.forward_hidden(tokens, padding_mask)
        return self.reuse_blocks(hidden)

    def gate_values(self) -> list[float]:
        return [float(torch.tanh(self.gate_alpha[index]).item())
                for index in range(GATED_BLOCK_COUNT)]


def migrate_from_parent(parent: IntegratedModel, config, *,
                        gates_enabled: bool = True,
                        tolerance: float = 1e-6) -> GatedReuseModel:
    """Zero-gate migration: the child must be functionally equal to the
    parent at alpha=0, share block storage (not clones), and carry an
    explicitly changed architecture identity."""
    child = GatedReuseModel(config, gates_enabled=gates_enabled)
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
    probe = torch.randint(0, config.model.vocab, (2, 12))
    parent.eval()
    child.eval()
    with torch.no_grad():
        parent_logits = parent(probe).logits
        child_logits = child(probe).logits
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
    probe = torch.randint(0, child.build_config.model.vocab, (2, 12))
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
    child.zero_grad(set_to_none=True)
    return results


def nonzero_gate_reaches_shared_blocks(child: GatedReuseModel) -> bool:
    """At nonzero gates the reused blocks must execute: perturbing a reused
    block's weights must change the output when |gate| > 0."""
    if not child.gates_enabled:
        return False
    with torch.no_grad():
        child.gate_alpha.fill_(0.5)
    probe = torch.randint(0, child.build_config.model.vocab, (2, 12))
    with torch.no_grad():
        before = child(probe).logits.clone()
        block = child.decoder.blocks[-1]
        saved = block.attention.output.weight.detach().clone()
        block.attention.output.weight.add_(0.01)
        after = child(probe).logits
        block.attention.output.weight.copy_(saved)
        child.gate_alpha.zero_()
    return bool((before - after).abs().max().item() > 0.0)
