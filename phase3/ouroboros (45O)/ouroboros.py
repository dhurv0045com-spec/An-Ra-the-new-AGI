"""
ouroboros.py — Phase 3 | Component 45O
OuroborosDecoder: Recursive depth without added parameters.

The Ouroboros is the ancient symbol of a snake eating its own tail.
Here it means: the same transformer layers are applied N times,
each pass feeding its output back as input to the next.

Same weights. Same size. Deeper thought.
Intelligence scales with passes, not parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class OuroborosDecoder(nn.Module):
    """
    Wraps an existing transformer model and runs its layers
    recursively for N passes.

    Each pass has a distinct cognitive purpose:
      Pass 1 — Semantic Anchoring   (what is being asked?)
      Pass 2 — Logic Integration    (what is the reasoning?)
      Pass 3 — Adversarial Check    (is the answer correct?)

    The only new parameters introduced:
      pass_gates    : (n_passes, d_model)  — pass-specific context vectors
      blend_weights : (n_passes,)          — how much of each pass to accumulate

    Everything else is borrowed from base_model. No new layers.
    No new attention heads. No new feedforward stacks.
    """

    def __init__(self, base_model: nn.Module, n_passes: int = 3):
        super().__init__()

        self.model    = base_model
        self.n_passes = n_passes
        self.d_model  = base_model.d_model  # expect base_model to expose this

        # ── Pass Gates ────────────────────────────────────────────────────────
        # One learned vector per pass. Injected additively into the hidden state
        # before each pass begins. This shifts the transformer's "attention bias"
        # so the same weights produce qualitatively different processing.
        #
        # Think of these as mode selectors:
        #   gate[0] nudges attention toward surface meaning
        #   gate[1] nudges attention toward logical connectives
        #   gate[2] nudges attention toward contradiction / error signals
        #
        # Shape: (n_passes, d_model)
        # New parameters: n_passes × d_model = 3 × 256 = 768
        self.pass_gates = nn.Parameter(
            torch.zeros(n_passes, self.d_model)
        )
        nn.init.normal_(self.pass_gates, mean=0.0, std=0.02)

        # ── Blend Weights ─────────────────────────────────────────────────────
        # Controls how much of each pass is accumulated into the running total.
        # Initialized equally (1/n_passes) so early training is stable.
        # The model learns to weight later passes more heavily as training matures.
        #
        # Shape: (n_passes,)
        # New parameters: n_passes = 3
        self.blend_weights = nn.Parameter(
            torch.ones(n_passes) / n_passes
        )

        # ── Total new parameters: 768 + 3 = 771 ──────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through N recursive loops of the base transformer.

        Args:
            x       : token indices, shape (batch, seq_len)
            targets : token indices for loss, shape (batch, seq_len), or None

        Returns:
            logits  : (batch, seq_len, vocab_size)
            loss    : scalar cross-entropy loss, or None if no targets given
        """
        # ── Embed input tokens once ───────────────────────────────────────────
        # Embeddings are shared across all passes — no duplication.
        hidden = self.model.embed(x)                  # (B, T, d_model)

        # ── Accumulator ───────────────────────────────────────────────────────
        # Each pass contributes a weighted slice to the accumulated state.
        # This is the "tail-eating" mechanism: accumulated feeds back
        # into the next pass as its starting hidden state.
        accumulated = torch.zeros_like(hidden)

        # Store per-pass hidden states for auxiliary losses during training
        pass_hiddens = []

        # Normalize blend weights so they sum to 1 (softmax-style stability)
        blend = F.softmax(self.blend_weights, dim=0)  # (n_passes,)

        for pass_idx in range(self.n_passes):

            # ── Inject pass-specific context ──────────────────────────────────
            # The gate vector is broadcast over batch and sequence dimensions.
            # It biases the hidden state before this pass starts, effectively
            # telling the transformer: "this time, focus on X."
            gate   = self.pass_gates[pass_idx]               # (d_model,)
            hidden = hidden + gate.unsqueeze(0).unsqueeze(0)  # (B, T, d_model)

            # ── Run all transformer layers (shared weights) ───────────────────
            hidden = self.model.run_all_layers(hidden)        # (B, T, d_model)

            # ── Store for auxiliary loss ──────────────────────────────────────
            pass_hiddens.append(hidden)

            # ── Accumulate with learned blend weight ──────────────────────────
            weight      = blend[pass_idx]
            accumulated = accumulated + weight * hidden

            # ── Ouroboros loop: feed accumulated back ─────────────────────────
            # This is what makes it recursive. The next pass doesn't start
            # from scratch — it starts from everything learned so far.
            hidden = accumulated

        # ── Project accumulated state to vocabulary logits ────────────────────
        logits = self.model.lm_head(accumulated)      # (B, T, vocab_size)

        # ── Compute loss if targets provided ──────────────────────────────────
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                targets.reshape(B * T),
                ignore_index=-1,
            )

        # Attach pass hiddens to allow auxiliary loss computation externally
        self._last_pass_hiddens = pass_hiddens

        return logits, loss

    @property
    def new_parameter_count(self) -> int:
        """Returns the count of parameters added by Ouroboros (not base model)."""
        return self.pass_gates.numel() + self.blend_weights.numel()

    def count_parameters(self) -> dict:
        base  = sum(p.numel() for p in self.model.parameters())
        added = self.new_parameter_count
        return {
            "base_model_params"     : base,
            "ouroboros_added_params": added,
            "total_params"          : base + added,
            "overhead_pct"          : round(100 * added / base, 4),
        }
