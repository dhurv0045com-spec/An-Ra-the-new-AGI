"""
adaptive.py — Phase 3 | Component 45O
AdaptiveController: dynamic pass count based on question complexity.

Not every question deserves three passes.
  "What is 2+2?" — one pass is enough.
  "Explain the proof of Fermat's Last Theorem" — use three.

The AdaptiveController measures the uncertainty in the model's
hidden state after the first pass and decides how many additional
passes are worth computing.

This preserves inference speed for simple queries while reserving
full recursive depth for queries that actually need it.

The measure of complexity used here is normalised entropy of the
hidden state softmax distribution. High entropy = high uncertainty
= the model is unsure = more passes needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AdaptiveController(nn.Module):
    """
    Decides how many Ouroboros passes to run for a given input.

    Decision is made after the first pass, by examining the uncertainty
    of the current hidden state. The intuition:

      Low uncertainty → the model already "knows" the answer → stop early.
      High uncertainty → the model is confused → keep refining.

    Thresholds are configurable. Defaults calibrated for a d_model=256 model:
      uncertainty < low_threshold  → 1 pass  (simple)
      uncertainty < high_threshold → 2 passes (moderate)
      uncertainty ≥ high_threshold → 3 passes (complex)

    Can be extended to 4-5 passes for very hard reasoning tasks by
    raising max_passes and adding an extra threshold.
    """

    def __init__(
        self,
        d_model: int,
        max_passes: int = 3,
        low_threshold: float  = 0.3,
        high_threshold: float = 0.7,
    ):
        super().__init__()
        self.d_model         = d_model
        self.max_passes      = max_passes
        self.low_threshold   = low_threshold
        self.high_threshold  = high_threshold

        # Optional: a tiny learned classifier that supplements entropy.
        # Takes the mean-pooled hidden state and predicts a complexity score.
        # This allows the controller to learn signal beyond raw entropy.
        # Parameters: d_model × 1 = 256 (training-time refinement only)
        self.complexity_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.complexity_head.weight)
        nn.init.zeros_(self.complexity_head.bias)

    def measure_uncertainty(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Compute normalised entropy of the hidden state distribution
        as a proxy for how much the model knows.

        Args:
            hidden: (B, T, d_model) hidden state tensor

        Returns:
            uncertainty: scalar in [0, 1]
              0 = completely certain (low entropy, peaked distribution)
              1 = completely uncertain (max entropy, flat distribution)
        """
        # Mean pool over batch and sequence → (d_model,)
        pooled = hidden.mean(dim=[0, 1])  # (d_model,)

        # Convert to a probability distribution via softmax
        probs = F.softmax(pooled, dim=-1)  # (d_model,)

        # Shannon entropy: H = -Σ p_i log(p_i)
        # Clamp to avoid log(0)
        entropy = -(probs * (probs + 1e-9).log()).sum()

        # Normalise by log(d_model) — the theoretical maximum entropy
        # for a d_model-dimensional uniform distribution.
        max_entropy    = torch.tensor(self.d_model, dtype=torch.float).log()
        normalised     = entropy / max_entropy  # in [0, 1]

        return normalised.clamp(0.0, 1.0)

    def learned_complexity(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Learned complexity score (supplement to entropy).
        Trained via auxiliary signal — e.g., sequences known to be
        hard (math proofs, multi-step reasoning) labelled as complexity=1.

        Args:
            hidden: (B, T, d_model)

        Returns:
            complexity: scalar in [0, 1] via sigmoid
        """
        pooled     = hidden.mean(dim=[0, 1])      # (d_model,)
        raw_score  = self.complexity_head(pooled) # (1,)
        return torch.sigmoid(raw_score).squeeze()

    def decide_passes(
        self,
        hidden: torch.Tensor,
        use_learned: bool = True,
        force_passes: Optional[int] = None,
    ) -> int:
        """
        Decide how many passes to run based on hidden state uncertainty.

        Args:
            hidden       : (B, T, d_model) hidden state after pass 1
            use_learned  : if True, blend entropy with learned complexity score
            force_passes : override and return this value (useful for debugging)

        Returns:
            n_passes: integer in [1, max_passes]
        """
        if force_passes is not None:
            return min(force_passes, self.max_passes)

        with torch.no_grad():
            uncertainty = self.measure_uncertainty(hidden)

            if use_learned:
                # Blend: 70% entropy, 30% learned signal
                learned  = self.learned_complexity(hidden)
                combined = 0.7 * uncertainty + 0.3 * learned
            else:
                combined = uncertainty

            u = combined.item()

        # Threshold decisions
        if u < self.low_threshold:
            return 1    # Simple question — one pass is enough
        elif u < self.high_threshold:
            return 2    # Moderate complexity — two passes
        else:
            return min(3, self.max_passes)  # Complex — full recursion

    def explain_decision(self, hidden: torch.Tensor) -> dict:
        """
        Diagnostic: explains why the controller chose a given pass count.
        Useful during development and evaluation.
        """
        with torch.no_grad():
            entropy_score  = self.measure_uncertainty(hidden).item()
            learned_score  = self.learned_complexity(hidden).item()
            combined       = 0.7 * entropy_score + 0.3 * learned_score
            n_passes       = self.decide_passes(hidden)

        return {
            "entropy_uncertainty" : round(entropy_score, 4),
            "learned_complexity"  : round(learned_score, 4),
            "combined_score"      : round(combined, 4),
            "low_threshold"       : self.low_threshold,
            "high_threshold"      : self.high_threshold,
            "passes_decided"      : n_passes,
            "reasoning"           : (
                "simple question — 1 pass"      if n_passes == 1 else
                "moderate complexity — 2 passes" if n_passes == 2 else
                f"complex reasoning — {n_passes} passes"
            ),
        }


class OuroborosAdaptive(nn.Module):
    """
    A thin wrapper that integrates OuroborosDecoder with AdaptiveController.

    Runs the first pass, consults the controller, then runs remaining passes.
    Combines the efficiency of early stopping with the depth of full recursion.
    """

    def __init__(self, ouroboros_decoder, adaptive_controller: AdaptiveController):
        super().__init__()
        self.decoder    = ouroboros_decoder
        self.controller = adaptive_controller

    def forward(self, x: torch.Tensor, targets=None, verbose: bool = False):
        """
        Forward pass with adaptive pass count.

        Runs pass 1, measures uncertainty, decides total passes,
        then continues with the remaining passes.
        """
        import torch.nn.functional as F

        model    = self.decoder.model
        blend    = F.softmax(self.decoder.blend_weights, dim=0)

        hidden      = model.embed(x)
        accumulated = torch.zeros_like(hidden)

        # ── Pass 1 always runs ────────────────────────────────────────────────
        gate        = self.decoder.pass_gates[0]
        hidden      = hidden + gate.unsqueeze(0).unsqueeze(0)
        hidden      = model.run_all_layers(hidden)
        accumulated = accumulated + blend[0] * hidden
        hidden      = accumulated

        # ── Controller decides how many more passes to run ────────────────────
        n_passes = self.controller.decide_passes(hidden)

        if verbose:
            decision = self.controller.explain_decision(hidden)
            print(f"[AdaptiveController] {decision}")

        # ── Remaining passes ──────────────────────────────────────────────────
        for pass_idx in range(1, n_passes):
            gate        = self.decoder.pass_gates[pass_idx]
            hidden      = hidden + gate.unsqueeze(0).unsqueeze(0)
            hidden      = model.run_all_layers(hidden)
            accumulated = accumulated + blend[pass_idx] * hidden
            hidden      = accumulated

        logits = model.lm_head(accumulated)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss    = F.cross_entropy(
                logits.view(B * T, V),
                targets.view(B * T),
                ignore_index=-1,
            )

        return logits, loss, n_passes   # also return pass count for logging
