"""
pass_gates.py — Phase 3 | Component 45O
Pass Specialization: teaching each recursive pass what it exists to do.

The three passes of Ouroboros are not identical repetitions.
They are a cognitive pipeline:
  Pass 1 — Semantic Anchoring      → understand the question
  Pass 2 — Logic Integration       → build the answer
  Pass 3 — Adversarial Verification → challenge the answer

This file defines the auxiliary losses that enforce this specialization
during training. Without these losses the passes would collapse into
doing the same thing, wasting the recursive structure entirely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# PASS 1 — SEMANTIC ANCHORING
# Goal: hidden state after pass 1 should strongly capture topic and intent.
# Proxy: a small auxiliary classifier trained to predict the domain/topic
#        label from the pass-1 hidden state (e.g. "math", "science", "chat").
#        If pass 1 is doing its job, its hidden state is topic-rich.
# ─────────────────────────────────────────────────────────────────────────────

class SemanticAnchorLoss(nn.Module):
    """
    Auxiliary loss for Pass 1.

    A lightweight probe (linear classifier) is placed on top of the
    pass-1 hidden state and trained to predict the semantic domain label.

    This creates gradient pressure for pass 1 to encode domain / intent
    information in its hidden state.

    The probe itself is discarded at inference — only the gradient signal
    matters during training.
    """

    def __init__(self, d_model: int, n_domains: int = 16):
        super().__init__()
        self.d_model   = d_model
        self.n_domains = n_domains

        # Lightweight linear probe — NOT part of the main forward pass.
        # Parameters: d_model × n_domains = 256 × 16 = 4096
        # These ARE extra parameters, but they are training-only probes.
        # They are stripped before saving the inference model.
        self.probe = nn.Linear(d_model, n_domains)

    def forward(
        self,
        pass1_hidden: torch.Tensor,   # (B, T, d_model)
        domain_labels: torch.Tensor,  # (B,) — integer domain class per sample
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for predicting domain from pass-1 hidden state.

        We use the mean-pooled hidden state (across the sequence dimension)
        as the sentence-level representation for classification.
        """
        # Mean pool over sequence → (B, d_model)
        pooled = pass1_hidden.mean(dim=1)

        # Classify → (B, n_domains)
        logits = self.probe(pooled)

        # Cross-entropy against the domain label
        loss = F.cross_entropy(logits, domain_labels)
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# PASS 2 — LOGIC INTEGRATION
# Goal: hidden state after pass 2 should reflect coherent causal structure.
# Proxy: penalise high variance between token representations in pass 2.
#        Coherent reasoning produces smoothly evolving hidden states.
#        Incoherent reasoning produces wildly fluctuating hidden states.
# ─────────────────────────────────────────────────────────────────────────────

class LogicIntegrationLoss(nn.Module):
    """
    Auxiliary loss for Pass 2.

    Coherence proxy: consecutive token representations in the sequence
    should not change too abruptly. A smooth trajectory in hidden-state
    space correlates with logically connected thought.

    Loss = mean squared difference between adjacent token hidden states.
    Minimising this encourages Pass 2 to produce locally consistent reasoning.

    We also add a contrastive term that rewards Pass 2 for being *more*
    coherent than Pass 1, ensuring genuine improvement, not just smoothness.
    """

    def __init__(self, smoothness_weight: float = 1.0, improvement_weight: float = 0.5):
        super().__init__()
        self.smoothness_weight   = smoothness_weight
        self.improvement_weight  = improvement_weight

    def forward(
        self,
        pass1_hidden: torch.Tensor,  # (B, T, d_model)
        pass2_hidden: torch.Tensor,  # (B, T, d_model)
    ) -> torch.Tensor:
        """
        Compute coherence loss for pass 2 relative to pass 1.
        """
        # ── Smoothness: penalise abrupt adjacent-token jumps in pass 2 ──────
        # Difference between token_i and token_{i+1}
        diff_p2 = pass2_hidden[:, 1:, :] - pass2_hidden[:, :-1, :]  # (B, T-1, d)
        smoothness_loss = (diff_p2 ** 2).mean()

        # ── Improvement: pass 2 should be smoother than pass 1 ──────────────
        diff_p1 = pass1_hidden[:, 1:, :] - pass1_hidden[:, :-1, :]  # (B, T-1, d)
        roughness_p1 = (diff_p1 ** 2).mean().detach()  # detach: treat as target
        roughness_p2 = (diff_p2 ** 2).mean()

        # Reward if pass 2 is smoother than pass 1; penalise if not.
        # hinge(0, rough_p2 - rough_p1): only penalise when p2 is rougher.
        improvement_loss = F.relu(roughness_p2 - roughness_p1)

        return (
            self.smoothness_weight  * smoothness_loss
          + self.improvement_weight * improvement_loss
        )


# ─────────────────────────────────────────────────────────────────────────────
# PASS 3 — ADVERSARIAL VERIFICATION
# Goal: pass 3 should catch errors that pass 2 made and correct them.
# Proxy: when the final output (from pass 3) is more correct than what
#        pass 2 would have produced alone, give a reward (negative loss).
# ─────────────────────────────────────────────────────────────────────────────

class VerificationRewardLoss(nn.Module):
    """
    Auxiliary loss for Pass 3.

    We compare the language-modelling loss from pass 2's hidden state
    against the LM loss from pass 3's hidden state (the final output).

    If pass 3 is better than pass 2 → reward (reduce loss).
    If pass 3 is the same or worse  → penalise.

    This directly incentivises pass 3 to be a self-correcting pass,
    not a redundant copy of pass 2.

    Also includes a divergence term to ensure pass 3 *changes* the
    representation meaningfully (not just copying pass 2 passively).
    """

    def __init__(
        self,
        lm_head: nn.Module,
        reward_scale: float = 0.5,
        divergence_weight: float = 0.1,
    ):
        super().__init__()
        self.lm_head           = lm_head
        self.reward_scale      = reward_scale
        self.divergence_weight = divergence_weight

    def forward(
        self,
        pass2_hidden: torch.Tensor,   # (B, T, d_model)
        pass3_hidden: torch.Tensor,   # (B, T, d_model)
        targets: torch.Tensor,        # (B, T) — ground truth token ids
    ) -> torch.Tensor:
        """
        Compute verification reward loss.
        """
        B, T, _ = pass2_hidden.shape
        V       = self.lm_head.out_features if hasattr(self.lm_head, 'out_features') \
                  else self.lm_head.weight.shape[0]

        # ── LM loss from pass 2 (detached — it's the baseline to beat) ──────
        with torch.no_grad():
            logits_p2 = self.lm_head(pass2_hidden)
            loss_p2   = F.cross_entropy(
                logits_p2.view(B * T, V),
                targets.view(B * T),
                ignore_index=-1,
                reduction='mean',
            )

        # ── LM loss from pass 3 ───────────────────────────────────────────────
        logits_p3 = self.lm_head(pass3_hidden)
        loss_p3   = F.cross_entropy(
            logits_p3.view(B * T, V),
            targets.view(B * T),
            ignore_index=-1,
            reduction='mean',
        )

        # ── Verification reward ───────────────────────────────────────────────
        # Positive when pass 3 improves over pass 2, negative when it doesn't.
        # We *minimise* this loss so reward (negative) is good.
        improvement    = loss_p2 - loss_p3               # positive = improvement
        reward_loss    = -self.reward_scale * improvement  # negate: minimise means maximise improvement

        # ── Divergence: ensure pass 3 is not identical to pass 2 ─────────────
        cosine_sim     = F.cosine_similarity(
            pass3_hidden.view(B * T, -1),
            pass2_hidden.view(B * T, -1).detach(),
            dim=-1,
        ).mean()
        # Penalise high cosine similarity → reward divergence
        divergence_loss = self.divergence_weight * cosine_sim

        return reward_loss + divergence_loss


# ─────────────────────────────────────────────────────────────────────────────
# PASS CONSISTENCY LOSS
# All three passes should converge toward the same answer.
# Penalise large divergence between final hidden states across passes.
# ─────────────────────────────────────────────────────────────────────────────

class PassConsistencyLoss(nn.Module):
    """
    Ensures the three passes cooperate rather than diverge wildly.

    The ideal behaviour: each pass refines the hidden state in the same
    direction, with each refinement getting closer to the true answer.

    Measured as: mean pairwise L2 distance between pass hidden states.
    Minimising this pulls all three passes toward agreement.

    Note: this is balanced against VerificationRewardLoss, which wants
    pass 3 to differ from pass 2. The tension is intentional:
      - Be different enough to correct errors
      - But not so different that you become incoherent
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, pass_hiddens: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            pass_hiddens: list of (B, T, d_model) tensors, one per pass
        """
        n = len(pass_hiddens)
        if n < 2:
            return torch.tensor(0.0, device=pass_hiddens[0].device)

        consistency_loss = torch.tensor(0.0, device=pass_hiddens[0].device)
        pair_count = 0

        for i in range(n):
            for j in range(i + 1, n):
                # L2 distance between hidden states of pass i and pass j
                diff  = (pass_hiddens[i] - pass_hiddens[j]) ** 2
                consistency_loss = consistency_loss + diff.mean()
                pair_count += 1

        return consistency_loss / (pair_count * self.temperature)
