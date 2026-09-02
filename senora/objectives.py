"""Real causal cross-entropy and query-swap contrastive objectives.

Implements:
1. Causal Next-Token Cross Entropy with FP32 reduction, ignore indexing,
   explicit token denominator, and non-finite loss guards.
2. Query-Swap Attribution Contrastive Loss: penalizes candidate-prior memorization
   and enforces query-conditioned value binding over counterfactual pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class NonFiniteLossError(RuntimeError):
    """Raised when training loss produces NaN or Inf."""


@dataclass(frozen=True, slots=True)
class LossReceipt:
    total_loss: float
    ce_loss: float
    query_swap_loss: float
    valid_token_count: int
    query_swap_pairs_count: int


try:
    import torch
    import torch.nn.functional as F

    def causal_cross_entropy(
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        ignore_index: int = -100,
    ) -> tuple[torch.Tensor, int]:
        """Compute causal next-token cross entropy loss in float32.
        
        Args:
            logits: Shape [batch, seq_len, vocab_size]
            targets: Shape [batch, seq_len]
            ignore_index: Target token ID to ignore (e.g. padding or prompt prefix)
            
        Returns:
            Tuple of (scalar loss tensor, count of valid supervised tokens)
        """
        # Targets are shifted relative to logits: predict target[t] from logit[t-1]
        # If input has already been shifted upstream, this function takes aligned logits and targets.
        # Here we accept aligned logits and targets: logits[:, :-1], targets[:, 1:]
        logits_flat = logits.float().view(-1, logits.shape[-1])
        targets_flat = targets.view(-1)

        valid_mask = targets_flat != ignore_index
        valid_token_count = int(valid_mask.sum().item())

        if valid_token_count == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=logits.device), 0

        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=ignore_index,
            reduction="sum",
        )
        mean_loss = loss / valid_token_count

        if not torch.isfinite(mean_loss):
            raise NonFiniteLossError(f"Causal Cross-Entropy produced non-finite loss: {mean_loss.item()}")

        return mean_loss, valid_token_count

    def query_swap_contrastive_loss(
        factual_target_logprob: torch.Tensor,
        counterfactual_distractor_logprob: torch.Tensor,
        counterfactual_target_logprob: torch.Tensor,
        factual_distractor_logprob: torch.Tensor,
    ) -> torch.Tensor:
        """Compute normalized query-attribution contrastive loss.
        
        Given a factual query q1 (with correct target y1 and distractor y2)
        and a counterfactual query q2 (with correct target y2 and distractor y1):
        
            l1 = -log sigmoid(log P(y1 | q1) - log P(y2 | q1))
            l2 = -log sigmoid(log P(y2 | q2) - log P(y1 | q2))
            L_qswap = mean(0.5 * (l1 + l2))
            
        This objective directly penalizes candidate priors and rewards query-driven binding.
        """
        diff1 = factual_target_logprob.float() - counterfactual_distractor_logprob.float()
        diff2 = counterfactual_target_logprob.float() - factual_distractor_logprob.float()

        # -log sigmoid(d) = softplus(-d) = log(1 + exp(-d))
        l1 = F.softplus(-diff1)
        l2 = F.softplus(-diff2)

        loss = 0.5 * (l1 + l2).mean()
        if not torch.isfinite(loss):
            raise NonFiniteLossError(f"Query-swap contrastive objective produced non-finite loss: {loss.item()}")
        return loss

    def compute_composite_training_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        ignore_index: int = -100,
        query_swap_lambda: float = 0.0,
        query_swap_payload: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, LossReceipt]:
        """Compute total training loss: L_ce + lambda * L_query_swap."""
        ce_loss, valid_tokens = causal_cross_entropy(logits, targets, ignore_index=ignore_index)

        qswap_loss = torch.tensor(0.0, dtype=torch.float32, device=logits.device)
        pair_count = 0

        if query_swap_lambda > 0.0 and query_swap_payload is not None:
            qswap_loss = query_swap_contrastive_loss(
                factual_target_logprob=query_swap_payload["factual_target_logprob"],
                counterfactual_distractor_logprob=query_swap_payload["counterfactual_distractor_logprob"],
                counterfactual_target_logprob=query_swap_payload["counterfactual_target_logprob"],
                factual_distractor_logprob=query_swap_payload["factual_distractor_logprob"],
            )
            pair_count = query_swap_payload["factual_target_logprob"].numel()

        total_loss = ce_loss + query_swap_lambda * qswap_loss

        receipt = LossReceipt(
            total_loss=float(total_loss.item()),
            ce_loss=float(ce_loss.item()),
            query_swap_loss=float(qswap_loss.item()),
            valid_token_count=valid_tokens,
            query_swap_pairs_count=pair_count,
        )
        return total_loss, receipt

except ImportError:  # pragma: no cover
    causal_cross_entropy = None  # type: ignore
    query_swap_contrastive_loss = None  # type: ignore
    compute_composite_training_loss = None  # type: ignore