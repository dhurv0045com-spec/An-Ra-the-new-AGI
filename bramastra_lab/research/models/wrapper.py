"""Canonical integrated-model wrapper (B01).

Extends the accepted BRAMASTRA ``TransformerDecoder`` with typed optional
outputs: token logits, optional hidden states, finite-action scores and a
value estimate. No second language model exists; every output shares the
decoder's representation. Head parameters are counted explicitly and the
base decoder's count is never advertised as the integrated count.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from bramastra_lab.research.config import BuildConfig


@dataclass(frozen=True)
class ModelOutput:
    """Typed results from one integrated forward pass.

    ``logits`` is always the full physical vocabulary; training-only logit
    treatments are applied by the trainer, never inside the model. ``action_scores``
    is present only when ``action_span_ends`` is supplied, and ``value`` only
    when ``return_value`` is set.
    """

    logits: Tensor
    hidden: Tensor | None = None
    action_scores: Tensor | None = None
    value: Tensor | None = None


class IntegratedModel(nn.Module):
    """One decoder plus explicit action/value probes on its hidden states."""

    def __init__(self, config: BuildConfig) -> None:
        super().__init__()
        if not isinstance(config, BuildConfig):
            raise TypeError("config must be a BuildConfig")
        self.build_config = config
        self.decoder = self._build_decoder(config)
        width = config.model.width
        self.action_head = nn.Linear(width, 1, bias=False)
        self.value_head = nn.Linear(width, 1, bias=False)

        actual = sum(parameter.numel() for parameter in self.parameters())
        expected = config.parameter_count()
        if actual != expected:
            raise RuntimeError(
                f"integrated model has {actual:,} parameters; expected {expected:,}")

    @staticmethod
    def _build_decoder(config: BuildConfig):
        from bramastra_lab.model import TransformerDecoder

        return TransformerDecoder(config.model_config())

    def base_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.decoder.parameters())

    def head_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.head_parameters())

    def head_parameters(self):
        yield from self.action_head.parameters()
        yield from self.value_head.parameters()

    def reset_parameters(self) -> None:
        """Re-initialize decoder and heads from the current global RNG state."""
        self.decoder.reset_parameters()
        for head in (self.action_head, self.value_head):
            nn.init.normal_(head.weight, mean=0.0, std=0.02)

    def forward_hidden(self, tokens: Tensor, padding_mask: Tensor | None = None,
                       attention_mask: Tensor | None = None,
                       segment_ids: Tensor | None = None) -> Tensor:
        """Canonical hidden path for scorers (R07).

        Base model: decoder pass with optional packed-segment isolation.
        GatedReuseModel overrides this to route through _decoder_with_reuse.
        Scorers must call this (or the model call with return_hidden) rather
        than decoder.forward_hidden directly when gates may be present.
        """
        if attention_mask is None and segment_ids is not None:
            if not isinstance(segment_ids, Tensor) or segment_ids.shape != tokens.shape:
                raise ValueError("segment_ids must be an integer tensor shaped like tokens")
            attention_mask = segment_ids[:, None, :] == segment_ids[:, :, None]
        return self.decoder.forward_hidden(tokens, padding_mask, attention_mask)

    def forward(
        self,
        tokens: Tensor,
        padding_mask: Tensor | None = None,
        *,
        segment_ids: Tensor | None = None,
        action_span_ends: Tensor | None = None,
        action_mask: Tensor | None = None,
        return_hidden: bool = False,
        return_value: bool = False,
    ) -> ModelOutput:
        """Run the shared decoder and optional heads.

        ``action_span_ends`` has shape ``[batch, candidates]`` and holds, for
        each legal-action candidate serialization, the sequence position of
        its final token. The action score is computed from the decoder hidden
        state at that position, so candidates share the representation and no
        per-candidate parameters exist. ``action_mask`` ([batch, candidates],
        bool) marks legal candidates; illegal scores are set to ``-inf`` and a
        row without any legal candidate is rejected.
        """
        if action_mask is not None and action_span_ends is None:
            raise ValueError("action_mask requires action_span_ends")
        attention_mask = None
        if segment_ids is not None:
            if not isinstance(segment_ids, Tensor) or segment_ids.shape != tokens.shape:
                raise ValueError("segment_ids must be an integer tensor shaped like tokens")
            if segment_ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("segment_ids must use torch.int32 or torch.int64")
            # A position may only attend to keys of its own episode segment.
            attention_mask = segment_ids[:, None, :] == segment_ids[:, :, None]
        hidden = self.decoder.forward_hidden(tokens, padding_mask, attention_mask)
        logits = nn.functional.linear(hidden, self.decoder.embedding.weight)

        action_scores: Tensor | None = None
        if action_span_ends is not None:
            if not isinstance(action_span_ends, Tensor) or action_span_ends.ndim != 2:
                raise ValueError("action_span_ends must have shape [batch, candidates]")
            if action_span_ends.shape[0] != tokens.shape[0]:
                raise ValueError("action_span_ends batch must match tokens batch")
            if action_span_ends.dtype not in (torch.int32, torch.int64):
                raise TypeError("action_span_ends must use torch.int32 or torch.int64")
            if bool((action_span_ends < 0).any()) or bool((action_span_ends >= tokens.shape[1]).any()):
                raise ValueError("action_span_ends positions must be inside the sequence")
            gathered = hidden.gather(
                1,
                action_span_ends.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]),
            )
            action_scores = self.action_head(gathered).squeeze(-1)
            if action_mask is not None:
                if action_mask.shape != action_span_ends.shape or action_mask.dtype != torch.bool:
                    raise ValueError("action_mask must be a bool tensor shaped like action_span_ends")
                if not bool(action_mask.any(dim=-1).all()):
                    raise ValueError("every row needs at least one legal action candidate")
                action_scores = action_scores.masked_fill(~action_mask, -math.inf)

        value: Tensor | None = None
        if return_value:
            if padding_mask is not None:
                lengths = padding_mask.sum(dim=-1).clamp_min(1) - 1
                last_positions = lengths.to(torch.long)
            else:
                last_positions = torch.full(
                    (tokens.shape[0],), tokens.shape[1] - 1, device=tokens.device, dtype=torch.long)
            last_hidden = hidden.gather(
                1, last_positions[:, None, None].expand(-1, 1, hidden.shape[-1])).squeeze(1)
            value = self.value_head(last_hidden).squeeze(-1)

        return ModelOutput(
            logits=logits,
            hidden=hidden if return_hidden else None,
            action_scores=action_scores,
            value=value,
        )
