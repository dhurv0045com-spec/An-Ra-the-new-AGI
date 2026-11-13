"""V8 model factory, typed heads (J, F, P, F-cap, U), and algebraic/parity controls.

All architectures share identical initial backbone weights for a given init_seed.
Heads are initialized inside an isolated RNG fork seeded deterministically from
(architecture, init_seed, v8_spec.sha256()).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from anra_v5.elendril.codec import digest
from anra_v5.elendril.representation import UTF8, SpecialToken
from v5_model.core import assert_receipt, assert_single_embedding, initialize, packed_layout
from v5_model.initialize import NORMAL_STD

from .contracts import ACTION_19_LABELS, SYMBOLS, V8ModelSpec

AP_MIN_TOKEN_ID = UTF8.encode("A")[0]  # 69
AP_MAX_TOKEN_ID = UTF8.encode("P")[0]  # 84
ANSWER_MARKER_TOKEN_ID = UTF8.special(SpecialToken.ANSWER)  # 3

OP_TO_INDEX = {"ANSWER": 0, "QUERY": 1, "RETRIEVE": 2, "ABSTAIN": 3}


def _init_linear(linear: nn.Linear, *, bias_value: float = 0.0) -> None:
    nn.init.normal_(linear.weight, mean=0.0, std=NORMAL_STD)
    if linear.bias is not None:
        nn.init.constant_(linear.bias, bias_value)


def _init_gru_cell(gru: nn.GRUCell) -> None:
    nn.init.normal_(gru.weight_ih, mean=0.0, std=NORMAL_STD)
    nn.init.normal_(gru.weight_hh, mean=0.0, std=NORMAL_STD)
    if gru.bias_ih is not None:
        nn.init.constant_(gru.bias_ih, 0.0)
    if gru.bias_hh is not None:
        nn.init.constant_(gru.bias_hh, 0.0)


@dataclass(frozen=True)
class TypedForwardOutput:
    log_probs_19: torch.Tensor
    log_p_op: torch.Tensor
    log_p_val: torch.Tensor
    gate: torch.Tensor | None = None
    attention_entropy: torch.Tensor | None = None
    has_pointer_candidates: torch.Tensor | None = None


class V8TypedModel(nn.Module):
    """Wrap the pure V5 Core with a versioned V8 typed action/argument head."""

    def __init__(self, core: nn.Module, v8_spec: V8ModelSpec) -> None:
        super().__init__()
        self.core = core
        self.v8_spec = v8_spec
        w = v8_spec.base_spec.width
        arch = v8_spec.architecture

        if arch == "J":
            self.joint_head = nn.Linear(w, 19, bias=True)
            _init_linear(self.joint_head)
        elif arch in ("F", "P", "F-cap", "U"):
            self.op_head = nn.Linear(w, 4, bias=True)
            self.val_head = nn.Linear(w, 16, bias=True)
            _init_linear(self.op_head)
            _init_linear(self.val_head)
            if arch == "P" and v8_spec.copy_enabled:
                d = v8_spec.pointer_dim
                self.q_proj = nn.Linear(w, d, bias=False)
                self.k_proj = nn.Linear(w, d, bias=False)
                self.gate = nn.Linear(w, 1, bias=True)
                _init_linear(self.q_proj)
                _init_linear(self.k_proj)
                _init_linear(self.gate, bias_value=-2.0)
            elif arch == "F-cap":
                r = v8_spec.bottleneck_width
                self.cap_in = nn.Linear(w, r, bias=True)
                self.cap_out = nn.Linear(r, w, bias=True)
                _init_linear(self.cap_in)
                _init_linear(self.cap_out)
            elif arch == "U":
                du = v8_spec.recurrent_dim
                self.u_q = nn.Linear(w, du, bias=False)
                self.u_k = nn.Linear(w, du, bias=False)
                self.u_v = nn.Linear(w, du, bias=False)
                self.u_out = nn.Linear(du, w, bias=True)
                self.u_gru = nn.GRUCell(w, w)
                _init_linear(self.u_q)
                _init_linear(self.u_k)
                _init_linear(self.u_v)
                _init_linear(self.u_out)
                _init_gru_cell(self.u_gru)
        else:
            raise ValueError(f"V8TypedModel does not support architecture {arch!r}")

    def forward_hidden_only(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        *,
        use_activation_checkpointing: bool = False,
    ) -> torch.Tensor:
        positions, mask = packed_layout(segment_ids, torch_module=torch)
        return self.core.forward_hidden(
            token_ids,
            positions,
            mask,
            use_activation_checkpointing=use_activation_checkpointing,
        )

    def forward_action(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        action_positions: torch.Tensor,
        *,
        action_batch_indices: torch.Tensor | None = None,
        copy_enabled_override: bool | None = None,
        value_permutation: torch.Tensor | None = None,
        use_activation_checkpointing: bool = False,
    ) -> TypedForwardOutput:
        """Evaluate typed 19-way log-probabilities at the specified public-prefix positions.

        Args:
            token_ids: [B, L] int64 tokens.
            segment_ids: [B, L] int64 segment IDs (-1 for trailing padding).
            action_positions: [N] int64 column indices of final public prefix token (ANSWER marker).
            action_batch_indices: optional [N] int64 row indices (defaults to 0..B-1 when N == B).
            copy_enabled_override: optional bool to disable pointer path at fixed weights.
            value_permutation: optional [N, L] or [16] tensor permuting candidate symbol bins
                for causal pointer interventions while holding attention weights fixed.
        """

        if action_positions.ndim != 1:
            raise ValueError("action_positions must be a 1D tensor")
        if action_batch_indices is None:
            if action_positions.shape[0] != token_ids.shape[0]:
                raise ValueError("action_batch_indices required when action count != batch size")
            b_idx = torch.arange(token_ids.shape[0], device=token_ids.device, dtype=torch.int64)
        else:
            b_idx = action_batch_indices.to(dtype=torch.int64)
        pos_idx = action_positions.to(dtype=torch.int64)
        if (segment_ids[b_idx, pos_idx] < 0).any().item():
            raise ValueError("action position cannot point to padding")

        H = self.forward_hidden_only(
            token_ids,
            segment_ids,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        h = H[b_idx, pos_idx].float()  # [N, W]
        arch = self.v8_spec.architecture

        if arch == "J":
            z = self.joint_head(h)
            log_probs_19 = F.log_softmax(z, dim=-1)
            p19 = log_probs_19.exp()
            p_ans_mass = p19[:, :16].sum(dim=-1, keepdim=True)
            log_p_op = torch.log(torch.cat([p_ans_mass, p19[:, 16:]], dim=-1).clamp_min(1e-30))
            log_p_val = torch.log((p19[:, :16] / p_ans_mass.clamp_min(1e-30)).clamp_min(1e-30))
            return TypedForwardOutput(log_probs_19=log_probs_19, log_p_op=log_p_op, log_p_val=log_p_val)

        if arch == "U":
            L = token_ids.shape[1]
            seg_row = segment_ids[b_idx]  # [N, L]
            q_seg = segment_ids[b_idx, pos_idx].unsqueeze(1)
            pos_grid = torch.arange(L, device=token_ids.device).unsqueeze(0)
            prefix_mask = (pos_grid <= pos_idx.unsqueeze(1)) & (seg_row == q_seg) & (seg_row >= 0)
            H_row = H[b_idx].float()  # [N, L, W]
            k_all = self.u_k(H_row)  # [N, L, D]
            v_all = self.u_v(H_row)  # [N, L, D]
            scale = math.sqrt(self.v8_spec.recurrent_dim)
            z_state = h
            for _ in range(self.v8_spec.recurrent_steps):
                q_t = self.u_q(z_state)  # [N, D]
                attn_scores = torch.bmm(q_t.unsqueeze(1), k_all.transpose(1, 2)).squeeze(1) / scale
                attn_weights = F.softmax(attn_scores.masked_fill(~prefix_mask, float("-inf")), dim=-1)
                read_vec = torch.bmm(attn_weights.unsqueeze(1), v_all).squeeze(1)
                z_state = self.u_gru(self.u_out(read_vec), z_state)
            h_op = z_state
            h_val = z_state
        elif arch == "F-cap":
            h_op = h
            h_val = h + self.cap_out(F.silu(self.cap_in(h)))
        else:
            h_op = h
            h_val = h

        log_p_op = F.log_softmax(self.op_head(h_op), dim=-1)
        z_val = self.val_head(h_val)

        use_copy = (
            arch == "P"
            and self.v8_spec.copy_enabled
            and (copy_enabled_override is not False)
        )
        if not use_copy:
            log_p_val = F.log_softmax(z_val, dim=-1)
            log_probs_19 = torch.cat([log_p_op[:, 0:1] + log_p_val, log_p_op[:, 1:]], dim=-1)
            return TypedForwardOutput(log_probs_19=log_probs_19, log_p_op=log_p_op, log_p_val=log_p_val)

        # Pointer-assisted distribution over public A-P prefix tokens in the same segment
        L = token_ids.shape[1]
        tok_row = token_ids[b_idx]  # [N, L]
        seg_row = segment_ids[b_idx]  # [N, L]
        q_seg = segment_ids[b_idx, pos_idx].unsqueeze(1)  # [N, 1]
        pos_grid = torch.arange(L, device=token_ids.device).unsqueeze(0)  # [1, L]
        is_prefix = pos_grid < pos_idx.unsqueeze(1)
        same_seg = (seg_row == q_seg) & (seg_row >= 0)
        is_ap = (tok_row >= AP_MIN_TOKEN_ID) & (tok_row <= AP_MAX_TOKEN_ID)
        cand_mask = is_prefix & same_seg & is_ap  # [N, L]

        H_row = H[b_idx].float()  # [N, L, W]
        q = self.q_proj(h)  # [N, D]
        K = self.k_proj(H_row)  # [N, L, D]
        scores = torch.bmm(q.unsqueeze(1), K.transpose(1, 2)).squeeze(1) / math.sqrt(self.v8_spec.pointer_dim)

        has_cand = cand_mask.any(dim=-1, keepdim=True)  # [N, 1]
        safe_mask = cand_mask.clone()
        safe_mask[~has_cand.squeeze(-1), 0] = True
        attn = F.softmax(scores.masked_fill(~safe_mask, float("-inf")), dim=-1) * cand_mask.to(scores.dtype)

        bins = (tok_row - AP_MIN_TOKEN_ID).clamp(0, 15)  # [N, L]
        if value_permutation is not None:
            if value_permutation.ndim == 1 and value_permutation.shape[0] == 16:
                bins = value_permutation[bins]
            elif value_permutation.shape == bins.shape:
                bins = value_permutation
            else:
                raise ValueError("value_permutation must have shape [16] or [N, L]")

        p_val = F.softmax(z_val, dim=-1)
        p_copy = torch.zeros_like(p_val)
        p_copy.scatter_add_(1, bins, attn)

        g = torch.sigmoid(self.gate(h))  # [N, 1]
        p_arg = torch.where(has_cand, (1.0 - g) * p_val + g * p_copy, p_val)
        log_p_val = torch.log(p_arg.clamp_min(1e-30))
        log_probs_19 = torch.cat([log_p_op[:, 0:1] + log_p_val, log_p_op[:, 1:]], dim=-1)

        # Entropy of pointer attention over candidate positions
        safe_attn = attn.clamp_min(1e-30)
        entropy = -(attn * torch.log(safe_attn)).sum(dim=-1)
        entropy = torch.where(has_cand.squeeze(-1), entropy, torch.zeros_like(entropy))
        return TypedForwardOutput(
            log_probs_19=log_probs_19,
            log_p_op=log_p_op,
            log_p_val=log_p_val,
            gate=g.squeeze(-1),
            attention_entropy=entropy,
            has_pointer_candidates=has_cand.squeeze(-1),
        )


def build_v8_model(v8_spec: V8ModelSpec, init_seed: int, *, torch_module: Any = None) -> nn.Module:
    """Construct a V8 model with identical backbone weights across architectures."""

    v8_spec.assert_valid()
    tor = torch_module or torch
    core = initialize(v8_spec.base_spec, init_seed, torch_module=tor)
    if v8_spec.architecture == "B0":
        assert_single_embedding(core)
        assert_receipt(core, v8_spec.base_spec)
        return core

    head_seed = v8_spec.extra_head_seed(init_seed)
    with tor.random.fork_rng():
        tor.manual_seed(head_seed)
        model = V8TypedModel(core, v8_spec)
    if v8_spec.architecture in ("F", "P", "F-cap", "U"):
        op_val_seed = int(
            digest(["elendril-v8-op-val-seed-v1", init_seed, v8_spec.base_spec.sha256()]),
            16,
        ) % (2**32)
        with tor.random.fork_rng():
            tor.manual_seed(op_val_seed)
            _init_linear(model.op_head)
            _init_linear(model.val_head)

    assert_single_embedding(model.core)
    assert_receipt(model.core, v8_spec.base_spec)
    total_params = sum(int(p.numel()) for p in model.parameters())
    if total_params != v8_spec.total_parameters():
        raise ValueError(
            f"model parameter count {total_params} != V8ModelSpec total {v8_spec.total_parameters()}"
        )
    if len({id(p) for p in model.parameters()}) != len(list(model.parameters())):
        raise ValueError("V8 model exposes duplicate parameter identities")
    return model


def compute_typed_nll(
    out: TypedForwardOutput,
    gold_ops: torch.Tensor,
    gold_args: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (summed_nll, per_state_nll) over the supplied states.

    Non-ANSWER states contribute 0 to argument NLL; empty ANSWER subsets never
    divide by zero.
    """

    op_nll = -out.log_p_op.gather(1, gold_ops.unsqueeze(1)).squeeze(1)
    is_answer = (gold_ops == 0).to(op_nll.dtype)
    val_nll = -out.log_p_val.gather(1, gold_args.clamp(0, 15).unsqueeze(1)).squeeze(1) * is_answer
    per_state = op_nll + val_nll
    return per_state.sum(), per_state


def algebraic_null_from_joint(joint_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Algebraically re-express J's [N, 19] logits as (log_p_op, log_p_val, recombined_log_p19).

    Must exactly match J's joint distribution, argmax, and gradients within
    floating-point tolerance.
    """

    log_j = F.log_softmax(joint_logits, dim=-1)
    log_p_ans = torch.logsumexp(log_j[:, :16], dim=-1, keepdim=True)
    log_p_op = torch.cat([log_p_ans, log_j[:, 16:]], dim=-1)
    log_p_val = log_j[:, :16] - log_p_ans
    recombined = torch.cat([log_p_op[:, 0:1] + log_p_val, log_p_op[:, 1:]], dim=-1)
    return log_p_op, log_p_val, recombined


def expected_active_parameter_names(
    model: nn.Module,
    v8_spec: V8ModelSpec,
    *,
    has_answer_targets: bool,
) -> set[str]:
    """Explicit per-objective contract of which parameters must receive gradients."""

    all_names = {name for name, _ in model.named_parameters()}
    if v8_spec.architecture in ("B0", "J", "U"):
        return all_names
    # In F / P / F-cap, val_head (and pointer/bottleneck) only enter the loss when >= 1 ANSWER state is in the batch.
    if has_answer_targets:
        return all_names
    inactive_prefixes = ("val_head.", "q_proj.", "k_proj.", "gate.", "cap_in.", "cap_out.")
    return {name for name in all_names if not name.startswith(inactive_prefixes)}


__all__ = [
    "ANSWER_MARKER_TOKEN_ID",
    "AP_MAX_TOKEN_ID",
    "AP_MIN_TOKEN_ID",
    "OP_TO_INDEX",
    "TypedForwardOutput",
    "V8TypedModel",
    "algebraic_null_from_joint",
    "build_v8_model",
    "compute_typed_nll",
    "expected_active_parameter_names",
]
