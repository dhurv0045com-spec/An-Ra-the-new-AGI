"""S20-backed ELENDRIL V8-X1 model factory: R, R-noedge, P-capR, U-1/2/4/U2-untied, and reused V8 heads."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from anra_v5.elendril.representation import UTF8, SpecialToken
from v5_model.core import assert_receipt, assert_single_embedding, initialize, packed_layout
from v5_model.initialize import NORMAL_STD

from .contracts import X1ModelSpec, X1_U_ARCHITECTURES

AP_MIN_TOKEN_ID = UTF8.encode("A")[0]
AP_MAX_TOKEN_ID = UTF8.encode("P")[0]
ANSWER_MARKER_TOKEN_ID = UTF8.special(SpecialToken.ANSWER)
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
class X1S20ForwardOutput:
    log_probs_19: torch.Tensor
    log_p_op: torch.Tensor
    log_p_val: torch.Tensor
    gate: torch.Tensor | None = None
    attention_entropy: torch.Tensor | None = None
    has_pointer_candidates: torch.Tensor | None = None
    decision_hidden: torch.Tensor | None = None
    slot_states: torch.Tensor | None = None


class _URecurrentStepBlock(nn.Module):
    def __init__(self, width: int, recurrent_dim: int) -> None:
        super().__init__()
        self.recurrent_dim = int(recurrent_dim)
        self.u_q = nn.Linear(width, self.recurrent_dim, bias=False)
        self.u_k = nn.Linear(width, self.recurrent_dim, bias=False)
        self.u_v = nn.Linear(width, self.recurrent_dim, bias=False)
        self.u_out = nn.Linear(self.recurrent_dim, width, bias=True)
        self.u_gru = nn.GRUCell(width, width)
        _init_linear(self.u_q)
        _init_linear(self.u_k)
        _init_linear(self.u_v)
        _init_linear(self.u_out)
        _init_gru_cell(self.u_gru)

    def step(
        self,
        z_state: torch.Tensor,
        H_row: torch.Tensor,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        q_t = self.u_q(z_state)
        k_all = self.u_k(H_row)
        v_all = self.u_v(H_row)
        scores = torch.bmm(q_t.unsqueeze(1), k_all.transpose(1, 2)).squeeze(1) / math.sqrt(self.recurrent_dim)
        weights = F.softmax(scores.masked_fill(~prefix_mask, float("-inf")), dim=-1)
        read_vec = torch.bmm(weights.unsqueeze(1), v_all).squeeze(1)
        return self.u_gru(self.u_out(read_vec), z_state)


class X1TypedModel(nn.Module):
    """Wrap the exact qualified V5 S20 Core with V8-X1 heads, relational reader R, controls, or U."""

    def __init__(self, core: nn.Module, x1_spec: X1ModelSpec) -> None:
        super().__init__()
        x1_spec.assert_valid()
        self.core = core
        self.x1_spec = x1_spec
        w = x1_spec.base_spec.width
        arch = x1_spec.architecture

        if arch == "J":
            self.joint_head = nn.Linear(w, 19, bias=True)
            _init_linear(self.joint_head)
            return

        self.op_head = nn.Linear(w, 4, bias=True)
        self.val_head = nn.Linear(w, 16, bias=True)
        _init_linear(self.op_head)
        _init_linear(self.val_head)

        if arch == "F-cap":
            r = x1_spec.bottleneck_width
            self.cap_in = nn.Linear(w, r, bias=True)
            self.cap_out = nn.Linear(r, w, bias=True)
            _init_linear(self.cap_in)
            _init_linear(self.cap_out)
        elif arch in ("P", "P-capR", "R", "R-noedge"):
            d = x1_spec.pointer_dim
            self.q_proj = nn.Linear(w, d, bias=False)
            self.k_proj = nn.Linear(w, d, bias=False)
            self.gate = nn.Linear(w, 1, bias=True)
            _init_linear(self.q_proj)
            _init_linear(self.k_proj)
            _init_linear(self.gate, bias_value=-2.0)

            if arch == "P-capR":
                c = x1_spec.cap_r_hidden
                self.cap_r_in = nn.Linear(w, c, bias=True)
                self.cap_r_out = nn.Linear(c, d, bias=True)
                _init_linear(self.cap_r_in)
                _init_linear(self.cap_r_out)
            elif arch in ("R", "R-noedge"):
                s, k = x1_spec.slot_dim, x1_spec.num_slots
                self.tok_ln = nn.LayerNorm(w)
                self.tok_proj = nn.Linear(w, s, bias=False)
                self.slots_init = nn.Parameter(torch.randn(k, s) * NORMAL_STD)
                self.slot_q = nn.Linear(s, s, bias=False)
                self.slot_k = nn.Linear(s, s, bias=False)
                self.slot_v = nn.Linear(s, s, bias=False)
                self.slot_gru = nn.GRUCell(s, s)
                self.agg_ln = nn.LayerNorm(s)
                self.agg_mlp_1 = nn.Linear(s, 2 * s, bias=True)
                self.agg_mlp_2 = nn.Linear(2 * s, s, bias=True)
                self.goal_q_proj = nn.Linear(w, s, bias=False)
                self.read_ln = nn.LayerNorm(s)
                self.r_to_ptr = nn.Linear(s, d, bias=False)
                for lin in (self.tok_proj, self.slot_q, self.slot_k, self.slot_v, self.agg_mlp_1, self.agg_mlp_2, self.goal_q_proj, self.r_to_ptr):
                    _init_linear(lin)
                _init_gru_cell(self.slot_gru)

                if arch == "R":
                    eh = x1_spec.slot_edge_hidden
                    self.rel_ln = nn.LayerNorm(s)
                    self.edge_mlp_1 = nn.Linear(3 * s, eh, bias=True)
                    self.edge_mlp_2 = nn.Linear(eh, 1, bias=True)
                    self.msg_proj = nn.Linear(s, s, bias=False)
                    self.rel_gru = nn.GRUCell(s, s)
                    for lin in (self.edge_mlp_1, self.edge_mlp_2, self.msg_proj):
                        _init_linear(lin)
                    _init_gru_cell(self.rel_gru)
                else:
                    m = x1_spec.noedge_mlp_dim
                    self.noedge_ln = nn.LayerNorm(s)
                    self.noedge_mlp_1 = nn.Linear(s, m, bias=True)
                    self.noedge_mlp_2 = nn.Linear(m, s, bias=True)
                    self.noedge_gru = nn.GRUCell(s, s)
                    _init_linear(self.noedge_mlp_1)
                    _init_linear(self.noedge_mlp_2)
                    _init_gru_cell(self.noedge_gru)
        elif arch in X1_U_ARCHITECTURES:
            n_blocks = 2 if arch == "U2-untied" else 1
            self.u_blocks = nn.ModuleList([_URecurrentStepBlock(w, x1_spec.recurrent_dim) for _ in range(n_blocks)])
        elif arch != "F":
            raise ValueError(f"X1TypedModel does not support architecture {arch!r}")

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
        lesion_mode: str | None = None,
        recurrent_steps_override: int | None = None,
        use_activation_checkpointing: bool = False,
    ) -> X1S20ForwardOutput:
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

        H = self.forward_hidden_only(token_ids, segment_ids, use_activation_checkpointing=use_activation_checkpointing).float()
        h = H[b_idx, pos_idx]
        arch = self.x1_spec.architecture

        if arch == "J":
            log_probs_19 = F.log_softmax(self.joint_head(h), dim=-1)
            p19 = log_probs_19.exp()
            p_ans_mass = p19[:, :16].sum(dim=-1, keepdim=True)
            log_p_op = torch.log(torch.cat([p_ans_mass, p19[:, 16:]], dim=-1).clamp_min(1e-30))
            log_p_val = torch.log((p19[:, :16] / p_ans_mass.clamp_min(1e-30)).clamp_min(1e-30))
            return X1S20ForwardOutput(
                log_probs_19=log_probs_19,
                log_p_op=log_p_op,
                log_p_val=log_p_val,
                decision_hidden=h,
            )

        L = token_ids.shape[1]
        seg_row = segment_ids[b_idx]
        q_seg = segment_ids[b_idx, pos_idx].unsqueeze(1)
        pos_grid = torch.arange(L, device=token_ids.device).unsqueeze(0)
        same_seg = (seg_row == q_seg) & (seg_row >= 0)
        valid_prefix = (pos_grid <= pos_idx.unsqueeze(1)) & same_seg

        if arch in X1_U_ARCHITECTURES:
            steps = self.x1_spec.recurrent_steps if recurrent_steps_override is None else int(recurrent_steps_override)
            H_row = H[b_idx]
            z_state = h
            for t in range(steps):
                blk = self.u_blocks[t % len(self.u_blocks)]
                z_state = blk.step(z_state, H_row, valid_prefix)
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

        use_pointer = arch in ("P", "P-capR", "R", "R-noedge")
        use_copy = use_pointer and self.x1_spec.copy_enabled and (copy_enabled_override is not False) and (lesion_mode != "disable_copy")

        slot_states: torch.Tensor | None = None
        if not use_copy:
            log_p_val = F.log_softmax(z_val, dim=-1)
            log_probs_19 = torch.cat([log_p_op[:, 0:1] + log_p_val, log_p_op[:, 1:]], dim=-1)
            return X1S20ForwardOutput(
                log_probs_19=log_probs_19,
                log_p_op=log_p_op,
                log_p_val=log_p_val,
                decision_hidden=h_val,
            )

        tok_row = token_ids[b_idx]
        H_row = H[b_idx]
        q_ptr = self.q_proj(h)

        if arch == "P-capR" and lesion_mode != "zero_r":
            q_ptr = q_ptr + self.cap_r_out(F.silu(self.cap_r_in(h)))
        elif arch in ("R", "R-noedge"):
            if lesion_mode == "zero_r":
                pass
            else:
                n_items = b_idx.shape[0]
                s = self.x1_spec.slot_dim
                k_slots = self.x1_spec.num_slots
                x_tok = self.tok_proj(self.tok_ln(H_row))
                slots = self.slots_init.unsqueeze(0).expand(n_items, k_slots, s).contiguous()
                k_tok = self.slot_k(x_tok)
                v_tok = self.slot_v(x_tok)
                for _ in range(self.x1_spec.agg_iterations):
                    q_s = self.slot_q(slots)
                    logits = torch.bmm(q_s, k_tok.transpose(1, 2)) / math.sqrt(s)
                    # Normalize over valid prefix tokens safely without NaN when rows are masked
                    a_ki = F.softmax(logits, dim=1) * valid_prefix.unsqueeze(1).to(logits.dtype)
                    denom = a_ki.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    w_ki = a_ki / denom
                    u_k = torch.bmm(w_ki, v_tok)
                    slots = self.slot_gru(u_k.reshape(n_items * k_slots, s), slots.reshape(n_items * k_slots, s)).view(n_items, k_slots, s)
                    slots = slots + self.agg_mlp_2(F.silu(self.agg_mlp_1(self.agg_ln(slots))))

                q_goal = self.goal_q_proj(h)
                for _ in range(self.x1_spec.msg_iterations):
                    if arch == "R":
                        s_ln = self.rel_ln(slots)
                        s_l = s_ln.unsqueeze(1).expand(n_items, k_slots, k_slots, s)
                        s_k = s_ln.unsqueeze(2).expand(n_items, k_slots, k_slots, s)
                        q_exp = q_goal.unsqueeze(1).unsqueeze(2).expand(n_items, k_slots, k_slots, s)
                        e_kl = self.edge_mlp_2(F.silu(self.edge_mlp_1(torch.cat([s_k, s_l, q_exp], dim=-1)))).squeeze(-1)
                        if k_slots > 1:
                            eye = torch.eye(k_slots, dtype=torch.bool, device=slots.device).unsqueeze(0)
                            e_kl = e_kl.masked_fill(eye, float("-inf"))
                        alpha = F.softmax(e_kl, dim=-1)
                        if lesion_mode == "permute_edges" and k_slots > 1:
                            alpha = torch.roll(alpha, shifts=1, dims=-1)
                        m_k = torch.bmm(alpha, self.msg_proj(slots))
                        slots = self.rel_gru(m_k.reshape(n_items * k_slots, s), slots.reshape(n_items * k_slots, s)).view(n_items, k_slots, s)
                    else:
                        up = self.noedge_mlp_2(F.silu(self.noedge_mlp_1(self.noedge_ln(slots))))
                        slots = self.noedge_gru(up.reshape(n_items * k_slots, s), slots.reshape(n_items * k_slots, s)).view(n_items, k_slots, s)
                slot_states = slots
                r_scores = torch.bmm(q_goal.unsqueeze(1), self.read_ln(slots).transpose(1, 2)).squeeze(1) / math.sqrt(s)
                r_alpha = F.softmax(r_scores, dim=-1)
                r_vec = torch.bmm(r_alpha.unsqueeze(1), slots).squeeze(1)
                r_contrib = self.r_to_ptr(r_vec)
                if lesion_mode == "sham":
                    r_contrib = r_contrib + 0.0
                q_ptr = q_ptr + r_contrib

        cand_mask = (pos_grid < pos_idx.unsqueeze(1)) & same_seg & (tok_row >= AP_MIN_TOKEN_ID) & (tok_row <= AP_MAX_TOKEN_ID)
        has_cand = cand_mask.any(dim=-1, keepdim=True)
        k_all = self.k_proj(H_row)
        scores = torch.bmm(q_ptr.unsqueeze(1), k_all.transpose(1, 2)).squeeze(1) / math.sqrt(self.x1_spec.pointer_dim)
        safe_mask = cand_mask.clone()
        safe_mask[~has_cand.squeeze(-1), 0] = True
        attn = F.softmax(scores.masked_fill(~safe_mask, float("-inf")), dim=-1) * cand_mask.to(scores.dtype)

        bins = (tok_row - AP_MIN_TOKEN_ID).clamp(0, 15)
        if lesion_mode == "permute_val_bins":
            perm = torch.arange(16, device=bins.device)
            perm = torch.roll(perm, shifts=1, dims=0)
            bins = perm[bins]
        elif value_permutation is not None:
            bins = value_permutation[bins] if value_permutation.ndim == 1 else value_permutation

        p_val = F.softmax(z_val, dim=-1)
        p_copy = torch.zeros_like(p_val)
        p_copy.scatter_add_(1, bins, attn)
        g = torch.sigmoid(self.gate(h))
        p_arg = torch.where(has_cand, (1.0 - g) * p_val + g * p_copy, p_val)
        log_p_val = torch.log(p_arg.clamp_min(1e-30))
        log_probs_19 = torch.cat([log_p_op[:, 0:1] + log_p_val, log_p_op[:, 1:]], dim=-1)
        entropy = -(attn * torch.log(attn.clamp_min(1e-30))).sum(dim=-1)
        entropy = torch.where(has_cand.squeeze(-1), entropy, torch.zeros_like(entropy))
        return X1S20ForwardOutput(
            log_probs_19=log_probs_19,
            log_p_op=log_p_op,
            log_p_val=log_p_val,
            gate=g.squeeze(-1),
            attention_entropy=entropy,
            has_pointer_candidates=has_cand.squeeze(-1),
            decision_hidden=h,
            slot_states=slot_states,
        )


def build_x1_s20_model(x1_spec: X1ModelSpec, init_seed: int, *, torch_module: Any = None) -> nn.Module:
    """Build X1 S20/tiny model with component-level identical shared initialization across arms."""

    x1_spec.assert_valid()
    tor = torch_module or torch
    core = initialize(x1_spec.base_spec, init_seed, torch_module=tor)
    if x1_spec.architecture == "B0":
        assert_single_embedding(core)
        assert_receipt(core, x1_spec.base_spec)
        return core

    with tor.random.fork_rng():
        tor.manual_seed(x1_spec.module_seed(init_seed))
        model = X1TypedModel(core, x1_spec)

    if x1_spec.architecture != "J":
        with tor.random.fork_rng():
            tor.manual_seed(x1_spec.f_head_seed(init_seed))
            _init_linear(model.op_head)
            _init_linear(model.val_head)
        if x1_spec.architecture in ("P", "P-capR", "R", "R-noedge"):
            with tor.random.fork_rng():
                tor.manual_seed(x1_spec.pointer_seed(init_seed))
                _init_linear(model.q_proj)
                _init_linear(model.k_proj)
                _init_linear(model.gate, bias_value=-2.0)

    assert_single_embedding(model.core)
    assert_receipt(model.core, x1_spec.base_spec)
    actual_params = sum(int(p.numel()) for p in model.parameters())
    if actual_params != x1_spec.total_parameters():
        raise ValueError(f"X1 parameter count mismatch: actual={actual_params} != spec={x1_spec.total_parameters()}")
    if len({id(p) for p in model.parameters()}) != len(list(model.parameters())):
        raise ValueError("X1 model exposes duplicate parameter identities")
    return model


def expected_x1_active_parameter_names(
    model: nn.Module,
    x1_spec: X1ModelSpec,
    *,
    has_answer_targets: bool,
) -> set[str]:
    all_names = {name for name, _ in model.named_parameters()}
    if x1_spec.architecture in ("B0", "J") or has_answer_targets:
        return all_names
    inactive_prefixes = (
        "val_head.",
        "q_proj.",
        "k_proj.",
        "gate.",
        "cap_in.",
        "cap_out.",
        "cap_r_",
        "tok_",
        "slots_init",
        "slot_",
        "agg_",
        "goal_q_proj.",
        "read_ln.",
        "r_to_ptr.",
        "rel_",
        "edge_",
        "msg_",
        "noedge_",
    )
    return {n for n in all_names if not n.startswith(inactive_prefixes)}


__all__ = [
    "ANSWER_MARKER_TOKEN_ID",
    "AP_MAX_TOKEN_ID",
    "AP_MIN_TOKEN_ID",
    "OP_TO_INDEX",
    "X1S20ForwardOutput",
    "X1TypedModel",
    "build_x1_s20_model",
    "expected_x1_active_parameter_names",
]
