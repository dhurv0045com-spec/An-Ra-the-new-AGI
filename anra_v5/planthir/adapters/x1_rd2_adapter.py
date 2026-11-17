"""Independent non-V8 X1 R/D2 backend, relational slot model, and grouped D2 loss."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from anra_v5.elendril.codec import canonical, digest
from anra_v5.elendril.representation import UTF8, SpecialToken
from anra_v5.elendril.training import _dump, _load, batch_tensors, configure_device, runtime_identity
from anra_v5.elendril.v8.contracts import ACTION_19_LABELS, SYMBOLS
from v5_training.checkpoint import CheckpointStore
from v5_training.optimizer import build_adamw_optimizer, optimizer_group_receipt, validate_parameter_ownership
from v5_training.state import CURSOR_SCHEMA, IDENTITY_SCHEMA, CursorState, IdentityBindings, TrainingState

from ..backend import BACKEND_CONTRACT_VERSION, BackendCapabilities
from ..contracts import SubgoalProbeSpec, UniversalAction, UniversalObservation

X1_ARCHITECTURES = ("X1-P", "X1-R", "X1-R-noedge", "X1-P-capR")
X1_OBJECTIVES = ("ce", "pair_ce", "pair_eq", "pair_eq_cf")
AP_MIN_TOKEN_ID = UTF8.encode("A")[0]
AP_MAX_TOKEN_ID = UTF8.encode("P")[0]
_OP_TO_IDX = {"ANSWER": 0, "QUERY": 1, "RETRIEVE": 2, "ABSTAIN": 3}


@dataclass(frozen=True, slots=True)
class X1RD2Recipe:
    """Versioned training recipe for the independent X1 R/D2 grouped-loss backend."""

    seed: int = 7
    total_updates: int = 10
    batch_size: int = 8
    micro_batch_size: int = 4
    lr: float = 1e-4
    clip: float = 1.0
    objective: str = "pair_eq_cf"
    eq_weight: float = 0.1
    cf_weight: float = 0.1
    cf_margin: float = 0.5
    sampling: str = "uniform"
    schedule: str = "cosine"
    precision: str = "fp32"
    version: str = "anra-x1-rd2-recipe-v1"

    def validate(self) -> None:
        if self.version != "anra-x1-rd2-recipe-v1":
            raise ValueError("unsupported X1RD2Recipe version")
        for name in ("seed", "total_updates", "batch_size", "micro_batch_size"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"X1RD2Recipe.{name} must be an integer")
        if not (0 <= self.seed < 2**32 and 1 <= self.total_updates <= 100_000 and 1 <= self.micro_batch_size <= self.batch_size <= 256):
            raise ValueError("X1RD2Recipe counters out of bounds")
        if not math.isfinite(self.lr) or self.lr <= 0 or not math.isfinite(self.clip) or self.clip <= 0:
            raise ValueError("X1RD2Recipe lr and clip must be positive and finite")
        if any(not math.isfinite(x) or x < 0 for x in (self.eq_weight, self.cf_weight, self.cf_margin)):
            raise ValueError("X1RD2Recipe loss weights/margin must be nonnegative and finite")
        if self.objective not in X1_OBJECTIVES or self.sampling != "uniform" or self.schedule not in ("constant", "cosine") or self.precision not in ("fp32", "fp16"):
            raise ValueError("unsupported X1RD2Recipe objective, sampling, schedule, or precision")


@dataclass(frozen=True, slots=True)
class X1RD2Spec:
    """Versioned model specification for the non-V8 X1 R/D2 relational backend."""

    architecture: str
    vocab_size: int = UTF8.vocabulary_size
    hidden_dim: int = 32
    num_layers: int = 1
    num_heads: int = 2
    pointer_dim: int = 16
    num_slots: int = 4
    slot_dim: int = 16
    slot_edge_hidden: int = 16
    agg_iterations: int = 2
    msg_iterations: int = 2
    cap_hidden_dim: int = 16
    max_sequence_length: int = 640
    schema: str = "anra-x1-rd2-model-spec-v1"

    def __post_init__(self) -> None:
        self.assert_valid()

    def assert_valid(self) -> None:
        if self.schema != "anra-x1-rd2-model-spec-v1":
            raise ValueError("unsupported X1RD2Spec schema")
        if self.architecture not in X1_ARCHITECTURES:
            raise ValueError(f"unsupported X1 architecture: {self.architecture!r}")
        for name in ("vocab_size", "hidden_dim", "num_layers", "num_heads", "pointer_dim", "max_sequence_length"):
            val = getattr(self, name)
            if type(val) is not int or val <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.architecture in ("X1-R", "X1-R-noedge"):
            for name in ("num_slots", "slot_dim", "slot_edge_hidden", "agg_iterations", "msg_iterations"):
                val = getattr(self, name)
                if type(val) is not int or val <= 0:
                    raise ValueError(f"{name} must be positive for {self.architecture}")
        if self.architecture == "X1-P-capR" and (type(self.cap_hidden_dim) is not int or self.cap_hidden_dim <= 0):
            raise ValueError("X1-P-capR requires positive cap_hidden_dim")

    def total_parameters(self) -> int:
        self.assert_valid()
        h, v, seq = self.hidden_dim, self.vocab_size, self.max_sequence_length
        p_dim = self.pointer_dim
        total = v * h + seq * h
        # Transformer layers: ln1(2h) + qkv(3h^2) + out(h^2) + ln2(2h) + mlp1(2h^2 + 2h) + mlp2(2h^2 + h)
        per_layer = 8 * h * h + 7 * h
        total += self.num_layers * per_layer
        # ln_f(2h) + op_head(4h+4) + val_head(16h+16) + q_proj(h*p) + k_proj(h*p) + gate(h+1)
        total += (2 * h) + (4 * h + 4) + (16 * h + 16) + 2 * (h * p_dim) + (h + 1)
        if self.architecture == "X1-P-capR":
            c = self.cap_hidden_dim
            total += (h * c + c) + (c * p_dim + p_dim)
        elif self.architecture in ("X1-R", "X1-R-noedge"):
            s, k, eh = self.slot_dim, self.num_slots, self.slot_edge_hidden
            # tok_ln(2h) + tok_proj(h*s) + slots_init(k*s) + q/k/v(3*s^2) + slot_gru(6*s^2+6*s)
            # + agg_ln(2s) + agg_mlp(4s^2 + 3s) + goal_q_proj(h*s) + read_ln(2s) + r_to_ptr(s*p_dim)
            total += (2 * h) + (h * s) + (k * s) + 3 * (s * s) + (6 * s * s + 6 * s)
            total += (2 * s) + (4 * s * s + 3 * s) + (h * s) + (2 * s) + (s * p_dim)
            if self.architecture == "X1-R":
                # rel_ln(2s) + edge_mlp1(3s*eh + eh) + edge_mlp2(eh + 1) + msg_proj(s^2) + rel_gru(6*s^2 + 6*s)
                total += (2 * s) + (3 * s * eh + eh) + (eh + 1) + (s * s) + (6 * s * s + 6 * s)
            else:
                # noedge_ln(2s) + noedge_mlp(4s^2 + 3s) + noedge_gru(6*s^2 + 6*s)
                total += (2 * s) + (4 * s * s + 3 * s) + (6 * s * s + 6 * s)
        return int(total)

    def canonical(self) -> dict[str, Any]:
        self.assert_valid()
        data = asdict(self)
        data["total_parameters"] = self.total_parameters()
        return data

    def sha256(self) -> str:
        return digest(self.canonical())


class _X1CausalBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn_qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.attn_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp_in = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.mlp_out = nn.Linear(2 * hidden_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        x1 = self.ln1(x)
        qkv = self.attn_qkv(x1).view(b, l, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~attn_mask.unsqueeze(1), float("-inf"))
        weights = F.softmax(scores, dim=-1).masked_fill(~attn_mask.unsqueeze(1), 0.0)
        att = (weights @ v).transpose(1, 2).contiguous().view(b, l, d)
        x = x + self.attn_out(att)
        x = x + self.mlp_out(F.silu(self.mlp_in(self.ln2(x))))
        return x


@dataclass(frozen=True)
class X1ForwardOutput:
    log_probs_19: torch.Tensor
    log_p_op: torch.Tensor
    log_p_val: torch.Tensor
    gate: torch.Tensor
    attention_entropy: torch.Tensor
    has_pointer_candidates: torch.Tensor
    decision_hidden: torch.Tensor


class X1RelationalModel(nn.Module):
    """Independent causal relational-slot + pointer-copy network for X1 R/D2."""

    def __init__(self, spec: X1RD2Spec) -> None:
        super().__init__()
        spec.assert_valid()
        self.x1_spec = spec
        h = spec.hidden_dim
        self.token_emb = nn.Embedding(spec.vocab_size, h)
        self.pos_emb = nn.Embedding(spec.max_sequence_length, h)
        self.blocks = nn.ModuleList([_X1CausalBlock(h, spec.num_heads) for _ in range(spec.num_layers)])
        self.ln_f = nn.LayerNorm(h)
        self.op_head = nn.Linear(h, 4, bias=True)
        self.val_head = nn.Linear(h, 16, bias=True)
        self.q_proj = nn.Linear(h, spec.pointer_dim, bias=False)
        self.k_proj = nn.Linear(h, spec.pointer_dim, bias=False)
        self.gate = nn.Linear(h, 1, bias=True)
        nn.init.constant_(self.gate.bias, -2.0)

        if spec.architecture == "X1-P-capR":
            self.cap_r_in = nn.Linear(h, spec.cap_hidden_dim, bias=True)
            self.cap_r_out = nn.Linear(spec.cap_hidden_dim, spec.pointer_dim, bias=True)
        elif spec.architecture in ("X1-R", "X1-R-noedge"):
            s, k, eh = spec.slot_dim, spec.num_slots, spec.slot_edge_hidden
            self.tok_ln = nn.LayerNorm(h)
            self.tok_proj = nn.Linear(h, s, bias=False)
            self.slots_init = nn.Parameter(torch.randn(k, s) * 0.02)
            self.slot_q = nn.Linear(s, s, bias=False)
            self.slot_k = nn.Linear(s, s, bias=False)
            self.slot_v = nn.Linear(s, s, bias=False)
            self.slot_gru = nn.GRUCell(s, s)
            self.agg_ln = nn.LayerNorm(s)
            self.agg_mlp_1 = nn.Linear(s, 2 * s, bias=True)
            self.agg_mlp_2 = nn.Linear(2 * s, s, bias=True)
            self.goal_q_proj = nn.Linear(h, s, bias=False)
            self.read_ln = nn.LayerNorm(s)
            self.r_to_ptr = nn.Linear(s, spec.pointer_dim, bias=False)
            if spec.architecture == "X1-R":
                self.rel_ln = nn.LayerNorm(s)
                self.edge_mlp_1 = nn.Linear(3 * s, eh, bias=True)
                self.edge_mlp_2 = nn.Linear(eh, 1, bias=True)
                self.msg_proj = nn.Linear(s, s, bias=False)
                self.rel_gru = nn.GRUCell(s, s)
            else:
                self.noedge_ln = nn.LayerNorm(s)
                self.noedge_mlp_1 = nn.Linear(s, 2 * s, bias=True)
                self.noedge_mlp_2 = nn.Linear(2 * s, s, bias=True)
                self.noedge_gru = nn.GRUCell(s, s)

    def forward_hidden_only(self, token_ids: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        b, l = token_ids.shape
        pos = torch.arange(l, device=token_ids.device).unsqueeze(0).expand(b, l)
        x = self.token_emb(token_ids) + self.pos_emb(pos)
        pos_row = torch.arange(l, device=token_ids.device)
        causal = pos_row.unsqueeze(1) >= pos_row.unsqueeze(0)
        same_seg = (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)) & (segment_ids.unsqueeze(1) >= 0)
        attn_mask = causal.unsqueeze(0) & same_seg
        for block in self.blocks:
            x = block(x, attn_mask)
        return self.ln_f(x)

    def forward_action(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        action_positions: torch.Tensor,
        *,
        lesion_relational: bool = False,
    ) -> X1ForwardOutput:
        b, l = token_ids.shape
        b_idx = torch.arange(b, device=token_ids.device, dtype=torch.int64)
        pos_idx = action_positions.to(dtype=torch.int64)
        H = self.forward_hidden_only(token_ids, segment_ids).float()
        h = H[b_idx, pos_idx]  # [B, W]

        log_p_op = F.log_softmax(self.op_head(h), dim=-1)
        z_val = self.val_head(h)

        pos_grid = torch.arange(l, device=token_ids.device).unsqueeze(0)
        same_seg = (segment_ids == segment_ids[b_idx, pos_idx].unsqueeze(1)) & (segment_ids >= 0)
        valid_prefix = (pos_grid <= pos_idx.unsqueeze(1)) & same_seg
        cand_mask = (pos_grid < pos_idx.unsqueeze(1)) & same_seg & (token_ids >= AP_MIN_TOKEN_ID) & (token_ids <= AP_MAX_TOKEN_ID)

        q_ptr = self.q_proj(h)
        arch = self.x1_spec.architecture
        if arch == "X1-P-capR":
            q_ptr = q_ptr + self.cap_r_out(F.silu(self.cap_r_in(h)))
        elif arch in ("X1-R", "X1-R-noedge") and not lesion_relational:
            x_tok = self.tok_proj(self.tok_ln(H))  # [B, L, S]
            k_slots = self.x1_spec.num_slots
            s = self.x1_spec.slot_dim
            slots = self.slots_init.unsqueeze(0).expand(b, k_slots, s).contiguous()
            k_tok = self.slot_k(x_tok)
            v_tok = self.slot_v(x_tok)
            for _ in range(self.x1_spec.agg_iterations):
                q_s = self.slot_q(slots)
                logits = torch.bmm(q_s, k_tok.transpose(1, 2)) / math.sqrt(s)
                # Softmax across slots (dim=1) per token; mask invalid positions only AFTER softmax so padded positions never yield softmax(-inf) -> NaN
                a_ki = F.softmax(logits, dim=1) * valid_prefix.unsqueeze(1).to(logits.dtype)
                denom = a_ki.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                w_ki = a_ki / denom
                u_k = torch.bmm(w_ki, v_tok)
                slots = self.slot_gru(u_k.reshape(b * k_slots, s), slots.reshape(b * k_slots, s)).view(b, k_slots, s)
                slots = slots + self.agg_mlp_2(F.silu(self.agg_mlp_1(self.agg_ln(slots))))
            q_goal = self.goal_q_proj(h)  # [B, S]
            for _ in range(self.x1_spec.msg_iterations):
                if arch == "X1-R":
                    s_ln = self.rel_ln(slots)
                    s_l = s_ln.unsqueeze(1).expand(b, k_slots, k_slots, s)
                    s_k = s_ln.unsqueeze(2).expand(b, k_slots, k_slots, s)
                    q_exp = q_goal.unsqueeze(1).unsqueeze(2).expand(b, k_slots, k_slots, s)
                    e_kl = self.edge_mlp_2(F.silu(self.edge_mlp_1(torch.cat([s_k, s_l, q_exp], dim=-1)))).squeeze(-1)
                    if k_slots > 1:
                        eye = torch.eye(k_slots, dtype=torch.bool, device=slots.device).unsqueeze(0)
                        e_kl = e_kl.masked_fill(eye, float("-inf"))
                    alpha = F.softmax(e_kl, dim=-1)
                    m_k = torch.bmm(alpha, self.msg_proj(slots))
                    slots = self.rel_gru(m_k.reshape(b * k_slots, s), slots.reshape(b * k_slots, s)).view(b, k_slots, s)
                else:
                    up = self.noedge_mlp_2(F.silu(self.noedge_mlp_1(self.noedge_ln(slots))))
                    slots = self.noedge_gru(up.reshape(b * k_slots, s), slots.reshape(b * k_slots, s)).view(b, k_slots, s)
            r_scores = torch.bmm(q_goal.unsqueeze(1), self.read_ln(slots).transpose(1, 2)).squeeze(1) / math.sqrt(s)
            r_alpha = F.softmax(r_scores, dim=-1)
            r_vec = torch.bmm(r_alpha.unsqueeze(1), slots).squeeze(1)
            q_ptr = q_ptr + self.r_to_ptr(r_vec)

        k_all = self.k_proj(H)
        ptr_scores = torch.bmm(q_ptr.unsqueeze(1), k_all.transpose(1, 2)).squeeze(1) / math.sqrt(self.x1_spec.pointer_dim)
        has_cand = cand_mask.any(dim=-1, keepdim=True)
        safe_mask = cand_mask.clone()
        safe_mask[~has_cand.squeeze(-1), 0] = True
        attn = F.softmax(ptr_scores.masked_fill(~safe_mask, float("-inf")), dim=-1) * cand_mask.to(ptr_scores.dtype)
        bins = (token_ids - AP_MIN_TOKEN_ID).clamp(0, 15)
        p_val = F.softmax(z_val, dim=-1)
        p_copy = torch.zeros_like(p_val)
        p_copy.scatter_add_(1, bins, attn)
        g = torch.sigmoid(self.gate(h))
        p_arg = torch.where(has_cand, (1.0 - g) * p_val + g * p_copy, p_val)
        log_p_val = torch.log(p_arg.clamp_min(1e-30))
        log_probs_19 = torch.cat([log_p_op[:, 0:1] + log_p_val, log_p_op[:, 1:]], dim=-1)
        entropy = -(attn * torch.log(attn.clamp_min(1e-30))).sum(dim=-1)
        entropy = torch.where(has_cand.squeeze(-1), entropy, torch.zeros_like(entropy))
        return X1ForwardOutput(
            log_probs_19=log_probs_19,
            log_p_op=log_p_op,
            log_p_val=log_p_val,
            gate=g.squeeze(-1),
            attention_entropy=entropy,
            has_pointer_candidates=has_cand.squeeze(-1),
            decision_hidden=h,
        )


def build_x1_model(spec: X1RD2Spec, init_seed: int) -> X1RelationalModel:
    spec.assert_valid()
    with torch.random.fork_rng():
        torch.manual_seed(init_seed)
        model = X1RelationalModel(spec)
    actual = sum(int(p.numel()) for p in model.parameters())
    if actual != spec.total_parameters():
        raise ValueError(f"X1 parameter mismatch: actual={actual} != spec={spec.total_parameters()}")
    return model


@dataclass(frozen=True, slots=True)
class X1D2Sample:
    """One supervised state inside a D2 group."""

    task_id: str
    group_id: str
    view_kind: str  # canonical | perm | irrel | cf
    tokens: tuple[int, ...]
    eligible: tuple[bool, ...]
    action_position: int
    gold_op: int
    gold_arg: int
    world_id: str
    state_mask: bool = True


@dataclass(frozen=True, slots=True)
class X1D2EqPair:
    """Index pair (within group) for JS equivariance under answer permutation Pi_T."""

    idx_a: int
    idx_b: int
    perm_16: tuple[int, ...]
    eligible: bool = True


@dataclass(frozen=True, slots=True)
class X1D2CfPair:
    """Index pair (within group) for counterfactual answer margin on distinct ANSWER targets."""

    idx_a: int
    idx_b: int
    target_a: int
    target_b: int
    eligible: bool = True


@dataclass(frozen=True, slots=True)
class X1D2Group:
    """Atomic 4-view base-world group preserving joint D2 gradient dependencies."""

    group_id: str
    samples: tuple[X1D2Sample, ...]
    eq_pairs: tuple[X1D2EqPair, ...]
    cf_pairs: tuple[X1D2CfPair, ...]


@dataclass(frozen=True)
class X1RD2TrainingData:
    dataset_name: str
    data_identity: str
    groups: tuple[X1D2Group, ...]
    manifest: dict[str, Any]


def compute_x1_d2_grouped_loss(
    log_probs_19: torch.Tensor,
    log_p_op: torch.Tensor,
    log_p_val: torch.Tensor,
    *,
    gold_ops: torch.Tensor,
    gold_args: torch.Tensor,
    state_mask: torch.Tensor,
    eq_a: torch.Tensor,
    eq_b: torch.Tensor,
    eq_perm_16: torch.Tensor,
    eq_mask: torch.Tensor,
    cf_a: torch.Tensor,
    cf_b: torch.Tensor,
    cf_target_a: torch.Tensor,
    cf_target_b: torch.Tensor,
    cf_mask: torch.Tensor,
    eq_weight: float = 0.0,
    cf_weight: float = 0.0,
    cf_margin: float = 0.5,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute exact X1 D2 grouped loss (CE + JS equivariance + counterfactual margin).

    Reduction and masking semantics:
    - CE denominator: max(sum(state_mask), 1); masked states contribute 0. When all
      states are masked, returns a differentiable zero tensor (0 * sum(log_probs_19)).
    - EQ denominator: max(sum(eq_mask), 1); permutes the 16 ANSWER bins of p(x_a) by
      eq_perm_16 while preserving bins 16..18 (QUERY, RETRIEVE, ABSTAIN), then computes
      symmetric Jensen-Shannon divergence JS(Pi_T p_a || p_b).
    - CF denominator: max(sum(cf_mask), 1); computes conditional value log-ratio margins
      m_a = log p_val(y_a|x_a) - log p_val(y_b|x_a) and symmetric m_b, applying
      0.5 * (relu(cf_margin - m_a) + relu(cf_margin - m_b)).
    """

    anchor_zero = log_probs_19.sum() * 0.0
    sm = state_mask.to(dtype=torch.float32)
    ce_denom = sm.sum()
    op_nll = -log_p_op.float().gather(1, gold_ops.unsqueeze(1)).squeeze(1)
    is_ans = (gold_ops == 0).to(dtype=torch.float32)
    val_nll = -log_p_val.float().gather(1, gold_args.clamp(0, 15).unsqueeze(1)).squeeze(1) * is_ans
    per_state_ce = (op_nll + val_nll) * sm
    if float(ce_denom.item()) > 0.0:
        l_ce = per_state_ce.sum() / ce_denom
    else:
        l_ce = anchor_zero

    em = eq_mask.to(dtype=torch.float32) if eq_mask.numel() else torch.zeros(0, device=log_probs_19.device)
    eq_denom = em.sum() if em.numel() else anchor_zero
    if eq_a.numel() > 0 and float(eq_denom.item()) > 0.0:
        p19 = log_probs_19.float().exp().clamp_min(1e-12)
        pa = p19[eq_a]
        pb = p19[eq_b]
        # Permute first 16 ANSWER bins cleanly via scatter according to eq_perm_16[i, a] = target bin
        pa_ans = torch.zeros_like(pa[:, :16])
        pa_ans.scatter_add_(1, eq_perm_16.long(), pa[:, :16])
        pa_perm = torch.cat([pa_ans, pa[:, 16:]], dim=-1).clamp_min(1e-12)
        pa_perm = pa_perm / pa_perm.sum(dim=-1, keepdim=True)
        pb_norm = pb / pb.sum(dim=-1, keepdim=True)
        mid = 0.5 * (pa_perm + pb_norm)
        kl_a = (pa_perm * (pa_perm.log() - mid.log())).sum(dim=-1)
        kl_b = (pb_norm * (pb_norm.log() - mid.log())).sum(dim=-1)
        js = 0.5 * (kl_a + kl_b)
        l_eq = (js * em).sum() / eq_denom
    else:
        l_eq = anchor_zero

    cm = cf_mask.to(dtype=torch.float32) if cf_mask.numel() else torch.zeros(0, device=log_probs_19.device)
    cf_denom = cm.sum() if cm.numel() else anchor_zero
    if cf_a.numel() > 0 and float(cf_denom.item()) > 0.0:
        lp_val = log_p_val.float()
        ma = lp_val[cf_a, cf_target_a] - lp_val[cf_a, cf_target_b]
        mb = lp_val[cf_b, cf_target_b] - lp_val[cf_b, cf_target_a]
        margin_loss = 0.5 * (F.relu(cf_margin - ma) + F.relu(cf_margin - mb))
        l_cf = (margin_loss * cm).sum() / cf_denom
    else:
        l_cf = anchor_zero

    total = l_ce + float(eq_weight) * l_eq + float(cf_weight) * l_cf
    metrics = {
        "total": float(total.detach().item()),
        "ce": float(l_ce.detach().item()),
        "eq": float(l_eq.detach().item()),
        "cf": float(l_cf.detach().item()),
        "active_states": int(ce_denom.item()),
        "eligible_eq_pairs": int(eq_denom.item()),
        "eligible_cf_pairs": int(cf_denom.item()),
    }
    return total, metrics


def hand_computed_x1_d2_reference(
    log_probs_19: torch.Tensor,
    log_p_op: torch.Tensor,
    log_p_val: torch.Tensor,
    *,
    gold_ops: Sequence[int],
    gold_args: Sequence[int],
    state_mask: Sequence[bool],
    eq_specs: Sequence[tuple[int, int, Sequence[int], bool]],
    cf_specs: Sequence[tuple[int, int, int, int, bool]],
    eq_weight: float = 0.0,
    cf_weight: float = 0.0,
    cf_margin: float = 0.5,
) -> torch.Tensor:
    """Explicit scalar loop reference implementation verifying D2 loss & gradient semantics."""

    active = [i for i, m in enumerate(state_mask) if m]
    if active:
        ce_terms = []
        for i in active:
            op = int(gold_ops[i])
            term = -log_p_op[i, op]
            if op == 0:
                term = term - log_p_val[i, int(gold_args[i])]
            ce_terms.append(term)
        l_ce = torch.stack(ce_terms).sum() / float(len(active))
    else:
        l_ce = log_probs_19.sum() * 0.0

    eq_terms = []
    for a, b, perm, ok in eq_specs:
        if not ok:
            continue
        pa = log_probs_19[a].exp().clamp_min(1e-12)
        pb = log_probs_19[b].exp().clamp_min(1e-12)
        perm_bins = [pa.new_tensor(0.0) for _ in range(16)]
        for src in range(16):
            perm_bins[int(perm[src])] = perm_bins[int(perm[src])] + pa[src]
        pa_p = torch.cat([torch.stack(perm_bins), pa[16:]]).clamp_min(1e-12)
        pa_p = pa_p / pa_p.sum()
        pb_n = pb / pb.sum()
        mid = 0.5 * (pa_p + pb_n)
        js = 0.5 * ((pa_p * (pa_p.log() - mid.log())).sum() + (pb_n * (pb_n.log() - mid.log())).sum())
        eq_terms.append(js)
    l_eq = (torch.stack(eq_terms).sum() / float(len(eq_terms))) if eq_terms else (log_probs_19.sum() * 0.0)

    cf_terms = []
    for a, b, ya, yb, ok in cf_specs:
        if not ok:
            continue
        ma = log_p_val[a, ya] - log_p_val[a, yb]
        mb = log_p_val[b, yb] - log_p_val[b, ya]
        cf_terms.append(0.5 * (F.relu(cf_margin - ma) + F.relu(cf_margin - mb)))
    l_cf = (torch.stack(cf_terms).sum() / float(len(cf_terms))) if cf_terms else (log_probs_19.sum() * 0.0)
    return l_ce + float(eq_weight) * l_eq + float(cf_weight) * l_cf


def _flatten_group_batch(groups: Sequence[X1D2Group], device: torch.device) -> dict[str, Any]:
    samples: list[X1D2Sample] = []
    eq_a, eq_b, eq_perm, eq_mask = [], [], [], []
    cf_a, cf_b, cf_ta, cf_tb, cf_mask = [], [], [], [], []
    for grp in groups:
        base = len(samples)
        samples.extend(grp.samples)
        for ep in grp.eq_pairs:
            eq_a.append(base + ep.idx_a)
            eq_b.append(base + ep.idx_b)
            eq_perm.append(list(ep.perm_16))
            eq_mask.append(bool(ep.eligible))
        for cp in grp.cf_pairs:
            cf_a.append(base + cp.idx_a)
            cf_b.append(base + cp.idx_b)
            cf_ta.append(int(cp.target_a))
            cf_tb.append(int(cp.target_b))
            cf_mask.append(bool(cp.eligible))

    tokens, segments, _ = batch_tensors([(s.tokens, s.eligible) for s in samples], device)
    apos = torch.tensor([s.action_position for s in samples], device=device, dtype=torch.int64)
    gold_ops = torch.tensor([s.gold_op for s in samples], device=device, dtype=torch.int64)
    gold_args = torch.tensor([s.gold_arg for s in samples], device=device, dtype=torch.int64)
    state_mask = torch.tensor([bool(s.state_mask) for s in samples], device=device, dtype=torch.bool)
    return {
        "samples": samples,
        "tokens": tokens,
        "segments": segments,
        "action_pos": apos,
        "gold_ops": gold_ops,
        "gold_args": gold_args,
        "state_mask": state_mask,
        "eq_a": torch.tensor(eq_a, device=device, dtype=torch.int64),
        "eq_b": torch.tensor(eq_b, device=device, dtype=torch.int64),
        "eq_perm_16": torch.tensor(eq_perm, device=device, dtype=torch.int64) if eq_perm else torch.zeros((0, 16), device=device, dtype=torch.int64),
        "eq_mask": torch.tensor(eq_mask, device=device, dtype=torch.bool),
        "cf_a": torch.tensor(cf_a, device=device, dtype=torch.int64),
        "cf_b": torch.tensor(cf_b, device=device, dtype=torch.int64),
        "cf_target_a": torch.tensor(cf_ta, device=device, dtype=torch.int64),
        "cf_target_b": torch.tensor(cf_tb, device=device, dtype=torch.int64),
        "cf_mask": torch.tensor(cf_mask, device=device, dtype=torch.bool),
    }


def _cyclic_shift_world(world: Mapping[str, Any], shift: int) -> dict[str, Any]:
    def _sh(sym: str | None) -> str | None:
        if sym is None or sym not in SYMBOLS:
            return sym
        return SYMBOLS[(SYMBOLS.index(sym) + shift) % len(SYMBOLS)]

    w = deepcopy(dict(world))
    w["world_id"] = f"{world['world_id']}::perm{shift}"
    w["goal_subject"] = _sh(str(w["goal_subject"]))
    for row in w["initial_facts"]:
        row["subject"] = _sh(str(row["subject"]))
        row["value"] = _sh(row["value"])
    for key in ("query_payload", "memory_payload"):
        for row in w["hidden"][key]:
            row["subject"] = _sh(str(row["subject"]))
            row["value"] = _sh(row["value"])
    w["hidden"]["expected"] = _sh(w["hidden"]["expected"])
    return w


def _irrel_world(world: Mapping[str, Any]) -> dict[str, Any]:
    w = deepcopy(dict(world))
    w["world_id"] = f"{world['world_id']}::irrel"
    w["initial_facts"] = list(reversed(w["initial_facts"]))
    return w


def _counterfactual_world(world: Mapping[str, Any]) -> dict[str, Any]:
    w = deepcopy(dict(world))
    w["world_id"] = f"{world['world_id']}::cf"
    exp = w["hidden"]["expected"]
    if isinstance(exp, str) and exp in SYMBOLS:
        alt = SYMBOLS[(SYMBOLS.index(exp) + 1) % len(SYMBOLS)]
        w["hidden"]["expected"] = alt
        for row in list(w["initial_facts"]) + list(w["hidden"]["query_payload"]) + list(w["hidden"]["memory_payload"]):
            if row["value"] == exp:
                row["value"] = alt
    return w


def prepare_x1_rd2_training_data(
    *,
    num_groups: int = 8,
    max_length: int = 640,
    seed: int = 8100,
) -> tuple[X1RD2TrainingData, tuple[dict[str, Any], ...]]:
    """Build deterministic D2 4-view groups stratified across canonical operations (ANSWER/QUERY/RETRIEVE/ABSTAIN)."""

    from anra_v5.elendril.v8.data import build_d1_training_dataset

    d1 = build_d1_training_dataset(seed=seed, quota_per_family=16)
    kernel = X1RD2EnvironmentKernel()
    buckets: dict[str, list[dict[str, Any]]] = {"ANSWER": [], "QUERY": [], "RETRIEVE": [], "ABSTAIN": []}
    for raw_w in d1.worlds:
        op0, _ = kernel.ideal_action(kernel.initial_state(raw_w))
        buckets.setdefault(op0, []).append(dict(raw_w))

    selected_worlds: list[dict[str, Any]] = []
    op_cycle = ("ANSWER", "QUERY", "RETRIEVE", "ABSTAIN")
    while len(selected_worlds) < num_groups:
        progressed = False
        for op_name in op_cycle:
            if buckets[op_name] and len(selected_worlds) < num_groups:
                selected_worlds.append(buckets[op_name].pop(0))
                progressed = True
        if not progressed:
            break
    answer_marker_id = UTF8.special(SpecialToken.ANSWER)

    def _to_sample(w: Mapping[str, Any], gid: str, view_kind: str) -> X1D2Sample:
        st = kernel.initial_state(w)
        obs = kernel.observe(st)
        ideal_op, ideal_arg = kernel.ideal_action(st)
        target_text = ideal_arg if ideal_op == "ANSWER" and ideal_arg else {"QUERY": "Q", "RETRIEVE": "R", "ABSTAIN": "!"}[ideal_op]
        framed = UTF8.frame(obs.rendered_text, target_text, max_length=max_length)
        apos = framed.answer_span[0] - 1
        assert apos >= 0 and framed.ids[apos] == answer_marker_id
        op_idx = _OP_TO_IDX[ideal_op]
        arg_idx = SYMBOLS.index(ideal_arg) if op_idx == 0 and ideal_arg in SYMBOLS else 0
        return X1D2Sample(
            task_id=f"{gid}:{view_kind}",
            group_id=gid,
            view_kind=view_kind,
            tokens=framed.ids,
            eligible=framed.loss_mask,
            action_position=apos,
            gold_op=op_idx,
            gold_arg=arg_idx,
            world_id=str(w["world_id"]),
        )

    perm_shift_1 = tuple((i + 1) % 16 for i in range(16))
    id_perm = tuple(range(16))
    groups: list[X1D2Group] = []
    for idx, base in enumerate(selected_worlds):
        gid = f"x1-d2-g{idx:03d}"
        s_can = _to_sample(base, gid, "canonical")
        s_perm = _to_sample(_cyclic_shift_world(base, 1), gid, "perm")
        s_irrel = _to_sample(_irrel_world(base), gid, "irrel")
        s_cf = _to_sample(_counterfactual_world(base), gid, "cf")
        cf_ok = (s_can.gold_op == 0 and s_cf.gold_op == 0 and s_can.gold_arg != s_cf.gold_arg)
        groups.append(
            X1D2Group(
                group_id=gid,
                samples=(s_can, s_perm, s_irrel, s_cf),
                eq_pairs=(
                    X1D2EqPair(0, 1, perm_shift_1, True),
                    X1D2EqPair(0, 2, id_perm, True),
                ),
                cf_pairs=(X1D2CfPair(0, 3, s_can.gold_arg, s_cf.gold_arg, cf_ok),),
            )
        )

    op_names = {0: "ANSWER", 1: "QUERY", 2: "RETRIEVE", 3: "ABSTAIN"}
    can_op_counts: dict[str, int] = {}
    for g in groups:
        op_label = op_names.get(int(g.samples[0].gold_op), "UNKNOWN")
        can_op_counts[op_label] = can_op_counts.get(op_label, 0) + 1
    manifest = {
        "schema": "anra-x1-d2-data-v1",
        "dataset": "X1-D2",
        "seed": seed,
        "group_count": len(groups),
        "sample_count": sum(len(g.samples) for g in groups),
        "world_ids": [str(w["world_id"]) for w in selected_worlds],
        "sampled_strata": {
            "canonical_operation_counts": dict(sorted(can_op_counts.items())),
            "interactive_claim_eligible": bool(
                can_op_counts.get("ANSWER", 0) > 0
                and (can_op_counts.get("QUERY", 0) + can_op_counts.get("RETRIEVE", 0)) > 0
                and can_op_counts.get("ABSTAIN", 0) > 0
            ),
        },
        "groups_sha256": digest([asdict(g) for g in groups]),
    }
    data = X1RD2TrainingData(
        dataset_name="X1-D2",
        data_identity=digest(manifest),
        groups=tuple(groups),
        manifest=manifest,
    )
    return data, tuple(selected_worlds)


class X1ObservablePolicy:
    """ObservablePolicy implementation for independent X1RelationalModel weights."""

    def __init__(
        self,
        model: X1RelationalModel,
        spec: X1RD2Spec,
        *,
        model_identity: str,
        mode: str = "raw",
    ) -> None:
        if mode not in ("raw", "grammar", "availability-mask", "direct-answer-only"):
            raise ValueError(f"unsupported policy mode: {mode!r}")
        self.model = model
        self.x1_spec = spec
        self.v8_spec = None
        self.model_identity = model_identity
        self.mode = mode
        self.policy_id = f"{model_identity}:{spec.architecture}:{mode}"

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _encode_prompt(self, text: str) -> list[int]:
        ids = [UTF8.bos_id, *UTF8.encode(text), UTF8.special(SpecialToken.ANSWER)]
        if len(ids) > self.x1_spec.max_sequence_length:
            raise ValueError("observation length exceeds X1 max_sequence_length")
        return ids

    def propose_with_telemetry(self, obs: UniversalObservation) -> UniversalAction:
        from anra_v5.elendril.v8.policies import _apply_mode_to_distribution, _index_to_action

        q_avail = bool(obs.available_tools.get("query", False))
        m_avail = bool(obs.available_tools.get("retrieve", False))
        prompt_ids = self._encode_prompt(obs.rendered_text)
        modes = [(m, m.training) for m in self.model.modules()]
        self.model.eval()
        try:
            with torch.inference_mode():
                tokens = torch.tensor([prompt_ids], device=self.device, dtype=torch.int64)
                segments = torch.zeros_like(tokens)
                apos = torch.tensor([len(prompt_ids) - 1], device=self.device, dtype=torch.int64)
                out = self.model.forward_action(tokens, segments, apos)
                probs_19 = out.log_probs_19[0].exp().cpu()
                tel = {
                    "model_calls": 1.0,
                    "evaluated_positions": float(len(prompt_ids)),
                    "pointer_gate": float(out.gate[0].item()),
                    "pointer_attention_entropy": float(out.attention_entropy[0].item()),
                    "has_pointer_candidates": 1.0 if bool(out.has_pointer_candidates[0].item()) else 0.0,
                }
        finally:
            for m, was in modes:
                m.training = was

        raw_idx, _ = _apply_mode_to_distribution(probs_19, mode="raw", query_available=q_avail, memory_available=m_avail)
        raw_op, raw_arg, _ = _index_to_action(raw_idx, query_available=q_avail)
        if self.mode in ("raw", "grammar"):
            exec_idx = raw_idx
            exec_logp = math.log(max(float(probs_19[raw_idx].item()), 1e-30))
        else:
            exec_idx, exec_logp = _apply_mode_to_distribution(
                probs_19, mode=self.mode, query_available=q_avail, memory_available=m_avail
            )
        exec_op, exec_arg, _ = _index_to_action(exec_idx, query_available=q_avail)
        dist = {label: float(probs_19[i].item()) for i, label in enumerate(ACTION_19_LABELS)}
        return UniversalAction(
            operation=exec_op,
            argument=exec_arg,
            log_probability=exec_logp,
            action_distribution=dist,
            mode=self.mode,
            raw_operation=raw_op,
            raw_argument=raw_arg,
            telemetry=tel,
        )

    def extract_representations(self, obs_batch: Sequence[UniversalObservation]) -> dict[str, torch.Tensor]:
        if not obs_batch:
            raise ValueError("obs_batch cannot be empty")
        encoded = [self._encode_prompt(obs.rendered_text) for obs in obs_batch]
        samples = [(tuple(ids), (False,) * len(ids)) for ids in encoded]
        tokens, segments, _ = batch_tensors(samples, self.device)
        apos = torch.tensor([len(ids) - 1 for ids in encoded], device=self.device, dtype=torch.int64)
        modes = [(m, m.training) for m in self.model.modules()]
        self.model.eval()
        try:
            with torch.inference_mode():
                out = self.model.forward_action(tokens, segments, apos)
                return {
                    "decision_hidden": out.decision_hidden.cpu(),
                    "log_probs_19": out.log_probs_19.cpu(),
                    "pointer_gate": out.gate.cpu(),
                    "pointer_entropy": out.attention_entropy.cpu(),
                }
        finally:
            for m, was in modes:
                m.training = was


class X1RD2Trainer:
    """Transactional trainer preserving D2 group ordering and CheckpointStore state."""

    def __init__(
        self,
        groups: Sequence[X1D2Group],
        *,
        x1_spec: X1RD2Spec,
        recipe: X1RD2Recipe | Any,
        order_seed: int,
        source_commit: str,
        root: Path,
        lineage: str,
        data_provenance: Mapping[str, Any],
        device: str = "cpu",
        cpu_canary: bool = False,
        resume: bool = False,
    ) -> None:
        x1_spec.assert_valid()
        recipe.validate()
        if recipe.objective not in X1_OBJECTIVES or recipe.sampling != "uniform":
            raise ValueError(f"unsupported X1 recipe objective/sampling: {(recipe.objective, recipe.sampling)!r}")
        if not groups:
            raise ValueError("X1RD2Trainer requires at least one X1D2Group")
        self.groups = tuple(groups)
        self.x1_spec = x1_spec
        self.recipe = recipe
        self.order_seed = order_seed
        self.device = torch.device(device)
        total_samples = sum(len(g.samples) for g in self.groups)
        if self.device.type == "cpu" and (not cpu_canary or total_samples > 32 or x1_spec.total_parameters() > 100_000 or recipe.total_updates > 8):
            raise ValueError("CPU canary limits exceeded for X1RD2Trainer")

        self.eq_weight = float(getattr(recipe, "eq_weight", 0.1)) if recipe.objective in ("pair_eq", "pair_eq_cf") else 0.0
        self.cf_weight = float(getattr(recipe, "cf_weight", 0.1)) if recipe.objective == "pair_eq_cf" else 0.0
        self.cf_margin = float(getattr(recipe, "cf_margin", 0.5))
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(recipe.seed)
        self.model = build_x1_model(x1_spec, recipe.seed).to(self.device)
        self.optimizer = build_adamw_optimizer(self.model, lr=recipe.lr, torch_module=torch)

        # Deterministic permutation order over D2 base groups across epochs
        self.group_plan: list[tuple[int, ...]] = []
        groups_per_step = max(1, recipe.batch_size // 4)
        n_g = len(self.groups)
        epoch_cache: dict[int, list[int]] = {}
        for u in range(recipe.total_updates):
            chosen: list[int] = []
            for slot in range(groups_per_step):
                g_pos = u * groups_per_step + slot
                epoch, offset = divmod(g_pos, n_g)
                if epoch not in epoch_cache:
                    idxs = list(range(n_g))
                    gen = torch.Generator(device="cpu")
                    gen.manual_seed(int(digest(["x1-group-order", order_seed, epoch]), 16) % (2**32))
                    perm = torch.randperm(n_g, generator=gen).tolist()
                    epoch_cache[epoch] = [idxs[p] for p in perm]
                chosen.append(epoch_cache[epoch][offset])
            self.group_plan.append(tuple(chosen))

        self.counts = tuple(
            sum(len(self.groups[gi].samples[j].tokens) for gi in g_idxs for j in range(len(self.groups[gi].samples)))
            for g_idxs in self.group_plan
        )
        self.data_sha = digest({"groups": [asdict(g) for g in self.groups], "provenance": dict(data_provenance)})
        self.runtime = runtime_identity(self.device)
        self.metadata = json.loads(
            canonical({
                "schema": "anra-x1-rd2-training-v1",
                "backend_id": "x1_rd2",
                "recipe": asdict(recipe),
                "order_seed": order_seed,
                "x1_model_spec": x1_spec.canonical(),
                "model_spec_sha256": x1_spec.sha256(),
                "group_plan_sha256": digest(self.group_plan),
                "tokenizer": UTF8.manifest.record(),
                "provenance": dict(data_provenance),
                "runtime": self.runtime,
            })
        )
        identities = IdentityBindings(
            IDENTITY_SCHEMA,
            source_commit,
            x1_spec.sha256(),
            UTF8.identity,
            self.data_sha,
            self.data_sha,
            digest(self.metadata),
            optimizer_group_receipt(self.model, self.optimizer)["sha256"],
            digest([recipe.schedule, recipe.lr, recipe.total_updates]),
            digest(["x1-group-uniform", order_seed]),
            sampler_spec_sha256=digest(self.group_plan),
        )
        self.store = CheckpointStore(Path(root), lineage, payload_validator=self._validate_payloads)
        self.parent = self.store.latest_sha256()
        self.state = TrainingState.initial(
            lineage_id=lineage,
            token_budget=sum(self.counts),
            tokens_per_update=max(self.counts),
            cursor=CursorState(CURSOR_SCHEMA, self.data_sha, 0, 0, 0),
            rng_state_sha256=sha256(self.rng_bytes()).hexdigest(),
            curriculum_phase=f"x1-{x1_spec.architecture}-{recipe.objective}",
            identities=identities,
        )
        self.faulted = False
        self.published_update = 0
        if resume:
            if self.parent is None:
                raise ValueError("no committed X1 checkpoint to resume")
            state, payloads = self.store.restore(self.parent)
            self.model.load_state_dict(_load(payloads["model.bin"]), strict=True)
            self.optimizer.load_state_dict(_load(payloads["optimizer.bin"]))
            self.rng.set_state(torch.tensor(list(payloads["rng.bin"]), dtype=torch.uint8))
            self.state = state
            self.published_update = state.global_update
        elif self.parent is not None:
            raise ValueError("existing X1 lineage requires explicit resume=True")

    def rng_bytes(self) -> bytes:
        return bytes(self.rng.get_state().tolist())

    def next_group_ids(self) -> tuple[str, ...]:
        if self.state.global_update >= len(self.group_plan):
            return ()
        return tuple(self.groups[i].group_id for i in self.group_plan[self.state.global_update])

    def learning_rate(self, update: int) -> float:
        if self.recipe.schedule == "constant":
            return float(self.recipe.lr)
        return float(
            self.recipe.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * update / self.recipe.total_updates)))
        )

    def _validate_payloads(self, state: TrainingState, payloads: dict[str, bytes]) -> None:
        if state.identities != self.state.identities:
            raise ValueError("X1 checkpoint identity mismatch")
        sched = json.loads(payloads["scheduler.json"])
        if sched.get("metadata") != self.metadata:
            raise ValueError("X1 checkpoint metadata mismatch")
        if sha256(payloads["rng.bin"]).hexdigest() != state.rng_state_sha256:
            raise ValueError("X1 checkpoint RNG digest mismatch")

    def step(self) -> dict[str, Any]:
        if self.faulted or self.state.complete:
            raise ValueError("complete or faulted X1 trainer requires restore")
        self.faulted = True
        u = self.state.global_update
        grp_idxs = self.group_plan[u]
        chosen_groups = [self.groups[i] for i in grp_idxs]
        batch = _flatten_group_batch(chosen_groups, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        lr = self.learning_rate(u)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        out = self.model.forward_action(batch["tokens"], batch["segments"], batch["action_pos"])
        loss, metrics = compute_x1_d2_grouped_loss(
            out.log_probs_19,
            out.log_p_op,
            out.log_p_val,
            gold_ops=batch["gold_ops"],
            gold_args=batch["gold_args"],
            state_mask=batch["state_mask"],
            eq_a=batch["eq_a"],
            eq_b=batch["eq_b"],
            eq_perm_16=batch["eq_perm_16"],
            eq_mask=batch["eq_mask"],
            cf_a=batch["cf_a"],
            cf_b=batch["cf_b"],
            cf_target_a=batch["cf_target_a"],
            cf_target_b=batch["cf_target_b"],
            cf_mask=batch["cf_mask"],
            eq_weight=self.eq_weight,
            cf_weight=self.cf_weight,
            cf_margin=self.cf_margin,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite X1 grouped loss")
        loss.backward()
        validate_parameter_ownership(self.model, self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.recipe.clip, error_if_nonfinite=True)
        self.optimizer.step()

        step_tokens = self.counts[u]
        next_cursor = replace(self.state.cursor, sequence_ordinal=u + 1)
        self.state = replace(
            self.state,
            generation=u + 1,
            global_update=u + 1,
            cumulative_tokens=self.state.cumulative_tokens + step_tokens,
            tokens_by_source={"x1_d2_groups": self.state.cumulative_tokens + step_tokens},
            optimizer_step_max=u + 1,
            schedule_tokens=self.state.cumulative_tokens + step_tokens,
            cursor=next_cursor,
            rng_state_sha256=sha256(self.rng_bytes()).hexdigest(),
            parent_checkpoint_sha256=self.parent,
        )
        self.state.assert_valid()
        self.faulted = False
        return {
            "update": u + 1,
            "loss": metrics["total"],
            "ce_loss": metrics["ce"],
            "eq_loss": metrics["eq"],
            "cf_loss": metrics["cf"],
            "pre_clip": float(norm),
            "lr": lr,
            "group_ids": [g.group_id for g in chosen_groups],
        }

    def save(self, *, inject_crash_at: str | None = None) -> str:
        if self.faulted:
            raise ValueError("cannot checkpoint faulted X1 trainer")
        self.state.assert_valid()
        payloads = {
            "model.bin": _dump({k: v.detach().cpu() for k, v in self.model.state_dict().items()}),
            "optimizer.bin": _dump(self.optimizer.state_dict()),
            "scheduler.json": json.dumps({"metadata": self.metadata, "scaler": {}}, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            "rng.bin": self.rng_bytes(),
            "cursor.json": json.dumps(asdict(self.state.cursor), sort_keys=True, separators=(",", ":")).encode("utf-8"),
            "ledger.json": json.dumps(dict(self.state.tokens_by_source), sort_keys=True, separators=(",", ":")).encode("utf-8"),
            "training_state.json": json.dumps(self.state.canonical(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
        }
        sha = self.store.publish(
            state=self.state,
            payloads=payloads,
            expected_parent_sha256=self.parent,
            inject_crash_at=inject_crash_at,
        )
        self.parent = sha
        self.published_update = self.state.global_update
        return sha


class X1RD2Backend:
    """Independent non-V8 PlanthirBackend for X1 R/D2 architectures and grouped D2 losses."""

    backend_id = "x1_rd2"
    backend_version = "anra-x1-rd2-1.0"
    contract_version = BACKEND_CONTRACT_VERSION

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id=self.backend_id,
            backend_version=self.backend_version,
            contract_version=self.contract_version,
            supported_architectures=X1_ARCHITECTURES,
            supported_objectives=X1_OBJECTIVES,
            supported_samplings=("uniform",),
            supported_devices=("cpu", "cuda"),
            supports_grouped_loss=True,
            supports_resume=True,
            supports_fp16=True,
        )

    def resolve_model_spec(self, arm: Any) -> X1RD2Spec:
        spec = getattr(arm, "model_spec", None)
        if not isinstance(spec, X1RD2Spec):
            raise ValueError("X1 R/D2 backend requires arm.model_spec to be an X1RD2Spec")
        spec.assert_valid()
        return spec

    def canonical_model_spec(self, model_spec: Any) -> dict[str, Any]:
        if not isinstance(model_spec, X1RD2Spec):
            raise ValueError("expected X1RD2Spec")
        return model_spec.canonical()

    def model_parameter_count(self, model_spec: Any) -> int:
        if not isinstance(model_spec, X1RD2Spec):
            raise ValueError("expected X1RD2Spec")
        return model_spec.total_parameters()

    def canonical_factor_config(self, arm: Any) -> dict[str, Any]:
        return self.resolve_model_spec(arm).canonical()

    def validate_kernel(self, kernel: Any) -> None:
        from ..contracts import EnvironmentKernel

        if not isinstance(kernel, EnvironmentKernel):
            raise ValueError("X1 R/D2 requires a valid EnvironmentKernel")

    def validate_arm(self, arm: Any, train_data: Any, *, device: str, cpu_canary: bool) -> dict[str, Any]:
        caps = self.capabilities()
        if getattr(arm, "trainer_factory", None) is not None:
            raise ValueError("custom trainer_factory is unsupported on qualified X1 backend")
        spec = self.resolve_model_spec(arm)
        if arm.architecture_family != spec.architecture:
            raise ValueError("arm.architecture_family differs from X1RD2Spec.architecture")
        if spec.architecture not in caps.supported_architectures:
            raise ValueError(f"unsupported X1 architecture: {spec.architecture!r}")
        arm.recipe.validate()
        if arm.recipe.objective not in caps.supported_objectives or arm.recipe.sampling not in caps.supported_samplings:
            raise ValueError(f"unsupported X1 recipe objective/sampling: {(arm.recipe.objective, arm.recipe.sampling)!r}")
        if not isinstance(train_data, X1RD2TrainingData) or arm.dataset_identity != train_data.data_identity:
            raise ValueError("X1 arm dataset identity or corpus schema mismatch")
        if any(type(u) is not int or not 1 <= u <= arm.recipe.total_updates for u in arm.milestone_updates):
            raise ValueError("milestone updates must fall inside total_updates")
        total_samples = sum(len(g.samples) for g in train_data.groups)
        if torch.device(device).type == "cpu" and (
            not cpu_canary
            or total_samples > caps.max_cpu_canary_samples
            or spec.total_parameters() > caps.max_cpu_canary_params
            or arm.recipe.total_updates > caps.max_cpu_canary_updates
        ):
            raise ValueError("CPU canary limits exceeded for X1 R/D2 arm")
        return {
            "backend_id": self.backend_id,
            "architecture": spec.architecture,
            "parameter_count": spec.total_parameters(),
            "sample_count": total_samples,
        }

    def validate_dataset_and_worlds(
        self,
        train_data: Any,
        train_worlds: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(train_data, X1RD2TrainingData):
            raise ValueError("X1 backend requires X1RD2TrainingData")
        recomputed = digest([asdict(g) for g in train_data.groups])
        if train_data.manifest.get("groups_sha256") != recomputed or digest(train_data.manifest) != train_data.data_identity:
            raise ValueError("X1RD2TrainingData manifest checksum mismatch")
        if not train_worlds:
            raise ValueError("X1 backend requires explicit train_worlds for disjointness checks")
        return {
            "dataset_name": train_data.dataset_name,
            "data_identity": train_data.data_identity,
            "content_sha256": recomputed,
            "world_count": len(train_worlds),
            "resolved_train_worlds": tuple(train_worlds),
        }

    def create_trainer(
        self,
        arm: Any,
        train_data: Any,
        *,
        source_commit: str,
        root: Path,
        lineage: str,
        device: str,
        cpu_canary: bool,
        resume: bool,
    ) -> X1RD2Trainer:
        spec = self.resolve_model_spec(arm)
        arm_resume = resume and (root / lineage / "LATEST").is_file()
        return X1RD2Trainer(
            train_data.groups,
            x1_spec=spec,
            recipe=arm.recipe,
            order_seed=arm.order_seed,
            source_commit=source_commit,
            root=root,
            lineage=lineage,
            data_provenance=train_data.manifest,
            device=device,
            cpu_canary=(torch.device(device).type == "cpu" and cpu_canary),
            resume=arm_resume,
        )

    def sample_gradient_loss_fn(
        self,
        trainer: X1RD2Trainer,
        arm: Any,
        train_data: X1RD2TrainingData,
        device: torch.device,
    ) -> Callable[[nn.Module], torch.Tensor]:
        batch = _flatten_group_batch(train_data.groups[:1], device)
        eq_w = float(getattr(arm.recipe, "eq_weight", 0.1)) if arm.recipe.objective in ("pair_eq", "pair_eq_cf") else 0.0
        cf_w = float(getattr(arm.recipe, "cf_weight", 0.1)) if arm.recipe.objective == "pair_eq_cf" else 0.0
        cf_m = float(getattr(arm.recipe, "cf_margin", 0.5))

        def _loss(m: X1RelationalModel) -> torch.Tensor:
            out = m.forward_action(batch["tokens"], batch["segments"], batch["action_pos"])
            total, _ = compute_x1_d2_grouped_loss(
                out.log_probs_19,
                out.log_p_op,
                out.log_p_val,
                gold_ops=batch["gold_ops"],
                gold_args=batch["gold_args"],
                state_mask=batch["state_mask"],
                eq_a=batch["eq_a"],
                eq_b=batch["eq_b"],
                eq_perm_16=batch["eq_perm_16"],
                eq_mask=batch["eq_mask"],
                cf_a=batch["cf_a"],
                cf_b=batch["cf_b"],
                cf_target_a=batch["cf_target_a"],
                cf_target_b=batch["cf_target_b"],
                cf_mask=batch["cf_mask"],
                eq_weight=eq_w,
                cf_weight=cf_w,
                cf_margin=cf_m,
            )
            return total

        return _loss

    def create_policy(
        self,
        model: Any,
        model_spec: Any,
        *,
        model_identity: str,
        mode: str = "raw",
        copy_enabled_override: bool | None = None,
    ) -> X1ObservablePolicy:
        if not isinstance(model_spec, X1RD2Spec):
            raise ValueError("expected X1RD2Spec")
        return X1ObservablePolicy(model, model_spec, model_identity=model_identity, mode=mode)


@dataclass(frozen=True, slots=True)
class X1RD2KernelState:
    """Immutable public state for the independent X1 relational-graph kernel."""

    world: dict[str, Any]
    public_facts: tuple[dict[str, Any], ...]
    query_avail: bool
    mem_avail: bool
    consumed_tools: frozenset[str]
    decision: int
    max_actions: int
    executed_actions: tuple[tuple[str, str | None], ...]
    terminal: bool = False


class X1RD2EnvironmentKernel:
    """Non-V8 relational-graph EnvironmentKernel for X1 R/D2 worlds and D1/D1-H style tasks."""

    kernel_id = "anra-x1-rd2-kernel-v1"

    def __init__(self, *, max_actions: int = 4) -> None:
        if type(max_actions) is not int or max_actions < 1:
            raise ValueError("max_actions must be a positive integer")
        self.max_actions = max_actions

    def canonical_config(self) -> dict[str, Any]:
        return {"kernel_id": self.kernel_id, "max_actions": self.max_actions}

    def initial_state(self, world: Mapping[str, Any]) -> X1RD2KernelState:
        tools = world.get("allowed_public_tools", {})
        return X1RD2KernelState(
            world=deepcopy(dict(world)),
            public_facts=tuple(dict(r) for r in world["initial_facts"]),
            query_avail=bool(tools.get("query_available", False)),
            mem_avail=bool(tools.get("memory_available", False)),
            consumed_tools=frozenset(),
            decision=0,
            max_actions=self.max_actions,
            executed_actions=(),
            terminal=False,
        )

    @staticmethod
    def _resolve(facts: Sequence[Mapping[str, Any]], goal: str, hops: int) -> tuple[bool, str | None]:
        # Strictly deterministic public precedence: highest (tick, source_rank) per subject
        best: dict[str, tuple[int, str, str | None]] = {}
        for row in facts:
            subj = str(row["subject"])
            tick = int(row.get("tick", 0))
            src = str(row.get("source", ""))
            cand = (tick, src, row.get("value"))
            if subj not in best or cand[:2] > best[subj][:2]:
                best[subj] = cand
        cur = goal
        for _ in range(max(1, int(hops))):
            if cur not in best:
                return False, None
            val = best[cur][2]
            if val is None:
                return False, None
            cur = str(val)
        return True, cur

    def observe(self, state: X1RD2KernelState) -> UniversalObservation:
        w = state.world
        facts_str = "; ".join(
            f"{r['subject']}->{('null' if r['value'] is None else r['value'])}@{r.get('tick', 0)}/{r.get('source', 's0')}"
            for r in state.public_facts
        )
        text = (
            f"X1 goal={w['goal_subject']} hops={w['hops']} "
            f"query={int(state.query_avail)} retrieve={int(state.mem_avail)} "
            f"facts=[{facts_str}] Action:"
        )
        obs_id = digest(["x1-obs", state.decision, text, state.query_avail, state.mem_avail])[:16]
        return UniversalObservation(
            obs_id=obs_id,
            rendered_text=text,
            available_tools={"query": state.query_avail, "retrieve": state.mem_avail},
            consumed_tools=state.consumed_tools,
            decision_index=state.decision,
            budget_remaining=max(1, state.max_actions - state.decision),
            metadata={"hops": int(w["hops"]), "kernel": self.kernel_id},
        )

    def ideal_action(self, state: X1RD2KernelState) -> tuple[str, str | None]:
        ok, val = self._resolve(state.public_facts, str(state.world["goal_subject"]), int(state.world["hops"]))
        if ok and val in SYMBOLS:
            return ("ANSWER", val)
        if state.mem_avail:
            return ("RETRIEVE", None)
        if state.query_avail:
            return ("QUERY", None)
        return ("ABSTAIN", None)

    def step(self, state: X1RD2KernelState, operation: str, argument: str | None) -> X1RD2KernelState:
        if state.terminal:
            raise ValueError("cannot step terminal X1RD2KernelState")
        facts = list(state.public_facts)
        q_avail = state.query_avail
        m_avail = state.mem_avail
        consumed = set(state.consumed_tools)
        executed = (*state.executed_actions, (operation, argument))
        next_dec = state.decision + 1
        terminal = next_dec >= state.max_actions

        if operation == "QUERY":
            if not q_avail:
                terminal = True
            else:
                consumed.add("query")
                q_avail = False
                for r in state.world.get("hidden", {}).get("query_payload", []):
                    facts.append(dict(r))
        elif operation == "RETRIEVE":
            if not m_avail:
                terminal = True
            else:
                consumed.add("retrieve")
                m_avail = False
                for r in state.world.get("hidden", {}).get("memory_payload", []):
                    facts.append(dict(r))
        else:
            terminal = True

        return X1RD2KernelState(
            world=state.world,
            public_facts=tuple(facts),
            query_avail=q_avail,
            mem_avail=m_avail,
            consumed_tools=frozenset(consumed),
            decision=next_dec,
            max_actions=state.max_actions,
            executed_actions=executed,
            terminal=terminal,
        )

    def verify(self, world: Mapping[str, Any], actions: Sequence[tuple[str, str | None]]) -> dict[str, Any]:
        st = self.initial_state(world)
        for op, arg in actions:
            if st.terminal:
                return {"correct": False, "terminal_status": "OVER_BUDGET"}
            st = self.step(st, op, arg)
        last_op, last_arg = actions[-1] if actions else ("INVALID", None)
        expected = world.get("hidden", {}).get("expected")
        if expected is None:
            ok = (last_op == "ABSTAIN" and last_arg is None)
        else:
            pub_ok, pub_val = self._resolve(st.public_facts, str(world["goal_subject"]), int(world["hops"]))
            ok = (last_op == "ANSWER" and last_arg == expected and pub_ok and pub_val == expected)
        return {"correct": bool(ok), "terminal_status": "OK" if ok else "WRONG"}

    def inject_oracle_information(self, world: Mapping[str, Any]) -> dict[str, Any]:
        w = deepcopy(dict(world))
        merged = list(w["initial_facts"])
        if w["allowed_public_tools"].get("query_available"):
            merged.extend(dict(r) for r in w["hidden"].get("query_payload", []))
        if w["allowed_public_tools"].get("memory_available"):
            merged.extend(dict(r) for r in w["hidden"].get("memory_payload", []))
        w["initial_facts"] = merged
        w["allowed_public_tools"] = {"query_available": False, "memory_available": False}
        w["hidden"]["query_payload"] = []
        w["hidden"]["memory_payload"] = []
        w["world_id"] = f"{world['world_id']}::oracle_injected"
        return w

    def decompose_subgoals(self, world: Mapping[str, Any]) -> tuple[SubgoalProbeSpec, ...]:
        hops = int(world.get("hops", 1))
        exp = world.get("hidden", {}).get("expected")
        if hops < 2 or exp is None:
            return ()
        ow = self.inject_oracle_information(world)
        goal = str(world["goal_subject"])
        ok1, mid = self._resolve(ow["initial_facts"], goal, 1)
        if not ok1 or mid is None:
            return ()
        ok2, end_v = self._resolve(ow["initial_facts"], mid, hops - 1)
        if not ok2 or end_v != exp:
            return ()
        h1 = deepcopy(ow)
        h1["world_id"] = f"{world['world_id']}::hop1"
        h1["hops"] = 1
        h1["goal_subject"] = goal
        h1["hidden"]["expected"] = mid
        h2 = deepcopy(ow)
        h2["world_id"] = f"{world['world_id']}::hop2_bridge"
        h2["hops"] = hops - 1
        h2["goal_subject"] = mid
        h2["hidden"]["expected"] = end_v
        return (
            SubgoalProbeSpec(f"{world['world_id']}::sg1", "hop_1", 1, h1, "ANSWER", mid, "X1 hop 1"),
            SubgoalProbeSpec(f"{world['world_id']}::sg2", "hop_2_given_bridge", 2, h2, "ANSWER", end_v, "X1 hop 2"),
        )


__all__ = [
    "X1D2CfPair",
    "X1D2EqPair",
    "X1D2Group",
    "X1D2Sample",
    "X1ForwardOutput",
    "X1ObservablePolicy",
    "X1RD2Backend",
    "X1RD2EnvironmentKernel",
    "X1RD2KernelState",
    "X1RD2Recipe",
    "X1RD2Spec",
    "X1RD2Trainer",
    "X1RD2TrainingData",
    "X1RelationalModel",
    "X1_ARCHITECTURES",
    "X1_OBJECTIVES",
    "build_x1_model",
    "compute_x1_d2_grouped_loss",
    "hand_computed_x1_d2_reference",
    "prepare_x1_rd2_training_data",
]
