"""Elendril V8-X2 S20-backed models, exact delta-memory update, active controls, and surgical lesions."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
import math
from typing import Any, Iterator, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from anra_v5.elendril.codec import digest
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from v5_contracts.model_spec import ModelSpec
from v5_model.core import initialize, packed_layout
from v5_model.initialize import NORMAL_STD

AP_MIN_TOKEN_ID = UTF8.encode("A")[0]
AP_MAX_TOKEN_ID = UTF8.encode("P")[0]
OP_TO_INDEX = {"ANSWER": 0, "QUERY": 1, "RETRIEVE": 2, "ABSTAIN": 3}

X2_ARCHITECTURES = (
    "P",
    "R",
    "P2",
    "P2-dense",
    "R2",
    "R2-pool",
    "U-P",
    "U-P-par",
    "DM",
    "DM-add",
    "PS",
    "PS-ctrl",
)
X2_NEW_PAIRS = (
    ("P2", "P2-dense"),
    ("R2", "R2-pool"),
    ("U-P", "U-P-par"),
    ("DM", "DM-add"),
    ("PS", "PS-ctrl"),
)
X2_MAX_S20_HEAD_OVERHEAD = 0.05
X2_ACTIVE_CONTROL_PARAM_TOL = 0.02
X2_MAX_SEQUENCE = 640


def build_s20_base_spec(context_length: int = X2_MAX_SEQUENCE) -> ModelSpec:
    """Return the exact frozen V5 S20 backbone (19,871,616 parameters, 7 layers, width 512)."""
    base = MODEL_PROFILES["S20"]
    return ModelSpec(
        schema=base.schema,
        family=base.family,
        vocabulary_size=base.vocabulary_size,
        width=base.width,
        layers=base.layers,
        query_heads=base.query_heads,
        kv_heads=base.kv_heads,
        head_dimension=base.head_dimension,
        ffn_width=base.ffn_width,
        context_length=int(context_length),
        rope_base=base.rope_base,
        norm_epsilon=base.norm_epsilon,
        tied_embeddings=base.tied_embeddings,
        qk_norm=base.qk_norm,
        qk_norm_affine=base.qk_norm_affine,
        linear_bias=base.linear_bias,
        dropout=base.dropout,
    )


def build_tiny_canary_base_spec(
    *,
    width: int = 16,
    layers: int = 1,
    query_heads: int = 2,
    kv_heads: int = 1,
    head_dimension: int = 8,
    ffn_width: int = 32,
    context_length: int = X2_MAX_SEQUENCE,
) -> ModelSpec:
    """Return a tiny CPU-canary backbone (<=100k params) sharing exact V5 core architecture."""
    return ModelSpec(
        schema="anra-v5-model-spec/v1",
        family="dense-decoder-transformer",
        vocabulary_size=UTF8.vocabulary_size,
        width=int(width),
        layers=int(layers),
        query_heads=int(query_heads),
        kv_heads=int(kv_heads),
        head_dimension=int(head_dimension),
        ffn_width=int(ffn_width),
        context_length=int(context_length),
        rope_base=10_000.0,
        norm_epsilon=1e-5,
        tied_embeddings=True,
        qk_norm=True,
        qk_norm_affine=True,
        linear_bias=False,
        dropout=0.0,
    )


def delta_memory_step(
    M: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    *,
    additive: bool = False,
) -> torch.Tensor:
    """Exact differentiable fast-weight memory update (§4, §11).

    Delta mode (`DM`):     M <- M + beta * (v - M k) k^T
    Additive mode (`DM-add`): M <- M + beta * v k^T
    where k is L2-normalized along its last dimension.
    Shapes: M (..., D, D), k (..., D), v (..., D), beta (..., 1) or scalar.
    """
    k_norm = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    beta_t = torch.as_tensor(beta, dtype=M.dtype, device=M.device)
    while beta_t.ndim < M.ndim:
        beta_t = beta_t.unsqueeze(-1)
    if additive:
        err = v
    else:
        mk = torch.matmul(M, k_norm.unsqueeze(-1)).squeeze(-1)
        err = v - mk
    outer = err.unsqueeze(-1) * k_norm.unsqueeze(-2)
    return M + beta_t * outer


def delta_memory_read(M: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Read vector M q for normalized or raw query q (..., D)."""
    return torch.matmul(M, q.unsqueeze(-1)).squeeze(-1)


def _init_linear(linear: nn.Linear, *, gen: torch.Generator | None = None, bias_value: float = 0.0) -> None:
    with torch.no_grad():
        linear.weight.normal_(mean=0.0, std=NORMAL_STD, generator=gen)
        if linear.bias is not None:
            linear.bias.fill_(bias_value)


def _init_gru_cell(gru: nn.GRUCell, *, gen: torch.Generator | None = None) -> None:
    with torch.no_grad():
        gru.weight_ih.normal_(mean=0.0, std=NORMAL_STD, generator=gen)
        gru.weight_hh.normal_(mean=0.0, std=NORMAL_STD, generator=gen)
        if gru.bias_ih is not None:
            gru.bias_ih.zero_()
        if gru.bias_hh is not None:
            gru.bias_hh.zero_()


@dataclass(frozen=True)
class X2ModelSpec:
    """Versioned Elendril V8-X2 model specification with exact parameter accounting."""

    architecture: str
    base_spec: ModelSpec
    pointer_dim: int = 64
    num_roles: int = 8
    dense_hidden: int = 128
    slot_dim: int = 96
    num_slots: int = 8
    slot_edge_hidden: int = 64
    pool_hidden: int = 256
    u_dim: int = 128
    u_steps: int = 4
    u_par_hidden: int = 128
    mem_dim: int = 64
    forecast_hidden: int = 128
    forecast_branches: int = 4
    forecast_classes: int = 4
    schema: str = "elendril-x2-model-spec-v1"

    def assert_valid(self) -> None:
        self.base_spec.assert_valid()
        if self.architecture not in X2_ARCHITECTURES:
            raise ValueError(f"unsupported X2 architecture {self.architecture!r}; expected one of {X2_ARCHITECTURES}")
        for name in (
            "pointer_dim",
            "num_roles",
            "dense_hidden",
            "slot_dim",
            "num_slots",
            "slot_edge_hidden",
            "pool_hidden",
            "u_dim",
            "u_steps",
            "u_par_hidden",
            "mem_dim",
            "forecast_hidden",
            "forecast_branches",
            "forecast_classes",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")

    def canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "architecture": self.architecture,
            "base_spec": asdict(self.base_spec),
            "pointer_dim": int(self.pointer_dim),
            "num_roles": int(self.num_roles),
            "dense_hidden": int(self.dense_hidden),
            "slot_dim": int(self.slot_dim),
            "num_slots": int(self.num_slots),
            "slot_edge_hidden": int(self.slot_edge_hidden),
            "pool_hidden": int(self.pool_hidden),
            "u_dim": int(self.u_dim),
            "u_steps": int(self.u_steps),
            "u_par_hidden": int(self.u_par_hidden),
            "mem_dim": int(self.mem_dim),
            "forecast_hidden": int(self.forecast_hidden),
            "forecast_branches": int(self.forecast_branches),
            "forecast_classes": int(self.forecast_classes),
        }

    def sha256(self) -> str:
        return digest(self.canonical())


@dataclass(frozen=True)
class X2ForwardOutput:
    """Structured forward output with separate op, conditional val, provisional belief, and telemetry."""

    log_probs_19: torch.Tensor
    log_p_op: torch.Tensor
    log_p_val: torch.Tensor
    provisional_val_logits: torch.Tensor
    certifiable_logit: torch.Tensor
    decision_hidden: torch.Tensor
    gate: torch.Tensor | None = None
    forecast_logits: torch.Tensor | None = None
    forecast_vector: torch.Tensor | None = None
    memory_read_vec: torch.Tensor | None = None
    memory_state: torch.Tensor | None = None
    u_step_states: tuple[torch.Tensor, ...] = ()
    slot_states: torch.Tensor | None = None


class X2TypedModel(nn.Module):
    """S20-backed X2 model supporting all 5 isolated pairs + P/R baselines, lesions, and episodic state."""

    def __init__(self, core: nn.Module, x2_spec: X2ModelSpec, *, head_seed: int = 0) -> None:
        super().__init__()
        x2_spec.assert_valid()
        self.core = core
        self.x2_spec = x2_spec
        w = int(x2_spec.base_spec.width)
        d = int(x2_spec.pointer_dim)
        arch = x2_spec.architecture

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(head_seed) + 10_003)

        # Shared typed interface + provisional belief head (§4, §7, §11)
        self.op_head = nn.Linear(w, 4, bias=True)
        self.val_head = nn.Linear(w, 16, bias=True)
        self.q_proj = nn.Linear(w, d, bias=False)
        self.k_proj = nn.Linear(w, d, bias=False)
        self.gate = nn.Linear(w, 1, bias=True)
        self.provisional_val_head = nn.Linear(w, 16, bias=True)
        self.provisional_cert_head = nn.Linear(w, 1, bias=True)
        for lin in (self.op_head, self.val_head, self.q_proj, self.k_proj, self.provisional_val_head, self.provisional_cert_head):
            _init_linear(lin, gen=gen)
        _init_linear(self.gate, gen=gen, bias_value=-1.0)

        # Architecture-specific isolated mechanism or matched active control
        if arch == "P2":
            nr = int(x2_spec.num_roles)
            self.role_head = nn.Linear(w, nr, bias=True)
            self.role_q_proj = nn.Linear(w, nr * d, bias=False)
            self.role_k_proj = nn.Linear(w, d, bias=False)
            for lin in (self.role_head, self.role_q_proj, self.role_k_proj):
                _init_linear(lin, gen=gen)
        elif arch == "P2-dense":
            dh = int(x2_spec.dense_hidden)
            self.dense_pos_1 = nn.Linear(w, dh, bias=True)
            self.dense_pos_2 = nn.Linear(dh, d, bias=True)
            self.dense_goal_1 = nn.Linear(w, dh, bias=True)
            self.dense_goal_2 = nn.Linear(dh, d, bias=True)
            for lin in (self.dense_pos_1, self.dense_pos_2, self.dense_goal_1, self.dense_goal_2):
                _init_linear(lin, gen=gen)
        elif arch in ("R", "R2"):
            s, k, eh = int(x2_spec.slot_dim), int(x2_spec.num_slots), int(x2_spec.slot_edge_hidden)
            self.tok_ln = nn.LayerNorm(w)
            self.tok_proj = nn.Linear(w, s, bias=False)
            self.slots_init = nn.Parameter(torch.empty(k, s))
            with torch.no_grad():
                self.slots_init.normal_(mean=0.0, std=NORMAL_STD, generator=gen)
            self.slot_q = nn.Linear(s, s, bias=False)
            self.slot_k = nn.Linear(s, s, bias=False)
            self.slot_v = nn.Linear(s, s, bias=False)
            self.slot_gru = nn.GRUCell(s, s)
            self.rel_ln = nn.LayerNorm(s)
            self.edge_mlp_1 = nn.Linear(3 * s, eh, bias=True)
            self.edge_mlp_2 = nn.Linear(eh, 1, bias=True)
            self.msg_proj = nn.Linear(s, s, bias=False)
            self.rel_gru = nn.GRUCell(s, s)
            self.goal_q_proj = nn.Linear(w, s, bias=False)
            self.read_ln = nn.LayerNorm(s)
            self.r_to_ptr = nn.Linear(s, d, bias=False)
            for lin in (
                self.tok_proj,
                self.slot_q,
                self.slot_k,
                self.slot_v,
                self.edge_mlp_1,
                self.edge_mlp_2,
                self.msg_proj,
                self.goal_q_proj,
                self.r_to_ptr,
            ):
                _init_linear(lin, gen=gen)
            _init_gru_cell(self.slot_gru, gen=gen)
            _init_gru_cell(self.rel_gru, gen=gen)
            if arch == "R2":
                self.r_to_op = nn.Linear(s, 4, bias=True)
                self.r_op_gate = nn.Linear(w, 1, bias=True)
                self.r_val_gate = nn.Linear(w, 1, bias=True)
                _init_linear(self.r_to_op, gen=gen)
                _init_linear(self.r_op_gate, gen=gen, bias_value=-1.5)
                _init_linear(self.r_val_gate, gen=gen, bias_value=-1.5)
        elif arch == "R2-pool":
            ph = int(x2_spec.pool_hidden)
            s = int(x2_spec.slot_dim)
            self.pool_ln = nn.LayerNorm(w)
            self.pool_mlp_1 = nn.Linear(w, ph, bias=True)
            self.pool_mlp_2 = nn.Linear(ph, s, bias=True)
            self.read_ln = nn.LayerNorm(s)
            self.r_to_ptr = nn.Linear(s, d, bias=False)
            self.r_to_op = nn.Linear(s, 4, bias=True)
            self.r_op_gate = nn.Linear(w, 1, bias=True)
            self.r_val_gate = nn.Linear(w, 1, bias=True)
            for lin in (self.pool_mlp_1, self.pool_mlp_2, self.r_to_ptr, self.r_to_op):
                _init_linear(lin, gen=gen)
            _init_linear(self.r_op_gate, gen=gen, bias_value=-1.5)
            _init_linear(self.r_val_gate, gen=gen, bias_value=-1.5)
        elif arch == "U-P":
            ud = int(x2_spec.u_dim)
            self.u_in = nn.Linear(w, ud, bias=True)
            self.u_k = nn.Linear(w, ud, bias=False)
            self.u_v = nn.Linear(w, ud, bias=False)
            self.u_gru = nn.GRUCell(ud, ud)
            self.u_out = nn.Linear(ud, w, bias=True)
            for lin in (self.u_in, self.u_k, self.u_v, self.u_out):
                _init_linear(lin, gen=gen)
            _init_gru_cell(self.u_gru, gen=gen)
        elif arch == "U-P-par":
            ph = int(x2_spec.u_par_hidden)
            self.u_par_q = nn.Linear(w, 4 * ph, bias=False)
            self.u_par_k = nn.Linear(w, ph, bias=False)
            self.u_par_v = nn.Linear(w, ph, bias=False)
            self.u_par_comb = nn.Linear(4 * ph, w, bias=True)
            for lin in (self.u_par_q, self.u_par_k, self.u_par_v, self.u_par_comb):
                _init_linear(lin, gen=gen)
        elif arch in ("DM", "DM-add"):
            md = int(x2_spec.mem_dim)
            self.mem_k = nn.Linear(w, md, bias=False)
            self.mem_v = nn.Linear(w, md, bias=False)
            self.mem_b = nn.Linear(w, 1, bias=True)
            self.mem_q = nn.Linear(w, md, bias=False)
            self.mem_to_op = nn.Linear(md, 4, bias=True)
            self.mem_to_ptr = nn.Linear(md, d, bias=False)
            self.mem_to_val = nn.Linear(md, 16, bias=True)
            self.mem_gate = nn.Linear(w, 1, bias=True)
            for lin in (self.mem_k, self.mem_v, self.mem_q, self.mem_to_op, self.mem_to_ptr, self.mem_to_val):
                _init_linear(lin, gen=gen)
            _init_linear(self.mem_b, gen=gen, bias_value=0.0)
            _init_linear(self.mem_gate, gen=gen, bias_value=0.0)
        elif arch in ("PS", "PS-ctrl"):
            fh = int(x2_spec.forecast_hidden)
            nb = int(x2_spec.forecast_branches)
            nc = int(x2_spec.forecast_classes)
            fvec_dim = nb * (nc + 1)  # action-conditioned outcome probs + scalar entropy per branch
            self.forecast_1 = nn.Linear(w, fh, bias=True)
            self.forecast_2 = nn.Linear(fh, nb * nc, bias=True)
            self.ps_to_op = nn.Linear(fvec_dim, 4, bias=True)
            self.ps_op_gate = nn.Linear(w, 1, bias=True)
            for lin in (self.forecast_1, self.forecast_2, self.ps_to_op):
                _init_linear(lin, gen=gen)
            _init_linear(self.ps_op_gate, gen=gen, bias_value=-0.5)
            if arch == "PS-ctrl":
                self.ctrl_state_proj = nn.Linear(w, fvec_dim, bias=True)
                _init_linear(self.ctrl_state_proj, gen=gen)

        self._active_lesion: str | None = None
        self._active_steps_override: int | None = None
        self._episodic_memory: torch.Tensor | None = None

    def reset_memory(self, batch_size: int = 1, device: torch.device | None = None) -> torch.Tensor:
        md = int(self.x2_spec.mem_dim)
        dev = device if device is not None else next(self.parameters()).device
        M = torch.zeros((batch_size, md, md), dtype=torch.float32, device=dev)
        self._episodic_memory = M
        return M

    def clone_memory(self) -> torch.Tensor | None:
        return None if self._episodic_memory is None else self._episodic_memory.detach().clone()

    def set_memory(self, M: torch.Tensor | None) -> None:
        self._episodic_memory = None if M is None else M.detach().clone()

    @contextmanager
    def ablate_mechanism(self, mode: str = "zero_r") -> Iterator[None]:
        prev = self._active_lesion
        self._active_lesion = str(mode)
        try:
            yield
        finally:
            self._active_lesion = prev

    @contextmanager
    def compute_depth_context(self, steps: int) -> Iterator[None]:
        prev = self._active_steps_override
        self._active_steps_override = int(steps)
        try:
            yield
        finally:
            self._active_steps_override = prev

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        backbone_params: list[nn.Parameter] = []
        mechanism_params: list[nn.Parameter] = []
        for name, p in self.named_parameters():
            if name.startswith("core."):
                backbone_params.append(p)
            else:
                mechanism_params.append(p)
        return {"backbone": backbone_params, "mechanism": mechanism_params}

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

    def _compute_slots(
        self,
        H_row: torch.Tensor,
        valid_prefix: torch.Tensor,
        h_goal: torch.Tensor,
        *,
        lesion: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = H_row.shape
        s = int(self.x2_spec.slot_dim)
        k = int(self.x2_spec.num_slots)
        tok_s = self.tok_proj(self.tok_ln(H_row))
        slots = self.slots_init.unsqueeze(0).expand(B, k, s).contiguous()
        for _ in range(2):
            sq = self.slot_q(slots)
            sk = self.slot_k(tok_s)
            sv = self.slot_v(tok_s)
            attn = torch.bmm(sq, sk.transpose(1, 2)) / math.sqrt(s)
            attn = attn.masked_fill(~valid_prefix.unsqueeze(1), float("-inf"))
            w_attn = F.softmax(attn, dim=-1)
            upd = torch.bmm(w_attn, sv)
            slots = self.slot_gru(upd.reshape(B * k, s), slots.reshape(B * k, s)).reshape(B, k, s)

        for _ in range(2):
            sn = self.rel_ln(slots)
            si = sn.unsqueeze(2).expand(B, k, k, s)
            sj = sn.unsqueeze(1).expand(B, k, k, s)
            edge_in = torch.cat([si, sj, si - sj], dim=-1)
            e_logits = self.edge_mlp_2(F.silu(self.edge_mlp_1(edge_in))).squeeze(-1)
            if lesion == "permute_edges":
                e_logits = torch.roll(e_logits, shifts=1, dims=-1)
            e_w = F.softmax(e_logits, dim=-1)
            msgs = torch.bmm(e_w, self.msg_proj(sn))
            slots = self.rel_gru(msgs.reshape(B * k, s), slots.reshape(B * k, s)).reshape(B, k, s)

        gq = self.goal_q_proj(h_goal)
        r_scores = torch.bmm(gq.unsqueeze(1), slots.transpose(1, 2)).squeeze(1) / math.sqrt(s)
        r_read = self.read_ln(torch.bmm(F.softmax(r_scores, dim=-1).unsqueeze(1), slots).squeeze(1))
        return r_read, slots

    def _run_delta_memory(
        self,
        H_row: torch.Tensor,
        valid_prefix: torch.Tensor,
        h_goal: torch.Tensor,
        *,
        additive: bool,
        init_memory: torch.Tensor | None,
        lesion: str | None,
        block_write_index: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = H_row.shape
        md = int(self.x2_spec.mem_dim)
        if init_memory is not None:
            M = init_memory.to(device=H_row.device, dtype=torch.float32)
            if M.shape[0] != B:
                M = M[:1].expand(B, md, md).contiguous()
        else:
            M = torch.zeros((B, md, md), dtype=torch.float32, device=H_row.device)

        # Chunk the valid prefix into up to 8 evenly spaced public event summaries for tractable fast-weight writes
        n_events = min(8, L)
        stride = max(1, L // n_events)
        k_all = self.mem_k(H_row)
        v_all = self.mem_v(H_row)
        b_all = torch.sigmoid(self.mem_b(H_row))
        event_idx = 0
        for start in range(0, L, stride):
            end = min(L, start + stride)
            cmask = valid_prefix[:, start:end]
            has_any = cmask.any(dim=1, keepdim=True)
            denom = cmask.to(torch.float32).sum(dim=1, keepdim=True).clamp_min(1.0)
            w_f = cmask.to(torch.float32).unsqueeze(-1)
            k_e = (k_all[:, start:end] * w_f).sum(dim=1) / denom
            v_e = (v_all[:, start:end] * w_f).sum(dim=1) / denom
            b_e = ((b_all[:, start:end] * w_f).sum(dim=1) / denom) * has_any.to(torch.float32)

            if lesion == "reset_mid_episode" and event_idx == max(1, n_events // 2):
                M = torch.zeros_like(M)
            if lesion == "write_block" and block_write_index is not None and event_idx == int(block_write_index):
                b_e = b_e * 0.0
            elif lesion == "write_sham" and block_write_index is not None:
                sham_idx = 0 if int(block_write_index) != 0 else 1
                if event_idx == sham_idx:
                    b_e = b_e * 0.0

            M = delta_memory_step(M, k_e, v_e, b_e, additive=additive)
            event_idx += 1

        if lesion in ("zero_mem", "zero_r"):
            M = torch.zeros_like(M)
        q_m = self.mem_q(h_goal)
        q_m = q_m / q_m.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        read_vec = delta_memory_read(M, q_m)
        return read_vec, M

    def forward_action(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        action_positions: torch.Tensor,
        *,
        action_batch_indices: torch.Tensor | None = None,
        copy_enabled_override: bool | None = None,
        lesion_mode: str | None = None,
        recurrent_steps_override: int | None = None,
        init_memory: torch.Tensor | None = None,
        block_write_index: int | None = None,
        interchange_step: int | None = None,
        interchange_state: torch.Tensor | None = None,
        forecast_override: torch.Tensor | None = None,
        use_activation_checkpointing: bool = False,
    ) -> X2ForwardOutput:
        if action_positions.ndim != 1:
            raise ValueError("action_positions must be 1D")
        if action_batch_indices is None:
            b_idx = torch.arange(token_ids.shape[0], device=token_ids.device, dtype=torch.int64)
        else:
            b_idx = action_batch_indices.to(dtype=torch.int64)
        pos_idx = action_positions.to(dtype=torch.int64)

        H = self.forward_hidden_only(token_ids, segment_ids, use_activation_checkpointing=use_activation_checkpointing).float()
        h = H[b_idx, pos_idx]
        H_row = H[b_idx]
        tok_row = token_ids[b_idx]
        L = token_ids.shape[1]
        seg_row = segment_ids[b_idx]
        q_seg = segment_ids[b_idx, pos_idx].unsqueeze(1)
        pos_grid = torch.arange(L, device=token_ids.device).unsqueeze(0)
        valid_prefix = (pos_grid <= pos_idx.unsqueeze(1)) & (seg_row == q_seg) & (seg_row >= 0)

        lesion = lesion_mode if lesion_mode is not None else self._active_lesion
        arch = self.x2_spec.architecture
        d = int(self.x2_spec.pointer_dim)

        h_op = h
        h_val = h
        extra_op_logits = torch.zeros((h.shape[0], 4), dtype=torch.float32, device=h.device)
        extra_ptr_q = torch.zeros((h.shape[0], d), dtype=torch.float32, device=h.device)
        extra_val_logits = torch.zeros((h.shape[0], 16), dtype=torch.float32, device=h.device)
        forecast_logits: torch.Tensor | None = None
        forecast_vector: torch.Tensor | None = None
        mem_read_vec: torch.Tensor | None = None
        mem_state: torch.Tensor | None = None
        u_step_states: list[torch.Tensor] = []
        slot_states: torch.Tensor | None = None

        if arch == "U-P":
            steps = (
                int(recurrent_steps_override)
                if recurrent_steps_override is not None
                else (int(self._active_steps_override) if self._active_steps_override is not None else int(self.x2_spec.u_steps))
            )
            steps = max(1, steps)
            ud = int(self.x2_spec.u_dim)
            q_state = self.u_in(h)
            k_u = self.u_k(H_row)
            v_u = self.u_v(H_row)
            for t in range(steps):
                if interchange_step is not None and interchange_state is not None and t == int(interchange_step):
                    q_state = interchange_state.to(device=q_state.device, dtype=q_state.dtype)
                sc = torch.bmm(q_state.unsqueeze(1), k_u.transpose(1, 2)).squeeze(1) / math.sqrt(ud)
                w_u = F.softmax(sc.masked_fill(~valid_prefix, float("-inf")), dim=-1)
                rd = torch.bmm(w_u.unsqueeze(1), v_u).squeeze(1)
                q_state = self.u_gru(rd, q_state)
                u_step_states.append(q_state)
            delta_u = self.u_out(q_state)
            if lesion in ("zero_r", "ablate_u"):
                delta_u = delta_u * 0.0
            h_op = h + delta_u
            h_val = h + delta_u
        elif arch == "U-P-par":
            ph = int(self.x2_spec.u_par_hidden)
            q_par = self.u_par_q(h).reshape(h.shape[0], 4, ph)
            k_par = self.u_par_k(H_row)
            v_par = self.u_par_v(H_row)
            sc = torch.bmm(q_par, k_par.transpose(1, 2)) / math.sqrt(ph)
            w_par = F.softmax(sc.masked_fill(~valid_prefix.unsqueeze(1), float("-inf")), dim=-1)
            reads = torch.bmm(w_par, v_par).reshape(h.shape[0], 4 * ph)
            delta_par = self.u_par_comb(reads)
            if lesion in ("zero_r", "ablate_u"):
                delta_par = delta_par * 0.0
            h_op = h + delta_par
            h_val = h + delta_par
        elif arch in ("R", "R2"):
            r_read, slot_states = self._compute_slots(H_row, valid_prefix, h, lesion=lesion)
            if lesion == "zero_r":
                r_read = r_read * 0.0
            if arch == "R":
                extra_ptr_q = self.r_to_ptr(r_read)
            else:
                op_g = torch.sigmoid(self.r_op_gate(h))
                val_g = torch.sigmoid(self.r_val_gate(h))
                op_res = op_g * self.r_to_op(r_read)
                val_res = val_g * self.r_to_ptr(r_read)
                if lesion == "op_only":
                    op_res = op_res * 0.0
                elif lesion == "val_only":
                    val_res = val_res * 0.0
                elif lesion == "both":
                    op_res = op_res * 0.0
                    val_res = val_res * 0.0
                extra_op_logits = op_res
                extra_ptr_q = val_res
        elif arch == "R2-pool":
            mask_f = valid_prefix.to(torch.float32).unsqueeze(-1)
            pooled = (self.pool_ln(H_row) * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
            p_read = self.read_ln(self.pool_mlp_2(F.silu(self.pool_mlp_1(pooled))))
            if lesion == "zero_r":
                p_read = p_read * 0.0
            op_res = torch.sigmoid(self.r_op_gate(h)) * self.r_to_op(p_read)
            val_res = torch.sigmoid(self.r_val_gate(h)) * self.r_to_ptr(p_read)
            if lesion == "op_only":
                op_res = op_res * 0.0
            elif lesion == "val_only":
                val_res = val_res * 0.0
            elif lesion == "both":
                op_res = op_res * 0.0
                val_res = val_res * 0.0
            extra_op_logits = op_res
            extra_ptr_q = val_res
        elif arch in ("DM", "DM-add"):
            mem_read_vec, mem_state = self._run_delta_memory(
                H_row,
                valid_prefix,
                h,
                additive=(arch == "DM-add"),
                init_memory=init_memory,
                lesion=lesion,
                block_write_index=block_write_index,
            )
            mg = torch.sigmoid(self.mem_gate(h))
            extra_op_logits = mg * self.mem_to_op(mem_read_vec)
            extra_ptr_q = mg * self.mem_to_ptr(mem_read_vec)
            extra_val_logits = mg * self.mem_to_val(mem_read_vec)
        elif arch in ("PS", "PS-ctrl"):
            nb = int(self.x2_spec.forecast_branches)
            nc = int(self.x2_spec.forecast_classes)
            f_raw = self.forecast_2(F.silu(self.forecast_1(h))).reshape(h.shape[0], nb, nc)
            forecast_logits = f_raw
            f_probs = F.softmax(f_raw, dim=-1)
            f_ent = -(f_probs * f_probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
            forecast_vector = torch.cat([f_probs, f_ent], dim=-1).reshape(h.shape[0], nb * (nc + 1))
            routed_vec = forecast_vector if arch == "PS" else self.ctrl_state_proj(h)
            if forecast_override is not None and arch == "PS":
                routed_vec = forecast_override.to(device=routed_vec.device, dtype=routed_vec.dtype)
            if lesion == "zero_r":
                routed_vec = routed_vec * 0.0
            extra_op_logits = torch.sigmoid(self.ps_op_gate(h)) * self.ps_to_op(routed_vec)

        op_logits = self.op_head(h_op) + extra_op_logits
        log_p_op = F.log_softmax(op_logits, dim=-1)
        z_val = self.val_head(h_val) + extra_val_logits

        # Pointer copy mechanism (role-factorized for P2, nonlinear dense for P2-dense, bilinear otherwise)
        cand_mask = valid_prefix & (tok_row >= AP_MIN_TOKEN_ID) & (tok_row <= AP_MAX_TOKEN_ID)
        has_cand = cand_mask.any(dim=1)
        copy_disabled = (copy_enabled_override is False) or (lesion == "disable_copy")

        if arch == "P2":
            nr = int(self.x2_spec.num_roles)
            role_probs = F.softmax(self.role_head(H_row), dim=-1)  # (B, L, nr)
            if lesion == "zero_r":
                role_probs = torch.full_like(role_probs, 1.0 / nr)
            q_roles = self.role_q_proj(h_val).reshape(h.shape[0], nr, d)
            k_roles = self.role_k_proj(H_row)  # (B, L, d)
            # Bilinear interaction between goal, position, and latent role
            role_scores = torch.einsum("bdr,bld->blr", q_roles.transpose(1, 2), k_roles) / math.sqrt(d)
            ptr_scores = (role_probs * role_scores).sum(dim=-1)
        elif arch == "P2-dense":
            k_d = self.dense_pos_2(F.silu(self.dense_pos_1(H_row)))
            q_d = self.dense_goal_2(F.silu(self.dense_goal_1(h_val)))
            ptr_scores = torch.bmm(k_d, q_d.unsqueeze(-1)).squeeze(-1) / math.sqrt(d)
        else:
            q_p = self.q_proj(h_val) + extra_ptr_q
            k_p = self.k_proj(H_row)
            ptr_scores = torch.bmm(k_p, q_p.unsqueeze(-1)).squeeze(-1) / math.sqrt(d)

        masked_scores = ptr_scores.masked_fill(~cand_mask, -1e9)
        attn_w = F.softmax(masked_scores, dim=-1) * cand_mask.to(torch.float32)
        sym_idx = (tok_row - AP_MIN_TOKEN_ID).clamp(0, 15)
        p_copy = torch.zeros((h.shape[0], 16), dtype=torch.float32, device=h.device)
        p_copy.scatter_add_(1, sym_idx, attn_w)

        g_val = torch.sigmoid(self.gate(h_val)) * has_cand.to(torch.float32).unsqueeze(-1)
        if copy_disabled:
            g_val = g_val * 0.0
        p_vocab = F.softmax(z_val, dim=-1)
        p_val = ((1.0 - g_val) * p_vocab + g_val * p_copy).clamp_min(1e-12)
        p_val = p_val / p_val.sum(dim=-1, keepdim=True)
        log_p_val = p_val.log()

        log_p_ans = log_p_op[:, :1] + log_p_val
        log_probs_19 = torch.cat([log_p_ans, log_p_op[:, 1:]], dim=-1)

        prov_val_logits = self.provisional_val_head(h_val)
        cert_logit = self.provisional_cert_head(h_op).squeeze(-1)

        return X2ForwardOutput(
            log_probs_19=log_probs_19,
            log_p_op=log_p_op,
            log_p_val=log_p_val,
            provisional_val_logits=prov_val_logits,
            certifiable_logit=cert_logit,
            decision_hidden=h_op,
            gate=g_val.squeeze(-1),
            forecast_logits=forecast_logits,
            forecast_vector=forecast_vector,
            memory_read_vec=mem_read_vec,
            memory_state=mem_state,
            u_step_states=tuple(u_step_states),
            slot_states=slot_states,
        )


def count_parameters(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def backbone_state_digest(model: X2TypedModel) -> str:
    """Return a deterministic SHA-256 over exact backbone parameter tensors (shared-init verification)."""
    h = sha256()
    for name in sorted(model.core.state_dict()):
        t = model.core.state_dict()[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _probe_param_count(spec: X2ModelSpec) -> int:
    core = initialize(spec.base_spec, 0)
    m = X2TypedModel(core, spec, head_seed=0)
    return count_parameters(m)


def build_x2_model_spec(
    architecture: str,
    *,
    base_spec: ModelSpec | None = None,
    tiny_canary: bool = False,
    **overrides: Any,
) -> X2ModelSpec:
    """Build a capacity-matched X2ModelSpec (active controls within <=2%, S20 head overhead <=5%)."""
    if base_spec is None:
        base_spec = build_tiny_canary_base_spec() if tiny_canary else build_s20_base_spec()

    is_tiny = base_spec.width <= 32
    defaults: dict[str, Any] = (
        {
            "pointer_dim": 8,
            "num_roles": 4,
            "dense_hidden": 12,
            "slot_dim": 12,
            "num_slots": 4,
            "slot_edge_hidden": 8,
            "pool_hidden": 42,
            "u_dim": 12,
            "u_steps": 4,
            "u_par_hidden": 12,
            "mem_dim": 16,
            "forecast_hidden": 16,
            "forecast_branches": 4,
            "forecast_classes": 4,
        }
        if is_tiny
        else {
            "pointer_dim": 64,
            "num_roles": 8,
            "dense_hidden": 286,
            "slot_dim": 96,
            "num_slots": 8,
            "slot_edge_hidden": 64,
            "pool_hidden": 592,
            "u_dim": 128,
            "u_steps": 4,
            "u_par_hidden": 88,
            "mem_dim": 64,
            "forecast_hidden": 128,
            "forecast_branches": 4,
            "forecast_classes": 4,
        }
    )
    defaults.update({k: v for k, v in overrides.items() if k in defaults})

    # Solve exact active-control width when not explicitly overridden
    if architecture == "P2-dense" and "dense_hidden" not in overrides:
        target = _probe_param_count(X2ModelSpec(architecture="P2", base_spec=base_spec, **defaults))
        w, d = int(base_spec.width), int(defaults["pointer_dim"])
        # Analytic solve: 2 * (w*h + h + h*d + d) ~= target_extra
        base_p = _probe_param_count(X2ModelSpec(architecture="P", base_spec=base_spec, **defaults))
        extra_needed = max(1, target - base_p - 2 * d)
        h_est = max(2, round(extra_needed / max(1, 2 * (w + d + 1))))
        defaults["dense_hidden"] = int(h_est)
    elif architecture == "R2-pool" and "pool_hidden" not in overrides:
        target = _probe_param_count(X2ModelSpec(architecture="R2", base_spec=base_spec, **defaults))
        base_p = _probe_param_count(X2ModelSpec(architecture="P", base_spec=base_spec, **defaults))
        w, s, d = int(base_spec.width), int(defaults["slot_dim"]), int(defaults["pointer_dim"])
        fixed_r2_pool = 2 * w + 2 * s + s * d + (s * 4 + 4) + 2 * (w + 1) + s
        extra_needed = max(1, target - base_p - fixed_r2_pool)
        defaults["pool_hidden"] = max(2, round(extra_needed / max(1, w + s + 1)))
    elif architecture == "U-P-par" and "u_par_hidden" not in overrides:
        target = _probe_param_count(X2ModelSpec(architecture="U-P", base_spec=base_spec, **defaults))
        base_p = _probe_param_count(X2ModelSpec(architecture="P", base_spec=base_spec, **defaults))
        w = int(base_spec.width)
        extra_needed = max(1, target - base_p - w)
        defaults["u_par_hidden"] = max(2, round(extra_needed / max(1, 10 * w)))

    spec = X2ModelSpec(architecture=architecture, base_spec=base_spec, **defaults)
    spec.assert_valid()
    return spec


def build_x2_model(
    spec: X2ModelSpec,
    init_seed: int,
    *,
    device: str | torch.device = "cpu",
) -> X2TypedModel:
    """Instantiate an X2TypedModel with byte-identical shared backbone weights per init_seed."""
    spec.assert_valid()
    core = initialize(spec.base_spec, int(init_seed))
    model = X2TypedModel(core, spec, head_seed=int(init_seed))
    return model.to(torch.device(device))


def verify_x2_parameter_contracts(
    *,
    tiny_canary: bool = False,
) -> dict[str, Any]:
    """Verify S20 <=5% head overhead and <=2% active-control parameter parity across all 5 X2 pairs."""
    base_spec = build_tiny_canary_base_spec() if tiny_canary else build_s20_base_spec()
    base_params = int(base_spec.parameter_receipt().total)
    counts: dict[str, int] = {}
    overheads: dict[str, float] = {}
    for arch in X2_ARCHITECTURES:
        spec = build_x2_model_spec(arch, base_spec=base_spec)
        tot = _probe_param_count(spec)
        counts[arch] = tot
        overheads[arch] = float((tot - base_params) / base_params)

    pair_diffs: dict[str, float] = {}
    for trt, ctrl in X2_NEW_PAIRS:
        rel = abs(counts[trt] - counts[ctrl]) / max(counts[trt], counts[ctrl])
        pair_diffs[f"{trt}_vs_{ctrl}"] = float(rel)

    s20_ok = True if tiny_canary else all(ov <= X2_MAX_S20_HEAD_OVERHEAD for ov in overheads.values())
    tol = 0.08 if tiny_canary else X2_ACTIVE_CONTROL_PARAM_TOL
    pairs_ok = all(d <= tol for d in pair_diffs.values())
    return {
        "base_parameters": base_params,
        "parameter_counts": counts,
        "head_overhead_ratios": overheads,
        "pair_relative_diffs": pair_diffs,
        "s20_overhead_within_5pct": s20_ok,
        "active_controls_within_tol": pairs_ok,
        "passed": bool(s20_ok and pairs_ok),
    }


__all__ = [
    "OP_TO_INDEX",
    "X2ForwardOutput",
    "X2ModelSpec",
    "X2TypedModel",
    "X2_ARCHITECTURES",
    "X2_MAX_S20_HEAD_OVERHEAD",
    "X2_NEW_PAIRS",
    "backbone_state_digest",
    "build_s20_base_spec",
    "build_tiny_canary_base_spec",
    "build_x2_model",
    "build_x2_model_spec",
    "count_parameters",
    "delta_memory_read",
    "delta_memory_step",
    "verify_x2_parameter_contracts",
]
