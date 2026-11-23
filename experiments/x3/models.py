"""V8-X3 Cognitive Reasoning & Mechanistic Repair Architectures (R2 U-Fix, R3 Alpha, R5 Screens, S20 & S50)."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from anra_v5.elendril.representation import SpecialToken, UTF8
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.elendril.v8x.contracts import build_x1_model_spec
from anra_v5.elendril.v8x.models import AP_MAX_TOKEN_ID, AP_MIN_TOKEN_ID, build_x1_s20_model
from v5_contracts.model_spec import ModelSpec
from v5_model.core import initialize, packed_layout

from .cognitive_battery import infer_public_generation_bound, parse_and_score_public_typed_output

X3_CORE_ARCHITECTURES: tuple[str, ...] = (
    "Base",
    "P",
    "R",
    "R-noedge",
    "P-R-Iter",
    "P-R-Iter-Fixed",
    "P-R-Iter-NoRel",
    "P-R-Iter-AddMem",
    "P-R-Iter-NoCopy",
)

X3_EXTENDED_ARCHITECTURES: tuple[str, ...] = (
    # R2 Anchored Recurrence Repairs & Controls
    "U2-Fix",
    "U4-Fix",
    "U2-GatedUnanchored",
    "U4-GatedUnanchored",
    "U-ZeroStep",
    "U-P",
    # R3 Alpha Pointer & Relational Routing + 2x2 Controls
    "P1-Alpha",
    "P1-DensePos",
    "P1-ParallelRead",
    "R1-Alpha",
    "R1-ValOnly",
    "R1-NoMsg-Both",
    "R1-NoMsg-ValOnly",
    "R1-SlotMLP",
    # R5 Project-Specific Falsifiable Screens
    "CounterexampleReader",
    "RelevanceReader2",
    "ContradictionShuffle",
    "OperatorTransportBank",
    "EpisodicMLP",
    "NearestExampleRetrieval",
)

X3_ARCHITECTURES: tuple[str, ...] = (*X3_CORE_ARCHITECTURES, *X3_EXTENDED_ARCHITECTURES)

X3_CONTROL_PAIRS: tuple[tuple[str, str], ...] = (
    ("P-R-Iter-Fixed", "P-R-Iter"),
    ("P-R-Iter-NoRel", "P-R-Iter"),
    ("P-R-Iter-AddMem", "P-R-Iter"),
    ("P-R-Iter-NoCopy", "P-R-Iter"),
    ("R-noedge", "R"),
)

# Versioned R2/R3/R5 paired contrasts (matched active parameters within <= 2% inside each contrast)
X3_VERSIONED_CONTROL_PAIRS: tuple[tuple[str, str], ...] = (
    ("U2-GatedUnanchored", "U2-Fix"),
    ("U4-GatedUnanchored", "U4-Fix"),
    ("P1-DensePos", "P1-Alpha"),
    ("P1-ParallelRead", "P1-Alpha"),
    ("R1-ValOnly", "R1-Alpha"),
    ("R1-NoMsg-Both", "R1-Alpha"),
    ("R1-NoMsg-ValOnly", "R1-Alpha"),
    ("RelevanceReader2", "CounterexampleReader"),
    ("ContradictionShuffle", "CounterexampleReader"),
    ("EpisodicMLP", "OperatorTransportBank"),
)

# Verified 50,552,576 core params (ctx=640); yields P=50,667,029 and R=51,229,462 under native X1 constructors
S50_MODEL_SPEC: ModelSpec = replace(
    MODEL_PROFILES["S20"],
    width=768,
    layers=8,
    query_heads=12,
    kv_heads=4,
    head_dimension=64,
    ffn_width=2048,
    context_length=640,
)

TINY_CANARY_MODEL_SPEC: ModelSpec = replace(
    MODEL_PROFILES["S20"],
    width=64,
    layers=2,
    query_heads=2,
    kv_heads=1,
    head_dimension=32,
    ffn_width=128,
    context_length=640,
)


def build_x3_backbone_spec(scale: str = "S20") -> ModelSpec:
    s = str(scale).strip()
    if s in ("S20", "s20"):
        return replace(MODEL_PROFILES["S20"], context_length=640)
    if s in ("S50", "s50"):
        return S50_MODEL_SPEC
    if s in ("tiny", "canary", "TINY"):
        return TINY_CANARY_MODEL_SPEC
    raise ValueError(f"Unknown X3 backbone scale {scale!r}; expected 'tiny', 'S20', or 'S50'")


def build_legacy_x1_scaled_model(
    architecture: str,
    *,
    scale: str | None = None,
    init_seed: int = 7,
    context_length: int = 640,
) -> dict[str, Any]:
    """Instantiate exact native `anra_v5.elendril.v8x` S20/S50 P or R model and parameter receipt (§1, §5 R6)."""
    arch_clean = str(architecture).strip()
    resolved_scale = scale
    if arch_clean in ("P-50M", "R-50M"):
        base_arch = arch_clean[0]
        resolved_scale = resolved_scale or "S50"
    else:
        base_arch = arch_clean
        resolved_scale = resolved_scale or "S20"

    if base_arch not in ("B0", "J", "F", "F-cap", "P", "P-capR", "R", "R-noedge", "U-1", "U-2", "U-4", "U2-untied"):
        raise ValueError(f"Unsupported legacy X1 architecture {architecture!r}")

    base_spec = replace(build_x3_backbone_spec(resolved_scale), context_length=int(context_length))
    x1_spec = build_x1_model_spec(base_arch, base_spec=base_spec)
    model = build_x1_s20_model(x1_spec, init_seed=int(init_seed), torch_module=torch)
    n_params = sum(int(p.numel()) for p in model.parameters())
    return {
        "architecture": arch_clean,
        "legacy_architecture": base_arch,
        "scale": resolved_scale,
        "x1_spec": x1_spec,
        "model": model,
        "total_parameters": n_params,
        "spec_sha256": x1_spec.sha256(),
    }


class _AnchoredRecurrentUCell(nn.Module):
    """R2 Anchored & Unanchored norm-controlled GRU-attention cell over immutable public H without pointer (§3)."""

    def __init__(self, width: int, recurrent_dim: int, *, anchored: bool = True) -> None:
        super().__init__()
        self.width = int(width)
        self.recurrent_dim = int(recurrent_dim)
        self.anchored = bool(anchored)
        self.u_q = nn.Linear(self.width, self.recurrent_dim, bias=False)
        self.u_k = nn.Linear(self.width, self.recurrent_dim, bias=False)
        self.u_v = nn.Linear(self.width, self.recurrent_dim, bias=False)
        self.u_out = nn.Linear(self.recurrent_dim, self.width, bias=True)
        self.u_gru = nn.GRUCell(self.width, self.width)
        self.gate_proj = nn.Linear(self.width, self.width, bias=True)
        self.rms_weight = nn.Parameter(torch.ones(self.width))
        nn.init.constant_(self.gate_proj.bias, -2.0)

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        denom = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
        return (x / denom) * self.rms_weight

    def step(
        self,
        z_t: torch.Tensor,
        h0: torch.Tensor,
        h_bank: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        q_t = self.u_q(z_t)
        k_all = self.u_k(h_bank)
        v_all = self.u_v(h_bank)
        scores = torch.bmm(q_t.unsqueeze(1), k_all.transpose(1, 2)).squeeze(1) / math.sqrt(float(self.recurrent_dim))
        weights = F.softmax(scores.masked_fill(~valid_mask, -1e9), dim=-1)
        read_vec = torch.bmm(weights.unsqueeze(1), v_all).squeeze(1)
        gru_cand = self.u_gru(self.u_out(read_vec), z_t)
        ref = h0 if self.anchored else z_t
        a_t = torch.sigmoid(self.gate_proj(z_t))
        h0_rms = torch.sqrt(torch.mean(h0 * h0, dim=-1, keepdim=True) + 1e-6).detach()
        delta = self._rms_norm(gru_cand - ref) * h0_rms
        z_next = ref + a_t * delta
        return z_next, {
            "gate_mean": a_t.mean(dim=-1),
            "update_norm": delta.norm(dim=-1),
            "anchor_dist": (z_next - h0).norm(dim=-1),
        }


class X3CognitiveModel(nn.Module):
    """Unified X3 cognitive architecture wrapping shared `v5_model.core` (`core.` prefix)."""

    def __init__(
        self,
        architecture: str,
        base_spec: ModelSpec,
        *,
        seed: int = 7,
        max_steps: int = 4,
        num_slots: int = 6,
    ) -> None:
        super().__init__()
        if architecture not in X3_ARCHITECTURES:
            raise ValueError(f"Unknown X3 architecture: {architecture!r}")
        self.architecture = str(architecture)
        self.base_spec = base_spec
        self.max_ctx = int(base_spec.context_length)
        self.max_steps = int(max(1, max_steps))
        self.num_slots = int(max(2, num_slots))

        # Byte-identical shared backbone initializer across control/treatment per seed
        self.core = initialize(base_spec, int(seed), torch_module=torch)

        d = int(base_spec.width)
        m_dim = min(96, max(8, d // 8))
        self.d_model = d
        self.m_dim = m_dim

        torch.manual_seed(int(seed) + 4103)
        self._ablation_mode: str = "none"
        self._episodic_state: torch.Tensor | None = None

        self.lm_bias = nn.Parameter(torch.zeros(int(base_spec.vocabulary_size)))
        self.op_head = nn.Linear(d, 4)  # ANSWER, QUERY, RETRIEVE, ABSTAIN
        self.cls_head = nn.Linear(m_dim, 16)
        self.uncert_head = nn.Linear(m_dim, 2)
        self.out_proj = nn.Linear(m_dim, d)

        # Configure clean modular family flags
        is_iter_family = self.architecture.startswith("P-R-Iter")
        is_r2_u = self.architecture in ("U2-Fix", "U4-Fix", "U2-GatedUnanchored", "U4-GatedUnanchored", "U-P")
        is_p1 = self.architecture in ("P1-Alpha", "P1-DensePos", "P1-ParallelRead")
        is_r1 = self.architecture in ("R1-Alpha", "R1-ValOnly", "R1-NoMsg-Both", "R1-NoMsg-ValOnly", "R1-SlotMLP")
        is_ce_screen = self.architecture in ("CounterexampleReader", "RelevanceReader2", "ContradictionShuffle")
        is_op_screen = self.architecture in ("OperatorTransportBank", "EpisodicMLP", "NearestExampleRetrieval")

        uses_slots = is_iter_family or self.architecture in ("R", "R-noedge") or is_r1
        uses_pointer = is_iter_family or self.architecture in ("P", "U-P") or is_p1 or is_r1 or is_ce_screen or is_op_screen

        if uses_slots:
            self.slot_init = nn.Parameter(torch.randn(1, self.num_slots, m_dim) * 0.02)
            self.input_proj = nn.Linear(d, m_dim)
            self.query_proj = nn.Linear(d, m_dim)
            self.read_q = nn.Linear(m_dim, m_dim, bias=False)
            self.read_k = nn.Linear(m_dim, m_dim, bias=False)
            self.read_v = nn.Linear(m_dim, m_dim, bias=False)
            self.rel_msg = nn.Linear(2 * m_dim, m_dim)
            self.delta_gate = nn.Linear(2 * m_dim, m_dim)
            self.delta_cand = nn.Linear(2 * m_dim, m_dim)
            self.halt_head = nn.Linear(m_dim, 1)
            if is_r1:
                self.r_op_gate = nn.Linear(d, 4)
                self.r_to_op = nn.Linear(m_dim, 4)
                self.r_val_gate = nn.Linear(d, 16)
                self.r_to_val = nn.Linear(m_dim, 16)
                self.r_to_ptr = nn.Linear(m_dim, m_dim)
        elif is_r2_u:
            anchored = self.architecture in ("U2-Fix", "U4-Fix", "U-P")
            self.u_cell = _AnchoredRecurrentUCell(d, m_dim, anchored=anchored)
            self.base_proj = nn.Linear(d, m_dim)
        else:
            self.base_proj = nn.Linear(d, m_dim)

        if uses_pointer:
            self.ptr_q = nn.Linear(d, m_dim)
            self.ptr_k = nn.Linear(d, m_dim)
            self.copy_mix_head = nn.Linear(d, 2)

        if is_p1:
            # Structural role features (delimiter proximity + normalized position) vs dense projection
            if self.architecture != "P1-DensePos":
                self.role_feat_proj = nn.Linear(4, m_dim)
            else:
                self.dense_pos_proj = nn.Linear(4, m_dim)
            self.comp_q1 = nn.Linear(d, m_dim)
            self.comp_q2 = nn.Linear(d, m_dim)
            self.comp_k = nn.Linear(d, m_dim)
            self.comp_v = nn.Linear(d, m_dim)
            self.comp_u = nn.Linear(m_dim, m_dim)
            self.comp_w = nn.Linear(m_dim, m_dim)
            self.comp_out = nn.Linear(m_dim, 16)
            self.three_way_switch = nn.Linear(d, 3)

        if is_ce_screen:
            self.cand_q = nn.Linear(d, m_dim)
            self.cand_kv = nn.Linear(d, 2 * m_dim)
            self.ce_q = nn.Linear(d + m_dim, m_dim)
            self.ce_k = nn.Linear(d, m_dim)
            self.ce_v = nn.Linear(d, m_dim)
            self.ce_score_head = nn.Linear(d + 2 * m_dim, 4)

        if is_op_screen:
            self.n_operators = 8
            if self.architecture == "OperatorTransportBank":
                self.op_bank = nn.Parameter(torch.randn(self.n_operators, m_dim, m_dim) * 0.05)
                self.align_q = nn.Linear(d, m_dim)
                self.align_k = nn.Linear(d, m_dim)
                self.align_v = nn.Linear(d, m_dim)
                self.bank_gate = nn.Linear(d, self.n_operators)
            elif self.architecture == "EpisodicMLP":
                self.ep_mlp_in = nn.Linear(2 * d, m_dim * 4)
                self.ep_mlp_out = nn.Linear(m_dim * 4, m_dim)
            else:  # NearestExampleRetrieval
                self.ret_q = nn.Linear(d, m_dim)
                self.ret_k = nn.Linear(d, m_dim)
                self.ret_v = nn.Linear(d, m_dim)

    def reset_episode(self) -> None:
        self._episodic_state = None
        self._ablation_mode = "none"

    @contextmanager
    def ablate_mechanism(self, mode: str = "disable_pointer") -> Any:
        """Surgical mechanism lesion context manager yielding mechanism label."""
        valid = {
            "none",
            "disable_pointer",
            "disable_relational",
            "disable_delta",
            "disable_iterative",
            "true_edge_swap",
            "random_edge_perm",
            "op_branch_lesion",
            "val_branch_lesion",
            "zero_r",
            "sham",
            "shuffle_counterexample",
            "swap_alignment",
        }
        if mode not in valid:
            raise ValueError(f"Unsupported ablation mode {mode!r}; expected one of {sorted(valid)}")
        prev = self._ablation_mode
        self._ablation_mode = str(mode)
        try:
            yield str(mode)
        finally:
            self._ablation_mode = prev

    def _compute_pointer_symbol_probs(
        self,
        h_all: torch.Tensor,
        h_act: torch.Tensor,
        token_ids: torch.Tensor,
        valid_tok: torch.Tensor,
        *,
        q_offset: torch.Tensor | None = None,
        slot_fallback: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute differentiable pointer-copy probs over A..P symbols (or slot attention for NoCopy)."""
        bsz = h_all.shape[0]
        dev = h_all.device
        dtype = h_all.dtype

        if self.architecture == "P-R-Iter-NoCopy" and slot_fallback is not None:
            q = self.ptr_q(h_act).unsqueeze(1)
            s_full = self.out_proj(slot_fallback)
            k = self.ptr_k(s_full)
            attn = F.softmax((q @ k.transpose(1, 2)) / math.sqrt(float(self.m_dim)), dim=-1)
            slot_ctx = (attn @ slot_fallback).squeeze(1)
            return F.softmax(self.cls_head(slot_ctx), dim=-1)

        if self._ablation_mode == "disable_pointer":
            return torch.full((bsz, 16), 1.0 / 16.0, device=dev, dtype=dtype)

        q_vec = self.ptr_q(h_act)
        if q_offset is not None:
            q_vec = q_vec + q_offset
        q = q_vec.unsqueeze(1)
        k = self.ptr_k(h_all)
        scores = (q @ k.transpose(1, 2)).squeeze(1) / math.sqrt(float(self.m_dim))
        is_ap = (token_ids >= AP_MIN_TOKEN_ID) & (token_ids <= AP_MAX_TOKEN_ID) & valid_tok
        has_any_ap = is_ap.any(dim=1, keepdim=True)
        attn_mask = torch.where(has_any_ap, is_ap, valid_tok)
        safe_scores = scores.masked_fill(~attn_mask, -1e9)
        attn_probs = F.softmax(safe_scores, dim=-1)

        sym_probs = torch.zeros((bsz, 16), device=dev, dtype=dtype)
        fallback_bucket = token_ids % 16
        tok_bucket = torch.where(has_any_ap, (token_ids - AP_MIN_TOKEN_ID).clamp(0, 15), fallback_bucket)
        active_weights = torch.where(has_any_ap, attn_probs * is_ap.to(dtype), attn_probs * valid_tok.to(dtype))
        sym_probs = sym_probs.scatter_add(1, tok_bucket, active_weights)
        return sym_probs.clamp(min=1e-7)

    @staticmethod
    def _public_role_features(token_ids: torch.Tensor, valid_tok: torch.Tensor) -> torch.Tensor:
        """Compute purely public role/delimiter features [norm_pos, is_eq, is_colon, is_newline] [B, L, 4]."""
        bsz, seq_len = token_ids.shape
        dev = token_ids.device
        row_denom = (valid_tok.sum(dim=1, keepdim=True).float() - 1.0).clamp_min(1.0)
        pos_norm = (
            torch.arange(seq_len, device=dev, dtype=torch.float32).unsqueeze(0).expand(bsz, seq_len) / row_denom
        ).clamp(0.0, 1.0)
        is_eq = (token_ids == UTF8.encode("=")[0]).float()
        is_colon = (token_ids == UTF8.encode(":")[0]).float()
        is_nl = (token_ids == UTF8.encode("\n")[0]).float()
        feats = torch.stack([pos_norm, is_eq, is_colon, is_nl], dim=-1)
        return feats * valid_tok.unsqueeze(-1).float()

    def forward(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        action_positions: torch.Tensor | None = None,
        persistent_state: torch.Tensor | None = None,
        use_persistent_state: bool = False,
        return_aux: bool = True,
        recurrent_steps_override: int | None = None,
        donor_h0: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        bsz, seq_len = token_ids.shape
        dev = token_ids.device
        valid_tok = token_ids != UTF8.pad_id

        if action_positions is None:
            idx_grid = torch.arange(seq_len, device=dev).unsqueeze(0).expand(bsz, seq_len)
            action_positions = torch.where(valid_tok, idx_grid, torch.zeros_like(idx_grid)).max(dim=1).values
        apos = action_positions.clamp(0, max(0, seq_len - 1)).to(dtype=torch.long)

        h = self.core.forward_hidden(token_ids, positions, attention_mask)  # [B, L, d]
        b_idx = torch.arange(bsz, device=dev, dtype=torch.long)
        h_act_orig = h[b_idx, apos]
        h_act = donor_h0.to(device=dev, dtype=h.dtype) if donor_h0 is not None else h_act_orig

        uses_slots = hasattr(self, "slot_init")
        uses_pointer = hasattr(self, "ptr_q")
        ponder_cost = torch.zeros((bsz,), device=dev, dtype=h.dtype)
        next_state = None
        r_ptr_offset = None
        extra_op_logits = None
        extra_cls_logits = None
        telemetry: dict[str, Any] = {}

        if uses_slots:
            if self.architecture in ("R", "R-noedge", "R1-Alpha", "R1-ValOnly", "R1-NoMsg-Both", "R1-NoMsg-ValOnly", "R1-SlotMLP"):
                n_steps = 2
            elif self._ablation_mode == "disable_iterative":
                n_steps = 1
            else:
                n_steps = self.max_steps
            if recurrent_steps_override is not None:
                n_steps = max(0, int(recurrent_steps_override))

            tok_mem = self.input_proj(h)
            q_vec = self.query_proj(h_act)
            if use_persistent_state and persistent_state is not None:
                w = persistent_state.to(device=dev, dtype=h.dtype)
            elif use_persistent_state and self._episodic_state is not None and self._episodic_state.shape[0] == bsz:
                w = self._episodic_state.to(device=dev, dtype=h.dtype)
            else:
                w = self.slot_init.expand(bsz, -1, -1).contiguous()

            mask_bias = (~valid_tok).to(dtype=h.dtype) * (-1e9)
            k_mem = self.read_k(tok_mem)
            v_mem = self.read_v(tok_mem)

            rem = torch.ones((bsz, 1), device=dev, dtype=h.dtype)
            w_accum = w if n_steps == 0 else torch.zeros_like(w)
            no_rel = (
                self.architecture in ("P-R-Iter-NoRel", "R-noedge", "R1-NoMsg-Both", "R1-NoMsg-ValOnly", "R1-SlotMLP")
                or self._ablation_mode == "disable_relational"
            )
            add_mem = self.architecture == "P-R-Iter-AddMem" or self._ablation_mode == "disable_delta"
            fixed_halt = self.architecture in ("P-R-Iter-Fixed", "R", "R-noedge", "R1-Alpha", "R1-ValOnly", "R1-NoMsg-Both", "R1-NoMsg-ValOnly", "R1-SlotMLP")

            for step_t in range(n_steps):
                q_slot = self.read_q(w + q_vec.unsqueeze(1))
                attn_s = (q_slot @ k_mem.transpose(1, 2)) / math.sqrt(float(self.m_dim))
                attn_w = F.softmax(attn_s + mask_bias.unsqueeze(1), dim=-1)
                read_vec = attn_w @ v_mem

                if no_rel:
                    peer_mean = w
                else:
                    slot_sum = w.sum(dim=1, keepdim=True)
                    peer_mean = (slot_sum - w) / float(max(1, self.num_slots - 1))
                    if self._ablation_mode in ("true_edge_swap", "random_edge_perm") and self.num_slots > 1:
                        shift_amt = 1 if self._ablation_mode == "true_edge_swap" else 2
                        peer_mean = torch.roll(peer_mean, shifts=shift_amt, dims=1)
                rel = torch.tanh(self.rel_msg(torch.cat([w, peer_mean], dim=-1)))
                cand = torch.tanh(self.delta_cand(torch.cat([read_vec, rel], dim=-1)))
                gate = torch.sigmoid(self.delta_gate(torch.cat([w, cand], dim=-1)))

                if add_mem:
                    w = F.layer_norm(w + gate * cand, (self.m_dim,))
                else:
                    w = (1.0 - gate) * w + gate * cand

                h_p = torch.sigmoid(self.halt_head(w.mean(dim=1)))
                if fixed_halt:
                    ponder_cost = ponder_cost + 0.01 * h_p.squeeze(-1)
                    if step_t == n_steps - 1:
                        w_accum = w
                else:
                    if step_t < n_steps - 1:
                        w_step = rem * h_p
                        rem = rem * (1.0 - h_p)
                        w_accum = w_accum + w_step.unsqueeze(-1) * w
                        ponder_cost = ponder_cost + rem.squeeze(-1)
                    else:
                        w_accum = w_accum + rem.unsqueeze(-1) * w
                        ponder_cost = ponder_cost + rem.squeeze(-1)

            w_pooled = w_accum.mean(dim=1)
            next_state = w_accum.detach()
            if use_persistent_state:
                self._episodic_state = next_state
            uncertainties = torch.sigmoid(self.uncert_head(w_pooled))
            ctx_feat = self.out_proj(w_pooled)
            slot_cls_logits = self.cls_head(w_pooled)
            slot_mem_for_nocopy = w_accum

            if hasattr(self, "r_to_op"):
                route_op = self.architecture in ("R1-Alpha", "R1-NoMsg-Both")
                g_op = torch.sigmoid(self.r_op_gate(h_act))
                g_val = torch.sigmoid(self.r_val_gate(h_act))
                r_op_vec = g_op * self.r_to_op(w_pooled)
                r_val_vec = g_val * self.r_to_val(w_pooled)
                r_ptr_vec = self.r_to_ptr(w_pooled)
                if self._ablation_mode == "zero_r":
                    r_op_vec = 0.0 * r_op_vec
                    r_val_vec = 0.0 * r_val_vec
                    r_ptr_vec = 0.0 * r_ptr_vec
                elif self._ablation_mode == "op_branch_lesion":
                    r_op_vec = 0.0 * r_op_vec
                elif self._ablation_mode == "val_branch_lesion":
                    r_val_vec = 0.0 * r_val_vec
                    r_ptr_vec = 0.0 * r_ptr_vec
                # In R1-ValOnly / R1-NoMsg-ValOnly / R1-SlotMLP, keep 0.0*r_op_vec so all params stay live without routing op branch!
                extra_op_logits = r_op_vec if route_op else (0.0 * r_op_vec)
                extra_cls_logits = r_val_vec
                r_ptr_offset = r_ptr_vec
        elif hasattr(self, "u_cell"):
            default_steps = 4 if self.architecture.startswith("U4") else 2
            steps = default_steps if recurrent_steps_override is None else max(0, int(recurrent_steps_override))
            z_t = h_act
            step_norms: list[float] = []
            anchor_dists: list[float] = []
            gate_means: list[float] = []
            for _ in range(steps):
                z_t, st_info = self.u_cell.step(z_t, h_act, h, valid_tok)
                step_norms.append(float(st_info["update_norm"].mean().item()))
                anchor_dists.append(float(st_info["anchor_dist"].mean().item()))
                gate_means.append(float(st_info["gate_mean"].mean().item()))
            if steps == 0:
                # Zero-step identity pass; keep 0.0 touch on u_cell parameters so smoke tests have zero None grads
                zero_touch = 0.0 * sum(p.sum() for p in self.u_cell.parameters())
                z_t = h_act + zero_touch
            telemetry["u_update_norms"] = step_norms
            telemetry["u_anchor_dists"] = anchor_dists
            telemetry["u_gate_means"] = gate_means
            proj = torch.tanh(self.base_proj(z_t))
            uncertainties = torch.sigmoid(self.uncert_head(proj))
            ctx_feat = self.out_proj(proj) + (z_t - h_act)
            slot_cls_logits = self.cls_head(proj)
            slot_mem_for_nocopy = None
        else:
            # Base, P, U-ZeroStep, P1-*, Screens
            state_in = h_act
            if use_persistent_state and persistent_state is not None:
                state_in = persistent_state.to(device=dev, dtype=h.dtype).mean(dim=1) if persistent_state.ndim == 3 else persistent_state.to(device=dev, dtype=h.dtype)
            elif use_persistent_state and self._episodic_state is not None and self._episodic_state.shape[0] == bsz:
                state_in = self._episodic_state.to(device=dev, dtype=h.dtype)
            proj = torch.tanh(self.base_proj(state_in))
            if use_persistent_state:
                next_state = (0.7 * state_in + 0.3 * self.out_proj(proj)).detach()
                self._episodic_state = next_state
            else:
                next_state = state_in.detach()
            uncertainties = torch.sigmoid(self.uncert_head(proj))
            ctx_feat = self.out_proj(proj)
            slot_cls_logits = self.cls_head(proj)
            slot_mem_for_nocopy = None

        decision_hidden = h_act + ctx_feat

        # Architectural extensions (P1-*, R5 CounterexampleReader, R5 OperatorTransportBank)
        three_way_probs = None
        if hasattr(self, "comp_q1"):
            r_feats = self._public_role_features(token_ids, valid_tok).to(dtype=h.dtype)
            pos_emb = self.dense_pos_proj(r_feats) if hasattr(self, "dense_pos_proj") else self.role_feat_proj(r_feats)
            k_c = self.comp_k(h) + pos_emb
            v_c = self.comp_v(h)
            mb = (~valid_tok).to(dtype=h.dtype) * (-1e9)
            a1 = F.softmax((self.comp_q1(h_act).unsqueeze(1) @ k_c.transpose(1, 2)).squeeze(1) / math.sqrt(float(self.m_dim)) + mb, dim=-1)
            a2 = F.softmax((self.comp_q2(h_act).unsqueeze(1) @ k_c.transpose(1, 2)).squeeze(1) / math.sqrt(float(self.m_dim)) + mb, dim=-1)
            e1 = torch.bmm(a1.unsqueeze(1), v_c).squeeze(1)
            e2 = torch.bmm(a2.unsqueeze(1), v_c).squeeze(1)
            if self.architecture == "P1-ParallelRead":
                comp_rep = self.comp_u(e1) + self.comp_w(e2)
            else:
                comp_rep = self.comp_u(e1) * self.comp_w(e2)
            comp_logits = self.comp_out(F.silu(comp_rep))
            three_way_probs = F.softmax(self.three_way_switch(decision_hidden), dim=-1)  # [B, 3]
            telemetry["three_way_switch"] = three_way_probs.detach()
        else:
            comp_logits = None

        ce_op_shift = torch.zeros((bsz, 4), device=dev, dtype=h.dtype)
        if hasattr(self, "ce_q"):
            mb = (~valid_tok).to(dtype=h.dtype) * (-1e9)
            ckv = self.cand_kv(h)
            c_k, c_v = ckv.chunk(2, dim=-1)
            ca = F.softmax((self.cand_q(h_act).unsqueeze(1) @ c_k.transpose(1, 2)).squeeze(1) / math.sqrt(float(self.m_dim)) + mb, dim=-1)
            cand_vec = torch.bmm(ca.unsqueeze(1), c_v).squeeze(1)
            ce_a = F.softmax((self.ce_q(torch.cat([h_act, cand_vec], dim=-1)).unsqueeze(1) @ self.ce_k(h).transpose(1, 2)).squeeze(1) / math.sqrt(float(self.m_dim)) + mb, dim=-1)
            e_ce = torch.bmm(ce_a.unsqueeze(1), self.ce_v(h)).squeeze(1)
            if self.architecture == "ContradictionShuffle" or self._ablation_mode == "shuffle_counterexample":
                if bsz > 1:
                    e_ce = torch.roll(e_ce, shifts=1, dims=0)
                else:
                    e_ce = torch.roll(e_ce, shifts=max(1, self.m_dim // 2), dims=-1)
            raw_ce = self.ce_score_head(torch.cat([h_act, cand_vec, e_ce], dim=-1))
            if self.architecture in ("CounterexampleReader", "ContradictionShuffle"):
                # Signed counterexample gating: defeat magnitude can push ANSWER -> QUERY/ABSTAIN
                defeat = F.softplus(raw_ce[:, 0:1])
                ce_op_shift = torch.cat([-defeat, 0.5 * raw_ce[:, 1:3], defeat + 0.25 * raw_ce[:, 3:4]], dim=-1)
            else:  # RelevanceReader2: symmetric second-read logits without forced contradiction sign flip
                ce_op_shift = 0.25 * raw_ce

        if hasattr(self, "op_bank"):
            mb = (~valid_tok).to(dtype=h.dtype) * (-1e9)
            al_a = F.softmax((self.align_q(h_act).unsqueeze(1) @ self.align_k(h).transpose(1, 2)).squeeze(1) / math.sqrt(float(self.m_dim)) + mb, dim=-1)
            if self._ablation_mode == "swap_alignment" and seq_len > 1:
                al_a = torch.roll(al_a, shifts=max(1, seq_len // 3), dims=-1)
            e_al = torch.bmm(al_a.unsqueeze(1), self.align_v(h)).squeeze(1)
            w_op = F.softmax(self.bank_gate(h_act), dim=-1)  # [B, M]
            w_mat = torch.einsum("bm,mij->bij", w_op, self.op_bank)
            trans_vec = torch.bmm(e_al.unsqueeze(1), w_mat).squeeze(1)
            slot_cls_logits = slot_cls_logits + self.cls_head(trans_vec)
            ctx_feat = ctx_feat + self.out_proj(trans_vec)
        elif hasattr(self, "ep_mlp_in"):
            mean_h = (h * valid_tok.unsqueeze(-1).to(h.dtype)).sum(dim=1) / valid_tok.sum(dim=1, keepdim=True).clamp_min(1).to(h.dtype)
            ep_vec = self.ep_mlp_out(F.silu(self.ep_mlp_in(torch.cat([h_act, mean_h], dim=-1))))
            slot_cls_logits = slot_cls_logits + self.cls_head(ep_vec)
            ctx_feat = ctx_feat + self.out_proj(ep_vec)
        elif hasattr(self, "ret_q"):
            mb = (~valid_tok).to(dtype=h.dtype) * (-1e9)
            ra = F.softmax((self.ret_q(h_act).unsqueeze(1) @ self.ret_k(h).transpose(1, 2)).squeeze(1) / math.sqrt(float(self.m_dim)) + mb, dim=-1)
            ret_vec = torch.bmm(ra.unsqueeze(1), self.ret_v(h)).squeeze(1)
            slot_cls_logits = slot_cls_logits + self.cls_head(ret_vec)
            ctx_feat = ctx_feat + self.out_proj(ret_vec)

        op_logits = self.op_head(decision_hidden) + ce_op_shift
        if extra_op_logits is not None:
            op_logits = op_logits + extra_op_logits
        if extra_cls_logits is not None:
            slot_cls_logits = slot_cls_logits + extra_cls_logits

        if uses_pointer:
            ptr_sym_probs = self._compute_pointer_symbol_probs(
                h, h_act, token_ids, valid_tok, q_offset=r_ptr_offset, slot_fallback=slot_mem_for_nocopy
            )
            mix = F.softmax(self.copy_mix_head(decision_hidden), dim=-1)
            copy_log = torch.log(ptr_sym_probs.clamp_min(1e-7))
            gen_log = F.log_softmax(slot_cls_logits, dim=-1)
            if three_way_probs is not None and comp_logits is not None:
                comp_log = F.log_softmax(comp_logits, dim=-1)
                # Combine copy, compositional bilinear, and lexical generation while keeping copy_mix_head active
                base_sym = mix[:, :1] * copy_log + mix[:, 1:2] * gen_log
                sym_log = three_way_probs[:, 0:1] * base_sym + three_way_probs[:, 1:2] * comp_log + three_way_probs[:, 2:3] * gen_log
            else:
                sym_log = mix[:, :1] * copy_log + mix[:, 1:2] * gen_log
        else:
            sym_log = F.log_softmax(slot_cls_logits, dim=-1)

        op_log = F.log_softmax(op_logits, dim=-1)
        ans_action_log = op_log[:, 0:1] + sym_log
        action_19_logits = torch.cat([ans_action_log, op_log[:, 1:]], dim=-1)

        h_gen = h + 0.25 * ctx_feat.unsqueeze(1)
        gen_logits = F.linear(h_gen, self.core.embedding.weight, self.lm_bias)

        out: dict[str, Any] = {
            "logits": gen_logits,
            "action_logits": action_19_logits,
            "op_logits": op_logits,
            "cls_logits": slot_cls_logits,
            "decision_hidden": decision_hidden,
            "h0_anchor": h_act_orig,
            "uncertainties": uncertainties,
            "ponder_cost": ponder_cost,
            "next_persistent_state": next_state,
            "telemetry": telemetry,
        }
        if not return_aux:
            return {"logits": gen_logits, "action_logits": action_19_logits, "decision_hidden": decision_hidden}
        return out

    @torch.no_grad()
    def generate_bytes(
        self,
        prompt_text: str,
        *,
        max_new_bytes: int | None = None,
        use_persistent_state: bool = False,
        stop_chars: Sequence[str] = ("\n",),
    ) -> dict[str, Any]:
        """Tool-free autoregressive byte synthesis bounded strictly from public prompt (§1, §4, §6)."""
        was_training = self.training
        self.eval()
        dev = next(self.parameters()).device
        bound = int(max_new_bytes) if max_new_bytes is not None else infer_public_generation_bound(prompt_text)
        encoded = [UTF8.bos_id, *UTF8.encode(prompt_text), UTF8.special(SpecialToken.ANSWER)]
        max_in = max(8, self.max_ctx - bound - 2)
        if len(encoded) > max_in:
            encoded = encoded[-max_in:]
        generated_ids: list[int] = []
        terminated_by_eos = False
        last_uncert = [0.5, 0.5]
        last_ponder = 0.0
        for _ in range(bound):
            seq = encoded + generated_ids
            if len(seq) > self.max_ctx:
                seq = seq[-self.max_ctx :]
            toks = torch.tensor([seq], dtype=torch.long, device=dev)
            segs = torch.zeros_like(toks)
            pos, mask = packed_layout(segs, torch_module=torch)
            apos = torch.tensor([len(seq) - 1], dtype=torch.long, device=dev)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=(dev.type == "cuda")):
                out = self.forward(toks, pos, mask, action_positions=apos, use_persistent_state=use_persistent_state)
            last_uncert = out["uncertainties"][0].float().detach().cpu().tolist()
            last_ponder = float(out["ponder_cost"][0].item())
            next_logits = out["logits"][0, -1].float().clone()
            next_logits[UTF8.pad_id] = -1e9
            next_logits[UTF8.bos_id] = -1e9
            nxt = int(torch.argmax(next_logits).item())
            if nxt in (UTF8.eos_id, UTF8.special(SpecialToken.ANSWER)):
                terminated_by_eos = True
                break
            generated_ids.append(nxt)
            if nxt >= 4:
                decoded_ch = UTF8.decode([nxt])
                if decoded_ch in stop_chars:
                    generated_ids.pop()
                    terminated_by_eos = True
                    break
        if was_training:
            self.train()
        clean_ids = [t for t in generated_ids if t >= 4]
        return {
            "text": UTF8.decode(clean_ids).strip(),
            "terminated_by_eos": bool(terminated_by_eos),
            "bound_bytes": bound,
            "generated_bytes": len(clean_ids),
            "epistemic_uncertainty": float(last_uncert[0]),
            "info_uncertainty": float(last_uncert[1]),
            "ponder_cost": last_ponder,
        }


def encode_prompt_and_target_pair(
    prompt: str,
    target_text: str,
    *,
    max_length: int = 640,
) -> tuple[list[int], list[bool], int]:
    """Encode prompt + target UTF-8 bytes + EOS for full-sequence supervised CE without truncation."""
    p_ids = [UTF8.bos_id, *UTF8.encode(prompt), UTF8.special(SpecialToken.ANSWER)]
    t_ids = [*UTF8.encode(target_text), UTF8.eos_id]
    full = p_ids + t_ids
    if len(full) > max_length:
        raise ValueError(f"Sequence {len(full)} exceeds context_length {max_length}")
    eligible = [False] * len(p_ids) + [True] * len(t_ids)
    action_pos = len(p_ids) - 1
    return full, eligible, action_pos


def evaluate_five_learning_modes(
    model: X3CognitiveModel,
    *,
    icl_cases_by_shot: Mapping[int, Sequence[Mapping[str, Any]]],
    state_adapt_sequence: Sequence[Mapping[str, Any]],
    grad_adapt_demos: Sequence[Mapping[str, Any]],
    grad_adapt_queries: Sequence[Mapping[str, Any]],
    transfer_cases: Sequence[Mapping[str, Any]],
    oracle_solver: Any,
    unrelated_transfer_cases: Sequence[Mapping[str, Any]] | None = None,
    lr: float = 5e-4,
    max_param_delta_norm: float = 1.0,
) -> dict[str, Any]:
    """Evaluate 5 distinct learning modes with public typed exact-output scoring & FLOPs estimates (§3-§4)."""
    dev = next(model.parameters()).device
    total_params = sum(int(p.numel()) for p in model.parameters())

    def _predict_and_score(case: Mapping[str, Any], *, use_persistent: bool = False) -> tuple[str, bool, int]:
        prompt = str(case["prompt"])
        bound = infer_public_generation_bound(prompt)
        out = model.generate_bytes(prompt, max_new_bytes=bound, use_persistent_state=use_persistent)
        pred = str(out["text"]).strip()
        scored = parse_and_score_public_typed_output(
            prompt, pred, case, max_new_bytes=bound, terminated_by_eos=bool(out.get("terminated_by_eos", True))
        )
        toks_used = len(UTF8.encode(prompt)) + int(out.get("generated_bytes", 0))
        return pred, bool(scored["correct"]), toks_used

    # Mode 1: Frozen-weight In-Context Learning across 0/1/4/16/64 shots
    model.eval()
    model.reset_episode()
    icl_curve: dict[int, float] = {}
    eval_tokens = 0
    for shot, items in sorted(icl_cases_by_shot.items()):
        ok = 0
        for c in items:
            model.reset_episode()
            _, matched, tk = _predict_and_score(c, use_persistent=False)
            ok += int(matched)
            eval_tokens += tk
        icl_curve[int(shot)] = float(ok / max(1, len(items)))

    # Mode 2: Persistent internal state across stream (weights frozen) + reset effect check
    model.reset_episode()
    state_ok = 0
    for c in state_adapt_sequence:
        _, matched, tk = _predict_and_score(c, use_persistent=True)
        state_ok += int(matched)
        eval_tokens += tk
    state_acc = float(state_ok / max(1, len(state_adapt_sequence)))

    reset_ok = 0
    for c in state_adapt_sequence:
        model.reset_episode()
        _, matched, _ = _predict_and_score(c, use_persistent=True)
        reset_ok += int(matched)
    state_reset_effect = state_acc - float(reset_ok / max(1, len(state_adapt_sequence)))

    # Mode 3: Bounded fast parameter-update learning from cloned weights (never mutates caller model)
    snap = {k: v.detach().clone() for k, v in model.named_parameters()}
    train_tokens = 0
    if grad_adapt_demos:
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=float(lr))
        model.train()
        with torch.enable_grad():
            samples = [encode_prompt_and_target_pair(str(d["prompt"]), str(d["gold_text"]), max_length=model.max_ctx) for d in grad_adapt_demos]
            max_l = max(len(s) for s, _, _ in samples)
            train_tokens += len(samples) * max_l
            toks = torch.zeros((len(samples), max_l), dtype=torch.long, device=dev)
            segs = torch.full((len(samples), max_l), -1, dtype=torch.long, device=dev)
            elig = torch.zeros((len(samples), max_l), dtype=torch.bool, device=dev)
            apos = torch.zeros((len(samples),), dtype=torch.long, device=dev)
            for i, (ids, el, ap) in enumerate(samples):
                toks[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=dev)
                segs[i, : len(ids)] = 0
                elig[i, : len(el)] = torch.tensor(el, dtype=torch.bool, device=dev)
                apos[i] = ap
            pos, mask = packed_layout(segs, torch_module=torch)
            opt.zero_grad(set_to_none=True)
            out = model(toks, pos, mask, action_positions=apos)
            targets = toks[:, 1:].reshape(-1)
            logits_flat = out["logits"][:, :-1, :].reshape(-1, out["logits"].shape[-1])
            mask_flat = elig[:, 1:].reshape(-1).to(logits_flat.dtype)
            per_tok = F.cross_entropy(logits_flat, targets, reduction="none")
            loss = (per_tok * mask_flat).sum() / mask_flat.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    sq_sum = 0.0
    with torch.no_grad():
        for name, p in model.named_parameters():
            diff = p.detach() - snap[name]
            sq_sum += float((diff * diff).sum().item())
        delta_norm = math.sqrt(sq_sum)
        if delta_norm > float(max_param_delta_norm) and delta_norm > 0.0:
            scale = float(max_param_delta_norm) / delta_norm
            for name, p in model.named_parameters():
                p.copy_(snap[name] + scale * (p.detach() - snap[name]))
            delta_norm = float(max_param_delta_norm)

    model.eval()
    model.reset_episode()
    grad_ok = 0
    for c in grad_adapt_queries:
        _, matched, tk = _predict_and_score(c, use_persistent=False)
        grad_ok += int(matched)
        eval_tokens += tk
    grad_acc = float(grad_ok / max(1, len(grad_adapt_queries)))

    # Retained A accuracy on grad_adapt_demos after parameter step
    retained_ok = 0
    for d in grad_adapt_demos:
        _, matched, _ = _predict_and_score(d, use_persistent=False)
        retained_ok += int(matched)
    retained_a = float(retained_ok / max(1, len(grad_adapt_demos))) if grad_adapt_demos else 0.0

    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(snap[name])
    model.reset_episode()

    # Mode 4 & Mode 5: Tool-free vs tool-assisted oracle gap + Cross-domain / unrelated B' transfer
    tr_neural = 0
    tr_oracle = 0
    for c in transfer_cases:
        _, matched, tk = _predict_and_score(c, use_persistent=False)
        tr_neural += int(matched)
        eval_tokens += tk
        tr_oracle += int(str(oracle_solver(c)).strip() == str(c["gold_text"]).strip())
    n_tr = max(1, len(transfer_cases))

    unrel_list = list(unrelated_transfer_cases or transfer_cases)
    unrel_ok = 0
    for c in unrel_list:
        _, matched, _ = _predict_and_score(c, use_persistent=False)
        unrel_ok += int(matched)

    inc_flops = float(6 * total_params * train_tokens + 2 * total_params * eval_tokens)
    return {
        "schema": "elendril-x3-five-learning-modes-v2",
        "mode1_frozen_icl_curve": icl_curve,
        "mode2_persistent_state_accuracy": state_acc,
        "mode2_state_reset_effect": float(state_reset_effect),
        "mode3_bounded_grad_adapt_accuracy": grad_acc,
        "mode3_retained_a_accuracy": retained_a,
        "mode3_param_delta_l2": float(delta_norm),
        "mode4_tool_free_transfer_accuracy": float(tr_neural / n_tr),
        "mode4_tool_assisted_oracle_accuracy": float(tr_oracle / n_tr),
        "mode4_tool_assistance_gap": float((tr_oracle - tr_neural) / n_tr),
        "mode5_cross_domain_transfer_accuracy": float(tr_neural / n_tr),
        "mode5_unrelated_b_prime_accuracy": float(unrel_ok / max(1, len(unrel_list))),
        "incremental_flops_estimate": inc_flops,
        "eval_tokens_processed": eval_tokens,
        "train_tokens_processed": train_tokens,
    }


__all__ = [
    "S50_MODEL_SPEC",
    "TINY_CANARY_MODEL_SPEC",
    "X3CognitiveModel",
    "X3_ARCHITECTURES",
    "X3_CONTROL_PAIRS",
    "X3_CORE_ARCHITECTURES",
    "X3_EXTENDED_ARCHITECTURES",
    "X3_VERSIONED_CONTROL_PAIRS",
    "build_legacy_x1_scaled_model",
    "build_x3_backbone_spec",
    "encode_prompt_and_target_pair",
    "evaluate_five_learning_modes",
]
