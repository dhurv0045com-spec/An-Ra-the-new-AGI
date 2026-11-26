"""V8-X4 Repaired Causal Architectures, Unified Pointer-Assisted Byte Generation & Controls (~20M S20)."""

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
from anra_v5.elendril.v8x.data import ACTION_19_LABELS
from v5_contracts.model_spec import ModelSpec
from v5_model.core import initialize, packed_layout

from .battery import infer_x4_generation_bound

X4_ARCHITECTURES: tuple[str, ...] = (
    # Cycle 1: Repaired historical foundations + pointer restoration (R-Ref has pointer!)
    "Base",
    "P-Ref",
    "R-Ref",
    "R-NoEdge",
    # Cycle 3: P1 / R1 / U2-Fix / U4-Fix / U-P + Original Hypotheses + Controls
    "P1-Alpha",
    "P1-ParallelRead",
    "R1-Alpha",
    "R1-DenseControl",
    "U2-Fix",
    "U4-Fix",
    "U2-Unanchored",
    "U-P",
    "P1-OpBank",
    "P1-OpBank-DenseCtrl",
    "P1-CE-Reader",
    # Cycle 4: Capability-preserving integration
    "ANRA-X4-Int",
)

X4_CANARY_ARCHITECTURES: tuple[str, ...] = X4_ARCHITECTURES

TINY_X4_SPEC: ModelSpec = replace(
    MODEL_PROFILES["S20"],
    width=48,
    layers=2,
    query_heads=2,
    kv_heads=1,
    head_dimension=24,
    ffn_width=96,
    context_length=640,
)


def build_x4_backbone_spec(scale: str = "S20") -> ModelSpec:
    """Return S20 (~19.87M) or tiny (<100k) backbone spec. S50 is explicitly disallowed in X4."""
    s = str(scale).strip()
    if s in ("S20", "s20"):
        return replace(MODEL_PROFILES["S20"], context_length=640)
    if s in ("tiny", "canary", "TINY"):
        return TINY_X4_SPEC
    raise ValueError(f"Unsupported X4 scale {scale!r}: only 'tiny' and 'S20' (~20M) are authorized in X4 (no S50).")


class _AnchoredRefinementCell(nn.Module):
    """Anchored iterative refinement cell retaining explicit access to immutable anchor h0 (§11)."""

    def __init__(self, width: int, hidden_dim: int, *, anchored: bool = True) -> None:
        super().__init__()
        self.width = int(width)
        self.hidden_dim = int(hidden_dim)
        self.anchored = bool(anchored)
        self.q_proj = nn.Linear(self.width, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.width, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(self.width, self.hidden_dim, bias=False)
        self.in_fuse = nn.Linear(2 * self.width + self.hidden_dim, self.hidden_dim)
        self.delta_out = nn.Linear(self.hidden_dim, self.width)
        self.gate_proj = nn.Linear(2 * self.width, self.width)
        self.ln = nn.LayerNorm(self.width)
        # Initialize gate bias moderately negative (-1.5 -> sigmoid ~0.18) so step-0 grads are live and bounded!
        nn.init.constant_(self.gate_proj.bias, -1.5)

    def step(
        self,
        z_t: torch.Tensor,
        h0: torch.Tensor,
        h_bank: torch.Tensor,
        valid_prompt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        q = self.q_proj(z_t).unsqueeze(1)
        k = self.k_proj(h_bank)
        v = self.v_proj(h_bank)
        scores = (q @ k.transpose(1, 2)).squeeze(1).float() / math.sqrt(float(self.hidden_dim))
        weights = F.softmax(scores.masked_fill(~valid_prompt, -1e9), dim=-1).to(dtype=v.dtype)
        read_v = torch.bmm(weights.unsqueeze(1), v).squeeze(1)

        anchor_feat = h0 if self.anchored else z_t
        fused = F.silu(self.in_fuse(torch.cat([z_t, anchor_feat, read_v], dim=-1)))
        delta = self.ln(self.delta_out(fused))
        gate = torch.sigmoid(self.gate_proj(torch.cat([z_t, anchor_feat], dim=-1)).float()).to(dtype=z_t.dtype)
        z_next = z_t + gate * delta
        return z_next, {
            "gate_mean": gate.float().mean(dim=-1),
            "delta_norm": delta.float().norm(dim=-1),
            "anchor_dist": (z_next - h0).float().norm(dim=-1),
        }


class X4CognitiveModel(nn.Module):
    """Unified X4 model enforcing prefix-only causality, pointer-assisted byte generation & FP32 probability mixture."""

    def __init__(
        self,
        architecture: str,
        base_spec: ModelSpec,
        *,
        seed: int = 7,
        num_slots: int = 8,
        edge_hidden: int = 64,
    ) -> None:
        super().__init__()
        if architecture not in X4_ARCHITECTURES:
            raise ValueError(f"Unknown X4 architecture: {architecture!r}")
        self.architecture = str(architecture)
        self.base_spec = base_spec
        self.max_ctx = int(base_spec.context_length)
        self.vocab_size = int(base_spec.vocabulary_size)
        self.d_model = int(base_spec.width)
        self.m_dim = min(96, max(16, self.d_model // 8))
        self.num_slots = int(max(4, num_slots))
        self.edge_hidden = min(int(edge_hidden), self.m_dim)

        self.core = initialize(base_spec, int(seed), torch_module=torch)
        torch.manual_seed(int(seed) + 6103)

        d = self.d_model
        m = self.m_dim
        self._ablation_mode: str = "none"
        self._episodic_state: torch.Tensor | None = None
        self._episodic_history_len: int = 0

        self.lm_bias = nn.Parameter(torch.zeros(self.vocab_size))
        self.base_proj = nn.Linear(d, m)
        self.ctx_proj = nn.Linear(m, d)
        self.uncert_head = nn.Linear(m, 2)
        self.uncert_to_ctx = nn.Linear(2, d)
        nn.init.normal_(self.uncert_to_ctx.weight, std=0.02)
        # Precompute exact UTF-8 byte IDs for OP initial bytes ['A', 'Q', 'R'] and action-19 map
        op_first_bytes = [UTF8.encode(ch)[0] for ch in ("A", "Q", "R", "A")]
        self.register_buffer("_op_byte_ids", torch.tensor(op_first_bytes, dtype=torch.long), persistent=False)
        compat_bytes = [UTF8.encode(lbl if len(lbl) == 1 else lbl[0])[0] for lbl in ACTION_19_LABELS]
        self.register_buffer("_compat_byte_ids", torch.tensor(compat_bytes, dtype=torch.long), persistent=False)

        # Modular architectural flags
        is_r_family = self.architecture in ("R-Ref", "R-NoEdge", "R1-Alpha", "R1-DenseControl", "ANRA-X4-Int")
        is_p1 = self.architecture in ("P1-Alpha", "P1-ParallelRead", "P1-OpBank", "P1-OpBank-DenseCtrl", "ANRA-X4-Int")
        is_u = self.architecture in ("U2-Fix", "U4-Fix", "U2-Unanchored", "U-P", "ANRA-X4-Int")
        is_opbank = self.architecture in ("P1-OpBank", "ANRA-X4-Int")
        is_opbank_dense = self.architecture == "P1-OpBank-DenseCtrl"
        is_ce = self.architecture == "P1-CE-Reader"
        # Defect A Fix: R-Ref and R-NoEdge explicitly retain pointer copy access!
        uses_pointer = self.architecture != "Base"

        self.uses_slots = is_r_family
        self.uses_pointer = uses_pointer
        self.uses_p1 = is_p1
        self.uses_u = is_u
        self.uses_opbank = is_opbank
        self.uses_opbank_dense = is_opbank_dense
        self.uses_ce = is_ce

        if uses_pointer:
            self.ptr_dec_q = nn.Linear(d, m)
            self.ptr_k = nn.Linear(d, m)
            self.copy_gate = nn.Linear(d, 1)
            nn.init.constant_(self.copy_gate.bias, -1.0)

        if is_r_family:
            self.slot_init = nn.Parameter(torch.randn(1, self.num_slots, m) * 0.02)
            self.tok_slot_proj = nn.Linear(d, m)
            self.slot_q = nn.Linear(m, m, bias=False)
            self.slot_k = nn.Linear(m, m, bias=False)
            self.slot_v = nn.Linear(m, m, bias=False)
            self.slot_gru = nn.GRUCell(m, m)
            if self.architecture == "R-NoEdge":
                self.noedge_mlp = nn.Sequential(nn.Linear(m, m), nn.SiLU(), nn.Linear(m, m))
            elif self.architecture == "R1-DenseControl":
                self.r1_dense = nn.Sequential(nn.Linear(2 * m, 2 * m), nn.SiLU(), nn.Linear(2 * m, m))
            else:
                # Pairwise directed edge_mlp (3*m -> edge_hidden -> n_edge_types) for R-Ref / R1-Alpha / ANRA-X4-Int
                n_edge_types = 4 if self.architecture in ("R1-Alpha", "ANRA-X4-Int") else 1
                self.edge_mlp = nn.Sequential(
                    nn.Linear(3 * m, self.edge_hidden),
                    nn.SiLU(),
                    nn.Linear(self.edge_hidden, n_edge_types),
                )
                self.edge_msg_proj = nn.Linear(m * n_edge_types, m)
            self.rel_gru = nn.GRUCell(m, m)
            self.r_to_ptr = nn.Linear(m, m)
            self.r_to_op = nn.Linear(m, 4)

        if is_p1:
            self.p1_q = nn.Linear(m, m, bias=False)
            self.p1_k = nn.Linear(d, m, bias=False)
            self.p1_v = nn.Linear(d, m, bias=False)
            self.p1_role_proj = nn.Linear(4, m)
            self.p1_gru = nn.GRUCell(m, m)
            self.p1_out = nn.Linear(m, m)

        if is_u:
            self.u_default_steps = 4 if self.architecture == "U4-Fix" else 2
            anchored = self.architecture != "U2-Unanchored"
            self.u_cell = _AnchoredRefinementCell(d, m, anchored=anchored)

        if is_opbank:
            self.n_operators = 8
            self.op_rank = max(8, m // 4)
            self.op_align_q = nn.Linear(d, m, bias=False)
            self.op_align_k = nn.Linear(d, m, bias=False)
            self.op_align_v = nn.Linear(d, m, bias=False)
            self.op_router = nn.Linear(d, self.n_operators)
            self.op_U = nn.Parameter(torch.randn(self.n_operators, m, self.op_rank) * 0.04)
            self.op_V = nn.Parameter(torch.randn(self.n_operators, self.op_rank, m) * 0.04)
            self.op_scale = nn.Parameter(torch.tensor(0.20))
        elif is_opbank_dense:
            self.op_align_q = nn.Linear(d, m, bias=False)
            self.op_align_k = nn.Linear(d, m, bias=False)
            self.op_align_v = nn.Linear(d, m, bias=False)
            self.op_dense_mlp = nn.Sequential(nn.Linear(m, 2 * m), nn.SiLU(), nn.Linear(2 * m, m))
            self.op_scale = nn.Parameter(torch.tensor(0.20))

        if is_ce:
            self.ce_supp_q = nn.Linear(d, m, bias=False)
            self.ce_supp_k = nn.Linear(d, m, bias=False)
            self.ce_supp_v = nn.Linear(d, m, bias=False)
            self.ce_ctr_q = nn.Linear(d + m, m, bias=False)
            self.ce_ctr_k = nn.Linear(d, m, bias=False)
            self.ce_ctr_v = nn.Linear(d, m, bias=False)
            self.ce_fuse = nn.Linear(2 * m, m)
            self.ce_conflict_gate = nn.Linear(2 * m, 1)

    def reset_episode(self) -> None:
        self._episodic_state = None
        self._episodic_history_len = 0
        self._ablation_mode = "none"

    def update_persistent_state(self, feedback_text: str) -> None:
        """Explicitly fold verified feedback string into frozen-weight episodic state (§17 Mode 2)."""
        dev = next(self.parameters()).device
        ids = [UTF8.bos_id, *UTF8.encode(feedback_text)[: self.max_ctx - 2], UTF8.special(SpecialToken.ANSWER)]
        toks = torch.tensor([ids], dtype=torch.long, device=dev)
        segs = torch.zeros_like(toks)
        pos, mask = packed_layout(segs, torch_module=torch)
        apos = torch.tensor([len(ids) - 1], dtype=torch.long, device=dev)
        with torch.no_grad():
            out = self.forward(toks, pos, mask, prompt_end_positions=apos, use_persistent_state=True)
        self._episodic_state = out["next_persistent_state"]
        self._episodic_history_len += 1

    @contextmanager
    def ablate_mechanism(self, mode: str = "disable_pointer") -> Any:
        valid = {"none", "disable_pointer", "disable_edge_mlp", "disable_u", "disable_opbank"}
        if mode not in valid:
            raise ValueError(f"Unsupported ablation mode {mode!r}")
        prev = self._ablation_mode
        self._ablation_mode = str(mode)
        try:
            yield mode
        finally:
            self._ablation_mode = prev

    @staticmethod
    def _public_role_feats(token_ids: torch.Tensor, valid_prompt: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = token_ids.shape
        dev = token_ids.device
        denom = (valid_prompt.sum(dim=1, keepdim=True).float() - 1.0).clamp_min(1.0)
        pos_norm = (torch.arange(seq_len, device=dev, dtype=torch.float32).unsqueeze(0).expand(bsz, seq_len) / denom).clamp(0.0, 1.0)
        is_eq = (token_ids == UTF8.encode("=")[0]).float()
        is_col = (token_ids == UTF8.encode(":")[0]).float()
        is_bar = (token_ids == UTF8.encode("|")[0]).float()
        return torch.stack([pos_norm, is_eq, is_col, is_bar], dim=-1) * valid_prompt.unsqueeze(-1).float()

    def _run_prompt_cognitive_modules(
        self,
        h: torch.Tensor,
        h_prompt: torch.Tensor,
        token_ids: torch.Tensor,
        valid_prompt: torch.Tensor,
        *,
        use_persistent_state: bool = False,
        persistent_state: torch.Tensor | None = None,
        eval_steps_override: int | None = None,
    ) -> dict[str, Any]:
        """Compute cognitive prompt representation strictly over `valid_prompt` (0..prompt_end)."""
        bsz, seq_len, _ = h.shape
        dev = h.device
        telemetry: dict[str, Any] = {}
        r_ptr_offset = torch.zeros((bsz, self.m_dim), device=dev, dtype=torch.float32)

        # Base pooled state from prompt_end representation
        state_vec = self.base_proj(h_prompt)  # [B, m]
        if use_persistent_state:
            prev = persistent_state if persistent_state is not None else self._episodic_state
            if prev is not None and prev.shape == state_vec.shape:
                state_vec = 0.65 * state_vec + 0.35 * prev.to(device=dev, dtype=state_vec.dtype)
        m_acc = torch.tanh(state_vec)

        op_bias = torch.zeros((bsz, 4), device=dev, dtype=torch.float32)
        if self.uses_slots:
            tok_mem = self.tok_slot_proj(h)
            w = self.slot_init.expand(bsz, -1, -1).contiguous()
            k_s = self.slot_k(tok_mem)
            v_s = self.slot_v(tok_mem)
            vp_slot = valid_prompt.unsqueeze(1)
            for _ in range(2):
                q_s = self.slot_q(w + m_acc.unsqueeze(1))
                sc = (q_s @ k_s.transpose(1, 2)).float() / math.sqrt(float(self.m_dim))
                aw = F.softmax(sc.masked_fill(~vp_slot, -1e9), dim=-1).to(dtype=v_s.dtype)
                read_s = aw @ v_s
                w = self.slot_gru(read_s.reshape(-1, self.m_dim), w.reshape(-1, self.m_dim)).reshape(bsz, self.num_slots, self.m_dim)

            # Pairwise relational communication (Restored edge_mlp on R-Ref, R1-Alpha, ANRA-X4-Int!)
            if self.architecture == "R-NoEdge" or self._ablation_mode == "disable_edge_mlp":
                msg = self.noedge_mlp(w) if hasattr(self, "noedge_mlp") else w
            elif self.architecture == "R1-DenseControl":
                peer_mean = (w.sum(dim=1, keepdim=True) - w) / float(max(1, self.num_slots - 1))
                msg = self.r1_dense(torch.cat([w, peer_mean], dim=-1))
            else:
                k_n = self.num_slots
                wi = w.unsqueeze(2).expand(bsz, k_n, k_n, self.m_dim)
                wj = w.unsqueeze(1).expand(bsz, k_n, k_n, self.m_dim)
                q_ctx = m_acc.unsqueeze(1).unsqueeze(2).expand(bsz, k_n, k_n, self.m_dim)
                e_logits = self.edge_mlp(torch.cat([wi, wj, q_ctx], dim=-1)).float()  # [B, K, K, T]
                eye = torch.eye(k_n, device=dev, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)
                e_w = F.softmax(e_logits.masked_fill(eye, -1e9), dim=2).to(dtype=w.dtype)
                # [B, K, T, K] @ [B, K, m] via einsum -> [B, K, T*m]
                msgs_t = torch.einsum("bikt,bkm->bitm", e_w, w).reshape(bsz, k_n, -1)
                msg = self.edge_msg_proj(msgs_t)
            w = self.rel_gru(msg.reshape(-1, self.m_dim), w.reshape(-1, self.m_dim)).reshape(bsz, self.num_slots, self.m_dim)
            w_pool = w.mean(dim=1)
            m_acc = m_acc + 0.35 * torch.tanh(w_pool)
            r_ptr_offset = r_ptr_offset + self.r_to_ptr(w_pool).float()
            op_bias = op_bias + self.r_to_op(w_pool).float()

        if self.uses_p1:
            r_feats = self._public_role_feats(token_ids, valid_prompt).to(dtype=h.dtype)
            k_p1 = self.p1_k(h) + self.p1_role_proj(r_feats)
            v_p1 = self.p1_v(h)
            w_p1 = m_acc
            if self.architecture == "P1-ParallelRead":
                # Matched single-step parallel multi-hop average
                reads = []
                for _ in range(3):
                    sc = (self.p1_q(w_p1).unsqueeze(1) @ k_p1.transpose(1, 2)).squeeze(1).float() / math.sqrt(float(self.m_dim))
                    aw = F.softmax(sc.masked_fill(~valid_prompt, -1e9), dim=-1).to(dtype=v_p1.dtype)
                    reads.append(torch.bmm(aw.unsqueeze(1), v_p1).squeeze(1))
                w_p1 = self.p1_gru(sum(reads) / 3.0, w_p1)
            else:
                for _ in range(3):
                    sc = (self.p1_q(w_p1).unsqueeze(1) @ k_p1.transpose(1, 2)).squeeze(1).float() / math.sqrt(float(self.m_dim))
                    aw = F.softmax(sc.masked_fill(~valid_prompt, -1e9), dim=-1).to(dtype=v_p1.dtype)
                    rd = torch.bmm(aw.unsqueeze(1), v_p1).squeeze(1)
                    w_p1 = self.p1_gru(rd, w_p1)
            p1_vec = self.p1_out(w_p1)
            m_acc = m_acc + 0.35 * torch.tanh(p1_vec)
            r_ptr_offset = r_ptr_offset + 0.35 * p1_vec.float()

        if self.uses_opbank and self._ablation_mode != "disable_opbank":
            sc = (self.op_align_q(h_prompt).unsqueeze(1) @ self.op_align_k(h).transpose(1, 2)).squeeze(1).float() / math.sqrt(float(self.m_dim))
            aw = F.softmax(sc.masked_fill(~valid_prompt, -1e9), dim=-1).to(dtype=h.dtype)
            e_al = torch.bmm(aw.unsqueeze(1), self.op_align_v(h)).squeeze(1)
            w_op = F.softmax(self.op_router(h_prompt).float(), dim=-1)  # [B, M]
            # Low-rank operator application: e_al @ U_m @ V_m -> [B, m]
            u_mats = torch.einsum("bm,mir->bir", w_op, self.op_U.float())
            v_mats = torch.einsum("bm,mrj->brj", w_op, self.op_V.float())
            mid = torch.bmm(e_al.float().unsqueeze(1), u_mats)
            op_vec = torch.bmm(mid, v_mats).squeeze(1).to(dtype=h.dtype)
            gate_op = self.op_scale.clamp(-0.5, 0.5).to(dtype=h.dtype)
            m_acc = m_acc + gate_op * torch.tanh(op_vec)
            telemetry["op_routing_weights"] = w_op.detach().cpu()
        elif self.uses_opbank_dense:
            sc = (self.op_align_q(h_prompt).unsqueeze(1) @ self.op_align_k(h).transpose(1, 2)).squeeze(1).float() / math.sqrt(float(self.m_dim))
            aw = F.softmax(sc.masked_fill(~valid_prompt, -1e9), dim=-1).to(dtype=h.dtype)
            e_al = torch.bmm(aw.unsqueeze(1), self.op_align_v(h)).squeeze(1)
            op_vec = self.op_dense_mlp(e_al)
            gate_op = self.op_scale.clamp(-0.5, 0.5).to(dtype=h.dtype)
            m_acc = m_acc + gate_op * torch.tanh(op_vec)

        if self.uses_ce:
            s_sc = (self.ce_supp_q(h_prompt).unsqueeze(1) @ self.ce_supp_k(h).transpose(1, 2)).squeeze(1).float() / math.sqrt(float(self.m_dim))
            s_aw = F.softmax(s_sc.masked_fill(~valid_prompt, -1e9), dim=-1).to(dtype=h.dtype)
            e_supp = torch.bmm(s_aw.unsqueeze(1), self.ce_supp_v(h)).squeeze(1)
            c_sc = (self.ce_ctr_q(torch.cat([h_prompt, e_supp], dim=-1)).unsqueeze(1) @ self.ce_ctr_k(h).transpose(1, 2)).squeeze(1).float() / math.sqrt(float(self.m_dim))
            c_aw = F.softmax(c_sc.masked_fill(~valid_prompt, -1e9), dim=-1).to(dtype=h.dtype)
            e_ctr = torch.bmm(c_aw.unsqueeze(1), self.ce_ctr_v(h)).squeeze(1)
            c_gate = torch.sigmoid(self.ce_conflict_gate(torch.cat([e_supp, e_ctr], dim=-1)).float()).to(dtype=h.dtype)
            ce_vec = self.ce_fuse(torch.cat([e_supp, (1.0 - 2.0 * c_gate) * e_ctr], dim=-1))
            m_acc = m_acc + 0.35 * torch.tanh(ce_vec)
            telemetry["ce_conflict_gate"] = c_gate.float().mean().item()

        h_refined = h_prompt
        if self.uses_u and self._ablation_mode != "disable_u":
            steps = self.u_default_steps if eval_steps_override is None else max(0, int(eval_steps_override))
            z_t = h_prompt
            norms, dists, gates = [], [], []
            for _ in range(steps):
                z_t, st = self.u_cell.step(z_t, h_prompt, h, valid_prompt)
                norms.append(float(st["delta_norm"].mean().item()))
                dists.append(float(st["anchor_dist"].mean().item()))
                gates.append(float(st["gate_mean"].mean().item()))
            if steps == 0:
                z_t = h_prompt + 0.0 * sum(p.sum() for p in self.u_cell.parameters())
            h_refined = z_t
            telemetry["u_steps"] = steps
            telemetry["u_delta_norms"] = norms
            telemetry["u_anchor_dists"] = dists
            telemetry["u_gate_means"] = gates

        uncert = torch.sigmoid(self.uncert_head(m_acc).float())
        prompt_ctx = self.ctx_proj(m_acc) + self.uncert_to_ctx(uncert.to(dtype=h.dtype)) + (h_refined - h_prompt)
        return {
            "prompt_ctx": prompt_ctx,
            "r_ptr_offset": r_ptr_offset,
            "op_bias": op_bias,
            "uncertainties": uncert,
            "next_persistent_state": m_acc.detach(),
            "telemetry": telemetry,
        }

    def forward(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        prompt_end_positions: torch.Tensor | None = None,
        action_positions: torch.Tensor | None = None,
        persistent_state: torch.Tensor | None = None,
        use_persistent_state: bool = False,
        eval_steps_override: int | None = None,
        debug_leak_future: bool = False,
    ) -> dict[str, Any]:
        """Strictly causal forward pass emitting unified copy+generate log-probabilities (`log_p_unified`)."""
        bsz, seq_len = token_ids.shape
        dev = token_ids.device
        valid_tok = token_ids != UTF8.pad_id

        end_pos = prompt_end_positions if prompt_end_positions is not None else action_positions
        idx_grid = torch.arange(seq_len, device=dev).unsqueeze(0).expand(bsz, seq_len)
        if end_pos is None:
            end_pos = torch.where(valid_tok, idx_grid, torch.zeros_like(idx_grid)).max(dim=1).values
        end_pos = end_pos.clamp(0, max(0, seq_len - 1)).to(dtype=torch.long)

        # CRITICAL INVARIANT (Defect C Fix): Readers ONLY ever see non-pad tokens at position <= prompt_end!
        if debug_leak_future:
            valid_prompt = valid_tok
        else:
            valid_prompt = valid_tok & (idx_grid <= end_pos.unsqueeze(1))

        h = self.core.forward_hidden(token_ids, positions, attention_mask)  # [B, L, d]
        b_idx = torch.arange(bsz, device=dev, dtype=torch.long)
        h_prompt = h[b_idx, end_pos]

        cog = self._run_prompt_cognitive_modules(
            h,
            h_prompt,
            token_ids,
            valid_prompt,
            use_persistent_state=use_persistent_state,
            persistent_state=persistent_state,
            eval_steps_override=eval_steps_override,
        )
        prompt_ctx = cog["prompt_ctx"]

        # Inject prompt_ctx strictly at target-generation positions t >= prompt_end (zero retroactive contamination!)
        after_mask = (idx_grid >= end_pos.unsqueeze(1)).unsqueeze(-1).to(dtype=h.dtype)
        h_gen = h + after_mask * (0.25 * prompt_ctx.unsqueeze(1))
        gen_logits = F.linear(h_gen, self.core.embedding.weight, self.lm_bias).float()  # [B, L, V]
        if self.uses_slots:
            # Route relational op bias directly onto initial op bytes at prompt_end position (A/Q/R)
            at_end = (idx_grid == end_pos.unsqueeze(1)).unsqueeze(-1).float()
            op_scatter = torch.zeros((bsz, 1, self.vocab_size), device=dev, dtype=torch.float32)
            op_scatter.scatter_add_(2, self._op_byte_ids.view(1, 1, 4).expand(bsz, 1, 4), cog["op_bias"].unsqueeze(1))
            gen_logits = gen_logits + at_end * op_scatter

        # Normalized FP32 probability-space copy + generate mixture (Defects B, D, E fixed!)
        p_gen = F.softmax(gen_logits, dim=-1)  # [B, L, V]
        if self.uses_pointer:
            q_dec = self.ptr_dec_q(h_gen).float() + cog["r_ptr_offset"].unsqueeze(1)  # [B, L, m]
            k_p = self.ptr_k(h).float()  # [B, L, m]
            ptr_scores = (q_dec @ k_p.transpose(1, 2)) / math.sqrt(float(self.m_dim))  # [B, L, L]
            # Mask pointer strictly to prompt positions 0..prompt_end (excluding BOS/PAD)
            ptr_cand = valid_prompt & (token_ids != UTF8.bos_id)
            has_cand = ptr_cand.any(dim=1, keepdim=True)  # [B, 1]
            safe_ptr_mask = torch.where(has_cand, ptr_cand, valid_prompt).unsqueeze(1).expand(bsz, seq_len, seq_len)
            attn_ptr = F.softmax(ptr_scores.masked_fill(~safe_ptr_mask, -1e9), dim=-1)  # [B, L, L]
            p_copy = torch.zeros((bsz, seq_len, self.vocab_size), device=dev, dtype=torch.float32)
            p_copy.scatter_add_(2, token_ids.clamp(0, self.vocab_size - 1).unsqueeze(1).expand(bsz, seq_len, seq_len), attn_ptr)

            copy_active = (
                torch.zeros((bsz, 1, 1), device=dev, dtype=torch.float32)
                if self._ablation_mode == "disable_pointer"
                else has_cand.unsqueeze(-1).float()
            )
            g_t = torch.sigmoid(self.copy_gate(h_gen.float())) * copy_active  # [B, L, 1]
            p_unified = ((1.0 - g_t) * p_gen + g_t * p_copy).clamp_min(1e-12)
            p_unified = p_unified / p_unified.sum(dim=-1, keepdim=True)
        else:
            g_t = torch.zeros((bsz, seq_len, 1), device=dev, dtype=torch.float32)
            p_unified = p_gen.clamp_min(1e-12)
            p_unified = p_unified / p_unified.sum(dim=-1, keepdim=True)

        log_p_unified = torch.log(p_unified)
        decision_vec = h_gen[b_idx, end_pos]
        action_logits = F.log_softmax(log_p_unified[b_idx, end_pos][:, self._compat_byte_ids].float(), dim=-1)
        return {
            "logits": log_p_unified,
            "log_p_unified": log_p_unified,
            "p_unified": p_unified,
            "gen_logits": gen_logits.float(),
            "copy_gate_mean": g_t[b_idx, end_pos, 0],
            "action_logits": action_logits,
            "decision_hidden": decision_vec,
            "uncertainties": cog["uncertainties"],
            "next_persistent_state": cog["next_persistent_state"],
            "telemetry": cog["telemetry"],
        }

    @torch.no_grad()
    def generate_bytes(
        self,
        prompt_text: str,
        *,
        max_new_bytes: int | None = None,
        use_persistent_state: bool = False,
        eval_steps_override: int | None = None,
        stop_chars: Sequence[str] = ("\n",),
    ) -> dict[str, Any]:
        """Tool-free autoregressive decoding with FIXED `prompt_end` matching training context exactly (Defect H)."""
        out_list = self.generate_bytes_batch(
            [prompt_text],
            max_new_bytes_list=[max_new_bytes if max_new_bytes is not None else infer_x4_generation_bound(prompt_text)],
            use_persistent_state=use_persistent_state,
            eval_steps_override=eval_steps_override,
            stop_chars=stop_chars,
        )
        return out_list[0]

    @torch.no_grad()
    def generate_bytes_batch(
        self,
        prompts: Sequence[str],
        *,
        max_new_bytes_list: Sequence[int] | None = None,
        use_persistent_state: bool = False,
        eval_steps_override: int | None = None,
        stop_chars: Sequence[str] = ("\n",),
    ) -> list[dict[str, Any]]:
        """Length-aware batched greedy decoding keeping each sequence's `prompt_end` fixed across all steps (§21)."""
        if not prompts:
            return []
        was_training = self.training
        self.eval()
        dev = next(self.parameters()).device
        bounds = (
            list(max_new_bytes_list)
            if max_new_bytes_list is not None
            else [infer_x4_generation_bound(p) for p in prompts]
        )
        bsz = len(prompts)
        seqs: list[list[int]] = []
        prompt_ends: list[int] = []
        for p, bd in zip(prompts, bounds):
            ids = [UTF8.bos_id, *UTF8.encode(str(p)), UTF8.special(SpecialToken.ANSWER)]
            max_in = max(8, self.max_ctx - int(bd) - 2)
            if len(ids) > max_in:
                ids = ids[-max_in:]
            seqs.append(ids)
            prompt_ends.append(len(ids) - 1)

        max_steps = max(bounds)
        gen_tokens: list[list[int]] = [[] for _ in range(bsz)]
        log_prob_sums = [0.0 for _ in range(bsz)]
        finished = [False for _ in range(bsz)]
        terminated_eos = [False for _ in range(bsz)]
        last_uncert = [[0.5, 0.5] for _ in range(bsz)]
        last_op_w: list[list[float] | None] = [None for _ in range(bsz)]

        for step_i in range(max_steps):
            active_idx = [i for i in range(bsz) if not finished[i] and step_i < bounds[i]]
            if not active_idx:
                break
            cur_seqs = [seqs[i] + gen_tokens[i] for i in active_idx]
            cur_ends = []
            for idx_k, seq in zip(range(len(active_idx)), cur_seqs):
                orig_end = prompt_ends[active_idx[idx_k]]
                if len(seq) > self.max_ctx:
                    shift = len(seq) - self.max_ctx
                    cur_seqs[idx_k] = seq[-self.max_ctx :]
                    cur_ends.append(max(0, orig_end - shift))
                else:
                    cur_ends.append(orig_end)

            max_l = max(len(s) for s in cur_seqs)
            toks = torch.zeros((len(active_idx), max_l), dtype=torch.long, device=dev)
            segs = torch.full((len(active_idx), max_l), -1, dtype=torch.long, device=dev)
            last_pos = torch.zeros((len(active_idx),), dtype=torch.long, device=dev)
            for r_i, s in enumerate(cur_seqs):
                toks[r_i, : len(s)] = torch.tensor(s, dtype=torch.long, device=dev)
                segs[r_i, : len(s)] = 0
                last_pos[r_i] = len(s) - 1
            p_end_t = torch.tensor(cur_ends, dtype=torch.long, device=dev)
            pos, mask = packed_layout(segs, torch_module=torch)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(dev.type == "cuda")):
                out = self.forward(
                    toks,
                    pos,
                    mask,
                    prompt_end_positions=p_end_t,
                    use_persistent_state=use_persistent_state,
                    eval_steps_override=eval_steps_override,
                )
            log_p_step = out["log_p_unified"][torch.arange(len(active_idx), device=dev), last_pos].float().detach().cpu()
            unc_step = out["uncertainties"].float().detach().cpu().tolist()
            op_w_t = out["telemetry"].get("op_routing_weights")

            for r_i, orig_i in enumerate(active_idx):
                last_uncert[orig_i] = unc_step[r_i]
                if op_w_t is not None and last_op_w[orig_i] is None:
                    last_op_w[orig_i] = [float(x) for x in op_w_t[r_i].tolist()]
                lp_row = log_p_step[r_i].clone()
                lp_row[UTF8.pad_id] = -1e9
                lp_row[UTF8.bos_id] = -1e9
                nxt = int(torch.argmax(lp_row).item())
                lp_val = float(log_p_step[r_i, nxt].item())
                log_prob_sums[orig_i] += lp_val
                if nxt in (UTF8.eos_id, UTF8.special(SpecialToken.ANSWER)) or UTF8.is_ascii_stop_token(nxt, stop_chars):
                    finished[orig_i] = True
                    terminated_eos[orig_i] = True
                else:
                    gen_tokens[orig_i].append(nxt)

        if was_training:
            self.train()

        results: list[dict[str, Any]] = []
        for i in range(bsz):
            clean = [t for t in gen_tokens[i] if 4 <= t < 260]
            txt, valid_utf8 = UTF8.decode_lossy(clean, strip_special=True)
            denom_len = max(1, len(clean) + int(terminated_eos[i]))
            results.append(
                {
                    "text": txt.strip(),
                    "valid_utf8": bool(valid_utf8),
                    "terminated_by_eos": bool(terminated_eos[i]),
                    "bound_bytes": int(bounds[i]),
                    "generated_bytes": len(clean),
                    "seq_log_prob": float(log_prob_sums[i]),
                    "mean_token_prob": float(math.exp(max(-30.0, min(0.0, log_prob_sums[i] / denom_len)))),
                    "epistemic_uncertainty": float(last_uncert[i][0]),
                    "info_uncertainty": float(last_uncert[i][1]),
                    "op_routing_weights": last_op_w[i],
                }
            )
        return results


def encode_prompt_and_target_pair(
    prompt: str,
    target_text: str,
    *,
    max_length: int = 640,
) -> tuple[list[int], list[bool], int]:
    """Encode `[BOS, prompt..., ANSWER, target_bytes..., EOS]` with exact `prompt_end` anchor."""
    p_ids = [UTF8.bos_id, *UTF8.encode(prompt), UTF8.special(SpecialToken.ANSWER)]
    t_ids = [*UTF8.encode(target_text), UTF8.eos_id]
    full = p_ids + t_ids
    if len(full) > max_length:
        raise ValueError(f"Sequence len {len(full)} exceeds max_length {max_length}")
    eligible = [False] * len(p_ids) + [True] * len(t_ids)
    prompt_end = len(p_ids) - 1
    return full, eligible, prompt_end


__all__ = [
    "TINY_X4_SPEC",
    "X4CognitiveModel",
    "X4_ARCHITECTURES",
    "X4_CANARY_ARCHITECTURES",
    "build_x4_backbone_spec",
    "encode_prompt_and_target_pair",
]
