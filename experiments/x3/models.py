"""V8-X3 Cognitive Reasoning Architectures (`P-R-Iter`, 4 Matched Controls, S20 & S50 Scaling) (§6-§9, §11-§13)."""

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
from anra_v5.elendril.v8x.models import AP_MAX_TOKEN_ID, AP_MIN_TOKEN_ID
from v5_contracts.model_spec import ModelSpec
from v5_model.core import initialize, packed_layout

X3_ARCHITECTURES: tuple[str, ...] = (
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

X3_CONTROL_PAIRS: tuple[tuple[str, str], ...] = (
    ("P-R-Iter-Fixed", "P-R-Iter"),
    ("P-R-Iter-NoRel", "P-R-Iter"),
    ("P-R-Iter-AddMem", "P-R-Iter"),
    ("P-R-Iter-NoCopy", "P-R-Iter"),
    ("R-noedge", "R"),
)

# Verified 50,552,576 core params (ctx=640)
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
        return MODEL_PROFILES["S20"]
    if s in ("S50", "s50"):
        return S50_MODEL_SPEC
    if s in ("tiny", "canary", "TINY"):
        return TINY_CANARY_MODEL_SPEC
    raise ValueError(f"Unknown X3 backbone scale {scale!r}; expected 'tiny', 'S20', or 'S50'")


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
        # Choose compact memory width m_dim so head overhead stays <= 5% even on 90.8k tiny canary
        m_dim = min(96, max(8, d // 8))
        self.d_model = d
        self.m_dim = m_dim

        torch.manual_seed(int(seed) + 4103)
        self._ablation_mode: str = "none"
        self._episodic_state: torch.Tensor | None = None

        # Common lean readout & dual-interface heads
        self.lm_bias = nn.Parameter(torch.zeros(int(base_spec.vocabulary_size)))
        self.op_head = nn.Linear(d, 4)  # ANSWER, QUERY, RETRIEVE, ABSTAIN

        is_iter_family = self.architecture.startswith("P-R-Iter")
        uses_slots = is_iter_family or self.architecture in ("R", "R-noedge")
        uses_pointer = is_iter_family or self.architecture == "P"

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
            self.uncert_head = nn.Linear(m_dim, 2)
            self.out_proj = nn.Linear(m_dim, d)
            self.cls_head = nn.Linear(m_dim, 16)
        else:
            # Base or pure P without slot memory
            self.base_proj = nn.Linear(d, m_dim)
            self.out_proj = nn.Linear(m_dim, d)
            self.cls_head = nn.Linear(m_dim, 16)
            self.uncert_head = nn.Linear(m_dim, 2)

        if uses_pointer:
            self.ptr_q = nn.Linear(d, m_dim)
            self.ptr_k = nn.Linear(d, m_dim)
            self.copy_mix_head = nn.Linear(d, 2)

    def reset_episode(self) -> None:
        self._episodic_state = None
        self._ablation_mode = "none"

    @contextmanager
    def ablate_mechanism(self, mode: str = "disable_pointer") -> Any:
        """Surgical mechanism lesion context manager yielding mechanism label (Palantiri Gate 4)."""
        valid = {"none", "disable_pointer", "disable_relational", "disable_delta", "disable_iterative"}
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
        slot_fallback: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute differentiable pointer-copy probs over A..P symbols (or slot attention for NoCopy)."""
        bsz, seq_len, _ = h_all.shape
        dev = h_all.device
        dtype = h_all.dtype

        if self.architecture == "P-R-Iter-NoCopy" and slot_fallback is not None:
            # Attend strictly over internal working memory slots without looking back at context bytes
            q = self.ptr_q(h_act).unsqueeze(1)  # [B, 1, m]
            s_full = self.out_proj(slot_fallback)  # [B, K, d]
            k = self.ptr_k(s_full)  # [B, K, m]
            attn = F.softmax((q @ k.transpose(1, 2)) / math.sqrt(float(self.m_dim)), dim=-1)  # [B, 1, K]
            slot_ctx = (attn @ slot_fallback).squeeze(1)  # [B, m]
            return F.softmax(self.cls_head(slot_ctx), dim=-1)

        if self._ablation_mode == "disable_pointer":
            return torch.full((bsz, 16), 1.0 / 16.0, device=dev, dtype=dtype)

        q = self.ptr_q(h_act).unsqueeze(1)  # [B, 1, m]
        k = self.ptr_k(h_all)  # [B, L, m]
        scores = (q @ k.transpose(1, 2)).squeeze(1) / math.sqrt(float(self.m_dim))  # [B, L]
        is_ap = (token_ids >= AP_MIN_TOKEN_ID) & (token_ids <= AP_MAX_TOKEN_ID) & valid_tok
        has_any_ap = is_ap.any(dim=1, keepdim=True)
        attn_mask = torch.where(has_any_ap, is_ap, valid_tok)
        safe_scores = scores.masked_fill(~attn_mask, -1e9)
        attn_probs = F.softmax(safe_scores, dim=-1)  # [B, L]

        sym_probs = torch.zeros((bsz, 16), device=dev, dtype=dtype)
        fallback_bucket = token_ids % 16
        tok_bucket = torch.where(has_any_ap, (token_ids - AP_MIN_TOKEN_ID).clamp(0, 15), fallback_bucket)
        active_weights = torch.where(has_any_ap, attn_probs * is_ap.to(dtype), attn_probs * valid_tok.to(dtype))
        sym_probs = sym_probs.scatter_add(1, tok_bucket, active_weights)
        return sym_probs.clamp(min=1e-7)

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
        h_act = h[b_idx, apos]  # [B, d]

        uses_slots = hasattr(self, "slot_init")
        uses_pointer = hasattr(self, "ptr_q")
        n_steps = 1 if (not uses_slots or self.architecture in ("R", "R-noedge") or self._ablation_mode == "disable_iterative") else self.max_steps
        if self.architecture in ("R", "R-noedge"):
            n_steps = 2

        ponder_cost = torch.zeros((bsz,), device=dev, dtype=h.dtype)
        next_state = None
        if uses_slots:
            tok_mem = self.input_proj(h)  # [B, L, m]
            q_vec = self.query_proj(h_act)  # [B, m]
            if use_persistent_state and persistent_state is not None:
                w = persistent_state.to(device=dev, dtype=h.dtype)
            elif use_persistent_state and self._episodic_state is not None and self._episodic_state.shape[0] == bsz:
                w = self._episodic_state.to(device=dev, dtype=h.dtype)
            else:
                w = self.slot_init.expand(bsz, -1, -1).contiguous()

            mask_bias = (~valid_tok).to(dtype=h.dtype) * (-1e9)  # [B, L]
            k_mem = self.read_k(tok_mem)  # [B, L, m]
            v_mem = self.read_v(tok_mem)  # [B, L, m]

            rem = torch.ones((bsz, 1), device=dev, dtype=h.dtype)
            w_accum = torch.zeros_like(w)
            halt_prob_trace: list[torch.Tensor] = []

            no_rel = self.architecture in ("P-R-Iter-NoRel", "R-noedge") or self._ablation_mode == "disable_relational"
            add_mem = self.architecture == "P-R-Iter-AddMem" or self._ablation_mode == "disable_delta"
            fixed_halt = self.architecture in ("P-R-Iter-Fixed", "R", "R-noedge")

            for step_t in range(n_steps):
                # 1. Query-conditioned cross-attention read from input bytes into K working-memory slots
                q_slot = self.read_q(w + q_vec.unsqueeze(1))  # [B, K, m]
                attn_s = (q_slot @ k_mem.transpose(1, 2)) / math.sqrt(float(self.m_dim))
                attn_w = F.softmax(attn_s + mask_bias.unsqueeze(1), dim=-1)  # [B, K, L]
                read_vec = attn_w @ v_mem  # [B, K, m]

                # 2. Cross-slot relational message passing (vs self-only MLP for NoRel)
                if no_rel:
                    peer_mean = w
                else:
                    slot_sum = w.sum(dim=1, keepdim=True)
                    peer_mean = (slot_sum - w) / float(max(1, self.num_slots - 1))
                rel = torch.tanh(self.rel_msg(torch.cat([w, peer_mean], dim=-1)))
                cand = torch.tanh(self.delta_cand(torch.cat([read_vec, rel], dim=-1)))
                gate = torch.sigmoid(self.delta_gate(torch.cat([w, cand], dim=-1)))

                # 3. Selective Delta Memory Revision vs Additive Accumulation
                if add_mem:
                    w = F.layer_norm(w + gate * cand, (self.m_dim,))
                else:
                    w = (1.0 - gate) * w + gate * cand

                # 4. Learned Halting Probability (ACT)
                h_p = torch.sigmoid(self.halt_head(w.mean(dim=1)))  # [B, 1]
                halt_prob_trace.append(h_p.squeeze(-1))
                if fixed_halt:
                    # Keep halt_head differentiably coupled in ponder_cost while executing all T steps cleanly
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

            w_pooled = w_accum.mean(dim=1)  # [B, m]
            next_state = w_accum.detach()
            if use_persistent_state:
                self._episodic_state = next_state
            uncertainties = torch.sigmoid(self.uncert_head(w_pooled))  # [B, 2]: [epistemic, info]
            ctx_feat = self.out_proj(w_pooled)  # [B, d]
            slot_cls_logits = self.cls_head(w_pooled)  # [B, 16]
            slot_mem_for_nocopy = w_accum
        else:
            proj = torch.tanh(self.base_proj(h_act))
            uncertainties = torch.sigmoid(self.uncert_head(proj))
            ctx_feat = self.out_proj(proj)
            slot_cls_logits = self.cls_head(proj)
            slot_mem_for_nocopy = None

        decision_hidden = h_act + ctx_feat
        op_logits = self.op_head(decision_hidden)  # [B, 4]: ANSWER, QUERY, RETRIEVE, ABSTAIN

        if uses_pointer:
            ptr_sym_probs = self._compute_pointer_symbol_probs(
                h, h_act, token_ids, valid_tok, slot_fallback=slot_mem_for_nocopy
            )
            mix = F.softmax(self.copy_mix_head(decision_hidden), dim=-1)  # [B, 2]
            copy_log = torch.log(ptr_sym_probs.clamp_min(1e-7))
            gen_log = F.log_softmax(slot_cls_logits, dim=-1)
            # For P-R-Iter-NoCopy, ptr_sym_probs attends strictly over internal memory slots (zero input-byte copy)
            sym_log = mix[:, :1] * copy_log + mix[:, 1:2] * gen_log
        else:
            sym_log = F.log_softmax(slot_cls_logits, dim=-1)

        op_log = F.log_softmax(op_logits, dim=-1)  # [B, 4]
        ans_action_log = op_log[:, 0:1] + sym_log  # [B, 16]
        action_19_logits = torch.cat([ans_action_log, op_log[:, 1:]], dim=-1)  # [B, 19]

        # Full byte-level autoregressive generative LM logits conditioned on reasoning context + backbone hidden
        h_gen = h + 0.25 * ctx_feat.unsqueeze(1)
        gen_logits = F.linear(h_gen, self.core.embedding.weight, self.lm_bias)

        out: dict[str, Any] = {
            "logits": gen_logits,
            "action_logits": action_19_logits,
            "op_logits": op_logits,
            "cls_logits": slot_cls_logits,
            "decision_hidden": decision_hidden,
            "uncertainties": uncertainties,
            "ponder_cost": ponder_cost,
            "next_persistent_state": next_state,
        }
        if not return_aux:
            return {"logits": gen_logits, "action_logits": action_19_logits, "decision_hidden": decision_hidden}
        return out

    @torch.no_grad()
    def generate_bytes(
        self,
        prompt_text: str,
        *,
        max_new_bytes: int = 24,
        use_persistent_state: bool = False,
        stop_chars: Sequence[str] = ("\n",),
    ) -> dict[str, Any]:
        """Tool-free autoregressive byte synthesis from the neural model alone (§4, §6)."""
        was_training = self.training
        self.eval()
        dev = next(self.parameters()).device
        encoded = [UTF8.bos_id, *UTF8.encode(prompt_text)]
        max_in = max(8, self.max_ctx - int(max_new_bytes))
        if len(encoded) > max_in:
            encoded = encoded[-max_in:]
        generated_ids: list[int] = []
        last_uncert = [0.5, 0.5]
        last_ponder = 0.0
        for _ in range(int(max_new_bytes)):
            seq = encoded + generated_ids
            toks = torch.tensor([seq], dtype=torch.long, device=dev)
            segs = torch.zeros_like(toks)
            pos, mask = packed_layout(segs, torch_module=torch)
            apos = torch.tensor([len(seq) - 1], dtype=torch.long, device=dev)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=(dev.type == "cuda")):
                out = self.forward(toks, pos, mask, action_positions=apos, use_persistent_state=use_persistent_state)
            last_uncert = out["uncertainties"][0].float().detach().cpu().tolist()
            last_ponder = float(out["ponder_cost"][0].item())
            next_logits = out["logits"][0, -1].float().clone()
            # Mask control tokens PAD/BOS while allowing EOS & printable bytes
            next_logits[UTF8.pad_id] = -1e9
            next_logits[UTF8.bos_id] = -1e9
            nxt = int(torch.argmax(next_logits).item())
            if nxt in (UTF8.eos_id, UTF8.special(SpecialToken.ANSWER)):
                break
            generated_ids.append(nxt)
            if nxt >= 4:
                decoded_ch = UTF8.decode([nxt])
                if decoded_ch in stop_chars:
                    generated_ids.pop()
                    break
        if was_training:
            self.train()
        clean_ids = [t for t in generated_ids if t >= 4]
        return {
            "text": UTF8.decode(clean_ids).strip(),
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
    """Encode prompt + target UTF-8 bytes for supervised CE + generative byte training without truncation."""
    p_ids = [UTF8.bos_id, *UTF8.encode(prompt), UTF8.special(SpecialToken.ANSWER)]
    t_ids = [*UTF8.encode(target_text), UTF8.eos_id]
    full = p_ids + t_ids
    if len(full) > max_length:
        raise ValueError(f"Sequence {len(full)} exceeds context_length {max_length}")
    eligible = [False] * len(p_ids) + [True] * len(t_ids)
    # Action readout happens at the ANSWER marker token index
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
    lr: float = 5e-4,
    max_param_delta_norm: float = 1.0,
) -> dict[str, Any]:
    """Evaluate all 5 distinct learning modes separately on the same neural system (§3-§4, §9)."""
    dev = next(model.parameters()).device

    def _predict_text_and_match(case: Mapping[str, Any], *, use_persistent: bool = False) -> tuple[str, bool]:
        out = model.generate_bytes(str(case["prompt"]), max_new_bytes=12, use_persistent_state=use_persistent)
        pred = str(out["text"]).strip()
        return pred, pred == str(case["gold_text"]).strip()

    # Mode 1: Frozen-weight In-Context Learning across 0/1/4/16/64 shots
    model.eval()
    model.reset_episode()
    icl_curve: dict[int, float] = {}
    for shot, items in sorted(icl_cases_by_shot.items()):
        ok = 0
        for c in items:
            model.reset_episode()
            _, matched = _predict_text_and_match(c, use_persistent=False)
            ok += int(matched)
        icl_curve[int(shot)] = float(ok / max(1, len(items)))

    # Mode 2: Persistent internal state / working-memory adaptation across sequential stream (weights frozen)
    model.reset_episode()
    state_ok = 0
    for c in state_adapt_sequence:
        _, matched = _predict_text_and_match(c, use_persistent=True)
        state_ok += int(matched)
    state_acc = float(state_ok / max(1, len(state_adapt_sequence)))

    # Mode 3: Bounded fast parameter-update learning (torch.enable_grad + floating-param named_parameters norm cap!)
    snap = {k: v.detach().clone() for k, v in model.named_parameters()}
    if grad_adapt_demos:
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=float(lr))
        model.train()
        with torch.enable_grad():
            samples = [encode_prompt_and_target_pair(str(d["prompt"]), str(d["gold_text"]), max_length=model.max_ctx) for d in grad_adapt_demos]
            max_l = max(len(s) for s, _, _ in samples)
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

    # Check floating-parameter L2 delta strictly via named_parameters() (never state_dict bool buffers!)
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
        _, matched = _predict_text_and_match(c, use_persistent=False)
        grad_ok += int(matched)
    grad_acc = float(grad_ok / max(1, len(grad_adapt_queries)))

    # Restore exact pre-adaptation weights so calling evaluator never mutates caller model
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(snap[name])
    model.reset_episode()

    # Mode 4 & Mode 5: Tool-free vs tool-assisted oracle gap + Cross-domain transfer
    tr_neural = 0
    tr_oracle = 0
    for c in transfer_cases:
        _, matched = _predict_text_and_match(c, use_persistent=False)
        tr_neural += int(matched)
        tr_oracle += int(str(oracle_solver(c)).strip() == str(c["gold_text"]).strip())
    n_tr = max(1, len(transfer_cases))
    return {
        "schema": "elendril-x3-five-learning-modes-v1",
        "mode1_frozen_icl_curve": icl_curve,
        "mode2_persistent_state_accuracy": state_acc,
        "mode3_bounded_grad_adapt_accuracy": grad_acc,
        "mode3_param_delta_l2": float(delta_norm),
        "mode4_tool_free_transfer_accuracy": float(tr_neural / n_tr),
        "mode4_tool_assisted_oracle_accuracy": float(tr_oracle / n_tr),
        "mode4_tool_assistance_gap": float((tr_oracle - tr_neural) / n_tr),
        "mode5_cross_domain_transfer_accuracy": float(tr_neural / n_tr),
    }


__all__ = [
    "S50_MODEL_SPEC",
    "TINY_CANARY_MODEL_SPEC",
    "X3CognitiveModel",
    "X3_ARCHITECTURES",
    "X3_CONTROL_PAIRS",
    "build_x3_backbone_spec",
    "encode_prompt_and_target_pair",
    "evaluate_five_learning_modes",
]
