"""Avendi V8-Y2 P4 intermediate-state transition models, parameter-matched controls, and hop-1 interchange protocol."""

from __future__ import annotations

from dataclasses import replace
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from anra_v5.elendril.representation import UTF8
from .oracle import parse_tokenized_public_observation
from .protocol import P4_ARCHITECTURE_SPEC, QUALIFICATION_GATES, SYMBOLS, derive_seed_int
from .render import RenderedRow, render_avendi_prompt
from .worlds import AvendiCase, AvendiFamily


SYM_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(SYMBOLS)}
IDX_TO_SYM: dict[int, str] = {i: s for i, s in enumerate(SYMBOLS)}


def parse_public_rows_and_query_for_p4(prompt: str) -> tuple[int, int, list[tuple[int, ...]]]:
    """Extract only public query symbol index (0..15), hop count, and raw UTF-8 byte rows from prompt.

    Attaches NO row labels for key, terminal, path membership, or evidence relevance, and NO row position index.
    """
    lines = prompt.splitlines()
    if len(lines) < 3:
        raise ValueError("malformed prompt for P4 row parser")
    header = lines[1]
    # Parse "Follow <hops> links from <SYM>."
    parts = header.split()
    hops = int(parts[1])
    q_sym = parts[4].rstrip(".")
    if q_sym not in SYM_TO_IDX:
        raise ValueError(f"P4 requires canonical A-P query symbol, got {q_sym!r}")
    q_idx = SYM_TO_IDX[q_sym]

    facts_line = lines[2].strip()
    row_chunks = [c.strip() for c in facts_line.split("; ") if c.strip()]
    row_byte_seqs = [tuple(int(t) for t in UTF8.encode(chunk)) for chunk in row_chunks]
    return q_idx, hops, row_byte_seqs


def _encode_rows_gru(
    byte_emb: nn.Embedding,
    row_gru: nn.GRU,
    row_byte_seqs: Sequence[Sequence[int]],
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Encode each public row independently with shared 1-layer GRU(64,64) without row-order embeddings."""
    lengths = [len(r) for r in row_byte_seqs]
    max_len = max(lengths)
    padded = torch.zeros((len(row_byte_seqs), max_len), dtype=torch.int64, device=device)
    for i, r in enumerate(row_byte_seqs):
        padded[i, : len(r)] = torch.tensor(r, dtype=torch.int64, device=device)
    emb = byte_emb(padded)  # (R, L, 64)
    out, _ = row_gru(emb)  # (R, L, 64)
    final_vecs = torch.stack([out[i, lengths[i] - 1] for i in range(len(row_byte_seqs))], dim=0)
    padded_positions = int(len(row_byte_seqs) * max_len)
    return final_vecs, padded_positions


def _state_entropy(probs: torch.Tensor) -> float:
    p = probs.detach().clamp_min(1e-12)
    return float(-(p * torch.log(p)).sum().item())


class AvendiSymbolicTransitionModel(nn.Module):
    """P4 controlled 16-way bottleneck transition model with shared row GRU(64,64) and repeated attention."""

    def __init__(self, *, seed: int = 7) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.vocab_size = UTF8.vocabulary_size
        self.width = int(P4_ARCHITECTURE_SPEC["byte_embedding_width"])  # 64
        self.num_states = int(P4_ARCHITECTURE_SPEC["state_cardinality"])  # 16
        self.attn_scale = float(P4_ARCHITECTURE_SPEC["attention_scale"])  # 8.0

        self.byte_emb = nn.Embedding(self.vocab_size, self.width)
        self.row_gru = nn.GRU(self.width, self.width, num_layers=1, batch_first=True)
        self.key_proj = nn.Linear(self.width, self.width)
        self.val_proj = nn.Linear(self.width, self.width)
        self.state_emb = nn.Parameter(torch.randn(self.num_states, self.width) * 0.1)
        self.readout = nn.Linear(self.width, self.num_states)

    def encode_rows(self, row_byte_seqs: Sequence[Sequence[int]]) -> tuple[torch.Tensor, torch.Tensor, int]:
        dev = self.state_emb.device
        row_vecs, pad_pos = _encode_rows_gru(self.byte_emb, self.row_gru, row_byte_seqs, dev)
        return self.key_proj(row_vecs), self.val_proj(row_vecs), pad_pos

    def transition_step(
        self,
        state_dist: torch.Tensor,
        row_keys: torch.Tensor,
        row_vals: torch.Tensor,
        *,
        hard_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single transition step mapping current 16-way state distribution -> (next_logits, next_state_dist)."""
        q_vec = state_dist @ self.state_emb  # (64,)
        scores = (row_keys @ q_vec) / self.attn_scale  # (R,)
        attn = F.softmax(scores, dim=-1)
        ctx = attn @ row_vals  # (64,)
        logits = self.readout(ctx)  # (16,)
        if hard_state:
            idx = int(torch.argmax(logits).item())
            next_dist = torch.zeros_like(logits)
            next_dist[idx] = 1.0
        else:
            next_dist = F.softmax(logits, dim=-1)
        return logits, next_dist

    def forward_prompt(
        self,
        prompt: str,
        *,
        hard_state: bool = False,
        override_after_step1: torch.Tensor | None = None,
        suffix_prompt_after_step1: str | None = None,
        row_perturbation_delta: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Run multi-step transition from public query one-hot with optional hop-1 state swap or suffix edit."""
        dev = self.state_emb.device
        q_idx, hops, row_seqs = parse_public_rows_and_query_for_p4(prompt)
        row_keys, row_vals, pad_pos = self.encode_rows(row_seqs)
        if row_perturbation_delta is not None:
            row_vals = row_vals + row_perturbation_delta.to(dev)

        state = torch.zeros(self.num_states, dtype=torch.float32, device=dev)
        state[q_idx] = 1.0

        step_logits: list[torch.Tensor] = []
        step_states: list[torch.Tensor] = []
        entropies: list[float] = []

        for step_idx in range(hops):
            if step_idx >= 1 and suffix_prompt_after_step1 is not None and step_idx == 1:
                _, _, suf_rows = parse_public_rows_and_query_for_p4(suffix_prompt_after_step1)
                row_keys, row_vals, suf_pad = self.encode_rows(suf_rows)
                pad_pos += suf_pad

            logits, next_state = self.transition_step(state, row_keys, row_vals, hard_state=hard_state)
            if step_idx == 0 and override_after_step1 is not None:
                next_state = override_after_step1.to(dev)
            step_logits.append(logits)
            step_states.append(next_state)
            entropies.append(_state_entropy(next_state))
            state = next_state

        final_logits = step_logits[-1]
        pred_idx = int(torch.argmax(final_logits).item())
        return {
            "final_logits": final_logits,
            "predicted_symbol": IDX_TO_SYM[pred_idx],
            "step_logits": step_logits,
            "step_states": step_states,
            "hop1_state": step_states[0],
            "entropies": entropies,
            "padded_positions": pad_pos,
        }


class AvendiContinuousRecurrentControl(nn.Module):
    """Control (a): same row GRU(64,64) encoder + repeated attention with continuous recurrent state."""

    def __init__(self, *, recurrent_width: int = 64, seed: int = 7) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.vocab_size = UTF8.vocabulary_size
        self.width = 64
        self.recurrent_width = int(recurrent_width)
        self.num_states = 16
        self.attn_scale = 8.0

        self.byte_emb = nn.Embedding(self.vocab_size, self.width)
        self.row_gru = nn.GRU(self.width, self.width, num_layers=1, batch_first=True)
        self.key_proj = nn.Linear(self.width, self.recurrent_width)
        self.val_proj = nn.Linear(self.width, self.recurrent_width)
        self.init_state_emb = nn.Embedding(self.num_states, self.recurrent_width)
        self.state_update = nn.Linear(self.recurrent_width, self.recurrent_width)
        self.readout = nn.Linear(self.recurrent_width, self.num_states)

    def forward_prompt(self, prompt: str) -> dict[str, Any]:
        dev = self.readout.weight.device
        q_idx, hops, row_seqs = parse_public_rows_and_query_for_p4(prompt)
        row_vecs, pad_pos = _encode_rows_gru(self.byte_emb, self.row_gru, row_seqs, dev)
        row_keys = self.key_proj(row_vecs)
        row_vals = self.val_proj(row_vecs)

        h = self.init_state_emb(torch.tensor(q_idx, dtype=torch.int64, device=dev))
        step_states: list[torch.Tensor] = []
        for _ in range(hops):
            attn = F.softmax((row_keys @ h) / self.attn_scale, dim=-1)
            ctx = attn @ row_vals
            h = torch.tanh(self.state_update(ctx))
            step_states.append(h)

        final_logits = self.readout(h)
        pred_idx = int(torch.argmax(final_logits).item())
        return {
            "final_logits": final_logits,
            "predicted_symbol": IDX_TO_SYM[pred_idx],
            "hop1_state": step_states[0],
            "padded_positions": pad_pos,
        }


class AvendiOneStepAttentionControl(nn.Module):
    """Control (b): one-step attention readout with the same shared row GRU(64,64) encoder."""

    def __init__(self, *, seed: int = 7) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.vocab_size = UTF8.vocabulary_size
        self.width = 64
        self.num_states = 16
        self.attn_scale = 8.0

        self.byte_emb = nn.Embedding(self.vocab_size, self.width)
        self.row_gru = nn.GRU(self.width, self.width, num_layers=1, batch_first=True)
        self.key_proj = nn.Linear(self.width, self.width)
        self.val_proj = nn.Linear(self.width, self.width)
        self.state_emb = nn.Parameter(torch.randn(self.num_states, self.width) * 0.1)
        self.readout = nn.Linear(self.width, self.num_states)

    def forward_prompt(self, prompt: str) -> dict[str, Any]:
        dev = self.state_emb.device
        q_idx, _, row_seqs = parse_public_rows_and_query_for_p4(prompt)
        row_vecs, pad_pos = _encode_rows_gru(self.byte_emb, self.row_gru, row_seqs, dev)
        row_keys = self.key_proj(row_vecs)
        row_vals = self.val_proj(row_vecs)
        q_vec = self.state_emb[q_idx]
        attn = F.softmax((row_keys @ q_vec) / self.attn_scale, dim=-1)
        ctx = attn @ row_vals
        final_logits = self.readout(ctx)
        pred_idx = int(torch.argmax(final_logits).item())
        return {
            "final_logits": final_logits,
            "predicted_symbol": IDX_TO_SYM[pred_idx],
            "padded_positions": pad_pos,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def select_matched_continuous_width() -> dict[str, Any]:
    """Preregistered CPU width search over (48, 56, 64, 72, 80) for continuous control, tie smaller."""
    sym_model = AvendiSymbolicTransitionModel(seed=7)
    target_params = count_parameters(sym_model)
    grid = list(P4_ARCHITECTURE_SPEC["continuous_width_candidates"])

    candidates: list[dict[str, Any]] = []
    for w in sorted(grid):
        c_mod = AvendiContinuousRecurrentControl(recurrent_width=w, seed=7)
        p = count_parameters(c_mod)
        diff = abs(p - target_params)
        rel = diff / max(1, target_params)
        candidates.append({
            "recurrent_width": w,
            "parameters": p,
            "abs_diff": diff,
            "relative_diff": rel,
        })

    # Choose closest, tie smaller width
    best = min(candidates, key=lambda r: (r["abs_diff"], r["recurrent_width"]))
    one_step_params = count_parameters(AvendiOneStepAttentionControl(seed=7))
    return {
        "schema": "anra-avendi-v8-y2-p4-width-search-v1",
        "symbolic_bottleneck_parameters": target_params,
        "selected_continuous_width": best["recurrent_width"],
        "selected_continuous_parameters": best["parameters"],
        "residual_parameter_diff": best["parameters"] - target_params,
        "relative_parameter_diff": best["relative_diff"],
        "within_10pct_tolerance": best["relative_diff"] <= float(QUALIFICATION_GATES["p4_transition"]["param_match_tolerance"]),
        "one_step_control_parameters": one_step_params,
        "candidates": candidates,
    }


def _build_edited_terminal_prompt(case: AvendiCase) -> tuple[str, dict[str, str]]:
    """Replace only downstream hop-3 terminal targets (e->g', f->h') with fresh symbols while keeping prefix edges fixed."""
    parsed = parse_tokenized_public_observation(case.prompt_tokens)
    used = {r.subject for r in parsed.rows} | {r.value for r in parsed.rows if r.value is not None}
    unused = [s for s in SYMBOLS if s not in used]
    new_g, new_h = unused[0], unused[1]
    remap = {case.endpoint_g: new_g, case.endpoint_h: new_h}

    new_rows = [
        RenderedRow(
            subject=r.subject,
            value=remap.get(r.value, r.value) if r.value is not None else None,
            tick=r.tick,
            source=r.source,
        )
        for r in parsed.rows
    ]
    edited_prompt = render_avendi_prompt(
        query_subject=case.query_subject,
        hops=case.hops,
        rows=new_rows,
        query_available=case.query_available,
        memory_available=case.memory_available,
        template=case.template,
    )
    return edited_prompt, remap


def _build_third_chain_donor_prompt(case: AvendiCase) -> tuple[str, str]:
    """Build an extended 3-chain prompt adding a disjoint chain u->v->w->z so donor state from u routes to z."""
    parsed = parse_tokenized_public_observation(case.prompt_tokens)
    used = {r.subject for r in parsed.rows} | {r.value for r in parsed.rows if r.value is not None}
    unused = [s for s in SYMBOLS if s not in used]
    u, v, w, z = unused[:4]
    extra_rows = [
        *(RenderedRow(subject=r.subject, value=r.value, tick=r.tick, source=r.source) for r in parsed.rows),
        RenderedRow(subject=u, value=v, tick=9101, source="s0"),
        RenderedRow(subject=v, value=w, tick=9102, source="s1"),
        RenderedRow(subject=w, value=z, tick=9103, source="s2"),
    ]
    prompt_u = render_avendi_prompt(
        query_subject=u,
        hops=3,
        rows=extra_rows,
        query_available=False,
        memory_available=False,
        template=0,
    )
    return prompt_u, z


def fit_state_alignment_bijection(
    model: AvendiSymbolicTransitionModel,
    calibration_families: Sequence[AvendiFamily],
) -> dict[str, int]:
    """Fit a 1-to-1 bijection from public intermediate symbol (A-P) to internal hop-1 state index on calibration data."""
    counts = np.zeros((16, 16), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for fam in calibration_families:
            for cell in fam.cells:
                if len(cell.proof_path) < 2:
                    continue
                true_hop1_sym = cell.proof_path[1]
                out = model.forward_prompt(cell.public_prompt)
                hop1_probs = out["hop1_state"].detach().cpu().numpy()
                counts[SYM_TO_IDX[true_hop1_sym]] += hop1_probs

    # Greedy maximum-weight bipartite matching (deterministic tie-break)
    assigned_internal: set[int] = set()
    mapping: dict[str, int] = {}
    for sym_idx in range(16):
        best_j = max(
            (j for j in range(16) if j not in assigned_internal),
            key=lambda j: (float(counts[sym_idx, j]), -j),
        )
        assigned_internal.add(best_j)
        mapping[IDX_TO_SYM[sym_idx]] = int(best_j)
    return mapping


def run_p4_interchange_protocol(
    model: AvendiSymbolicTransitionModel,
    three_hop_families: Sequence[AvendiFamily],
    *,
    calibration_families: Sequence[AvendiFamily] | None = None,
    seed: int = 7,
) -> dict[str, Any]:
    """Execute complete P4 hop-1 state interchange protocol + all 7 controls + terminal-edited suffix test."""
    model.eval()
    dev = model.state_emb.device
    cal_fams = calibration_families if calibration_families is not None else three_hop_families[: min(16, len(three_hop_families))]
    aligned_map = fit_state_alignment_bijection(model, cal_fams)

    rng = random.Random(derive_seed_int("avendi-p4-interchange-perm", seed))
    perm = list(range(16))
    while any(perm[i] == i for i in range(16)):
        rng.shuffle(perm)
    perm_t = torch.tensor(perm, dtype=torch.int64, device=dev)

    all_four_unswapped = 0
    all_four_hard_unswapped = 0
    eligible_both_correct_pairs = 0
    total_pairs = 0  # 2 rewire conditions per family -> 2 * len(families) pairs, 4 directed swaps per family
    total_directed = 0
    eligible_directed = 0

    identity_swap_matches = 0
    donor_unconditional_hits = 0
    donor_conditional_hits = 0
    donor_hard_unconditional_hits = 0
    donor_edited_suffix_unconditional_hits = 0
    donor_edited_suffix_conditional_hits = 0
    permuted_state_hits = 0
    row_perturbation_hits = 0
    third_chain_donor_hits = 0
    oracle_public_onehot_hits = 0
    oracle_aligned_onehot_hits = 0
    log_odds_shifts_toward_donor: list[float] = []
    entropies: list[float] = []

    with torch.no_grad():
        for fam in three_hop_families:
            cmap = fam.cell_map()
            fam_soft_ok = 0
            fam_hard_ok = 0
            for c_key in ("00", "01", "10", "11"):
                c = cmap[c_key]
                r_soft = model.forward_prompt(c.public_prompt, hard_state=False)
                r_hard = model.forward_prompt(c.public_prompt, hard_state=True)
                fam_soft_ok += int(r_soft["predicted_symbol"] == c.expected_answer)
                fam_hard_ok += int(r_hard["predicted_symbol"] == c.expected_answer)
                entropies.extend(r_soft["entropies"])
            if fam_soft_ok == 4:
                all_four_unswapped += 1
            if fam_hard_ok == 4:
                all_four_hard_unswapped += 1

            # Evaluate swaps on both original graph (00 <-> 10) and rewired graph (01 <-> 11)
            for k_a, k_b in (("00", "10"), ("01", "11")):
                ca, cb = cmap[k_a], cmap[k_b]
                total_pairs += 1
                res_a = model.forward_prompt(ca.public_prompt, hard_state=False)
                res_b = model.forward_prompt(cb.public_prompt, hard_state=False)
                base_a_ok = res_a["predicted_symbol"] == ca.expected_answer
                base_b_ok = res_b["predicted_symbol"] == cb.expected_answer
                both_ok = bool(base_a_ok and base_b_ok)
                if both_ok:
                    eligible_both_correct_pairs += 1

                for rec_case, don_case, res_rec, res_don in (
                    (ca, cb, res_a, res_b),
                    (cb, ca, res_b, res_a),
                ):
                    total_directed += 1
                    if both_ok:
                        eligible_directed += 1

                    # 1. Identity swap control
                    id_res = model.forward_prompt(
                        rec_case.public_prompt,
                        override_after_step1=res_rec["hop1_state"].clone(),
                    )
                    identity_swap_matches += int(id_res["predicted_symbol"] == res_rec["predicted_symbol"])

                    # 2. Donor swap (soft)
                    swap_res = model.forward_prompt(
                        rec_case.public_prompt,
                        override_after_step1=res_don["hop1_state"].clone(),
                    )
                    donor_target = don_case.expected_answer
                    own_target = rec_case.expected_answer
                    hit_swap = int(swap_res["predicted_symbol"] == donor_target)
                    donor_unconditional_hits += hit_swap
                    if both_ok:
                        donor_conditional_hits += hit_swap

                    # Log-odds shift toward donor endpoint
                    d_idx = SYM_TO_IDX[donor_target]
                    o_idx = SYM_TO_IDX[own_target]
                    base_diff = float((res_rec["final_logits"][d_idx] - res_rec["final_logits"][o_idx]).item())
                    swap_diff = float((swap_res["final_logits"][d_idx] - swap_res["final_logits"][o_idx]).item())
                    log_odds_shifts_toward_donor.append(swap_diff - base_diff)

                    # 3. Hard argmax donor swap
                    res_don_hard = model.forward_prompt(don_case.public_prompt, hard_state=True)
                    swap_hard = model.forward_prompt(
                        rec_case.public_prompt,
                        hard_state=True,
                        override_after_step1=res_don_hard["hop1_state"].clone(),
                    )
                    donor_hard_unconditional_hits += int(swap_hard["predicted_symbol"] == donor_target)

                    # 4. Downstream terminal-edge edit after hop-1 cut
                    edited_rec_prompt, remap = _build_edited_terminal_prompt(rec_case)
                    expected_new_donor_terminal = remap[donor_target]
                    edit_swap_res = model.forward_prompt(
                        rec_case.public_prompt,
                        override_after_step1=res_don["hop1_state"].clone(),
                        suffix_prompt_after_step1=edited_rec_prompt,
                    )
                    hit_edit = int(edit_swap_res["predicted_symbol"] == expected_new_donor_terminal)
                    donor_edited_suffix_unconditional_hits += hit_edit
                    if both_ok:
                        donor_edited_suffix_conditional_hits += hit_edit

                    # 5. Random permuted state control
                    perm_state = res_don["hop1_state"][perm_t]
                    perm_res = model.forward_prompt(rec_case.public_prompt, override_after_step1=perm_state)
                    permuted_state_hits += int(perm_res["predicted_symbol"] == donor_target)

                    # 6. Norm-matched row encoder perturbation control (no state swap)
                    state_diff_norm = float(torch.linalg.vector_norm(res_don["hop1_state"] - res_rec["hop1_state"]).item())
                    noise = torch.randn((6, model.width), device=dev)
                    noise_norm = float(torch.linalg.vector_norm(noise).item())
                    scaled_noise = noise * (state_diff_norm / max(1e-8, noise_norm))
                    pert_res = model.forward_prompt(rec_case.public_prompt, row_perturbation_delta=scaled_noise)
                    row_perturbation_hits += int(pert_res["predicted_symbol"] == donor_target)

                    # 7. Oracle one-hot injection (fixed public symbol vs calibration-aligned bijection)
                    true_donor_hop1_sym = don_case.proof_path[1]
                    pub_oh = torch.zeros(16, dtype=torch.float32, device=dev)
                    pub_oh[SYM_TO_IDX[true_donor_hop1_sym]] = 1.0
                    pub_inj = model.forward_prompt(rec_case.public_prompt, override_after_step1=pub_oh)
                    oracle_public_onehot_hits += int(pub_inj["predicted_symbol"] == donor_target)

                    al_oh = torch.zeros(16, dtype=torch.float32, device=dev)
                    al_oh[aligned_map[true_donor_hop1_sym]] = 1.0
                    al_inj = model.forward_prompt(rec_case.public_prompt, override_after_step1=al_oh)
                    oracle_aligned_onehot_hits += int(al_inj["predicted_symbol"] == donor_target)

                    # 8. Third-chain donor control
                    u_prompt, z_target = _build_third_chain_donor_prompt(rec_case)
                    res_u = model.forward_prompt(u_prompt)
                    res_third = model.forward_prompt(u_prompt, override_after_step1=res_u["hop1_state"].clone())
                    third_chain_donor_hits += int(res_third["predicted_symbol"] == z_target)

    n_fams = max(1, len(three_hop_families))
    n_dir = max(1, total_directed)
    n_elig = max(1, eligible_directed)

    soft_rate = all_four_unswapped / n_fams
    hard_rate = all_four_hard_unswapped / n_fams
    id_parity = identity_swap_matches / n_dir
    uncond_orig = donor_unconditional_hits / n_dir
    cond_orig = donor_conditional_hits / n_elig if eligible_directed > 0 else 0.0
    uncond_edit = donor_edited_suffix_unconditional_hits / n_dir
    cond_edit = donor_edited_suffix_conditional_hits / n_elig if eligible_directed > 0 else 0.0
    perm_rate = permuted_state_hits / n_dir

    return {
        "schema": "anra-avendi-v8-y2-p4-interchange-v1",
        "families_evaluated": len(three_hop_families),
        "directed_swaps_evaluated": total_directed,
        "eligible_both_correct_pairs": eligible_both_correct_pairs,
        "unswapped_three_hop_all_four_soft_rate": soft_rate,
        "unswapped_three_hop_all_four_hard_rate": hard_rate,
        "soft_minus_hard_all_four_change": soft_rate - hard_rate,
        "identity_swap_parity_rate": id_parity,
        "identity_swap_drop": 1.0 - id_parity,
        "donor_unconditional_accuracy_original_suffix": uncond_orig,
        "donor_conditional_accuracy_original_suffix": cond_orig,
        "donor_unconditional_accuracy_edited_terminal_suffix": uncond_edit,
        "donor_conditional_accuracy_edited_terminal_suffix": cond_edit,
        "donor_hard_argmax_unconditional_accuracy": donor_hard_unconditional_hits / n_dir,
        "random_permuted_state_accuracy": perm_rate,
        "donor_excess_over_permuted": uncond_orig - perm_rate,
        "row_encoder_perturbation_accuracy": row_perturbation_hits / n_dir,
        "third_chain_donor_accuracy": third_chain_donor_hits / n_dir,
        "oracle_public_onehot_accuracy": oracle_public_onehot_hits / n_dir,
        "oracle_aligned_onehot_accuracy": oracle_aligned_onehot_hits / n_dir,
        "mean_log_odds_shift_toward_donor": sum(log_odds_shifts_toward_donor) / max(1, len(log_odds_shifts_toward_donor)),
        "mean_state_entropy": sum(entropies) / max(1, len(entropies)),
    }
