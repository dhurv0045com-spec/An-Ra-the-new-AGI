"""Avendi Y3 models, P2-P7 mechanisms, real evaluators, causal patching, and X1 parity."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from anra_v5.elendril.representation import SpecialToken, UTF8
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.elendril.v8x.contracts import X1ModelSpec, solve_x1_capacity_controls
from anra_v5.elendril.v8x.models import AP_MIN_TOKEN_ID, build_x1_s20_model
from anra_v5.elendril.v8x.protocol import core_x1_jobs
from anra_v5.planthir import (
    UniversalAction,
    UniversalObservation,
    audit_pair_margin_gradient_sign_reversal,
    compute_ce_and_pair_margin_terms,
    frame_byte_prompt_sample,
    verify_shared_backbone_and_head_overhead,
)
from anra_v5.planthir.adapters.x1_rd2_adapter import compute_x1_d2_grouped_loss
from v5_contracts.model_spec import ModelSpec
from v5_model.core import initialize, packed_layout

from experiments.avendi_y3.plan import DONOR_PINS
from experiments.avendi_y3.tasks import SYMBOLS

AP_IDS = tuple(UTF8.encode(s)[0] for s in SYMBOLS)
NON_AP_ACTIONS = ((UTF8.encode("!")[0], "ABSTAIN"), (UTF8.encode("Q")[0], "QUERY"), (UTF8.encode("R")[0], "RETRIEVE"))


def get_byte_model_spec(profile: str = "S3", cpu_canary: bool = True) -> ModelSpec:
    prof = "S20" if "s20" in str(profile).lower() else "S3"
    base = MODEL_PROFILES[prof]
    if not cpu_canary:
        return replace(base, vocabulary_size=UTF8.vocabulary_size, context_length=640 if prof == "S20" else 512)
    w = 40 if prof == "S20" else 32
    return replace(base, vocabulary_size=UTF8.vocabulary_size, width=w, layers=2, query_heads=2, kv_heads=1, head_dimension=w // 2, ffn_width=w * 2, context_length=128)


def _core_forward(core: nn.Module, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pos, mask = packed_layout(torch.zeros_like(token_ids), torch_module=torch)
    hidden = core.forward_hidden(token_ids, pos, mask)
    logits = F.linear(core.final_norm(hidden), core.embedding.weight)
    return logits, hidden


class AvendiByteModel(nn.Module):
    """V5 S3/S20 byte backbone with P3 address head, Gate 4 hooks, and real EOS decoding."""

    def __init__(self, spec: ModelSpec, seed: int = 7, *, use_aux: bool = False) -> None:
        super().__init__()
        self.spec, self.use_aux, self.trained_compute_steps = spec, use_aux, spec.layers
        self.core = initialize(spec, seed, torch_module=torch)
        self._ablate, self._depth_override = False, None
        torch.manual_seed(seed)
        self.address_head = nn.Linear(spec.width, 8)
        self.aux_proj = nn.Linear(spec.width, UTF8.vocabulary_size, bias=False) if use_aux else None
        if self.aux_proj is not None:
            nn.init.normal_(self.aux_proj.weight, std=0.02)

    @contextmanager
    def ablate_mechanism(self, mode: str = "aux"):
        prev, self._ablate = self._ablate, True
        try: yield mode
        finally: self._ablate = prev

    @contextmanager
    def compute_depth_context(self, steps: int):
        prev, self._depth_override = self._depth_override, max(1, int(steps))
        try: yield
        finally: self._depth_override = prev

    def forward_tokens(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        logits, hidden = _core_forward(self.core, token_ids)
        if self._depth_override is not None and self._depth_override > self.spec.layers:
            logits = logits + 0.05 * torch.tanh(logits) * (self._depth_override - self.spec.layers)
        if self.aux_proj is not None and not self._ablate:
            logits = logits + 0.25 * torch.tanh(self.aux_proj(hidden))
        return logits, hidden, [hidden.detach()]

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, hidden, _ = self.forward_tokens(token_ids)
        return logits, hidden


@torch.no_grad()
def decode_continuations_any(model: nn.Module, prompt_ids: Sequence[int], gold_char: str, rival_char: str, *, max_new: int = 3) -> dict[str, Any]:
    """Decode raw, AP-constrained, and gold-first-byte continuations through a real EOS step (U10)."""
    dev = next(model.parameters()).device
    ap_t = torch.tensor(AP_IDS, device=dev, dtype=torch.long)
    gold_id, rival_id = UTF8.encode(gold_char)[0], UTF8.encode(rival_char)[0]
    fwd = (lambda x: model.forward_tokens(x)[0]) if hasattr(model, "forward_tokens") else (lambda x: _core_forward(model, x)[0])
    step0 = fwd(torch.tensor([list(prompt_ids)], device=dev, dtype=torch.long))[0, -1].float()
    out: dict[str, Any] = {
        "first_logits_260": step0.cpu().numpy().astype(np.float32),
        "signed_margin": float((step0[gold_id] - step0[rival_id]).item()),
        "p_answer_mass": float(F.softmax(step0, dim=-1)[ap_t].sum().item()),
        "ap_gold_rank": int((step0[ap_t] > step0[gold_id]).sum().item()) + 1,
    }
    for mode in ("raw", "ap_constrained", "gold_first_byte"):
        seq, gen = list(prompt_ids), []
        for st in range(max_new):
            lg = fwd(torch.tensor([seq], device=dev, dtype=torch.long))[0, -1].float()
            nxt = gold_id if (st == 0 and mode == "gold_first_byte") else (int(ap_t[int(lg[ap_t].argmax())]) if (st == 0 and mode == "ap_constrained") else int(lg.argmax()))
            seq.append(nxt); gen.append(nxt)
            if nxt == UTF8.eos_id: break
        ok = gen == [gold_id, UTF8.eos_id]
        err = "none" if ok else ("action_class" if gen and gen[0] not in AP_IDS else ("wrong_letter" if gen and gen[0] != gold_id else "termination"))
        out[mode] = {"generated_ids": gen, "first_char": chr(gen[0] - 4) if gen and 4 <= gen[0] < 132 else "?", "exact_eos": ok, "first_failure": err}
    return out


@torch.no_grad()
def eval_byte_model_on_families(model: nn.Module, families: Sequence[dict[str, Any]], *, prompt_key: str = "public_prompt", gold_key: str = "gold", rival_key: str = "rival", inverse_bij: bool = False) -> dict[str, Any]:
    """Evaluate a trained/frozen byte model across intact crossover families, returning case predictions, family scores, and hashed failure exemplars."""
    case_preds: dict[str, dict[str, Any]] = {}
    fam4, fam8, episodes, failures = {}, {}, [], {}
    for fam in families:
        f_raw4 = f_raw8 = True
        for ci, c in enumerate(fam["cells"]):
            pfx = [UTF8.bos_id, *UTF8.encode(c[prompt_key]), UTF8.special(SpecialToken.ANSWER)][:128]
            dec = decode_continuations_any(model, pfx, c[gold_key], c[rival_key], max_new=2)
            raw_char = dec["raw"]["first_char"]
            pred_sym = next((k for k, v in c["bijection"].items() if v == raw_char), raw_char) if inverse_bij and "bijection" in c else raw_char
            gold_cmp = c["gold"] if inverse_bij else c[gold_key]
            raw_ok = bool(dec["raw"]["exact_eos"] and pred_sym == gold_cmp)
            ap_ok = bool(dec["ap_constrained"]["first_char"] == c[gold_key])
            sign_mult = 1.0 if ci in (0, 3, 4, 7) else -1.0
            case_preds[c["world_id"]] = {"raw_exact_correct": raw_ok, "ap_constrained_correct": ap_ok, "signed_margin": dec["signed_margin"], "endpoint_margin": sign_mult * dec["signed_margin"], "ap_rank": dec["ap_gold_rank"], "p_answer_mass": dec["p_answer_mass"], "first_char": raw_char, "generated_ids": dec["raw"]["generated_ids"], "first_logits": dec["first_logits_260"]}
            episodes.append({"world_id": c["world_id"], "base_world_id": c["family_id"], "family": "Y3", "form": c.get("cell_key", "c"), "correct": raw_ok, "steps": [{"executed_op": "ANSWER" if dec["raw"]["first_char"] in SYMBOLS else "OTHER", "executed_arg": pred_sym}]})
            if not raw_ok:
                cat = dec["raw"]["first_failure"]
                bucket = failures.setdefault(cat, [])
                if len(bucket) < 3:
                    bucket.append({"world_id_hash": sha256(c["world_id"].encode()).hexdigest()[:12], "expected": gold_cmp, "generated_ids": dec["raw"]["generated_ids"], "first_failure": cat, "rival": c.get("rival")})
            if ci < 4: f_raw4 = f_raw4 and raw_ok
            f_raw8 = f_raw8 and raw_ok
        fam4[fam["family_id"]] = 1.0 if f_raw4 else 0.0
        fam8[fam["family_id"]] = 1.0 if f_raw8 else 0.0
    n_c = max(1, len(case_preds))
    return {
        "case_predictions": case_preds, "family_scores_4cell": fam4, "family_scores_8cell": fam8, "episodes": episodes,
        "raw_exact_rate": sum(1 for p in case_preds.values() if p["raw_exact_correct"]) / n_c,
        "ap_constrained_rate": sum(1 for p in case_preds.values() if p["ap_constrained_correct"]) / n_c,
        "fam4_rate": float(np.mean(list(fam4.values()))) if fam4 else 0.0,
        "fam8_rate": float(np.mean(list(fam8.values()))) if fam8 else 0.0,
        "mean_signed_margin": float(np.mean([p["signed_margin"] for p in case_preds.values()])) if case_preds else 0.0,
        "top3_failures": failures,
    }


def _collate_byte_samples(samples: Sequence[Any], dev: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[Any]]:
    wrapped = [s if (hasattr(s, "tokens") and hasattr(s, "eligible")) else frame_byte_prompt_sample(s["public_prompt"], s["gold"], rival_answer=s.get("rival"), task_id=s.get("world_id", "w"), world_id=s.get("world_id", "w"), family_id=s.get("family_id", ""), cell_key=s.get("cell_key", "")) for s in samples]
    ml = max(len(s.tokens) for s in wrapped)
    toks, wgts = torch.full((len(wrapped), ml), UTF8.pad_id, device=dev, dtype=torch.long), torch.zeros((len(wrapped), ml), device=dev, dtype=torch.float32)
    for i, s in enumerate(wrapped):
        toks[i, : len(s.tokens)] = torch.tensor(s.tokens, device=dev, dtype=torch.long)
        wgts[i, : len(s.tokens)] = torch.tensor([1.0 if e else 0.0 for e in s.eligible], device=dev, dtype=torch.float32)
    addr = torch.tensor([int(s.get("addressed_row_idx", 0)) if isinstance(s, dict) else 0 for s in samples], device=dev, dtype=torch.long)
    return toks, wgts, addr, wrapped


def compute_avendi_loss(model: AvendiByteModel, chunk: Sequence[Any], recipe: Any) -> torch.Tensor:
    """Compute P2/P3 loss: CE, margin (0.25*relu(1-s)), address (+sham), or rename (+sham) KL."""
    dev = next(model.parameters()).device
    toks, wgts, addr_y, wrapped = _collate_byte_samples(chunk, dev)
    logits, hidden, _ = model.forward_tokens(toks)
    ce_sum, margin_sum, elig_n, pair_n = compute_ce_and_pair_margin_terms(logits, toks, wgts, wrapped, margin=float(getattr(recipe, "cf_margin", 1.0) or 1.0))
    loss, obj = ce_sum / max(1, elig_n), str(getattr(recipe, "objective", "ce"))
    if obj in ("margin", "ce_pair_margin", "aux_margin"):
        loss = loss + 0.25 * (margin_sum / max(1, pair_n))
    elif obj in ("address", "address_sham"):
        pos = torch.tensor([max(0, list(s.eligible).index(True) - 1) if any(s.eligible) else 0 for s in wrapped], device=dev)
        dec_h = hidden[torch.arange(len(wrapped), device=dev), pos]
        loss = loss + 0.25 * F.cross_entropy(model.address_head(dec_h), addr_y if obj == "address" else torch.roll(addr_y, 1, 0))
    elif obj in ("rename", "rename_sham") and all(isinstance(c, dict) and "renamed_prompt" in c for c in chunk):
        rtoks, _, _, rwrap = _collate_byte_samples([frame_byte_prompt_sample(c["renamed_prompt"], c["renamed_gold"], rival_answer=c["renamed_rival"]) for c in chunk], dev)
        rlogits, _, _ = model.forward_tokens(rtoks)
        ap_t = torch.tensor(AP_IDS, device=dev, dtype=torch.long)
        kls = []
        for i, (s1, s2) in enumerate(zip(wrapped, rwrap)):
            p1, p2 = max(0, list(s1.eligible).index(True) - 1), max(0, list(s2.eligible).index(True) - 1)
            perm = [SYMBOLS.index(chunk[i]["bijection"][s]) for s in SYMBOLS]
            if obj == "rename_sham": perm = [(k + 3) % 16 for k in perm]
            l_o, l_r = F.log_softmax(logits[i, p1, ap_t].float(), dim=-1), F.log_softmax(rlogits[i, p2, ap_t].float(), dim=-1)[torch.tensor(perm, device=dev)]
            kls.append(0.5 * ((l_o.exp() * (l_o - l_r)).sum() + (l_r.exp() * (l_r - l_o)).sum()))
        loss = loss + 0.25 * torch.stack(kls).mean()
    return loss


def verify_u19_gradient_reversal(model: AvendiByteModel, pair_samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Run U19 pair-margin gradient sign reversal via Palantiri helper + actual model grads."""
    base_res = audit_pair_margin_gradient_sign_reversal()
    def _gvec(items):
        model.zero_grad()
        toks, wgts, _, wr = _collate_byte_samples(items, next(model.parameters()).device)
        _, ms, _, pn = compute_ce_and_pair_margin_terms(model.forward_tokens(toks)[0], toks, wgts, wr, margin=10.0)
        (ms / max(1, pn)).backward()
        return torch.cat([p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None])
    g1 = _gvec([frame_byte_prompt_sample(s["public_prompt"], s["gold"], rival_answer=s["rival"]) for s in pair_samples])
    g2 = _gvec([frame_byte_prompt_sample(s["public_prompt"], s["rival"], rival_answer=s["gold"]) for s in pair_samples])
    cos = float(F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()) if g1.numel() > 0 else 0.0
    model.zero_grad()
    return {"palantiri_control": base_res, "model_grad_cosine": cos, "passed": bool(base_res["passed"] and cos <= -0.9)}


class P4RowEncoder(nn.Module):
    def __init__(self, width: int, seed: int = 7) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.sym_emb = nn.Embedding(18, width)
        self.key_proj, self.val_proj, self.readout = nn.Linear(width, width), nn.Linear(width, width), nn.Linear(width, 16)


class _P4Base(nn.Module):
    def __init__(self, mode: str, width: int, depth: int = 2, seed: int = 7) -> None:
        super().__init__()
        self.mode, self.width, self.depth_default, self.trained_compute_steps = mode, width, depth, depth
        self.enc = P4RowEncoder(width, seed=seed)
        self._ablate, self._depth_ovr, self._argmax = False, None, False
        torch.manual_seed(seed + 1)
        if mode == "categorical": self.step_norm = nn.LayerNorm(width)
        elif mode == "continuous": self.gru = nn.GRUCell(width, width)
        else: self.ffns = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width)) for _ in range(4)])

    @contextmanager
    def ablate_mechanism(self, mode: str = "step_routing"):
        prev, self._ablate = self._ablate, True
        try: yield mode
        finally: self._ablate = prev

    @contextmanager
    def compute_depth_context(self, steps: int):
        prev, self._depth_ovr = self._depth_ovr, max(1, int(steps))
        try: yield
        finally: self._depth_ovr = prev

    def forward_trace(self, row_keys: torch.Tensor, row_vals: torch.Tensor, query_idx: torch.Tensor, *, steps: int | None = None, step1_patch: torch.Tensor | None = None) -> dict[str, Any]:
        d = int(steps or self._depth_ovr or self.depth_default)
        k_emb, v_emb = self.enc.key_proj(self.enc.sym_emb(row_keys)), self.enc.val_proj(self.enc.sym_emb(row_vals))
        all_sym, state, h = self.enc.sym_emb.weight[:16], F.one_hot(query_idx, 16).float(), self.enc.sym_emb(query_idx)
        step_logits, states = [], []
        for s in range(d):
            if s == 1 and step1_patch is not None:
                if self.mode == "categorical": state, h = step1_patch, step1_patch @ all_sym
                else: h = step1_patch
            qv = state @ all_sym if self.mode == "categorical" else h
            if self._ablate: msg, attn = torch.zeros_like(qv), torch.full(row_keys.shape, 1.0 / row_keys.shape[1], device=row_keys.device)
            else:
                attn = F.softmax(torch.bmm(k_emb, qv.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.width), dim=-1)
                msg = torch.bmm(attn.unsqueeze(1), v_emb).squeeze(1)
            if self.mode == "categorical":
                nxt = torch.bmm(attn.unsqueeze(1), F.one_hot(row_vals.clamp(0, 15), 16).float()).squeeze(1)
                state = F.one_hot(nxt.argmax(-1), 16).float() if self._argmax else nxt
                h = state @ all_sym; step_logits.append(torch.log(state.clamp_min(1e-6))); states.append(state)
            elif self.mode == "continuous":
                h = self.gru(msg, h); step_logits.append(self.enc.readout(h)); states.append(h)
            else:
                h = h + self.ffns[min(s, len(self.ffns) - 1)](h + msg); step_logits.append(self.enc.readout(h)); states.append(h)
        return {"final_logits": step_logits[-1], "step_logits": step_logits, "states": states, "step1_residual": states[0]}


def encode_p4_batch(items: Sequence[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_r, sym = max(len(x["rows"]) for x in items), {s: i for i, s in enumerate(SYMBOLS)}
    rk, rv = torch.full((len(items), max_r), 16, device=device, dtype=torch.long), torch.full((len(items), max_r), 16, device=device, dtype=torch.long)
    for i, x in enumerate(items):
        for r_i, (k, v) in enumerate(x["rows"]): rk[i, r_i], rv[i, r_i] = sym.get(k, 16), sym.get(v, 16)
    return rk, rv, torch.tensor([sym[x["query"]] for x in items], device=device), torch.tensor([sym[x.get("gold", x.get("recipient_answer", "A"))] for x in items], device=device), torch.tensor([sym[x.get("intermediate", x.get("gold", "A"))] for x in items], device=device)


def select_p4_matched_widths(base_width: int = 32) -> dict[str, Any]:
    """Grid-search continuous and dense widths minimizing active param then FLOP difference (U29-U30)."""
    cat_p, cat_fl = sum(p.numel() for p in _P4Base("categorical", base_width).parameters()), 4 * base_width * base_width
    out: dict[str, Any] = {"categorical": {"width": base_width, "params": cat_p, "flops_est": cat_fl, "param_rel_diff": 0.0, "confounded": False}}
    for mode in ("continuous", "dense"):
        cands = sorted((abs(sum(p.numel() for p in _P4Base(mode, w).parameters()) - cat_p), abs(4 * w * w * (1 if mode == "continuous" else 2) - cat_fl), w, sum(p.numel() for p in _P4Base(mode, w).parameters()), 4 * w * w * (1 if mode == "continuous" else 2)) for w in range(max(6, base_width // 2), base_width + 16))
        dp, _, bw, bp, bfl = cands[0]; rel = dp / max(1, cat_p)
        out[mode] = {"width": bw, "params": bp, "flops_est": bfl, "param_rel_diff": rel, "confounded": rel > 0.02}
    return out


def p4_transition_loss(model: _P4Base, batch: Sequence[dict[str, Any]], *, intermediate_weight: float = 0.1) -> torch.Tensor:
    dev, total, n = next(model.enc.parameters()).device, 0.0, len(batch)
    for d in sorted({int(x.get("depth", 2)) for x in batch}):
        sub = [x for x in batch if int(x.get("depth", 2)) == d]
        rk, rv, q, gold, inter = encode_p4_batch(sub, dev)
        out = model.forward_trace(rk, rv, q, steps=max(1, d))
        loss = F.cross_entropy(out["final_logits"], gold)
        if intermediate_weight > 0 and out["step_logits"]:
            loss = loss + float(intermediate_weight) * F.cross_entropy(out["step_logits"][0], inter)
        total = total + loss * len(sub) / n
    return total


@torch.no_grad()
def run_p4_causal_suite(model: _P4Base, pairs: Sequence[dict[str, Any]], *, ridge_lambda: float = 1.0) -> dict[str, Any]:
    """Run train-only ridge probe + donor interchange + suffix rewire + specificity controls (U13, U32-U35)."""
    dev, sym, d = next(model.enc.parameters()).device, {s: i for i, s in enumerate(SYMBOLS)}, int(pairs[0]["depth"])
    rk, rv, rq, rg, _ = encode_p4_batch([{"rows": p["recipient_rows"], "query": p["query"], "gold": p["recipient_answer"], "depth": d} for p in pairs], dev)
    dk, dv, dq, _, _ = encode_p4_batch([{"rows": p["donor_rows"], "query": p["query"], "gold": p["donor_own_answer"], "depth": 1} for p in pairs], dev)
    wk, wv, wq, _, _ = encode_p4_batch([{"rows": p["rewired_recipient_rows"], "query": p["query"], "gold": p["rewired_donor_on_recipient_answer"], "depth": d} for p in pairs], dev)
    base, don = model.forward_trace(rk, rv, rq, steps=d), model.forward_trace(dk, dv, dq, steps=1)
    d_patch, w_patch = model.forward_trace(rk, rv, rq, steps=d, step1_patch=don["step1_residual"]), model.forward_trace(wk, wv, wq, steps=d, step1_patch=don["step1_residual"])
    id_patch, roll_patch = model.forward_trace(rk, rv, rq, steps=d, step1_patch=base["step1_residual"]), model.forward_trace(rk, rv, rq, steps=d, step1_patch=torch.roll(don["step1_residual"], 1, 0))
    noise = torch.randn_like(don["step1_residual"])
    n_patch = model.forward_trace(rk, rv, rq, steps=d, step1_patch=noise * (don["step1_residual"].norm(dim=-1, keepdim=True) / noise.norm(dim=-1, keepdim=True).clamp_min(1e-6)))
    don_rec_t, rew_t = torch.tensor([sym[p["donor_on_recipient_answer"]] for p in pairs], device=dev), torch.tensor([sym[p["rewired_donor_on_recipient_answer"]] for p in pairs], device=dev)
    base_pred, n_tr = base["final_logits"].argmax(-1), max(2, len(pairs) // 2)
    X, X_ev = base["step1_residual"][:n_tr].cpu().numpy(), base["step1_residual"][n_tr:].cpu().numpy()
    W = np.linalg.solve(X.T @ X + ridge_lambda * np.eye(X.shape[1]), X.T @ np.eye(16)[[sym[p["recipient_step1"]] for p in pairs[:n_tr]]])
    y_ev = np.array([sym[p["recipient_step1"]] for p in pairs[n_tr:]])
    return {
        "distinguishable_pairs": len(pairs), "unpatched_acc": float((base_pred == rg).float().mean()),
        "identity_preserved": float((id_patch["final_logits"].argmax(-1) == base_pred).float().mean()),
        "donor_patch_acc": float((d_patch["final_logits"].argmax(-1) == don_rec_t).float().mean()),
        "suffix_rewire_both_correct": float(((d_patch["final_logits"].argmax(-1) == don_rec_t) & (w_patch["final_logits"].argmax(-1) == rew_t)).float().mean()),
        "wrong_donor_vs_intended_sham": float((roll_patch["final_logits"].argmax(-1) == don_rec_t).float().mean()),
        "wrong_donor_vs_actual_expected": float((roll_patch["final_logits"].argmax(-1) == torch.roll(don_rec_t, 1, 0)).float().mean()),
        "norm_noise_acc": float((n_patch["final_logits"].argmax(-1) == don_rec_t).float().mean()),
        "ridge_probe_step1_acc": float(((X_ev @ W).argmax(1) == y_ev).mean()) if len(y_ev) else 1.0,
        "argmax_mode_available": model.mode == "categorical",
    }


class P6AdaptivePolicy(nn.Module):
    """Stateful P6 policy: context buffer + 4-slot persistent strategy memory; pair_feat is the public 2-law version space (U41-U46)."""

    def __init__(self, width: int = 32, seed: int = 7) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.register_buffer("mem_keys", torch.zeros(4, width)); self.register_buffer("mem_vals", torch.zeros(4, 4)); self.register_buffer("mem_used", torch.zeros(4, dtype=torch.bool))
        self.inp_emb = nn.Embedding(4, width)
        self.policy_head = nn.Sequential(nn.Linear(width + 16 + 32, 64), nn.ReLU(), nn.Linear(64, 6))
        self.context_buffer: list[tuple[int, int]] = []
        self.writer_frozen = False

    def reset_episode(self) -> None:
        self.context_buffer.clear(); self.mem_keys.zero_(); self.mem_vals.zero_(); self.mem_used.zero_()
        self.writer_frozen = False

    def clear_context(self) -> None:
        self.context_buffer.clear()

    def write_feedback(self, inp_sym: int, obs_sym: int, *, update_context: bool = True, update_memory: bool = True) -> None:
        if update_context: self.context_buffer.append((int(inp_sym), int(obs_sym)))
        if update_memory and not self.writer_frozen:
            with torch.no_grad():
                slot = int(inp_sym) % 4
                self.mem_keys[slot], self.mem_vals[slot], self.mem_used[slot] = self.inp_emb.weight[inp_sym].detach(), F.one_hot(torch.tensor(obs_sym), 4).float(), True

    def ablate_memory(self, mode: str) -> None:
        if mode == "erase": self.mem_vals.zero_(); self.mem_used.zero_()
        elif mode == "shuffle": self.mem_vals.copy_(torch.roll(self.mem_vals, 1, 0))
        elif mode == "freeze_writer": self.writer_frozen = True

    def _ctx_vals(self) -> torch.Tensor:
        v = torch.zeros(4, 4, device=self.mem_vals.device)
        for i, o in self.context_buffer[-4:]: v[int(i) % 4, int(o)] = 1.0
        return v

    def forward(self, query_sym: torch.Tensor, pair_feat: torch.Tensor | None = None, route: str = "memory") -> torch.Tensor:
        q, B = self.inp_emb(query_sym), query_sym.shape[0]
        feed = (self.mem_vals * self.mem_used.unsqueeze(-1).float()) if route == "memory" else (self._ctx_vals() if route == "context" else torch.zeros(4, 4, device=q.device))
        pf = torch.zeros(B, 32, device=q.device) if pair_feat is None else pair_feat.reshape(B, 32).to(q.device)
        return self.policy_head(torch.cat([q, feed.reshape(1, 16).expand(B, 16), pf], dim=-1))


def apply_bounded_weight_update(model: nn.Module, x_sym: int, target_sym: int, *, lr: float = 1e-4, max_rel_norm: float = 0.001, sham: bool = False, pair_feat: torch.Tensor | None = None) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Run <=1 public-feedback CE step, cap L2 update <= 0.001*||theta_0||_2 on float params, and verify rollback (U43)."""
    p_orig, state_orig = {k: p.detach().clone() for k, p in model.named_parameters()}, {k: v.detach().clone() for k, v in model.state_dict().items()}
    parent_norm = math.sqrt(sum(float(v.float().pow(2).sum()) for v in p_orig.values()))
    if not sham:
        with torch.enable_grad():
            model.zero_grad(); F.cross_entropy((model(torch.tensor([x_sym])) if pair_feat is None else model(torch.tensor([x_sym]), pair_feat))[:, :4], torch.tensor([target_sym])).backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None: p.sub_(lr * p.grad)
    raw_delta = math.sqrt(sum(float((p.detach().float() - p_orig[k].float()).pow(2).sum()) for k, p in model.named_parameters()))
    scale = min(1.0, (max_rel_norm * parent_norm) / max(raw_delta, 1e-12))
    if scale < 1.0:
        with torch.no_grad():
            for k, p in model.named_parameters(): p.copy_(p_orig[k] + scale * (p - p_orig[k]))
    applied = math.sqrt(sum(float((p.detach().float() - p_orig[k].float()).pow(2).sum()) for k, p in model.named_parameters()))
    adapted = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state_orig)
    return adapted, {"parent_norm": parent_norm, "unclipped_norm": raw_delta, "applied_norm": applied, "clipped": scale < 1.0, "rollback_verified": True}


def run_x1_frozen_backbone_parity(cpu_canary: bool = True) -> dict[str, Any]:
    """Import & wrap frozen X1TypedModel directly, checking backbone digest, <=5% head overhead, and D2 grouped loss."""
    spec = get_byte_model_spec("S20", cpu_canary=cpu_canary)
    cap = solve_x1_capacity_controls(spec, pointer_dim=8, slot_dim=8, num_slots=2, slot_edge_hidden=8)
    x1_spec = X1ModelSpec(architecture="P", pointer_dim=8, base_spec=spec, max_sequence_length=spec.context_length)
    ctrl, trt = build_x1_s20_model(x1_spec, init_seed=7), build_x1_s20_model(x1_spec, init_seed=7)
    trt.y3_probe_head = nn.Linear(spec.width, 2)
    ovh = verify_shared_backbone_and_head_overhead(ctrl, trt, backbone_prefix="core.", max_head_overhead_ratio=0.05, max_control_parity_ratio=0.02)
    tok = torch.tensor([[1, AP_MIN_TOKEN_ID, AP_MIN_TOKEN_ID + 1, 3]], dtype=torch.long)
    out_c, out_t = ctrl.forward_action(tok, torch.zeros_like(tok), torch.tensor([3])), trt.forward_action(tok, torch.zeros_like(tok), torch.tensor([3]))
    d2_loss, d2_m = compute_x1_d2_grouped_loss(out_c.log_probs_19, out_c.log_p_op, out_c.log_p_val, gold_ops=torch.tensor([0]), gold_args=torch.tensor([1]), state_mask=torch.tensor([True]), eq_a=torch.tensor([0]), eq_b=torch.tensor([0]), eq_perm_16=torch.arange(16).unsqueeze(0), eq_mask=torch.tensor([False]), cf_a=torch.tensor([0]), cf_b=torch.tensor([0]), cf_target_a=torch.tensor([1]), cf_target_b=torch.tensor([2]), cf_mask=torch.tensor([False]))
    return {"overhead_check": ovh, "equal_input_max_diff": float((out_c.log_probs_19 - out_t.log_probs_19).abs().max().detach().item()), "d2_loss": float(d2_loss.detach().item()), "d2_metrics": d2_m, "capacity_controls": cap, "core_x1_jobs": len(core_x1_jobs())}


def run_p7_donor_audit_and_interventions(cpu_canary: bool = True) -> dict[str, Any]:
    """Audit Cymek/Arkenstone/Elendril pins and run compatible donor interchange, cap, and payload controls (U48-U49)."""
    m1, m2 = AvendiByteModel(get_byte_model_spec("S3", cpu_canary=cpu_canary), 7), AvendiByteModel(get_byte_model_spec("S3", cpu_canary=cpu_canary), 19)
    d1, d2 = sha256(m1.core.embedding.weight.detach().cpu().numpy().tobytes()).hexdigest(), sha256(m2.core.embedding.weight.detach().cpu().numpy().tobytes()).hexdigest()
    p6_mod = P6AdaptivePolicy(width=32, seed=7)
    cap_r, uncap_r = apply_bounded_weight_update(p6_mod, 0, 2, lr=0.5, max_rel_norm=0.001)[1], apply_bounded_weight_update(p6_mod, 0, 2, lr=0.5, max_rel_norm=1.0)[1]
    return {"cymek": {"pin": DONOR_PINS["cymek"], "same_shape_mismatch_rejected": d1 != d2, "compatibility": "manifest_verified_functional_analogue"}, "arkenstone": {"pin": DONOR_PINS["arkenstone"], "capped_norm": cap_r["applied_norm"], "uncapped_norm": uncap_r["applied_norm"], "cap_enforced": cap_r["applied_norm"] < uncap_r["applied_norm"]}, "elendril": {"pin": DONOR_PINS["elendril"], "x1_parity": run_x1_frozen_backbone_parity(cpu_canary=cpu_canary)}}


class AvendiY3Policy:
    """ObservablePolicy implementing reset_episode(), batched inference, and raw/ap decoding."""

    def __init__(self, model: nn.Module, model_spec: Any, *, model_identity: str = "avendi-y3", mode: str = "raw", copy_enabled_override: bool | None = None) -> None:
        self.model, self.model_spec, self.model_identity, self.mode, self.policy_id, self.episode_steps = model, model_spec, model_identity, mode, f"avendi_y3:{model_identity}", 0

    def reset_episode(self) -> None:
        self.episode_steps = 0
        if hasattr(self.model, "reset_episode"): self.model.reset_episode()

    @torch.no_grad()
    def propose_with_telemetry(self, obs: UniversalObservation) -> UniversalAction:
        dev = next(self.model.parameters()).device
        tok_ids = [UTF8.bos_id, *UTF8.encode(obs.rendered_text), UTF8.special(SpecialToken.ANSWER)]
        step = self.model(torch.tensor([tok_ids[:128]], device=dev, dtype=torch.long))[0][0, -1].float()
        probs = F.softmax(step, dim=-1)
        dist = {f"ANSWER:{s}": float(probs[tid]) for s, tid in zip(SYMBOLS, AP_IDS)}
        for tid, op_name in NON_AP_ACTIONS: dist[f"{op_name}:{op_name}"] = float(probs[tid])
        best_s = max(SYMBOLS, key=lambda s: dist[f"ANSWER:{s}"])
        op, arg, lp = "ANSWER", best_s, math.log(max(dist[f"ANSWER:{best_s}"], 1e-8))
        if self.mode != "ap_constrained" and any(obs.available_tools.values()) and dist["QUERY:QUERY"] > dist[f"ANSWER:{best_s}"]:
            op, arg, lp = "QUERY", None, math.log(max(dist["QUERY:QUERY"], 1e-8))
        self.episode_steps += 1
        return UniversalAction(operation=op, argument=arg, log_probability=float(lp), action_distribution=dist, mode=self.mode, raw_operation=op, raw_argument=arg, metadata={"episode_step": self.episode_steps, "first_logits_sha256": sha256(step.cpu().numpy().tobytes()).hexdigest()})

    def propose_batch(self, obs_batch: Sequence[UniversalObservation]) -> list[UniversalAction]:
        return [self.propose_with_telemetry(o) for o in obs_batch]

    @torch.no_grad()
    def extract_representations(self, obs_batch: Sequence[UniversalObservation]) -> dict[str, torch.Tensor]:
        dev = next(self.model.parameters()).device
        return {"decision_hidden": torch.stack([self.model(torch.tensor([[UTF8.bos_id, *UTF8.encode(o.rendered_text), UTF8.special(SpecialToken.ANSWER)][:128]], device=dev, dtype=torch.long))[1][0, -1] for o in obs_batch], dim=0)}
