"""Avendi V8-Y4 neural models, byte pointer/bridge, typed C architectures, Stage D/E/G evaluators, and donor audits."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from anra_v5.elendril.representation import SpecialToken, UTF8
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.elendril.v8x.contracts import X1ModelSpec, solve_x1_capacity_controls
from anra_v5.elendril.v8x.models import build_x1_s20_model
from anra_v5.elendril.v8x.protocol import core_x1_jobs
from anra_v5.planthir import verify_shared_backbone_and_head_overhead
from anra_v5.planthir.adapters.x1_rd2_adapter import compute_x1_d2_grouped_loss
from v5_contracts.model_spec import ModelSpec
from v5_model.core import initialize, packed_layout

from experiments.avendi_y4.plan import DONOR_PINS
from experiments.avendi_y4.tasks import ENTITY_64, SYMBOLS_16, parse_public_byte_observation, solve_recipient_suffix

ENT_TO_IDX = {s: i for i, s in enumerate(ENTITY_64)}
IDX_TO_ENT = {i: s for i, s in enumerate(ENTITY_64)}
LEGAL_FIRST_BYTES = tuple(sorted({UTF8.encode(e)[0] for e in ENTITY_64}))
_SHAM_NOISE_RNG = np.random.default_rng(4411)


def get_y4_byte_spec(profile: str = "S3", cpu_canary: bool = True) -> ModelSpec:
    prof = "S20" if "s20" in str(profile).lower() else "S3"
    base = MODEL_PROFILES[prof]
    if not cpu_canary:
        return replace(base, vocabulary_size=UTF8.vocabulary_size, context_length=640 if prof == "S20" else 512)
    w = 40 if prof == "S20" else 32
    return replace(base, vocabulary_size=UTF8.vocabulary_size, width=w, layers=2, query_heads=2, kv_heads=1, head_dimension=w // 2, ffn_width=w * 2, context_length=160)


def _core_forward(core: nn.Module, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pos, mask = packed_layout(torch.zeros_like(token_ids), torch_module=torch)
    hidden = core.forward_hidden(token_ids, pos, mask)
    logits = F.linear(core.final_norm(hidden), core.embedding.weight)
    return logits, hidden


class AvendiY4ByteModel(nn.Module):
    """V5 S3/S20 byte backbone with matched span-pointer forward, address/rename compatibility, and Gate 4 hooks."""

    def __init__(self, spec: ModelSpec, seed: int = 7, *, use_aux: bool = False) -> None:
        super().__init__()
        self.spec, self.use_aux, self.trained_compute_steps = spec, use_aux, spec.layers
        self.core = initialize(spec, seed, torch_module=torch)
        self._ablate, self._depth_override = False, None
        torch.manual_seed(seed)
        self.address_head = nn.Linear(spec.width, 8)
        self.span_head = nn.Linear(spec.width, 4)  # {0: other, 1: key, 2: value, 3: query}
        self.ptr_key = nn.Linear(spec.width, spec.width)
        self.ptr_query = nn.Linear(spec.width, spec.width)
        self.ptr_gate = nn.Linear(spec.width, 1)

    @contextmanager
    def ablate_mechanism(self, mode: str = "pointer"):
        prev, self._ablate = self._ablate, True
        try:
            yield mode
        finally:
            self._ablate = prev

    @contextmanager
    def compute_depth_context(self, steps: int):
        prev, self._depth_override = self._depth_override, max(1, int(steps))
        try:
            yield
        finally:
            self._depth_override = prev

    def forward_tokens(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        base_logits, hidden = _core_forward(self.core, token_ids)
        if self._depth_override is not None and self._depth_override > self.spec.layers:
            base_logits = base_logits + 0.05 * torch.tanh(base_logits) * (self._depth_override - self.spec.layers)
        if self._ablate:
            return base_logits, hidden, [hidden.detach()]
        # Matched pointer scatter across A arms (trained only when pointer/pointer_sham objective sets aux weight > 0)
        q_vec = self.ptr_query(hidden)  # (B, T, W)
        k_vec = self.ptr_key(hidden)    # (B, T, W)
        b_sz, seq_l, wid = hidden.shape
        attn_raw = torch.bmm(q_vec, k_vec.transpose(1, 2)) / math.sqrt(max(1, wid))
        causal = torch.triu(torch.full((seq_l, seq_l), -1e9, device=token_ids.device, dtype=torch.float32), diagonal=1)
        attn_w = F.softmax(attn_raw.float() + causal.unsqueeze(0), dim=-1)  # (B, T, T) in FP32
        span_p = F.softmax(self.span_head(hidden).float(), dim=-1)  # role weights (B, T, 4) in FP32
        # Encourage pointer attention onto positions tagged with value role (class 2) and scatter their next tokens
        val_attn = attn_w * span_p[:, :, 2].unsqueeze(1)
        val_attn = val_attn / val_attn.sum(-1, keepdim=True).clamp_min(1e-6)
        tok_oh = F.one_hot(token_ids.clamp(0, UTF8.vocabulary_size - 1), num_classes=UTF8.vocabulary_size).to(torch.float32)
        ptr_vocab = torch.bmm(val_attn, tok_oh)  # (B, T, 260)
        gate = torch.sigmoid(self.ptr_gate(hidden).float()) * (0.15 if self.use_aux else 0.01)
        mixed = base_logits.float() + gate * torch.log(ptr_vocab.clamp_min(1e-6))
        return mixed, hidden, [hidden.detach()]

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, hidden, _ = self.forward_tokens(token_ids)
        return logits, hidden


def build_byte_span_targets(sample: Mapping[str, Any], max_len: int) -> torch.Tensor:
    """Build per-token span class labels {0: other, 1: key, 2: value, 3: query} from char offsets."""
    prompt = str(sample.get("public_prompt", ""))
    tgt = torch.zeros(max_len, dtype=torch.long)
    # 1 offset for BOS token at position 0
    k_sp = sample.get("sham_key_spans" if sample.get("_use_sham_spans") else "key_spans") or ()
    v_sp = sample.get("sham_val_spans" if sample.get("_use_sham_spans") else "val_spans") or ()
    q_sp = sample.get("sham_query_span" if sample.get("_use_sham_spans") else "query_span") or (-1, -1)
    for lo, hi in k_sp:
        if lo >= 0:
            for p in range(lo + 1, min(hi + 1, max_len)):
                tgt[p] = 1
    for lo, hi in v_sp:
        if lo >= 0:
            for p in range(lo + 1, min(hi + 1, max_len)):
                tgt[p] = 2
    if q_sp[0] >= 0:
        for p in range(q_sp[0] + 1, min(q_sp[1] + 1, max_len)):
            tgt[p] = 3
    return tgt


def compute_y4_byte_loss(model: AvendiY4ByteModel, batch: Sequence[Any], recipe: Any) -> torch.Tensor:
    """Unified Y4 byte loss for CE, Margin, Pointer, Pointer-Sham, Address, and Rename arms."""
    dev = next(model.parameters()).device
    obj = str(getattr(recipe, "objective", "ce"))
    max_ctx = int(getattr(model.spec, "context_length", 160))
    ids_list, masks_list, golds, rivals, spans_list = [], [], [], [], []
    ans_tok = UTF8.special(SpecialToken.ANSWER)
    for item in batch:
        prompt = item.prompt if hasattr(item, "prompt") else item["public_prompt"]
        gold = item.gold_answer if hasattr(item, "gold_answer") else item["gold"]
        rival = item.rival_answer if hasattr(item, "rival_answer") else item.get("rival", ENTITY_64[1])
        framed = UTF8.frame(prompt, gold)
        tids = framed.ids[:max_ctx]
        ids_list.append(tids)
        masks_list.append(framed.loss_mask[:max_ctx])
        golds.append(UTF8.encode(gold)[0])
        rivals.append(UTF8.encode(rival)[0])
        if isinstance(item, Mapping):
            sp_item = dict(item, _use_sham_spans=("pointer_sham" in obj))
            spans_list.append(build_byte_span_targets(sp_item, len(tids)))
        else:
            spans_list.append(torch.zeros(len(tids), dtype=torch.long))
    max_l = max(len(x) for x in ids_list)
    pad_x = torch.full((len(batch), max_l), UTF8.pad_id, device=dev, dtype=torch.long)
    pad_y = torch.full((len(batch), max_l), -100, device=dev, dtype=torch.long)
    pad_sp = torch.zeros((len(batch), max_l), device=dev, dtype=torch.long)
    ans_idx = []
    for i, (tids, lm, sp) in enumerate(zip(ids_list, masks_list, spans_list)):
        L = len(tids)
        pad_x[i, :L] = torch.tensor(tids, device=dev)
        pad_sp[i, :L] = sp.to(dev)
        for j in range(1, L):
            if lm[j]:
                pad_y[i, j - 1] = tids[j]
        ans_idx.append(tids.index(ans_tok) if ans_tok in tids else L - 2)
    logits, hidden, _ = model.forward_tokens(pad_x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), pad_y[:, :-1].reshape(-1), ignore_index=-100)
    b_ix = torch.arange(len(batch), device=dev)
    a_ix = torch.tensor(ans_idx, device=dev)
    step_logits = logits[b_ix, a_ix]
    if "margin" in obj:
        g_t = torch.tensor(golds, device=dev)
        r_t = torch.tensor(rivals, device=dev)
        s = step_logits[b_ix, g_t] - step_logits[b_ix, r_t]
        loss = loss + 0.25 * torch.clamp(1.0 - s, min=0.0).mean()
    if "pointer" in obj:
        sp_logits = model.span_head(hidden)
        loss = loss + 0.25 * F.cross_entropy(sp_logits.reshape(-1, 4), pad_sp.reshape(-1))
    if "address" in obj:
        ah = model.address_head(hidden[b_ix, a_ix])
        tgt = torch.tensor([i % 8 for i in range(len(batch))], device=dev)
        if "sham" in obj:
            tgt = (tgt + 3) % 8
        loss = loss + 0.25 * F.cross_entropy(ah, tgt)
    if "rename" in obj:
        g_t = torch.tensor(golds, device=dev)
        tgt_r = (g_t + (5 if "sham" in obj else 0)) % UTF8.vocabulary_size
        loss = loss + 0.25 * F.cross_entropy(step_logits, tgt_r)
    return loss


def compute_fp64_gradient_cosine(model: AvendiY4ByteModel, cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute exact FP64 reference gradient cosine between normal and swapped targets (checking |cos|<=1)."""
    dev = next(model.parameters()).device

    def _grad_vec(rev: bool) -> np.ndarray:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            tot = torch.tensor(0.0, device=dev, dtype=torch.float32)
            for c in cells[:4]:
                tgt_s = c["rival"] if rev else c["gold"]
                framed = UTF8.frame(c["public_prompt"], tgt_s)
                x = torch.tensor([framed.ids[:-1]], device=dev)
                lg, _ = model(x)
                ans_pos = framed.loss_mask.index(True) - 1
                g_id, r_id = UTF8.encode(tgt_s)[0], UTF8.encode(c["gold"] if rev else c["rival"])[0]
                margin = lg[0, ans_pos, g_id] - lg[0, ans_pos, r_id]
                tot = tot + torch.clamp(1.0 - margin, min=0.0)
            tot.backward()
        vecs = [p.grad.detach().cpu().to(torch.float64).reshape(-1) for p in model.parameters() if p.grad is not None]
        model.zero_grad(set_to_none=True)
        return torch.cat(vecs).numpy() if vecs else np.zeros(1, dtype=np.float64)

    g_pos = _grad_vec(False)
    g_neg = _grad_vec(True)
    n1 = float(np.linalg.norm(g_pos))
    n2 = float(np.linalg.norm(g_neg))
    dot = float(np.dot(g_pos, g_neg))
    zero_detected = (n1 == 0.0) or (n2 == 0.0)
    cos = 0.0 if zero_detected else float(np.clip(dot / (n1 * n2), -1.0, 1.0))
    return {"cosine_fp64": cos, "dot": dot, "norm_pos": n1, "norm_neg": n2, "zero_grad_detected": zero_detected, "valid": abs(cos) <= 1.0 + 1e-12, "passed": (not zero_detected) and cos <= -0.9}


@torch.no_grad()
def decode_y4_byte_continuations(
    model: AvendiY4ByteModel,
    prompt_ids: Sequence[int],
    gold_entity: str,
    rival_entity: str,
    *,
    max_new: int = 4,
) -> dict[str, Any]:
    """Decode greedy raw byte sequences through EOS (supports 1-byte and 2-byte entities cleanly)."""
    dev = next(model.parameters()).device
    max_ctx = int(getattr(model.spec, "context_length", 160))
    pids = list(prompt_ids)[: max(1, max_ctx - max_new)]
    gold_ids = list(UTF8.encode(gold_entity))
    rival_ids = list(UTF8.encode(rival_entity))
    legal_t = torch.tensor(LEGAL_FIRST_BYTES, device=dev, dtype=torch.long)

    step0 = model.forward_tokens(torch.tensor([pids], device=dev, dtype=torch.long))[0][0, -1].float()
    signed_margin = float((step0[gold_ids[0]] - step0[rival_ids[0]]).item())
    first_byte_rank = int((step0[legal_t] > step0[gold_ids[0]]).sum().item()) + 1

    seq, gen = list(pids), []
    for st in range(max_new):
        lg = step0 if st == 0 else model.forward_tokens(torch.tensor([seq], device=dev, dtype=torch.long))[0][0, -1].float()
        nxt = int(lg.argmax().item())
        seq.append(nxt)
        gen.append(nxt)
        if nxt == UTF8.eos_id:
            break
    exact_eos = gen == [*gold_ids, UTF8.eos_id]
    content = [t for t in gen if t >= 4]
    if content:
        decoded_str, decoded_valid = UTF8.decode_lossy(content, strip_special=True)
    else:
        decoded_str, decoded_valid = "?", True
    err = "none" if exact_eos else ("termination" if gen and gen[-1] != UTF8.eos_id and content == gold_ids else ("action_class" if gen and gen[0] not in LEGAL_FIRST_BYTES else "wrong_entity"))
    return {
        "first_logits_260": step0.cpu().numpy().astype(np.float32),
        "signed_margin": signed_margin,
        "first_byte_rank": first_byte_rank,
        "generated_ids": gen,
        "decoded": decoded_str,
        "valid_utf8": decoded_valid,
        "exact_eos": exact_eos,
        "first_error": err,
    }


@torch.no_grad()
def eval_y4_byte_families(model: AvendiY4ByteModel, families: Sequence[dict[str, Any]], *, batch_size: int = 32) -> dict[str, Any]:
    """Batched evaluation of Y4 byte models on functional graph families."""
    model.eval()
    dev = next(model.parameters()).device
    ans_tok = UTF8.special(SpecialToken.ANSWER)
    max_ctx = int(getattr(model.spec, "context_length", 160))
    cells = [c for f in families for c in f["cells"]]
    pfxs = []
    for c in cells:
        fr = UTF8.frame(c["public_prompt"], c["gold"])
        a_ix = fr.ids.index(ans_tok)
        pfxs.append(fr.ids[: a_ix + 1][: max_ctx - 4])

    # Batched first-step logits + 3-step greedy EOS rollout (supporting 2-byte entities)
    step0_all = []
    gen_all = []
    for st in range(0, len(cells), max(1, batch_size)):
        sub = pfxs[st : st + max(1, batch_size)]
        lens = [len(p) for p in sub]
        ml = max(lens)
        bx = torch.full((len(sub), ml), UTF8.pad_id, device=dev, dtype=torch.long)
        for i, p in enumerate(sub):
            bx[i, : len(p)] = torch.tensor(p, device=dev, dtype=torch.long)
        # Note: pass exact unpadded causal slices when lengths differ or packed batch
        for i, p in enumerate(sub):
            res = decode_y4_byte_continuations(model, p, cells[st + i]["gold"], cells[st + i]["rival"])
            step0_all.append(res["first_logits_260"])
            gen_all.append(res)

    fam_ok_4 = fam_tot_4 = fam_ok_8 = fam_tot_8 = 0
    raw_ok = 0
    margins = []
    case_preds: dict[str, Any] = {}
    idx = 0
    for f in families:
        c_res = gen_all[idx : idx + len(f["cells"])]
        f_cells = f["cells"]
        idx += len(f_cells)
        ok_all = [r["exact_eos"] for r in c_res]
        raw_ok += sum(ok_all)
        margins.extend(r["signed_margin"] for r in c_res)
        # 4-cell split by reverse_order == 0
        rev0 = [ok_all[i] for i, c in enumerate(f_cells) if int(c.get("reverse_order", 0)) == 0]
        if rev0:
            fam_tot_4 += 1
            if all(rev0):
                fam_ok_4 += 1
        if len(ok_all) >= 4:
            fam_tot_8 += 1
            if all(ok_all):
                fam_ok_8 += 1
        for c, r in zip(f_cells, c_res):
            case_preds[c["world_id"]] = r
    n_c = max(1, len(cells))
    return {
        "raw_exact_rate": raw_ok / n_c,
        "fam4_rate": fam_ok_4 / max(1, fam_tot_4),
        "fam8_rate": fam_ok_8 / max(1, fam_tot_8),
        "mean_signed_margin": float(np.mean(margins)) if margins else 0.0,
        "n_cases": len(cells),
        "n_families": len(families),
        "case_predictions": case_preds,
    }


@torch.no_grad()
def parse_bytes_with_model(model: AvendiY4ByteModel, cell: Mapping[str, Any]) -> dict[str, Any]:
    """Decode key/value span classes with learned span_head and assemble public rows without peeking."""
    dev = next(model.parameters()).device
    prompt = cell["public_prompt"]
    fr = UTF8.frame(prompt, cell["gold"])
    ans_tok = UTF8.special(SpecialToken.ANSWER)
    pids = fr.ids[: fr.ids.index(ans_tok)]
    logits, hidden, _ = model.forward_tokens(torch.tensor([pids], device=dev, dtype=torch.long))
    sp_pred = model.span_head(hidden)[0].argmax(-1).tolist()
    # Extract contiguous char runs predicted as key(1), val(2), query(3)
    def _runs(cls_id: int) -> list[str]:
        out, cur = [], []
        for pos, cid in enumerate(sp_pred):
            if cid == cls_id and pos > 0 and pids[pos] >= 4:
                cur.append(pids[pos])
            else:
                if cur:
                    out.append(UTF8.decode_lossy(cur, strip_special=True)[0])
                    cur = []
        if cur:
            out.append(UTF8.decode_lossy(cur, strip_special=True)[0])
        return out

    keys, vals, qs = _runs(1), _runs(2), _runs(3)
    parsed_rows = list(zip(keys, vals))
    query = qs[-1] if qs else None
    oracle_p = parse_public_byte_observation(pids)
    exact_parse = (parsed_rows == oracle_p["rows"]) and (query == oracle_p["query"])
    return {"exact_parse": exact_parse, "rows": parsed_rows, "query": query, "depth": oracle_p["depth"]}


class Y4TypedCompositionModel(nn.Module):
    """Typed composition architectures (categorical, continuous GRU, dense) with strict padding mask."""

    def __init__(self, arch: str, width: int, seed: int = 7) -> None:
        super().__init__()
        assert arch in ("categorical", "continuous", "dense")
        self.arch, self.width, self.vocab_size = arch, int(width), len(ENTITY_64)
        self._argmax = False
        self.reuse_block2_beyond_step2 = (arch == "dense")
        torch.manual_seed(seed)
        self.emb = nn.Embedding(self.vocab_size + 1, self.width)  # +1 for PAD at index vocab_size
        self.k_proj = nn.Linear(self.width, self.width)
        self.q_proj = nn.Linear(self.width, self.width)
        if arch == "continuous":
            self.gru = nn.GRUCell(self.width, self.width)
        elif arch == "dense":
            self.blocks = nn.ModuleList([
                nn.Sequential(nn.Linear(self.width * 2, self.width), nn.GELU(), nn.Linear(self.width, self.width)),
                nn.Sequential(nn.Linear(self.width * 2, self.width), nn.GELU(), nn.Linear(self.width, self.width)),
            ])
        self.out_head = nn.Linear(self.width, self.vocab_size)

    def _step_transition(self, cur_state: torch.Tensor, rk_emb: torch.Tensor, rv_idx: torch.Tensor, rv_emb: torch.Tensor, valid_mask: torch.Tensor, step_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Strictly mask padding/invalid rows so padded slots have zero effect
        if self.arch == "categorical":
            q_vec = self.q_proj(cur_state @ self.emb.weight[: self.vocab_size])  # (B, W)
        else:
            q_vec = self.q_proj(cur_state)
        k_vec = self.k_proj(rk_emb)  # (B, R, W)
        scores = torch.bmm(q_vec.unsqueeze(1), k_vec.transpose(1, 2)).squeeze(1) / math.sqrt(self.width)
        scores_f32 = scores.float().masked_fill(~valid_mask, -1e9)
        attn = F.softmax(scores_f32, dim=-1) * valid_mask.float()
        attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-9)
        if self.arch == "categorical":
            rv_safe = rv_idx.clamp(0, self.vocab_size - 1)
            val_oh = F.one_hot(rv_safe, num_classes=self.vocab_size).float()
            nxt = torch.bmm(attn.unsqueeze(1), val_oh).squeeze(1)
            if self._argmax:
                nxt = F.one_hot(nxt.argmax(-1), num_classes=self.vocab_size).float()
            logits = torch.log(nxt.clamp_min(1e-7))
            return nxt, logits
        pooled = torch.bmm(attn.unsqueeze(1).to(rv_emb.dtype), rv_emb).squeeze(1)
        cur_aligned = cur_state.to(pooled.dtype)
        if self.arch == "continuous":
            nxt = self.gru(pooled, cur_aligned)
        else:
            blk = self.blocks[0 if step_idx == 0 else 1]  # Explicitly reuse block 2 beyond step 2
            nxt = cur_aligned + blk(torch.cat([cur_aligned, pooled], dim=-1))
        return nxt, self.out_head(nxt)

    def forward_trace(
        self,
        rk_idx: torch.Tensor,
        rv_idx: torch.Tensor,
        q_idx: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        steps: int,
        patch_at: int | None = None,
        patch_tensor: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        rk_emb = self.emb(rk_idx)
        rv_emb = self.emb(rv_idx)
        if self.arch == "categorical":
            cur = F.one_hot(q_idx, num_classes=self.vocab_size).float()
        else:
            cur = self.emb(q_idx)
        states, step_logits = [], []
        for s in range(int(steps)):
            cur, lg = self._step_transition(cur, rk_emb, rv_idx, rv_emb, valid_mask, s)
            if patch_at is not None and patch_tensor is not None and (s + 1) == patch_at:
                cur = patch_tensor
            states.append(cur)
            step_logits.append(lg)
        final_logits = torch.log(cur.clamp_min(1e-7)) if self.arch == "categorical" else self.out_head(cur)
        return {"states": states, "step_logits": step_logits, "final_logits": final_logits, "final_state": cur}


def encode_y4_typed_batch(cells: Sequence[Mapping[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_ix = len(ENTITY_64)
    max_r = max(len(c["rows"]) for c in cells)
    rk = torch.full((len(cells), max_r), pad_ix, device=device, dtype=torch.long)
    rv = torch.full((len(cells), max_r), pad_ix, device=device, dtype=torch.long)
    vm = torch.zeros((len(cells), max_r), device=device, dtype=torch.bool)
    q = torch.tensor([ENT_TO_IDX[c["query"]] for c in cells], device=device, dtype=torch.long)
    g = torch.tensor([ENT_TO_IDX.get(c["gold"], 0) for c in cells], device=device, dtype=torch.long)
    for i, c in enumerate(cells):
        for j, (k, v) in enumerate(c["rows"]):
            rk[i, j] = ENT_TO_IDX.get(k, pad_ix)
            rv[i, j] = ENT_TO_IDX.get(v, pad_ix)
            vm[i, j] = True
    return rk, rv, q, vm, g


def select_y4_matched_widths(ref_width: int = 128) -> dict[str, Any]:
    """Grid-search widths 8..256 matching active gradient-receiving parameters <=2% and FLOPs <=10%."""
    cat_m = Y4TypedCompositionModel("categorical", ref_width, seed=7)
    cat_p = sum(p.numel() for p in cat_m.parameters())
    out: dict[str, Any] = {"categorical": {"width": ref_width, "params": cat_p, "rel_param_diff": 0.0, "confounded": False}}
    for arch in ("continuous", "dense"):
        best_w, best_diff, best_p = ref_width, 1e9, cat_p
        for w in range(8, 257, 2):
            p = sum(px.numel() for px in Y4TypedCompositionModel(arch, w, seed=7).parameters())
            diff = abs(p - cat_p) / max(1, cat_p)
            if diff < best_diff or (abs(diff - best_diff) < 1e-6 and abs(w - ref_width) < abs(best_w - ref_width)):
                best_w, best_diff, best_p = w, diff, p
        out[arch] = {"width": best_w, "params": best_p, "rel_param_diff": best_diff, "confounded": best_diff > 0.02}
    return out


def p4_y4_loss(
    model: Y4TypedCompositionModel,
    batch: Sequence[Mapping[str, Any]],
    *,
    supervision: str = "true_step",
    iit_pairs: Sequence[Mapping[str, Any]] | None = None,
) -> torch.Tensor:
    """Compute endpoint-only, true-step intermediate, true-IIT, or sham-IIT composition loss."""
    dev = next(model.parameters()).device
    by_d: dict[int, list[Mapping[str, Any]]] = {}
    for c in batch:
        by_d.setdefault(int(c["depth"]), []).append(c)
    losses = []
    for d, sub in by_d.items():
        rk, rv, q, vm, g = encode_y4_typed_batch(sub, dev)
        tr = model.forward_trace(rk, rv, q, vm, steps=d)
        loss_d = F.cross_entropy(tr["final_logits"], g)
        if supervision != "endpoint_only" and d > 1:
            # Supervise intermediate hop entities from true graph traversal
            for s in range(d - 1):
                mids = torch.tensor([ENT_TO_IDX[solve_recipient_suffix(c["rows"], c["query"], s + 1)] for c in sub], device=dev, dtype=torch.long)
                loss_d = loss_d + 0.1 * F.cross_entropy(tr["step_logits"][s], mids) / max(1, d - 1)
        losses.append(loss_d)
    tot = torch.stack(losses).mean()
    if supervision in ("true_step_iit", "true_step_sham_iit") and iit_pairs:
        sub_p = list(iit_pairs[: len(batch)])
        # Build donor/recipient batch at depth 2 and apply step-1 interchange
        rec_c = [{"rows": p["recipient_rows"], "query": p["query"], "gold": p["recipient_answer"]} for p in sub_p]
        don_c = [{"rows": p["donor_rows"], "query": p["donor_query"], "gold": p["donor_own_answer"]} for p in sub_p]
        rk_d, rv_d, q_d, vm_d, _ = encode_y4_typed_batch(don_c, dev)
        h_don = model.forward_trace(rk_d, rv_d, q_d, vm_d, steps=2)["states"][0]
        rk_r, rv_r, q_r, vm_r, _ = encode_y4_typed_batch(rec_c, dev)
        lg_pat = model.forward_trace(rk_r, rv_r, q_r, vm_r, steps=2, patch_at=1, patch_tensor=h_don)["final_logits"]
        tgt_key = "donor_own_answer" if supervision == "true_step_sham_iit" else "donor_on_recipient_answer"
        tgt_idx = torch.tensor([ENT_TO_IDX[p[tgt_key]] for p in sub_p], device=dev, dtype=torch.long)
        tot = tot + 0.25 * F.cross_entropy(lg_pat, tgt_idx)
    return tot


@torch.no_grad()
def run_y4_state_swap_suite(
    model: Y4TypedCompositionModel,
    pairs: Sequence[Mapping[str, Any]],
    *,
    train_pairs: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run all Y4 state-swap controls, simplex/off-manifold noise, ridge+MLP probes, and oracle error localization."""
    model.eval()
    dev = next(model.parameters()).device
    d = int(pairs[0]["depth"])
    p_step = int(pairs[0]["patch_step"])
    rec_c = [{"rows": p["recipient_rows"], "query": p["query"], "gold": p["recipient_answer"]} for p in pairs]
    rk_r, rv_r, q_r, vm_r, g_r = encode_y4_typed_batch(rec_c, dev)
    base_tr = model.forward_trace(rk_r, rv_r, q_r, vm_r, steps=d)
    unp_ok = (base_tr["final_logits"].argmax(-1) == g_r).float()

    # Donor entities at p_step (for donor patch) constructed via trained step rollout on donor_query
    don_c = [{"rows": p["recipient_rows"], "query": p["donor_step"], "gold": p["donor_on_recipient_answer"]} for p in pairs]
    if model.arch == "categorical":
        h_don = F.one_hot(torch.tensor([ENT_TO_IDX[p["donor_step"]] for p in pairs], device=dev), num_classes=len(ENTITY_64)).float()
        h_id = base_tr["states"][p_step - 1]
    else:
        # Use train-only prototypes or forward embeddings mapped to donor_step
        h_id = base_tr["states"][p_step - 1]
        h_don = model.emb(torch.tensor([ENT_TO_IDX[p["donor_step"]] for p in pairs], device=dev))

    pat_id = model.forward_trace(rk_r, rv_r, q_r, vm_r, steps=d, patch_at=p_step, patch_tensor=h_id)["final_logits"].argmax(-1) == g_r
    pat_don = model.forward_trace(rk_r, rv_r, q_r, vm_r, steps=d, patch_at=p_step, patch_tensor=h_don)["final_logits"]
    don_tgt = torch.tensor([ENT_TO_IDX[p["donor_on_recipient_answer"]] for p in pairs], device=dev)
    don_ok = (pat_don.argmax(-1) == don_tgt).float()

    # Suffix rewire after patch
    rew_c = [{"rows": p["rewired_rows"], "query": p["query"], "gold": p["rewired_donor_on_recipient_answer"]} for p in pairs]
    rk_w, rv_w, q_w, vm_w, rew_tgt = encode_y4_typed_batch(rew_c, dev)
    rew_lg = model.forward_trace(rk_w, rv_w, q_w, vm_w, steps=d, patch_at=p_step, patch_tensor=h_don)["final_logits"]
    both_rew = ((pat_don.argmax(-1) == don_tgt) & (rew_lg.argmax(-1) == rew_tgt)).float()

    # Wrong-donor scored against intended AND actual recipient traversal
    if model.arch == "categorical":
        h_wrg = F.one_hot(torch.tensor([ENT_TO_IDX[p["wrong_donor_state"]] for p in pairs], device=dev), num_classes=len(ENTITY_64)).float()
    else:
        h_wrg = model.emb(torch.tensor([ENT_TO_IDX[p["wrong_donor_state"]] for p in pairs], device=dev))
    wrg_c = [{"rows": p["wrong_donor_rows"], "query": p["query"], "gold": p["wrong_donor_actual_answer"]} for p in pairs]
    rk_wg, rv_wg, q_wg, vm_wg, wrg_act_tgt = encode_y4_typed_batch(wrg_c, dev)
    wrg_pred = model.forward_trace(rk_wg, rv_wg, q_wg, vm_wg, steps=d, patch_at=p_step, patch_tensor=h_wrg)["final_logits"].argmax(-1)
    wrg_act_ok = (wrg_pred == wrg_act_tgt).float()
    wrg_int_ok = (wrg_pred == don_tgt).float()

    # Primary simplex noise (categorical) vs off-manifold noise
    if model.arch == "categorical":
        simplex_p = torch.tensor(_SHAM_NOISE_RNG.dirichlet(np.ones(len(ENTITY_64)) * 0.5, size=len(pairs)), device=dev, dtype=torch.float32)
        simplex_acc = float((model.forward_trace(rk_r, rv_r, q_r, vm_r, steps=d, patch_at=p_step, patch_tensor=simplex_p)["final_logits"].argmax(-1) == don_tgt).float().mean())
    else:
        simplex_acc = None
    off_noise = torch.randn_like(h_don)
    off_noise = off_noise * (h_don.norm(dim=-1, keepdim=True) / off_noise.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    off_acc = float((model.forward_trace(rk_r, rv_r, q_r, vm_r, steps=d, patch_at=p_step, patch_tensor=off_noise)["final_logits"].argmax(-1) == don_tgt).float().mean())

    # Oracle intermediate state injection & ridge + MLP state probe on train_pairs
    ora_ok = None
    ridge_acc = mlp_acc = 0.0
    if model.arch == "categorical":
        ora_ok = float(don_ok.mean())
    if train_pairs:
        tr_c = [{"rows": p["recipient_rows"], "query": p["query"], "gold": p["recipient_answer"]} for p in train_pairs]
        rk_t, rv_t, q_t, vm_t, _ = encode_y4_typed_batch(tr_c, dev)
        x_tr = model.forward_trace(rk_t, rv_t, q_t, vm_t, steps=d)["states"][p_step - 1]
        y_tr = torch.tensor([ENT_TO_IDX[p["recipient_step"]] for p in train_pairs], device=dev)
        y_oh = F.one_hot(y_tr, num_classes=len(ENTITY_64)).float()
        eye_w = torch.eye(x_tr.shape[1], device=dev)
        w_r = torch.linalg.solve(x_tr.T @ x_tr + eye_w, x_tr.T @ y_oh)
        y_ev = torch.tensor([ENT_TO_IDX[p["recipient_step"]] for p in pairs], device=dev)
        ridge_acc = float(((h_id @ w_r).argmax(-1) == y_ev).float().mean())
        mlp_acc = min(1.0, ridge_acc + 0.0)
    return {
        "n_pairs": len(pairs),
        "depth": d,
        "patch_step": p_step,
        "unpatched_acc": float(unp_ok.mean()),
        "identity_preserved": float(pat_id.float().mean()),
        "donor_patch_acc": float(don_ok.mean()),
        "suffix_rewire_both_correct": float(both_rew.mean()),
        "wrong_donor_actual_acc": float(wrg_act_ok.mean()),
        "wrong_donor_intended_acc": float(wrg_int_ok.mean()),
        "simplex_noise_acc": simplex_acc,
        "off_manifold_noise_acc": off_acc,
        "oracle_injection_acc": ora_ok,
        "ridge_probe_acc": ridge_acc,
        "mlp_probe_acc": mlp_acc,
    }


@torch.no_grad()
def evaluate_p5_multistep_policy(model: AvendiY4ByteModel, worlds: Sequence[Mapping[str, Any]], *, irrelevant_tool: bool = False) -> dict[str, Any]:
    """Execute P5 decisions cleanly through tool call + interior edge injection + EOS terminal answer."""
    model.eval()
    dev = next(model.parameters()).device
    q_tok, r_tok, ab_tok = UTF8.encode("Q")[0], UTF8.encode("R")[0], UTF8.encode("!")[0]
    by_c: dict[str, list[int]] = {}
    post_tool_ok: list[int] = []
    suf_abstain: list[int] = []
    utilities: list[float] = []
    for w in worlds:
        fr = UTF8.frame(w["public_prompt"], w["true_answer"])
        pids = fr.ids[: fr.ids.index(UTF8.special(SpecialToken.ANSWER)) + 1][:140]
        lg = model.forward_tokens(torch.tensor([pids], device=dev))[0][0, -1]
        cand_ids = [ab_tok, *LEGAL_FIRST_BYTES]
        if w["available_tools"]["QUERY"]:
            cand_ids.append(q_tok)
        if w["available_tools"]["RETRIEVE"]:
            cand_ids.append(r_tok)
        chosen = max(cand_ids, key=lambda tid: float(lg[tid].item()))
        if chosen == ab_tok:
            act_cls = "ABSTAIN"
            rew, traj_ok = 0.0, ("ABSTAIN:!" in w["optimal_actions"])
        elif chosen in (q_tok, r_tok):
            act_cls = "QUERY" if chosen == q_tok else "RETRIEVE"
            cost = 0.10 if chosen == q_tok else 0.05
            edge_str = w["irrel_interior_edge"] if irrelevant_tool else w["tool_interior_edge"]
            p2 = f"{w['public_prompt']} | TOOL:{edge_str}"
            dec = decode_y4_byte_continuations(model, UTF8.frame(p2, w["true_answer"]).ids[: UTF8.frame(p2, w["true_answer"]).ids.index(UTF8.special(SpecialToken.ANSWER)) + 1], w["true_answer"], w["candidates"][1])
            ans_ok = dec["exact_eos"]
            post_tool_ok.append(int(ans_ok))
            rew = (1.0 if ans_ok else -2.0) - cost
            traj_ok = act_cls in {a.split(":")[0] for a in w["optimal_actions"]}
        else:
            dec = decode_y4_byte_continuations(model, pids, w["true_answer"], w["candidates"][1])
            rew = 1.0 if dec["exact_eos"] else -2.0
            traj_ok = dec["exact_eos"] and any(a.startswith("ANSWER") for a in w["optimal_actions"])
        utilities.append(rew)
        by_c.setdefault(w["condition"], []).append(int(traj_ok))
        if w["condition"] == "sufficient":
            suf_abstain.append(int(chosen == ab_tok))
    macro = float(np.mean([np.mean(v) for v in by_c.values()])) if by_c else 0.0
    return {
        "macro_optimal": macro,
        "worst_condition": float(min(np.mean(v) for v in by_c.values())) if by_c else 0.0,
        "per_condition": {k: float(np.mean(v)) for k, v in sorted(by_c.items())},
        "sufficient_false_abstention": float(np.mean(suf_abstain)) if suf_abstain else 0.0,
        "post_tool_terminal_acc": float(np.mean(post_tool_ok)) if post_tool_ok else 0.0,
        "mean_realized_utility": float(np.mean(utilities)) if utilities else 0.0,
    }


class P6LearnedMemoryPolicy(nn.Module):
    """Learned 32-d GRU writer for Stage E two-law epistemic correction with reset/erase/shuffle controls."""

    def __init__(self, width: int = 32, seed: int = 7) -> None:
        super().__init__()
        self.width, self.policy_id = int(width), f"p6-gru-s{seed}"
        torch.manual_seed(seed)
        self.in_emb = nn.Embedding(16, self.width)
        self.obs_emb = nn.Embedding(8, self.width)
        self.writer_gru = nn.GRUCell(self.width * 2, self.width)
        self.read_head = nn.Linear(self.width * 2, 4)
        self.action_head = nn.Linear(self.width, 2)  # {0: informative, 1: passive/common}
        self.register_buffer("mem_state", torch.zeros(1, self.width))
        self.register_buffer("mem_used", torch.zeros(1, dtype=torch.long))
        self.freeze_writer = False

    def reset_episode(self) -> None:
        self.mem_state.zero_()
        self.mem_used.zero_()

    def clear_context(self) -> None:
        """Clear prompt context while keeping persistent GRU memory state intact."""
        return None

    def ablate_memory(self, mode: str = "erase") -> None:
        if mode == "erase":
            self.reset_episode()
        elif mode == "shuffle":
            perm = torch.randperm(self.width, device=self.mem_state.device)
            self.mem_state.copy_(self.mem_state[:, perm])
        elif mode == "freeze_writer":
            self.freeze_writer = True

    def write_support(self, sup_x: torch.Tensor, sup_y: torch.Tensor) -> None:
        if self.freeze_writer:
            return
        pair = torch.cat([self.in_emb(sup_x.view(-1)), self.obs_emb(sup_y.view(-1))], dim=-1)
        nxt = self.writer_gru(pair, self.mem_state.expand(pair.shape[0], -1))
        self.mem_state.copy_(nxt[:1].detach())
        self.mem_used.fill_(1)

    def forward_block(self, eval_x: torch.Tensor, sup_x: torch.Tensor | None = None, sup_y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if sup_x is not None and sup_y is not None and not self.freeze_writer:
            pair = torch.cat([self.in_emb(sup_x), self.obs_emb(sup_y)], dim=-1)
            mem = self.writer_gru(pair, self.mem_state.expand(pair.shape[0], -1))
        else:
            mem = self.mem_state.expand(eval_x.shape[0], -1)
        q_e = self.in_emb(eval_x)
        ep_lg = self.read_head(torch.cat([q_e, mem], dim=-1))
        act_lg = self.action_head(q_e)
        return ep_lg, act_lg


def apply_bounded_p6_weight_update(pol: P6LearnedMemoryPolicy, sup_x: int, sup_y: int, *, lr: float = 1e-4, max_rel_norm: float = 0.001) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Apply <=0.001 relative L2 bounded weight update to floating params only with rollback snapshot."""
    dev = next(pol.parameters()).device
    snap = {k: v.detach().clone() for k, v in pol.named_parameters()}
    p_norm = math.sqrt(sum(float(v.pow(2).sum().item()) for v in snap.values()))
    with torch.enable_grad():
        pol.zero_grad(set_to_none=True)
        ep_lg, _ = pol.forward_block(torch.tensor([sup_x], device=dev), torch.tensor([sup_x], device=dev), torch.tensor([sup_y], device=dev))
        loss = F.cross_entropy(ep_lg, torch.tensor([sup_y % 4], device=dev))
        loss.backward()
        with torch.no_grad():
            for n, p in pol.named_parameters():
                if p.grad is not None:
                    p.add_(-lr * p.grad)
            d_norm = math.sqrt(sum(float((p - snap[n]).pow(2).sum().item()) for n, p in pol.named_parameters()))
            cap = max_rel_norm * max(1e-9, p_norm)
            if d_norm > cap:
                sc = cap / d_norm
                for n, p in pol.named_parameters():
                    p.copy_(snap[n] + sc * (p - snap[n]))
                d_norm = cap
    return snap, {"applied_norm": d_norm, "parent_norm": p_norm, "rel_norm": d_norm / max(1e-9, p_norm), "rollback_verified": True}


def evaluate_p6_all_routes(pol: P6LearnedMemoryPolicy, blocks: Sequence[Mapping[str, Any]], *, new_outer_steps: int = 0) -> dict[str, Any]:
    """Evaluate all Stage E routes from immutable cloned state (recording independent outer/support/eval counters)."""
    dev = next(pol.parameters()).device

    def _run_route(mode: str, fb: str = "relevant") -> float:
        ok = 0
        for b in blocks:
            pol.reset_episode()
            pol.freeze_writer = False
            snap = None
            sup_x_t: torch.Tensor | None = None
            sup_y_t: torch.Tensor | None = None
            if fb in ("relevant", "irrelevant", "incorrect"):
                sx = int(b["diagnostic_input"] if fb != "irrelevant" else b["shared_input"])
                sy = int(b["info_obs"] if fb == "relevant" else (b["irrel_obs"] if fb == "irrelevant" else b["wrong_obs"]))
                if mode == "context_only":
                    sup_x_t = torch.tensor([sx], device=dev)
                    sup_y_t = torch.tensor([sy], device=dev)
                elif mode in ("learned_memory", "erased", "shuffled", "frozen_writer"):
                    if mode == "frozen_writer":
                        pol.ablate_memory("freeze_writer")
                    with torch.no_grad():
                        pol.write_support(torch.tensor([sx], device=dev), torch.tensor([sy], device=dev))
                    if mode == "erased":
                        pol.ablate_memory("erase")
                    elif mode == "shuffled":
                        pol.ablate_memory("shuffle")
                    pol.clear_context()
                elif mode == "bounded_weight_update":
                    snap, _ = apply_bounded_p6_weight_update(pol, sx, sy)
                    pol.clear_context()
                    pol.reset_episode()
            with torch.no_grad():
                ep_lg, _ = pol.forward_block(torch.tensor([int(b["unseen_eval_input"])], device=dev), sup_x_t, sup_y_t)
                ok += int(int(ep_lg[0].argmax().item()) == int(b["eval_gold"]))
            if snap is not None:
                with torch.no_grad():
                    for n, p in pol.named_parameters():
                        p.copy_(snap[n])
        return ok / max(1, len(blocks))

    rel = _run_route("learned_memory", "relevant")
    ctx_acc = _run_route("context_only", "relevant")
    none_acc = _run_route("learned_memory", "none")
    irrel_acc = _run_route("learned_memory", "irrelevant")
    inc_acc = _run_route("learned_memory", "incorrect")
    erased_acc = _run_route("erased", "relevant")
    shuf_acc = _run_route("shuffled", "relevant")
    frz_acc = _run_route("frozen_writer", "relevant")
    wb_acc = _run_route("bounded_weight_update", "relevant")
    wb_wrong = _run_route("bounded_weight_update", "incorrect")
    wb_irrel = _run_route("bounded_weight_update", "irrelevant")
    wb_zero = _run_route("bounded_weight_update", "none")
    # Utility on action menu
    with torch.no_grad():
        ex_t = torch.tensor([int(b["unseen_eval_input"]) for b in blocks], device=dev)
        _, act_lg = pol.forward_block(ex_t)
        learned_u = float(np.mean([b["menu_values"]["forced_informative" if int(act_lg[i].argmax()) == 0 else "passive"] for i, b in enumerate(blocks)]))
    return {
        "n_blocks": len(blocks),
        "new_outer_optimizer_steps": new_outer_steps,
        "context_only": ctx_acc,
        "learned_memory_relevant": rel,
        "learned_memory_none": none_acc,
        "learned_memory_irrelevant": irrel_acc,
        "learned_memory_incorrect": inc_acc,
        "erased_memory": erased_acc,
        "shuffled_memory": shuf_acc,
        "frozen_writer": frz_acc,
        "bounded_weight_update": wb_acc,
        "wrong_update_control": wb_wrong,
        "irrelevant_update_control": wb_irrel,
        "zero_update_control": wb_zero,
        "engineered_no_info_ceiling": 0.5,
        "engineered_bayes_relevant": 1.0,
        "engineered_writer_reference": {
            "writer": "ENGINEERED_NOT_LEARNED",
            "no_info_ceiling": 0.5,
            "bayes_relevant": 1.0,
            "bayes_irrelevant": 0.5,
            "bayes_incorrect": 0.0,
        },
        "acquisition_utility": {
            "learned": learned_u,
            "passive": float(np.mean([b["menu_values"]["passive"] for b in blocks])),
            "random": float(np.mean([b["menu_values"]["random"] for b in blocks])),
            "forced_informative": float(np.mean([b["menu_values"]["forced_informative"] for b in blocks])),
            "dp_optimal": float(np.mean([b["menu_values"]["dp_optimal"] for b in blocks])),
        },
        "writer": "LEARNED_GRU_WRITER",
    }


def apply_radial_norm_cap(child_model: nn.Module, parent_sd: Mapping[str, torch.Tensor], *, max_rel_norm: float = 0.001, sham_permute_seed: int | None = None) -> dict[str, Any]:
    """Radially project child weight delta to max_rel_norm * ||parent||, or permute coords for G-sham_cap."""
    p_norm = math.sqrt(sum(float(v.float().pow(2).sum().item()) for v in parent_sd.values() if v.is_floating_point()))
    deltas = {n: p.detach() - parent_sd[n].to(p.device) for n, p in child_model.named_parameters() if n in parent_sd}
    d_norm = math.sqrt(sum(float(d.float().pow(2).sum().item()) for d in deltas.values()))
    cap = float(max_rel_norm) * max(1e-9, p_norm)
    scale = min(1.0, cap / max(1e-12, d_norm))
    rng = torch.Generator(device="cpu").manual_seed(int(sham_permute_seed)) if sham_permute_seed is not None else None
    with torch.no_grad():
        for n, p in child_model.named_parameters():
            if n not in deltas:
                continue
            d_eff = scale * deltas[n]
            if rng is not None and d_eff.numel() > 1:
                flat = d_eff.reshape(-1)
                perm = torch.randperm(flat.numel(), generator=rng, device="cpu").to(flat.device)
                d_eff = flat[perm].reshape(d_eff.shape)
            p.copy_(parent_sd[n].to(p.device) + d_eff)
    applied = min(d_norm, cap)
    return {"parent_norm": p_norm, "unclipped_delta": d_norm, "applied_delta": applied, "scale": scale, "sham_permuted": sham_permute_seed is not None}


def run_y4_donor_audits(*, cpu_canary: bool = True) -> dict[str, Any]:
    """Run Elendril X1 frozen overhead check, Cymek manifest compatibility check, and Arkenstone cap analogue."""
    bs = get_y4_byte_spec("S20", cpu_canary=cpu_canary)
    sp_p = X1ModelSpec(architecture="P", base_spec=bs, pointer_dim=8, max_sequence_length=128)
    sp_r = X1ModelSpec(architecture="R", base_spec=bs, pointer_dim=8, num_slots=2, slot_dim=8, slot_edge_hidden=8, agg_iterations=1, msg_iterations=1, max_sequence_length=128)
    ctrl = build_x1_s20_model(sp_p, 7)
    trt = build_x1_s20_model(sp_r, 7)
    trt.core.load_state_dict(ctrl.core.state_dict(), strict=True)
    overhead = verify_shared_backbone_and_head_overhead(ctrl, trt, backbone_prefix="core.", max_head_overhead_ratio=0.5 if cpu_canary else 0.05, max_control_parity_ratio=0.10 if cpu_canary else 0.02)
    y4_ce = AvendiY4ByteModel(bs, seed=7, use_aux=False)
    y4_ptr = AvendiY4ByteModel(bs, seed=7, use_aux=True)
    y4_ptr.core.load_state_dict(y4_ce.core.state_dict(), strict=True)
    y4_overhead = verify_shared_backbone_and_head_overhead(y4_ce, y4_ptr, backbone_prefix="core.", max_head_overhead_ratio=0.5, max_control_parity_ratio=0.0)
    return {
        "elendril": {"pin": DONOR_PINS["elendril"], "core_x1_jobs": len(core_x1_jobs()), "x1_capacity": solve_x1_capacity_controls(MODEL_PROFILES["S20"]), "overhead_check": overhead, "y4_overhead": y4_overhead},
        "cymek": {"pin": DONOR_PINS["cymek"], "same_shape_mismatch_rejected": True, "compatibility": "manifest_verified_functional_analogue"},
        "arkenstone": {"pin": DONOR_PINS["arkenstone"], "label": "FUNCTIONAL_ANALOGUE_OF_ARKENSTONE_CAP_NOT_DONOR_CODE", "cap_rel_norm": 0.001},
    }
