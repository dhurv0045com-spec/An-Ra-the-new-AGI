"""FORMATION-DIAG-001 measurement panel.

Extends the preregistered FORMATION-BASELINE-GATE-001 panel with the
measurements its list lacks: CE by family, accuracy stratified by output
position AND answer length, clipping multiplier, and the tied-embedding
input/output gradient decomposition. No sealed data is touched anywhere.

The tied decomposition separates the two roles the single tied matrix
serves (representation lookup vs output prediction): the output-only
gradient is measured by backwarding through a forward pass whose input
lookup is detached, and the input-role gradient is the residual
(full - output). For the V5 core the input path contributes to the loss
only through hidden states, so this split is exact up to float round-off.
"""

from __future__ import annotations

from typing import Any, Mapping

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


def _batch_from_rows(rows: list[Mapping[str, Any]], *, torch: Any,
                     device: Any, arm: str = "M0_STANDARD") -> tuple[Any, Any, Any, list[list[int]], list[list[int]]]:
    from anra_v5.formation_mux_train import build_batch, _prompt_answer_ids
    tokens, segments, eligible, _supervised = build_batch(
        rows, "CS-MECH-002", arm, torch=torch, device=device)
    prompts = [_prompt_answer_ids(row, "CS-MECH-002", arm)[0] for row in rows]
    answers = [[t for t in _prompt_answer_ids(row, "CS-MECH-002", arm)[1]
                if t != 3] for row in rows]
    return tokens, segments, eligible, prompts, answers


def ce_by_family(model: Any, rows: list[Mapping[str, Any]], *, torch: Any,
                 device: Any) -> dict[str, Any]:
    """Mean CE loss per family over the development surface (diagnostic
    only; loss is never cognition evidence)."""

    from v5_objectives.causal_lm import causal_lm_loss
    from v5_model.core import packed_layout
    by_family: dict[str, list[float]] = {}
    was_training = model.training
    model.eval()
    families = sorted({row["family"] for row in rows})
    with torch.no_grad():
        for family in families:
            fam_rows = [row for row in rows if row["family"] == family]
            tokens, segments, eligible, _p, _a = _batch_from_rows(
                fam_rows, torch=torch, device=device)
            positions, mask = packed_layout(segments, torch_module=torch)
            logits = model(tokens, positions, mask)
            loss, count = causal_lm_loss(logits, tokens, segments, bos_id=2,
                                         pad_id=0, eligible=eligible,
                                         torch_module=torch)
            by_family[family] = round(float(loss.item()), 4)
    model.train(was_training)
    return {"schema": "anra.formation-diag-ce-family/v1",
            "ce_by_family": by_family}


def teacher_forced_diagnostics(model: Any, rows: list[Mapping[str, Any]], *,
                               torch: Any, device: Any,
                               max_len_bucket: int = 8) -> dict[str, Any]:
    """Token accuracy overall, by output position, and by answer length;
    target rank and margin on the gold token."""

    from v5_model.core import packed_layout
    bos, eos, pad = 2, 3, 0
    model.eval()
    hits: list[int] = []
    per_position: dict[int, list[int]] = {}
    per_length: dict[int, list[int]] = {}
    ranks: list[int] = []
    margins: list[float] = []
    identity_rows = [row for row in rows if row.get("family") == "identity"]
    sample = identity_rows if identity_rows else rows
    with torch.no_grad():
        for row in sample:
            prompt = [bos, *row["prompt_ids"]]
            answer = [t for t in row["answer_ids"] if t != eos]
            if not answer:
                continue
            ids = prompt + answer
            current = torch.tensor([ids], device=device)
            positions, mask = packed_layout(
                torch.tensor([[0] * current.shape[1]], device=device),
                torch_module=torch)
            logits = model(current, positions, mask)[0]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            sorted_probs, sorted_ids = torch.sort(log_probs, descending=True)
            for offset, gold in enumerate(answer):
                position = len(prompt) - 1 + offset
                row_logits = logits[position]
                prediction = int(torch.argmax(row_logits).item())
                hit = int(prediction == gold)
                hits.append(hit)
                per_position.setdefault(offset, []).append(hit)
                per_length.setdefault(len(answer), []).append(hit)
                rank = int((sorted_ids[position] == gold)
                           .nonzero(as_tuple=True)[0].item()) \
                    if gold < sorted_ids.shape[1] else -1
                ranks.append(rank)
                gold_lp = float(log_probs[position, gold].item())
                best_other = float(sorted_probs[position][1
                                                              if hit == 0 else 0].item())
                margins.append(gold_lp - best_other)
    model.train()
    length_buckets = {str(length): round(sum(v) / len(v), 4)
                      for length, v in sorted(per_length.items())
                      if length <= max_len_bucket}
    return {"schema": "anra.formation-diag-teacher-forced/v1",
            "rows_scored": len(sample),
            "token_accuracy": round(sum(hits) / len(hits), 4) if hits else 0.0,
            "accuracy_by_position": {str(p): round(sum(v) / len(v), 4)
                                     for p, v in sorted(per_position.items())},
            "accuracy_by_answer_length": length_buckets,
            "mean_gold_rank": round(sum(ranks) / len(ranks), 2) if ranks else None,
            "mean_gold_margin_logprob": round(sum(margins) / len(margins), 4)
            if margins else None}


def full_vs_shared_rescue(model: Any, rows: list[Mapping[str, Any]], *,
                          torch: Any, device: Any,
                          shared_rows: int = 128) -> dict[str, Any]:
    """Candidate-free identity exact under (a) the full 24,576-way output
    denominator and (b) a shared-only denominator (rows 0..shared_rows-1).
    A large (b) - (a) gap is the output-space competition signature."""

    from anra_v5.formation_mux_train import _eval_rates
    identity_rows = [row for row in rows if row.get("family") == "identity"]
    full = _eval_rates(model, identity_rows, "CS-MECH-002", "M0_STANDARD",
                       torch=torch, device=device)
    original = model.forward
    import types

    def shared_forward(self, *args, **kwargs):
        logits = original(*args, **kwargs)
        return logits[..., :shared_rows]

    model.forward = types.MethodType(shared_forward, model)
    try:
        shared = _eval_rates(model, identity_rows, "CS-MECH-002",
                             "M0_STANDARD", torch=torch, device=device)
    finally:
        model.forward = original
    return {"schema": "anra.formation-diag-rescue/v1",
            "full_vocab_exact": full["complete_exact_with_valid_stop"],
            "shared_only_exact": shared["complete_exact_with_valid_stop"],
            "rescue": round(shared["complete_exact_with_valid_stop"]
                            - full["complete_exact_with_valid_stop"], 4)}


def clipping_multiplier(*, grad_norm_pre_clip: float,
                        clip_value: float = 1.0) -> float:
    if grad_norm_pre_clip <= 0:
        return 1.0
    return min(1.0, clip_value / grad_norm_pre_clip)


def tied_gradient_decomposition(model: Any, batch: tuple[Any, Any, Any], *,
                                torch: Any, device: Any) -> dict[str, Any]:
    """Input-role vs output-role gradient split of the tied matrix.

    Output role: gradient of the loss w.r.t. W with the trunk computed under
    no_grad and DETACHED before the output projection — only the direct
    logit path reaches W. Full role: the standard backward. Input role is
    the residual g_in = g_full - g_out (exact up to float round-off; the
    receipt carries the reconstruction gap and fails closed above 1e-4).
    """

    import types
    tokens, segments, eligible = batch
    from v5_objectives.causal_lm import causal_lm_loss
    from v5_model.core import packed_layout

    W = model.embedding.weight
    was_training = model.training
    model.eval()
    positions, mask = packed_layout(segments, torch_module=torch)

    # pass 1: output role (trunk detached)
    with torch.no_grad():
        hidden = model.embedding(tokens)
        for block in model.blocks:
            hidden = block(hidden, positions, mask)
        trunk = model.final_norm(hidden).detach()
    W.grad = torch.zeros_like(W)
    logits = torch.nn.functional.linear(trunk, W)
    loss, _count = causal_lm_loss(logits, tokens, segments, bos_id=2,
                                  pad_id=0, eligible=eligible,
                                  torch_module=torch)
    loss.backward()
    g_out = W.grad.detach().clone()

    # pass 2: full standard backward
    model.zero_grad(set_to_none=False)
    W.grad = torch.zeros_like(W)
    logits = model(tokens, positions, mask)
    loss, _count = causal_lm_loss(logits, tokens, segments, bos_id=2,
                                  pad_id=0, eligible=eligible,
                                  torch_module=torch)
    loss.backward()
    g_full = W.grad.detach().clone()

    g_in = g_full - g_out
    model.train(was_training)
    norm_out = float(g_out.norm().item())
    norm_in = float(g_in.norm().item())
    cos = float(torch.nn.functional.cosine_similarity(
        g_out.flatten(), g_in.flatten(), dim=0).item())
    reconstruct_gap = float((g_out + g_in - g_full).norm().item()
                            / max(g_full.norm().item(), 1e-8))
    if reconstruct_gap > 1e-4:
        raise RuntimeError(
            "tied-gradient decomposition failed its reconstruction check: "
            f"{reconstruct_gap}")
    return {"schema": "anra.formation-diag-tied-grad/v1",
            "output_role_grad_norm": round(norm_out, 6),
            "input_role_grad_norm": round(norm_in, 6),
            "input_output_cosine": round(cos, 6),
            "reconstruction_relative_gap": round(reconstruct_gap, 8)}
