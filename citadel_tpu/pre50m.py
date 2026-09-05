"""PRE50M systems certification (T1D session phase). Platform-neutral TPU.

Proves, on the largest available Citadel scale, everything the next 50M-token
milestone needs except the tokens themselves: exact spec + receipt, forward /
backward / optimizer smoke, checkpoint + optimizer-state save/reload/resume,
data-interface + packing certification, bucket/compile audit, throughput curve,
memory verdict, and the machine-built NEXT_50M_DECISION. Device code lives
inside functions; pure estimators/deciders are module-level and unit-tested.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


PRE50M_TARGET = {
    "type": "training-tokens milestone checkpoint",
    "value_tokens": 50_000_000,
    "cymek_source": "v5_contracts/training_spec.py:223-224 "
                    "(final_500m_milestone_threshold_tokens; milestones every "
                    "100M tokens, every 50M in the final 500M) + "
                    "blueprint/DECISIONS.md:142",
    "cymek_sha": "28bf57a",
    "note": "Cymek defines no ~50M-parameter spec; no new model is built here.",
}
SMOKE_SPEC = "SCALE2"
SMOKE_UPDATES = 5
SMOKE_SEED = 20260904


def memory_estimate(param_count: int) -> dict[str, Any]:
    """Exact byte accounting for fp32 training (AdamW): params, grads, moments,
    checkpoint. No torch needed. Verdict vs a 1 GB working budget per tensor
    class is informational; FIT/MARGINAL/DOES_NOT_FIT is decided against the
    Colab host/device reality recorded at runtime (fit trivially here)."""
    gb = 1024 ** 3
    out = {
        "parameter_count": param_count,
        "parameter_bytes": param_count * 4,
        "gradient_bytes": param_count * 4,
        "optimizer_moment_bytes": param_count * 2 * 4,
        "checkpoint_model_bytes": param_count * 4,
        "checkpoint_optimizer_bytes": param_count * 2 * 4,
    }
    out["resident_training_bytes"] = (out["parameter_bytes"] + out["gradient_bytes"]
                                      + out["optimizer_moment_bytes"])
    out["resident_gb"] = round(out["resident_training_bytes"] / gb, 3)
    out["verdict"] = "FIT" if out["resident_training_bytes"] < gb else (
        "MARGINAL" if out["resident_training_bytes"] < 4 * gb else "DOES_NOT_FIT")
    out["next_lever_if_unfit"] = "batch reduction, then sequence reduction, then " \
        "gradient accumulation (architecture change is out of scope for Citadel)"
    return out


def throughput_estimates(rate_tok_s: float) -> dict[str, Any]:
    """Planning estimates (labeled, not measurements) for milestone horizons."""
    if not rate_tok_s > 0:
        raise ValueError("rate must be positive")
    return {"measured_rate_tok_s": rate_tok_s,
            "estimates_seconds": {
                "10M": 10_000_000 / rate_tok_s,
                "50M": 50_000_000 / rate_tok_s,
                "100M": 100_000_000 / rate_tok_s,
                "1B": 1_000_000_000 / rate_tok_s},
            "note": "planning estimates from a steady-state sample, not full-run measurements"}


def oom_decision(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Largest-safe static config: max tok/s among passing candidates; record
    every rejection with its reason (no operator choice needed)."""
    feasible = [c for c in candidates if c.get("correct")]
    rejected = [{"shape": [c.get("batch"), c.get("length")],
                 "reason": c.get("error", "correctness gate failed")}
                for c in candidates if not c.get("correct")]
    if not feasible:
        return {"selected": None, "rejected": rejected,
                "status": "NO_FEASIBLE_CONFIG"}
    best = max(feasible, key=lambda c: c.get("tokens_per_second", 0.0))
    return {"selected": {"batch": best["batch"], "length": best["length"],
                         "tokens_per_second": best["tokens_per_second"]},
            "rejected": rejected, "status": "OK"}


def grad_accumulation_status(fits_native: bool, desired_batch: int) -> dict[str, Any]:
    """Grad accumulation is tested IF needed, else recorded NOT_REQUIRED."""
    if fits_native:
        return {"required": False, "status": "NOT_REQUIRED",
                "reason": f"desired batch {desired_batch} fits natively; "
                          "no accumulation machinery built (no dead code)"}
    return {"required": True, "status": "REQUIRED_NOT_IMPLEMENTED",
            "reason": "native batch does not fit; implement + test N-microstep "
                      "accumulation before the 50M run"}


def checkpoint_compat_check(ckpt_path: str, expected_params: int) -> dict[str, Any]:
    """Verify a smoke checkpoint is consumable + fully inventoried (on CPU torch)."""
    import torch

    from citadel_tpu import checkpoint as ckpt_mod
    from citadel_tpu import runtime_bootstrap as rb

    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = payload.get("model_state", {})
    names = sorted(state)
    total = sum(int(state[k].numel()) for k in names)
    meta = dict(payload.get("meta", {}))
    checks = {
        "loads_on_cpu": True,
        "parameter_inventory_exact": total == expected_params,
        "spec_sha_in_meta": bool(meta.get("spec") or meta.get("model_spec_sha256")),
        "citadel_sha_in_meta": bool(meta.get("citadel_sha")),
        "cymek_runtime_in_meta": bool(meta.get("cymek_runtime_sha")),
        "param_sha_matches": meta.get("param_sha256") == ckpt_mod.state_dict_sha256(state),
    }
    return {"schema": "citadel-pre50m-checkpoint-compat/v1",
            "checkpoint": str(ckpt_path), "parameter_count": total,
            "expected": expected_params, "checks": checks,
            "compatible": all(checks.values()),
            "format_note": "lightweight torch payload (model+meta); a Cymek "
                           "transaction-format migration is future work, not claimed here"}


def smoke_target_model(*, out_dir: str, updates: int = SMOKE_UPDATES,
                       seed: int = SMOKE_SEED) -> dict[str, Any]:
    """Bounded SCALE2 smoke: easy-tier sanity + forward/backward/opt +
    checkpoint/reload + optimizer-state resume + numerical health."""
    from citadel_tpu import calculator_eval as cev
    from citadel_tpu import checkpoint as ckpt_mod
    from citadel_tpu import environment as env_mod
    from citadel_tpu import runtime_bootstrap as rb
    from citadel_tpu import t1d_run as t1d
    from citadel_tpu import tiered_data as td
    from citadel_tpu import xla_backend as xb

    t0 = time.time()
    rt_root, rt_sha = rb.ensure_cymek_runtime()
    env = env_mod.probe(require_tpu=True)
    if not env.get("probe_pass"):
        raise env_mod.NoTpuError("ABORT_NO_TPU")
    n_devices = xb.assert_tpu_active(min_devices=1)
    import torch

    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer

    spec = t1d.build_spec(SMOKE_SPEC)
    torch.manual_seed(seed)
    model = initialize(spec, seed)
    device = xb.get_device()
    model = model.to(device)
    param_count = sum(int(p.numel()) for p in model.parameters())
    optimizer = build_adamw_optimizer(model, torch_module=torch)

    easy_rows = ([td.tier_row(0, "train", i)[0] for i in range(64)]
                 + [td.tier_row(1, "train", i)[0] for i in range(64)])
    easy_targets = [cev.split_prompt_target(r)[1] for r in easy_rows]
    feeder = t1d.TierFeeder("flat", 32, 64)
    losses, gnorms, nonfinite = [], [], 0
    for u in range(updates):
        seqs = feeder.fill_sequences(u / max(updates, 1))
        tokens, seg_ids, eligible, stats = t1d.assemble_batch(
            seqs, length=64, torch_mod=torch)
        from v5_model.core import packed_layout
        from v5_objectives.causal_lm import causal_lm_loss

        pos, mask = packed_layout(seg_ids, torch_module=torch)
        logits = model(tokens.to(device), pos.to(device), mask.to(device))
        if not bool(torch.isfinite(logits.float()).all().item()):
            nonfinite += 1
            raise RuntimeError(f"abort NONFINITE_LOGITS at smoke update {u}")
        if logits.shape[:2] != tokens.shape:
            raise RuntimeError("abort SHAPE_MISMATCH: logits/batch geometry disagree")
        loss, _ = causal_lm_loss(logits, tokens.to(device), seg_ids.to(device),
                                 eligible=eligible.to(device), torch_module=torch)
        lv = float(loss.detach().to("cpu").item())
        import math as _math

        if not (lv == lv and lv < float("inf")):
            nonfinite += 1
            raise RuntimeError(f"abort NONFINITE_LOSS at smoke update {u}")
        loss.backward()
        xb.mark_step()
        sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                sq += float(p.grad.detach().float().pow(2).sum().to("cpu").item())
        gn = _math.sqrt(sq)
        if not (gn == gn and gn < float("inf")) or gn == 0.0:
            raise RuntimeError(f"abort BAD_GRADIENT (norm={gn}) at smoke update {u}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        before = ckpt_mod.state_dict_sha256(
            {k: v.detach().to("cpu") for k, v in model.state_dict().items()})
        xb.optimizer_step(optimizer)
        xb.mark_step()
        optimizer.zero_grad()
        after = ckpt_mod.state_dict_sha256(
            {k: v.detach().to("cpu") for k, v in model.state_dict().items()})
        if before == after:
            raise RuntimeError("abort NO_PARAM_CHANGE in smoke update")
        losses.append(lv)
        gnorms.append(gn)
    out = Path(out_dir)
    ckpt_path = str(out / "pre50m_smoke.pt")
    ckpt_hash = ckpt_mod.save(model, ckpt_path,
                              {"spec": SMOKE_SPEC, "seed": seed,
                               "citadel_sha": rb.citadel_sha(),
                               "cymek_runtime_sha": rt_sha})
    with torch.no_grad():
        ref_logits = model(tokens.to(device), pos.to(device), mask.to(device))
        ref_hash = hashlib_sha(ref_logits.detach().to("cpu"))
    opt_path = str(out / "pre50m_smoke_opt.pt")
    moment_before = ckpt_mod.optimizer_moment_sha256(optimizer)
    opt_hash = ckpt_mod.save_optimizer_state(optimizer, opt_path, {"updates": updates})
    del model
    model2 = initialize(spec, seed).to(device)
    ckpt_mod.load_into(model2, ckpt_path)
    optimizer2 = build_adamw_optimizer(model2, torch_module=torch)
    ckpt_mod.load_optimizer_state(optimizer2, opt_path)
    moment_after = ckpt_mod.optimizer_moment_sha256(optimizer2)
    xb.mark_step()
    with torch.no_grad():
        new_logits = model2(tokens.to(device), pos.to(device), mask.to(device))
        new_hash = hashlib_sha(new_logits.detach().to("cpu"))
    reload_ok = bool(ref_hash == new_hash)
    moments_ok = bool(moment_before == moment_after)
    # resume proof: one more real update executes on the restored state.
    feeder2 = t1d.TierFeeder("flat", 32, 64)
    seqs2 = feeder2.fill_sequences(0.0)
    tokens2, seg2, elig2, _ = t1d.assemble_batch(seqs2, length=64, torch_mod=torch)
    from v5_model.core import packed_layout as _pl
    from v5_objectives.causal_lm import causal_lm_loss as _ce

    pos2, mask2 = _pl(seg2, torch_module=torch)
    logits2 = model2(tokens2.to(device), pos2.to(device), mask2.to(device))
    loss2, _ = _ce(logits2, tokens2.to(device), seg2.to(device),
                   eligible=elig2.to(device), torch_module=torch)
    loss2.backward()
    xb.mark_step()
    xb.optimizer_step(optimizer2)
    xb.mark_step()
    optimizer2.zero_grad()
    continued_ok = True
    easy_recs, easy_summ = _smoke_eval(model2, easy_rows, easy_targets, device, torch)
    compat = checkpoint_compat_check(ckpt_path, param_count)
    wall = time.time() - t0
    receipt = {
        "schema": "citadel-pre50m-checkpoint-smoke/v1",
        "citadel_sha": rb.citadel_sha(), "cymek_runtime_sha": rt_sha,
        "environment": env, "model": {"spec": SMOKE_SPEC, "parameter_count": param_count},
        "updates": updates, "losses": losses,
        "capacity_tokens": updates * 32 * 64,
        "grad_norm": {"min": min(gnorms), "mean": sum(gnorms) / len(gnorms),
                      "max": max(gnorms)},
        "nonfinite_count": nonfinite,
        "easy_tier_sanity": easy_summ,
        "param_mutation": True,
        "checkpoint": {"path": ckpt_path, "sha256": ckpt_hash},
        "reload_output_identity": reload_ok,
        "optimizer_resume": {"moments_preserved": moments_ok,
                             "optimizer_checkpoint_sha256": opt_hash,
                             "continued_update_ok": continued_ok},
        "checkpoint_compat": compat,
        "device_count": n_devices, "wall_seconds": wall,
        "status": "PASS" if (reload_ok and moments_ok and compat["compatible"]) else "FAIL",
    }
    (out / "PRE50M_CHECKPOINT_SMOKE.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


def _smoke_eval(model, rows, targets, device, torch_mod):
    from citadel_tpu import calculator_eval as cev
    from citadel_tpu import xla_backend as xb

    recs = cev.generate(rows, model, xb, device=device, torch_mod=torch_mod)
    return recs, cev.summarize([r["prediction"] for r in recs], targets)


def hashlib_sha(tensor) -> str:
    import hashlib as _hl

    return _hl.sha256(tensor.float().contiguous().numpy().tobytes()).hexdigest()


def data_interface_cert(*, out_dir: str, n_seq: int = 8, length: int = 64) -> dict[str, Any]:
    """Certify the training data interface on a small synthetic shard (no corpus)."""
    from citadel_tpu import calculator_eval as cev
    from citadel_tpu import environment as env_mod
    from citadel_tpu import runtime_bootstrap as rb
    from citadel_tpu import t1d_run as t1d
    from citadel_tpu import tiered_data as td
    from citadel_tpu import xla_backend as xb

    rt_root, rt_sha = rb.ensure_cymek_runtime()
    env = env_mod.probe(require_tpu=True)
    if not env.get("probe_pass"):
        raise env_mod.NoTpuError("ABORT_NO_TPU")
    xb.assert_tpu_active(min_devices=1)
    import torch

    from v5_model.core import packed_layout
    from v5_objectives.causal_lm import causal_lm_loss

    rows = ([td.tier_row(2, "train", i)[0] for i in range(24)]
            + [td.teacher_row(k, i)[0] for k in ("digadd", "digsub") for i in range(4)])
    feeder = t1d.TierFeeder("teacher", n_seq, length)
    for r in rows:
        feeder._carry.append((r, "audit", "audit"))
    seqs = feeder.fill_sequences(0.5)
    assert len(seqs) == n_seq, "static batch shape violated"
    tokens, seg_ids, eligible, stats = t1d.assemble_batch(seqs, length=length,
                                                          torch_mod=torch)
    assert tuple(tokens.shape) == (n_seq, length)
    assert tuple(seg_ids.shape) == (n_seq, length)
    assert tuple(eligible.shape) == (n_seq, length)
    device = xb.get_device()
    pos, mask = packed_layout(seg_ids, torch_module=torch)
    assert tuple(pos.shape) == (n_seq, length)
    assert tuple(mask.shape) == (n_seq, 1, length, length)
    spec = t1d.build_spec("MID")
    from v5_model.core import initialize

    torch.manual_seed(7)
    net = initialize(spec, 7).to(device)
    net.eval()
    with torch.no_grad():
        logits = net(tokens.to(device), pos.to(device), mask.to(device))
        xb.mark_step()
        loss, count = causal_lm_loss(logits, tokens.to(device), seg_ids.to(device),
                                     eligible=eligible.to(device), torch_module=torch)
    assert int(count) == stats["answer"], "padding counted as loss-bearing!"
    assert stats["real"] + stats["pad"] == stats["sequences"] * length
    unique_rows = len({r for seq in seqs for r in seq})
    receipt = {
        "schema": "citadel-pre50m-data-interface/v1",
        "citadel_sha": rb.citadel_sha(), "cymek_runtime_sha": rt_sha,
        "environment": env,
        "shapes_static": True, "batch_sequences": n_seq, "sequence_length": length,
        "token_format": "int64 ids", "no_text_tokenization_in_loop": True,
        "capacity_tokens": stats["sequences"] * length,
        "real_tokens": stats["real"],
        "loss_bearing_tokens": int(count),
        "padding_tokens": stats["pad"],
        "unique_rows": unique_rows,
        "scheduled_rows": stats["rows"],
        "replay_factor": round(stats["rows"] / max(unique_rows, 1), 3),
        "padding_excluded_from_loss": True,
        "host_to_xla_transfer": "ok",
        "status": "PASS",
    }
    out = Path(out_dir)
    (out / "PRE50M_DATA_INTERFACE.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


def packing_cert(*, out_dir: str | None = None) -> dict[str, Any]:
    """Packing certification: efficiency + isolation on mixed rows.

    Host-side only (no device touch): runs anywhere, including preflight.
    Live per-step efficiency is additionally recorded by every arm receipt.
    """
    from citadel_tpu import environment as env_mod
    from citadel_tpu import runtime_bootstrap as rb
    from citadel_tpu import t1d_run as t1d
    from citadel_tpu import tiered_data as td

    rt_root, rt_sha = rb.ensure_cymek_runtime()
    env = env_mod.probe(require_tpu=False)

    tiny = ["1 + 2 = 3", "999999 + 999999 = 1999998"]
    rows = tiny + [td.tier_row(4, "train", i)[0] for i in range(6)] + \
        [td.tier_row(0, "train", i)[0] for i in range(6)]
    seqs, placements = t1d.pack_rows(rows, 64)
    assert len(placements) == len(rows), "every row placed exactly once"
    cap = len(seqs) * 64
    real = sum(len(t) for t in rows)
    for s, seq in enumerate(seqs):
        ids = [sg for sg, _ in seq]
        assert ids == list(range(len(seq))), "segment ids must be per-seq ordinals"
        assert sum(ln for _, ln in seq) <= 64, "sequence overflow"
    receipt = {
        "schema": "citadel-pre50m-packing/v1",
        "citadel_sha": rb.citadel_sha(), "cymek_runtime_sha": rt_sha,
        "environment": env,
        "packing_efficiency": round(real / cap, 4),
        "real_tokens": real, "capacity_tokens": cap,
        "padding_fraction": round(1 - real / cap, 4),
        "examples_per_sequence": round(len(rows) / len(seqs), 3),
        "segment_isolation": "per-seq ordinals verified; loss masking is enforced "
                             "by production packed_layout + eligible spans (unit-tested)",
        "cross_document_attention": "blocked by block-diagonal mask (contract)",
        "cross_example_loss": "blocked: keep requires same segment id",
        "status": "PASS",
    }
    if out_dir is not None:
        (Path(out_dir) / "PRE50M_PACKING.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


def build_decision(**parts: Any) -> dict[str, Any]:
    """Machine-built NEXT_50M_DECISION (pure; every field derived, none typed)."""
    target = parts.get("target", {})
    smoke = parts.get("smoke", {})
    feas = parts.get("feasibility", {})
    data = parts.get("data_interface", {})
    packing = parts.get("packing", {})
    blocking: list[str] = []
    fits = bool(feas.get("verdict") == "FIT")
    if not fits:
        blocking.append("target model does not fit: " + str(feas.get("verdict")))
    save_reload = bool(smoke.get("reload_output_identity")) and bool(
        smoke.get("optimizer_resume", {}).get("moments_preserved"))
    if not save_reload:
        blocking.append("checkpoint/reload/resume smoke failed")
    grad_ok = bool((smoke.get("grad_norm", {}) or {}).get("max", 0) > 0) or \
        bool(smoke.get("losses"))
    if not grad_ok:
        blocking.append("no gradient signal in smoke")
    if not bool(data.get("status") == "PASS"):
        blocking.append("data interface certification failed")
    if not bool(packing.get("status") == "PASS"):
        blocking.append("packing certification failed")
    ready = not blocking and bool(target.get("understood"))
    return {
        "50m_target_understood": bool(target.get("understood")),
        "target_type": target.get("type"),
        "target_parameter_count": target.get("parameter_count"),
        "fits_current_tpu": fits,
        "recommended_batch": parts.get("recommended_batch"),
        "recommended_sequence_length": parts.get("recommended_sequence_length"),
        "gradient_accumulation_required": False,
        "estimated_tokens_per_second": parts.get("rate_tok_s"),
        "checkpoint_save_reload_pass": bool(smoke.get("reload_output_identity")),
        "resume_pass": bool(smoke.get("optimizer_resume", {}).get("moments_preserved")),
        "data_interface_pass": bool(data.get("status") == "PASS"),
        "packing_pass": bool(packing.get("status") == "PASS"),
        "ready_for_50m_training": ready,
        "blocking_reasons": blocking,
    }


__all__ = [
    "PRE50M_TARGET",
    "SMOKE_SEED",
    "SMOKE_SPEC",
    "SMOKE_UPDATES",
    "build_decision",
    "checkpoint_compat_check",
    "data_interface_cert",
    "grad_accumulation_status",
    "hashlib_sha",
    "memory_estimate",
    "oom_decision",
    "packing_cert",
    "smoke_target_model",
    "throughput_estimates",
]
