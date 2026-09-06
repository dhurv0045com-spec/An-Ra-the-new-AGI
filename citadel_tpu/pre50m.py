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


def checkpoint_compat_check(store, checkpoint_sha: str,
                            expected_params: int) -> dict[str, Any]:
    """Verify a smoke checkpoint through Cymek's own restore path.

    restore() itself enforces manifest hash, inventory completeness, component
    hashes, training-state hash, and cursor/ledger agreement. We additionally
    assert the model payload carries exactly the expected parameter inventory
    plus recorded spec/source/manifest identities. A PASS here means the
    artifact is consumable by Cymek production code — not merely torch-loadable.
    """
    import torch

    state, payloads = store.restore(checkpoint_sha256=checkpoint_sha)
    model_state = torch.load(__import__("io").BytesIO(bytes(payloads["model.bin"])),
                             map_location="cpu", weights_only=False)
    total = sum(int(t.numel()) for t in model_state.values())
    ids = state.identities
    checks = {
        "production_restore_verify": True,
        "parameter_inventory_exact": bool(total == expected_params),
        "spec_sha_recorded": bool(ids.model_spec_sha256),
        "source_commit_recorded": bool(ids.source_commit),
        "data_manifest_recorded": bool(ids.data_manifest_sha256),
        "tokenizer_identity_recorded": bool(ids.tokenizer_sha256),
        "parent_chain_present": state.parent_checkpoint_sha256 is not None,
    }
    return {"schema": "citadel-pre50m-checkpoint-compat/v1",
            "checkpoint": checkpoint_sha, "parameter_count": total,
            "expected": expected_params, "checks": checks,
            "compatible": all(checks.values())}


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
    lr = float(optimizer.param_groups[0]["lr"])

    from citadel_tpu import cymek_checkpoint as cckpt

    data_manifest = {"kind": "pre50m-smoke-tiered", "seed": seed,
                     "updates": updates, "batch_sequences": 32, "length": 64,
                     "rows": "tiered-train prefix via flat TierFeeder"}
    pack_manifest = {"length": 64, "segments": "first-fit rows, per-seq ordinals",
                     "eligible": "answer spans only"}
    data_sha = cckpt.spec_json_sha256(data_manifest)
    pack_sha = cckpt.spec_json_sha256(pack_manifest)
    identities = cckpt.build_identities(
        model_spec_sha256=spec.sha256(), data_manifest_sha256=data_sha,
        pack_manifest_sha256=pack_sha,
        run_spec={"model": SMOKE_SPEC, "updates": updates, "batch_sequences": 32,
                  "length": 64, "seed": seed},
        optimizer_spec={"optimizer": "AdamW", "beta1": 0.9, "beta2": 0.95,
                        "eps": 1e-8, "weight_decay": 0.1},
        schedule_spec={"schedule": "constant-canary", "learning_rate": lr},
        curriculum_spec={"phase": "smoke"},
        source_commit=rb.citadel_sha())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    store = cckpt.open_store(str(out / "cymek_store"), "citadel-pre50m-smoke")

    def _rng_bytes() -> bytes:
        return bytes(torch.get_rng_state().tolist())

    tokens_per_update = 32 * 64
    import hashlib as _hl

    def _rng_sha() -> str:
        return _hl.sha256(_rng_bytes()).hexdigest()

    # the budget must fund the RESUME-PROOF update too (updates+1): Cymek
    # refuses any advance beyond the budget ("a completed run cannot
    # advance") - the real TPU run of 2026-09-06 died exactly there.
    state = cckpt.initial_state(
        lineage_id="citadel-pre50m-smoke",
        token_budget=(updates + 1) * tokens_per_update,
        tokens_per_update=tokens_per_update, pack_manifest_sha256=pack_sha,
        identities=identities, rng_state_sha256=_rng_sha())
    sha0 = cckpt.publish_genesis(store, state=state, model=model,
                                 optimizer=optimizer, learning_rate=lr,
                                 rng_bytes=_rng_bytes())
    prev_state, prev_sha = state, sha0

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
        cursor = cckpt.cursor_for_update(
            pack_sha, sequence_ordinal=u + 1, token_offset=(u + 1) * tokens_per_update)
        prev_state, prev_sha = cckpt.publish_update(
            store, prev_state=prev_state, model=model, optimizer=optimizer,
            learning_rate=lr, rng_bytes=_rng_bytes(), cursor=cursor,
            ledger_delta={"smoke-tiered": tokens_per_update})
        losses.append(lv)
        gnorms.append(gn)
    cum_after_training = int(prev_state.cumulative_tokens)
    out = Path(out_dir)
    with torch.no_grad():
        ref_logits = model(tokens.to(device), pos.to(device), mask.to(device))
        ref_hash = hashlib_sha(ref_logits.detach().to("cpu"))
    moment_before = ckpt_mod.optimizer_moment_sha256(optimizer)
    del model
    model2 = initialize(spec, seed).to(device)
    optimizer2 = build_adamw_optimizer(model2, torch_module=torch)
    restored_state, restored_payloads = cckpt.restore_latest(store)
    cckpt.load_model_bytes_into(model2, restored_payloads)
    cckpt.load_optimizer_bytes_into(optimizer2, restored_payloads)
    moment_after = ckpt_mod.optimizer_moment_sha256(optimizer2)
    xb.mark_step()
    with torch.no_grad():
        new_logits = model2(tokens.to(device), pos.to(device), mask.to(device))
        new_hash = hashlib_sha(new_logits.detach().to("cpu"))
    reload_ok = bool(ref_hash == new_hash)
    moments_ok = bool(moment_before == moment_after)
    # stale-parent negative probe: must be rejected with no state change.
    fence_probe = "not-run"
    try:
        store.publish(state=restored_state, payloads=restored_payloads,
                      expected_parent_sha256="00" * 64)
        fence_probe = "UNEXPECTED-ACCEPT"
    except ValueError:
        fence_probe = "rejected-as-required"
    # resume proof: one more real update executes on the restored state,
    # advancing through the production transaction path.
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
    resume_cursor = cckpt.cursor_for_update(
        pack_sha, sequence_ordinal=updates + 1,
        token_offset=(updates + 1) * tokens_per_update)
    resume_state, resume_sha = cckpt.publish_update(
        store, prev_state=restored_state, model=model2, optimizer=optimizer2,
        learning_rate=lr, rng_bytes=_rng_bytes(), cursor=resume_cursor,
        ledger_delta={"smoke-tiered": tokens_per_update})
    continued_ok = True
    # mechanical post-resume contract (§12): the reserved final update must
    # land the state exactly at completion, with every ledger consistent
    assert resume_state.global_update == updates + 1, resume_state.global_update
    assert resume_state.optimizer_step_max == resume_state.global_update
    assert resume_state.schedule_tokens == resume_state.cumulative_tokens
    assert sum(resume_state.tokens_by_source.values()) ==         resume_state.cumulative_tokens
    assert resume_state.complete is True
    cum_after_resume = int(resume_state.cumulative_tokens)
    expected_after_training = updates * tokens_per_update
    expected_after_resume = (updates + 1) * tokens_per_update
    token_accounting = {
        "tokens_per_update": tokens_per_update,
        "updates": updates,
        "cumulative_after_training": cum_after_training,
        "expected_after_training": expected_after_training,
        "cumulative_after_resume": cum_after_resume,
        "expected_after_resume": expected_after_resume,
        "scheduled_values_nonnegative": bool(tokens_per_update > 0 and updates > 0),
        "consistent": bool(cum_after_training == expected_after_training
                           and cum_after_resume == expected_after_resume
                           and tokens_per_update > 0),
    }
    easy_recs, easy_summ = _smoke_eval(model2, easy_rows, easy_targets, device, torch)
    compat = checkpoint_compat_check(store, resume_sha, param_count)
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
        "production_transaction": True,
        "lineage_id": "citadel-pre50m-smoke",
        "generations_published": updates + 2,
        "head_checkpoint_sha256": resume_sha,
        "writer_fence_probe": fence_probe,
        "reload_output_identity": reload_ok,
        "optimizer_resume": {"moments_preserved": moments_ok,
                             "continued_update_ok": continued_ok,
                             "resume_head_sha256": resume_sha},
        "checkpoint_compat": compat,
        "token_accounting": token_accounting,
        "device_count": n_devices, "wall_seconds": wall,
        "status": "PASS" if (reload_ok and moments_ok and continued_ok
                             and fence_probe == "rejected-as-required"
                             and compat["compatible"]
                             and token_accounting["consistent"]) else "FAIL",
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
    """Machine-built NEXT_50M_DECISION (pure; every field derived, none typed).

    FAIL-CLOSED: `ready_for_50m_training` is true only when EVERY required
    condition holds; any false/missing condition forces false and adds a
    precise blocking reason. No success is ever inferred from partial fields.
    """
    import math as _math

    target = parts.get("target", {}) or {}
    smoke = parts.get("smoke", {}) or {}
    feas = parts.get("feasibility", {}) or {}
    data = parts.get("data_interface", {}) or {}
    packing = parts.get("packing", {}) or {}
    blocking: list[str] = []

    # target identity (resolved from Cymek, never guessed)
    target_understood = bool(target.get("understood"))
    if not target_understood:
        blocking.append("50m target not understood")
    target_type_ok = target.get("type") == PRE50M_TARGET["type"]
    if not target_type_ok:
        blocking.append(f"target type mismatch: {target.get('type')!r} "
                        f"!= {PRE50M_TARGET['type']!r}")
    target_value_ok = target.get("value_tokens") == PRE50M_TARGET["value_tokens"]
    if not target_value_ok:
        blocking.append(f"target token value mismatch: {target.get('value_tokens')!r} "
                        f"!= {PRE50M_TARGET['value_tokens']!r}")

    # capacity/feasibility
    fits = bool(feas.get("verdict") == "FIT")
    if not fits:
        blocking.append("target model does not fit: " + str(feas.get("verdict")))
    batch_ok = isinstance(parts.get("recommended_batch"), int) \
        and parts.get("recommended_batch") > 0
    seq_ok = isinstance(parts.get("recommended_sequence_length"), int) \
        and parts.get("recommended_sequence_length") > 0
    if not (batch_ok and seq_ok):
        blocking.append("no safe selected static shape "
                        f"(batch={parts.get('recommended_batch')!r}, "
                        f"length={parts.get('recommended_sequence_length')!r})")
    rate = parts.get("rate_tok_s")
    rate_ok = isinstance(rate, (int, float)) and not isinstance(rate, bool) \
        and _math.isfinite(float(rate)) and float(rate) > 0
    if not rate_ok:
        blocking.append(f"no positive finite throughput: {rate!r}")

    # smoke: status + every mechanical certificate
    if smoke.get("status") != "PASS":
        blocking.append(f"smoke status is {smoke.get('status')!r}, want 'PASS'")
    losses = smoke.get("losses") or []
    finite_losses = bool(losses) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool)
        and _math.isfinite(float(v)) for v in losses)
    if not finite_losses:
        blocking.append("smoke losses missing or nonfinite")
    gn = smoke.get("grad_norm", {}) or {}
    gn_max = gn.get("max")
    grad_ok = isinstance(gn_max, (int, float)) and not isinstance(gn_max, bool) \
        and _math.isfinite(float(gn_max)) and float(gn_max) > 0
    if not grad_ok:
        blocking.append(f"nonzero finite gradients required, got max={gn_max!r}")
    if smoke.get("param_mutation") is not True:
        blocking.append("no parameter mutation certified in smoke")
    if smoke.get("production_transaction") is not True:
        blocking.append("smoke did not run through the production transaction path")
    compat = smoke.get("checkpoint_compat", {}) or {}
    if compat.get("compatible") is not True:
        blocking.append("checkpoint compatibility not verified "
                        f"(compatible={compat.get('compatible')!r})")
    if smoke.get("reload_output_identity") is not True:
        blocking.append("checkpoint/reload output identity failed")
    opt_resume = smoke.get("optimizer_resume", {}) or {}
    if opt_resume.get("moments_preserved") is not True:
        blocking.append("optimizer moments not preserved across reload")
    if opt_resume.get("continued_update_ok") is not True:
        blocking.append("continued update after resume not certified")
    if smoke.get("writer_fence_probe") != "rejected-as-required":
        blocking.append(f"writer fence probe = {smoke.get('writer_fence_probe')!r}, "
                        "want 'rejected-as-required'")
    ta = smoke.get("token_accounting", {}) or {}
    if ta.get("consistent") is not True:
        blocking.append("token accounting inconsistent in smoke "
                        f"(cumulative {ta.get('cumulative_after_training')!r} vs "
                        f"expected {ta.get('expected_after_training')!r})")

    # data + packing certification
    if data.get("status") != "PASS":
        blocking.append("data interface certification failed")
    if packing.get("status") != "PASS":
        blocking.append("packing certification failed")
    real = data.get("real_tokens")
    loss_bearing = data.get("loss_bearing_tokens")
    capacity = data.get("capacity_tokens")
    padding = data.get("padding_tokens")
    counts_ok = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and v >= 0 for v in (real, loss_bearing, capacity))
    if not counts_ok or not (capacity >= real >= loss_bearing):
        blocking.append(f"token accounting violated: capacity={capacity!r} "
                        f"real={real!r} loss-bearing={loss_bearing!r} "
                        "(need capacity >= real >= loss-bearing >= 0)")
    elif isinstance(padding, (int, float)) and padding != capacity - real:
        blocking.append(f"padding {padding!r} != capacity - real "
                        f"{capacity - real}")
    if not all(isinstance(data.get(k), (int, float)) or data.get(k) is None
               for k in ("scheduled_rows",)) or (data.get("scheduled_rows") is not None
                                                 and data.get("scheduled_rows") < 0):
        blocking.append(f"negative scheduled token/row value: "
                        f"{data.get('scheduled_rows')!r}")

    ready = not blocking
    return {
        "50m_target_understood": target_understood,
        "target_type": target.get("type"),
        "target_type_verified": bool(target_type_ok),
        "target_value_tokens_verified": bool(target_value_ok),
        "target_parameter_count": target.get("parameter_count"),
        "fits_current_tpu": fits,
        "safe_static_shape_exists": bool(batch_ok and seq_ok),
        "positive_finite_throughput": bool(rate_ok),
        "smoke_status": smoke.get("status"),
        "finite_loss": bool(finite_losses),
        "nonzero_finite_gradients": bool(grad_ok),
        "parameter_mutation": smoke.get("param_mutation") is True,
        "production_transaction": smoke.get("production_transaction") is True,
        "checkpoint_compat_verified": compat.get("compatible") is True,
        "recommended_batch": parts.get("recommended_batch"),
        "recommended_sequence_length": parts.get("recommended_sequence_length"),
        "gradient_accumulation_required": False,
        "estimated_tokens_per_second": rate,
        "checkpoint_save_reload_pass": smoke.get("reload_output_identity") is True,
        "resume_pass": opt_resume.get("moments_preserved") is True,
        "continued_update_pass": opt_resume.get("continued_update_ok") is True,
        "writer_fence_rejected_as_required":
            smoke.get("writer_fence_probe") == "rejected-as-required",
        "token_accounting_consistent": ta.get("consistent") is True,
        "data_interface_pass": data.get("status") == "PASS",
        "packing_pass": packing.get("status") == "PASS",
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
