"""CYR-GPU-014 / R1C operator runner.

Large, prospective fixed-matrix softmax mechanism campaign. Every arm owns the
same physical V24576 Cymek V5 model; only training-time output competition is
changed. Evaluation always uses the unmodified full vocabulary unless a metric
is explicitly labelled ACTIVE_ONLY_DIAGNOSTIC.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
import time
import traceback
import types
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping

from anra_v5 import cyr_gpu006_run as legacy
from anra_v5 import cyr_gpu011_run as inherited
from anra_v5.cyr_gpu011_optimizer_compat import canonical_optimizer_compat
from v5_experiments import cyr_gpu011 as base
from v5_experiments import cyr_gpu012_r1 as r1core
from v5_experiments import cyr_gpu014_r1c as core

BUNDLE_NAME = "CYMEK_R1C_SOFTMAX_MECHANISM_RESULTS.zip"
PARTIAL_BUNDLE_NAME = "CYMEK_R1C_SOFTMAX_MECHANISM_PARTIAL.zip"


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text("utf-8"))


def setup_reproducibility(torch: Any) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except Exception:
        torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass


def environment(torch: Any, device: Any) -> dict[str, Any]:
    return {
        "schema": "anra-cyr-gpu014-r1c-environment/v1",
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "tf32_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
    }


def state_fingerprint(model: Any, optimizer: Any, *, torch: Any) -> dict[str, str]:
    mb, ob = io.BytesIO(), io.BytesIO()
    torch.save(model.state_dict(), mb)
    torch.save(optimizer.state_dict(), ob)
    return {
        "model_sha256": hashlib.sha256(mb.getvalue()).hexdigest(),
        "optimizer_sha256": hashlib.sha256(ob.getvalue()).hexdigest(),
    }


def model_hash(model: Any, *, torch: Any) -> str:
    mb = io.BytesIO()
    torch.save(model.state_dict(), mb)
    return hashlib.sha256(mb.getvalue()).hexdigest()


def patch_training_treatment(model: Any, arm: str, *, torch: Any) -> Any:
    """Patch only this model instance; parameter/state inventory is unchanged."""
    original = model.forward

    def forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        logits = original(*args, **kwargs)
        if self.training and bool(getattr(self, "_r1c_treatment_enabled", True)):
            logits = core.apply_training_logits(logits, arm, torch_module=torch)
        eval_k = getattr(self, "_r1c_eval_candidate_count", None)
        if (not self.training) and eval_k is not None and int(eval_k) < core.PHYSICAL_VOCAB:
            logits = logits.clone()
            logits[..., int(eval_k):] = -1.0e4
        return logits

    model.forward = types.MethodType(forward, model)
    model._r1c_arm = arm
    model._r1c_treatment_enabled = True
    model._r1c_eval_candidate_count = None
    return model


def build_model(seed: int, arm: str, *, torch: Any, device: Any) -> Any:
    from v5_model.core import initialize
    model = initialize(r1core.spec_for(core.PHYSICAL_VOCAB), int(seed), torch_module=torch).to(device)
    return patch_training_treatment(model, arm, torch=torch)


def build_optimizer(model: Any, *, torch: Any) -> Any:
    from v5_training.optimizer import build_adamw_optimizer
    return build_adamw_optimizer(
        model, torch_module=torch, lr=base.CYR11_HIGH_LR,
        betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
    )


def make_backend(model: Any, optimizer: Any, tokenizer: Any, *, torch: Any, device: Any) -> Any:
    return legacy._make_backend(
        model=model, optimizer=optimizer, special=tokenizer.special,
        device=device, torch=torch, lr=base.CYR11_HIGH_LR,
    )


def active_only_rates(model: Any, tokenizer: Any, rows: list[dict[str, Any]], *,
                      torch: Any, device: Any) -> dict[str, Any]:
    old = model._r1c_eval_candidate_count
    try:
        model._r1c_eval_candidate_count = core.ACTIVE_VOCAB
        return legacy.generate_rates_batched(
            model, tokenizer, rows, torch=torch, device=device,
            special=tokenizer.special, batch_size=32,
        )
    finally:
        model._r1c_eval_candidate_count = old


def diagnostic_keep(tokens: Any, segment_ids: Any, eligible: Any, *, special: Mapping[str, int]) -> Any:
    targets = tokens[:, 1:]
    keep = (segment_ids[:, 1:] == segment_ids[:, :-1]) & (segment_ids[:, 1:] >= 0)
    keep = keep & (targets != int(special["bos_id"])) & (targets != int(special["pad_id"]))
    return keep & eligible[:, 1:]


def core_grad_snapshot(model: Any, loss: Any, arm: str, *, torch: Any) -> tuple[dict[str, float], list[Any]]:
    model.zero_grad(set_to_none=True)
    loss.backward()
    emb = dict(model.named_parameters())["embedding.weight"].grad.detach().float()
    spec = core.treatment_spec(arm)
    k = int(spec["candidate_count"]) if spec["kind"] == "hard_mask" else core.PHYSICAL_VOCAB
    active = float(torch.linalg.vector_norm(emb[:core.ACTIVE_VOCAB]).item())
    participating = float(torch.linalg.vector_norm(emb[core.ACTIVE_VOCAB:k]).item()) if k > core.ACTIVE_VOCAB else 0.0
    excluded = float(torch.linalg.vector_norm(emb[k:]).item()) if k < core.PHYSICAL_VOCAB else 0.0
    chunks: list[Any] = []
    sq = 0.0
    for name, p in model.named_parameters():
        if name == "embedding.weight" or p.grad is None:
            continue
        g = p.grad.detach().float().cpu().reshape(-1).clone()
        chunks.append(g)
        sq += float(torch.dot(g, g).item())
    model.zero_grad(set_to_none=True)
    return {
        "active_embedding_grad_l2": active,
        "participating_inactive_embedding_grad_l2": participating,
        "excluded_inactive_embedding_grad_l2": excluded,
        "core_grad_l2": math.sqrt(max(sq, 0.0)),
    }, chunks


def vector_cosine(a: list[Any], b: list[Any], *, torch: Any) -> float:
    if len(a) != len(b):
        raise RuntimeError("counterfactual gradient inventory mismatch")
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += float(torch.dot(x, y).item())
        na += float(torch.dot(x, x).item())
        nb += float(torch.dot(y, y).item())
    den = math.sqrt(na) * math.sqrt(nb)
    return dot / den if den > 0 else 0.0


def mechanism_diagnostic(*, model: Any, optimizer: Any, tokenizer: Any,
                         rows: list[dict[str, Any]], arm: str,
                         torch: Any, device: Any) -> dict[str, Any]:
    """Read-only probability/gradient probe; mechanically checks no state mutation."""
    from v5_model.core import packed_layout
    from v5_objectives.causal_lm import causal_lm_loss

    before = state_fingerprint(model, optimizer, torch=torch)
    was_training = model.training
    special = tokenizer.special
    tokens, segments, eligible, counted = legacy.render_batch(
        tokenizer, rows, torch=torch, device=device, special=special,
    )
    hidden_box: dict[str, Any] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        hidden_box["hidden"] = output.detach()

    handle = model.final_norm.register_forward_hook(hook)
    try:
        model.eval()
        pos, mask = packed_layout(segments, torch_module=torch)
        with torch.no_grad():
            logits = model(tokens, pos.to(device), mask.to(device)).float()
        keep = diagnostic_keep(tokens, segments, eligible, special=special)
        targets = tokens[:, 1:][keep]
        selected = logits[:, :-1][keep]
        probs = torch.softmax(selected, dim=-1)
        target_p = probs.gather(1, targets[:, None]).squeeze(1)
        active_mass = probs[:, :core.ACTIVE_VOCAB].sum(-1)
        inactive_mass = probs[:, core.ACTIVE_VOCAB:].sum(-1)
        max_inactive_p = probs[:, core.ACTIVE_VOCAB:].max(-1).values
        max_inactive_logit = selected[:, core.ACTIVE_VOCAB:].max(-1).values
        target_logit = selected.gather(1, targets[:, None]).squeeze(1)
        active_wrong = selected[:, :core.ACTIVE_VOCAB].clone()
        active_wrong.scatter_(1, targets[:, None], -1.0e9)
        best_active_wrong = active_wrong.max(-1).values
        entropy = -(probs.clamp_min(1e-30) * probs.clamp_min(1e-30).log()).sum(-1)
        ap = torch.softmax(selected[:, :core.ACTIVE_VOCAB], dim=-1)
        active_entropy = -(ap.clamp_min(1e-30) * ap.clamp_min(1e-30).log()).sum(-1)
        pred = selected.argmax(-1)
        pred_counts = {
            "correct": int((pred == targets).sum().item()),
            "wrong_active": int(((pred < core.ACTIVE_VOCAB) & (pred != targets)).sum().item()),
            "inactive": int((pred >= core.ACTIVE_VOCAB).sum().item()),
        }
        hidden_stats = None
        if "hidden" in hidden_box:
            hs = hidden_box["hidden"][:, :-1][keep].float()
            norms = torch.linalg.vector_norm(hs, dim=-1)
            hidden_stats = {
                "mean_l2": float(norms.mean().item()),
                "std_l2": float(norms.std(unbiased=False).item()),
            }

        model.train()
        model._r1c_treatment_enabled = False
        pos, mask = packed_layout(segments, torch_module=torch)

        raw = model(tokens, pos.to(device), mask.to(device), use_activation_checkpointing=False)
        treated = core.apply_training_logits(raw, arm, torch_module=torch)
        loss_arm, _ = causal_lm_loss(
            treated, tokens, segments, bos_id=special["bos_id"], pad_id=special["pad_id"],
            eligible=eligible, torch_module=torch,
        )
        arm_norms, arm_core = core_grad_snapshot(model, loss_arm, arm, torch=torch)

        raw = model(tokens, pos.to(device), mask.to(device), use_activation_checkpointing=False)
        loss_full, _ = causal_lm_loss(
            raw, tokens, segments, bos_id=special["bos_id"], pad_id=special["pad_id"],
            eligible=eligible, torch_module=torch,
        )
        _n, full_core = core_grad_snapshot(model, loss_full, "FULL_24576", torch=torch)

        raw = model(tokens, pos.to(device), mask.to(device), use_activation_checkpointing=False)
        mask4096 = core.apply_training_logits(raw, "MASK_4096", torch_module=torch)
        loss4096, _ = causal_lm_loss(
            mask4096, tokens, segments, bos_id=special["bos_id"], pad_id=special["pad_id"],
            eligible=eligible, torch_module=torch,
        )
        _n, mask4096_core = core_grad_snapshot(model, loss4096, "MASK_4096", torch=torch)
        model._r1c_treatment_enabled = True

        after = state_fingerprint(model, optimizer, torch=torch)
        if before != after:
            raise RuntimeError("R1C mechanism diagnostic mutated model/optimizer state")
        return {
            "schema": "anra-cyr-gpu014-r1c-mechanism-diagnostic/v1",
            "supervised_tokens": int(counted["supervised_tokens"]),
            "target_probability_mean": float(target_p.mean().item()),
            "active_probability_mass_mean": float(active_mass.mean().item()),
            "inactive_probability_mass_mean": float(inactive_mass.mean().item()),
            "max_inactive_probability_mean": float(max_inactive_p.mean().item()),
            "max_inactive_logit_mean": float(max_inactive_logit.mean().item()),
            "target_vs_best_active_wrong_margin_mean": float((target_logit - best_active_wrong).mean().item()),
            "target_vs_best_inactive_margin_mean": float((target_logit - max_inactive_logit).mean().item()),
            "full_softmax_entropy_mean": float(entropy.mean().item()),
            "active_only_entropy_mean": float(active_entropy.mean().item()),
            "prediction_counts": pred_counts,
            "hidden_final_norm": hidden_stats,
            "arm_gradient_partition": arm_norms,
            "core_gradient_cosine_vs_full": vector_cosine(arm_core, full_core, torch=torch),
            "core_gradient_cosine_vs_mask4096": vector_cosine(arm_core, mask4096_core, torch=torch),
        }
    finally:
        handle.remove()
        model._r1c_treatment_enabled = True
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def diagnostic_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = list(data["train"][:16])
    if len(rows) != 16:
        raise RuntimeError("R1C expected 16 frozen diagnostic rows")
    return rows


def save_checkpoint(path: Path, *, model: Any, optimizer: Any, torch: Any,
                    payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body.update({
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    })
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(body, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, *, model: Any, optimizer: Any, torch: Any,
                    expected: Mapping[str, Any]) -> dict[str, Any]:
    body = torch.load(path, map_location="cpu", weights_only=False)
    for k, v in expected.items():
        if body.get(k) != v:
            raise RuntimeError(f"R1C checkpoint identity mismatch {k}: {body.get(k)!r} != {v!r}")
    model.load_state_dict(body["model"])
    optimizer.load_state_dict(body["optimizer"])
    torch.set_rng_state(body["cpu_rng"])
    torch.cuda.set_rng_state_all(body["cuda_rng"])
    return body


def one_update(*, model: Any, backend: Any, tokenizer: Any, rows: list[dict[str, Any]],
               torch: Any, device: Any, real_tokens: int, update: int, data_sha: str) -> dict[str, Any]:
    model.train()
    return legacy._one_update(
        backend=backend, tokenizer=tokenizer, rows=rows, torch=torch, device=device,
        special=tokenizer.special, cumulative=real_tokens, update=update, data_sha=data_sha,
    )


def exact_resume_smoke(*, data: Mapping[str, Any], out: Path, torch: Any, device: Any) -> dict[str, Any]:
    """Require 10 uninterrupted updates == 5 + exact save/load + 5."""
    tokenizer = r1core.tokenizer_for(core.PHYSICAL_VOCAB)
    seed, order_seed, arm = 379991, 609991, "MASK_4096"
    stream = inherited._index_stream(
        torch=torch, seed=order_seed, world_count=len(data["train"]), rows=10 * core.BATCH_ROWS,
    )
    data_sha = str(data["source_split_sha256"])

    def fresh() -> tuple[Any, Any, Any]:
        model = build_model(seed, arm, torch=torch, device=device)
        opt = build_optimizer(model, torch=torch)
        return model, opt, make_backend(model, opt, tokenizer, torch=torch, device=device)

    def advance(model: Any, backend: Any, start: int, stop: int, real: int) -> int:
        for u in range(start, stop):
            idx = stream[u * core.BATCH_ROWS:(u + 1) * core.BATCH_ROWS].tolist()
            rows = [data["train"][int(i)] for i in idx]
            rec = one_update(
                model=model, backend=backend, tokenizer=tokenizer, rows=rows, torch=torch, device=device,
                real_tokens=real, update=u + 1, data_sha=data_sha,
            )
            real += int(rec["counted"]["real_tokens"])
        return real

    m1, o1, b1 = fresh()
    r1 = advance(m1, b1, 0, 10, 0)
    f1 = state_fingerprint(m1, o1, torch=torch)

    m2, o2, b2 = fresh()
    r2 = advance(m2, b2, 0, 5, 0)
    smoke_path = out / "PREEXECUTION_EXACT_RESUME_SMOKE.pt"
    identity = {"experiment": core.EXPERIMENT, "arm": arm, "seed": seed, "order_seed": order_seed}
    save_checkpoint(smoke_path, model=m2, optimizer=o2, torch=torch,
                    payload={**identity, "updates": 5, "real_tokens": r2})
    del m2, o2, b2
    gc.collect(); torch.cuda.empty_cache()

    m3, o3, b3 = fresh()
    saved = load_checkpoint(smoke_path, model=m3, optimizer=o3, torch=torch, expected=identity)
    r3 = advance(m3, b3, int(saved["updates"]), 10, int(saved["real_tokens"]))
    f3 = state_fingerprint(m3, o3, torch=torch)
    identical = f1 == f3 and r1 == r3
    if not identical:
        raise RuntimeError("R1C exact-resume smoke failed")
    smoke_path.unlink(missing_ok=True)
    result = {
        "schema": "anra-cyr-gpu014-r1c-exact-resume-smoke/v1",
        "status": "PASS",
        "model_optimizer_hash_identical": True,
        "real_token_counter_identical": True,
        "updates": 10,
        "arm": arm,
        "uninterrupted": f1,
        "resumed": f3,
    }
    write_json(out / "EXACT_RESUME_SMOKE.json", result)
    del m1, o1, b1, m3, o3, b3
    gc.collect(); torch.cuda.empty_cache()
    return result


def calibrate_arm(*, arm: str, data: Mapping[str, Any], torch: Any, device: Any) -> dict[str, Any]:
    tokenizer = r1core.tokenizer_for(core.PHYSICAL_VOCAB)
    model = build_model(379900 + core.ARMS.index(arm), arm, torch=torch, device=device)
    optimizer = build_optimizer(model, torch=torch)
    backend = make_backend(model, optimizer, tokenizer, torch=torch, device=device)
    stream = inherited._index_stream(
        torch=torch, seed=609900 + core.ARMS.index(arm), world_count=len(data["train"]), rows=4 * core.BATCH_ROWS,
    )
    data_sha = str(data["source_split_sha256"])
    real = 0
    try:
        for u in range(1):
            idx = stream[u * core.BATCH_ROWS:(u + 1) * core.BATCH_ROWS].tolist()
            rows = [data["train"][int(i)] for i in idx]
            rec = one_update(model=model, backend=backend, tokenizer=tokenizer, rows=rows,
                             torch=torch, device=device, real_tokens=real, update=u + 1, data_sha=data_sha)
            real += int(rec["counted"]["real_tokens"])
        torch.cuda.synchronize()
        started = time.monotonic()
        for u in range(1, 4):
            idx = stream[u * core.BATCH_ROWS:(u + 1) * core.BATCH_ROWS].tolist()
            rows = [data["train"][int(i)] for i in idx]
            rec = one_update(model=model, backend=backend, tokenizer=tokenizer, rows=rows,
                             torch=torch, device=device, real_tokens=real, update=u + 1, data_sha=data_sha)
            real += int(rec["counted"]["real_tokens"])
        torch.cuda.synchronize()
        train_sec = max(time.monotonic() - started, 1e-9)
        eval_rows = list(data["dev_measurement"][:32])
        torch.cuda.synchronize(); started = time.monotonic()
        legacy.generate_rates_batched(model, tokenizer, eval_rows, torch=torch, device=device,
                                      special=tokenizer.special, batch_size=32)
        torch.cuda.synchronize(); eval_sec = max(time.monotonic() - started, 1e-9)
        drows = diagnostic_rows(data)[:8]
        torch.cuda.synchronize(); started = time.monotonic()
        mechanism_diagnostic(model=model, optimizer=optimizer, tokenizer=tokenizer, rows=drows,
                             arm=arm, torch=torch, device=device)
        torch.cuda.synchronize(); diag_sec = max(time.monotonic() - started, 1e-9)
        return {
            "schema": "anra-cyr-gpu014-r1c-calibration/v1",
            "status": "PASS", "arm": arm, "batch_rows": core.BATCH_ROWS,
            "training_updates_per_sec": 3.0 / train_sec,
            "generation_examples_per_sec": len(eval_rows) / eval_sec,
            "diagnostic_seconds": diag_sec * 2.0,
            "diagnostic_calibration_rows": len(drows),
            "diagnostic_science_rows": 16,
            "parameters": sum(p.numel() for p in model.parameters()),
            "physical_vocabulary": core.PHYSICAL_VOCAB,
            "treatment": core.treatment_spec(arm),
        }
    finally:
        del model, optimizer, backend
        gc.collect(); torch.cuda.empty_cache()


def preflight(*, repo: Path, out: Path, torch: Any, device: Any) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    setup_reproducibility(torch)
    manifest = repo / "docs/cymek/experiments/CYR-GPU-011/ARK002B_TASK_MANIFEST.json"
    data = base.load_ark002b_manifest(manifest)
    if data["source_split_sha256"] != "0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236":
        raise RuntimeError("R1C ARK-002B split identity drift")
    smoke = exact_resume_smoke(data=data, out=out, torch=torch, device=device)
    calibrations: dict[str, Any] = {}
    for arm in core.ARMS:
        print(f"R1C calibration: {arm}", flush=True)
        calibrations[arm] = calibrate_arm(arm=arm, data=data, torch=torch, device=device)
    resolved = core.resolve_from_calibrations(calibrations)

    # Same seed must produce byte-identical full V24576 initialization across all treatments.
    init_hashes: dict[str, str] = {}
    for arm in core.ARMS:
        m = build_model(core.MODEL_SEEDS[0], arm, torch=torch, device=device)
        init_hashes[arm] = model_hash(m, torch=torch)
        del m; gc.collect(); torch.cuda.empty_cache()
    if len(set(init_hashes.values())) != 1:
        raise RuntimeError("R1C matched physical initialization gate failed")
    matched = {"status": "PASS", "seed": core.MODEL_SEEDS[0], "model_sha256_by_arm": init_hashes}
    write_json(out / "CALIBRATION.json", calibrations)
    write_json(out / "RESOLVED.json", resolved)
    write_json(out / "MATCHED_INIT_PREFLIGHT.json", matched)
    gate = {
        "schema": "anra-cyr-gpu014-r1c-preexecution-gate/v1", "status": "PASS",
        "environment": environment(torch, device), "data_split_sha256": data["source_split_sha256"],
        "exact_resume": smoke, "matched_initialization": matched,
        "resolved": resolved,
    }
    write_json(out / "PREEXECUTION_GATE.json", gate)
    print("R1C PREEXECUTION GATE: PASS", flush=True)
    print("Estimated full campaign minutes:", round(float(resolved["estimated_campaign_seconds"]) / 60.0, 1), flush=True)
    return gate


def run_arm(*, out: Path, seed_index: int, arm: str, data: Mapping[str, Any], battery: Mapping[str, Any],
            torch: Any, device: Any, deadline: float,
            progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    label = core.arm_label(seed_index, arm)
    model_seed = core.MODEL_SEEDS[seed_index]
    order_seed = core.ORDER_SEEDS[seed_index]
    root = out / "arms" / label
    final_path = root / "R1C_ARM.json"
    old = read_json(final_path)
    if old is not None:
        acq = old.get("acquisition", {})
        if (old.get("experiment") == core.EXPERIMENT and old.get("arm") == arm
                and int(old.get("model_seed", -1)) == model_seed
                and int(old.get("order_seed", -1)) == order_seed
                and int(acq.get("updates", -1)) == core.UPDATES):
            old["resume_action"] = "SKIPPED_COMPLETED_ARM"
            return old
        raise RuntimeError(f"incompatible completed arm exists: {final_path}")

    root.mkdir(parents=True, exist_ok=True)
    tokenizer = r1core.tokenizer_for(core.PHYSICAL_VOCAB)
    model = build_model(model_seed, arm, torch=torch, device=device)
    optimizer = build_optimizer(model, torch=torch)
    backend = make_backend(model, optimizer, tokenizer, torch=torch, device=device)
    initial_flat = inherited._flat_snapshot(model, torch=torch)
    initial_model_sha = model_hash(model, torch=torch)
    stream = inherited._index_stream(
        torch=torch, seed=order_seed, world_count=len(data["train"]), rows=core.ROW_PRESENTATIONS,
    )
    stream_sha = hashlib.sha256(stream.numpy().tobytes()).hexdigest()
    data_sha = str(data["source_split_sha256"])
    drows = diagnostic_rows(data)
    drows_sha = hashlib.sha256(canonical([(r["prompt"], r["answer"]) for r in drows])).hexdigest()
    identity = {
        "experiment": core.EXPERIMENT, "label": label, "arm": arm,
        "model_seed": model_seed, "order_seed": order_seed,
        "data_sha256": data_sha, "semantic_stream_sha256": stream_sha,
        "target_updates": core.UPDATES, "physical_vocabulary": core.PHYSICAL_VOCAB,
        "treatment": core.treatment_spec(arm), "diagnostic_rows_sha256": drows_sha,
    }
    checkpoint = root / "resume.pt"
    updates = real_tokens = 0
    trace: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    structural: list[dict[str, Any]] = []

    if checkpoint.exists():
        saved = load_checkpoint(checkpoint, model=model, optimizer=optimizer, torch=torch, expected=identity)
        updates = int(saved["updates"]); real_tokens = int(saved["real_tokens"])
        trace = list(saved.get("trace", [])); diagnostics = list(saved.get("diagnostics", []))
        structural = list(saved.get("structural", []))
        if saved.get("initial_model_sha256") != initial_model_sha:
            raise RuntimeError("R1C initial-model identity drift on resume")
        print(f"R1C RESUME {label}: {updates}/{core.UPDATES}", flush=True)
    else:
        ctrl = legacy.generate_rates_batched(model, tokenizer, list(data["dev_controller"]),
                                             torch=torch, device=device, special=tokenizer.special)
        meas = legacy.generate_rates_batched(model, tokenizer, list(data["dev_measurement"]),
                                             torch=torch, device=device, special=tokenizer.special)
        active = active_only_rates(model, tokenizer, list(data["dev_measurement"]), torch=torch, device=device)
        trace.append({"update": 0, "row_presentations": 0, "dev_controller": ctrl,
                      "dev_measurement": meas, "active_only_measurement_diagnostic": active})
        diagnostics.append({"update": 0, **mechanism_diagnostic(
            model=model, optimizer=optimizer, tokenizer=tokenizer, rows=drows,
            arm=arm, torch=torch, device=device)})
        structural.append({"update": 0, "battery": inherited.reasoning_battery(
            model, tokenizer, battery, torch=torch, device=device,
            special=tokenizer.special, include_verbal=False)})

    while updates < core.UPDATES:
        if time.monotonic() >= deadline - 90.0:
            save_checkpoint(checkpoint, model=model, optimizer=optimizer, torch=torch,
                            payload={**identity, "updates": updates, "real_tokens": real_tokens,
                                     "trace": trace, "diagnostics": diagnostics, "structural": structural,
                                     "initial_model_sha256": initial_model_sha})
            partial = {**identity, "status": "PARTIAL_TIMEBOX", "updates": updates,
                       "real_tokens": real_tokens, "trace": trace, "diagnostics": diagnostics,
                       "structural": structural, "initial_model_sha256": initial_model_sha}
            write_json(root / "PARTIAL.json", partial)
            del model, optimizer, backend
            gc.collect(); torch.cuda.empty_cache()
            return {"experiment": core.EXPERIMENT, "arm": arm, "model_seed": model_seed,
                    "order_seed": order_seed, "status": "PARTIAL_TIMEBOX", "acquisition": partial}

        start = updates * core.BATCH_ROWS
        indices = stream[start:start + core.BATCH_ROWS].tolist()
        rows = [data["train"][int(i)] for i in indices]
        rec = one_update(
            model=model, backend=backend, tokenizer=tokenizer, rows=rows,
            torch=torch, device=device, real_tokens=real_tokens,
            update=updates + 1, data_sha=data_sha,
        )
        real_tokens += int(rec["counted"]["real_tokens"])
        updates += 1

        if progress and updates % 150 == 0:
            progress(f"R1C {label}: {updates}/{core.UPDATES}")

        if updates % core.EVAL_EVERY == 0:
            ctrl = legacy.generate_rates_batched(model, tokenizer, list(data["dev_controller"]),
                                                 torch=torch, device=device, special=tokenizer.special)
            meas = legacy.generate_rates_batched(model, tokenizer, list(data["dev_measurement"]),
                                                 torch=torch, device=device, special=tokenizer.special)
            probe = legacy.generate_rates_batched(model, tokenizer, list(data["train"][:100]),
                                                  torch=torch, device=device, special=tokenizer.special)
            active = active_only_rates(model, tokenizer, list(data["dev_measurement"]), torch=torch, device=device)
            entry = {
                "update": updates, "row_presentations": updates * core.BATCH_ROWS,
                "actual_real_tokens": real_tokens, "dev_controller": ctrl,
                "dev_measurement": meas, "train_probe": probe,
                "active_only_measurement_diagnostic": active,
                "relative_displacement": inherited._relative_displacement(model, initial_flat, torch=torch),
            }
            trace.append(entry)
            if updates in core.DIAGNOSTIC_UPDATES:
                diagnostics.append({"update": updates, **mechanism_diagnostic(
                    model=model, optimizer=optimizer, tokenizer=tokenizer, rows=drows,
                    arm=arm, torch=torch, device=device)})
            if updates in {1500, core.UPDATES}:
                structural.append({"update": updates, "battery": inherited.reasoning_battery(
                    model, tokenizer, battery, torch=torch, device=device,
                    special=tokenizer.special, include_verbal=False)})
            write_json(root / "PROGRESS.json", {**identity, "status": "RUNNING", "updates": updates,
                       "real_tokens": real_tokens, "trace": trace, "diagnostics": diagnostics,
                       "structural": structural, "initial_model_sha256": initial_model_sha})

        if updates % core.CHECKPOINT_EVERY == 0:
            save_checkpoint(checkpoint, model=model, optimizer=optimizer, torch=torch,
                            payload={**identity, "updates": updates, "real_tokens": real_tokens,
                                     "trace": trace, "diagnostics": diagnostics, "structural": structural,
                                     "initial_model_sha256": initial_model_sha})

    final_battery = inherited.reasoning_battery(
        model, tokenizer, battery, torch=torch, device=device, special=tokenizer.special,
        include_verbal=False, include_predictions=True,
    )
    final_fp = state_fingerprint(model, optimizer, torch=torch)
    acquisition = {
        "schema": "anra-cyr-gpu014-r1c-acquisition/v1",
        "label": label, "status": "COMPLETE", "model_seed": model_seed, "order_seed": order_seed,
        "batch_rows": core.BATCH_ROWS, "updates": updates,
        "row_presentations": updates * core.BATCH_ROWS, "actual_real_tokens": real_tokens,
        "semantic_stream_sha256": stream_sha, "trace": trace,
        "diagnostics": diagnostics, "structural_batteries": structural,
        "reasoning_battery_final": final_battery,
        "relative_displacement_final": inherited._relative_displacement(model, initial_flat, torch=torch),
    }
    body = {
        "schema": "anra-cyr-gpu014-r1c-arm/v1", "experiment": core.EXPERIMENT,
        "label": label, "arm": arm, "treatment": core.treatment_spec(arm),
        "physical_vocabulary": core.PHYSICAL_VOCAB, "active_token_ids": list(range(core.ACTIVE_VOCAB)),
        "model_seed": model_seed, "order_seed": order_seed,
        "target_updates": core.UPDATES, "target_row_presentations": core.ROW_PRESENTATIONS,
        "initial_model_sha256": initial_model_sha, "final_state": final_fp,
        "data_sha256": data_sha, "diagnostic_rows_sha256": drows_sha,
        "acquisition": acquisition,
    }
    write_json(final_path, body)
    save_checkpoint(checkpoint, model=model, optimizer=optimizer, torch=torch,
                    payload={**identity, "updates": updates, "real_tokens": real_tokens,
                             "trace": trace, "diagnostics": diagnostics, "structural": structural,
                             "initial_model_sha256": initial_model_sha, "complete": True,
                             "final_state": final_fp})
    del model, optimizer, backend
    gc.collect(); torch.cuda.empty_cache()
    return body


def collect_completed_arms(out: Path) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for i in range(len(core.MODEL_SEEDS)):
        for arm in core.ARMS:
            label = core.arm_label(i, arm)
            body = read_json(out / "arms" / label / "R1C_ARM.json")
            if body is not None:
                arms[label] = body
    return arms


def package(out: Path, campaign: Mapping[str, Any], preregistration: Mapping[str, Any] | None,
            failure: Mapping[str, Any] | None) -> dict[str, Any]:
    complete = campaign.get("status") == "COMPLETE"
    bundle = out / (BUNDLE_NAME if complete else PARTIAL_BUNDLE_NAME)
    payload: dict[str, Any] = {
        "SESSION_MANIFEST.json": {
            "experiment": core.EXPERIMENT, "status": campaign.get("status"),
            "wall_seconds": campaign.get("wall_seconds"),
        },
        "PREREGISTRATION.json": dict(preregistration or {}),
        "ENVIRONMENT.json": campaign.get("environment", {}),
        "PREEXECUTION_GATE.json": read_json(out / "PREEXECUTION_GATE.json") or {},
        "CALIBRATION.json": read_json(out / "CALIBRATION.json") or {},
        "RESOLVED.json": read_json(out / "RESOLVED.json") or {},
        "DATA_RECEIPT.json": campaign.get("data_receipt", {}),
        "DECISION.json": campaign.get("decision", {}),
        "CAMPAIGN.json": dict(campaign),
    }
    for label, body in sorted(collect_completed_arms(out).items()):
        payload[f"ARMS/{label}.json"] = body
    if failure is not None:
        payload["FAILURE.json"] = dict(failure)
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, body in payload.items():
            zf.writestr(name, json.dumps(body, indent=2, sort_keys=True, default=str))
    digest = sha256_file(bundle)
    (bundle.with_suffix(bundle.suffix + ".sha256")).write_text(digest + "  " + bundle.name + "\n", encoding="utf-8")
    return {"path": str(bundle), "sha256": digest, "entries": sorted(payload)}


def run_campaign(*, repo: Path, out: Path, preregistration: Mapping[str, Any] | None,
                 torch: Any, device: Any) -> dict[str, Any]:
    gate = read_json(out / "PREEXECUTION_GATE.json")
    if not gate or gate.get("status") != "PASS":
        raise RuntimeError("R1C preexecution gate missing; run preflight first")
    resolved = read_json(out / "RESOLVED.json")
    if not resolved:
        raise RuntimeError("R1C resolved calibration missing")
    setup_reproducibility(torch)
    manifest = repo / "docs/cymek/experiments/CYR-GPU-011/ARK002B_TASK_MANIFEST.json"
    data = base.load_ark002b_manifest(manifest)
    if data["source_split_sha256"] != "0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236":
        raise RuntimeError("R1C data drift")
    battery = base.make_reasoning_battery(data)
    started = time.monotonic()
    hard_deadline = started + core.WALL_MINUTES * 60.0
    science_deadline = hard_deadline - core.PACKAGING_RESERVE_MINUTES * 60.0
    campaign: dict[str, Any] = {
        "schema": "anra-cyr-gpu014-r1c-campaign/v1", "experiment": core.EXPERIMENT,
        "status": "RUNNING", "environment": environment(torch, device),
        "resolved": resolved,
        "data_receipt": {
            "source_split_sha256": data["source_split_sha256"],
            "source_blob_sha": data["source_blob_sha"],
            "train": len(data["train"]), "dev_controller": len(data["dev_controller"]),
            "dev_measurement": len(data["dev_measurement"]),
        },
        "completed_before_session": sorted(collect_completed_arms(out)),
    }
    failure = None
    partial = False
    try:
        for seed_index in range(len(core.MODEL_SEEDS)):
            for arm in core.ARM_ORDERS[seed_index]:
                if time.monotonic() >= science_deadline - 90.0:
                    partial = True
                    break
                print(f"\n=== R1C {core.arm_label(seed_index, arm)} ===", flush=True)
                body = run_arm(
                    out=out, seed_index=seed_index, arm=arm, data=data, battery=battery,
                    torch=torch, device=device, deadline=science_deadline,
                    progress=lambda s: print(s, flush=True),
                )
                if body.get("status") == "PARTIAL_TIMEBOX" or body.get("acquisition", {}).get("status") == "PARTIAL_TIMEBOX":
                    partial = True
                    break
            if partial:
                break
        arms = collect_completed_arms(out)
        campaign["completed_arms"] = sorted(arms)
        campaign["completed_arm_count"] = len(arms)
        if len(arms) == len(core.MODEL_SEEDS) * len(core.ARMS):
            # Cross-arm physical-init equality is a scientific invariant.
            for i in range(len(core.MODEL_SEEDS)):
                shas = {arms[core.arm_label(i, a)]["initial_model_sha256"] for a in core.ARMS}
                if len(shas) != 1:
                    raise RuntimeError(f"R1C matched initialization drift in seed {i + 1}")
            campaign["decision"] = core.decision(arms)
            campaign["status"] = "COMPLETE"
        else:
            campaign["decision"] = {
                "verdict": "INCONCLUSIVE_PARTIAL_SESSION",
                "completed_arms": len(arms), "required_arms": len(core.MODEL_SEEDS) * len(core.ARMS),
                "instruction": "rerun the frozen notebook; exact checkpoints resume unfinished work",
                "pre500m_authorized": False, "training_500m_authorized": False,
            }
            campaign["status"] = "PARTIAL_SESSION"
    except Exception as exc:
        failure = {
            "schema": "anra-cyr-gpu014-r1c-failure/v1", "exception": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(),
            "wall_seconds": time.monotonic() - started,
        }
        campaign["status"] = "FAILED"
        campaign["decision"] = {
            "verdict": "INCONCLUSIVE_RUNTIME_FAILURE", "pre500m_authorized": False,
            "training_500m_authorized": False,
        }
    finally:
        campaign["wall_seconds"] = time.monotonic() - started
        write_json(out / "CAMPAIGN_RECEIPT.json", campaign)
        campaign["bundle"] = package(out, campaign, preregistration, failure)
        write_json(out / "CAMPAIGN_RECEIPT.json", campaign)
    if failure is not None:
        raise RuntimeError(f"R1C failed after packaging: {failure['message']}")
    return campaign


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preflight", "run", "all"], default="all")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("/content/drive/MyDrive/CYMEK/CYR-GPU-014-R1C"))
    parser.add_argument("--prereg", type=Path, default=None)
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CYR-GPU-014 R1C requires CUDA/T4-class execution")
    device = torch.device("cuda")
    prereg = read_json(args.prereg) if args.prereg else None
    if args.mode in {"preflight", "all"}:
        preflight(repo=args.repo, out=args.out, torch=torch, device=device)
    if args.mode in {"run", "all"}:
        result = run_campaign(repo=args.repo, out=args.out, preregistration=prereg, torch=torch, device=device)
        print("R1C STATUS:", result["status"], flush=True)
        print("R1C VERDICT:", result.get("decision", {}).get("verdict"), flush=True)
        print("R1C BUNDLE:", result["bundle"]["path"], flush=True)
        print("R1C BUNDLE SHA256:", result["bundle"]["sha256"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
