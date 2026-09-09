"""CYR-GPU-011 long exposure-matched capability-emergence runner."""
from __future__ import annotations

import gc
import hashlib
import json
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping

from anra_v5 import cyr_gpu006_run as legacy
from v5_experiments import cyr_gpu011 as core

BUNDLE_NAME = "CYMEK_GPU_RESEARCH_V11_RESULTS.zip"
production_tokenizer = legacy.production_tokenizer
write_json = legacy.write_json
read_json = legacy.read_json


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment(torch: Any, device: Any) -> dict[str, Any]:
    return {"schema": "anra-cyr-gpu011-environment/v1", "torch": torch.__version__,
            "device": str(device), "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30}


def _build_model(spec: Any, seed: int, *, torch: Any, device: Any):
    from v5_model.core import initialize
    model = initialize(spec, seed, torch_module=torch).to(device)
    expected = int(spec.parameter_receipt().total)
    actual = sum(p.numel() for p in model.parameters())
    if actual != expected:
        raise ValueError(f"live parameter count {actual} != receipt {expected}")
    return model


def _save_checkpoint(root: Path, name: str, *, model: Any, optimizer: Any,
                     torch: Any, counters: Mapping[str, Any]) -> dict[str, Any]:
    """Persist a research checkpoint and surface its byte identities in receipts."""
    path = root / "checkpoints" / name
    receipt = legacy._save_checkpoint(
        path, model=model, optimizer=optimizer, torch=torch, counters=counters
    )
    return {"path": str(path), **dict(receipt)}


def _checkpoint_path(checkpoint: str | Path | Mapping[str, Any]) -> Path:
    if isinstance(checkpoint, Mapping):
        value = checkpoint.get("path")
        if not value:
            raise ValueError("checkpoint receipt has no path")
        return Path(str(value))
    return Path(checkpoint)


def _flat_snapshot(model: Any, *, torch: Any) -> Any:
    return torch.cat([p.detach().float().cpu().reshape(-1) for p in model.parameters()])


def _relative_displacement(model: Any, initial_flat: Any, *, torch: Any) -> dict[str, float]:
    current = _flat_snapshot(model, torch=torch)
    delta = float((current - initial_flat).norm().item())
    base = float(initial_flat.norm().item())
    return {"absolute_l2": delta, "relative_l2": delta / max(base, 1e-12), "initial_l2": base}


def _generate_texts(model: Any, tokenizer: Any, rows: list[dict[str, Any]], *,
                    torch: Any, device: Any, special: Mapping[str, int],
                    batch_size: int = 32, max_new_tokens: int = 8) -> list[dict[str, Any]]:
    """Candidate-free greedy generation with the same 8-token cap as the capability gate."""
    from v5_model.core import packed_layout
    bos, eos, pad = int(special["bos_id"]), int(special["eos_id"]), int(special["pad_id"])
    buckets: dict[int, list[tuple[dict[str, Any], list[int]]]] = {}
    for row in rows:
        prompt = [bos, *tokenizer.encode(row["prompt"])]
        buckets.setdefault(len(prompt), []).append((row, prompt))
    outputs: list[dict[str, Any]] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _length, items in sorted(buckets.items()):
            for start in range(0, len(items), batch_size):
                chunk = items[start:start + batch_size]
                prompts = [p for _r, p in chunk]
                generated: list[list[int]] = [[] for _ in chunk]
                done = [False] * len(chunk)
                stops = ["MAX_TOKENS"] * len(chunk)
                for _ in range(max_new_tokens):
                    seqs = [prompt + gen for prompt, gen in zip(prompts, generated)]
                    width = max(len(seq) for seq in seqs)
                    padded = [seq + [eos] * (width - len(seq)) for seq in seqs]
                    tokens = torch.tensor(padded, dtype=torch.long, device=device)
                    segments = torch.zeros_like(tokens)
                    positions, mask = packed_layout(segments, torch_module=torch)
                    logits = model(tokens, positions.to(device), mask.to(device))[:, -1]
                    next_ids = torch.argmax(logits, dim=-1).tolist()
                    for i, nxt in enumerate(next_ids):
                        nxt = int(nxt)
                        if done[i]:
                            generated[i].append(eos)
                        elif nxt == eos:
                            generated[i].append(eos); done[i] = True; stops[i] = "EOS"
                        elif nxt == pad:
                            generated[i].append(pad); done[i] = True; stops[i] = "PAD"
                        else:
                            generated[i].append(nxt)
                    if all(done):
                        break
                for i, (row, _prompt) in enumerate(chunk):
                    raw = [tok for tok in generated[i] if tok not in (eos, pad)]
                    text = tokenizer.decode(raw) if raw else ""
                    outputs.append({"world_id": row["world_id"], "pair_id": row.get("pair_id"),
                                    "role": row.get("role"), "expected_delta": row.get("expected_delta"),
                                    "expected": row["answer"], "generated": text, "stop": stops[i],
                                    "exact": text == row["answer"]})
    if was_training:
        model.train()
    return outputs


def _score_texts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows); denom = max(total, 1)
    exact = valid_stop = numeric = 0
    abs_errors: list[float] = []
    for row in rows:
        is_exact = bool(row["exact"])
        exact += int(is_exact)
        valid_stop += int(is_exact and row["stop"] == "EOS")
        try:
            pred = int(str(row["generated"]).strip())
            target = int(str(row["expected"]).strip())
            numeric += 1; abs_errors.append(abs(pred - target))
        except Exception:
            pass
    return {"n": total, "content_exact": exact / denom,
            "complete_exact_with_valid_stop": valid_stop / denom,
            "numeric_rate": numeric / denom,
            "numeric_mae": (sum(abs_errors) / len(abs_errors)) if abs_errors else None}


def _prediction_receipt(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return auditable final predictions plus a canonical digest."""
    body = [dict(row) for row in rows]
    return {"schema": "anra-cyr-gpu011-predictions/v1", "count": len(body),
            "sha256": hashlib.sha256(_canonical(body)).hexdigest(), "rows": body}


def _score_locality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("pair_id")), {})[str(row.get("role"))] = row
    usable = relation_ok = both_exact = 0
    for pair in groups.values():
        if "base" not in pair or "counterfactual" not in pair:
            continue
        try:
            base = int(str(pair["base"]["generated"]).strip())
            cf = int(str(pair["counterfactual"]["generated"]).strip())
        except Exception:
            continue
        usable += 1
        delta = int(pair["base"].get("expected_delta") or 0)
        relation_ok += int(cf - base == delta)
        both_exact += int(bool(pair["base"]["exact"]) and bool(pair["counterfactual"]["exact"]))
    denom = max(len(groups), 1)
    return {"pairs": len(groups), "numeric_usable_pairs": usable,
            "numeric_usable_fraction": usable / denom,
            "counterfactual_relation_consistency": relation_ok / denom,
            "both_exact": both_exact / denom}


def _digit_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tens = ones = denom = 0
    for row in rows:
        expected = str(row["expected"]).strip(); generated = str(row["generated"]).strip()
        if len(expected) != 2:
            continue
        denom += 1
        if len(generated) == 2:
            tens += int(generated[0] == expected[0]); ones += int(generated[1] == expected[1])
    d = max(denom, 1)
    return {"two_digit_examples": denom, "tens_digit_exact": tens / d, "ones_digit_exact": ones / d}


def reasoning_battery(model: Any, tokenizer: Any, battery: Mapping[str, Any], *,
                      torch: Any, device: Any, special: Mapping[str, int],
                      include_verbal: bool, include_predictions: bool = False) -> dict[str, Any]:
    names = ["STANDARD", "COMMUTED", "LOCALITY", "CARRY", "TRIPLE_ADD", "THREE_DIGIT"]
    if include_verbal:
        names.append("VERBAL")
    result: dict[str, Any] = {"schema": "anra-cyr-gpu011-reasoning-battery/v2",
                              "manifest": battery["_manifest"], "verbal_evaluated": include_verbal,
                              "generation_max_new_tokens": 8}
    raw: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        raw[name] = _generate_texts(model, tokenizer, list(battery[name]), torch=torch,
                                    device=device, special=special)
        result[name] = _score_texts(raw[name])
    result["STANDARD"]["digit_accuracy"] = _digit_accuracy(raw["STANDARD"])
    result["LOCALITY"]["structural"] = _score_locality(raw["LOCALITY"])
    if not include_verbal:
        result["VERBAL"] = {"status": "NOT_APPLICABLE_COMPACT_VOCAB"}
    if include_predictions:
        result["candidate_free_predictions"] = {
            name: _prediction_receipt(raw[name]) for name in names
        }
    result["structural_flags"] = core.structural_flags(result)
    result["claim_note"] = "Diagnostics probe controlled structural transfer; they are not broad reasoning or AGI claims."
    return result


def _calibrate_one(*, regime: str, spec: Any, tokenizer: Any, special: Mapping[str, int],
                   batch_rows: int, train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]],
                   torch: Any, device: Any) -> dict[str, Any]:
    from v5_training.optimizer import build_adamw_optimizer
    if getattr(device, "type", None) != "cuda":
        raise ValueError("CYR-GPU-011 calibration requires CUDA")
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = _build_model(spec, 9911 + batch_rows, torch=torch, device=device)
        optimizer = build_adamw_optimizer(model, torch_module=torch, lr=core.CYR11_HIGH_LR,
                                          betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
        backend = legacy._make_backend(model=model, optimizer=optimizer, special=special,
                                       device=device, torch=torch, lr=core.CYR11_HIGH_LR)
        batch = [train_rows[i % len(train_rows)] for i in range(batch_rows)]
        data_sha = hashlib.sha256(_canonical([r["prompt"] + r["answer"] for r in train_rows])).hexdigest()
        cumulative = 0
        first = legacy._one_update(backend=backend, tokenizer=tokenizer, rows=batch, torch=torch,
                                   device=device, special=special, cumulative=cumulative, update=1,
                                   data_sha=data_sha)
        cumulative += int(first["counted"]["real_tokens"])
        torch.cuda.synchronize()
        timed_updates = 2; timed_tokens = 0
        started = time.monotonic()
        for i in range(timed_updates):
            rec = legacy._one_update(backend=backend, tokenizer=tokenizer, rows=batch, torch=torch,
                                     device=device, special=special, cumulative=cumulative, update=i + 2,
                                     data_sha=data_sha)
            cumulative += int(rec["counted"]["real_tokens"]); timed_tokens += int(rec["counted"]["real_tokens"])
        torch.cuda.synchronize(); train_wall = max(time.monotonic() - started, 1e-9)
        eval_subset = eval_rows[:min(32, len(eval_rows))]
        torch.cuda.synchronize(); eval_started = time.monotonic()
        legacy.generate_rates_batched(model, tokenizer, eval_subset, torch=torch, device=device,
                                      special=special, batch_size=min(32, batch_rows))
        torch.cuda.synchronize(); eval_wall = max(time.monotonic() - eval_started, 1e-9)
        return {"schema": "anra-cyr-gpu011-calibration/v1", "status": "PASS", "regime": regime,
                "batch_rows": batch_rows, "parameters": sum(p.numel() for p in model.parameters()),
                "training_updates_per_sec": timed_updates / train_wall,
                "semantic_rows_per_sec": timed_updates * batch_rows / train_wall,
                "training_real_tokens_per_sec": timed_tokens / train_wall,
                "generation_examples_per_sec": len(eval_subset) / eval_wall,
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
                "optimizer_step_included": True, "candidate_free_generation_included": True}
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        return {"schema": "anra-cyr-gpu011-calibration/v1", "status": "OOM", "regime": regime,
                "batch_rows": batch_rows, "error": str(exc)}
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def calibrate_all(*, production_tok: Any, production_special: Mapping[str, int],
                  compact_tok: Any, compact_special: Mapping[str, int],
                  production_spec: Any, compact_spec: Any,
                  train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]],
                  torch: Any, device: Any) -> dict[str, Any]:
    receipts: dict[str, Any] = {}
    for regime, spec, tokenizer, special in (
        ("COMPACT", compact_spec, compact_tok, compact_special),
        ("PRODUCTION", production_spec, production_tok, production_special),
    ):
        for batch in (64, 32, 16):
            key = core.calibration_key(regime, batch)
            receipts[key] = _calibrate_one(regime=regime, spec=spec, tokenizer=tokenizer,
                                           special=special, batch_rows=batch,
                                           train_rows=train_rows, eval_rows=eval_rows,
                                           torch=torch, device=device)
    return receipts


def _index_stream(*, torch: Any, seed: int, world_count: int, rows: int) -> Any:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randint(0, int(world_count), (int(rows),), generator=gen,
                         dtype=torch.int64, device="cpu")


def run_acquisition(*, label: str, model_seed: int, order_seed: int, spec: Any,
                    tokenizer: Any, special: Mapping[str, int], batch_rows: int,
                    data: Mapping[str, Any], battery: Mapping[str, Any],
                    torch: Any, device: Any, deadline: float, out: Path,
                    include_verbal: bool, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    from v5_training.optimizer import build_adamw_optimizer
    model = _build_model(spec, model_seed, torch=torch, device=device)
    optimizer = build_adamw_optimizer(model, torch_module=torch, lr=core.CYR11_HIGH_LR,
                                      betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    backend = legacy._make_backend(model=model, optimizer=optimizer, special=special,
                                   device=device, torch=torch, lr=core.CYR11_HIGH_LR)
    initial_flat = _flat_snapshot(model, torch=torch)
    train = list(data["train"]); controller_rows = list(data["dev_controller"])
    measurement_rows = list(data["dev_measurement"]); probe_rows = train[:100]
    stream = _index_stream(torch=torch, seed=order_seed, world_count=len(train),
                           rows=core.CYR11_MAX_UPDATES * batch_rows)
    stream_sha = hashlib.sha256(stream.numpy().tobytes()).hexdigest()
    data_sha = str(data["source_split_sha256"])
    updates = real_tokens = row_presentations = 0
    next_eval_rows = core.CYR11_EVAL_EVERY_ROW_PRESENTATIONS
    trace: list[dict[str, Any]] = []
    milestones: dict[str, Any] = {}
    streaks = {"M99": 0, "G50": 0, "G90": 0}
    first_cross = {"M99": None, "G50": None, "G90": None}
    confirms = {"M99": None, "G50": None, "G90": None}

    baseline_battery = reasoning_battery(model, tokenizer, battery, torch=torch, device=device,
                                         special=special, include_verbal=include_verbal)
    baseline = {"update": 0, "row_presentations": 0, "actual_real_tokens": 0,
                "reasoning_battery": baseline_battery,
                "relative_displacement": {"absolute_l2": 0.0, "relative_l2": 0.0,
                                          "initial_l2": float(initial_flat.norm().item())}}
    trace.append(baseline)

    while updates < core.CYR11_MAX_UPDATES and time.monotonic() < deadline:
        start = updates * batch_rows
        indices = stream[start:start + batch_rows].tolist()
        rows = [train[int(i)] for i in indices]
        receipt = legacy._one_update(backend=backend, tokenizer=tokenizer, rows=rows, torch=torch,
                                     device=device, special=special, cumulative=real_tokens,
                                     update=updates + 1, data_sha=data_sha)
        updates += 1; row_presentations += len(rows)
        real_tokens += int(receipt["counted"]["real_tokens"])
        if progress and (updates % max(1, 12_800 // batch_rows) == 0):
            progress(f"CYR11 {label}: update={updates} rows={row_presentations} "
                     f"ARK_exposure={row_presentations/core.CYR11_ARK_MAX_ROW_PRESENTATIONS:.3f}")
        if row_presentations < next_eval_rows and updates < core.CYR11_MAX_UPDATES:
            continue
        while next_eval_rows <= row_presentations:
            next_eval_rows += core.CYR11_EVAL_EVERY_ROW_PRESENTATIONS

        controller = legacy.generate_rates_batched(model, tokenizer, controller_rows,
                                                    torch=torch, device=device, special=special)
        measurement = legacy.generate_rates_batched(model, tokenizer, measurement_rows,
                                                     torch=torch, device=device, special=special)
        probe = legacy.generate_rates_batched(model, tokenizer, probe_rows,
                                               torch=torch, device=device, special=special)
        scores = {"M99": float(probe["complete_exact_with_valid_stop"]),
                  "G50": float(controller["complete_exact_with_valid_stop"]),
                  "G90": float(controller["complete_exact_with_valid_stop"])}
        thresholds = {"M99": 0.99, "G50": core.CYR11_G50, "G90": core.CYR11_G90}
        reasons: list[str] = []
        for key in ("M99", "G50", "G90"):
            hit = scores[key] >= thresholds[key]
            if hit and first_cross[key] is None:
                first_cross[key] = updates; reasons.append(key + "_ONSET")
            streaks[key] = streaks[key] + 1 if hit else 0
            if streaks[key] >= core.CYR11_CONFIRMATIONS and confirms[key] is None:
                confirms[key] = updates; reasons.append(key + "_CONFIRMED")
                milestones[key] = _save_checkpoint(out, key, model=model, optimizer=optimizer,
                    torch=torch, counters={"label": label, "model_seed": model_seed,
                    "order_seed": order_seed, "updates": updates, "row_presentations": row_presentations,
                    "actual_real_tokens": real_tokens, "stream_sha256": stream_sha})

        entry: dict[str, Any] = {"update": updates, "row_presentations": row_presentations,
            "ark_reference_exposure_fraction": row_presentations / core.CYR11_ARK_MAX_ROW_PRESENTATIONS,
            "actual_real_tokens": real_tokens, "train_probe": probe,
            "dev_controller": controller, "dev_measurement": measurement,
            "milestone_events": reasons}
        periodic = row_presentations in {core.CYR11_EVAL_EVERY_ROW_PRESENTATIONS,
                                        5 * core.CYR11_EVAL_EVERY_ROW_PRESENTATIONS,
                                        25 * core.CYR11_EVAL_EVERY_ROW_PRESENTATIONS,
                                        50 * core.CYR11_EVAL_EVERY_ROW_PRESENTATIONS,
                                        75 * core.CYR11_EVAL_EVERY_ROW_PRESENTATIONS}
        if reasons or periodic or updates == core.CYR11_MAX_UPDATES:
            entry["reasoning_battery"] = reasoning_battery(model, tokenizer, battery, torch=torch,
                device=device, special=special, include_verbal=include_verbal)
            entry["relative_displacement"] = _relative_displacement(model, initial_flat, torch=torch)
        trace.append(entry)
        write_json(out / "progress.json", {"schema": "anra-cyr-gpu011-progress/v1", "label": label,
            "updates": updates, "row_presentations": row_presentations, "actual_real_tokens": real_tokens,
            "first_cross": first_cross, "confirmed": confirms, "trace": trace, "milestones": milestones})
        if confirms["G90"] is not None:
            break

    final_battery = reasoning_battery(model, tokenizer, battery, torch=torch, device=device,
                                      special=special, include_verbal=include_verbal,
                                      include_predictions=True)
    final_controller_rows = _generate_texts(model, tokenizer, controller_rows, torch=torch,
                                            device=device, special=special)
    final_controller = {**_score_texts(final_controller_rows),
                        "prediction_receipt": _prediction_receipt(final_controller_rows)}
    final_displacement = _relative_displacement(model, initial_flat, torch=torch)
    final_checkpoint = _save_checkpoint(out, "FINAL", model=model, optimizer=optimizer, torch=torch,
        counters={"label": label, "model_seed": model_seed, "order_seed": order_seed,
                  "updates": updates, "row_presentations": row_presentations,
                  "actual_real_tokens": real_tokens, "stream_sha256": stream_sha,
                  "g90_confirm_update": confirms["G90"]})
    status = "G90_CONFIRMED" if confirms["G90"] is not None else (
        "MAX_UPDATES_NO_G90" if updates >= core.CYR11_MAX_UPDATES else "TIMEBOX_NO_G90")
    receipt = {"schema": "anra-cyr-gpu011-acquisition/v2", "label": label, "status": status,
        "model_seed": model_seed, "order_seed": order_seed, "batch_rows": batch_rows,
        "updates": updates, "row_presentations": row_presentations,
        "ark_reference_max_row_presentations": core.CYR11_ARK_MAX_ROW_PRESENTATIONS,
        "ark_exposure_fraction": row_presentations / core.CYR11_ARK_MAX_ROW_PRESENTATIONS,
        "actual_real_tokens": real_tokens, "first_cross": first_cross, "confirmed": confirms,
        "m99_confirm_update": confirms["M99"], "g50_confirm_update": confirms["G50"],
        "g90_confirm_update": confirms["G90"], "semantic_stream_sha256": stream_sha,
        "trace": trace, "dev_controller_final": final_controller,
        "reasoning_battery_final": final_battery,
        "structural_flags_final": final_battery["structural_flags"],
        "relative_displacement_final": final_displacement,
        "milestone_checkpoints": milestones, "final_checkpoint": final_checkpoint}
    write_json(out / "acquisition.json", receipt)
    return receipt


def _measure_sealed(*, checkpoint: str | Path | Mapping[str, Any], spec: Any, seed: int,
                    tokenizer: Any, special: Mapping[str, int], rows: list[dict[str, Any]],
                    torch: Any, device: Any) -> dict[str, Any]:
    from v5_training.optimizer import build_adamw_optimizer
    model = _build_model(spec, seed, torch=torch, device=device)
    optimizer = build_adamw_optimizer(model, torch_module=torch, lr=core.CYR11_HIGH_LR,
                                      betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    checkpoint_path = _checkpoint_path(checkpoint)
    legacy._load_checkpoint(checkpoint_path, model=model, optimizer=optimizer, torch=torch)
    generated = _generate_texts(model, tokenizer, rows, torch=torch, device=device, special=special)
    return {"status": "MEASURED_AFTER_DECISION", "score": _score_texts(generated),
            "prediction_receipt": _prediction_receipt(generated),
            "checkpoint": dict(checkpoint) if isinstance(checkpoint, Mapping) else {"path": str(checkpoint_path)}}


def _package(out: Path, *, campaign: Mapping[str, Any], preregistration: Mapping[str, Any],
             failure: Mapping[str, Any] | None) -> dict[str, Any]:
    bundle = out / BUNDLE_NAME
    payload = {
        "SESSION_MANIFEST.json": {"experiment": core.CYR11_ID, "status": campaign.get("status"),
                                  "wall_seconds": campaign.get("wall_seconds")},
        "ENVIRONMENT.json": campaign.get("environment", {}),
        "PREREGISTRATION.json": dict(preregistration),
        "RESOLVED_PREREGISTRATION.json": campaign.get("resolved", {}),
        "CALIBRATION.json": campaign.get("calibrations", {}),
        "ARK002B_MANIFEST_RECEIPT.json": campaign.get("data_receipt", {}),
        "REASONING_BATTERY_MANIFEST.json": campaign.get("reasoning_battery_manifest", {}),
        "COMPACT_BRIDGE.json": campaign.get("compact_bridge", {}),
        "PRODUCTION_PRIMARY.json": campaign.get("production_primary", {}),
        "PRODUCTION_REPLICATION.json": campaign.get("production_replication", {}),
        "SEALED.json": campaign.get("sealed", {}),
        "DECISION.json": campaign.get("decision", {}),
    }
    if failure is not None:
        payload["FAILURE.json"] = dict(failure)
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, body in payload.items():
            archive.writestr(name, json.dumps(body, indent=2, sort_keys=True, default=str))
    return {"path": str(bundle), "sha256": _sha256(bundle), "entries": sorted(payload)}


def run_campaign(*, repo: Path, out: Path, preregistration: Mapping[str, Any],
                 resolved: Mapping[str, Any], calibrations: Mapping[str, Any],
                 torch: Any = None, device: Any = None,
                 progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    if torch is None:
        import torch as torch_module
        torch = torch_module
    if not torch.cuda.is_available():
        raise RuntimeError("CYR-GPU-011 full experiment requires CUDA")
    if device is None:
        device = torch.device("cuda")
    if getattr(device, "type", None) != "cuda":
        raise RuntimeError("CYR-GPU-011 refuses non-CUDA scientific execution")
    resolved = core.validate_resolved(resolved)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    hard_deadline = started + float(resolved["wall_budget_minutes"]) * 60.0
    science_deadline = hard_deadline - float(resolved["packaging_reserve_minutes"]) * 60.0
    campaign: dict[str, Any] = {"schema": "anra-cyr-gpu011-campaign/v2", "experiment": core.CYR11_ID,
        "status": "RUNNING", "environment": _environment(torch, device),
        "resolved": dict(resolved), "calibrations": dict(calibrations),
        "arkenstone_audited_sha": core.CYR11_ARKENSTONE_AUDIT_SHA,
        "bramastra_audited_sha": core.CYR11_BRAMASTRA_AUDIT_SHA,
        "v9_bundle_sha256": core.CYR11_V9_BUNDLE_SHA256}
    failure = None
    try:
        prod_tok, prod_identity = production_tokenizer(Path(repo))
        expected_tok = preregistration.get("tokenizer", {}).get("artifact_sha256")
        if expected_tok and expected_tok != prod_identity["artifact_sha256"]:
            raise ValueError("production tokenizer drift")
        prod_special = {"pad_id": prod_identity["pad_id"], "bos_id": prod_identity["bos_id"],
                        "eos_id": prod_identity["eos_id"]}
        compact_tok = core.CompactCharTokenizer(); compact_special = compact_tok.special

        manifest_path = Path(repo) / "docs/cymek/experiments/CYR-GPU-011/ARK002B_TASK_MANIFEST.json"
        data = core.load_ark002b_manifest(manifest_path)
        expected_split = preregistration.get("data", {}).get("split_sha256")
        if expected_split and expected_split != data["source_split_sha256"]:
            raise ValueError("ARK-002B manifest split drift")
        campaign["data_receipt"] = {"source_split_sha256": data["source_split_sha256"],
            "source_blob_sha": data["source_blob_sha"], "role_sha256": data["role_sha256"],
            "train": len(data["train"]), "dev_controller": len(data["dev_controller"]),
            "dev_measurement": len(data["dev_measurement"]), "sealed_reserved": len(data["sealed_reserved"]),
            "train_test_canonical_overlap": data["train_test_canonical_overlap"]}
        battery = core.make_reasoning_battery(data)
        campaign["reasoning_battery_manifest"] = battery["_manifest"]

        compact_spec = core.research_small_spec(compact_tok.vocabulary_size)
        production_spec = core.research_small_spec(prod_identity["vocabulary_size"])

        compact_deadline = min(science_deadline,
            started + float(resolved["compact_stage_cap_minutes"]) * 60.0)
        compact = run_acquisition(label="COMPACT_BRIDGE", model_seed=core.CYR11_MODEL_SEEDS["compact"],
            order_seed=core.CYR11_ORDER_SEEDS["compact"], spec=compact_spec,
            tokenizer=compact_tok, special=compact_special,
            batch_rows=int(resolved["compact_batch_rows"]), data=data, battery=battery,
            torch=torch, device=device, deadline=compact_deadline, out=out / "compact_bridge",
            include_verbal=False, progress=progress)
        campaign["compact_bridge"] = compact

        production_primary = None; production_replication = None
        if bool(resolved.get("production_available")) and time.monotonic() < science_deadline - 60.0:
            production_primary = run_acquisition(label="PRODUCTION_PRIMARY",
                model_seed=core.CYR11_MODEL_SEEDS["production_primary"],
                order_seed=core.CYR11_ORDER_SEEDS["production_primary"], spec=production_spec,
                tokenizer=prod_tok, special=prod_special,
                batch_rows=int(resolved["production_batch_rows"]), data=data, battery=battery,
                torch=torch, device=device, deadline=science_deadline,
                out=out / "production_primary", include_verbal=True, progress=progress)
        campaign["production_primary"] = production_primary or {"status": "NOT_RUN_HARDWARE_OR_WALL"}

        remaining_minutes = max(0.0, (science_deadline - time.monotonic()) / 60.0)
        primary_measurement = float((production_primary or {}).get("reasoning_battery_final", {})
                                    .get("STANDARD", {}).get("complete_exact_with_valid_stop", 0.0))
        primary_qualified = bool(production_primary
                                 and production_primary.get("g90_confirm_update") is not None
                                 and primary_measurement >= core.CYR11_G90)
        if primary_qualified and remaining_minutes >= float(resolved["second_seed_launch_minutes"]):
            production_replication = run_acquisition(label="PRODUCTION_REPLICATION",
                model_seed=core.CYR11_MODEL_SEEDS["production_replication"],
                order_seed=core.CYR11_ORDER_SEEDS["production_replication"], spec=production_spec,
                tokenizer=prod_tok, special=prod_special,
                batch_rows=int(resolved["production_batch_rows"]), data=data, battery=battery,
                torch=torch, device=device, deadline=science_deadline,
                out=out / "production_replication", include_verbal=True, progress=progress)
        campaign["production_replication"] = production_replication or {"status": "NOT_LAUNCHED"}

        decision = core.final_decision(compact=compact, production_primary=production_primary,
                                       production_replication=production_replication)
        campaign["decision"] = decision
        sealed: dict[str, Any] = {"status": "NOT_MEASURED"}
        if production_primary and production_primary.get("final_checkpoint") and time.monotonic() < hard_deadline - 20.0:
            sealed = _measure_sealed(checkpoint=production_primary["final_checkpoint"],
                spec=production_spec, seed=core.CYR11_MODEL_SEEDS["production_primary"],
                tokenizer=prod_tok, special=prod_special, rows=list(data["sealed_reserved"]),
                torch=torch, device=device)
        campaign["sealed"] = sealed
        campaign["status"] = "COMPLETE"
    except Exception as exc:
        failure = {"schema": "anra-cyr-gpu011-failure/v1", "exception": type(exc).__name__,
                   "message": str(exc), "traceback": traceback.format_exc(),
                   "wall_seconds": time.monotonic() - started}
        campaign["status"] = "FAILED"
        campaign.setdefault("decision", {"verdict": "INCONCLUSIVE_RUNTIME_FAILURE",
            "production_promotion_authorized": False, "pre500m_authorized": False,
            "training_500m_authorized": False})
    finally:
        campaign["wall_seconds"] = time.monotonic() - started
        bundle = _package(out, campaign=campaign, preregistration=preregistration, failure=failure)
        campaign["bundle"] = bundle
        write_json(out / "campaign_receipt.json", campaign)
    if failure is not None:
        raise RuntimeError(f"CYR-GPU-011 failed after packaging: {failure['message']}")
    return campaign
