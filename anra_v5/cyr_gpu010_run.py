"""CYR-GPU-010 long capability-emergence runner.

One fixed-wall session first answers the scale/dose question on Cymek's real
4L/128w RESEARCH_SMALL V5 proxy. Candidate-free G90 is monitored every 200
optimizer updates, matching the temporal resolution used by Arkenstone's
memorize->generalize studies. Structural diagnostics are measurement-only.
If G90 is confirmed with enough wall remaining, an optional matched stress
compares canonical-only HIGH continuation with 1/16 commuted-render replay at
the same HIGH LR and identical semantic-example stream.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping

from anra_v5 import cyr_gpu006_run as legacy
from v5_experiments import cyr_gpu010 as core

BUNDLE_NAME = "CYMEK_GPU_RESEARCH_V10_RESULTS.zip"
production_tokenizer = legacy.production_tokenizer
write_json = legacy.write_json
read_json = legacy.read_json


def calibrate_candidates(*, registry: Mapping[str, Mapping[str, Any]], tokenizer: Any,
                         torch: Any, device: Any, special: Mapping[str, int],
                         train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    receipts: dict[str, Any] = {}
    for name in ("RESEARCH_SMALL", "TINY"):
        receipts[name] = legacy.calibrate_candidate(
            name=name, spec=registry[name]["spec"], tokenizer=tokenizer,
            torch=torch, device=device, special=special,
            train_rows=train_rows, eval_rows=eval_rows)
    return receipts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _environment(torch: Any, device: Any) -> dict[str, Any]:
    return {"schema": "anra-cyr-gpu010-environment/v1", "torch": torch.__version__,
            "device": str(device), "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30}


def _deterministic_index_stream(*, seed: int, world_count: int, rows: int) -> list[int]:
    rng = random.Random(seed); stream: list[int] = []
    while len(stream) < rows:
        epoch = list(range(world_count)); rng.shuffle(epoch); stream.extend(epoch)
    return stream[:rows]


def _generate_texts(model: Any, tokenizer: Any, rows: list[dict[str, Any]], *,
                    torch: Any, device: Any, special: Mapping[str, int],
                    batch_size: int = 32, max_new_tokens: int = 10) -> list[dict[str, Any]]:
    from v5_model.core import packed_layout
    bos, eos, pad = special["bos_id"], special["eos_id"], special["pad_id"]
    buckets: dict[int, list[tuple[dict[str, Any], list[int]]]] = {}
    for row in rows:
        prompt = [bos, *tokenizer.encode(row["prompt"])]
        buckets.setdefault(len(prompt), []).append((row, prompt))
    outputs: list[dict[str, Any]] = []; was_training = model.training; model.eval()
    with torch.no_grad():
        for _length, items in sorted(buckets.items()):
            for start in range(0, len(items), batch_size):
                chunk = items[start:start + batch_size]; prompts = [prompt for _row, prompt in chunk]
                generated: list[list[int]] = [[] for _ in chunk]; done = [False] * len(chunk)
                stops = ["MAX_TOKENS"] * len(chunk)
                for _step in range(max_new_tokens):
                    seqs = [prompt + gen for prompt, gen in zip(prompts, generated)]
                    width = max(len(seq) for seq in seqs)
                    padded = [seq + [eos] * (width - len(seq)) for seq in seqs]
                    tokens = torch.tensor(padded, dtype=torch.long, device=device)
                    segments = torch.zeros_like(tokens)
                    positions, mask = packed_layout(segments, torch_module=torch)
                    logits = model(tokens, positions.to(device), mask.to(device))[:, -1]
                    next_ids = torch.argmax(logits, dim=-1).tolist()
                    for i, nxt in enumerate(next_ids):
                        if done[i]: generated[i].append(eos)
                        elif int(nxt) == eos: done[i] = True; stops[i] = "EOS"; generated[i].append(eos)
                        elif int(nxt) == pad: done[i] = True; stops[i] = "PAD"; generated[i].append(pad)
                        else: generated[i].append(int(nxt))
                    if all(done): break
                for i, (row, _prompt) in enumerate(chunk):
                    raw = [tok for tok in generated[i] if tok not in (eos, pad)]
                    text = tokenizer.decode(raw) if raw else ""
                    outputs.append({"world_id": row["world_id"], "pair_id": row.get("pair_id"),
                                    "role": row.get("role"), "expected_delta": row.get("expected_delta"),
                                    "expected": row["answer"], "generated": text, "stop": stops[i],
                                    "exact": text == row["answer"]})
    if was_training: model.train()
    return outputs


def _score_texts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(len(rows), 1); exact = sum(bool(r["exact"]) for r in rows)
    valid_stop = sum(bool(r["exact"]) and r["stop"] == "EOS" for r in rows); numeric = 0
    for r in rows:
        try: int(r["generated"].strip()); numeric += 1
        except Exception: pass
    return {"n": len(rows), "content_exact": exact / n,
            "complete_exact_with_valid_stop": valid_stop / n, "numeric_rate": numeric / n}


def _score_locality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows: groups.setdefault(str(row.get("pair_id")), {})[str(row.get("role"))] = row
    relation_ok = both_exact = usable = 0
    for pair in groups.values():
        if "base" not in pair or "counterfactual" not in pair: continue
        try:
            base = int(pair["base"]["generated"].strip()); cf = int(pair["counterfactual"]["generated"].strip())
        except Exception: continue
        usable += 1; delta = int(pair["base"].get("expected_delta") or 0)
        relation_ok += int(cf - base == delta); both_exact += int(pair["base"]["exact"] and pair["counterfactual"]["exact"])
    denom = max(len(groups), 1)
    return {"pairs": len(groups), "numeric_usable_pairs": usable,
            "counterfactual_relation_consistency": relation_ok / denom, "both_exact": both_exact / denom}


def _digit_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    tens = ones = denom = 0
    for row in rows:
        expected = str(row["expected"]).strip(); generated = str(row["generated"]).strip()
        if len(expected) != 2: continue
        denom += 1
        if len(generated) == 2:
            tens += int(generated[0] == expected[0]); ones += int(generated[1] == expected[1])
    d = max(denom, 1)
    return {"two_digit_examples": denom, "tens_digit_exact": tens / d, "ones_digit_exact": ones / d}


def reasoning_battery(model: Any, tokenizer: Any, battery: Mapping[str, Any], *,
                      torch: Any, device: Any, special: Mapping[str, int]) -> dict[str, Any]:
    result: dict[str, Any] = {"schema": "anra-cyr-gpu010-reasoning-battery/v1", "manifest": battery["_manifest"]}
    raw: dict[str, list[dict[str, Any]]] = {}
    for name in ("STANDARD", "COMMUTED", "LOCALITY", "RENDERING", "THREE_DIGIT"):
        raw[name] = _generate_texts(model, tokenizer, list(battery[name]), torch=torch, device=device, special=special)
        result[name] = _score_texts(raw[name])
    result["STANDARD"]["digit_accuracy"] = _digit_accuracy(raw["STANDARD"])
    result["LOCALITY"]["structural"] = _score_locality(raw["LOCALITY"])
    result["claim_note"] = "STANDARD is the held-out capability measure; all other sets are structural diagnostics, not AGI claims."
    return result


def _save_named_checkpoint(root: Path, name: str, *, model: Any, optimizer: Any,
                           torch: Any, counters: Mapping[str, Any]) -> str:
    path = root / "checkpoints" / name
    legacy._save_checkpoint(path, model=model, optimizer=optimizer, torch=torch, counters=counters)
    return str(path)


def acquire(*, seed: int, spec: Any, proxy_name: str, tokenizer: Any, torch: Any,
            device: Any, special: Mapping[str, int], splits: Mapping[str, list[dict[str, Any]]],
            battery: Mapping[str, Any], max_updates: int, eval_every: int,
            deadline: float, out: Path, progress: Callable[[str], None] | None) -> dict[str, Any]:
    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer
    model = initialize(spec, seed, torch_module=torch).to(device)
    core.assert_proxy_in_registry(proxy_name, sum(p.numel() for p in model.parameters()), core.proxy_registry(vocab_size=spec.vocabulary_size))
    optimizer = build_adamw_optimizer(model, torch_module=torch, lr=core.CYR10_HIGH_LR)
    backend = legacy._make_backend(model=model, optimizer=optimizer, special=special, device=device, torch=torch, lr=core.CYR10_HIGH_LR)
    train = list(splits["train"])
    stream = _deterministic_index_stream(seed=seed + 9100, world_count=len(train), rows=max_updates * core.CYR10_BATCH_ROWS)
    stream_sha = hashlib.sha256(_canonical(stream)).hexdigest()
    data_sha = hashlib.sha256(_canonical([row["prompt"] + row["answer"] for row in train])).hexdigest()
    updates = actual_tokens = 0; g50_onset = g90_onset = g90_confirm = m99_onset = None
    g90_flags: list[bool] = []; trace: list[dict[str, Any]] = []; milestone_checkpoints: dict[str, str] = {}
    last_battery: dict[str, Any] | None = None
    while updates < max_updates and time.monotonic() < deadline:
        start = updates * core.CYR10_BATCH_ROWS; indices = stream[start:start + core.CYR10_BATCH_ROWS]
        rows = [train[i] for i in indices]
        receipt = legacy._one_update(backend=backend, tokenizer=tokenizer, rows=rows, torch=torch, device=device,
                                     special=special, cumulative=actual_tokens, update=updates + 1, data_sha=data_sha)
        updates += 1; actual_tokens += int(receipt["counted"]["real_tokens"])
        if progress and updates % 200 == 0: progress(f"CYR10 {proxy_name} seed {seed}: update {updates}/{max_updates}, real_tokens={actual_tokens}")
        if updates % eval_every != 0 and updates != max_updates: continue
        controller = legacy.generate_rates_batched(model, tokenizer, list(splits["dev_controller"]), torch=torch, device=device, special=special)
        measurement = legacy.generate_rates_batched(model, tokenizer, list(splits["dev_measurement"]), torch=torch, device=device, special=special)
        probe = legacy.generate_rates_batched(model, tokenizer, train[:16], torch=torch, device=device, special=special)
        ctrl = float(controller["complete_exact_with_valid_stop"]); train_score = float(probe["complete_exact_with_valid_stop"])
        if m99_onset is None and train_score >= 0.99:
            m99_onset = updates; milestone_checkpoints["M99"] = _save_named_checkpoint(out, "M99", model=model, optimizer=optimizer, torch=torch, counters={"seed": seed, "updates": updates, "actual_real_tokens": actual_tokens, "stream_sha256": stream_sha})
        if g50_onset is None and ctrl >= core.CYR10_G50:
            g50_onset = updates; milestone_checkpoints["G50"] = _save_named_checkpoint(out, "G50", model=model, optimizer=optimizer, torch=torch, counters={"seed": seed, "updates": updates, "actual_real_tokens": actual_tokens, "stream_sha256": stream_sha})
        if g90_onset is None and ctrl >= core.CYR10_G90: g90_onset = updates
        g90_flags.append(ctrl >= core.CYR10_G90)
        if len(g90_flags) >= core.CYR10_CONFIRMATIONS and all(g90_flags[-core.CYR10_CONFIRMATIONS:]) and g90_confirm is None:
            g90_confirm = updates; milestone_checkpoints["G90"] = _save_named_checkpoint(out, "G90", model=model, optimizer=optimizer, torch=torch, counters={"seed": seed, "updates": updates, "actual_real_tokens": actual_tokens, "stream_sha256": stream_sha})
        entry = {"update": updates, "actual_real_tokens": actual_tokens, "train_probe": probe, "dev_controller": controller, "dev_measurement": measurement}
        if updates in {200, 1000} or updates == m99_onset or updates == g50_onset or updates == g90_confirm or updates == max_updates:
            last_battery = reasoning_battery(model, tokenizer, battery, torch=torch, device=device, special=special); entry["reasoning_battery"] = last_battery
        trace.append(entry)
        write_json(out / "acquisition_progress.json", {"schema": "anra-cyr-gpu010-acquisition-progress/v1", "seed": seed,
                   "updates": updates, "actual_real_tokens": actual_tokens, "g90_confirm_update": g90_confirm,
                   "trace": trace, "milestone_checkpoints": milestone_checkpoints})
        if g90_confirm is not None: break
    if last_battery is None or (trace and "reasoning_battery" not in trace[-1]):
        last_battery = reasoning_battery(model, tokenizer, battery, torch=torch, device=device, special=special)
    final_checkpoint = _save_named_checkpoint(out, "ACQUISITION_FINAL", model=model, optimizer=optimizer, torch=torch,
        counters={"seed": seed, "updates": updates, "actual_real_tokens": actual_tokens, "stream_sha256": stream_sha, "g90_confirm_update": g90_confirm})
    standard_final = float(last_battery["STANDARD"]["complete_exact_with_valid_stop"])
    locality_final = float(last_battery["LOCALITY"]["structural"]["counterfactual_relation_consistency"])
    return {"schema": "anra-cyr-gpu010-acquisition/v1", "seed": seed, "proxy": proxy_name,
            "status": "G90_CONFIRMED" if g90_confirm is not None else ("TIMEBOX" if time.monotonic() >= deadline else "NO_G90"),
            "updates": updates, "actual_real_tokens": actual_tokens, "m99_onset_update": m99_onset,
            "g50_onset_update": g50_onset, "g90_onset_update": g90_onset, "g90_confirm_update": g90_confirm,
            "trace": trace, "reasoning_battery_final": last_battery, "milestone_checkpoints": milestone_checkpoints,
            "final_checkpoint": final_checkpoint, "train_stream_sha256": stream_sha,
            "decision": core.classify_acquisition(g90_confirm_update=g90_confirm, final_standard=standard_final,
                                                   final_locality=locality_final, proxy=proxy_name, updates=updates)}


def _stress_render(rows: list[dict[str, Any]], *, arm: str, step: int) -> list[dict[str, Any]]:
    rendered = [dict(row) for row in rows]
    if arm == "SUPPORT_HIGH_1OF16":
        pos = step % len(rendered); row = rendered[pos]
        rendered[pos] = {**row, "prompt": f"{row['b']} + {row['a']} = ", "world_id": row["world_id"] + "/commuted-support"}
    return rendered


def run_stress(*, acquisition: Mapping[str, Any], steps_per_arm: int, spec: Any,
               tokenizer: Any, torch: Any, device: Any, special: Mapping[str, int],
               splits: Mapping[str, list[dict[str, Any]]], battery: Mapping[str, Any],
               deadline: float, out: Path, progress: Callable[[str], None] | None) -> dict[str, Any]:
    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer
    parent_path = acquisition.get("milestone_checkpoints", {}).get("G90")
    if not parent_path or steps_per_arm <= 0:
        return {"schema": "anra-cyr-gpu010-stress/v1", "status": "NOT_RUN", "reason": "no G90 parent or insufficient remaining wall"}
    train = list(splits["train"])
    semantic = _deterministic_index_stream(seed=5510, world_count=len(train), rows=steps_per_arm * core.CYR10_BATCH_ROWS)
    semantic_sha = hashlib.sha256(_canonical(semantic)).hexdigest(); results: dict[str, Any] = {}
    for arm in ("NARROW_HIGH", "SUPPORT_HIGH_1OF16"):
        if time.monotonic() >= deadline: results[arm] = {"status": "TIMEBOX", "updates": 0}; continue
        model = initialize(spec, int(acquisition["seed"]), torch_module=torch).to(device)
        optimizer = build_adamw_optimizer(model, torch_module=torch, lr=core.CYR10_HIGH_LR)
        legacy._load_checkpoint(Path(parent_path), model=model, optimizer=optimizer, torch=torch)
        backend = legacy._make_backend(model=model, optimizer=optimizer, special=special, device=device, torch=torch, lr=core.CYR10_HIGH_LR)
        data_sha = hashlib.sha256(_canonical([row["prompt"] + row["answer"] for row in train])).hexdigest()
        actual_tokens = 0; trace = []; consumed_shas = []; executed = 0
        for step in range(steps_per_arm):
            if time.monotonic() >= deadline: break
            start = step * core.CYR10_BATCH_ROWS; idx = semantic[start:start + core.CYR10_BATCH_ROWS]
            consumed_shas.append(hashlib.sha256(_canonical(idx)).hexdigest())
            rows = _stress_render([train[i] for i in idx], arm=arm, step=step)
            receipt = legacy._one_update(backend=backend, tokenizer=tokenizer, rows=rows, torch=torch, device=device,
                                         special=special, cumulative=actual_tokens, update=step + 1, data_sha=data_sha)
            actual_tokens += int(receipt["counted"]["real_tokens"]); executed += 1
            if executed % core.CYR10_STRESS_EVAL_EVERY == 0 or executed == steps_per_arm:
                batt = reasoning_battery(model, tokenizer, battery, torch=torch, device=device, special=special)
                trace.append({"update": executed, "actual_real_tokens": actual_tokens, "reasoning_battery": batt})
            if progress and executed % 200 == 0: progress(f"CYR10 stress {arm}: {executed}/{steps_per_arm}")
        final_cp = _save_named_checkpoint(out / "stress" / arm, "FINAL", model=model, optimizer=optimizer, torch=torch,
                                          counters={"arm": arm, "updates": executed, "actual_real_tokens": actual_tokens, "semantic_stream_sha256": semantic_sha})
        results[arm] = {"status": "COMPLETE" if executed == steps_per_arm else "TIMEBOX", "updates": executed,
                        "actual_real_tokens": actual_tokens, "semantic_stream_sha256": semantic_sha,
                        "consumed_batch_shas": consumed_shas, "trace": trace, "final_checkpoint": final_cp}
    compared = min((len(r.get("consumed_batch_shas", [])) for r in results.values()), default=0)
    matched = bool(compared and all(results[a].get("consumed_batch_shas", [])[:compared] == results["NARROW_HIGH"].get("consumed_batch_shas", [])[:compared] for a in results))
    return {"schema": "anra-cyr-gpu010-stress/v1", "status": "COMPLETE" if all(r.get("status") == "COMPLETE" for r in results.values()) else "PARTIAL",
            "steps_per_arm": steps_per_arm, "semantic_stream_sha256": semantic_sha, "matched_semantic_batches": matched,
            "batches_compared": compared, "arms": results,
            "interpretation_boundary": "exploratory single-parent Cymek stress informed by ARK-015; only rendering support differs; no production claim"}


def _package(out: Path, *, campaign: Mapping[str, Any], preregistration: Mapping[str, Any], failure: Mapping[str, Any] | None) -> dict[str, Any]:
    bundle = out / BUNDLE_NAME
    payload = {"SESSION_MANIFEST.json": {"experiment": core.CYR10_ID, "status": campaign.get("status"), "wall_seconds": campaign.get("wall_seconds")},
               "ENVIRONMENT.json": campaign.get("environment", {}), "PREREGISTRATION.json": dict(preregistration),
               "RESOLVED_PREREGISTRATION.json": campaign.get("resolved", {}), "CALIBRATION.json": campaign.get("calibrations", {}),
               "DATA_MANIFEST.json": campaign.get("data_manifest", {}), "REASONING_BATTERY_MANIFEST.json": campaign.get("reasoning_battery_manifest", {}),
               "ACQUISITION.json": campaign.get("acquisition", {}), "STRESS.json": campaign.get("stress", {}),
               "SEALED.json": campaign.get("sealed", {}), "DECISION.json": campaign.get("decision", {})}
    if failure is not None: payload["FAILURE.json"] = dict(failure)
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, body in payload.items(): archive.writestr(name, json.dumps(body, indent=2, sort_keys=True, default=str))
    return {"path": str(bundle), "sha256": _sha256(bundle), "entries": sorted(payload)}


def run_campaign(*, repo: Path, out: Path, preregistration: Mapping[str, Any], resolved: Mapping[str, Any],
                 calibrations: Mapping[str, Any], torch: Any = None, device: Any = None,
                 progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    if torch is None:
        import torch as torch_module; torch = torch_module
    if not torch.cuda.is_available(): raise RuntimeError("CYR-GPU-010 requires CUDA")
    if device is None: device = torch.device("cuda")
    if getattr(device, "type", None) != "cuda": raise RuntimeError("CYR-GPU-010 refuses non-CUDA scientific execution")
    resolved = core.validate_resolved(resolved); selected = calibrations.get(resolved["proxy"], {})
    if selected.get("status") != "PASS": raise ValueError("selected proxy lacks passing calibration")
    out = Path(out); out.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    hard_deadline = started + float(resolved["wall_budget_minutes"]) * 60.0
    science_deadline = hard_deadline - float(resolved["packaging_reserve_minutes"]) * 60.0
    campaign: dict[str, Any] = {"schema": "anra-cyr-gpu010-campaign/v1", "experiment": core.CYR10_ID,
        "status": "RUNNING", "resolved": dict(resolved), "calibrations": dict(calibrations),
        "environment": _environment(torch, device), "arkenstone_audited_sha": core.CYR10_ARKENSTONE_SHA,
        "v9_bundle_sha256": core.CYR10_V9_BUNDLE_SHA256}
    failure = None
    try:
        tokenizer, identity = production_tokenizer(Path(repo)); expected_tok = preregistration.get("tokenizer", {}).get("artifact_sha256")
        if expected_tok and expected_tok != identity["artifact_sha256"]: raise ValueError("tokenizer drift")
        special = {"pad_id": identity["pad_id"], "bos_id": identity["bos_id"], "eos_id": identity["eos_id"]}
        splits = core.render_t2_worlds(); manifest = core.build_data_manifest(splits); core.assert_manifest_sha(manifest)
        expected_manifest = preregistration.get("data", {}).get("data_manifest_sha256_full")
        if expected_manifest and manifest["sha256"] != expected_manifest: raise ValueError("rendered data manifest drift")
        leak = core.commutation_audit(splits, tv_bound=0.20)
        if not leak["commutation_free"]: raise ValueError(f"data audit failed: {leak['findings']}")
        battery = core.make_reasoning_battery(splits); campaign["data_manifest"] = manifest; campaign["reasoning_battery_manifest"] = battery["_manifest"]
        registry = core.proxy_registry(vocab_size=identity["vocabulary_size"]); spec = registry[resolved["proxy"]]["spec"]
        acquisition = acquire(seed=int(resolved["parent_seed"]), spec=spec, proxy_name=resolved["proxy"], tokenizer=tokenizer,
            torch=torch, device=device, special=special, splits=splits, battery=battery,
            max_updates=int(resolved["max_acquisition_updates"]), eval_every=int(resolved["eval_every_updates"]),
            deadline=science_deadline, out=out, progress=progress)
        campaign["acquisition"] = acquisition; campaign["decision"] = acquisition["decision"]
        stress = {"status": "NOT_RUN", "reason": "G90 not confirmed or no wall"}; remaining = science_deadline - time.monotonic()
        if acquisition.get("g90_confirm_update") is not None and remaining > 0:
            steps = core.stress_steps_from_remaining(remaining_seconds=remaining, updates_per_sec=float(resolved["training_updates_per_sec"]))
            if steps: stress = run_stress(acquisition=acquisition, steps_per_arm=steps, spec=spec, tokenizer=tokenizer,
                torch=torch, device=device, special=special, splits=splits, battery=battery,
                deadline=science_deadline, out=out, progress=progress)
        campaign["stress"] = stress
        sealed = {"status": "NOT_MEASURED_BUDGET"}; final_cp = acquisition.get("final_checkpoint")
        if final_cp and time.monotonic() < science_deadline - 20.0:
            from v5_model.core import initialize
            from v5_training.optimizer import build_adamw_optimizer
            model = initialize(spec, int(resolved["parent_seed"]), torch_module=torch).to(device)
            optimizer = build_adamw_optimizer(model, torch_module=torch)
            legacy._load_checkpoint(Path(final_cp), model=model, optimizer=optimizer, torch=torch)
            sealed = {"status": "MEASURED_AFTER_DECISION", "standard": legacy.generate_rates_batched(model, tokenizer, list(splits["sealed_reserved"]), torch=torch, device=device, special=special)}
        campaign["sealed"] = sealed; campaign["status"] = "COMPLETE"
    except Exception as exc:
        failure = {"schema": "anra-cyr-gpu010-failure/v1", "exception": type(exc).__name__, "message": str(exc),
                   "traceback": traceback.format_exc(), "wall_seconds": time.monotonic() - started}
        campaign["status"] = "FAILED"; campaign.setdefault("decision", {"verdict": "INCONCLUSIVE_RUNTIME_FAILURE",
            "production_promotion_authorized": False, "pre500m_authorized": False, "training_500m_authorized": False})
    finally:
        campaign["wall_seconds"] = time.monotonic() - started
        bundle = _package(out, campaign=campaign, preregistration=preregistration, failure=failure)
        campaign["bundle"] = bundle; write_json(out / "campaign_receipt.json", campaign)
    if failure is not None: raise RuntimeError(f"CYR-GPU-010 failed after packaging: {failure['message']}")
    return campaign
