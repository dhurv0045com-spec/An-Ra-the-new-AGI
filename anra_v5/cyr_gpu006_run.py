"""CYR-GPU-006 torch orchestrator.

Engineering closure over CYR-GPU-005:
- full mode executes on CUDA (never the CPU fallback that invalidated 005);
- CELL-0 calibration/resolution is consumed verbatim by CELL 1;
- all three acquisition parents are attempted; no break after first G90;
- every qualified parent forks all four policies from identical bytes/tail;
- candidate-free generation is batched and included in runtime calibration;
- stage-level durable resume skips completed parents/arms after Colab restart;
- failure paths always emit FAILURE.json + a partial evidence bundle.
"""
from __future__ import annotations

import gc
import hashlib
import io
import json
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping

from v5_experiments import cyr_gpu006 as core
from v5_experiments.cyr_tournament import HysteresisController

BUNDLE_NAME = "CYMEK_GPU_RESEARCH_V6_RESULTS.zip"
TOKENIZER_ARTIFACT = "artifacts/e1/local_tournament/tokenizer-24576.json.gz"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, body: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(body, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text("utf-8"))


def production_tokenizer(repo: Path) -> tuple[Any, dict[str, Any]]:
    from v5_data.corpus_loading import _load_tokenizer
    import gzip
    tokenizer, _evaluation = _load_tokenizer(repo.resolve())
    artifact = repo / TOKENIZER_ARTIFACT
    identity = tokenizer.identity
    body = json.loads(gzip.decompress(artifact.read_bytes()).decode("utf-8"))
    added = {entry["content"]: entry["id"] for entry in body.get("added_tokens", [])}
    for required in ("<pad>", "<unk>", "<bos>", "<eos>"):
        if required not in added:
            raise ValueError(f"tokenizer artifact missing {required}")
    if sha256_file(artifact) != identity.artifact_sha256:
        raise ValueError("tokenizer artifact hash differs from TokenizerIdentity")
    receipt = {
        "schema": "anra-cyr-gpu006-tokenizer/v1", "artifact": TOKENIZER_ARTIFACT,
        "artifact_sha256": identity.artifact_sha256,
        "vocabulary_size": int(tokenizer.vocabulary_size),
        "pad_id": int(added["<pad>"]), "unk_id": int(added["<unk>"]),
        "bos_id": int(added["<bos>"]), "eos_id": int(added["<eos>"]),
        "identity_special_ids": dict(identity.special_token_ids),
    }
    if receipt["vocabulary_size"] != 24576:
        raise ValueError("full CYR experiment requires the frozen 24,576 tokenizer")
    expected = (identity.special_token_ids["pad"], identity.special_token_ids["bos"], identity.special_token_ids["eos"])
    actual = (receipt["pad_id"], receipt["bos_id"], receipt["eos_id"])
    if actual != expected:
        raise ValueError("tokenizer special IDs disagree with frozen identity")
    return tokenizer, receipt


def render_batch(tokenizer: Any, rows: list[dict[str, Any]], *, torch: Any,
                 device: Any, special: Mapping[str, int]) -> tuple[Any, Any, Any, dict[str, int]]:
    bos, eos, pad = special["bos_id"], special["eos_id"], special["pad_id"]
    encoded: list[tuple[list[int], int]] = []
    for row in rows:
        prompt_ids = tokenizer.encode(row["prompt"])
        answer_ids = tokenizer.encode(row["answer"])
        encoded.append(([bos, *prompt_ids, *answer_ids, eos], 1 + len(prompt_ids)))
    width = max(len(ids) for ids, _ in encoded)
    tokens = torch.tensor([ids + [pad] * (width - len(ids)) for ids, _ in encoded], dtype=torch.long, device=device)
    segment_ids = torch.tensor([[0] * len(ids) + [-1] * (width - len(ids)) for ids, _ in encoded], dtype=torch.long, device=device)
    eligible_rows = []
    for ids, prompt_len in encoded:
        eligible_rows.append([False] * prompt_len + [True] * (len(ids) - prompt_len) + [False] * (width - len(ids)))
    eligible = torch.tensor(eligible_rows, dtype=torch.bool, device=device)
    return tokens, segment_ids, eligible, {
        "real_tokens": int((segment_ids >= 0).sum().item()),
        "supervised_tokens": int(eligible.sum().item()),
    }


def generate_rates_batched(model: Any, tokenizer: Any, rows: list[dict[str, Any]], *,
                           torch: Any, device: Any, special: Mapping[str, int],
                           batch_size: int = 32, max_new_tokens: int = 8) -> dict[str, Any]:
    """Candidate-free greedy generation, bucketed by prompt token length."""
    from v5_model.core import packed_layout
    bos, eos, pad = special["bos_id"], special["eos_id"], special["pad_id"]
    buckets: dict[int, list[tuple[dict[str, Any], list[int]]]] = {}
    for row in rows:
        prompt = [bos, *tokenizer.encode(row["prompt"])]
        buckets.setdefault(len(prompt), []).append((row, prompt))
    content_exact = complete = eos_stops = caps = prefix_extra = invalid = 0
    total = len(rows)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _length, items in sorted(buckets.items()):
            for start in range(0, len(items), batch_size):
                chunk = items[start:start + batch_size]
                prompts = [prompt for _row, prompt in chunk]
                generated: list[list[int]] = [[] for _ in chunk]
                done = [False] * len(chunk)
                stop_reason = ["MAX_TOKENS"] * len(chunk)
                for _step in range(max_new_tokens):
                    sequences = [prompt + gen for prompt, gen in zip(prompts, generated)]
                    width = max(len(seq) for seq in sequences)
                    padded = [seq + [eos] * (width - len(seq)) for seq in sequences]
                    tokens = torch.tensor(padded, dtype=torch.long, device=device)
                    segments = torch.zeros_like(tokens)
                    positions, mask = packed_layout(segments, torch_module=torch)
                    logits = model(tokens, positions.to(device), mask.to(device))[:, -1]
                    next_ids = torch.argmax(logits, dim=-1).tolist()
                    for index, next_id in enumerate(next_ids):
                        if done[index]:
                            generated[index].append(eos)
                            continue
                        if int(next_id) == eos:
                            done[index] = True
                            stop_reason[index] = "EOS"
                            generated[index].append(eos)
                        elif int(next_id) == pad:
                            done[index] = True
                            stop_reason[index] = "PAD"
                            generated[index].append(pad)
                        else:
                            generated[index].append(int(next_id))
                    if all(done):
                        break
                for index, (row, _prompt) in enumerate(chunk):
                    raw = [token for token in generated[index] if token not in (eos, pad)]
                    text = tokenizer.decode(raw) if raw else ""
                    expected = row["answer"]
                    exact = text == expected
                    if exact:
                        content_exact += 1
                    if exact and stop_reason[index] == "EOS":
                        complete += 1
                    if stop_reason[index] == "EOS":
                        eos_stops += 1
                    elif stop_reason[index] == "MAX_TOKENS":
                        caps += 1
                    else:
                        invalid += 1
                    if text != expected and text.startswith(expected):
                        prefix_extra += 1
    if was_training:
        model.train()
    denom = max(total, 1)
    return {
        "content_exact": content_exact / denom,
        "complete_exact_with_valid_stop": complete / denom,
        "eos_rate": eos_stops / denom,
        "max_tokens_rate": caps / denom,
        "invalid_rate": invalid / denom,
        "prefix_correct_extra": prefix_extra / denom,
        "total": total,
    }


def _stub_state(cumulative_tokens: int) -> Any:
    return type("ResearchState", (), {"cumulative_tokens": int(cumulative_tokens)})()


def _cursor(*, update: int, offset: int, data_sha: str) -> Any:
    from v5_training.state import CURSOR_SCHEMA, CursorState
    return CursorState(CURSOR_SCHEMA, data_sha, update, offset, 0)


def _make_backend(*, model: Any, optimizer: Any, special: Mapping[str, int],
                  device: Any, torch: Any, lr: float) -> Any:
    from v5_training.production_backend import ProductionTrainingBackend
    return ProductionTrainingBackend(
        model=model, optimizer=optimizer, bos_id=special["bos_id"], pad_id=special["pad_id"],
        device=device, schedule=lambda cumulative_tokens: lr,
        bfloat16_autocast=getattr(device, "type", None) == "cuda",
        torch_module=torch, activation_checkpointing=False)


def _one_update(*, backend: Any, tokenizer: Any, rows: list[dict[str, Any]],
                torch: Any, device: Any, special: Mapping[str, int],
                cumulative: int, update: int, data_sha: str) -> dict[str, Any]:
    tokens, segment_ids, eligible, counted = render_batch(tokenizer, rows, torch=torch, device=device, special=special)
    state = _stub_state(cumulative)
    ctx = backend.begin_update(state)
    ctx = backend.accumulate_microstep(
        ctx, tokens=tokens, segment_ids=segment_ids, eligible=eligible,
        tokens_by_source={"cyr": counted["supervised_tokens"]}, planned_total=counted["supervised_tokens"])
    backend.finish_update(state, ctx, planned_total=counted["supervised_tokens"],
                          cursor=_cursor(update=update, offset=update * len(rows), data_sha=data_sha))
    receipt = backend.last_receipt
    if receipt is None:
        raise RuntimeError("production backend produced no update receipt")
    return {"counted": counted, "backend_receipt": receipt}


def calibrate_candidate(*, name: str, spec: Any, tokenizer: Any, torch: Any,
                        device: Any, special: Mapping[str, int],
                        train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Representative calibration includes optimizer.step and generation."""
    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer
    if getattr(device, "type", None) != "cuda":
        raise ValueError("full calibration requires CUDA")
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        created = time.monotonic()
        model = initialize(spec, 123, torch_module=torch).to(device)
        optimizer = build_adamw_optimizer(model, torch_module=torch, lr=1e-3)
        backend = _make_backend(model=model, optimizer=optimizer, special=special, device=device, torch=torch, lr=1e-3)
        cold_startup = time.monotonic() - created
        data_sha = hashlib.sha256(_canonical_json([row["prompt"] + row["answer"] for row in train_rows])).hexdigest()
        batch = train_rows[:16]
        cumulative = 0
        first = _one_update(backend=backend, tokenizer=tokenizer, rows=batch, torch=torch, device=device,
                            special=special, cumulative=cumulative, update=1, data_sha=data_sha)
        cumulative += first["counted"]["real_tokens"]
        torch.cuda.synchronize()
        timed_tokens = 0
        timed_updates = 3
        started = time.monotonic()
        for index in range(timed_updates):
            result = _one_update(backend=backend, tokenizer=tokenizer, rows=batch, torch=torch, device=device,
                                 special=special, cumulative=cumulative, update=index + 2, data_sha=data_sha)
            cumulative += result["counted"]["real_tokens"]
            timed_tokens += result["counted"]["real_tokens"]
        torch.cuda.synchronize()
        train_wall = max(time.monotonic() - started, 1e-9)
        eval_subset = eval_rows[:min(32, len(eval_rows))]
        torch.cuda.synchronize()
        eval_started = time.monotonic()
        generate_rates_batched(model, tokenizer, eval_subset, torch=torch, device=device, special=special, batch_size=32)
        torch.cuda.synchronize()
        eval_wall = max(time.monotonic() - eval_started, 1e-9)
        peak = torch.cuda.max_memory_allocated() / 1e9
        parameters = sum(parameter.numel() for parameter in model.parameters())
        return {
            "schema": "anra-cyr-gpu006-calibration/v1", "status": "PASS", "proxy": name,
            "parameters": parameters, "gpu_name": torch.cuda.get_device_name(0),
            "cold_startup_seconds": round(cold_startup, 3),
            "training_real_tokens_per_sec": timed_tokens / train_wall,
            "training_updates_per_sec": timed_updates / train_wall,
            "generation_examples_per_sec": len(eval_subset) / eval_wall,
            "generation_examples_benchmarked": len(eval_subset),
            "peak_vram_gb": peak, "optimizer_step_included": True,
            "candidate_free_generation_included": True,
        }
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        return {"schema": "anra-cyr-gpu006-calibration/v1", "status": "OOM", "proxy": name, "error": str(exc)}
    finally:
        gc.collect()
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def calibrate_candidates(*, registry: Mapping[str, Mapping[str, Any]], tokenizer: Any,
                         torch: Any, device: Any, special: Mapping[str, int],
                         train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    receipts: dict[str, Any] = {}
    for name in ("MIDI", "MICRO", "RESEARCH_SMALL"):
        receipts[name] = calibrate_candidate(name=name, spec=registry[name]["spec"], tokenizer=tokenizer,
                                             torch=torch, device=device, special=special,
                                             train_rows=train_rows, eval_rows=eval_rows)
    return receipts


def _state_fingerprint(model: Any, optimizer: Any, *, torch: Any) -> dict[str, str]:
    model_buffer, optimizer_buffer = io.BytesIO(), io.BytesIO()
    torch.save(model.state_dict(), model_buffer)
    torch.save(optimizer.state_dict(), optimizer_buffer)
    return {"model_sha256": hashlib.sha256(model_buffer.getvalue()).hexdigest(),
            "optimizer_sha256": hashlib.sha256(optimizer_buffer.getvalue()).hexdigest()}


def _save_checkpoint(path: Path, *, model: Any, optimizer: Any, torch: Any,
                     counters: Mapping[str, Any]) -> dict[str, Any]:
    from anra_v5.cyr_execute import save_research_checkpoint
    return save_research_checkpoint(path, model=model, optimizer=optimizer, torch=torch, counters=dict(counters))


def _load_checkpoint(path: Path, *, model: Any, optimizer: Any, torch: Any) -> dict[str, Any]:
    from anra_v5.cyr_execute import load_research_checkpoint
    return load_research_checkpoint(path, model=model, optimizer=optimizer, torch=torch)


def _checkpoint_valid(path: str | Path) -> bool:
    root = Path(path)
    return all((root / name).exists() for name in ("model.bin", "optimizer.bin", "counters.json", "receipt.json"))


def acquire_parent(*, seed: int, spec: Any, proxy_name: str, tokenizer: Any,
                   torch: Any, device: Any, special: Mapping[str, int],
                   train_rows: list[dict[str, Any]], controller_rows: list[dict[str, Any]],
                   probe_rows: list[dict[str, Any]], target_actual_tokens: int,
                   stream: Mapping[str, Any], store_root: Path,
                   stage_deadline: float, eval_interval_tokens: int,
                   progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer
    run_root = store_root / f"parent-{seed}"
    receipt_path = run_root / "acquisition.json"
    existing = read_json(receipt_path)
    if existing is not None:
        if existing.get("parent_status") == "G90_CONFIRMED" and _checkpoint_valid(existing.get("parent_checkpoint", "")):
            existing["resume_action"] = "SKIPPED_COMPLETED_ACQUISITION"
            return existing
        if existing.get("parent_status") == "NOT_QUALIFIED" and existing.get("terminal", False):
            existing["resume_action"] = "SKIPPED_TERMINAL_NOT_QUALIFIED"
            return existing
    torch.manual_seed(seed)
    model = initialize(spec, seed, torch_module=torch).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    core.assert_proxy_in_registry(proxy_name, parameters, core.proxy_registry(vocab_size=spec.vocabulary_size))
    optimizer = build_adamw_optimizer(model, torch_module=torch, lr=core.CYR6_LRS["HIGH"])
    backend = _make_backend(model=model, optimizer=optimizer, special=special, device=device, torch=torch, lr=core.CYR6_LRS["HIGH"])
    prefix = list(stream["stream"][:stream["fork_boundary"]])
    data_sha = hashlib.sha256(_canonical_json([row["prompt"] + row["answer"] for row in train_rows])).hexdigest()
    consumed = updates = cursor_offset = 0
    next_eval = eval_interval_tokens
    flags: list[bool] = []
    trace: list[dict[str, Any]] = []
    g90_onset = g90_confirm = None
    status = "RUNNING"
    while consumed < target_actual_tokens:
        if time.monotonic() >= stage_deadline:
            status = "TIMEBOX"
            break
        indices = prefix[cursor_offset:cursor_offset + 16]
        if not indices:
            cursor_offset = 0
            indices = prefix[:16]
        cursor_offset += len(indices)
        rows = [train_rows[index] for index in indices]
        result = _one_update(backend=backend, tokenizer=tokenizer, rows=rows, torch=torch, device=device,
                             special=special, cumulative=consumed, update=updates + 1, data_sha=data_sha)
        consumed += result["counted"]["real_tokens"]
        updates += 1
        if progress and updates % 25 == 0:
            progress(f"parent {seed}: {consumed}/{target_actual_tokens} real tokens")
        if consumed >= next_eval or consumed >= target_actual_tokens:
            controller = generate_rates_batched(model, tokenizer, controller_rows, torch=torch, device=device, special=special)
            probe = generate_rates_batched(model, tokenizer, probe_rows, torch=torch, device=device, special=special)
            trace.append({"update": updates, "real_tokens": consumed, "dev_controller": controller, "train_probe": probe})
            passed = controller["complete_exact_with_valid_stop"] >= 0.90
            flags.append(passed)
            if passed and g90_onset is None:
                g90_onset = consumed
            if len(flags) >= 3 and all(flags[-3:]):
                g90_confirm = consumed
                status = "G90_CONFIRMED"
                break
            while next_eval <= consumed:
                next_eval += eval_interval_tokens
    if status == "RUNNING":
        status = "NO_G90"
    parent_status = "G90_CONFIRMED" if status == "G90_CONFIRMED" else "NOT_QUALIFIED"
    receipt: dict[str, Any] = {
        "schema": "anra-cyr-gpu006-acquisition/v1", "seed": seed, "status": status,
        "parent_status": parent_status, "terminal": status in ("G90_CONFIRMED", "NO_G90"),
        "updates": updates, "actual_real_tokens": consumed,
        "target_actual_real_tokens": target_actual_tokens, "parameters": parameters,
        "eval_trace": trace, "g90_onset_real_tokens": g90_onset,
        "g90_confirm_real_tokens": g90_confirm,
        "future_stream": {key: stream[key] for key in ("prefix_sha256", "tail_sha256", "fork_boundary")},
    }
    if parent_status == "G90_CONFIRMED":
        checkpoint = run_root / "parent-checkpoint"
        cp_receipt = _save_checkpoint(checkpoint, model=model, optimizer=optimizer, torch=torch,
                                      counters={"seed": seed, "updates": updates, "actual_real_tokens": consumed,
                                                "cursor_offset": cursor_offset, "tail_sha256": stream["tail_sha256"],
                                                "status": status})
        receipt["parent_checkpoint"] = str(checkpoint)
        receipt["parent_model_sha256"] = cp_receipt["model_sha256"]
        receipt["parent_optimizer_sha256"] = cp_receipt["optimizer_sha256"]
    write_json(receipt_path, receipt)
    return receipt


def verify_parent_equivalence(*, parent: Mapping[str, Any], spec: Any, torch: Any, device: Any) -> dict[str, Any]:
    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer
    fingerprints: dict[str, Any] = {}
    for arm in core.CYR6_ARMS:
        model = initialize(spec, int(parent["seed"]), torch_module=torch).to(device)
        optimizer = build_adamw_optimizer(model, torch_module=torch)
        restored = _load_checkpoint(Path(parent["parent_checkpoint"]), model=model, optimizer=optimizer, torch=torch)
        fingerprints[arm] = {**_state_fingerprint(model, optimizer, torch=torch),
                             "checkpoint_counters_sha256": hashlib.sha256(_canonical_json(restored["counters"])).hexdigest()}
        del optimizer, model
        if getattr(device, "type", None) == "cuda":
            torch.cuda.empty_cache()
    reference = next(iter(fingerprints.values()))
    identical = all(value == reference for value in fingerprints.values())
    if not identical:
        raise ValueError("fork parent equivalence failed")
    return {"schema": "anra-cyr-gpu006-parent-equivalence/v1", "identical": True,
            "forks": fingerprints, "checked": "model bytes + optimizer bytes + checkpoint counters"}


def _optimizer_moment_norms(optimizer: Any, *, torch: Any) -> dict[str, float]:
    exp_avg = exp_avg_sq = 0.0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            first, second = state.get("exp_avg"), state.get("exp_avg_sq")
            if torch.is_tensor(first):
                exp_avg += float((first.detach().float() ** 2).sum().item())
            if torch.is_tensor(second):
                exp_avg_sq += float((second.detach().float() ** 2).sum().item())
    return {"exp_avg_norm": exp_avg ** 0.5, "exp_avg_sq_norm": exp_avg_sq ** 0.5}


def continuation_arm(*, arm: str, parent: Mapping[str, Any], spec: Any,
                     tokenizer: Any, torch: Any, device: Any, special: Mapping[str, int],
                     train_rows: list[dict[str, Any]], controller_rows: list[dict[str, Any]],
                     measurement_rows: list[dict[str, Any]], stream: Mapping[str, Any],
                     target_actual_tokens: int, store_root: Path, stage_deadline: float,
                     eval_interval_tokens: int, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer
    if parent.get("parent_status") != "G90_CONFIRMED":
        raise ValueError("continuation requires G90 parent")
    seed = int(parent["seed"])
    arm_root = store_root / f"parent-{seed}" / f"arm-{arm}"
    receipt_path = arm_root / "receipt.json"
    existing = read_json(receipt_path)
    if existing is not None and existing.get("status") == "COMPLETE" and _checkpoint_valid(existing.get("final_checkpoint", "")):
        existing["resume_action"] = "SKIPPED_COMPLETED_ARM"
        return existing
    torch.manual_seed(seed + 6060)
    model = initialize(spec, seed, torch_module=torch).to(device)
    optimizer = build_adamw_optimizer(model, torch_module=torch)
    _load_checkpoint(Path(parent["parent_checkpoint"]), model=model, optimizer=optimizer, torch=torch)
    parent_flat = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()]).to(device)
    controller = None
    if arm == "HYSTERETIC_HIGH_LOW":
        controller = HysteresisController(enter_retention=core.CYR6_HYSTERESIS["enter_retention"],
                                          reenter_plasticity=core.CYR6_HYSTERESIS["reenter_plasticity"],
                                          confirmations=core.CYR6_HYSTERESIS["confirmations"])
        controller.assert_valid()
    backend = _make_backend(model=model, optimizer=optimizer, special=special, device=device, torch=torch, lr=core.CYR6_LRS["HIGH"])
    tail = list(stream["stream"][stream["fork_boundary"]:])
    switch_point = core.fixed_time_switch_point(continuation_target_tokens=target_actual_tokens)
    data_sha = hashlib.sha256(_canonical_json([row["prompt"] + row["answer"] for row in train_rows])).hexdigest()
    consumed = updates = tail_offset = 0
    next_eval = eval_interval_tokens
    batch_shas: list[str] = []
    eval_trace: list[dict[str, Any]] = []
    lr_trace: list[float] = []
    high_tokens = low_tokens = 0
    grad_pre_sum = grad_post_sum = 0.0
    status = "RUNNING"
    while consumed < target_actual_tokens:
        if time.monotonic() >= stage_deadline:
            status = "TIMEBOX"
            break
        lr = core.lr_for_token(arm, consumed, switch_point=switch_point, controller=controller)
        backend.schedule = lambda cumulative_tokens, _lr=lr: _lr
        for group in optimizer.param_groups:
            group["lr"] = lr
        indices = tail[tail_offset:tail_offset + 16]
        if not indices:
            tail_offset = 0
            indices = tail[:16]
        tail_offset += len(indices)
        batch_shas.append(hashlib.sha256(_canonical_json(indices)).hexdigest())
        rows = [train_rows[index] for index in indices]
        result = _one_update(backend=backend, tokenizer=tokenizer, rows=rows, torch=torch, device=device,
                             special=special, cumulative=consumed, update=updates + 1, data_sha=data_sha)
        amount = result["counted"]["real_tokens"]
        consumed += amount
        updates += 1
        receipt = result["backend_receipt"]
        grad_pre_sum += float(receipt.get("grad_norm_pre_clip", 0.0))
        grad_post_sum += float(receipt.get("grad_norm_post_clip", 0.0))
        lr_trace.append(lr)
        if lr >= core.CYR6_LRS["HIGH"]:
            high_tokens += amount
        else:
            low_tokens += amount
        if progress and updates % 25 == 0:
            progress(f"parent {seed}/{arm}: {consumed}/{target_actual_tokens}")
        if consumed >= next_eval or consumed >= target_actual_tokens:
            ctrl = generate_rates_batched(model, tokenizer, controller_rows, torch=torch, device=device, special=special)
            measure = generate_rates_batched(model, tokenizer, measurement_rows, torch=torch, device=device, special=special)
            current = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
            delta = float((current - parent_flat).float().norm().item())
            base_norm = max(float(parent_flat.float().norm().item()), 1e-12)
            eval_trace.append({"update": updates, "continuation_real_tokens": consumed, "lr": lr,
                               "dev_controller": ctrl, "dev_measurement": measure,
                               "parameter_displacement": {"l2": delta, "relative": delta / base_norm},
                               "adam_moments": _optimizer_moment_norms(optimizer, torch=torch)})
            if controller is not None:
                controller.observe(metric=ctrl["complete_exact_with_valid_stop"],
                                   threshold_note="DEV_CONTROLLER candidate-free complete exact",
                                   token_position=consumed, lr_before=lr,
                                   lr_plasticity=core.CYR6_LRS["HIGH"], lr_retention=core.CYR6_LRS["LOW"])
            while next_eval <= consumed:
                next_eval += eval_interval_tokens
    if status == "RUNNING":
        status = "COMPLETE"
    final_checkpoint = arm_root / "final"
    _save_checkpoint(final_checkpoint, model=model, optimizer=optimizer, torch=torch,
                     counters={"seed": seed, "arm": arm, "updates": updates,
                               "continuation_real_tokens": consumed, "tail_offset": tail_offset, "status": status})
    trajectory = [entry["dev_measurement"]["complete_exact_with_valid_stop"] for entry in eval_trace]
    ret90 = (sum(value >= 0.90 for value in trajectory) / len(trajectory) if trajectory else 0.0)
    last_displacement = eval_trace[-1]["parameter_displacement"] if eval_trace else None
    redteam = {
        "schema": "anra-cyr-gpu006-redteam/v1",
        "parameter_displacement": last_displacement,
        "adam_moment_norms": eval_trace[-1]["adam_moments"] if eval_trace else None,
        "integrated_grad_norm_pre_clip": grad_pre_sum,
        "integrated_grad_norm_post_clip": grad_post_sum,
        "high_exposure_tokens": high_tokens, "low_exposure_tokens": low_tokens,
        "update_norm_measured": False,
        "note": "near-freezing is interpreted from displacement + gradients + moments; no false claim of direct per-step update-norm measurement",
        "pass": bool(eval_trace and last_displacement is not None),
    }
    result = {
        "schema": "anra-cyr-gpu006-arm/v1", "seed": seed, "arm": arm, "status": status,
        "updates": updates, "actual_real_tokens": consumed,
        "target_actual_real_tokens": target_actual_tokens, "shares_valid_parent": True,
        "parent_model_sha256": parent["parent_model_sha256"],
        "future_tail_sha256": stream["tail_sha256"], "consumed_batch_shas": batch_shas,
        "retention_ret90": ret90, "final_g": trajectory[-1] if trajectory else 0.0,
        "lr_trace": lr_trace, "eval_trace": eval_trace,
        "high_exposure_tokens": high_tokens, "low_exposure_tokens": low_tokens,
        "redteam": redteam, "redteam_pass": redteam["pass"],
        "controller_snapshot": controller.snapshot() if controller is not None else None,
        "final_checkpoint": str(final_checkpoint),
        "resume_semantics": "completed arms skipped; incomplete arm restarts cleanly from the frozen parent/tail",
    }
    write_json(receipt_path, result)
    return result


def _binding_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    from v5_experiments.cyr_tournament import render_worlds
    worlds = render_worlds(family=core.CYR6_TRANSFER_FAMILY,
                           split_seeds={"train": 9001, "heldout": 9002},
                           worlds_per_split={"train": 64, "heldout": 32})
    def to_row(record: Mapping[str, Any]) -> dict[str, str]:
        text = record["text"]
        answer = record["answer"]
        return {"prompt": text[:len(text) - len(answer)], "answer": answer}
    return ([to_row(world["base"]) for world in worlds["train"]],
            [to_row(world["base"]) for world in worlds["heldout"]])


def transfer_pair(*, parent_run: Mapping[str, Any], winner: str, spec: Any,
                  tokenizer: Any, torch: Any, device: Any, special: Mapping[str, int],
                  target_actual_tokens: int, deadline: float) -> dict[str, Any]:
    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer
    train_rows, heldout = _binding_rows()
    outputs: dict[str, Any] = {}
    for label, checkpoint in (("baseline", parent_run["parent"]["parent_checkpoint"]),
                              ("finalist", parent_run["arms"][winner]["final_checkpoint"])):
        if time.monotonic() >= deadline:
            outputs[label] = {"status": "TIMEBOX"}
            continue
        seed = int(parent_run["seed"])
        model = initialize(spec, seed, torch_module=torch).to(device)
        optimizer = build_adamw_optimizer(model, torch_module=torch)
        _load_checkpoint(Path(checkpoint), model=model, optimizer=optimizer, torch=torch)
        backend = _make_backend(model=model, optimizer=optimizer, special=special, device=device, torch=torch, lr=core.CYR6_LRS["HIGH"])
        consumed = updates = 0
        data_sha = hashlib.sha256(_canonical_json(train_rows)).hexdigest()
        while consumed < target_actual_tokens and time.monotonic() < deadline:
            batch = [train_rows[(updates * 16 + i) % len(train_rows)] for i in range(16)]
            update = _one_update(backend=backend, tokenizer=tokenizer, rows=batch, torch=torch, device=device,
                                 special=special, cumulative=consumed, update=updates + 1, data_sha=data_sha)
            consumed += update["counted"]["real_tokens"]
            updates += 1
        status = "COMPLETE" if consumed >= target_actual_tokens else "TIMEBOX"
        generated = generate_rates_batched(model, tokenizer, heldout, torch=torch, device=device, special=special) if status == "COMPLETE" else None
        outputs[label] = {"status": status, "actual_real_tokens": consumed, "heldout_generated": generated}
    return {"seed": parent_run["seed"], "winner": winner, "results": outputs}


def transfer_replicated(*, parent_runs: list[Mapping[str, Any]], winner: str, spec: Any,
                        tokenizer: Any, torch: Any, device: Any, special: Mapping[str, int],
                        target_actual_tokens: int, deadline: float) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for parent in parent_runs:
        if len(pairs) >= core.CYR6_TRANSFER_MIN_PARENTS:
            break
        if parent.get("parent_status") != "G90_CONFIRMED" or winner not in parent.get("arms", {}):
            continue
        pairs.append(transfer_pair(parent_run=parent, winner=winner, spec=spec, tokenizer=tokenizer,
                                   torch=torch, device=device, special=special,
                                   target_actual_tokens=target_actual_tokens, deadline=deadline))
    complete = [pair for pair in pairs if all(entry.get("status") == "COMPLETE" for entry in pair["results"].values())]
    if len(complete) < core.CYR6_TRANSFER_MIN_PARENTS:
        return {"schema": "anra-cyr-gpu006-transfer/v1", "status": "INCONCLUSIVE", "pairs": pairs,
                "reason": "fewer than two complete transfer pairs"}
    diffs = []
    informative = True
    for pair in complete:
        base = pair["results"]["baseline"]["heldout_generated"]["complete_exact_with_valid_stop"]
        final = pair["results"]["finalist"]["heldout_generated"]["complete_exact_with_valid_stop"]
        informative = informative and max(base, final) > 0
        diffs.append(final - base)
    status = "REPLICATED_COMPLETE" if informative and min(diffs) >= -0.05 else "INCONCLUSIVE"
    return {"schema": "anra-cyr-gpu006-transfer/v1", "status": status, "pairs": pairs,
            "finalist_minus_baseline": diffs,
            "criterion": "two complete nonzero-event pairs; finalist no worse than -0.05 each"}


def _environment(torch: Any, device: Any) -> dict[str, Any]:
    return {"schema": "anra-cyr-gpu006-environment/v1", "torch": torch.__version__,
            "device": str(device), "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None}


def package_bundle(out: Path, *, campaign: Mapping[str, Any], preregistration: Mapping[str, Any],
                   failure: Mapping[str, Any] | None = None) -> dict[str, Any]:
    bundle = out / BUNDLE_NAME
    payload: dict[str, Any] = {
        "SESSION_MANIFEST.json": {"experiment": core.CYR6_ID, "status": campaign.get("status"),
                                  "wall_seconds": campaign.get("wall_seconds")},
        "ENVIRONMENT.json": campaign.get("environment", {}),
        "PREREGISTRATION.json": dict(preregistration),
        "RESOLVED_PREREGISTRATION.json": campaign.get("resolved", {}),
        "CALIBRATION.json": campaign.get("calibrations", {}),
        "MODEL_REGISTRY.json": campaign.get("proxy_registry", {}),
        "DATA_MANIFEST.json": campaign.get("data_manifest", {}),
        "SPLIT_MANIFEST.json": campaign.get("split_manifest", {}),
        "DECISION.json": campaign.get("decision", {}),
        "TRANSFER.json": campaign.get("transfer", {}),
        "SEALED.json": campaign.get("sealed", {}),
        "ACQUISITION/parents.json": [parent.get("parent", {}) for parent in campaign.get("parent_runs", [])],
        "RETENTION/parents.json": campaign.get("parent_runs", []),
        "REDTEAM.json": {str(parent.get("seed")): {arm: receipt.get("redteam") for arm, receipt in parent.get("arms", {}).items()}
                         for parent in campaign.get("parent_runs", [])},
    }
    if failure is not None:
        payload["FAILURE.json"] = dict(failure)
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, body in payload.items():
            archive.writestr(name, json.dumps(body, indent=2, default=str))
    return {"path": str(bundle), "sha256": sha256_file(bundle), "entries": sorted(payload)}


def run_campaign(*, repo: Path, out: Path, preregistration: Mapping[str, Any],
                 resolved: Mapping[str, Any], calibrations: Mapping[str, Any],
                 torch: Any = None, device: Any = None,
                 progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Execute the frozen 006 plan; no calibration/re-resolution occurs here."""
    if torch is None:
        import torch as torch_module
        torch = torch_module
    if not torch.cuda.is_available():
        raise RuntimeError("CYR-GPU-006 requires a CUDA Colab GPU")
    if device is None:
        device = torch.device("cuda")
    if getattr(device, "type", None) != "cuda":
        raise RuntimeError("CYR-GPU-006 full run refuses non-CUDA device")
    resolved = core.validate_resolved(resolved)
    if resolved["proxy"] not in calibrations or calibrations[resolved["proxy"]].get("status") != "PASS":
        raise ValueError("resolved proxy lacks a passing CELL-0 calibration")

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    hard_deadline = started + float(resolved["wall_budget_minutes"]) * 60.0
    packaging_deadline = hard_deadline - core.CYR6_PACKAGING_RESERVE_MINUTES * 60.0
    acquisition_deadline = started + (packaging_deadline - started) * 0.42
    continuation_deadline = started + (packaging_deadline - started) * 0.90
    transfer_deadline = packaging_deadline
    campaign: dict[str, Any] = {
        "schema": "anra-cyr-gpu006-campaign/v1", "experiment": core.CYR6_ID,
        "status": "RUNNING", "resolved": dict(resolved), "calibrations": dict(calibrations),
        "environment": _environment(torch, device),
        "resume_semantics": "stage-level: completed parents/arms are skipped; incomplete arms restart from the identical parent/tail",
    }
    failure: dict[str, Any] | None = None
    try:
        tokenizer, identity = production_tokenizer(repo)
        special = {"pad_id": identity["pad_id"], "bos_id": identity["bos_id"], "eos_id": identity["eos_id"]}
        if preregistration.get("tokenizer", {}).get("artifact_sha256") != identity["artifact_sha256"]:
            raise ValueError("runtime tokenizer differs from preregistration")
        splits = core.render_t2_worlds()
        manifest = core.build_data_manifest(splits)
        core.assert_manifest_sha(manifest)
        expected_manifest = preregistration.get("data", {}).get("data_manifest_sha256_full")
        if expected_manifest and manifest["sha256"] != expected_manifest:
            raise ValueError("runtime rendered-data manifest differs from preregistration")
        audit = core.commutation_audit(splits, tv_bound=0.20)
        if not audit["commutation_free"]:
            raise ValueError(f"data leak audit failed: {audit['findings']}")
        campaign["data_manifest"] = manifest
        campaign["split_manifest"] = {name: [row["world_id"] for row in rows] for name, rows in splits.items()}
        campaign["leak_audit"] = audit
        registry = core.proxy_registry(vocab_size=identity["vocabulary_size"])
        campaign["proxy_registry"] = {name: {"parameters": entry["parameters"], "role": entry["role"]}
                                      for name, entry in registry.items()}
        spec = registry[resolved["proxy"]]["spec"]

        parent_receipts: list[dict[str, Any]] = []
        streams: dict[int, Any] = {}
        for seed in core.CYR6_PARENT_SEEDS:
            stream = core.build_future_stream(seed=seed, world_count=len(splits["train"]),
                                              prefix_rows=max(64, int(resolved["target_actual_tokens_acquisition"] // 128)),
                                              tail_rows=max(64, int(resolved["target_actual_tokens_continuation"] // 128)))
            streams[seed] = stream
            parent_receipts.append(acquire_parent(
                seed=seed, spec=spec, proxy_name=resolved["proxy"], tokenizer=tokenizer,
                torch=torch, device=device, special=special, train_rows=splits["train"],
                controller_rows=splits["dev_controller"], probe_rows=splits["train"][:16],
                target_actual_tokens=int(resolved["target_actual_tokens_acquisition"]), stream=stream,
                store_root=out, stage_deadline=acquisition_deadline,
                eval_interval_tokens=int(resolved["acquisition_eval_interval_tokens"]), progress=progress))
        campaign["parents"] = parent_receipts

        parent_runs: list[dict[str, Any]] = []
        for parent_index, parent in enumerate(parent_receipts):
            seed = int(parent["seed"])
            run_path = out / f"parent-{seed}" / "parent-run.json"
            existing_run = read_json(run_path)
            if existing_run and existing_run.get("complete_parent_experiment"):
                parent_runs.append(existing_run)
                continue
            if parent.get("parent_status") != "G90_CONFIRMED":
                record = {"seed": seed, "parent_status": parent["parent_status"], "parent": parent,
                          "arms": {}, "parent_equivalence": {"identical": False},
                          "future_tail": {"identical": False}, "complete_parent_experiment": True}
                write_json(run_path, record)
                parent_runs.append(record)
                continue
            equivalence = verify_parent_equivalence(parent=parent, spec=spec, torch=torch, device=device)
            arms: dict[str, Any] = {}
            for arm in core.arm_order(parent_index):
                arms[arm] = continuation_arm(
                    arm=arm, parent=parent, spec=spec, tokenizer=tokenizer, torch=torch, device=device,
                    special=special, train_rows=splits["train"], controller_rows=splits["dev_controller"],
                    measurement_rows=splits["dev_measurement"], stream=streams[seed],
                    target_actual_tokens=int(resolved["target_actual_tokens_continuation"]),
                    store_root=out, stage_deadline=continuation_deadline,
                    eval_interval_tokens=int(resolved["continuation_eval_interval_tokens"]), progress=progress)
            compared = min((len(receipt["consumed_batch_shas"]) for receipt in arms.values()), default=0)
            tail = ({"identical": False, "batches_compared": 0} if compared == 0 else
                    core.assert_future_tail_equality({arm: receipt["consumed_batch_shas"][:compared]
                                                      for arm, receipt in arms.items()}))
            record = {"seed": seed, "parent_status": parent["parent_status"], "parent": parent,
                      "arms": arms, "parent_equivalence": equivalence, "future_tail": tail,
                      "complete_parent_experiment": all(receipt.get("status") == "COMPLETE" for receipt in arms.values())}
            write_json(run_path, record)
            parent_runs.append(record)
        campaign["parent_runs"] = parent_runs

        preliminary = core.decide_campaign(parent_runs)
        transfer = None
        if preliminary.get("verdict") == "REPLICATED_WINNER" and resolved.get("transfer_enabled") and time.monotonic() < transfer_deadline:
            transfer = transfer_replicated(
                parent_runs=parent_runs, winner=preliminary["winner"], spec=spec, tokenizer=tokenizer,
                torch=torch, device=device, special=special,
                target_actual_tokens=int(resolved.get("transfer_target_actual_tokens", core.CYR6_TRANSFER_TOKENS)),
                deadline=transfer_deadline)
        campaign["transfer"] = transfer
        decision = core.decide_campaign(parent_runs, transfer=transfer)
        campaign["decision"] = decision

        sealed: dict[str, Any] = {"status": "SKIPPED_NO_REPLICATED_WINNER"}
        if decision.get("winner") and time.monotonic() < packaging_deadline:
            sealed = {"status": "MEASURED_AFTER_DECISION", "winner": decision["winner"], "parents": {}}
            for parent_run in parent_runs:
                if decision["winner"] not in parent_run.get("arms", {}):
                    continue
                winner_receipt = parent_run["arms"][decision["winner"]]
                if winner_receipt.get("status") != "COMPLETE":
                    continue
                seed = int(parent_run["seed"])
                from v5_model.core import initialize
                from v5_training.optimizer import build_adamw_optimizer
                model = initialize(spec, seed, torch_module=torch).to(device)
                optimizer = build_adamw_optimizer(model, torch_module=torch)
                _load_checkpoint(Path(winner_receipt["final_checkpoint"]), model=model, optimizer=optimizer, torch=torch)
                sealed["parents"][str(seed)] = generate_rates_batched(model, tokenizer, splits["sealed_reserved"],
                                                                       torch=torch, device=device, special=special)
                del model, optimizer
        campaign["sealed"] = sealed
        campaign["status"] = "COMPLETE" if decision.get("verdict") != "INCONCLUSIVE" else "COMPLETE_INCONCLUSIVE"
    except Exception as exc:
        failure = {"schema": "anra-cyr-gpu006-failure/v1", "exception": type(exc).__name__,
                   "message": str(exc), "traceback": traceback.format_exc(),
                   "wall_seconds": time.monotonic() - started}
        campaign["status"] = "FAILED"
        campaign["decision"] = campaign.get("decision", {"verdict": "INCONCLUSIVE", "winner": None,
                                                          "reason": "campaign failure; see FAILURE.json"})
    finally:
        campaign["wall_seconds"] = time.monotonic() - started
        bundle = package_bundle(out, campaign=campaign, preregistration=preregistration, failure=failure)
        campaign["bundle"] = bundle
        write_json(out / "campaign_receipt.json", campaign)
    if failure is not None:
        raise RuntimeError(f"CYR-GPU-006 failed after packaging partial evidence: {failure['message']}")
    return campaign
