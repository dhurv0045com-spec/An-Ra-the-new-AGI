from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from anra_v5 import x_factor_pilot_model_v1 as model_module
from v5_experiments import x_factor_pilot_protocol_v1 as protocol


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _surface_splits(surface: Mapping[str, Any], mode: str) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], str]:
    if "sealed" in surface.get("splits", {}):
        raise RuntimeError("SEALED_ROWS_PRESENT: worker surface contains sealed rows")
    if mode == "official":
        if surface.get("seed") != protocol.SURFACE_SEED or int(surface.get("physical_vocabulary", 0)) != protocol.PHYSICAL_VOCAB:
            raise RuntimeError("OFFICIAL_SURFACE_IDENTITY_MISMATCH")
    if "splits" in surface:
        splits = surface["splits"]
        if set(splits) != {"training", "development"}:
            raise RuntimeError("official worker surface must contain training and development only")
        return list(splits["training"]), list(splits["development"]), str(surface["sha256"])
    protocol.validate_positive_control_surface(surface)
    if int(surface.get("seed", -1)) != protocol.CONTROL_SEED:
        raise RuntimeError("CONTROL_SURFACE_IDENTITY_MISMATCH")
    return list(surface["training"]), list(surface["development"]), str(surface["sha256"])


def _encode_row(row: Mapping[str, Any], mode: str) -> tuple[list[int], list[int]]:
    if mode == "official":
        if "r0_prompt_ids" not in row or "r0_answer_ids" not in row:
            raise RuntimeError("OFFICIAL_RENDERING_MISSING")
        prompt = [int(value) for value in row["r0_prompt_ids"]]
        answer = [int(value) for value in row["r0_answer_ids"]]
        if not prompt or not answer or prompt[0] != 2 or answer[-1] != 3:
            raise RuntimeError("OFFICIAL_RENDERING_BOUNDARY_MISMATCH")
    else:
        prompt = [2, *[int(value) for value in row["prompt_ids"]]]
        answer = [*[int(value) for value in row["answer_ids"]], 3]
    if min(prompt + answer) < 0 or max(prompt + answer) >= protocol.PHYSICAL_VOCAB:
        raise RuntimeError("encoded row escaped the physical vocabulary")
    return prompt, answer


def build_batch(rows: list[Mapping[str, Any]], *, mode: str, torch: Any, device: Any) -> tuple[Any, Any, Any, int, int]:
    encoded = [_encode_row(row, mode) for row in rows]
    width = max(len(prompt) + len(answer) for prompt, answer in encoded)
    tokens: list[list[int]] = []
    segments: list[list[int]] = []
    eligible: list[list[bool]] = []
    supervised = 0
    processed = 0
    for prompt, answer in encoded:
        ids = prompt + answer
        padding = width - len(ids)
        tokens.append(ids + [0] * padding)
        segments.append([0] * len(ids) + [-1] * padding)
        eligible.append([False] * len(prompt) + [True] * len(answer) + [False] * padding)
        supervised += len(answer)
        processed += len(ids)
    return (
        torch.tensor(tokens, dtype=torch.long, device=device),
        torch.tensor(segments, dtype=torch.long, device=device),
        torch.tensor(eligible, dtype=torch.bool, device=device),
        supervised,
        processed,
    )


def _greedy_rates(model: Any, rows: list[Mapping[str, Any]], *, mode: str, torch: Any, device: Any) -> dict[str, float]:
    was_training = model.training
    model.eval()
    complete = 0
    content = 0
    stopped_count = 0
    cap_count = 0
    with torch.no_grad():
        for row in rows:
            prompt, answer = _encode_row(row, mode)
            expected = [value for value in answer if value != 3]
            generated: list[int] = []
            stopped = False
            for _ in range(protocol.MAX_GENERATION_TOKENS):
                current = torch.tensor([prompt + generated], dtype=torch.long, device=device)
                segments = torch.zeros_like(current)
                from v5_model.core import packed_layout
                positions, mask = packed_layout(segments, torch_module=torch)
                logits = model(current, positions, mask)[0, -1]
                next_id = int(torch.argmax(logits).item())
                if next_id == 3:
                    stopped = True
                    break
                if next_id == 0:
                    break
                generated.append(next_id)
            else:
                cap_count += 1
            if generated == expected:
                content += 1
                if stopped:
                    complete += 1
            stopped_count += int(stopped)
    if was_training:
        model.train()
    count = max(len(rows), 1)
    return {
        "content_exact": content / count,
        "complete_exact_with_valid_stop": complete / count,
        "eos_rate": stopped_count / count,
        "max_tokens_rate": cap_count / count,
        "n": len(rows),
    }


def evaluate_development(model: Any, rows: list[Mapping[str, Any]], *, mode: str, torch: Any, device: Any) -> dict[str, Any]:
    families = sorted({str(row["family"]) for row in rows})
    per_family: dict[str, dict[str, float]] = {}
    for family in families:
        per_family[family] = _greedy_rates(
            model,
            [row for row in rows if row["family"] == family],
            mode=mode,
            torch=torch,
            device=device,
        )
    if "identity" not in per_family:
        raise RuntimeError("evaluation surface has no identity family")
    return {
        "identity_exact_valid_eos": per_family["identity"]["complete_exact_with_valid_stop"],
        "per_family": per_family,
    }


def formation_summary(trace: list[Mapping[str, Any]], eligible_from: int) -> dict[str, Any]:
    points = [(int(item["axis"]), float(item["identity_exact_valid_eos"])) for item in trace if int(item["axis"]) >= eligible_from]
    if not points:
        return {"formation_auc": 0.0, "endpoint": 0.0, "first_acquisition_axis": None, "points": 0}
    if len(points) == 1:
        area = points[0][1]
    else:
        area = sum((x1 - x0) * (y0 + y1) / 2.0 for (x0, y0), (x1, y1) in zip(points[:-1], points[1:])) / max(points[-1][0] - points[0][0], 1)
    first = next((axis for axis, score in points if score >= 0.5), None)
    return {
        "formation_auc": round(float(area), 6),
        "endpoint": round(float(points[-1][1]), 6),
        "first_acquisition_axis": first,
        "points": len(points),
    }


def _optimizer(model: Any, *, torch: Any) -> Any:
    from v5_training.optimizer import build_adamw_optimizer
    return build_adamw_optimizer(model, torch_module=torch, lr=protocol.LEARNING_RATE)


def _data_order(rows: list[Mapping[str, Any]], seed: int, torch: Any) -> tuple[list[int], str]:
    generator = torch.Generator().manual_seed(int(seed) + 0xA11CE)
    order = torch.randperm(len(rows), generator=generator).tolist()
    digest = sha256_bytes(canonical(order))
    return [int(value) for value in order], digest


def _save_checkpoint(path: Path, *, model: Any, optimizer: Any, torch: Any, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema": "anra.x-factor-pilot-checkpoint/v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        **dict(payload),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(body, temporary)
    digest = file_sha256(temporary)
    os.replace(temporary, path)
    receipt = {"schema": "anra.x-factor-pilot-checkpoint-receipt/v1", "sha256": digest, **{key: value for key, value in payload.items() if key not in {"trace", "diagnostics"}}}
    atomic_json(path.parent / "CHECKPOINT_RECEIPT.json", receipt)
    return digest


def _load_checkpoint(path: Path, *, model: Any, optimizer: Any, torch: Any, identity: Mapping[str, Any]) -> dict[str, Any]:
    receipt_path = path.parent / "CHECKPOINT_RECEIPT.json"
    if not receipt_path.exists():
        raise RuntimeError("CHECKPOINT_RECEIPT_MISSING")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "anra.x-factor-pilot-checkpoint-receipt/v1" or receipt.get("sha256") != file_sha256(path):
        raise RuntimeError("CHECKPOINT_RECEIPT_MISMATCH")
    for key, value in identity.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"checkpoint receipt identity mismatch for {key}")
    body = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(body, dict) or body.get("schema") != "anra.x-factor-pilot-checkpoint/v1":
        raise RuntimeError("CHECKPOINT_SCHEMA_MISMATCH")
    for key, value in identity.items():
        if body.get(key) != value:
            raise RuntimeError(f"checkpoint identity mismatch for {key}")
    if not isinstance(body.get("model"), dict) or not isinstance(body.get("optimizer"), dict):
        raise RuntimeError("CHECKPOINT_PAYLOAD_MALFORMED")
    model.load_state_dict(body["model"])
    optimizer.load_state_dict(body["optimizer"])
    if body.get("current_model_sha256") != model_module.model_state_sha(model):
        raise RuntimeError("CHECKPOINT_MODEL_HASH_MISMATCH")
    if body.get("cpu_rng_state") is not None:
        torch.set_rng_state(body["cpu_rng_state"])
    if torch.cuda.is_available() and body.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(body["cuda_rng_state"])
    return body


def _write_progress(root: Path, payload: Mapping[str, Any], checkpoint_sha: str) -> None:
    progress = {
        "schema": "anra.x-factor-pilot-progress/v1",
        "mode": payload["mode"],
        "arm": payload["arm"],
        "seed": payload["seed"],
        "protocol_sha256": payload["protocol_sha256"],
        "architecture_id": payload["architecture_id"],
        "data_manifest_sha256": payload["data_manifest_sha256"],
        "updates": int(payload["updates"]),
        "processed_tokens": int(payload["processed_tokens"]),
        "supervised_tokens": int(payload["supervised_tokens"]),
        "stream_cursor": int(payload["stream_cursor"]),
        "latest_development": payload["trace"][-1] if payload["trace"] else None,
        "clip_events": int(payload["clip_events"]),
        "checkpoint_sha256": checkpoint_sha,
        "diagnostic_only": True,
    }
    atomic_json(root / "LATEST_PROGRESS.json", progress)
    atomic_json(root / "progress" / f"UPDATE_{int(payload['updates']):08d}.json", progress)


def train_arm(
    *,
    mode: str,
    arm: str,
    seed: int,
    surface: Mapping[str, Any],
    out_dir: Path,
    torch: Any,
    device: Any,
    target_updates: int | None = None,
    stop_after: int | None = None,
    deadline_epoch: float | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if mode not in {"official", "control", "canary"}:
        raise ValueError(f"unknown training mode: {mode}")
    if arm not in protocol.ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if mode == "official" and seed not in protocol.MODEL_SEEDS:
        raise RuntimeError("official seed is not registered")
    if mode == "control" and seed != protocol.CONTROL_SEED:
        raise RuntimeError("control seed is not registered")
    if deadline_epoch is not None and mode != "official":
        raise RuntimeError("deadline is only an official-session operational control")
    train_rows, dev_rows, surface_sha = _surface_splits(surface, mode)
    if mode == "control" and len(train_rows) != protocol.CONTROL_SYMBOL_COUNT * protocol.CONTROL_TRAIN_CONTEXTS:
        raise RuntimeError("control training surface count mismatch")
    torch.manual_seed(int(seed))
    model = model_module.build_model(arm, int(seed), torch_module=torch, device=device)
    optimizer = _optimizer(model, torch=torch)
    order, order_sha = _data_order(train_rows, int(seed), torch)
    initial_sha = model_module.model_state_sha(model)
    architecture = model_module.architecture_id(arm)
    if target_updates is None:
        target_updates = protocol.OFFICIAL_UPDATES if mode == "official" else protocol.CONTROL_UPDATES
    if target_updates <= 0:
        raise ValueError("target_updates must be positive")
    identity = {
        "mode": mode,
        "arm": arm,
        "seed": int(seed),
        "data_manifest_sha256": surface_sha,
        "data_order_sha256": order_sha,
        "protocol_sha256": protocol.protocol_sha256(),
        "architecture_id": architecture,
        "target_updates": int(target_updates),
    }
    root = out_dir / mode / arm / protocol.seed_label(int(seed))
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "resume.pt"
    state: dict[str, Any] = {
        "updates": 0,
        "processed_tokens": 0,
        "supervised_tokens": 0,
        "stream_cursor": 0,
        "trace": [],
        "clip_events": 0,
        "timing": {"train_seconds": 0.0, "eval_seconds": 0.0, "checkpoint_seconds": 0.0},
        "last_eval_axis": 0,
        "last_checkpoint_axis": 0,
        "resume_count": 0,
    }
    resume_checkpoint_sha = None
    if checkpoint.exists():
        resume_checkpoint_sha = file_sha256(checkpoint)
        saved = _load_checkpoint(checkpoint, model=model, optimizer=optimizer, torch=torch, identity=identity)
        if saved.get("initial_model_sha256") != initial_sha:
            raise RuntimeError("initial model state changed across resume")
        for key in state:
            if key in saved:
                state[key] = saved[key]
        state["resume_count"] = int(state.get("resume_count", 0)) + 1
        if progress:
            progress(f"RESUME {mode}/{arm}/{protocol.seed_label(int(seed))}: {state['updates']} updates")
    started = time.monotonic()
    eval_every = protocol.OFFICIAL_EVAL_EVERY if mode == "official" else protocol.CONTROL_EVAL_EVERY
    checkpoint_every = protocol.OFFICIAL_CHECKPOINT_EVERY if mode == "official" else protocol.CONTROL_CHECKPOINT_EVERY
    eligible_from = protocol.OFFICIAL_ELIGIBLE_FROM if mode == "official" else protocol.CONTROL_ELIGIBLE_FROM
    while state["updates"] < int(target_updates):
        if deadline_epoch is not None and time.time() >= float(deadline_epoch) and stop_after is None:
            next_checkpoint = ((int(state["updates"]) // checkpoint_every) + 1) * checkpoint_every
            stop_after = min(int(target_updates), int(next_checkpoint))
        selected: list[Mapping[str, Any]] = []
        cursor = int(state["stream_cursor"])
        for offset in range(protocol.BATCH_ROWS):
            selected.append(train_rows[order[(cursor + offset) % len(order)]])
        tokens, segments, eligible, supervised, processed = build_batch(selected, mode=mode, torch=torch, device=device)
        from v5_model.core import packed_layout
        from v5_objectives.causal_lm import causal_lm_loss
        positions, mask = packed_layout(segments, torch_module=torch)
        mask = mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        update_started = time.monotonic()
        logits = model(tokens, positions, mask)
        loss, loss_tokens = causal_lm_loss(
            logits,
            tokens,
            segments,
            bos_id=2,
            pad_id=0,
            eligible=eligible,
            torch_module=torch,
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("NONFINITE_LOSS")
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        if not gradients or any(gradient is None for gradient in gradients):
            raise RuntimeError("missing trainable gradient")
        if any(not bool(torch.isfinite(gradient).all().item()) for gradient in gradients if gradient is not None):
            raise RuntimeError("NONFINITE_GRADIENT")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norm_value = float(grad_norm.item())
        if not math.isfinite(grad_norm_value):
            raise RuntimeError("NONFINITE_GRADIENT")
        optimizer.step()
        state["timing"]["train_seconds"] += time.monotonic() - update_started
        state["updates"] += 1
        state["processed_tokens"] += processed
        state["supervised_tokens"] += supervised
        state["stream_cursor"] = (cursor + len(selected)) % len(order)
        if grad_norm_value > 1.0:
            state["clip_events"] += 1
        if state["updates"] % eval_every == 0 or state["updates"] >= int(target_updates) or (stop_after is not None and state["updates"] >= int(stop_after)):
            eval_started = time.monotonic()
            evaluation = evaluate_development(model, dev_rows, mode=mode, torch=torch, device=device)
            state["timing"]["eval_seconds"] += time.monotonic() - eval_started
            state["trace"].append({"axis": state["updates"], **evaluation})
            state["last_eval_axis"] = state["updates"]
        should_checkpoint = state["updates"] % checkpoint_every == 0 or state["updates"] >= int(target_updates) or (stop_after is not None and state["updates"] >= int(stop_after))
        if should_checkpoint:
            checkpoint_started = time.monotonic()
            state["last_checkpoint_axis"] = state["updates"]
            payload = {
                **identity,
                "initial_model_sha256": initial_sha,
                "parameterization": model_module.parameterization_receipt(model),
                "current_model_sha256": model_module.model_state_sha(model),
                "target_updates": int(target_updates),
                "resume_checkpoint_sha256": resume_checkpoint_sha,
                **state,
            }
            checkpoint_sha = _save_checkpoint(checkpoint, model=model, optimizer=optimizer, torch=torch, payload=payload)
            _write_progress(root, payload, checkpoint_sha)
            state["timing"]["checkpoint_seconds"] += time.monotonic() - checkpoint_started
            resume_checkpoint_sha = checkpoint_sha
        if progress and state["updates"] % max(eval_every, 1) == 0:
            progress(f"{mode}/{arm}/{protocol.seed_label(int(seed))}: updates={state['updates']} loss={float(loss.item()):.6f}")
        if stop_after is not None and state["updates"] >= int(stop_after):
            break
    formation = formation_summary(state["trace"], eligible_from)
    status = "COMPLETE" if state["updates"] >= int(target_updates) else "PARTIAL"
    result = {
        "schema": "anra.x-factor-pilot-arm-result/v1",
        **identity,
        "status": status,
        "target_updates": int(target_updates),
        "stop_after": stop_after,
        "deadline_epoch": deadline_epoch,
        "updates": int(state["updates"]),
        "processed_tokens": int(state["processed_tokens"]),
        "supervised_tokens": int(state["supervised_tokens"]),
        "stream_cursor": int(state["stream_cursor"]),
        "formation": formation,
        "trace": state["trace"],
        "clip_fraction": state["clip_events"] / max(state["updates"], 1),
        "timing": state["timing"],
        "wall_seconds": time.monotonic() - started,
        "parameterization": model_module.parameterization_receipt(model),
        "initial_model_sha256": initial_sha,
        "final_model_sha256": model_module.model_state_sha(model),
        "resume_checkpoint_sha256": resume_checkpoint_sha,
        "resume_count": int(state["resume_count"]),
    }
    atomic_json(root / "ARM_RESULT.json", result)
    return result


def load_model_for_evaluation(
    checkpoint: Path,
    *,
    mode: str,
    arm: str,
    seed: int,
    data_manifest_sha256: str,
    torch: Any,
    device: Any,
) -> Any:
    receipt_path = checkpoint.parent / "CHECKPOINT_RECEIPT.json"
    if not receipt_path.exists():
        raise RuntimeError("CHECKPOINT_RECEIPT_MISSING")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "anra.x-factor-pilot-checkpoint-receipt/v1" or receipt.get("sha256") != file_sha256(checkpoint):
        raise RuntimeError("CHECKPOINT_RECEIPT_MISMATCH")
    model = model_module.build_model(arm, int(seed), torch_module=torch, device=device)
    body = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(body, dict) or body.get("schema") != "anra.x-factor-pilot-checkpoint/v1":
        raise RuntimeError("CHECKPOINT_SCHEMA_MISMATCH")
    expected = {
        "mode": mode,
        "arm": arm,
        "seed": int(seed),
        "data_manifest_sha256": data_manifest_sha256,
        "protocol_sha256": protocol.protocol_sha256(),
        "architecture_id": model_module.architecture_id(arm),
        "target_updates": protocol.OFFICIAL_UPDATES if mode == "official" else protocol.CONTROL_UPDATES,
    }
    for key, value in expected.items():
        if body.get(key) != value or receipt.get(key) != value:
            raise RuntimeError(f"checkpoint identity mismatch for {key}")
    if int(body.get("updates", -1)) != int(expected["target_updates"]):
        raise RuntimeError("checkpoint update target mismatch")
    if not isinstance(body.get("model"), dict):
        raise RuntimeError("CHECKPOINT_PAYLOAD_MALFORMED")
    model.load_state_dict(body["model"])
    if body.get("current_model_sha256") != model_module.model_state_sha(model):
        raise RuntimeError("CHECKPOINT_MODEL_HASH_MISMATCH")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
