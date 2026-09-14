"""FORMATION-MUX-001 Amendment-1 training/evaluation path.

Key invariants:
- CS-MECH primary metric is IDENTITY family only.
- REP-FORM is stopped by processed non-padding token exposure, not equal steps.
- frozen extra-row gradients are zeroed before the one global clip.
- the tied row optimizer is stepped only by ProductionTrainingBackend after clip.
- sealed evaluation can restore model weights without reconstructing optimizer topology.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from anra_v5 import formation_mux_model_v2 as fxm
from v5_experiments import formation_mux_protocol_v2 as proto


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _model_sha(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _encode_row(row: Mapping[str, Any], experiment: str, arm: str) -> tuple[list[int], list[int]]:
    if experiment == proto.EXPERIMENT_A or arm == "R1_ISOMORPHIC_RENDERING":
        prompt = [2, *[int(x) for x in row["prompt_ids"]]]
        answer = [*[int(x) for x in row["answer_ids"]], 3]
        fxm.assert_latent_ids_are_shared(prompt[1:] + answer[:-1])
        return prompt, answer
    if "r0_prompt_ids" not in row or "r0_answer_ids" not in row:
        raise RuntimeError("R0 token ids missing: frozen production tokenizer surface is required")
    return [int(x) for x in row["r0_prompt_ids"]], [int(x) for x in row["r0_answer_ids"]]


def _row_processed_tokens(row: Mapping[str, Any], experiment: str, arm: str) -> int:
    p, a = _encode_row(row, experiment, arm)
    return len(p) + len(a)


def build_batch(
    rows: list[Mapping[str, Any]],
    experiment: str,
    arm: str,
    *,
    torch: Any,
    device: Any,
) -> tuple[Any, Any, Any, int, int]:
    encoded = [_encode_row(r, experiment, arm) for r in rows]
    width = max(len(p) + len(a) for p, a in encoded)
    tokens: list[list[int]] = []
    segments: list[list[int]] = []
    eligible: list[list[bool]] = []
    supervised = 0
    processed = 0
    for prompt, answer in encoded:
        ids = prompt + answer
        pad_n = width - len(ids)
        tokens.append(ids + [0] * pad_n)
        segments.append([0] * len(ids) + [-1] * pad_n)
        eligible.append([False] * len(prompt) + [True] * len(answer) + [False] * pad_n)
        supervised += len(answer)
        processed += len(ids)
    return (
        torch.tensor(tokens, dtype=torch.long, device=device),
        torch.tensor(segments, dtype=torch.long, device=device),
        torch.tensor(eligible, dtype=torch.bool, device=device),
        supervised,
        processed,
    )


def _greedy_rates(
    model: Any,
    rows: list[Mapping[str, Any]],
    experiment: str,
    arm: str,
    *,
    torch: Any,
    device: Any,
) -> dict[str, float]:
    from v5_model.core import packed_layout

    was_training = model.training
    model.eval()
    complete = content = stopped_count = cap_count = 0
    with torch.no_grad():
        for row in rows:
            prompt, answer = _encode_row(row, experiment, arm)
            expected = [x for x in answer if x != 3]
            generated: list[int] = []
            stopped = False
            for _ in range(proto.MAX_GENERATION_TOKENS):
                current = torch.tensor([prompt + generated], dtype=torch.long, device=device)
                seg = torch.zeros_like(current)
                positions, mask = packed_layout(seg, torch_module=torch)
                logits = model(current, positions, mask)[0, -1]
                nxt = int(torch.argmax(logits).item())
                if nxt == 3:
                    stopped = True
                    break
                if nxt == 0:
                    break
                generated.append(nxt)
            else:
                cap_count += 1
            if generated == expected:
                content += 1
                if stopped:
                    complete += 1
            stopped_count += int(stopped)
    if was_training:
        model.train()
    n = max(len(rows), 1)
    return {
        "content_exact": content / n,
        "complete_exact_with_valid_stop": complete / n,
        "eos_rate": stopped_count / n,
        "max_tokens_rate": cap_count / n,
        "n": len(rows),
    }


def evaluate_development(
    model: Any,
    dev_rows: list[Mapping[str, Any]],
    experiment: str,
    arm: str,
    *,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    families = sorted({str(r["family"]) for r in dev_rows})
    per_family: dict[str, dict[str, float]] = {}
    for family in families:
        subset = [r for r in dev_rows if r["family"] == family]
        per_family[family] = _greedy_rates(
            model, subset, experiment, arm, torch=torch, device=device
        )
    if "identity" not in per_family:
        raise RuntimeError("development surface missing identity family")
    return {
        "identity_exact_valid_eos": per_family["identity"]["complete_exact_with_valid_stop"],
        "per_family": per_family,
    }


def _formation_summary(trace: list[dict[str, Any]], eligible_from: int) -> dict[str, Any]:
    points = [
        (int(x["axis"]), float(x["identity_exact_valid_eos"]))
        for x in trace
        if int(x["axis"]) >= int(eligible_from)
    ]
    if not points:
        return {"formation_auc": 0.0, "endpoint": 0.0, "first_acquisition_axis": None, "points": 0}
    if len(points) == 1:
        auc = points[0][1]
    else:
        area = 0.0
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            area += (x1 - x0) * (y0 + y1) / 2.0
        span = max(points[-1][0] - points[0][0], 1)
        auc = area / span
    first = next((axis for axis, score in points if score >= 0.5), None)
    return {
        "formation_auc": round(float(auc), 6),
        "endpoint": round(float(points[-1][1]), 6),
        "first_acquisition_axis": first,
        "points": len(points),
    }


def _save_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizers: Mapping[str, Any],
    torch: Any,
    payload: Mapping[str, Any],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "model": model.state_dict(),
        "main_optimizer": optimizers["main"].state_dict(),
        "row_optimizer": optimizers["rows"].state_dict() if optimizers.get("rows") else None,
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        **dict(payload),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(body, tmp)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    os.replace(tmp, path)
    (path.parent / "CHECKPOINT_RECEIPT.json").write_text(
        json.dumps({"sha256": digest, **{k: v for k, v in payload.items() if k not in {"trace", "diagnostics"}}}, indent=2, default=str),
        encoding="utf-8",
    )
    return digest


def _load_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizers: Mapping[str, Any],
    torch: Any,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    body = torch.load(path, map_location="cpu", weights_only=False)
    for key, value in expected.items():
        if body.get(key) != value:
            raise RuntimeError(f"checkpoint identity mismatch {key}: {body.get(key)!r} != {value!r}")
    model.load_state_dict(body["model"])
    optimizers["main"].load_state_dict(body["main_optimizer"])
    if optimizers.get("rows") is not None:
        if body.get("row_optimizer") is None:
            raise RuntimeError("checkpoint identity mismatch: row optimizer state missing")
        optimizers["rows"].load_state_dict(body["row_optimizer"])
    if body.get("cpu_rng_state") is not None:
        torch.set_rng_state(body["cpu_rng_state"])
    if torch.cuda.is_available() and body.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(body["cuda_rng_state"])
    return body


def load_model_for_evaluation(
    checkpoint: Path,
    *,
    experiment: str,
    arm: str,
    seed_bundle: int,
    data_manifest_sha256: str,
    torch: Any,
    device: Any,
) -> Any:
    """Verified model-only restore for sealed scoring; no optimizer topology reconstruction."""

    from v5_model.core import initialize

    seed = seed_bundle if experiment == proto.EXPERIMENT_A else proto.b_seed(seed_bundle)
    model = (
        fxm.build_model(seed, arm, torch=torch, device=device)
        if experiment == proto.EXPERIMENT_A
        else initialize(fxm.spec(), int(seed), torch_module=torch).to(device)
    )
    body = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = {
        "experiment": experiment,
        "arm": arm,
        "seed_bundle": seed_bundle,
        "data_manifest_sha256": data_manifest_sha256,
        "protocol_sha256": proto.protocol_sha(experiment),
    }
    for key, value in expected.items():
        if body.get(key) != value:
            raise RuntimeError(f"checkpoint identity mismatch {key}: {body.get(key)!r} != {value!r}")
    model.load_state_dict(body["model"])
    model.eval()
    return model


def _make_model_and_optimizers(experiment: str, arm: str, seed: int, *, torch: Any, device: Any):
    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer

    if experiment == proto.EXPERIMENT_A:
        model = fxm.build_model(seed, arm, torch=torch, device=device)
        optimizers = fxm.make_optimizers(model, arm, torch=torch, lr=proto.LR)
    else:
        model = initialize(fxm.spec(), int(seed), torch_module=torch).to(device)
        main = build_adamw_optimizer(model, torch_module=torch, lr=proto.LR)
        optimizers = {"main": main, "rows": None, "view": main}
    return model, optimizers


def _batch_rows(
    train_rows: list[Mapping[str, Any]],
    order: list[int],
    cursor: int,
    *,
    experiment: str,
    arm: str,
    remaining_processed: int | None,
) -> tuple[list[Mapping[str, Any]], int]:
    selected: list[Mapping[str, Any]] = []
    consumed = 0
    for offset in range(proto.BATCH_ROWS):
        row = train_rows[order[(cursor + offset) % len(order)]]
        cost = _row_processed_tokens(row, experiment, arm)
        if remaining_processed is not None and selected and consumed + cost > remaining_processed:
            break
        selected.append(row)
        consumed += cost
        if remaining_processed is not None and consumed >= remaining_processed:
            break
    if not selected:
        raise RuntimeError("deterministic batch planner produced no rows")
    return selected, cursor + len(selected)


def train_arm(
    *,
    experiment: str,
    arm: str,
    seed_bundle: int,
    surface: Mapping[str, Any],
    out_dir: Path,
    torch: Any,
    device: Any,
    engineering_only: bool = False,
    a_updates_override: int | None = None,
    b_token_budget_override: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    from v5_training.production_backend import ProductionTrainingBackend
    from v5_training.state import CURSOR_SCHEMA, CursorState

    if experiment not in proto.EXPERIMENTS:
        raise RuntimeError(f"experiment not registered: {experiment}")
    valid_arms = proto.ARMS_A if experiment == proto.EXPERIMENT_A else proto.ARMS_B
    if arm not in valid_arms:
        raise RuntimeError(f"arm not registered: {experiment}/{arm}")
    official = seed_bundle in proto.SEED_BUNDLES
    if not official and not engineering_only:
        raise RuntimeError(f"seed bundle not preregistered: {seed_bundle}")
    if experiment == proto.EXPERIMENT_B:
        for split in ("training", "development", "sealed"):
            if any("r0_prompt_ids" not in r or "r0_answer_ids" not in r for r in surface["splits"][split]):
                raise RuntimeError("R0 token ids missing: production-tokenizer surface invalid")

    seed = seed_bundle if experiment == proto.EXPERIMENT_A else proto.b_seed(seed_bundle)
    torch.manual_seed(int(seed))
    model, optimizers = _make_model_and_optimizers(experiment, arm, seed, torch=torch, device=device)
    backend = ProductionTrainingBackend(
        model=model,
        optimizer=optimizers["view"],
        bos_id=2,
        pad_id=0,
        device=device,
        schedule=lambda cumulative_tokens: proto.LR,
        bfloat16_autocast=False,
        torch_module=torch,
        activation_checkpointing=False,
    )

    train_rows = list(surface["splits"]["training"])
    dev_rows = list(surface["splits"]["development"])
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(train_rows), generator=generator).tolist()
    data_sha = str(surface["sha256"])
    target_updates = (
        int(a_updates_override) if a_updates_override is not None else proto.A_UPDATES
    )
    target_processed = (
        int(b_token_budget_override)
        if b_token_budget_override is not None
        else proto.B_PROCESSED_TOKEN_BUDGET
    )
    identity = {
        "experiment": experiment,
        "arm": arm,
        "seed_bundle": seed_bundle,
        "data_manifest_sha256": data_sha,
        "protocol_sha256": proto.protocol_sha(experiment),
        "engineering_only": bool(engineering_only),
    }

    seed_label = f"S{proto.SEED_BUNDLES.index(seed_bundle) + 1}" if official else f"CAL{seed_bundle}"
    root = Path(out_dir) / experiment / arm / seed_label
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "resume.pt"
    initial_sha = _model_sha(model)
    state = {
        "updates": 0,
        "processed_tokens": 0,
        "supervised_tokens": 0,
        "stream_cursor": 0,
        "trace": [],
        "diagnostics": [],
        "timing": {"train_seconds": 0.0, "eval_seconds": 0.0, "checkpoint_seconds": 0.0},
        "last_eval_axis": 0,
        "last_checkpoint_axis": 0,
        "clip_events": 0,
    }
    if checkpoint.exists():
        saved = _load_checkpoint(
            checkpoint,
            model=model,
            optimizers=optimizers,
            torch=torch,
            expected=identity,
        )
        if saved.get("initial_model_sha256") != initial_sha:
            raise RuntimeError("checkpoint identity drift: initial model hash changed")
        for key in state:
            if key in saved:
                state[key] = saved[key]
        if progress:
            progress(f"RESUME {experiment}/{arm}/{seed_label}: update={state['updates']} processed={state['processed_tokens']}")

    started = time.monotonic()
    while True:
        if experiment == proto.EXPERIMENT_A:
            if state["updates"] >= target_updates:
                break
            remaining = None
        else:
            if state["processed_tokens"] >= target_processed:
                break
            remaining = target_processed - state["processed_tokens"]

        rows, next_cursor = _batch_rows(
            train_rows,
            order,
            int(state["stream_cursor"]),
            experiment=experiment,
            arm=arm,
            remaining_processed=remaining,
        )
        tokens, segments, eligible, supervised, processed = build_batch(
            rows, experiment, arm, torch=torch, device=device
        )

        diag_due = state["updates"] % 100 == 0
        embedding_before = model.embedding.weight.detach().clone() if diag_due else None
        update_started = time.monotonic()
        ctx = backend.begin_update(type("S", (), {"cumulative_tokens": state["supervised_tokens"]})())
        ctx = backend.accumulate_microstep(
            ctx,
            tokens=tokens,
            segment_ids=segments,
            eligible=eligible,
            tokens_by_source={"mux": supervised},
            planned_total=supervised,
        )
        grad = model.embedding.weight.grad
        if grad is None:
            raise RuntimeError("embedding gradient missing after backward")
        diag: dict[str, Any] | None = None
        if diag_due:
            active_grad = float(grad[: fxm.SHARED_VOCAB].float().norm().item())
            extra_grad = float(grad[fxm.SHARED_VOCAB :].float().norm().item())
            core_sq = 0.0
            for name, p in model.named_parameters():
                if p is model.embedding.weight or p.grad is None:
                    continue
                core_sq += float(p.grad.detach().float().norm().item()) ** 2
            diag = {
                "update": state["updates"] + 1,
                "processed_tokens_before": state["processed_tokens"],
                "shared_row_grad_norm_pre_mask": active_grad,
                "extra_row_grad_norm_pre_mask": extra_grad,
                "shared_core_grad_norm_pre_clip": math.sqrt(core_sq),
            }
        fxm.mask_frozen_gradients_before_clip(model, arm) if experiment == proto.EXPERIMENT_A else None
        report = backend.finish_update(
            type("S", (), {"cumulative_tokens": state["supervised_tokens"]})(),
            ctx,
            planned_total=supervised,
            cursor=CursorState(CURSOR_SCHEMA, data_sha, state["updates"] + 1, 0, 0),
        )
        state["timing"]["train_seconds"] += time.monotonic() - update_started
        state["updates"] += 1
        state["processed_tokens"] += processed
        state["supervised_tokens"] += supervised
        state["stream_cursor"] = next_cursor
        receipt = backend.last_receipt or {}
        if float(receipt.get("grad_norm_pre_clip", 0.0)) > 1.0:
            state["clip_events"] += 1
        if diag is not None and embedding_before is not None:
            after = model.embedding.weight.detach()
            shared_w = embedding_before[: fxm.SHARED_VOCAB].float().norm().item()
            shared_delta = (after[: fxm.SHARED_VOCAB] - embedding_before[: fxm.SHARED_VOCAB]).float().norm().item()
            extra_w = embedding_before[fxm.SHARED_VOCAB :].float().norm().item()
            extra_delta = (after[fxm.SHARED_VOCAB :] - embedding_before[fxm.SHARED_VOCAB :]).float().norm().item()
            diag.update(
                {
                    "shared_update_weight_ratio": shared_delta / max(shared_w, 1e-30),
                    "extra_update_weight_ratio": extra_delta / max(extra_w, 1e-30),
                    "grad_norm_pre_clip": receipt.get("grad_norm_pre_clip"),
                    "grad_norm_post_clip": receipt.get("grad_norm_post_clip"),
                }
            )
            state["diagnostics"].append(diag)

        axis = state["updates"] if experiment == proto.EXPERIMENT_A else state["processed_tokens"]
        eval_interval = proto.A_EVAL_EVERY_UPDATES if experiment == proto.EXPERIMENT_A else proto.B_EVAL_EVERY_TOKENS
        final_now = (
            state["updates"] >= target_updates
            if experiment == proto.EXPERIMENT_A
            else state["processed_tokens"] >= target_processed
        )
        if axis - state["last_eval_axis"] >= eval_interval or final_now:
            t0 = time.monotonic()
            dev = evaluate_development(model, dev_rows, experiment, arm, torch=torch, device=device)
            state["timing"]["eval_seconds"] += time.monotonic() - t0
            state["trace"].append({"axis": int(axis), **dev})
            state["last_eval_axis"] = int(axis)

        ckpt_interval = proto.A_CHECKPOINT_EVERY_UPDATES if experiment == proto.EXPERIMENT_A else proto.B_CHECKPOINT_EVERY_TOKENS
        if axis - state["last_checkpoint_axis"] >= ckpt_interval or final_now:
            t0 = time.monotonic()
            payload = {
                **identity,
                "initial_model_sha256": initial_sha,
                **state,
            }
            _save_checkpoint(
                checkpoint,
                model=model,
                optimizers=optimizers,
                torch=torch,
                payload=payload,
            )
            state["timing"]["checkpoint_seconds"] += time.monotonic() - t0
            state["last_checkpoint_axis"] = int(axis)

        if progress and state["updates"] % 100 == 0:
            progress(
                f"{experiment}/{arm}/{seed_label}: updates={state['updates']} processed={state['processed_tokens']}"
            )

    eligible_from = (
        proto.A_ELIGIBLE_FROM_UPDATE
        if experiment == proto.EXPERIMENT_A
        else proto.B_ELIGIBLE_FROM_TOKENS
    )
    formation = _formation_summary(state["trace"], eligible_from)
    result = {
        "schema": "anra.formation-mux-arm-result/v2",
        **identity,
        "status": "COMPLETE",
        "updates": state["updates"],
        "processed_tokens": state["processed_tokens"],
        "supervised_tokens": state["supervised_tokens"],
        "stream_cursor": state["stream_cursor"],
        "formation": formation,
        "trace": state["trace"],
        "diagnostics": state["diagnostics"],
        "clip_fraction": state["clip_events"] / max(state["updates"], 1),
        "timing": state["timing"],
        "wall_seconds": time.monotonic() - started,
        "exposure_target": target_updates if experiment == proto.EXPERIMENT_A else target_processed,
        "exposure_axis": "updates" if experiment == proto.EXPERIMENT_A else "processed_nonpadding_tokens",
    }
    final_path = root / "ARM_RESULT.json"
    tmp = final_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, final_path)
    return result
