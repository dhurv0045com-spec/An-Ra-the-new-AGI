"""Exact-resume training binding for the preregistered TIE-ROLE frontier.

Reuses the audited FORMATION-MUX training transaction while rebinding protocol
and model construction. Workers receive only training+development rows.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from anra_v5 import formation_mux_train_v2 as _base
from anra_v5 import tie_role_model_v1 as fxm
from v5_experiments import tie_role_protocol_v1 as proto
from v5_experiments.formation_mux_surface_v5 import validate_public_surface

_BASE_SAVE = _base._save_checkpoint


def _bind() -> None:
    _base.fxm = fxm
    _base.proto = proto
    _base._make_model_and_optimizers = _make_model_and_optimizers


def _worker_surface(public_surface: Mapping[str, Any]) -> dict[str, Any]:
    validate_public_surface(public_surface)
    splits = dict(public_surface["splits"])
    if "sealed" in splits:
        raise RuntimeError("SEALED_FIREWALL_BREACH: frontier worker received sealed rows")
    return {**dict(public_surface), "splits": {**splits, "sealed": []}}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _make_model_and_optimizers(experiment: str, arm: str, seed: int, *, torch: Any, device: Any):
    model = fxm.build_model(seed, arm, torch=torch, device=device)
    optimizers = fxm.make_optimizers(model, arm, torch=torch, lr=proto.LR)
    return model, optimizers


def _save_checkpoint_with_progress(path: Path, *, model: Any, optimizers: Mapping[str, Any], torch: Any, payload: Mapping[str, Any]) -> str:
    digest = _BASE_SAVE(path, model=model, optimizers=optimizers, torch=torch, payload=payload)
    experiment = str(payload["experiment"])
    eligible_from = proto.A_ELIGIBLE_FROM_UPDATE if experiment == proto.EXPERIMENT_A else proto.B_ELIGIBLE_FROM_TOKENS
    trace = list(payload.get("trace", []))
    formation = _base._formation_summary(trace, eligible_from)
    updates = int(payload.get("updates", 0))
    progress = {
        "schema": "anra.tie-role-progress/v1",
        "extension": proto.EXTENSION,
        "experiment": experiment,
        "arm": payload.get("arm"),
        "seed_bundle": payload.get("seed_bundle"),
        "protocol_sha256": payload.get("protocol_sha256"),
        "data_manifest_sha256": payload.get("data_manifest_sha256"),
        "updates": updates,
        "processed_tokens": int(payload.get("processed_tokens", 0)),
        "supervised_tokens": int(payload.get("supervised_tokens", 0)),
        "latest_development": trace[-1] if trace else None,
        "formation_so_far": formation,
        "clip_events": int(payload.get("clip_events", 0)),
        "timing": payload.get("timing", {}),
        "checkpoint_sha256": digest,
        "checkpoint_file": path.name,
        "science_s5_unchanged": True,
    }
    _atomic_json(path.parent / "LATEST_PROGRESS.json", progress)
    axis = updates if experiment == proto.EXPERIMENT_A else int(payload.get("processed_tokens", 0))
    _atomic_json(path.parent / "progress" / f"AXIS_{axis:09d}.json", progress)
    return digest


def train_arm(**kwargs):
    _bind()
    _base._save_checkpoint = _save_checkpoint_with_progress
    if "surface" not in kwargs:
        raise RuntimeError("frontier public worker surface missing")
    kwargs = dict(kwargs)
    kwargs["surface"] = _worker_surface(kwargs["surface"])
    return _base.train_arm(**kwargs)


def build_batch(*args, **kwargs):
    _bind()
    return _base.build_batch(*args, **kwargs)


def evaluate_development(*args, **kwargs):
    _bind()
    return _base.evaluate_development(*args, **kwargs)


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
    _bind()
    if experiment not in proto.EXPERIMENTS:
        raise RuntimeError(f"experiment not registered: {experiment}")
    seed = seed_bundle if experiment == proto.EXPERIMENT_A else proto.b_seed(seed_bundle)
    model = fxm.build_model(seed, arm, torch=torch, device=device)
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


# Expose the deterministic helpers after binding for diagnostics.
_bind()
_encode_row = _base._encode_row
_row_processed_tokens = _base._row_processed_tokens
_formation_summary = _base._formation_summary
