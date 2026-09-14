"""Science-S5 training binding with exact-resume progress snapshots."""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any, Mapping
from anra_v5 import formation_mux_train_v2 as _base
from anra_v5 import formation_mux_model_v3 as fxm
from v5_experiments import formation_mux_protocol_v5 as proto
from v5_experiments.formation_mux_surface_v5 import validate_public_surface

_base.fxm = fxm
_base.proto = proto

build_batch = _base.build_batch
evaluate_development = _base.evaluate_development
load_model_for_evaluation = _base.load_model_for_evaluation
_encode_row = _base._encode_row
_row_processed_tokens = _base._row_processed_tokens
_greedy_rates = _base._greedy_rates
_ORIGINAL_SAVE = _base._save_checkpoint


def _worker_surface(public_surface: Mapping[str, Any]) -> dict[str, Any]:
    validate_public_surface(public_surface)
    splits = dict(public_surface["splits"])
    if "sealed" in splits:
        raise RuntimeError("SEALED_FIREWALL_BREACH: S5 worker received sealed rows")
    return {**dict(public_surface), "splits": {**splits, "sealed": []}}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _save_checkpoint_with_progress(path: Path, *, model: Any, optimizers: Mapping[str, Any], torch: Any, payload: Mapping[str, Any]) -> str:
    digest = _ORIGINAL_SAVE(
        path, model=model, optimizers=optimizers, torch=torch, payload=payload
    )
    experiment = str(payload["experiment"])
    eligible_from = (
        proto.A_ELIGIBLE_FROM_UPDATE
        if experiment == proto.EXPERIMENT_A
        else proto.B_ELIGIBLE_FROM_TOKENS
    )
    trace = list(payload.get("trace", []))
    formation = _base._formation_summary(trace, eligible_from)
    updates = int(payload.get("updates", 0))
    progress = {
        "schema": "anra.formation-mux-progress/v5",
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
        "diagnostic_only": True,
        "may_not_change_frozen_science": True,
    }
    _atomic_json(path.parent / "LATEST_PROGRESS.json", progress)
    _atomic_json(path.parent / "progress" / f"UPDATE_{updates:08d}.json", progress)
    return digest


def train_arm(**kwargs):
    _base.fxm = fxm
    _base.proto = proto
    _base._save_checkpoint = _save_checkpoint_with_progress
    if "surface" not in kwargs:
        raise RuntimeError("S5 public worker surface missing")
    kwargs = dict(kwargs)
    kwargs["surface"] = _worker_surface(kwargs["surface"])
    return _base.train_arm(**kwargs)
