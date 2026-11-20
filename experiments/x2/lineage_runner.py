"""Explicit lineage, frozen-probe, task-adaptation, horizon-continuation & single-arm routes for Elendril V8-X2."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from anra_v5.elendril.codec import digest
from v5_contracts.model_spec import ModelSpec
from anra_v5.elendril.v8x.contracts import X1ModelSpec, build_x1_model_spec
from anra_v5.elendril.v8x.models import X1TypedModel, build_x1_s20_model
from anra_v5.planthir.runtime import (
    CURSOR_SCHEMA,
    CursorState,
    GenericExperimentTrainer,
    GenericTrainRecipe,
    IDENTITY_SCHEMA,
    IdentityBindings,
    TrainingState,
    UTF8,
    _dump,
    _load,
    build_adamw_optimizer,
    canonical,
    optimizer_group_receipt,
    quarantine_interrupted_publication,
    runtime_identity,
    validate_parameter_ownership,
    writer_lock,
)
from v5_model.initialize import NORMAL_STD
from v5_training.checkpoint import CheckpointStore

from ..elendril_x2 import _encode_batch_samples, x2_dataset_factory, x2_loss_fn
from .models import (
    X2ForwardOutput,
    X2ModelSpec,
    X2TypedModel,
    build_s20_base_spec,
    build_tiny_canary_base_spec,
    build_x2_model,
    build_x2_model_spec,
)
from .tasks import (
    T13_DOSES,
    X2EnvironmentKernel,
    X2ObservablePolicy,
    build_generator_b_multihop_panel,
    build_selective_edit_panel,
    build_t13_episode_stream,
    build_t14_transition_stream,
    select_balanced_matched_cohort,
    solve_t14_public_dp,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
A0_REFERENCE_METRICS_PATH = REPO_ROOT / "Experiment Result" / "V8-X1" / "A0" / "METRICS.json"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    body = "".join(json.dumps(dict(r), sort_keys=True, allow_nan=False) + "\n" for r in rows)
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)


def weights_state_digest(model: nn.Module) -> str:
    """Deterministic SHA-256 across all parameters & buffers of a model."""
    h = sha256()
    sd = model.state_dict()
    for name in sorted(sd):
        t = sd[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


class X1AdaptationWrapper(nn.Module):
    """Wraps a loaded X1TypedModel (e.g. A0 R) with X2 provisional heads for lossless parity + X2 task adaptation."""

    def __init__(self, base_model: X1TypedModel, x2_spec: X2ModelSpec, *, head_seed: int = 0) -> None:
        super().__init__()
        self.base = base_model
        self.x2_spec = x2_spec
        w = int(base_model.x1_spec.base_spec.width)
        self.provisional_val_head = nn.Linear(w, 16, bias=True)
        self.provisional_cert_head = nn.Linear(w, 1, bias=True)
        gen = torch.Generator(device="cpu").manual_seed(int(head_seed) + 20_009)
        with torch.no_grad():
            self.provisional_val_head.weight.normal_(mean=0.0, std=NORMAL_STD, generator=gen)
            self.provisional_val_head.bias.zero_()
            self.provisional_cert_head.weight.normal_(mean=0.0, std=NORMAL_STD, generator=gen)
            self.provisional_cert_head.bias.zero_()

    @property
    def core(self) -> nn.Module:
        return self.base.core

    def forward_hidden_only(
        self, token_ids: torch.Tensor, segment_ids: torch.Tensor, *, use_activation_checkpointing: bool = False
    ) -> torch.Tensor:
        return self.base.forward_hidden_only(token_ids, segment_ids, use_activation_checkpointing=use_activation_checkpointing)

    def forward_action(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        action_positions: torch.Tensor,
        *,
        action_batch_indices: torch.Tensor | None = None,
        copy_enabled_override: bool | None = None,
        lesion_mode: str | None = None,
        recurrent_steps_override: int | None = None,
        init_memory: torch.Tensor | None = None,
        block_write_index: int | None = None,
        forecast_override: torch.Tensor | None = None,
        use_activation_checkpointing: bool = False,
        **_: Any,
    ) -> X2ForwardOutput:
        # Auxiliary provisional heads do NOT perturb log_probs_19, log_p_op, log_p_val, or gate.
        _ = (init_memory, block_write_index, forecast_override)
        out = self.base.forward_action(
            token_ids,
            segment_ids,
            action_positions,
            action_batch_indices=action_batch_indices,
            copy_enabled_override=copy_enabled_override,
            lesion_mode=lesion_mode,
            recurrent_steps_override=recurrent_steps_override,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        h = out.decision_hidden
        return X2ForwardOutput(
            log_probs_19=out.log_probs_19,
            log_p_op=out.log_p_op,
            log_p_val=out.log_p_val,
            provisional_val_logits=self.provisional_val_head(h),
            certifiable_logit=self.provisional_cert_head(h).squeeze(-1),
            decision_hidden=h,
            gate=out.gate,
            slot_states=out.slot_states,
        )


def _base_spec_from_dict_or_weights(meta_spec: Mapping[str, Any] | None, weights: Mapping[str, torch.Tensor], tiny_canary: bool) -> ModelSpec:
    if isinstance(meta_spec, Mapping) and isinstance(meta_spec.get("base_spec"), Mapping):
        bs = dict(meta_spec["base_spec"])
        return ModelSpec(**{k: bs[k] for k in ModelSpec.__dataclass_fields__ if k in bs})
    w = int(weights["op_head.weight"].shape[1]) if "op_head.weight" in weights else (16 if tiny_canary else 512)
    return build_tiny_canary_base_spec() if w <= 64 else build_s20_base_spec()


def _is_x1_checkpoint(weights: Mapping[str, Any], meta: Mapping[str, Any], architecture: str | None) -> bool:
    if architecture in ("F-cap", "R-noedge", "P-capR", "J", "F"):
        return True
    if any(k in weights for k in ("agg_mlp_1.weight", "cap_in.weight", "noedge_gru.weight_ih", "joint_head.weight")):
        return True
    ms = meta.get("model_spec", {})
    if isinstance(ms, Mapping) and ms.get("schema") == "elendril-x1-model-spec-v1":
        return True
    if str(meta.get("backend_id", "")) in ("x1_s20", "x1_s20_legacy"):
        return True
    return False


def _reconstruct_x1_spec(
    weights: Mapping[str, torch.Tensor],
    meta: Mapping[str, Any],
    architecture: str | None,
    tiny_canary: bool,
) -> X1ModelSpec:
    ms = meta.get("model_spec") if isinstance(meta.get("model_spec"), Mapping) else None
    arch = str(architecture or (ms.get("architecture") if ms else None) or ("R" if "slots_init" in weights else "P"))
    if ms and ms.get("schema") == "elendril-x1-model-spec-v1":
        bs = _base_spec_from_dict_or_weights(ms, weights, tiny_canary)
        kw = {k: ms[k] for k in X1ModelSpec.__dataclass_fields__ if k in ms and k != "base_spec"}
        if "output_mapping" in kw:
            kw["output_mapping"] = tuple(kw["output_mapping"])
        return X1ModelSpec(base_spec=bs, **kw)
    bs = _base_spec_from_dict_or_weights(ms, weights, tiny_canary)
    ptr_dim = int(weights["q_proj.weight"].shape[0]) if "q_proj.weight" in weights else (8 if bs.width <= 64 else 64)
    if arch in ("R", "R-noedge", "P-capR") and "slots_init" in weights:
        n_slots, s_dim = int(weights["slots_init"].shape[0]), int(weights["slots_init"].shape[1])
        eh = int(weights["edge_mlp_1.weight"].shape[0]) if "edge_mlp_1.weight" in weights else (8 if bs.width <= 64 else 64)
        return build_x1_model_spec(
            arch, base_spec=bs, pointer_dim=ptr_dim, slot_dim=s_dim, num_slots=n_slots, slot_edge_hidden=eh
        )
    return build_x1_model_spec(arch, base_spec=bs, pointer_dim=ptr_dim)


def _reconstruct_x2_spec(
    weights: Mapping[str, torch.Tensor],
    meta: Mapping[str, Any],
    architecture: str | None,
    tiny_canary: bool,
) -> X2ModelSpec:
    ms = meta.get("model_spec") if isinstance(meta.get("model_spec"), Mapping) else None
    arch = str(architecture or (ms.get("architecture") if ms else None) or "P")
    if ms and ms.get("schema") == "elendril-x2-model-spec-v1":
        bs = _base_spec_from_dict_or_weights(ms, weights, tiny_canary)
        kw = {k: ms[k] for k in X2ModelSpec.__dataclass_fields__ if k in ms and k != "base_spec"}
        return X2ModelSpec(base_spec=bs, **kw)
    bs = _base_spec_from_dict_or_weights(ms, weights, tiny_canary)
    is_tiny = tiny_canary or bs.width <= 64
    return build_x2_model_spec(arch, base_spec=bs, tiny_canary=is_tiny)


def _find_lineage_dirs(checkpoint_path: Path) -> list[Path]:
    dirs: list[Path] = []
    if (checkpoint_path / "LATEST").is_file():
        dirs.append(checkpoint_path)
    ckpt_sub = checkpoint_path / "checkpoints"
    search_roots = [ckpt_sub] if ckpt_sub.is_dir() else [checkpoint_path]
    for sr in search_roots:
        for latest in sorted(sr.rglob("LATEST")):
            if latest.is_file() and latest.parent not in dirs:
                dirs.append(latest.parent)
    return dirs


def _select_lineage_dir(lineage_dirs: Sequence[Path], architecture: str | None) -> Path:
    if len(lineage_dirs) == 1:
        return lineage_dirs[0]
    if architecture:
        want = str(architecture)
        matches: list[Path] = []
        for ldir in lineage_dirs:
            store = CheckpointStore(ldir.parent, ldir.name)
            sha = store.latest_sha256()
            if sha is None:
                continue
            try:
                _, payloads = store.restore(sha)
                sched = json.loads(payloads["scheduler.json"])
                meta = sched.get("metadata", sched)
                if str(meta.get("architecture", "")) == want:
                    matches.append(ldir)
            except Exception:
                continue
        if len(matches) == 1:
            return matches[0]
        if matches:
            return matches[-1]
        norm_want = want.replace("-", "_")
        token_matches = [
            d for d in lineage_dirs
            if f"_{norm_want}_" in d.name or f"-{want}-" in d.name or d.name.endswith(f"-{want}")
        ]
        if len(token_matches) == 1:
            return token_matches[0]
    raise ValueError(
        f"AMBIGUOUS_PARENT_CHECKPOINT_LINEAGE: found {[d.name for d in lineage_dirs]} for architecture={architecture!r}"
    )


def _find_direct_weights_file(checkpoint_path: Path) -> Path | None:
    if checkpoint_path.is_file() and checkpoint_path.suffix in (".bin", ".pt"):
        return checkpoint_path
    if checkpoint_path.is_dir():
        for name in ("model.bin", "weights.bin", "model_inference.bin"):
            cand = checkpoint_path / name
            if cand.is_file():
                return cand
    return None


def load_parent_checkpoint(
    checkpoint_path: Path | str | None,
    *,
    architecture: str | None = None,
    tiny_canary: bool = False,
    device: str = "cpu",
    require_x1_native: bool = False,
) -> dict[str, Any]:
    """Load an immutable parent checkpoint (Palantiri CheckpointStore, X1 job dir, or .bin) failing closed if missing."""
    if checkpoint_path is None or not str(checkpoint_path).strip():
        raise FileNotFoundError("MISSING_PARENT_CHECKPOINT: empty path")
    p = Path(checkpoint_path)
    if not p.exists():
        raise FileNotFoundError(f"MISSING_PARENT_CHECKPOINT: {p}")

    sched: dict[str, Any] = {}
    meta: dict[str, Any] = {}
    optimizer_state: dict[str, Any] | None = None
    parent_global_update = 0
    parent_sha = ""
    parent_source_commit = "0" * 40
    ckpt_format = "direct_weights"
    resolved_lineage_dir: str | None = None

    lineage_dirs = _find_lineage_dirs(p) if p.is_dir() else []
    if lineage_dirs:
        ldir = _select_lineage_dir(lineage_dirs, architecture)
        resolved_lineage_dir = str(ldir)
        store = CheckpointStore(ldir.parent, ldir.name)
        sha = store.latest_sha256()
        if sha is None:
            raise FileNotFoundError(f"MISSING_PARENT_CHECKPOINT: LATEST missing in {ldir}")
        state, payloads = store.restore(sha)
        weights = _load(payloads["model.bin"])
        if "optimizer.bin" in payloads:
            optimizer_state = _load(payloads["optimizer.bin"])
        sched = json.loads(payloads["scheduler.json"])
        meta = dict(sched.get("metadata", sched))
        parent_global_update = int(state.global_update)
        parent_sha = str(sha)
        parent_source_commit = str(state.identities.source_commit)
        ckpt_format = "checkpoint_store"
    else:
        wfile = _find_direct_weights_file(p)
        if wfile is None:
            raise FileNotFoundError(f"MISSING_PARENT_CHECKPOINT: no LATEST store or weight file at {p}")
        raw_bytes = wfile.read_bytes()
        weights = _load(raw_bytes)
        parent_sha = sha256(raw_bytes).hexdigest()

    is_x1 = _is_x1_checkpoint(weights, meta, architecture)
    target_arch = str(architecture or meta.get("architecture") or ("R" if "slots_init" in weights else "P"))

    dev = torch.device(device)
    if is_x1:
        x1_spec = _reconstruct_x1_spec(weights, meta, target_arch, tiny_canary)
        model: nn.Module = build_x1_s20_model(x1_spec, 0).to(dev)
        model.load_state_dict(weights, strict=True)
        model.eval()
        spec_obj: Any = x1_spec
        if not require_x1_native and target_arch == "P":
            # X1 P weights load losslessly into X2TypedModel("P") except provisional_* heads
            x2_spec = build_x2_model_spec("P", base_spec=x1_spec.base_spec, tiny_canary=x1_spec.base_spec.width <= 64)
            x2_m = build_x2_model(x2_spec, 0, device=device)
            missing, unexpected = x2_m.load_state_dict(weights, strict=False)
            allowed_missing = {
                "provisional_val_head.weight",
                "provisional_val_head.bias",
                "provisional_cert_head.weight",
                "provisional_cert_head.bias",
            }
            if unexpected or not set(missing).issubset(allowed_missing):
                raise RuntimeError(f"X1->X2 P load key mismatch: missing={missing}, unexpected={unexpected}")
            x2_m.eval()
            spec_obj = x1_spec
    else:
        x2_spec = _reconstruct_x2_spec(weights, meta, target_arch, tiny_canary)
        model = build_x2_model(x2_spec, 0, device=device)
        model.load_state_dict(weights, strict=True)
        model.eval()
        spec_obj = x2_spec

    return {
        "model": model,
        "model_spec": spec_obj,
        "weights": weights,
        "optimizer_state": optimizer_state,
        "architecture": target_arch,
        "is_x1_parent": is_x1,
        "parent_global_update": parent_global_update,
        "parent_checkpoint_sha256": parent_sha,
        "parent_source_commit": parent_source_commit,
        "checkpoint_format": ckpt_format,
        "resolved_lineage_dir": resolved_lineage_dir,
        "metadata": meta,
    }


def _clone_child_from_loaded_parent(
    loaded: Mapping[str, Any],
    *,
    child_architecture: str,
    seed: int,
    device: str = "cpu",
) -> tuple[nn.Module, Any]:
    parent_model = loaded["model"]
    weights = loaded["weights"]
    is_x1 = bool(loaded["is_x1_parent"])
    pspec = loaded["model_spec"]

    if is_x1:
        assert isinstance(pspec, X1ModelSpec)
        is_tiny = pspec.base_spec.width <= 64
        x2_arch = child_architecture if child_architecture in ("P", "R") else ("P" if child_architecture == "F-cap" else "R")
        x2_spec = build_x2_model_spec(x2_arch, base_spec=pspec.base_spec, tiny_canary=is_tiny)
        if child_architecture == "P":
            child: nn.Module = build_x2_model(x2_spec, seed, device=device)
            missing, unexpected = child.load_state_dict(weights, strict=False)
            allowed_missing = {
                "provisional_val_head.weight",
                "provisional_val_head.bias",
                "provisional_cert_head.weight",
                "provisional_cert_head.bias",
            }
            if unexpected or not set(missing).issubset(allowed_missing):
                raise RuntimeError(f"X1->X2 P clone mismatch: missing={missing}, unexpected={unexpected}")
            return child, x2_spec
        # Wrap native X1TypedModel (e.g. R with slot_dim=128 + agg_mlp) so 100% parent params + exact logits match
        x1_clone = build_x1_s20_model(pspec, seed).to(torch.device(device))
        x1_clone.load_state_dict({k: v.clone() for k, v in weights.items()}, strict=True)
        return X1AdaptationWrapper(x1_clone, x2_spec, head_seed=seed).to(torch.device(device)), x2_spec

    assert isinstance(pspec, X2ModelSpec)
    child_x2 = build_x2_model(pspec, seed, device=device)
    child_x2.load_state_dict({k: v.clone() for k, v in weights.items()}, strict=True)
    return child_x2, pspec


@torch.no_grad()
def verify_parent_child_prediction_parity(
    parent_model: nn.Module,
    child_model: nn.Module,
    probe_worlds: Sequence[Mapping[str, Any]],
    *,
    device: str = "cpu",
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Verify exact pre-update parent vs child output parity before ANY optimizer step."""
    if not probe_worlds:
        raise ValueError("probe_worlds must be non-empty for parent/child parity check")
    kernel = X2EnvironmentKernel()
    samples = []
    for w in probe_worlds:
        st = kernel.initial_state(w)
        obs = kernel.observe(st)
        samples.append({"rendered_text": obs.rendered_text})

    max_ctx = 640
    for m in (parent_model, child_model):
        if hasattr(m, "x2_spec"):
            max_ctx = min(max_ctx, int(m.x2_spec.base_spec.context_length))
        elif hasattr(m, "x1_spec"):
            max_ctx = min(max_ctx, int(m.x1_spec.max_sequence_length))

    dev = torch.device(device)
    toks, segs, apos = _encode_batch_samples(samples, dev, max_ctx)
    parent_model.eval()
    child_model.eval()
    p_out = parent_model.forward_action(toks, segs, apos)
    c_out = child_model.forward_action(toks, segs, apos)

    max_diff = float((p_out.log_probs_19.float() - c_out.log_probs_19.float()).abs().max().item())
    p_arg = torch.argmax(p_out.log_probs_19, dim=-1).tolist()
    c_arg = torch.argmax(c_out.log_probs_19, dim=-1).tolist()
    arg_match = p_arg == c_arg
    if max_diff > float(atol) or not arg_match:
        raise ValueError(
            f"PARENT_CHILD_PARITY_VIOLATION: max_abs_logit_diff={max_diff:.6g} > {atol}, argmax_match={arg_match}"
        )
    return {
        "verified": True,
        "checked_cases": len(samples),
        "max_abs_logit_diff": round(max_diff, 12),
        "argmax_exact_match": True,
        "tolerance": float(atol),
    }


def _evaluate_worlds_with_model(
    model: nn.Module,
    model_spec: Any,
    worlds: Sequence[Mapping[str, Any]],
    *,
    lesion_mode: str | None = None,
    copy_enabled_override: bool | None = None,
) -> dict[str, Any]:
    """Evaluate a model (X2TypedModel, X1AdaptationWrapper, or X1TypedModel) on X2 worlds via X2EnvironmentKernel."""
    kernel = X2EnvironmentKernel()
    max_ctx = int(
        model_spec.base_spec.context_length
        if hasattr(model_spec, "base_spec")
        else getattr(model_spec, "max_sequence_length", 640)
    )
    is_x1_raw = isinstance(model, X1TypedModel)
    if is_x1_raw:
        dev = next(model.parameters()).device
        eval_mod = X1AdaptationWrapper(
            model,
            build_x2_model_spec(
                model.x1_spec.architecture if model.x1_spec.architecture in ("P", "R") else "P",
                base_spec=model.x1_spec.base_spec,
                tiny_canary=model.x1_spec.base_spec.width <= 64,
            ),
        ).to(dev)
    else:
        eval_mod = model
    policy = X2ObservablePolicy(
        eval_mod,  # type: ignore[arg-type]
        model_spec,
        mode="raw",
        copy_enabled_override=copy_enabled_override,
        lesion_mode=lesion_mode,
    )
    records: list[dict[str, Any]] = []
    batch_size = 32
    eval_mod.eval()
    for start in range(0, len(worlds), batch_size):
        chunk = worlds[start : start + batch_size]
        states = [kernel.initial_state(w) for w in chunk]
        obs_list = [kernel.observe(st) for st in states]
        acts = policy.propose_batch(obs_list)
        for w, st, obs, act in zip(chunk, states, obs_list, acts):
            gold_op, gold_arg = kernel.ideal_action(st)
            res = kernel.verify(w, [(act.operation, act.argument)])
            records.append(
                {
                    "world_id": str(w.get("world_id", "")),
                    "task_type": str(w.get("task_type", "")),
                    "fact_style": str(w.get("fact_style", "canonical")),
                    "header_style": str(w.get("header_style", "canonical")),
                    "hops": int(w.get("hops", 1)),
                    "support_dose": w.get("support_dose"),
                    "gold_op": gold_op,
                    "gold_arg": gold_arg,
                    "pred_op": act.operation,
                    "pred_arg": act.argument,
                    "correct": bool(res.get("exact", False)),
                    "max_ctx": max_ctx,
                    "obs_id": getattr(obs, "obs_id", digest(obs.rendered_text)),
                }
            )
    acc = sum(1 for r in records if r["correct"]) / max(1, len(records))
    return {"accuracy": round(acc, 6), "correct": sum(1 for r in records if r["correct"]), "total": len(records), "records": records}


def _recompute_a0_parity(architecture: str, dev_accuracy: float) -> dict[str, Any]:
    if not A0_REFERENCE_METRICS_PATH.is_file():
        return {"reference_available": False, "path": str(A0_REFERENCE_METRICS_PATH)}
    doc = json.loads(A0_REFERENCE_METRICS_PATH.read_text(encoding="utf-8"))
    ctr = doc.get("corrected_gate", {}).get("paired_seed_stats", {}).get("contrasts", {}).get("default", {})
    ref_vals = [
        float(e["control_value" if architecture == "P" else "treatment_value"])
        for e in ctr.get("per_seed_effects", ())
        if "control_value" in e and "treatment_value" in e
    ]
    ref_mean = sum(ref_vals) / max(1, len(ref_vals)) if ref_vals else None
    return {
        "reference_available": True,
        "reference_metrics_path": str(A0_REFERENCE_METRICS_PATH),
        "reference_archive_sha256": doc.get("archive_sha256"),
        "a0_reference_mean_accuracy": round(ref_mean, 6) if ref_mean is not None else None,
        "reproduced_dev_accuracy": round(float(dev_accuracy), 6),
        "abs_diff_vs_a0_mean": round(abs(float(dev_accuracy) - ref_mean), 6) if ref_mean is not None else None,
    }


def run_frozen_probe(
    *,
    job_id: str,
    architecture: str,
    work_root: Path | str,
    source_commit: str,
    seed: int = 7,
    parent_job_id: str | None = None,
    parent_checkpoint: Path | str | None = None,
    missing_original_receipt: Path | str | None = None,
    reproduced_checkpoint: Path | str | None = None,
    dataset_name: str = "x1-a0-frozen-dev2304-depth640-v1",
    device: str = "cpu",
    canary: bool = False,
) -> dict[str, Any]:
    """Run BII-FROZEN A0 diagnosis: strict 0 optimizer steps, renderer 2x2 factorial, lesions & depth cohorts."""
    wroot = Path(work_root)
    wroot.mkdir(parents=True, exist_ok=True)

    using_reproduced = bool(missing_original_receipt and reproduced_checkpoint and not parent_checkpoint)
    ckpt_target = reproduced_checkpoint if using_reproduced else parent_checkpoint
    if using_reproduced:
        mpath = Path(str(missing_original_receipt))
        if not mpath.is_file():
            raise FileNotFoundError(f"missing_original_receipt file not found: {mpath}")
        evidence_source = "reproduced_with_missing_original_receipt"
        missing_receipt_doc = json.loads(mpath.read_text(encoding="utf-8"))
    else:
        evidence_source = "historical_original"
        missing_receipt_doc = None

    loaded = load_parent_checkpoint(
        ckpt_target, architecture=architecture, tiny_canary=canary, device=device, require_x1_native=True
    )
    model = loaded["model"]
    model_spec = loaded["model_spec"]
    pre_digest = weights_state_digest(model)

    # Build 2x2 renderer factorial + depth cohorts (3, 4, 5) + D1-H probe panel
    gen_b = build_generator_b_multihop_panel(
        split="dev",
        num_skeletons=2 if canary else 8,
        seed=int(seed) + 8101,
        depths=(3, 4, 5),
        include_exhaustive_perm=False,
    )
    dev_worlds = select_balanced_matched_cohort(
        gen_b["episodes"],
        max_total=12 if canary else 96,
        stratum_keys=("hops", "fact_style", "header_style"),
    )

    lesions: tuple[str | None, ...] = (None, "disable_copy", "zero_r") if architecture in ("R", "R2", "U-P", "DM") else (None, "disable_copy")
    lesion_results: dict[str, Any] = {}
    all_episodes: list[dict[str, Any]] = []
    for les in lesions:
        label = "intact" if les is None else str(les)
        res = _evaluate_worlds_with_model(
            model,
            model_spec,
            dev_worlds,
            lesion_mode=les,
            copy_enabled_override=False if les == "disable_copy" else None,
        )
        lesion_results[label] = {"accuracy": res["accuracy"], "correct": res["correct"], "total": res["total"]}
        if les is None:
            intact_records = res["records"]
        for r in res["records"]:
            all_episodes.append({**r, "lesion": label})

    # Breakdown intact records into 2x2 renderer factorial and depth cohorts (3, 4, 5)
    renderer_factorial: dict[str, Any] = {}
    for fs in ("canonical", "narrative"):
        for hs in ("canonical", "narrative"):
            sub = [r for r in intact_records if r["fact_style"] == fs and r["header_style"] == hs]
            renderer_factorial[f"{fs}__{hs}"] = {
                "accuracy": round(sum(1 for r in sub if r["correct"]) / max(1, len(sub)), 6),
                "correct": sum(1 for r in sub if r["correct"]),
                "total": len(sub),
            }

    depth_cohorts: dict[str, Any] = {}
    for dp in (3, 4, 5):
        sub = [r for r in intact_records if int(r["hops"]) == dp]
        depth_cohorts[f"depth_{dp}"] = {
            "accuracy": round(sum(1 for r in sub if r["correct"]) / max(1, len(sub)), 6),
            "correct": sum(1 for r in sub if r["correct"]),
            "total": len(sub),
        }

    post_digest = weights_state_digest(model)
    if post_digest != pre_digest:
        raise RuntimeError("FROZEN_PROBE_MUTATED_WEIGHTS")

    intact_acc = float(lesion_results["intact"]["accuracy"])
    parity_ref = _recompute_a0_parity(architecture, intact_acc) if using_reproduced else None

    probe_receipt = {
        "schema": "elendril-x2-frozen-probe-v1",
        "job_id": str(job_id),
        "architecture": str(architecture),
        "parent_job_id": parent_job_id,
        "dataset": str(dataset_name),
        "source_commit": str(source_commit),
        "evidence_source": evidence_source,
        "missing_original_receipt": missing_receipt_doc,
        "reproduced_a0_parity": parity_ref,
        "parent_checkpoint_sha256": loaded["parent_checkpoint_sha256"],
        "parent_global_update": int(loaded["parent_global_update"]),
        "optimizer_steps_executed": 0,
        "pre_eval_weights_sha256": pre_digest,
        "post_eval_weights_sha256": post_digest,
        "weights_immutable_verified": True,
        "eval_rows": len(all_episodes),
        "intact_eval_rows": len(intact_records),
        "lesion_summary": lesion_results,
        "renderer_factorial": renderer_factorial,
        "depth_cohorts": depth_cohorts,
    }
    summary = {
        "schema": "elendril-x2-frozen-summary-v1",
        "job_id": str(job_id),
        "architecture": str(architecture),
        "evidence_source": evidence_source,
        "optimizer_steps_executed": 0,
        "episodes": len(all_episodes),
        "intact_accuracy": intact_acc,
        "renderer_factorial": renderer_factorial,
        "lesion_summary": lesion_results,
        "depth_cohorts": depth_cohorts,
    }
    final_receipt = {
        "schema": "anra-planthir-final-receipt-v1",
        "campaign_id": str(job_id),
        "execution_route": "x2_frozen_probe",
        "status": "COMPLETE",
        "source_commit": str(source_commit),
        "optimizer_steps_executed": 0,
        "eval_rows": len(all_episodes),
        "evidence_source": evidence_source,
        "parent_checkpoint_sha256": loaded["parent_checkpoint_sha256"],
    }
    _atomic_write_json(wroot / "FROZEN_PROBE_RECEIPT.json", probe_receipt)
    _atomic_write_json(wroot / "SCIENTIFIC_SUMMARY.json", summary)
    _atomic_write_jsonl(wroot / "EPISODES.jsonl", all_episodes)
    _atomic_write_json(wroot / "FINAL_RECEIPT.json", final_receipt)
    return probe_receipt


def _evaluate_t13_suite(model: nn.Module, model_spec: Any, *, seed: int, canary: bool, split: str) -> dict[str, Any]:
    eval_split = "confirmation" if split == "confirmation" else "dev"
    streams_by_dose = {}
    all_recs: list[dict[str, Any]] = []
    for dose in T13_DOSES:
        worlds = [
            w
            for w in build_t13_episode_stream(split=eval_split, doses=(dose,), laws_per_dose=1 if canary else 2, seed=seed + 100 + dose)
            if w.get("eligible", True)
        ]
        ev = _evaluate_worlds_with_model(model, model_spec, worlds)
        streams_by_dose[f"k_{dose:02d}"] = {
            "accuracy": ev["accuracy"],
            "correct": ev["correct"],
            "total": ev["total"],
        }
        all_recs.extend({**r, "dose": dose} for r in ev["records"])

    # D1-H selective-edit retention + forward unseen-rule transfer
    ret_worlds = build_selective_edit_panel(num_histories=2 if canary else 8, seed=seed + 303, split="dev")
    ret_ev = _evaluate_worlds_with_model(model, model_spec, ret_worlds)
    xfer_worlds = [
        w
        for w in build_t13_episode_stream(split=eval_split, doses=(16,), history_kind="hist_a", laws_per_dose=1 if canary else 2, seed=seed + 505)
        if w.get("eligible", True)
    ]
    xfer_ev = _evaluate_worlds_with_model(model, model_spec, xfer_worlds)
    return {
        "schema": "elendril-t13-suite-v1",
        "dose_curves": streams_by_dose,
        "retention_d1h": {"accuracy": ret_ev["accuracy"], "correct": ret_ev["correct"], "total": ret_ev["total"]},
        "unseen_rule_transfer": {"accuracy": xfer_ev["accuracy"], "correct": xfer_ev["correct"], "total": xfer_ev["total"]},
        "total_eval_rows": len(all_recs) + ret_ev["total"] + xfer_ev["total"],
        "records": all_recs,
    }


def _evaluate_t14_suite(model: nn.Module, model_spec: Any, *, seed: int, canary: bool, split: str) -> dict[str, Any]:
    eval_split = "confirmation" if split == "confirmation" else "dev"
    worlds = build_t14_transition_stream(split=eval_split, histories_per_stratum=1 if canary else 3, seed=seed + 707)
    ev = _evaluate_worlds_with_model(model, model_spec, worlds)
    dp_refs = {str(w["world_id"]): solve_t14_public_dp(w) for w in worlds}

    cheap_recs: list[dict[str, Any]] = []
    exp_recs: list[dict[str, Any]] = []
    free_recs: list[dict[str, Any]] = []
    forced_recs: list[dict[str, Any]] = []
    tool_costs: list[float] = []
    ans_total = 0
    ans_correct = 0
    unsupported_count = 0

    for w, r in zip(worlds, ev["records"]):
        pb = w["public_belief"]
        mean_cost = 0.5 * (float(pb.get("query_cost", 0.0)) + float(pb.get("retrieve_cost", 0.0)))
        pred_op = r["pred_op"]
        pred_arg = r["pred_arg"]
        cost = float(pb["query_cost"]) if pred_op == "QUERY" else (float(pb["retrieve_cost"]) if pred_op == "RETRIEVE" else 0.0)
        tool_costs.append(cost)
        if not w.get("hidden", {}).get("supported", True) and pred_op == "ANSWER":
            unsupported_count += 1
        if pred_op == "ANSWER":
            ans_total += 1
            if r["correct"]:
                ans_correct += 1
        dp_opt_op = dp_refs[str(w["world_id"])]["optimal_operation"]
        rec_item = {
            **r,
            "mean_tool_cost": mean_cost,
            "paid_cost": cost,
            "dp_optimal_operation": dp_opt_op,
            "matched_dp": pred_op == dp_opt_op,
        }
        if mean_cost <= 0.10:
            cheap_recs.append(rec_item)
        else:
            exp_recs.append(rec_item)
        if w.get("transition_stage") == "post_tool_certified":
            forced_recs.append(rec_item)
        else:
            free_recs.append(rec_item)

    return {
        "schema": "elendril-t14-suite-v1",
        "overall_accuracy": ev["accuracy"],
        "cheap_regime": {
            "accuracy": round(sum(1 for x in cheap_recs if x["correct"]) / max(1, len(cheap_recs)), 6),
            "dp_match_rate": round(sum(1 for x in cheap_recs if x["matched_dp"]) / max(1, len(cheap_recs)), 6),
            "total": len(cheap_recs),
        },
        "expensive_regime": {
            "accuracy": round(sum(1 for x in exp_recs if x["correct"]) / max(1, len(exp_recs)), 6),
            "dp_match_rate": round(sum(1 for x in exp_recs if x["matched_dp"]) / max(1, len(exp_recs)), 6),
            "total": len(exp_recs),
        },
        "free_choice_accuracy": round(sum(1 for x in free_recs if x["correct"]) / max(1, len(free_recs)), 6),
        "forced_observation_revision_accuracy": round(sum(1 for x in forced_recs if x["correct"]) / max(1, len(forced_recs)), 6),
        "conditional_answer_denominator": ans_total,
        "conditional_answer_accuracy": round(ans_correct / max(1, ans_total), 6),
        "mean_tool_cost": round(sum(tool_costs) / max(1, len(tool_costs)), 6),
        "unsupported_answer_count": unsupported_count,
        "total_eval_rows": ev["total"],
        "records": ev["records"],
    }


def _train_standalone_with_trainer(
    *,
    job_id: str,
    model: nn.Module,
    model_spec: Any,
    dataset_name: str,
    objective: str,
    seed: int,
    order_seed: int,
    updates: int,
    milestones: Sequence[int],
    source_commit: str,
    work_root: Path,
    device: str,
    canary: bool,
    architecture: str,
) -> dict[str, Any]:
    split = "confirmation" if "confirmation" in dataset_name else ("dev" if "t14" in dataset_name else "train")
    ds_bundle = x2_dataset_factory({"dataset_name": dataset_name, "seed": seed, "split": split}, cpu_canary=canary)
    train_data = ds_bundle["train_data"]
    batch_sz = 4 if canary else 16
    lr = 1e-3 if canary else 1e-4
    recipe = GenericTrainRecipe(
        seed=seed,
        total_updates=int(updates),
        batch_size=batch_sz,
        micro_batch_size=2,
        lr=lr,
        objective=objective,
        sampling="uniform",
        schedule="cosine",
        precision="fp32",
    )
    lineage_id = f"{job_id.lower()}-{architecture.lower()}-s{order_seed}"
    trainer = GenericExperimentTrainer(
        backend_id="elendril_x2",
        model=model,
        model_spec=model_spec,
        recipe=recipe,
        loss_fn=x2_loss_fn,
        items=train_data.items,
        data_provenance=train_data.manifest,
        order_seed=order_seed,
        source_commit=source_commit,
        root=work_root / "checkpoints",
        lineage=lineage_id,
        device=device,
        cpu_canary=canary,
        resume=False,
    )
    milestone_set = {int(m) for m in milestones if int(m) > 0}
    step_logs: list[dict[str, Any]] = []
    recorded_milestones: list[dict[str, Any]] = []
    for _ in range(int(updates)):
        info = trainer.step()
        step_logs.append(info)
        u_cur = int(info["update"])
        if u_cur in milestone_set or u_cur == int(updates):
            ckpt_sha = trainer.save(prune_old=False)
            trainer.store.record_milestone(threshold_tokens=trainer.state.cumulative_tokens, checkpoint_sha256=ckpt_sha)
            recorded_milestones.append(
                {
                    "global_update": u_cur,
                    "cumulative_tokens": int(trainer.state.cumulative_tokens),
                    "checkpoint_sha256": ckpt_sha,
                    "loss": float(info["loss"]),
                    "lr": float(info["lr"]),
                }
            )
    if trainer.published_update < trainer.state.global_update:
        ckpt_sha = trainer.save(prune_old=False)
        recorded_milestones.append(
            {
                "global_update": int(trainer.state.global_update),
                "cumulative_tokens": int(trainer.state.cumulative_tokens),
                "checkpoint_sha256": ckpt_sha,
            }
        )
    return {
        "ds_bundle": ds_bundle,
        "split": split,
        "trainer": trainer,
        "lineage_id": lineage_id,
        "step_logs": step_logs,
        "milestones": recorded_milestones,
        "optimizer_group_receipt": optimizer_group_receipt(model, trainer.optimizer),
    }


def run_task_adaptation(
    *,
    job_id: str,
    architecture: str,
    parent_job_id: str | None,
    parent_checkpoint: Path | str | None,
    dataset_name: str,
    objective: str,
    work_root: Path | str,
    source_commit: str,
    seed: int = 7,
    order_seed: int | None = None,
    updates: int = 2368,
    milestones: Sequence[int] = (0, 592, 1184, 1472, 2368),
    device: str = "cpu",
    canary: bool = False,
) -> dict[str, Any]:
    """Run BIV-T13 / BIV-T14 / CONF-T13/T14 / EXTRA-T14 task adaptation: clone parent, check parity, reset AdamW."""
    wroot = Path(work_root)
    wroot.mkdir(parents=True, exist_ok=True)
    oseed = int(seed if order_seed is None else order_seed)
    loaded = load_parent_checkpoint(
        parent_checkpoint, architecture=architecture, tiny_canary=canary, device=device, require_x1_native=True
    )
    parent_model = loaded["model"]
    child_model, child_spec = _clone_child_from_loaded_parent(
        loaded, child_architecture=architecture, seed=seed, device=device
    )
    probe_worlds = build_selective_edit_panel(num_histories=2, seed=seed + 17, split="dev")[:6]
    parity = verify_parent_child_prediction_parity(parent_model, child_model, probe_worlds, device=device)

    split = "confirmation" if "confirmation" in dataset_name else ("dev" if "t14" in dataset_name else "train")
    frozen_state_eval = (
        _evaluate_t13_suite(parent_model, loaded["model_spec"], seed=seed, canary=canary, split=split)
        if "t13" in dataset_name.lower() or objective == "T13_ADAPT"
        else None
    )

    eff_updates = 2 if canary and updates > 8 else int(updates)
    eff_ms = (1, eff_updates) if canary and updates > 8 else tuple(int(m) for m in milestones)
    trained = _train_standalone_with_trainer(
        job_id=job_id,
        model=child_model,
        model_spec=child_spec,
        dataset_name=dataset_name,
        objective=objective,
        seed=seed,
        order_seed=oseed,
        updates=eff_updates,
        milestones=eff_ms,
        source_commit=source_commit,
        work_root=wroot,
        device=device,
        canary=canary,
        architecture=architecture,
    )

    if "t13" in dataset_name.lower() or objective == "T13_ADAPT":
        task_eval = _evaluate_t13_suite(child_model, child_spec, seed=seed, canary=canary, split=split)
    else:
        task_eval = _evaluate_t14_suite(child_model, child_spec, seed=seed, canary=canary, split=split)
    eval_recs = task_eval.pop("records", [])

    parent_receipt = {
        "schema": "elendril-x2-parent-lineage-v1",
        "job_id": str(job_id),
        "architecture": str(architecture),
        "parent_job_id": parent_job_id,
        "parent_checkpoint_path": str(parent_checkpoint),
        "parent_checkpoint_sha256": loaded["parent_checkpoint_sha256"],
        "parent_global_update": int(loaded["parent_global_update"]),
        "parent_is_x1": bool(loaded["is_x1_parent"]),
        "optimizer_reset": True,
        "optimizer_step_starts_at": 0,
        "pre_update_parity": parity,
    }
    arm_key = f"ADAPT_{architecture}_s{seed}".replace("-", "_")
    arm_results = {
        arm_key: {
            "architecture": str(architecture),
            "lineage_id": trained["lineage_id"],
            "milestones": trained["milestones"],
            "process_funnel": {"episodes": len(eval_recs), "updates": eff_updates},
            "task_evaluation": task_eval,
            "frozen_state_baseline": frozen_state_eval,
        }
    }
    summary = {
        "schema": "elendril-x2-adaptation-summary-v1",
        "job_id": str(job_id),
        "architecture": str(architecture),
        "dataset": str(dataset_name),
        "objective": str(objective),
        "parent_lineage": parent_receipt,
        "task_evaluation": task_eval,
        "frozen_state_baseline": frozen_state_eval,
        "episodes": len(eval_recs),
        "step_logs": trained["step_logs"],
    }
    final_receipt = {
        "schema": "anra-planthir-final-receipt-v1",
        "campaign_id": str(job_id),
        "execution_route": "x2_task_adaptation",
        "status": "COMPLETE",
        "source_commit": str(source_commit),
        "completed_updates": eff_updates,
        "eval_rows": len(eval_recs),
        "parent_checkpoint_sha256": loaded["parent_checkpoint_sha256"],
        "optimizer_reset": True,
    }
    _atomic_write_json(wroot / "PARENT_LINEAGE_RECEIPT.json", parent_receipt)
    _atomic_write_json(wroot / "ARM_RESULTS.json", arm_results)
    _atomic_write_json(wroot / "SCIENTIFIC_SUMMARY.json", summary)
    _atomic_write_jsonl(wroot / "EPISODES.jsonl", eval_recs)
    _atomic_write_json(wroot / "FINAL_RECEIPT.json", final_receipt)
    return {"parent_receipt": parent_receipt, "arm_results": arm_results, "summary": summary, "final_receipt": final_receipt}


def run_horizon_continuation(
    *,
    job_id: str,
    architecture: str,
    parent_job_id: str | None,
    parent_checkpoint: Path | str | None,
    dataset_name: str = "x2-d1h-selective-edit-v1",
    objective: str = "standard",
    work_root: Path | str,
    source_commit: str,
    seed: int = 7,
    order_seed: int | None = None,
    parent_updates: int | None = None,
    extension_updates: int = 2368,
    mid_milestone: int | None = None,
    end_milestone: int | None = None,
    device: str = "cpu",
    canary: bool = False,
) -> dict[str, Any]:
    """Run BV-H4736 horizon continuation: restore prefix weights, verify parity, step continuously 2368->3552->4736."""
    wroot = Path(work_root)
    wroot.mkdir(parents=True, exist_ok=True)
    oseed = int(seed if order_seed is None else order_seed)
    loaded = load_parent_checkpoint(
        parent_checkpoint, architecture=architecture, tiny_canary=canary, device=device, require_x1_native=True
    )
    parent_model = loaded["model"]
    child_model, child_spec = _clone_child_from_loaded_parent(
        loaded, child_architecture=architecture, seed=seed, device=device
    )
    probe_worlds = build_selective_edit_panel(num_histories=2, seed=seed + 29, split="dev")[:6]
    parity = verify_parent_child_prediction_parity(parent_model, child_model, probe_worlds, device=device)

    u_parent = int(parent_updates if parent_updates is not None else (loaded["parent_global_update"] or (2 if canary else 2368)))
    u_ext = 2 if (canary and extension_updates > 6) else int(extension_updates)
    u_end = int(end_milestone if end_milestone is not None else (u_parent + u_ext))
    if u_end != u_parent + u_ext:
        raise ValueError(f"end_milestone ({u_end}) != parent_updates ({u_parent}) + extension_updates ({u_ext})")
    u_mid = int(mid_milestone if mid_milestone is not None else (u_parent + max(1, u_ext // 2)))

    ds_bundle = x2_dataset_factory({"dataset_name": dataset_name, "seed": seed, "split": "train"}, cpu_canary=canary)
    items = ds_bundle["train_data"].items
    batch_sz = 4 if canary else 16
    base_lr = 1e-3 if canary else 1e-4
    peak_lr = 0.3 * base_lr
    term_lr = 0.1 * base_lr

    dev = torch.device(device)
    child_model = child_model.to(dev)
    opt = build_adamw_optimizer(child_model, lr=peak_lr, torch_module=torch)
    opt_restored = False
    if loaded["optimizer_state"] is not None and isinstance(child_model, X2TypedModel):
        try:
            opt.load_state_dict(loaded["optimizer_state"])
            opt_restored = True
        except Exception:
            opt_restored = False

    rng = torch.Generator(device="cpu").manual_seed(int(seed) + 991)
    lineage_id = f"{job_id.lower()}-{architecture.lower()}-h{u_end}"
    recipe_canon = {
        "seed": seed,
        "total_updates": u_end,
        "parent_updates": u_parent,
        "extension_updates": u_ext,
        "batch_size": batch_sz,
        "micro_batch_size": 2,
        "base_lr": base_lr,
        "peak_lr": peak_lr,
        "term_lr": term_lr,
        "objective": objective,
        "schedule": "continuation_cosine_0.3x_to_0.1x",
    }
    plan: list[tuple[int, ...]] = []
    n_items = len(items)
    orders: dict[int, list[int]] = {}
    for k in range(u_ext):
        chosen: list[int] = []
        for slot in range(batch_sz):
            pos = k * batch_sz + slot
            ep, off = divmod(pos, n_items)
            if ep not in orders:
                perm = list(range(n_items))
                random.Random(int(digest(["planthir-horizon-order-v1", oseed, u_parent, ep]), 16)).shuffle(perm)
                orders[ep] = perm
            chosen.append(orders[ep][off])
        plan.append(tuple(chosen))

    spec_canon = child_spec.canonical()
    spec_sha = child_spec.sha256()
    data_sha = digest({"count": len(items), "provenance": ds_bundle["train_data"].manifest})
    meta_doc = json.loads(
        canonical(
            {
                "schema": "anra-planthir-horizon-continuation-v1",
                "backend_id": "elendril_x2",
                "architecture": architecture,
                "parameter_count": sum(int(p.numel()) for p in child_model.parameters()),
                "model_spec": spec_canon,
                "model_spec_sha256": spec_sha,
                "recipe": recipe_canon,
                "order_seed": oseed,
                "plan_sha256": digest(plan),
                "provenance": ds_bundle["train_data"].manifest,
                "runtime": runtime_identity(dev),
            }
        )
    )
    identities = IdentityBindings(
        IDENTITY_SCHEMA,
        source_commit,
        spec_sha,
        UTF8.identity,
        data_sha,
        data_sha,
        digest(meta_doc),
        optimizer_group_receipt(child_model, opt)["sha256"],
        digest(["continuation_cosine_0.3x_to_0.1x", peak_lr, term_lr, u_parent, u_end]),
        digest(["uniform", oseed]),
        sampler_spec_sha256=digest(plan),
    )

    store = CheckpointStore(wroot / "checkpoints", lineage_id)
    base_state = TrainingState(
        schema="anra-v5-training-state/v1",
        lineage_id=lineage_id,
        generation=u_parent,
        global_update=u_parent,
        cumulative_tokens=u_parent * batch_sz,
        token_budget=u_end * batch_sz,
        tokens_per_update=batch_sz,
        tokens_by_source={"elendril_x2": u_parent * batch_sz},
        optimizer_step_max=u_parent,
        schedule_tokens=u_parent * batch_sz,
        cursor=CursorState(CURSOR_SCHEMA, data_sha, 0, u_parent, 0),
        rng_state_sha256=sha256(bytes(rng.get_state().tolist())).hexdigest(),
        curriculum_phase=f"horizon-{architecture}-{u_parent}to{u_end}",
        identities=identities,
        parent_checkpoint_sha256=None,
    )

    def _publish(st: TrainingState) -> str:
        payloads = {
            "model.bin": _dump({k: v.detach().cpu() for k, v in child_model.state_dict().items()}),
            "optimizer.bin": _dump(opt.state_dict()),
            "rng.bin": bytes(rng.get_state().tolist()),
            "scheduler.json": canonical({"metadata": meta_doc, "scaler": {}}),
            "cursor.json": canonical(asdict(st.cursor)),
            "ledger.json": canonical(dict(st.tokens_by_source)),
            "training_state.json": canonical(st.canonical()),
        }
        with writer_lock(store.lineage_root):
            quarantine_interrupted_publication(store.lineage_root, st.generation)
            sha = store.publish(state=st, payloads=payloads, expected_parent_sha256=store.latest_sha256())
        return sha

    prefix_sha = _publish(base_state)
    store.record_milestone(threshold_tokens=base_state.cumulative_tokens, checkpoint_sha256=prefix_sha)
    milestones_out: list[dict[str, Any]] = [
        {"global_update": u_parent, "cumulative_tokens": base_state.cumulative_tokens, "checkpoint_sha256": prefix_sha, "role": "restored_prefix"}
    ]

    class _MinRecipe:
        pass

    r_obj = _MinRecipe()
    r_obj.objective = objective  # type: ignore[attr-defined]
    r_obj.eq_weight = 0.0  # type: ignore[attr-defined]
    r_obj.cf_weight = 0.0  # type: ignore[attr-defined]
    r_obj.cf_margin = 0.5  # type: ignore[attr-defined]

    cur_state = base_state
    parent_sha = prefix_sha
    step_logs: list[dict[str, Any]] = []
    for k in range(u_ext):
        denom_cos = max(1, u_ext - 1) if u_ext > 1 else 1
        lr_k = term_lr + (peak_lr - term_lr) * 0.5 * (1.0 + math.cos(math.pi * k / denom_cos))
        opt.zero_grad(set_to_none=True)
        child_model.train()
        for pg in opt.param_groups:
            pg["lr"] = lr_k
        chosen = [items[i] for i in plan[k]]
        acc_loss = 0.0
        for s in range(0, len(chosen), 2):
            sub = chosen[s : s + 2]
            scale = float(len(sub)) / float(max(1, len(chosen)))
            loss_t = x2_loss_fn(child_model, sub, r_obj) * scale  # type: ignore[arg-type]
            loss_t.backward()
            acc_loss += float(loss_t.item())
        validate_parameter_ownership(child_model, opt)
        norm = float(torch.nn.utils.clip_grad_norm_(child_model.parameters(), 1.0, error_if_nonfinite=True))
        opt.step()

        cur_state = cur_state.advance(
            tokens_by_source={"elendril_x2": len(chosen)},
            cursor=CursorState(CURSOR_SCHEMA, data_sha, 0, cur_state.global_update + 1, 0),
            rng_state_sha256=sha256(bytes(rng.get_state().tolist())).hexdigest(),
            parent_checkpoint_sha256=parent_sha,
        )
        gstep = int(cur_state.global_update)
        step_logs.append({"local_step": k + 1, "global_update": gstep, "lr": round(lr_k, 10), "loss": acc_loss, "pre_clip": norm})
        if gstep in (u_mid, u_end):
            parent_sha = _publish(cur_state)
            store.record_milestone(threshold_tokens=cur_state.cumulative_tokens, checkpoint_sha256=parent_sha)
            milestones_out.append(
                {
                    "global_update": gstep,
                    "cumulative_tokens": int(cur_state.cumulative_tokens),
                    "checkpoint_sha256": parent_sha,
                    "lr": round(lr_k, 10),
                    "role": "midpoint" if gstep == u_mid and gstep != u_end else "final",
                }
            )

    eval_ev = _evaluate_worlds_with_model(child_model, child_spec, ds_bundle["eval_worlds"])
    horizon_receipt = {
        "schema": "elendril-x2-horizon-continuation-v1",
        "job_id": str(job_id),
        "architecture": str(architecture),
        "parent_job_id": parent_job_id,
        "external_parent_checkpoint_sha256": loaded["parent_checkpoint_sha256"],
        "prefix_global_update": u_parent,
        "mid_milestone_update": u_mid,
        "end_milestone_update": u_end,
        "global_steps_executed": [s["global_update"] for s in step_logs],
        "schedule": "second_cosine_peak_0.3x_term_0.1x",
        "base_lr": base_lr,
        "peak_lr": peak_lr,
        "terminal_lr": term_lr,
        "lr_start": step_logs[0]["lr"],
        "lr_end": step_logs[-1]["lr"],
        "optimizer_state_restored": opt_restored,
        "pre_update_parity": parity,
        "milestones": milestones_out,
        "heldout_eval_accuracy": eval_ev["accuracy"],
    }
    arm_key = f"HORIZON_{architecture}_s{seed}".replace("-", "_")
    arm_results = {
        arm_key: {
            "architecture": str(architecture),
            "lineage_id": lineage_id,
            "milestones": milestones_out,
            "process_funnel": {"episodes": eval_ev["total"], "updates": u_ext, "end_global_update": u_end},
            "eval_accuracy": eval_ev["accuracy"],
        }
    }
    summary = {
        "schema": "elendril-x2-horizon-summary-v1",
        "job_id": str(job_id),
        "architecture": str(architecture),
        "horizon_receipt": horizon_receipt,
        "heldout_eval": {"accuracy": eval_ev["accuracy"], "correct": eval_ev["correct"], "total": eval_ev["total"]},
        "step_logs": step_logs,
        "episodes": eval_ev["total"],
    }
    final_receipt = {
        "schema": "anra-planthir-final-receipt-v1",
        "campaign_id": str(job_id),
        "execution_route": "x2_horizon_continuation",
        "status": "COMPLETE",
        "source_commit": str(source_commit),
        "completed_updates": u_ext,
        "end_global_update": u_end,
        "eval_rows": eval_ev["total"],
    }
    _atomic_write_json(wroot / "HORIZON_CONTINUATION_RECEIPT.json", horizon_receipt)
    _atomic_write_json(wroot / "ARM_RESULTS.json", arm_results)
    _atomic_write_json(wroot / "SCIENTIFIC_SUMMARY.json", summary)
    _atomic_write_jsonl(wroot / "EPISODES.jsonl", eval_ev["records"])
    _atomic_write_json(wroot / "FINAL_RECEIPT.json", final_receipt)
    return horizon_receipt


def run_scratch_single(
    *,
    job_id: str,
    architecture: str,
    dataset_name: str,
    objective: str = "standard",
    work_root: Path | str,
    source_commit: str,
    seed: int = 7,
    order_seed: int | None = None,
    updates: int = 2368,
    milestones: Sequence[int] = (0, 592, 1184, 1472, 2368),
    device: str = "cpu",
    canary: bool = False,
) -> dict[str, Any]:
    """Train one standalone unpaired `elendril_x2` arm without synthesizing a hidden companion arm."""
    wroot = Path(work_root)
    wroot.mkdir(parents=True, exist_ok=True)
    oseed = int(seed if order_seed is None else order_seed)
    spec = build_x2_model_spec(architecture, tiny_canary=canary)
    model = build_x2_model(spec, init_seed=seed, device=device)
    eff_updates = 2 if canary and updates > 8 else int(updates)
    eff_ms = (1, eff_updates) if canary and updates > 8 else tuple(int(m) for m in milestones)
    trained = _train_standalone_with_trainer(
        job_id=job_id,
        model=model,
        model_spec=spec,
        dataset_name=dataset_name,
        objective=objective,
        seed=seed,
        order_seed=oseed,
        updates=eff_updates,
        milestones=eff_ms,
        source_commit=source_commit,
        work_root=wroot,
        device=device,
        canary=canary,
        architecture=architecture,
    )
    ds_bundle = trained["ds_bundle"]
    eval_ev = _evaluate_worlds_with_model(model, spec, ds_bundle["eval_worlds"])
    arm_key = f"SINGLE_{architecture}_s{seed}".replace("-", "_")
    arm_results = {
        arm_key: {
            "architecture": str(architecture),
            "lineage_id": trained["lineage_id"],
            "milestones": trained["milestones"],
            "process_funnel": {"episodes": eval_ev["total"], "updates": eff_updates},
            "eval_accuracy": eval_ev["accuracy"],
            "synthesized_extra_arms": False,
        }
    }
    summary = {
        "schema": "elendril-x2-scratch-single-v1",
        "job_id": str(job_id),
        "architecture": str(architecture),
        "dataset": str(dataset_name),
        "objective": str(objective),
        "arm_count": 1,
        "synthesized_extra_arms": False,
        "eval_accuracy": eval_ev["accuracy"],
        "episodes": eval_ev["total"],
        "step_logs": trained["step_logs"],
    }
    final_receipt = {
        "schema": "anra-planthir-final-receipt-v1",
        "campaign_id": str(job_id),
        "execution_route": "x2_scratch_single",
        "status": "COMPLETE",
        "source_commit": str(source_commit),
        "completed_updates": eff_updates,
        "eval_rows": eval_ev["total"],
        "arm_count": 1,
    }
    _atomic_write_json(wroot / "ARM_RESULTS.json", arm_results)
    _atomic_write_json(wroot / "SCIENTIFIC_SUMMARY.json", summary)
    _atomic_write_jsonl(wroot / "EPISODES.jsonl", eval_ev["records"])
    _atomic_write_json(wroot / "FINAL_RECEIPT.json", final_receipt)
    return {"arm_results": arm_results, "summary": summary, "final_receipt": final_receipt}


def _parse_milestones(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return (0, 592, 1184, 1472, 2368)
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Elendril V8-X2 Explicit Lineage & Single-Arm Runner")
    sub = parser.add_subparsers(dest="route", required=True)

    p_fr = sub.add_parser("frozen-probe")
    p_fr.add_argument("--job-id", required=True)
    p_fr.add_argument("--architecture", required=True)
    p_fr.add_argument("--work-root", required=True)
    p_fr.add_argument("--source", required=True)
    p_fr.add_argument("--seed", type=int, default=7)
    p_fr.add_argument("--parent-job-id", default=None)
    p_fr.add_argument("--parent-checkpoint", default=None)
    p_fr.add_argument("--missing-original-receipt", default=None)
    p_fr.add_argument("--reproduced-checkpoint", default=None)
    p_fr.add_argument("--dataset", default="x1-a0-frozen-dev2304-depth640-v1")
    p_fr.add_argument("--device", default="cpu")
    p_fr.add_argument("--canary", action="store_true")

    p_ad = sub.add_parser("task-adapt")
    p_ad.add_argument("--job-id", required=True)
    p_ad.add_argument("--architecture", required=True)
    p_ad.add_argument("--parent-job-id", default=None)
    p_ad.add_argument("--parent-checkpoint", required=True)
    p_ad.add_argument("--dataset", required=True)
    p_ad.add_argument("--objective", default="T13_ADAPT")
    p_ad.add_argument("--work-root", required=True)
    p_ad.add_argument("--source", required=True)
    p_ad.add_argument("--seed", type=int, default=7)
    p_ad.add_argument("--order-seed", type=int, default=None)
    p_ad.add_argument("--updates", type=int, default=2368)
    p_ad.add_argument("--milestones", default="0,592,1184,1472,2368")
    p_ad.add_argument("--device", default="cpu")
    p_ad.add_argument("--canary", action="store_true")

    p_hc = sub.add_parser("horizon-continue")
    p_hc.add_argument("--job-id", required=True)
    p_hc.add_argument("--architecture", required=True)
    p_hc.add_argument("--parent-job-id", default=None)
    p_hc.add_argument("--parent-checkpoint", required=True)
    p_hc.add_argument("--dataset", default="x2-d1h-selective-edit-v1")
    p_hc.add_argument("--objective", default="standard")
    p_hc.add_argument("--work-root", required=True)
    p_hc.add_argument("--source", required=True)
    p_hc.add_argument("--seed", type=int, default=7)
    p_hc.add_argument("--order-seed", type=int, default=None)
    p_hc.add_argument("--parent-updates", type=int, default=None)
    p_hc.add_argument("--updates", type=int, default=2368)
    p_hc.add_argument("--mid-milestone", type=int, default=None)
    p_hc.add_argument("--end-milestone", type=int, default=None)
    p_hc.add_argument("--device", default="cpu")
    p_hc.add_argument("--canary", action="store_true")

    p_ss = sub.add_parser("scratch-single")
    p_ss.add_argument("--job-id", required=True)
    p_ss.add_argument("--architecture", required=True)
    p_ss.add_argument("--dataset", required=True)
    p_ss.add_argument("--objective", default="standard")
    p_ss.add_argument("--work-root", required=True)
    p_ss.add_argument("--source", required=True)
    p_ss.add_argument("--seed", type=int, default=7)
    p_ss.add_argument("--order-seed", type=int, default=None)
    p_ss.add_argument("--updates", type=int, default=2368)
    p_ss.add_argument("--milestones", default="0,592,1184,1472,2368")
    p_ss.add_argument("--device", default="cpu")
    p_ss.add_argument("--canary", action="store_true")

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.route == "frozen-probe":
        run_frozen_probe(
            job_id=args.job_id,
            architecture=args.architecture,
            work_root=args.work_root,
            source_commit=args.source,
            seed=args.seed,
            parent_job_id=args.parent_job_id,
            parent_checkpoint=args.parent_checkpoint,
            missing_original_receipt=args.missing_original_receipt,
            reproduced_checkpoint=args.reproduced_checkpoint,
            dataset_name=args.dataset,
            device=args.device,
            canary=args.canary,
        )
        return 0
    if args.route == "task-adapt":
        run_task_adaptation(
            job_id=args.job_id,
            architecture=args.architecture,
            parent_job_id=args.parent_job_id,
            parent_checkpoint=args.parent_checkpoint,
            dataset_name=args.dataset,
            objective=args.objective,
            work_root=args.work_root,
            source_commit=args.source,
            seed=args.seed,
            order_seed=args.order_seed,
            updates=args.updates,
            milestones=_parse_milestones(args.milestones),
            device=args.device,
            canary=args.canary,
        )
        return 0
    if args.route == "horizon-continue":
        run_horizon_continuation(
            job_id=args.job_id,
            architecture=args.architecture,
            parent_job_id=args.parent_job_id,
            parent_checkpoint=args.parent_checkpoint,
            dataset_name=args.dataset,
            objective=args.objective,
            work_root=args.work_root,
            source_commit=args.source,
            seed=args.seed,
            order_seed=args.order_seed,
            parent_updates=args.parent_updates,
            extension_updates=args.updates,
            mid_milestone=args.mid_milestone,
            end_milestone=args.end_milestone,
            device=args.device,
            canary=args.canary,
        )
        return 0
    if args.route == "scratch-single":
        run_scratch_single(
            job_id=args.job_id,
            architecture=args.architecture,
            dataset_name=args.dataset,
            objective=args.objective,
            work_root=args.work_root,
            source_commit=args.source,
            seed=args.seed,
            order_seed=args.order_seed,
            updates=args.updates,
            milestones=_parse_milestones(args.milestones),
            device=args.device,
            canary=args.canary,
        )
        return 0
    return 1


__all__ = [
    "X1AdaptationWrapper",
    "load_parent_checkpoint",
    "main",
    "run_frozen_probe",
    "run_horizon_continuation",
    "run_scratch_single",
    "run_task_adaptation",
    "verify_parent_child_prediction_parity",
    "weights_state_digest",
]

if __name__ == "__main__":
    raise SystemExit(main())
