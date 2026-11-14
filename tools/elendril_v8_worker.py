"""Isolated V8 GPU/CPU worker subprocess CLI for canary, profile, train, recover, and frozen eval."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import signal
import tempfile
import threading
import time
from typing import Any

import torch

from anra_v5.elendril.codec import canonical, digest
from anra_v5.elendril.foundation_training import Recipe
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.training import batch_tensors, configure_device
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.elendril.v717_training import IndependentOrderTrainer
from anra_v5.elendril.v8 import evaluation as _v8_eval
from anra_v5.elendril.v8.contracts import POLICY_MODES, SYMBOLS, PublicInput, V8ModelSpec, build_v8_model_spec
from anra_v5.elendril.v8.data import (
    V8Dataset,
    build_d1_dev_suite,
    build_d1_training_dataset,
    load_d0_dataset,
    make_positive_canary_cohort,
    render_public_v8,
)
from anra_v5.elendril.v8.evaluation import (
    EpisodeRecord,
    diagnostic_probe,
    evaluate_a20_recovery_extension,
    evaluate_f03_reverse_extension,
    evaluate_worlds_by_mode,
    run_pointer_interventions,
    summarize_suite_episodes,
)
from anra_v5.elendril.v8.holdout import (
    audit_split_disjointness,
    build_confirmation_suite,
    canonical_graph_signature,
    render_confirmation_public,
)
from anra_v5.elendril.v8.models import compute_typed_nll
from anra_v5.elendril.v8.policies import make_policy
from anra_v5.elendril.v8.protocol import OPTIMIZER_DEFAULTS, PHASES, jobs as protocol_jobs
from anra_v5.elendril.v8.teacher import PublicFact, TeacherStep
from anra_v5.elendril.v8.training import (
    V8TrainingData,
    V8TypedTrainer,
    create_v8_trainer,
    prepare_v8_training_data,
    teacher_steps_to_byte_samples,
    teacher_steps_to_typed_samples,
)
from anra_v5.elendril.v8.verifier import independent_expected_outcome, verify_episode_trace
from v5_training.checkpoint import CheckpointStore

_DEFAULT_CONFIRMATION_SEED = 90901


def _find_job(job_id: str) -> dict[str, object]:
    for phase in PHASES:
        for job in protocol_jobs(phase):
            if job["job_id"] == job_id:
                return job
    raise ValueError(f"unknown V8 job_id: {job_id!r}")


def _resolve_profile_job(
    *,
    job_id: str | None = None,
    arm: str | None = None,
    dataset: str | None = None,
) -> dict[str, object]:
    if job_id is not None:
        job = _find_job(job_id)
        if arm is not None and str(job["arm"]) != arm:
            raise ValueError(f"job {job_id!r} arm {job['arm']!r} != requested arm {arm!r}")
        if dataset is not None and str(job["dataset"]) != dataset:
            raise ValueError(f"job {job_id!r} dataset {job['dataset']!r} != requested dataset {dataset!r}")
        return job
    target_arm = arm or "F"
    target_ds = dataset or "D1"
    for phase in PHASES:
        for job in protocol_jobs(phase):
            if str(job["arm"]) == target_arm and str(job["dataset"]) == target_ds and int(job["epochs"]) > 0:
                return job
    raise ValueError(f"no registered trainable V8 job matching arm={target_arm!r}, dataset={target_ds!r}")


def _tiny_cpu_spec(arm: str) -> V8ModelSpec:
    tiny_base = replace(
        MODEL_PROFILES["S20"],
        width=32,
        layers=1,
        query_heads=2,
        kv_heads=1,
        head_dimension=16,
        ffn_width=64,
        context_length=512,
    )
    return build_v8_model_spec(arm, base_spec=tiny_base, pointer_dim=16 if arm == "P" else 64)


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _atomic_write_json(path: Path, payload: object, *, compact: bool = False) -> None:
    if compact:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(path, text)


def _file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _world_structure_hash(world: dict[str, object]) -> str:
    h = world.get("structure_hash") or world.get("semantic_hash")
    if isinstance(h, str) and h:
        return h
    return str(canonical_graph_signature(world)["structure_hash"])


def _is_isolated_probe(world: dict[str, object]) -> bool:
    if "is_isolated_one_edge_probe" in world:
        return bool(world["is_isolated_one_edge_probe"])
    return bool(canonical_graph_signature(world)["is_isolated_one_edge_probe"])


def _serialize_step(step: TeacherStep) -> dict[str, object]:
    return asdict(step)


def _deserialize_step(row: dict[str, Any]) -> TeacherStep:
    pub = PublicInput(**row["public_input"])
    return TeacherStep(
        task_id=str(row["task_id"]),
        world_id=str(row["world_id"]),
        family=str(row["family"]),
        stage=str(row["stage"]),
        public_input=pub,
        gold_op=str(row["gold_op"]),
        gold_arg=(None if row.get("gold_arg") is None else str(row["gold_arg"])),
        target_text=str(row["target_text"]),
        prompt_bytes=int(row["prompt_bytes"]),
        is_answer_step=bool(row["is_answer_step"]),
    )


def prepare_verified_splits(
    work_root: Path | str,
    *,
    cpu_smoke: bool = False,
    dev_worlds_per_cohort: int | None = None,
    conf_worlds_per_cohort: int | None = None,
    confirmation_seed: int | None = None,
) -> dict[str, Any]:
    """Build or load cached, structurally disjoint D1/Dev/Confirmation splits under <work_root>/splits."""

    splits_dir = Path(work_root) / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    audit_path = splits_dir / "SPLIT_AUDIT.json"
    d1_worlds_path = splits_dir / "d1_worlds.json"
    d1_steps_path = splits_dir / "d1_steps.json"
    d1_hashes_path = splits_dir / "d1_structural_hashes.json"
    dev_path = splits_dir / "dev_cohorts.json"
    conf_path = splits_dir / "conf_cohorts.json"
    data_files = {
        "d1_worlds.json": d1_worlds_path,
        "d1_steps.json": d1_steps_path,
        "d1_structural_hashes.json": d1_hashes_path,
        "dev_cohorts.json": dev_path,
        "conf_cohorts.json": conf_path,
    }

    d1_quota = 16 if cpu_smoke else 512
    dev_wpc = int(dev_worlds_per_cohort) if dev_worlds_per_cohort is not None else (2 if cpu_smoke else 128)
    conf_wpc = int(conf_worlds_per_cohort) if conf_worlds_per_cohort is not None else (2 if cpu_smoke else 256)

    # Attempt verified cache load before regenerating
    if audit_path.exists() and all(p.exists() for p in data_files.values()):
        try:
            cached_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if cached_audit.get("schema") == "elendril-v8-split-audit-v1" and cached_audit.get("audit_passed") is True:
                cfg_matches = (
                    bool(cached_audit.get("cpu_smoke")) == bool(cpu_smoke)
                    and int(cached_audit.get("d1_quota_per_family", -1)) == d1_quota
                    and int(cached_audit.get("dev_worlds_per_cohort", -1)) == dev_wpc
                    and int(cached_audit.get("conf_worlds_per_cohort", -1)) == conf_wpc
                    and (
                        confirmation_seed is None
                        or int(cached_audit.get("confirmation_seed", -1)) == int(confirmation_seed)
                    )
                )
                expected_sha = cached_audit.get("split_audit_sha256")
                body_for_hash = {k: v for k, v in cached_audit.items() if k != "split_audit_sha256"}
                file_hashes = cached_audit.get("file_hashes", {})
                hashes_ok = (
                    isinstance(expected_sha, str)
                    and digest(body_for_hash) == expected_sha
                    and isinstance(file_hashes, dict)
                    and set(file_hashes) == set(data_files)
                    and all(_file_sha256(p) == file_hashes[name] for name, p in data_files.items())
                )
                if cfg_matches and hashes_ok:
                    d1_worlds_list = json.loads(d1_worlds_path.read_text(encoding="utf-8"))
                    d1_steps_list = [_deserialize_step(r) for r in json.loads(d1_steps_path.read_text(encoding="utf-8"))]
                    d1_hashes_list = json.loads(d1_hashes_path.read_text(encoding="utf-8"))
                    dev_raw = json.loads(dev_path.read_text(encoding="utf-8"))
                    conf_raw = json.loads(conf_path.read_text(encoding="utf-8"))
                    if digest(sorted(d1_hashes_list)) == cached_audit.get("d1_structural_hashes_sha256"):
                        dev_cohorts = {k: tuple(v) for k, v in dev_raw.items()}
                        conf_cohorts = {k: tuple(v) for k, v in conf_raw.items()}
                        d1_dataset = V8Dataset(
                            split="train",
                            identity=str(cached_audit["d1_identity"]),
                            worlds=tuple(d1_worlds_list),
                            steps=tuple(d1_steps_list),
                            audit=dict(cached_audit["d1_audit"]),
                            shortcuts=dict(cached_audit["d1_shortcuts"]),
                            crosstabs=dict(cached_audit["d1_crosstabs"]),
                            structural_hashes=frozenset(d1_hashes_list),
                        )
                        return {
                            "splits_dir": str(splits_dir),
                            "cached": True,
                            "audit": cached_audit,
                            "split_audit_sha256": str(expected_sha),
                            "d1_identity": d1_dataset.identity,
                            "d1_structural_hashes_sha256": str(cached_audit["d1_structural_hashes_sha256"]),
                            "dev_identity": str(cached_audit["dev_identity"]),
                            "conf_identity": str(cached_audit["conf_identity"]),
                            "d1_dataset": d1_dataset,
                            "d1_structural_hashes": d1_dataset.structural_hashes,
                            "dev_cohorts": dev_cohorts,
                            "conf_cohorts": conf_cohorts,
                        }
        except Exception:
            # Fall through to deterministic clean regeneration if cache is corrupt/incomplete
            pass

    conf_seed = int(confirmation_seed) if confirmation_seed is not None else _DEFAULT_CONFIRMATION_SEED
    d1_dataset = build_d1_training_dataset(quota_per_family=d1_quota)
    d1_exclusions = set(d1_dataset.structural_hashes)

    dev_cohorts = build_d1_dev_suite(
        worlds_per_cohort=dev_wpc,
        excluded_structure_hashes=d1_exclusions,
    )
    dev_worlds = [w for rows in dev_cohorts.values() for w in rows]
    dev_and_d1_exclusions = d1_exclusions | {
        _world_structure_hash(w) for w in dev_worlds if not _is_isolated_probe(w)
    }

    conf_cohorts = build_confirmation_suite(
        confirmation_seed=conf_seed,
        worlds_per_cohort=conf_wpc,
        excluded_structure_hashes=dev_and_d1_exclusions,
    )
    conf_worlds = [w for rows in conf_cohorts.values() for w in rows]

    # Mandatory full-corpus split disjointness audit across D1 + Dev + Confirmation
    disjoint_audit = audit_split_disjointness(
        list(d1_dataset.worlds) + dev_worlds + conf_worlds,
        min_distinct_classes_per_split=(2 if cpu_smoke else 32),
    )
    if disjoint_audit.get("status") != "PASS":
        raise ValueError(f"split disjointness audit failed: {disjoint_audit}")

    d1_hashes_sorted = sorted(d1_dataset.structural_hashes)
    d1_hashes_sha = digest(d1_hashes_sorted)
    dev_identity = digest({k: [w["content_hash"] for w in v] for k, v in sorted(dev_cohorts.items())})
    conf_identity = digest({k: [w["content_hash"] for w in v] for k, v in sorted(conf_cohorts.items())})

    _atomic_write_json(d1_worlds_path, list(d1_dataset.worlds), compact=True)
    _atomic_write_json(d1_steps_path, [_serialize_step(s) for s in d1_dataset.steps], compact=True)
    _atomic_write_json(d1_hashes_path, d1_hashes_sorted, compact=True)
    _atomic_write_json(dev_path, {k: list(v) for k, v in dev_cohorts.items()}, compact=True)
    _atomic_write_json(conf_path, {k: list(v) for k, v in conf_cohorts.items()}, compact=True)

    file_hashes = {name: _file_sha256(p) for name, p in data_files.items()}
    counts = {
        "d1_worlds": len(d1_dataset.worlds),
        "d1_steps": len(d1_dataset.steps),
        "dev_records": len(dev_worlds),
        "conf_records": len(conf_worlds),
    }
    audit_payload: dict[str, Any] = {
        "schema": "elendril-v8-split-audit-v1",
        "cpu_smoke": bool(cpu_smoke),
        "d1_quota_per_family": d1_quota,
        "dev_worlds_per_cohort": dev_wpc,
        "conf_worlds_per_cohort": conf_wpc,
        "confirmation_seed": conf_seed,
        "d1_identity": d1_dataset.identity,
        "d1_structural_hashes_sha256": d1_hashes_sha,
        "dev_identity": dev_identity,
        "conf_identity": conf_identity,
        "counts": counts,
        **counts,
        "d1_audit": d1_dataset.audit,
        "d1_shortcuts": d1_dataset.shortcuts,
        "d1_crosstabs": d1_dataset.crosstabs,
        "disjointness_audit": disjoint_audit,
        "file_hashes": file_hashes,
        "audit_passed": True,
    }
    audit_payload["split_audit_sha256"] = digest(audit_payload)
    _atomic_write_json(audit_path, audit_payload, compact=False)

    return {
        "splits_dir": str(splits_dir),
        "cached": False,
        "audit": audit_payload,
        "split_audit_sha256": str(audit_payload["split_audit_sha256"]),
        "d1_identity": d1_dataset.identity,
        "d1_structural_hashes_sha256": d1_hashes_sha,
        "dev_identity": dev_identity,
        "conf_identity": conf_identity,
        "d1_dataset": d1_dataset,
        "d1_structural_hashes": d1_dataset.structural_hashes,
        "dev_cohorts": dev_cohorts,
        "conf_cohorts": conf_cohorts,
    }


def _select_stratified_probe_steps(
    steps: tuple[TeacherStep, ...] | list[TeacherStep],
    *,
    max_steps: int,
) -> tuple[TeacherStep, ...]:
    fn = getattr(_v8_eval, "select_stratified_probe_steps", None)
    if callable(fn):
        return tuple(fn(steps, max_steps=max_steps))
    if len(steps) <= max_steps:
        return tuple(steps)
    by_family: dict[str, list[TeacherStep]] = defaultdict(list)
    for st in steps:
        by_family[str(st.family)].append(st)
    families = sorted(by_family)
    if not families:
        return ()
    selected: list[TeacherStep] = []
    idx = 0
    while len(selected) < max_steps:
        added = False
        for fam in families:
            pool = by_family[fam]
            if idx < len(pool):
                selected.append(pool[idx])
                added = True
                if len(selected) >= max_steps:
                    break
        if not added:
            break
        idx += 1
    return tuple(selected)


def _d0_teacher_steps() -> tuple[TeacherStep, ...]:
    d0 = load_d0_dataset()
    steps: list[TeacherStep] = []
    for ex in d0.examples:
        if ex.target == "Q":
            gold_op, gold_arg = "QUERY", None
        elif ex.target == "R":
            gold_op, gold_arg = "RETRIEVE", None
        elif ex.target == "!":
            gold_op, gold_arg = "ABSTAIN", None
        elif len(ex.target) == 1 and ex.target in SYMBOLS:
            gold_op, gold_arg = "ANSWER", ex.target
        else:
            # Skip non-V8 action tokens (e.g. deliberate) if any ever existed
            continue
        pub = PublicInput(
            text_utf8=ex.prompt,
            query_available=("Query=1" in ex.prompt),
            memory_available=("Memory=1" in ex.prompt),
            action_budget=4,
        )
        steps.append(
            TeacherStep(
                task_id=ex.task_id,
                world_id=ex.world_id,
                family=ex.kind,
                stage=ex.stage,
                public_input=pub,
                gold_op=gold_op,
                gold_arg=gold_arg,
                target_text=ex.target,
                prompt_bytes=len(ex.prompt.encode("utf-8")),
                is_answer_step=(gold_op == "ANSWER"),
            )
        )
    return tuple(steps)


def _compute_exact_probe_breakdown(
    model: Any,
    v8_spec: V8ModelSpec,
    probe_steps: tuple[TeacherStep, ...] | list[TeacherStep],
) -> dict[str, Any]:
    """Compute exact full-action accuracy and separated op/val metrics without approximations."""

    n = len(probe_steps)
    if n == 0:
        raise ValueError("probe requires at least one TeacherStep")
    exact_matches = 0
    op_matches = 0
    ans_total = 0
    ans_op_matches = 0
    ans_val_matches = 0
    ans_val_given_op = 0
    non_ans_total = 0
    non_ans_op_matches = 0

    if v8_spec.architecture == "B0":
        pol = make_policy(model, v8_spec, model_identity="probe", mode="raw")
        for st in probe_steps:
            raw_prop, _ = pol.propose(st.public_input)
            pred_op = raw_prop.operation
            pred_arg = raw_prop.argument
            gold_op = st.gold_op
            gold_arg = st.gold_arg if gold_op == "ANSWER" else None
            # For QUERY, policy populates argument with channel name when available
            eff_pred_arg = pred_arg if pred_op == "ANSWER" else None
            op_ok = pred_op == gold_op
            if op_ok:
                op_matches += 1
            if gold_op == "ANSWER":
                ans_total += 1
                if op_ok:
                    ans_op_matches += 1
                if eff_pred_arg == gold_arg:
                    ans_val_matches += 1
                    if op_ok:
                        ans_val_given_op += 1
                        exact_matches += 1
            else:
                non_ans_total += 1
                if op_ok:
                    non_ans_op_matches += 1
                    exact_matches += 1
        return {
            "exact_action_accuracy": exact_matches / n,
            "op_accuracy": op_matches / n,
            "val_accuracy_on_answer_steps": (ans_val_matches / ans_total) if ans_total else None,
            "val_accuracy_given_op_correct": (ans_val_given_op / ans_op_matches) if ans_op_matches else None,
            "non_answer_op_accuracy": (non_ans_op_matches / non_ans_total) if non_ans_total else None,
        }

    device = next(model.parameters()).device
    samples = teacher_steps_to_typed_samples(tuple(probe_steps), max_length=v8_spec.max_sequence_length)
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(samples), 8):
                chunk = samples[start : start + 8]
                tokens, segments, _ = batch_tensors([(s.tokens, s.eligible) for s in chunk], device)
                action_pos = torch.tensor([s.action_position for s in chunk], device=device, dtype=torch.int64)
                gold_ops = torch.tensor([s.gold_op for s in chunk], device=device, dtype=torch.int64)
                gold_args = torch.tensor([s.gold_arg for s in chunk], device=device, dtype=torch.int64)
                out = model.forward_action(tokens, segments, action_pos)
                pred_op = out.log_p_op.argmax(dim=-1)
                pred_val = out.log_p_val.argmax(dim=-1)
                op_ok = pred_op == gold_ops
                is_ans = gold_ops == 0
                val_ok = pred_val == gold_args
                exact = torch.where(is_ans, op_ok & val_ok, op_ok)
                exact_matches += int(exact.sum().item())
                op_matches += int(op_ok.sum().item())
                ans_total += int(is_ans.sum().item())
                ans_op_matches += int((op_ok & is_ans).sum().item())
                ans_val_matches += int((val_ok & is_ans).sum().item())
                ans_val_given_op += int((op_ok & val_ok & is_ans).sum().item())
                non_ans_total += int((~is_ans).sum().item())
                non_ans_op_matches += int((op_ok & (~is_ans)).sum().item())
    finally:
        for m, was in modes:
            m.training = was

    return {
        "exact_action_accuracy": exact_matches / n,
        "op_accuracy": op_matches / n,
        "val_accuracy_on_answer_steps": (ans_val_matches / ans_total) if ans_total else None,
        "val_accuracy_given_op_correct": (ans_val_given_op / ans_op_matches) if ans_op_matches else None,
        "non_answer_op_accuracy": (non_ans_op_matches / non_ans_total) if non_ans_total else None,
    }


def _run_probe(
    model: Any,
    v8_spec: V8ModelSpec,
    probe_steps: tuple[TeacherStep, ...] | list[TeacherStep],
    *,
    dataset_name: str,
) -> dict[str, Any]:
    diag = dict(diagnostic_probe(model, v8_spec, probe_steps, max_probe_steps=len(probe_steps)))
    families_covered = sorted({str(st.family) for st in probe_steps})
    if (
        "exact_action_accuracy" not in diag
        or diag["exact_action_accuracy"] is None
        or "non_answer_op_accuracy" not in diag
    ):
        exact_stats = _compute_exact_probe_breakdown(model, v8_spec, probe_steps)
        for k, v in exact_stats.items():
            if k not in diag or diag[k] is None:
                diag[k] = v

    if "teacher_loss" not in diag or diag["teacher_loss"] is None:
        if "mean_total_nll" in diag and diag["mean_total_nll"] is not None:
            diag["teacher_loss"] = float(diag["mean_total_nll"])
        elif "first_action_byte_nll" in diag and diag["first_action_byte_nll"] is not None:
            diag["teacher_loss"] = float(diag["first_action_byte_nll"])
        else:
            diag["teacher_loss"] = float(diag.get("byte_nll_per_token", 0.0))

    diag.setdefault("families_covered", families_covered)
    existing_scope = diag.get("probe_scope")
    scope_dict: dict[str, Any] = {
        "dataset": dataset_name,
        "count": len(probe_steps),
        "families_covered": families_covered,
    }
    if isinstance(existing_scope, str):
        scope_dict["strategy"] = existing_scope
    elif isinstance(existing_scope, dict):
        scope_dict = {**existing_scope, **scope_dict}
    diag["probe_scope"] = scope_dict
    return diag


def run_canary(
    *,
    arm: str,
    source_commit: str,
    work_dir: Path,
    device: str = "cuda:0",
    cpu_canary: bool = False,
    lr: float | None = None,
) -> dict[str, object]:
    """Run the positive-control canary check (action/episode accuracy & loss drop gates) for B0 or typed arms."""

    d1_src = build_d1_training_dataset(quota_per_family=16 if cpu_canary else 64)
    canary = make_positive_canary_cohort(d1_src, worlds_target=32)
    v8_spec = _tiny_cpu_spec(arm) if cpu_canary else build_v8_model_spec(arm)
    steps = _select_stratified_probe_steps(canary.steps, max_steps=32) if cpu_canary else canary.steps
    updates = 6 if cpu_canary else 120
    batch = min(16, len(steps))
    effective_lr = float(lr) if lr is not None else (1e-3 if cpu_canary else float(OPTIMIZER_DEFAULTS["peak_lr"]))
    recipe = Recipe(
        seed=7,
        total_updates=updates,
        batch_size=batch,
        micro_batch_size=min(2, batch),
        lr=effective_lr,
        schedule="cosine",
        precision="fp32",
    )
    prov = {
        "split": "train",
        "license": "internal-v8-canary",
        "canary_identity": canary.identity,
    }
    configure_device(device)
    if arm == "B0":
        byte_samples = teacher_steps_to_byte_samples(steps, max_length=v8_spec.max_sequence_length)
        trainer = IndependentOrderTrainer(
            byte_samples,
            spec=v8_spec.base_spec,
            recipe=recipe,
            source_commit=source_commit,
            root=work_dir / "canary_ckpt",
            lineage=f"canary-{arm}",
            data_provenance=prov,
            tokenizer_manifest=UTF8.manifest.record(),
            device=device,
            method="uniform",
            order_seed=7,
        )
        trainer.v8_spec = v8_spec
    else:
        typed_samples = teacher_steps_to_typed_samples(steps, max_length=v8_spec.max_sequence_length)
        trainer = V8TypedTrainer(
            typed_samples,
            v8_spec=v8_spec,
            recipe=recipe,
            order_seed=7,
            source_commit=source_commit,
            root=work_dir / "canary_ckpt",
            lineage=f"canary-{arm}",
            data_provenance=prov,
            device=device,
            cpu_canary=cpu_canary,
        )

    initial_diag = _run_probe(trainer.model, v8_spec, steps, dataset_name="D1-canary")
    initial_probe_nll = float(initial_diag["teacher_loss"])
    losses: list[float] = []
    t0 = time.perf_counter()
    for _ in range(updates):
        m = trainer.step()
        losses.append(float(m["loss"]))
    wall = time.perf_counter() - t0
    trainer.save()

    diag = _run_probe(trainer.model, v8_spec, steps, dataset_name="D1-canary")
    final_probe_nll = float(diag["teacher_loss"])
    eps = evaluate_worlds_by_mode(
        trainer.model, v8_spec, canary.worlds[:32], model_identity=f"canary-{arm}", modes=("raw",)
    )
    ep_acc = sum(1 for e in eps["raw"] if e.outcome.correct) / len(eps["raw"])
    exact_action_acc = float(diag["exact_action_accuracy"])
    loss_ratio = final_probe_nll / max(initial_probe_nll, 1e-12)
    passed = (
        (ep_acc >= 0.90 and exact_action_acc >= 0.95 and loss_ratio <= 0.20)
        if not cpu_canary
        else (final_probe_nll < initial_probe_nll)
    )

    result = {
        "schema": "elendril-v8-canary-v1",
        "arm": arm,
        "cpu_canary": cpu_canary,
        "lr": effective_lr,
        "updates": updates,
        "initial_loss": initial_probe_nll,
        "final_loss": final_probe_nll,
        "teacher_loss": final_probe_nll,
        "loss_ratio": loss_ratio,
        "raw_episode_accuracy": ep_acc,
        "exact_action_accuracy": exact_action_acc,
        "raw_action_accuracy": exact_action_acc,
        "op_accuracy": diag.get("op_accuracy"),
        "val_accuracy_on_answer_steps": diag.get("val_accuracy_on_answer_steps"),
        "val_accuracy_given_op_correct": diag.get("val_accuracy_given_op_correct"),
        "non_answer_op_accuracy": diag.get("non_answer_op_accuracy"),
        "families_covered": diag.get("families_covered"),
        "probe_scope": diag.get("probe_scope"),
        "probe": diag,
        "wall_seconds": wall,
        "passed": passed,
    }
    _atomic_write_json(work_dir / "CANARY_RESULT.json", result)
    return result


def _compare_optimizer_states(sd_a: dict[str, Any], sd_b: dict[str, Any]) -> bool:
    if sd_a.get("param_groups") != sd_b.get("param_groups"):
        return False
    state_a = sd_a.get("state", {})
    state_b = sd_b.get("state", {})
    if set(state_a) != set(state_b):
        return False
    for pid in state_a:
        slot_a = state_a[pid]
        slot_b = state_b[pid]
        if set(slot_a) != set(slot_b):
            return False
        for key in slot_a:
            va = slot_a[key]
            vb = slot_b[key]
            if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
                if not torch.equal(va, vb):
                    return False
            elif va != vb:
                return False
    return True


def run_profile(
    *,
    job_id: str | None = None,
    arm: str | None = None,
    dataset: str | None = None,
    device_index: int | None = None,
    source_commit: str,
    work_dir: Path,
    device: str = "cuda:0",
    cpu_canary: bool = False,
) -> dict[str, object]:
    """Measure full-horizon save@2->resume@4 state parity, update speed, and interactive eval timing."""

    job = _resolve_profile_job(job_id=job_id, arm=arm, dataset=dataset)
    resolved_job_id = str(job["job_id"])
    resolved_arm = str(job["arm"])
    resolved_dataset = str(job["dataset"])
    if device_index is not None and torch.cuda.is_available() and not cpu_canary and device != "cpu":
        torch.cuda.set_device(int(device_index))

    v8_spec = _tiny_cpu_spec(resolved_arm) if cpu_canary else None
    data = prepare_v8_training_data(
        resolved_dataset,
        d1_worlds_per_family=16 if cpu_canary else 512,
        sample_limit_for_cpu=32 if cpu_canary else None,
    )
    prof_updates = 4 if cpu_canary else 20

    # 1. Full-horizon resume parity test (save@2 -> fresh CheckpointStore/trainer resume@4 vs continuous 4)
    # Never pass updates_override so full registered cosine horizon is constructed!
    cont_root = work_dir / "prof_cont"
    res_root = work_dir / "prof_res"
    t_cont = create_v8_trainer(
        data,
        job,
        source=source_commit,
        root=cont_root,
        device=device,
        cpu_canary=cpu_canary,
        v8_spec_override=v8_spec,
        updates_override=None,
    )
    for _ in range(4):
        t_cont.step()

    t_res_a = create_v8_trainer(
        data,
        job,
        source=source_commit,
        root=res_root,
        device=device,
        cpu_canary=cpu_canary,
        v8_spec_override=v8_spec,
        updates_override=None,
    )
    t_res_a.step()
    t_res_a.step()
    sha2 = t_res_a.save()

    # Verify independent CheckpointStore readback before resuming fresh trainer instance
    ckpt_store = CheckpointStore(res_root, resolved_job_id)
    restored_state, _ = ckpt_store.restore(sha2)

    t_res_b = create_v8_trainer(
        data,
        job,
        source=source_commit,
        root=res_root,
        resume=True,
        device=device,
        cpu_canary=cpu_canary,
        v8_spec_override=v8_spec,
        updates_override=None,
    )
    t_res_b.step()
    t_res_b.step()

    weights_match = set(t_cont.model.state_dict()) == set(t_res_b.model.state_dict()) and all(
        torch.equal(t_cont.model.state_dict()[k], t_res_b.model.state_dict()[k])
        for k in t_cont.model.state_dict()
    )
    optimizer_match = _compare_optimizer_states(t_cont.optimizer.state_dict(), t_res_b.optimizer.state_dict())
    horizon = t_cont.recipe.total_updates
    probe_u = sorted({0, 1, 2, 3, 4, max(0, horizon // 2), max(0, horizon - 1)})
    scheduler_match = (
        t_cont.recipe.total_updates == t_res_b.recipe.total_updates
        and t_cont.scaler.state_dict() == t_res_b.scaler.state_dict()
        and t_cont.metadata == t_res_b.metadata
        and all(t_cont.learning_rate(u) == t_res_b.learning_rate(u) for u in probe_u)
    )
    cursor_match = (
        t_cont.state.cursor == t_res_b.state.cursor
        and t_cont.sample_cursor(t_cont.state.global_update) == t_res_b.sample_cursor(t_res_b.state.global_update)
        and t_cont.plan == t_res_b.plan
    )
    rng_match = (
        t_cont.rng_bytes() == t_res_b.rng_bytes()
        and t_cont.state.rng_state_sha256 == t_res_b.state.rng_state_sha256
        and torch.equal(t_cont.rng.get_state(), t_res_b.rng.get_state())
    )
    per_epoch_cont = max(1, math.ceil(len(t_cont.samples) / t_cont.recipe.batch_size))
    per_epoch_res = max(1, math.ceil(len(t_res_b.samples) / t_res_b.recipe.batch_size))
    cont_epoch = (t_cont.state.global_update - 1) // per_epoch_cont
    res_epoch = (t_res_b.state.global_update - 1) // per_epoch_res
    cont_samples_seen = t_cont.state.global_update * t_cont.recipe.batch_size
    res_samples_seen = t_res_b.state.global_update * t_res_b.recipe.batch_size
    training_state_match = (
        restored_state.global_update == 2
        and t_cont.state.global_update == t_res_b.state.global_update == 4
        and t_cont.state.generation == t_res_b.state.generation
        and t_cont.state.cumulative_tokens == t_res_b.state.cumulative_tokens
        and t_cont.state.token_budget == t_res_b.state.token_budget
        and t_cont.state.tokens_per_update == t_res_b.state.tokens_per_update
        and t_cont.state.tokens_by_source == t_res_b.state.tokens_by_source
        and t_cont.state.optimizer_step_max == t_res_b.state.optimizer_step_max
        and t_cont.state.schedule_tokens == t_res_b.state.schedule_tokens
        and t_cont.state.identities == t_res_b.state.identities
        and cont_epoch == res_epoch
        and cont_samples_seen == res_samples_seen
    )
    recovery_parity = {
        "weights_match": bool(weights_match),
        "optimizer_match": bool(optimizer_match),
        "scheduler_match": bool(scheduler_match),
        "cursor_match": bool(cursor_match),
        "rng_match": bool(rng_match),
        "training_state_match": bool(training_state_match),
        "all_match": bool(
            weights_match
            and optimizer_match
            and scheduler_match
            and cursor_match
            and rng_match
            and training_state_match
        ),
        "full_horizon_total_updates": horizon,
    }
    if not recovery_parity["all_match"]:
        raise AssertionError(f"profile save@2 -> resume@4 failed recovery parity: {recovery_parity}")

    # 2. Timing profile over prof_updates steps (using full registered horizon)
    time_root = work_dir / "prof_timing"
    if not cpu_canary and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_prof = create_v8_trainer(
        data,
        job,
        source=source_commit,
        root=time_root,
        device=device,
        cpu_canary=cpu_canary,
        v8_spec_override=v8_spec,
        updates_override=None,
    )
    actual_prof_updates = min(prof_updates, t_prof.recipe.total_updates)
    t_prof.step()  # warmup step
    t0 = time.perf_counter()
    for _ in range(max(1, actual_prof_updates - 1)):
        t_prof.step()
    if torch.cuda.is_available() and t_prof.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - t0, 1e-6)
    measured_steps = max(1, actual_prof_updates - 1)
    peak_mb = (
        float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        if (torch.cuda.is_available() and t_prof.device.type == "cuda")
        else 0.0
    )

    # 3. Interactive episode evaluation timing on a real dev world sample
    sample_suite = build_d1_dev_suite(worlds_per_cohort=1)
    sample_worlds = list(sample_suite["R09"][:5])
    t_eval0 = time.perf_counter()
    sample_eps = evaluate_worlds_by_mode(
        t_prof.model,
        t_prof.v8_spec,
        sample_worlds,
        model_identity=f"prof-{resolved_job_id}",
        modes=("raw",),
    )
    if torch.cuda.is_available() and t_prof.device.type == "cuda":
        torch.cuda.synchronize()
    eval_elapsed = max(time.perf_counter() - t_eval0, 1e-6)
    measured_eval_episodes = max(1, len(sample_eps["raw"]))
    eval_seconds_per_episode = eval_elapsed / measured_eval_episodes

    result = {
        "schema": "elendril-v8-profile-v1",
        "job_id": resolved_job_id,
        "arm": resolved_arm,
        "dataset": resolved_dataset,
        "device_index": device_index,
        "source_commit": source_commit,
        "measured_updates": measured_steps,
        "seconds_per_update": elapsed / measured_steps,
        "measured_eval_episodes": measured_eval_episodes,
        "eval_seconds_per_episode": eval_seconds_per_episode,
        "peak_vram_mb": peak_mb,
        "recovery_parity": recovery_parity,
        "resume_parity_at_4_verified": recovery_parity["all_match"],
        "total_training_samples": len(t_prof.samples),
        "estimated_full_updates": int(t_prof.recipe.total_updates),
    }
    _atomic_write_json(work_dir / "WORKER_PROFILE.json", result)
    return result


def _reconstruct_episode_steps(
    ep: EpisodeRecord,
    world: dict[str, object],
    *,
    render_fn=render_public_v8,
) -> tuple[list[dict[str, Any]], int | None]:
    """Reconstruct full public text and tool responses per step without touching model weights."""

    ordered_facts = [PublicFact(**row) for row in world["initial_facts"]]
    seen_facts = set(ordered_facts)
    query_avail = bool(world["allowed_public_tools"]["query_available"])
    mem_avail = bool(world["allowed_public_tools"]["memory_available"])
    goal_subject = str(world["goal_subject"])
    hops = int(world["hops"])
    template_id = int(world["renderer_template"])
    alias_prefix = str(world["alias_prefix"])
    order_seed = int(world["row_order_seed"])
    queried = False
    retrieved = False

    serialized_steps: list[dict[str, Any]] = []
    first_div_step: int | None = None

    for st in ep.steps:
        text = render_fn(
            ordered_facts=tuple(ordered_facts),
            goal_subject=goal_subject,
            hops=hops,
            query_available=query_avail,
            memory_available=mem_avail,
            template_id=template_id,
            alias_prefix=alias_prefix,
            order_seed=order_seed,
            decision=st.decision,
        )
        text_sha = sha256(text.encode("utf-8")).hexdigest()
        if text_sha != st.public_text_sha256:
            raise AssertionError(
                f"reconstructed public text sha256 mismatch for {ep.world_id} decision={st.decision}"
            )
        op = st.executed_proposal.operation
        tool_responses: list[dict[str, object]] = []
        if op == "QUERY" and query_avail:
            tool_responses = [dict(r) for r in world["hidden"]["query_payload"]]
            if not queried:
                queried = True
                for row in world["hidden"]["query_payload"]:
                    f = PublicFact(**row)
                    if f not in seen_facts:
                        seen_facts.add(f)
                        ordered_facts.append(f)
        elif op == "RETRIEVE" and mem_avail:
            tool_responses = [dict(r) for r in world["hidden"]["memory_payload"]]
            if not retrieved:
                retrieved = True
                for row in world["hidden"]["memory_payload"]:
                    f = PublicFact(**row)
                    if f not in seen_facts:
                        seen_facts.add(f)
                        ordered_facts.append(f)

        if first_div_step is None and (st.diverged_from_raw or st.diverged_from_teacher):
            first_div_step = int(st.decision)

        serialized_steps.append(
            {
                "step_index": int(st.decision),
                "public_text_utf8": text,
                "public_text_sha256": st.public_text_sha256,
                "raw_proposal": asdict(st.raw_proposal),
                "executed_proposal": asdict(st.executed_proposal),
                "ideal_operation": st.ideal_operation,
                "ideal_argument": st.ideal_argument,
                "tool_responses": tool_responses,
                "diverged_from_raw": bool(st.diverged_from_raw),
                "diverged_from_teacher": bool(st.diverged_from_teacher),
            }
        )

    return serialized_steps, first_div_step


def _serialize_episode_record(
    *,
    job_id: str,
    milestone: str,
    ep: EpisodeRecord,
    world: dict[str, object],
    render_fn=render_public_v8,
) -> dict[str, Any]:
    steps_payload, first_div_step = _reconstruct_episode_steps(ep, world, render_fn=render_fn)
    predicted_answer: str | None = None
    if ep.steps and ep.steps[-1].executed_proposal.operation == "ANSWER":
        predicted_answer = ep.steps[-1].executed_proposal.argument
    return {
        "job_id": job_id,
        "milestone": milestone,
        "split": str(world.get("split", "validation")),
        "cohort": ep.cohort,
        "world_id": ep.world_id,
        "base_world_id": ep.base_world_id,
        "family": ep.family,
        "form": ep.form,
        "structure_hash": _world_structure_hash(world),
        "is_isolated_one_edge_probe": _is_isolated_probe(world),
        "mode": ep.mode,
        "exact_match": bool(ep.outcome.correct),
        "correct": bool(ep.outcome.correct),
        "terminal_status": ep.outcome.terminal_status,
        "predicted_answer": predicted_answer,
        "expected_answer": world["hidden"]["expected"],
        "first_divergence_step": first_div_step,
        "first_observable_divergence": ep.first_observable_divergence,
        "model_calls": ep.model_calls,
        "evaluated_positions": ep.evaluated_positions,
        "allowed_public_tools": dict(world["allowed_public_tools"]),
        "goal_subject": str(world["goal_subject"]),
        "hops": int(world["hops"]),
        "initial_facts": [dict(r) for r in world["initial_facts"]],
        "query_payload": [dict(r) for r in world["hidden"]["query_payload"]],
        "memory_payload": [dict(r) for r in world["hidden"]["memory_payload"]],
        "steps": steps_payload,
    }


def _parse_tools_from_public_text(text: str) -> tuple[bool, bool] | None:
    m = re.search(r"Query=([01])\s+Memory=([01])", text)
    if m:
        return bool(int(m.group(1))), bool(int(m.group(2)))
    return None


def rescore_saved_episode_jsonl(episode_jsonl_path: Path | str) -> dict[str, Any]:
    """Pure non-neural independent validator for saved episodes/<job_id>.jsonl traces."""

    path = Path(episode_jsonl_path)
    if not path.exists():
        return {
            "verified": False,
            "episode_count": 0,
            "errors": [f"episode file not found: {path}"],
            "exact_accuracy": 0.0,
            "non_isolated_exact_accuracy": None,
            "by_cohort": {},
            "by_mode": {},
            "first_divergence_histogram": {},
        }

    errors: list[str] = []
    episode_count = 0
    total_correct = 0
    non_iso_total = 0
    non_iso_correct = 0
    iso_total = 0
    iso_correct = 0
    raw_total = 0
    raw_correct = 0
    raw_non_iso_total = 0
    raw_non_iso_correct = 0

    cohort_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "correct": 0, "non_iso_n": 0, "non_iso_correct": 0}
    )
    mode_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "correct": 0, "non_iso_n": 0, "non_iso_correct": 0}
    )
    div_hist: Counter[str] = Counter()

    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            episode_count += 1
            try:
                row = json.loads(line)
            except Exception as exc:
                errors.append(f"line {line_no}: invalid JSON ({exc})")
                continue

            wid = str(row.get("world_id", f"line-{line_no}"))
            mode = str(row.get("mode", "unknown"))
            cohort = str(row.get("cohort", "unknown"))
            is_isolated = bool(row.get("is_isolated_one_edge_probe", False))
            steps = row.get("steps")

            if not isinstance(steps, list) or len(steps) == 0:
                if row.get("explicitly_invalid") is True:
                    saved_match = bool(row.get("exact_match", row.get("correct", False)))
                    if saved_match:
                        errors.append(f"line {line_no} ({wid}/{mode}): explicitly_invalid episode marked exact_match=True")
                    div_hist["none"] += 1
                    cohort_stats[cohort]["n"] += 1
                    mode_stats[mode]["n"] += 1
                    if not is_isolated:
                        non_iso_total += 1
                        cohort_stats[cohort]["non_iso_n"] += 1
                        mode_stats[mode]["non_iso_n"] += 1
                    else:
                        iso_total += 1
                    if mode == "raw":
                        raw_total += 1
                        if not is_isolated:
                            raw_non_iso_total += 1
                    continue
                errors.append(f"line {line_no} ({wid}/{mode}): missing or empty steps")
                continue

            # Resolve allowed_public_tools from saved world metadata or step 0 public text
            allowed_tools = row.get("allowed_public_tools")
            if isinstance(allowed_tools, dict):
                query_avail = bool(allowed_tools.get("query_available", False))
                mem_avail = bool(allowed_tools.get("memory_available", False))
            else:
                parsed = _parse_tools_from_public_text(str(steps[0].get("public_text_utf8", "")))
                if parsed is None:
                    errors.append(f"line {line_no} ({wid}/{mode}): missing allowed_public_tools")
                    continue
                query_avail, mem_avail = parsed

            expected_answer = row.get("expected_answer")
            if all(k in row for k in ("initial_facts", "goal_subject", "hops")):
                recon_world = {
                    "goal_subject": row["goal_subject"],
                    "hops": int(row["hops"]),
                    "initial_facts": list(row["initial_facts"]),
                    "allowed_public_tools": {"query_available": query_avail, "memory_available": mem_avail},
                    "hidden": {
                        "expected": expected_answer,
                        "query_payload": list(row.get("query_payload", [])),
                        "memory_payload": list(row.get("memory_payload", [])),
                    },
                }
                ind_exp = independent_expected_outcome(recon_world)
                if ind_exp != expected_answer:
                    errors.append(
                        f"line {line_no} ({wid}/{mode}): independent expected {ind_exp!r} != saved {expected_answer!r}"
                    )

            executed_actions: list[tuple[str, str | None]] = []
            recon_first_div: int | None = None
            q_avail_sim = query_avail
            m_avail_sim = mem_avail

            for idx, st in enumerate(steps):
                pub_txt = st.get("public_text_utf8")
                pub_sha = st.get("public_text_sha256")
                if isinstance(pub_txt, str) and isinstance(pub_sha, str):
                    if sha256(pub_txt.encode("utf-8")).hexdigest() != pub_sha:
                        errors.append(f"line {line_no} ({wid}/{mode}) step {idx}: public_text_sha256 mismatch")

                raw_prop = st.get("raw_proposal", {})
                exec_prop = st.get("executed_proposal", {})
                raw_op = raw_prop.get("operation")
                raw_arg = raw_prop.get("argument")
                exec_op = exec_prop.get("operation")
                exec_arg = exec_prop.get("argument")
                ideal_op = st.get("ideal_operation")
                ideal_arg = st.get("ideal_argument")

                expected_div_raw = (exec_op, exec_arg) != (raw_op, raw_arg)
                if "diverged_from_raw" in st and bool(st["diverged_from_raw"]) != expected_div_raw:
                    errors.append(f"line {line_no} ({wid}/{mode}) step {idx}: diverged_from_raw mismatch")

                expected_div_teacher = (exec_op, exec_arg) != (
                    ideal_op,
                    ideal_arg if ideal_op == "ANSWER" else None,
                )
                if exec_op == "QUERY" and ideal_op == "QUERY":
                    expected_div_teacher = False
                if "diverged_from_teacher" in st and bool(st["diverged_from_teacher"]) != expected_div_teacher:
                    errors.append(f"line {line_no} ({wid}/{mode}) step {idx}: diverged_from_teacher mismatch")

                if recon_first_div is None and (expected_div_raw or expected_div_teacher):
                    recon_first_div = int(st.get("step_index", idx))

                executed_actions.append((str(exec_op), exec_arg))
                if exec_op == "INVALID":
                    break
                if exec_op == "QUERY":
                    if not q_avail_sim:
                        break
                    continue
                if exec_op == "RETRIEVE":
                    if not m_avail_sim:
                        break
                    continue
                if exec_op in ("ANSWER", "ABSTAIN"):
                    break

            if "first_divergence_step" in row and row["first_divergence_step"] != recon_first_div:
                errors.append(
                    f"line {line_no} ({wid}/{mode}): first_divergence_step {row['first_divergence_step']!r} != {recon_first_div!r}"
                )

            verify_world = {
                "hidden": {"expected": expected_answer},
                "allowed_public_tools": {"query_available": query_avail, "memory_available": mem_avail},
            }
            outcome = verify_episode_trace(verify_world, executed_actions)
            saved_match = row.get("exact_match", row.get("correct"))
            if saved_match is None or bool(saved_match) != outcome.correct:
                errors.append(
                    f"line {line_no} ({wid}/{mode}): saved exact_match={saved_match!r} != verifier {outcome.correct!r}"
                )
            if "correct" in row and bool(row["correct"]) != outcome.correct:
                errors.append(
                    f"line {line_no} ({wid}/{mode}): saved correct={row['correct']!r} != verifier {outcome.correct!r}"
                )

            if "predicted_answer" in row:
                last_op, last_arg = executed_actions[-1]
                expected_pred = last_arg if last_op == "ANSWER" else None
                if row["predicted_answer"] != expected_pred:
                    errors.append(
                        f"line {line_no} ({wid}/{mode}): predicted_answer={row['predicted_answer']!r} != {expected_pred!r}"
                    )

            div_key = "none" if recon_first_div is None else str(recon_first_div)
            div_hist[div_key] += 1

            is_ok = outcome.correct
            if is_ok:
                total_correct += 1
            cohort_stats[cohort]["n"] += 1
            mode_stats[mode]["n"] += 1
            if is_ok:
                cohort_stats[cohort]["correct"] += 1
                mode_stats[mode]["correct"] += 1

            if is_isolated:
                iso_total += 1
                if is_ok:
                    iso_correct += 1
            else:
                non_iso_total += 1
                cohort_stats[cohort]["non_iso_n"] += 1
                mode_stats[mode]["non_iso_n"] += 1
                if is_ok:
                    non_iso_correct += 1
                    cohort_stats[cohort]["non_iso_correct"] += 1
                    mode_stats[mode]["non_iso_correct"] += 1

            if mode == "raw":
                raw_total += 1
                if is_ok:
                    raw_correct += 1
                if not is_isolated:
                    raw_non_iso_total += 1
                    if is_ok:
                        raw_non_iso_correct += 1

    if episode_count == 0:
        errors.append(f"episode file has 0 episodes: {path}")

    def _fmt_bucket(stats: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for k in sorted(stats):
            s = stats[k]
            out[k] = {
                "n": s["n"],
                "correct": s["correct"],
                "exact_accuracy": (s["correct"] / s["n"]) if s["n"] else None,
                "non_isolated_n": s["non_iso_n"],
                "non_isolated_exact_accuracy": (
                    (s["non_iso_correct"] / s["non_iso_n"]) if s["non_iso_n"] else None
                ),
            }
        return out

    return {
        "verified": (len(errors) == 0 and episode_count > 0),
        "episode_count": episode_count,
        "errors": errors,
        "exact_accuracy": (total_correct / episode_count) if episode_count else 0.0,
        "non_isolated_exact_accuracy": (non_iso_correct / non_iso_total) if non_iso_total else None,
        "isolated_exact_accuracy": (iso_correct / iso_total) if iso_total else None,
        "raw_exact_accuracy": (raw_correct / raw_total) if raw_total else None,
        "raw_non_isolated_exact_accuracy": (
            (raw_non_iso_correct / raw_non_iso_total) if raw_non_iso_total else None
        ),
        "by_cohort": _fmt_bucket(cohort_stats),
        "by_mode": _fmt_bucket(mode_stats),
        "first_divergence_histogram": dict(sorted(div_hist.items())),
    }


def _reconcile_steps_jsonl_on_resume(steps_path: Path, resume_u: int) -> float | None:
    """Keep only committed step records <= resume_u and return the latest loss."""

    if not steps_path.exists():
        return None
    kept: dict[int, dict[str, Any]] = {}
    for raw_line in steps_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            u = int(row["global_update"])
        except Exception:
            continue
        if 1 <= u <= resume_u:
            kept[u] = row
    ordered = [kept[u] for u in sorted(kept)]
    text = "".join(canonical(r).decode("utf-8") + "\n" for r in ordered)
    _atomic_write_text(steps_path, text)
    return float(ordered[-1]["loss"]) if ordered else None


def _load_persisted_milestones(milestones_dir: Path, resume_u: int) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    if not milestones_dir.exists():
        return loaded
    for p in sorted(milestones_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            u = int(data.get("global_update", -1))
            name = str(data.get("milestone_name") or p.stem)
            if 0 <= u <= resume_u:
                loaded[name] = data
        except Exception:
            continue
    return loaded


def _save_milestone_file(
    milestones_dir: Path,
    *,
    milestone_name: str,
    global_update: int,
    checkpoint_generation: int,
    checkpoint_sha256: str | None,
    probe: dict[str, Any],
    suite_summary: dict[str, Any],
    loss: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "elendril-v8-milestone-v1",
        "milestone_name": milestone_name,
        "global_update": int(global_update),
        "checkpoint_generation": int(checkpoint_generation),
        "checkpoint_sha256": checkpoint_sha256,
        "loss": loss,
        "probe": probe,
        "suite_summary": suite_summary,
    }
    _atomic_write_json(milestones_dir / f"{milestone_name}.json", payload)
    return payload


def _build_training_data_from_splits(
    dataset_name: str,
    splits: dict[str, Any],
    *,
    sample_limit_for_cpu: int | None,
) -> V8TrainingData:
    if dataset_name == "D0":
        return prepare_v8_training_data("D0", sample_limit_for_cpu=sample_limit_for_cpu)
    if dataset_name == "D1":
        d1: V8Dataset = splits["d1_dataset"]
        steps = d1.steps[:sample_limit_for_cpu] if sample_limit_for_cpu is not None else d1.steps
        typed = teacher_steps_to_typed_samples(steps)
        byte_s = teacher_steps_to_byte_samples(steps)
        manifest = {
            "split": "train",
            "license": "internal-v8-procedural-corpus",
            "dataset": "D1",
            "d1_identity": d1.identity,
            "d1_audit": d1.audit,
            "v7_sampling_method": "uniform",
            "world_count": len(d1.worlds),
            "sample_count": len(steps),
            "slice_limit": sample_limit_for_cpu,
        }
        return V8TrainingData(
            dataset_name="D1",
            data_identity=digest(manifest) if sample_limit_for_cpu is not None else d1.identity,
            typed_samples=typed,
            byte_samples=byte_s,
            manifest=manifest,
        )
    raise ValueError(f"unsupported V8 dataset: {dataset_name!r}")


def run_job_worker(
    *,
    job_id: str,
    source_commit: str,
    work_dir: Path,
    device: str = "cuda:0",
    resume: bool = False,
    cpu_canary: bool = False,
    cpu_smoke: bool | None = None,
    updates_override: int | None = None,
    dev_worlds_per_cohort: int | None = None,
    conf_worlds_per_cohort: int | None = None,
    save_every: int = 64,
    confirmation_seed: int | None = None,
) -> dict[str, object]:
    """Train or recover one registered V8 job and run milestone + final evaluations."""

    use_cpu_smoke = bool(cpu_canary or cpu_smoke)
    job = _find_job(job_id)
    arm = str(job["arm"])
    dataset_name = str(job["dataset"])
    v8_spec = _tiny_cpu_spec(arm) if use_cpu_smoke else build_v8_model_spec(arm)

    splits = prepare_verified_splits(
        work_dir,
        cpu_smoke=use_cpu_smoke,
        dev_worlds_per_cohort=dev_worlds_per_cohort,
        conf_worlds_per_cohort=conf_worlds_per_cohort,
        confirmation_seed=confirmation_seed,
    )
    sample_limit = 32 if (use_cpu_smoke or device == "cpu") else None
    data = _build_training_data_from_splits(dataset_name, splits, sample_limit_for_cpu=sample_limit)

    ckpt_root = work_dir / "checkpoints"
    trainer = create_v8_trainer(
        data,
        job,
        source=source_commit,
        root=ckpt_root,
        resume=resume,
        device=device,
        cpu_canary=use_cpu_smoke,
        v8_spec_override=v8_spec if use_cpu_smoke else None,
        updates_override=updates_override,
    )
    total_updates = trainer.recipe.total_updates
    steps_per_epoch = max(1, math.ceil(len(trainer.samples) / trainer.recipe.batch_size))

    eval_cohorts: dict[str, tuple[dict[str, object], ...]] = splits["dev_cohorts"]
    all_eval_worlds = [w for rows in eval_cohorts.values() for w in rows]
    world_by_id = {str(w["world_id"]): w for w in all_eval_worlds}
    sub_worlds = [
        w
        for rows in eval_cohorts.values()
        for w in rows[: (5 * 2 if use_cpu_smoke else 5 * 32)]
    ]

    if dataset_name == "D0":
        all_probe_source = _d0_teacher_steps()
        probe_cap = 28 if use_cpu_smoke else 192
    else:
        d1_dataset: V8Dataset = splits["d1_dataset"]
        all_probe_source = d1_dataset.steps
        probe_cap = 28 if use_cpu_smoke else (len(all_probe_source) if len(all_probe_source) <= 512 else 192)
    probe_steps = _select_stratified_probe_steps(all_probe_source, max_steps=probe_cap)

    steps_dir = work_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    steps_path = steps_dir / f"{job_id}.jsonl"
    milestones_dir = work_dir / "milestones" / job_id
    milestones_dir.mkdir(parents=True, exist_ok=True)

    resume_u = int(trainer.state.global_update)
    if resume:
        last_loss = _reconcile_steps_jsonl_on_resume(steps_path, resume_u)
        milestones = _load_persisted_milestones(milestones_dir, resume_u)
    else:
        _atomic_write_text(steps_path, "")
        last_loss = None
        milestones = {}

    t0 = time.perf_counter()
    if trainer.state.global_update == 0 and "initial" not in milestones:
        init_worlds = sub_worlds if use_cpu_smoke else all_eval_worlds
        init_eps = evaluate_worlds_by_mode(
            trainer.model, v8_spec, init_worlds, model_identity=job_id, modes=("raw",)
        )
        init_summary = summarize_suite_episodes(init_eps, init_worlds)
        init_probe = _run_probe(trainer.model, v8_spec, probe_steps, dataset_name=dataset_name)
        milestones["initial"] = _save_milestone_file(
            milestones_dir,
            milestone_name="initial",
            global_update=0,
            checkpoint_generation=int(trainer.state.generation),
            checkpoint_sha256=trainer.parent,
            probe=init_probe,
            suite_summary=init_summary["raw"],
        )

    epoch_len = max(total_updates // 4, 1)
    milestone_updates = {
        epoch_len: "epoch-1",
        2 * epoch_len: "epoch-2",
        total_updates: "epoch-4",
    }

    stop_requested = False
    signal_received: list[int] = []

    def _handle_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        signal_received.append(signum)

    in_main_thread = threading.current_thread() is threading.main_thread()
    prev_sigterm = signal.getsignal(signal.SIGTERM) if in_main_thread else None
    prev_sigint = signal.getsignal(signal.SIGINT) if in_main_thread else None
    if in_main_thread:
        signal.signal(signal.SIGTERM, _handle_stop)
        signal.signal(signal.SIGINT, _handle_stop)

    partial_path = work_dir / "jobs" / f"{job_id}.partial.json"
    interrupted = False
    try:
        with steps_path.open("a", encoding="utf-8") as step_fh:
            while trainer.state.global_update < total_updates:
                if stop_requested:
                    interrupted = True
                    break
                metrics = trainer.step()
                last_loss = float(metrics["loss"])
                u = int(trainer.state.global_update)
                step_row = {
                    "global_update": u,
                    "epoch": int((u - 1) // steps_per_epoch),
                    "loss": last_loss,
                    "lr": float(metrics["lr"]),
                    "grad_norm": float(metrics["pre_clip"]),
                    "batch_size": int(
                        metrics.get("states") or metrics.get("examples") or len(metrics.get("indices", ()))
                    ),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                step_fh.write(canonical(step_row).decode("utf-8") + "\n")
                step_fh.flush()
                os.fsync(step_fh.fileno())

                if u % save_every == 0 or u == total_updates or u in milestone_updates or stop_requested:
                    if u > trainer.published_update:
                        trainer.save()

                if u in milestone_updates and u < total_updates and not stop_requested:
                    m_name = milestone_updates[u]
                    m_eps = evaluate_worlds_by_mode(
                        trainer.model, v8_spec, sub_worlds, model_identity=job_id, modes=("raw",)
                    )
                    m_summary = summarize_suite_episodes(m_eps, sub_worlds)["raw"]
                    m_probe = _run_probe(trainer.model, v8_spec, probe_steps, dataset_name=dataset_name)
                    milestones[m_name] = _save_milestone_file(
                        milestones_dir,
                        milestone_name=m_name,
                        global_update=u,
                        checkpoint_generation=int(trainer.state.generation),
                        checkpoint_sha256=trainer.parent,
                        probe=m_probe,
                        suite_summary=m_summary,
                        loss=last_loss,
                    )

                if stop_requested:
                    interrupted = True
                    break

        if interrupted:
            if trainer.state.global_update > trainer.published_update:
                trainer.save()
            partial_payload = {
                "schema": "elendril-v8-job-partial-v1",
                "status": "INTERRUPTED_CHECKPOINTED",
                "signal": signal_received[-1] if signal_received else None,
                "job_id": job_id,
                "condition_id": f"{arm}-{dataset_name}",
                "phase": job["phase"],
                "arm": arm,
                "dataset": dataset_name,
                "init_seed": int(job["init_seed"]),
                "order_seed": int(job["order_seed"]),
                "source_commit": source_commit,
                "completed_updates": int(trainer.state.global_update),
                "target_updates": int(total_updates),
                "checkpoint_sha256": trainer.parent,
                "split_audit_sha256": splits["split_audit_sha256"],
                "d1_structural_hashes_sha256": splits["d1_structural_hashes_sha256"],
                "dev_identity": splits["dev_identity"],
                "conf_identity": splits["conf_identity"],
                "steps_file": str(steps_path),
                "milestones_dir": str(milestones_dir),
                "milestones": milestones,
                "wall_seconds": time.perf_counter() - t0,
            }
            _atomic_write_json(partial_path, partial_payload)
            return partial_payload
    finally:
        if in_main_thread:
            if prev_sigterm is not None:
                signal.signal(signal.SIGTERM, prev_sigterm)
            if prev_sigint is not None:
                signal.signal(signal.SIGINT, prev_sigint)

    train_wall = time.perf_counter() - t0
    final_eps = evaluate_worlds_by_mode(trainer.model, v8_spec, all_eval_worlds, model_identity=job_id)
    final_summary = summarize_suite_episodes(final_eps, all_eval_worlds)
    final_probe = _run_probe(trainer.model, v8_spec, probe_steps, dataset_name=dataset_name)

    if "epoch-4" not in milestones:
        milestones["epoch-4"] = _save_milestone_file(
            milestones_dir,
            milestone_name="epoch-4",
            global_update=int(trainer.state.global_update),
            checkpoint_generation=int(trainer.state.generation),
            checkpoint_sha256=trainer.parent,
            probe=final_probe,
            suite_summary=final_summary["raw"],
            loss=last_loss,
        )

    pointer_interv = (
        run_pointer_interventions(trainer.model, v8_spec, probe_steps[:64])
        if arm == "P"
        else None
    )
    f03_ext = evaluate_f03_reverse_extension(trainer.model, v8_spec, model_identity=job_id)
    a20_ext = evaluate_a20_recovery_extension(
        trainer.model,
        v8_spec,
        eval_cohorts["E13"][: 5 * 4],
        model_identity=job_id,
    )

    ep_dir = work_dir / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)
    ep_file = ep_dir / f"{job_id}.jsonl"
    final_milestone_label = "epoch-4" if trainer.state.global_update >= total_updates else f"update-{trainer.state.global_update}"
    ep_lines: list[str] = []
    for mode, eps in final_eps.items():
        for ep in eps:
            world = world_by_id[ep.world_id]
            row = _serialize_episode_record(
                job_id=job_id,
                milestone=final_milestone_label,
                ep=ep,
                world=world,
                render_fn=render_public_v8,
            )
            ep_lines.append(canonical(row).decode("utf-8") + "\n")
    _atomic_write_text(ep_file, "".join(ep_lines))

    rescore_summary = rescore_saved_episode_jsonl(ep_file)
    if not rescore_summary["verified"]:
        raise ValueError(f"saved episode rescore failed for {job_id}: {rescore_summary['errors'][:5]}")

    raw_sum = final_summary["raw"]
    non_iso_acc = raw_sum.get("non_isolated_exact_accuracy")
    if non_iso_acc is None:
        non_iso_acc = rescore_summary.get("raw_non_isolated_exact_accuracy")

    result: dict[str, object] = {
        "schema": "elendril-v8-job-result-v1",
        "job_id": job_id,
        "condition_id": f"{arm}-{dataset_name}",
        "phase": job["phase"],
        "arm": arm,
        "dataset": dataset_name,
        "init_seed": int(job["init_seed"]),
        "order_seed": int(job["order_seed"]),
        "source_commit": source_commit,
        "completed_updates": int(trainer.state.global_update),
        "target_updates": int(total_updates),
        "checkpoint_sha256": trainer.parent,
        "split_audit_sha256": splits["split_audit_sha256"],
        "d1_structural_hashes_sha256": splits["d1_structural_hashes_sha256"],
        "dev_identity": splits["dev_identity"],
        "conf_identity": splits["conf_identity"],
        "total_parameters": v8_spec.total_parameters(),
        "wall_seconds": train_wall,
        "final_loss": last_loss,
        "raw_overall_accuracy": raw_sum["exact_episode_accuracy"],
        "raw_non_isolated_accuracy": non_iso_acc,
        "raw_r09_accuracy": raw_sum["by_cohort"].get("R09", {}).get("exact_accuracy"),
        "relevant_cf_both_correct": raw_sum["linked_counterfactuals"]["relevant_cf_both_correct"],
        "irrelevant_cf_stability": raw_sum["linked_counterfactuals"]["irrelevant_cf_stability"],
        "eval_model_calls": raw_sum["work_totals"]["model_calls"],
        "eval_evaluated_positions": raw_sum["work_totals"]["evaluated_positions"],
        "summary_by_mode": final_summary,
        "final_probe": final_probe,
        "pointer_interventions": pointer_interv,
        "extensions": {"F03": f03_ext, "A20": a20_ext},
        "milestones": milestones,
        "steps_file": str(steps_path),
        "milestones_dir": str(milestones_dir),
        "episodes_file": str(ep_file),
        "rescore_summary": rescore_summary,
    }

    if confirmation_seed is not None or str(job["phase"]) == "replication":
        conf_cohorts: dict[str, tuple[dict[str, object], ...]] = splits["conf_cohorts"]
        conf_worlds = [w for rows in conf_cohorts.values() for w in rows]
        conf_eps = evaluate_worlds_by_mode(
            trainer.model,
            v8_spec,
            conf_worlds,
            model_identity=job_id,
            modes=("raw",),
            render_fn=render_confirmation_public,
        )
        result["confirmation_summary"] = summarize_suite_episodes(conf_eps, conf_worlds)["raw"]

    out_dir = work_dir / "jobs"
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out_dir / f"{job_id}.json", result)
    if partial_path.exists():
        partial_path.unlink()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ELENDRIL V8 isolated GPU/CPU worker")
    parser.add_argument("mode", choices=("canary", "profile", "train", "recover", "frozen"))
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--arm", default="F")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--source", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-canary", action="store_true")
    parser.add_argument("--updates-override", type=int, default=None)
    parser.add_argument("--dev-worlds", type=int, default=None)
    parser.add_argument("--confirmation-seed", type=int, default=None)
    args = parser.parse_args(argv)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "canary":
        res = run_canary(
            arm=args.arm,
            source_commit=args.source,
            work_dir=args.work_dir,
            device=args.device,
            cpu_canary=args.cpu_canary,
        )
    elif args.mode == "profile":
        if not args.job_id and args.dataset is None:
            raise ValueError("--job-id or --dataset/--arm required for profile")
        res = run_profile(
            job_id=args.job_id,
            arm=args.arm if args.job_id is None else None,
            dataset=args.dataset,
            device_index=args.device_index,
            source_commit=args.source,
            work_dir=args.work_dir,
            device=args.device,
            cpu_canary=args.cpu_canary,
        )
    elif args.mode in ("train", "recover"):
        if not args.job_id:
            raise ValueError("--job-id required for train/recover")
        res = run_job_worker(
            job_id=args.job_id,
            source_commit=args.source,
            work_dir=args.work_dir,
            device=args.device,
            resume=(args.mode == "recover"),
            cpu_canary=args.cpu_canary,
            updates_override=args.updates_override,
            dev_worlds_per_cohort=args.dev_worlds,
            confirmation_seed=args.confirmation_seed,
        )
    else:
        if not args.job_id:
            raise ValueError("--job-id required for frozen")
        # Frozen evaluation restores latest checkpoint and evaluates 0 additional steps
        res = run_job_worker(
            job_id=args.job_id,
            source_commit=args.source,
            work_dir=args.work_dir,
            device=args.device,
            resume=True,
            cpu_canary=args.cpu_canary,
            updates_override=args.updates_override,
            dev_worlds_per_cohort=args.dev_worlds,
            confirmation_seed=args.confirmation_seed,
        )
    print(json.dumps({"status": res.get("status", "OK"), "mode": args.mode}, indent=2))
    return 0


__all__ = [
    "prepare_verified_splits",
    "rescore_saved_episode_jsonl",
    "run_canary",
    "run_job_worker",
    "run_profile",
]

if __name__ == "__main__":
    raise SystemExit(main())
