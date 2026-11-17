"""Isolated GPU/CPU worker CLI for ELENDRIL V8-X1 split prep, canaries, profiling, training, and frozen eval."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from anra_v5.elendril.codec import digest
from anra_v5.elendril.training import batch_tensors, configure_device, runtime_identity
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.elendril.v8x.analysis import summarize_x1_episodes
from anra_v5.elendril.v8x.contracts import (
    POLICY_MODES,
    X1ModelSpec,
    X1_D2_DATASET_SCHEMA,
    X1_DATASET_SCHEMA,
    build_x1_model_spec,
)
from anra_v5.elendril.v8x.data import (
    X1D1HDataset,
    X1D2Dataset,
    build_x1_d1h_training_dataset,
    build_x1_d2_paired_dataset,
    build_x1_eval_splits,
)
from anra_v5.elendril.v8x.evaluation import (
    X1S20EnvironmentKernel,
    X1S20ObservablePolicy,
    evaluate_x1_worlds_batched,
    run_x1_lesion_and_intervention_battery,
)
from anra_v5.elendril.v8x.models import build_x1_s20_model, expected_x1_active_parameter_names
from anra_v5.elendril.v8x.objectives import compute_x1_objective_batch
from anra_v5.elendril.v8x.protocol import (
    X1_INTERMEDIATE_WEIGHT_MILESTONES,
    X1_MILESTONES,
    X1_OPTIMIZER_DEFAULTS,
    X1_SESSIONS,
    session_jobs,
)
from anra_v5.elendril.v8x.training import X1S20Trainer, X1TrainRecipe, create_x1_s20_trainer
from v5_model.core import ModelSpec, packed_layout
from v5_training.checkpoint import CheckpointStore


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _cpu_micro_base() -> ModelSpec:
    return replace(
        MODEL_PROFILES["S20"],
        family="v5-s20-cpu-fixture",
        layers=1,
        width=16,
        query_heads=2,
        kv_heads=1,
        head_dimension=8,
        ffn_width=32,
        context_length=640,
    )


def _resolve_x1_spec(arch: str, *, cpu_fixtures: bool) -> X1ModelSpec:
    if cpu_fixtures:
        return build_x1_model_spec(
            arch,
            base_spec=_cpu_micro_base(),
            pointer_dim=8,
            slot_dim=8,
            num_slots=4,
            slot_edge_hidden=8,
            agg_iterations=2,
            msg_iterations=2,
            recurrent_dim=8,
        )
    return build_x1_model_spec(arch)


def find_x1_job(job_id: str) -> dict[str, Any]:
    for sess in X1_SESSIONS:
        for job in session_jobs(sess):
            if str(job["job_id"]) == job_id:
                return job
    raise ValueError(f"unknown X1 job_id: {job_id!r}")


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def prepare_splits(
    *,
    work_dir: Path,
    cpu_fixtures: bool = False,
) -> dict[str, Any]:
    """Generate and persist structurally verified D1-H / D2 train & evaluation splits."""

    quota = 1 if cpu_fixtures else 512
    d1h = build_x1_d1h_training_dataset(quota_per_family=quota)
    d2 = build_x1_d2_paired_dataset(base_worlds=d1h.worlds[: (12 if cpu_fixtures else len(d1h.worlds))])
    splits = build_x1_eval_splits(
        train_worlds=d1h.worlds,
        dev_per_family=1 if cpu_fixtures else 32,
        conf_per_family=1 if cpu_fixtures else 128,
        depth_per_hop=1 if cpu_fixtures else 64,
        milestone_per_family=1 if cpu_fixtures else 8,
    )
    data_dir = work_dir / "data"
    manifest = {
        "schema": "elendril-x1-prepared-splits-v1",
        "cpu_fixtures": bool(cpu_fixtures),
        "d1h_identity": d1h.data_identity,
        "d1h_world_count": len(d1h.worlds),
        "d1h_state_count": len(d1h.states),
        "d2_identity": d2.data_identity,
        "d2_group_count": len(d2.groups),
        "dev_views_count": len(splits["dev_views"]),
        "milestone_views_count": len(splits["milestone_views"]),
        "conf_views_count": len(splits["conf_views"]),
        "depth_stress_views_count": len(splits["depth_stress_views"]),
        "dev_views_sha256": digest([w["world_id"] for w in splits["dev_views"]]),
        "milestone_views_sha256": digest([w["world_id"] for w in splits["milestone_views"]]),
        "conf_views_sha256": digest([w["world_id"] for w in splits["conf_views"]]),
        "depth_stress_views_sha256": digest([w["world_id"] for w in splits["depth_stress_views"]]),
    }
    manifest["manifest_sha256"] = digest(manifest)
    _atomic_write_json(data_dir / "manifest.json", manifest)
    _atomic_write_json(
        data_dir / "eval_splits.json",
        {
            "dev_views": list(splits["dev_views"]),
            "milestone_views": list(splits["milestone_views"]),
            "conf_views": list(splits["conf_views"]),
            "depth_stress_views": list(splits["depth_stress_views"]),
        },
    )
    return manifest


def _load_or_build_splits(work_dir: Path, *, cpu_fixtures: bool) -> dict[str, Any]:
    cache = work_dir / "data" / "eval_splits.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    prepare_splits(work_dir=work_dir, cpu_fixtures=cpu_fixtures)
    return json.loads(cache.read_text(encoding="utf-8"))


def _load_training_corpus(dataset_name: str, *, objective: str, cpu_fixtures: bool) -> X1D1HDataset | X1D2Dataset:
    quota = 1 if cpu_fixtures else 512
    d1h = build_x1_d1h_training_dataset(quota_per_family=quota)
    if dataset_name == X1_D2_DATASET_SCHEMA or objective != "standard":
        base_w = d1h.worlds[:12] if cpu_fixtures else d1h.worlds
        return build_x1_d2_paired_dataset(base_worlds=base_w)
    return d1h


def run_arm_canary(
    *,
    arm: str,
    objective: str,
    device: str,
    work_dir: Path,
    source_commit: str,
    cpu_fixtures: bool = False,
) -> dict[str, Any]:
    """Run bounded Stage A canary: finite forward/backward, all active grads, 2-step update & resume parity."""

    dev = configure_device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device {device} without available GPU")

    spec = _resolve_x1_spec(arm, cpu_fixtures=cpu_fixtures)
    is_paired = objective != "standard"
    d1h = build_x1_d1h_training_dataset(quota_per_family=1)
    if is_paired:
        d2 = build_x1_d2_paired_dataset(base_worlds=d1h.worlds[:4])
        train_ds: X1D1HDataset | X1D2Dataset = d2
        groups = tuple(g for g in d2.groups if any(s.gold_op == 0 for s in g.samples))[:2] or d2.groups[:2]
        states = None
        sched = None
    else:
        ans = [s for s in d1h.states if s.gold_op == 0][:4]
        non = [s for s in d1h.states if s.gold_op != 0][:4]
        limited = tuple([*ans, *non])
        train_ds = X1D1HDataset(
            dataset_name=d1h.dataset_name,
            split=d1h.split,
            data_identity=d1h.data_identity,
            worlds=d1h.worlds,
            states=limited,
            interleaved_schedule=tuple(range(len(limited))),
            manifest=d1h.manifest,
        )
        states = limited
        sched = train_ds.interleaved_schedule
        groups = None

    recipe = X1TrainRecipe(
        seed=7,
        total_updates=2,
        batch_size=2 if is_paired else 4,
        micro_batch_size=1 if is_paired else 2,
        lr=1e-4,
        objective=objective,
        sampling="uniform" if is_paired else "interleaved",
    )
    canary_root = work_dir / f"canary_{arm}_{objective}"
    t0 = time.perf_counter()

    # Clean 0->2 trajectory
    tr_clean = X1S20Trainer(
        x1_spec=spec,
        recipe=recipe,
        order_seed=7,
        source_commit=source_commit,
        root=canary_root / "clean",
        lineage="canary_clean",
        data_provenance=train_ds.manifest,
        states=states,
        interleaved_schedule=sched,
        groups=groups,
        device=device,
        cpu_canary=cpu_fixtures,
    )
    # Check active parameter gradients explicitly on update 1
    step1 = tr_clean.step()
    tr_clean.save()
    step2 = tr_clean.step()
    tr_clean.save()

    # Restarted 0->1 -> save -> resume -> 2 trajectory
    tr_part = X1S20Trainer(
        x1_spec=spec,
        recipe=recipe,
        order_seed=7,
        source_commit=source_commit,
        root=canary_root / "resumed",
        lineage="canary_resume",
        data_provenance=train_ds.manifest,
        states=states,
        interleaved_schedule=sched,
        groups=groups,
        device=device,
        cpu_canary=cpu_fixtures,
    )
    tr_part.step()
    tr_part.save()
    tr_res = X1S20Trainer(
        x1_spec=spec,
        recipe=recipe,
        order_seed=7,
        source_commit=source_commit,
        root=canary_root / "resumed",
        lineage="canary_resume",
        data_provenance=train_ds.manifest,
        states=states,
        interleaved_schedule=sched,
        groups=groups,
        device=device,
        cpu_canary=cpu_fixtures,
        resume=True,
    )
    step2_res = tr_res.step()
    tr_res.save()

    weights_match = all(
        torch.equal(p1.detach().cpu(), p2.detach().cpu())
        for (_, p1), (_, p2) in zip(tr_clean.model.named_parameters(), tr_res.model.named_parameters())
    )
    loss_diff = abs(float(step2["loss"]) - float(step2_res["loss"]))
    wall = time.perf_counter() - t0
    passed = bool(
        tr_clean._first_update_norm_ok
        and weights_match
        and loss_diff <= 1e-6
        and os.path.exists(canary_root / "clean")
    )
    return {
        "schema": "elendril-x1-canary-v1",
        "arm": arm,
        "objective": objective,
        "device": str(dev),
        "total_parameters": spec.total_parameters(),
        "passed": passed,
        "step1_loss": float(step1["loss"]),
        "step2_loss": float(step2["loss"]),
        "step2_resumed_loss": float(step2_res["loss"]),
        "resume_weights_match": weights_match,
        "resume_loss_diff": loss_diff,
        "wall_seconds": float(wall),
        "runtime": runtime_identity(dev),
    }


def profile_x1_condition(
    *,
    job: Mapping[str, Any],
    device: str,
    work_dir: Path,
    source_commit: str,
    cpu_fixtures: bool = False,
) -> dict[str, Any]:
    """Measure real median/p95 step seconds, eval timing, and serialized checkpoint bytes."""

    dev = configure_device(device)
    arch = str(job["architecture"])
    obj = str(job["objective"])
    spec = _resolve_x1_spec(arch, cpu_fixtures=cpu_fixtures)
    ds = _load_training_corpus(str(job["dataset"]), objective=obj, cpu_fixtures=True if cpu_fixtures else False)
    prof_updates = 2 if cpu_fixtures else 4
    prof_root = work_dir / f"profile_{job['job_id']}"
    tr = create_x1_s20_trainer(
        ds,
        job,
        source_commit=source_commit,
        root=prof_root / "ckpts",
        device=device,
        cpu_canary=cpu_fixtures,
        x1_spec_override=spec,
        updates_override=prof_updates,
        batch_size_override=(2 if obj != "standard" else 4) if cpu_fixtures else None,
    )
    step_times: list[float] = []
    for _ in range(prof_updates):
        t0 = time.perf_counter()
        tr.step()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        step_times.append(time.perf_counter() - t0)
    tr.save()
    ms_dir = prof_root / "ms"
    tr.save_inference_milestone_weights(ms_dir, update=0, shared_backbone_dir=prof_root / "shared_bb")
    tr.save_inference_milestone_weights(ms_dir, update=prof_updates)

    splits = _load_or_build_splits(work_dir, cpu_fixtures=cpu_fixtures)
    sample_worlds = splits["milestone_views"][: (4 if cpu_fixtures else 16)]
    t_e0 = time.perf_counter()
    pol = X1S20ObservablePolicy(tr.model, spec, model_identity=f"prof:{job['job_id']}", mode="raw")
    _ = evaluate_x1_worlds_batched(pol, X1S20EnvironmentKernel(), sample_worlds, batch_size=8)
    eval_sec_per_ep = (time.perf_counter() - t_e0) / max(len(sample_worlds), 1)

    sorted_t = sorted(step_times)
    med = sorted_t[len(sorted_t) // 2]
    p95 = sorted_t[min(len(sorted_t) - 1, int(0.95 * (len(sorted_t) - 1)))]
    ckpt_bytes = _dir_size_bytes(prof_root / "ckpts")
    ms_bytes = _dir_size_bytes(ms_dir)
    return {
        "schema": "elendril-x1-profile-v1",
        "job_id": str(job["job_id"]),
        "architecture": arch,
        "objective": obj,
        "device": str(dev),
        "total_parameters": spec.total_parameters(),
        "median_step_seconds": float(med),
        "p95_step_seconds": float(p95),
        "eval_seconds_per_episode": float(eval_sec_per_ep),
        "measured_full_checkpoint_bytes": int(ckpt_bytes),
        "measured_milestone_bytes": int(ms_bytes),
    }


def _save_and_eval_milestone(
    *,
    trainer: X1S20Trainer,
    spec: X1ModelSpec,
    job: Mapping[str, Any],
    update: int,
    job_dir: Path,
    shared_bb_dir: Path,
    splits: Mapping[str, Any],
    is_final: bool,
) -> dict[str, Any]:
    ms_root = job_dir / "milestones"
    ms_dir = trainer.save_inference_milestone_weights(
        ms_root,
        update=update,
        shared_backbone_dir=shared_bb_dir,
    )
    worlds = splits["dev_views"] if is_final else splits["milestone_views"]
    kernel = X1S20EnvironmentKernel()
    pol = X1S20ObservablePolicy(
        trainer.model,
        spec,
        model_identity=f"{job['job_id']}:u{update}",
        mode="raw",
    )
    recs = evaluate_x1_worlds_batched(pol, kernel, worlds, batch_size=16)
    summary = summarize_x1_episodes(recs)
    _atomic_write_json(ms_dir / "summary.json", summary)
    _atomic_write_jsonl(ms_dir / "episodes.jsonl", recs)

    extra: dict[str, Any] = {}
    if is_final:
        subset_96 = splits["milestone_views"][:96]
        diag_modes: dict[str, Any] = {}
        for mode in POLICY_MODES:
            if mode == "raw":
                continue
            mpol = X1S20ObservablePolicy(
                trainer.model,
                spec,
                model_identity=f"{job['job_id']}:u{update}:{mode}",
                mode=mode,
            )
            mrecs = evaluate_x1_worlds_batched(mpol, kernel, subset_96, batch_size=16)
            diag_modes[mode] = summarize_x1_episodes(mrecs)
        lesions = run_x1_lesion_and_intervention_battery(
            trainer.model,
            spec,
            splits["dev_views"],
            model_identity=f"{job['job_id']}:u{update}",
            u_sweep_worlds=splits["depth_stress_views"],
        )
        depth_recs = evaluate_x1_worlds_batched(pol, kernel, splits["depth_stress_views"], batch_size=16)
        depth_sum = summarize_x1_episodes(depth_recs)
        extra = {
            "diagnostic_modes_subset": diag_modes,
            "lesions": lesions,
            "depth_stress": depth_sum,
        }
        _atomic_write_json(ms_dir / "diagnostics.json", extra)
        _atomic_write_jsonl(ms_dir / "depth_episodes.jsonl", depth_recs)

    return {"update": int(update), "summary": summary, **extra}


def run_x1_job(
    *,
    job_id: str,
    work_dir: Path,
    source_commit: str,
    device: str = "cpu",
    checkpoint_interval: int = 64,
    resume: bool = False,
    deadline_epoch: float | None = None,
    cpu_fixtures: bool = False,
    updates_override: int | None = None,
) -> dict[str, Any]:
    """Execute or resume a single registered V8-X1 job atomically with milestones & eval."""

    t_job0 = time.perf_counter()
    job = find_x1_job(job_id)
    arch = str(job["architecture"])
    obj = str(job["objective"])
    spec = _resolve_x1_spec(arch, cpu_fixtures=cpu_fixtures)
    ds = _load_training_corpus(str(job["dataset"]), objective=obj, cpu_fixtures=cpu_fixtures)
    splits = _load_or_build_splits(work_dir, cpu_fixtures=cpu_fixtures)

    target_updates = int(updates_override) if updates_override is not None else (4 if cpu_fixtures else int(job["updates"]))
    milestones = (0, max(1, target_updates // 2), target_updates) if cpu_fixtures else X1_MILESTONES
    ckpt_root = work_dir / "checkpoints"
    job_dir = work_dir / "jobs" / job_id
    shared_bb_dir = work_dir / "shared_backbones"
    job_dir.mkdir(parents=True, exist_ok=True)

    has_existing = CheckpointStore(ckpt_root, job_id).latest_sha256() is not None
    should_resume = bool(resume and has_existing)

    trainer = create_x1_s20_trainer(
        ds,
        job,
        source_commit=source_commit,
        root=ckpt_root,
        device=device,
        cpu_canary=cpu_fixtures,
        resume=should_resume,
        x1_spec_override=spec,
        updates_override=target_updates,
        batch_size_override=(2 if obj != "standard" else 4) if cpu_fixtures else None,
    )

    milestone_results: dict[str, Any] = {}
    updates_log = job_dir / "updates.jsonl"
    if not should_resume and updates_log.exists():
        updates_log.unlink()

    if trainer.state.global_update == 0 and 0 in milestones:
        milestone_results["u0000"] = _save_and_eval_milestone(
            trainer=trainer,
            spec=spec,
            job=job,
            update=0,
            job_dir=job_dir,
            shared_bb_dir=shared_bb_dir,
            splits=splits,
            is_final=False,
        )

    interrupted = False
    with updates_log.open("a", encoding="utf-8") as log_fh:
        while trainer.state.global_update < target_updates:
            if deadline_epoch is not None and time.time() >= deadline_epoch:
                interrupted = True
                break
            t_u0 = time.perf_counter()
            step_stats = trainer.step()
            step_stats["wall_seconds"] = float(time.perf_counter() - t_u0)
            log_fh.write(json.dumps(step_stats, sort_keys=True) + "\n")
            log_fh.flush()

            cur_u = int(trainer.state.global_update)
            need_save = (cur_u % max(1, int(checkpoint_interval)) == 0) or (cur_u == target_updates) or (cur_u in milestones)
            if need_save and cur_u > trainer.published_update:
                trainer.save(prune_old=True)

            if cur_u in milestones and cur_u > 0:
                ms_key = f"u{cur_u:04d}"
                milestone_results[ms_key] = _save_and_eval_milestone(
                    trainer=trainer,
                    spec=spec,
                    job=job,
                    update=cur_u,
                    job_dir=job_dir,
                    shared_bb_dir=shared_bb_dir,
                    splits=splits,
                    is_final=(cur_u == target_updates),
                )

    if interrupted and trainer.state.global_update > trainer.published_update:
        trainer.save(prune_old=True)

    status = "COMPLETE" if trainer.state.global_update >= target_updates else "PARTIAL"
    receipt = {
        "schema": "elendril-x1-job-receipt-v1",
        "job_id": job_id,
        "session": str(job["session"]),
        "architecture": arch,
        "objective": obj,
        "dataset": str(job["dataset"]),
        "init_seed": int(job["init_seed"]),
        "order_seed": int(job["order_seed"]),
        "status": status,
        "source_commit": source_commit,
        "target_updates": target_updates,
        "completed_updates": int(trainer.state.global_update),
        "latest_checkpoint_sha256": trainer.parent,
        "model_spec_sha256": spec.sha256(),
        "total_parameters": spec.total_parameters(),
        "milestones_recorded": sorted(milestone_results),
        "wall_seconds": float(time.perf_counter() - t_job0),
    }
    _atomic_write_json(job_dir / "job.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ELENDRIL V8-X1 isolated GPU/CPU worker CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare-splits", help="Generate and verify D1-H/D2 splits.")
    p_prep.add_argument("--work-dir", type=Path, required=True)
    p_prep.add_argument("--cpu-fixtures", action="store_true")

    p_can = sub.add_parser("canary", help="Run Stage A bounded arm canary.")
    p_can.add_argument("--arm", required=True)
    p_can.add_argument("--objective", default="standard")
    p_can.add_argument("--device", default="cpu")
    p_can.add_argument("--work-dir", type=Path, required=True)
    p_can.add_argument("--source", required=True)
    p_can.add_argument("--cpu-fixtures", action="store_true")
    p_can.add_argument("--json-out", type=Path, default=None)

    p_prof = sub.add_parser("profile", help="Run Stage B timing & storage profile for a job condition.")
    p_prof.add_argument("--job-id", required=True)
    p_prof.add_argument("--device", default="cpu")
    p_prof.add_argument("--work-dir", type=Path, required=True)
    p_prof.add_argument("--source", required=True)
    p_prof.add_argument("--cpu-fixtures", action="store_true")
    p_prof.add_argument("--json-out", type=Path, default=None)

    p_run = sub.add_parser("run-job", help="Run or resume a single registered X1 job.")
    p_run.add_argument("--job-id", required=True)
    p_run.add_argument("--work-dir", type=Path, required=True)
    p_run.add_argument("--source", required=True)
    p_run.add_argument("--device", default="cpu")
    p_run.add_argument("--checkpoint-interval", type=int, default=64)
    p_run.add_argument("--resume", action="store_true")
    p_run.add_argument("--deadline-epoch", type=float, default=None)
    p_run.add_argument("--cpu-fixtures", action="store_true")
    p_run.add_argument("--updates-override", type=int, default=None)
    p_run.add_argument("--json-out", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "prepare-splits":
        res = prepare_splits(work_dir=args.work_dir, cpu_fixtures=bool(args.cpu_fixtures))
    elif args.command == "canary":
        res = run_arm_canary(
            arm=args.arm,
            objective=args.objective,
            device=args.device,
            work_dir=args.work_dir,
            source_commit=args.source,
            cpu_fixtures=bool(args.cpu_fixtures),
        )
    elif args.command == "profile":
        res = profile_x1_condition(
            job=find_x1_job(args.job_id),
            device=args.device,
            work_dir=args.work_dir,
            source_commit=args.source,
            cpu_fixtures=bool(args.cpu_fixtures),
        )
    else:
        res = run_x1_job(
            job_id=args.job_id,
            work_dir=args.work_dir,
            source_commit=args.source,
            device=args.device,
            checkpoint_interval=int(args.checkpoint_interval),
            resume=bool(args.resume),
            deadline_epoch=args.deadline_epoch,
            cpu_fixtures=bool(args.cpu_fixtures),
            updates_override=args.updates_override,
        )

    if getattr(args, "json_out", None) is not None:
        _atomic_write_json(args.json_out, res)
    print(json.dumps(res, indent=2, sort_keys=True))
    return 0


__all__ = [
    "find_x1_job",
    "prepare_splits",
    "profile_x1_condition",
    "run_arm_canary",
    "run_x1_job",
]

if __name__ == "__main__":
    raise SystemExit(main())
