"""Isolated V8 GPU/CPU worker subprocess CLI for canary, profile, train, recover, and frozen eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

from anra_v5.elendril.codec import canonical
from anra_v5.elendril.foundation_training import Recipe
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.training import configure_device
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.elendril.v8.contracts import build_v8_model_spec
from anra_v5.elendril.v8.data import (
    build_d1_dev_suite,
    build_d1_training_dataset,
    make_positive_canary_cohort,
)
from anra_v5.elendril.v8.evaluation import (
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
    render_confirmation_public,
)
from anra_v5.elendril.v8.protocol import PHASES, jobs as protocol_jobs
from anra_v5.elendril.v8.training import (
    V8TrainingData,
    V8TypedTrainer,
    create_v8_trainer,
    prepare_v8_training_data,
    teacher_steps_to_byte_samples,
    teacher_steps_to_typed_samples,
)
from dataclasses import asdict, replace


def _find_job(job_id: str) -> dict[str, object]:
    for phase in PHASES:
        for job in protocol_jobs(phase):
            if job["job_id"] == job_id:
                return job
    raise ValueError(f"unknown V8 job_id: {job_id!r}")


def _tiny_cpu_spec(arm: str):
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


def run_canary(
    *,
    arm: str,
    source_commit: str,
    work_dir: Path,
    device: str = "cuda:0",
    cpu_canary: bool = False,
) -> dict[str, object]:
    """Run the positive-control canary check (action/episode accuracy & loss drop gates)."""

    canary = make_positive_canary_cohort(num_worlds=32 if cpu_canary else 32)
    v8_spec = _tiny_cpu_spec(arm) if cpu_canary else build_v8_model_spec(arm)
    steps = canary.steps[:32] if cpu_canary else canary.steps
    typed_samples = teacher_steps_to_typed_samples(steps)
    updates = 6 if cpu_canary else 120
    batch = min(16, len(typed_samples))
    recipe = Recipe(
        seed=7,
        total_updates=updates,
        batch_size=batch,
        micro_batch_size=min(2, batch),
        lr=3e-4 if not cpu_canary else 1e-3,
        schedule="cosine",
        precision="fp32",
    )
    prov = {
        "split": "train",
        "license": "internal-v8-canary",
        "canary_identity": canary.identity,
    }
    if arm == "B0":
        raise ValueError("canary positive-control typed check uses a typed arm (e.g. F or P)")
    configure_device(device)
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
    losses: list[float] = []
    t0 = time.perf_counter()
    for _ in range(updates):
        m = trainer.step()
        losses.append(float(m["loss"]))
    wall = time.perf_counter() - t0
    trainer.save()

    diag = diagnostic_probe(trainer.model, v8_spec, steps, max_probe_steps=32)
    eps = evaluate_worlds_by_mode(trainer.model, v8_spec, canary.worlds[:32], model_identity=f"canary-{arm}", modes=("raw",))
    ep_acc = sum(1 for e in eps["raw"] if e.outcome.correct) / len(eps["raw"])
    action_acc = float(diag["op_accuracy"]) * float(diag.get("val_accuracy_given_op_correct") or 1.0)
    loss_ratio = (losses[-1] / max(losses[0], 1e-12)) if losses else 1.0
    passed = (ep_acc >= 0.90 and action_acc >= 0.95 and loss_ratio <= 0.20) if not cpu_canary else (losses[-1] <= losses[0])

    result = {
        "schema": "elendril-v8-canary-v1",
        "arm": arm,
        "cpu_canary": cpu_canary,
        "updates": updates,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_ratio": loss_ratio,
        "raw_episode_accuracy": ep_acc,
        "raw_action_accuracy": action_acc,
        "wall_seconds": wall,
        "passed": passed,
    }
    (work_dir / "CANARY_RESULT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_profile(
    *,
    job_id: str,
    source_commit: str,
    work_dir: Path,
    device: str = "cuda:0",
    cpu_canary: bool = False,
) -> dict[str, object]:
    """Measure 16+ GPU steps (or bounded CPU canary steps) + save@2 -> fresh resume@4 parity."""

    job = _find_job(job_id)
    v8_spec = _tiny_cpu_spec(str(job["arm"])) if cpu_canary else None
    data = prepare_v8_training_data(
        str(job["dataset"]),
        d1_worlds_per_family=16 if cpu_canary else 512,
        sample_limit_for_cpu=32 if cpu_canary else None,
    )
    prof_updates = 4 if cpu_canary else 20

    # 1. Resume parity test (save@2 -> resume@4 vs continuous 4)
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
        updates_override=4,
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
        updates_override=4,
    )
    t_res_a.step()
    t_res_a.step()
    t_res_a.save()

    t_res_b = create_v8_trainer(
        data,
        job,
        source=source_commit,
        root=res_root,
        resume=True,
        device=device,
        cpu_canary=cpu_canary,
        v8_spec_override=v8_spec,
        updates_override=4,
    )
    t_res_b.step()
    t_res_b.step()
    parity_ok = all(
        torch.equal(t_cont.model.state_dict()[k], t_res_b.model.state_dict()[k])
        for k in t_cont.model.state_dict()
    )
    if not parity_ok:
        raise AssertionError("profile save@2 -> resume@4 failed exact weight parity")

    # 2. Timing profile over prof_updates steps
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
        updates_override=prof_updates,
    )
    t_prof.step()  # warmup step
    t0 = time.perf_counter()
    for _ in range(prof_updates - 1):
        t_prof.step()
    if torch.cuda.is_available() and t_prof.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - t0, 1e-6)
    measured_steps = prof_updates - 1
    peak_mb = (
        float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        if (torch.cuda.is_available() and t_prof.device.type == "cuda")
        else 0.0
    )
    result = {
        "schema": "elendril-v8-profile-v1",
        "job_id": job_id,
        "source_commit": source_commit,
        "measured_updates": measured_steps,
        "seconds_per_update": elapsed / measured_steps,
        "peak_vram_mb": peak_mb,
        "resume_parity_at_4_verified": True,
        "total_training_samples": len(t_prof.samples),
        "estimated_full_updates": 4 * ((len(t_prof.samples) + int(job["batch_size"]) - 1) // int(job["batch_size"])),
    }
    (work_dir / "WORKER_PROFILE.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_job_worker(
    *,
    job_id: str,
    source_commit: str,
    work_dir: Path,
    device: str = "cuda:0",
    resume: bool = False,
    cpu_canary: bool = False,
    updates_override: int | None = None,
    dev_worlds_per_cohort: int | None = None,
    save_every: int = 64,
    confirmation_seed: int | None = None,
) -> dict[str, object]:
    """Train or recover one registered V8 job and run milestone + final evaluations."""

    job = _find_job(job_id)
    arm = str(job["arm"])
    v8_spec = _tiny_cpu_spec(arm) if cpu_canary else build_v8_model_spec(arm)
    data = prepare_v8_training_data(
        str(job["dataset"]),
        d1_worlds_per_family=16 if cpu_canary else 512,
        sample_limit_for_cpu=32 if cpu_canary else None,
    )
    ckpt_root = work_dir / "checkpoints"
    trainer = create_v8_trainer(
        data,
        job,
        source=source_commit,
        root=ckpt_root,
        resume=resume,
        device=device,
        cpu_canary=cpu_canary,
        v8_spec_override=v8_spec if cpu_canary else None,
        updates_override=updates_override,
    )
    total_updates = trainer.recipe.total_updates
    eval_cohorts = build_d1_dev_suite(
        worlds_per_cohort=dev_worlds_per_cohort or (2 if cpu_canary else 128)
    )
    all_eval_worlds = [w for rows in eval_cohorts.values() for w in rows]
    probe_d1 = build_d1_training_dataset(quota_per_family=16 if cpu_canary else 64)

    t0 = time.perf_counter()
    milestones: dict[str, object] = {}
    if trainer.state.global_update == 0:
        milestones["initial"] = diagnostic_probe(trainer.model, v8_spec, probe_d1.steps, max_probe_steps=32)

    epoch_len = max(total_updates // 4, 1)
    milestone_updates = {
        epoch_len: "epoch-1",
        2 * epoch_len: "epoch-2",
        total_updates: "epoch-4",
    }

    last_loss = None
    while trainer.state.global_update < total_updates:
        metrics = trainer.step()
        last_loss = float(metrics["loss"])
        u = int(trainer.state.global_update)
        if u % save_every == 0 or u == total_updates or u in milestone_updates:
            if u > trainer.published_update:
                trainer.save()
        if u in milestone_updates and u < total_updates:
            # Intermediate 32-world cohort check
            sub_worlds = [w for rows in eval_cohorts.values() for w in rows[: (5 * 2 if cpu_canary else 5 * 32)]]
            m_eps = evaluate_worlds_by_mode(trainer.model, v8_spec, sub_worlds, model_identity=job_id, modes=("raw",))
            milestones[milestone_updates[u]] = {
                "update": u,
                "loss": last_loss,
                "raw_summary": summarize_suite_episodes(m_eps, sub_worlds)["raw"],
                "probe": diagnostic_probe(trainer.model, v8_spec, probe_d1.steps, max_probe_steps=32),
            }

    train_wall = time.perf_counter() - t0
    final_eps = evaluate_worlds_by_mode(trainer.model, v8_spec, all_eval_worlds, model_identity=job_id)
    final_summary = summarize_suite_episodes(final_eps, all_eval_worlds)
    final_probe = diagnostic_probe(trainer.model, v8_spec, probe_d1.steps, max_probe_steps=32)
    pointer_interv = (
        run_pointer_interventions(trainer.model, v8_spec, probe_d1.steps[:64])
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
    with ep_file.open("w", encoding="utf-8") as fh:
        for mode, eps in final_eps.items():
            for ep in eps:
                fh.write(
                    canonical(
                        {
                            "world_id": ep.world_id,
                            "base_world_id": ep.base_world_id,
                            "cohort": ep.cohort,
                            "family": ep.family,
                            "form": ep.form,
                            "mode": mode,
                            "correct": ep.outcome.correct,
                            "terminal_status": ep.outcome.terminal_status,
                            "first_observable_divergence": ep.first_observable_divergence,
                            "model_calls": ep.model_calls,
                            "evaluated_positions": ep.evaluated_positions,
                        }
                    )
                    + "\n"
                )

    raw_sum = final_summary["raw"]
    result = {
        "schema": "elendril-v8-job-result-v1",
        "job_id": job_id,
        "phase": job["phase"],
        "arm": arm,
        "dataset": job["dataset"],
        "source_commit": source_commit,
        "completed_updates": trainer.state.global_update,
        "target_updates": total_updates,
        "checkpoint_sha256": trainer.parent,
        "total_parameters": v8_spec.total_parameters(),
        "wall_seconds": train_wall,
        "final_loss": last_loss,
        "raw_overall_accuracy": raw_sum["exact_episode_accuracy"],
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
        "episodes_file": str(ep_file),
    }

    if confirmation_seed is not None:
        conf_cohorts = build_confirmation_suite(
            confirmation_seed=confirmation_seed,
            worlds_per_cohort=2 if cpu_canary else 256,
            excluded_structure_hashes=probe_d1.structural_hashes,
        )
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
    (out_dir / f"{job_id}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ELENDRIL V8 isolated GPU/CPU worker")
    parser.add_argument("mode", choices=("canary", "profile", "train", "recover", "frozen"))
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--arm", default="F")
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
        if not args.job_id:
            raise ValueError("--job-id required for profile")
        res = run_profile(
            job_id=args.job_id,
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
    print(json.dumps({"status": "OK", "mode": args.mode}, indent=2))
    _ = res
    return 0


__all__ = ["run_canary", "run_job_worker", "run_profile"]

if __name__ == "__main__":
    raise SystemExit(main())
