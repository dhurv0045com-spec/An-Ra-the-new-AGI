"""Isolated single-GPU worker for Hitchhiker profiling, recovery, Stage 0 audit, and training."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any
from uuid import uuid4

import torch

from anra_v5.elendril.codec import canonical
from anra_v5.elendril.cognition import (
    ExperienceEvent,
    Goal,
    MemoryVersion,
    PublicEnvironment,
    Truth,
)
from anra_v5.elendril.hitchhiker_data import (
    HitchhikerSurface,
    build_counterfactual_suite,
    build_health_probes,
    build_structural_holdout_suite,
    build_training_stream,
)
from anra_v5.elendril.hitchhiker_diagnostics import (
    ExternalSlotCanonicalizerPolicy,
    RawHitchhikerPolicy,
    admit_historical_s20_checkpoint,
    score_entity_equivariance,
)
from anra_v5.elendril.hitchhiker_evaluation import (
    ORIGINAL_V6_KINDS,
    evaluate_counterfactual_suite,
    evaluate_health_probes,
    evaluate_hitchhiker,
    evaluate_original_kinds,
)
from anra_v5.elendril.hitchhiker_protocol import (
    AUDITED_V717_COMMIT,
    HISTORICAL_S20_LOWLR_MANIFESTS,
    PROTOCOL_SHA,
    resolve_scale_profile,
    stage_jobs,
    validate_job,
    validate_stage_selection,
)
from anra_v5.elendril.hitchhiker_training import (
    aux_penalty_gradient_probe,
    create_hitchhiker_trainer,
    resolve_exposure_schedule,
    run_hitchhiker_updates,
)
from anra_v5.elendril.representation import UTF8
from anra_v5.elendril.training import MODEL, configure_cpu
from anra_v5.elendril.v6_data import Surface, V6Case, make_dataset, read_dataset
from tools.elendril_action_cloud import compare_payloads
from tools.elendril_hitchhiker_runner import (
    _evaluate_backend_decoders,
    _resolve_historical_source,
    write_stage0_evidence_package,
    verify_and_rescore_stage0_package,
)
from tools.elendril_v6_export import _write_derived, immutable_json
from v5_training.checkpoint import CheckpointStore


REPO_ROOT = Path(__file__).resolve().parents[1]


class WorkerInterrupted(BaseException):
    """Raised on SIGTERM/SIGINT so interrupted checkpoints flush cleanly."""


def hitchhiker_case_to_record(case: V6Case) -> dict[str, Any]:
    """Serialize V6Case (with Surface or HitchhikerSurface) losslessly."""
    surf_cls = "HitchhikerSurface" if isinstance(case.surface, HitchhikerSurface) else "Surface"
    return {
        "surface_type": surf_cls,
        "case_id": case.case_id,
        "world_id": case.world_id,
        "family": case.family,
        "kind": case.kind,
        "goal": asdict(case.goal),
        "initial": [asdict(e) for e in case.environment.initial],
        "channels": {k: [asdict(e) for e in v] for k, v in case.environment.channels.items()},
        "truth": asdict(case.truth),
        "surface": asdict(case.surface),
        "memory": None if case.memory is None else asdict(case.memory),
        "pair_id": case.pair_id,
        "panel": case.panel,
    }


def hitchhiker_case_from_record(row: dict[str, Any]) -> V6Case:
    """Deserialize a V6Case with Surface or HitchhikerSurface deterministically."""
    event = lambda item: ExperienceEvent(**{**item, "parents": tuple(item["parents"])})
    goal = Goal(**{**row["goal"], "slots": tuple(row["goal"]["slots"])})
    mem = row["memory"]
    memory = None if mem is None else MemoryVersion(tuple(event(e) for e in mem["events"]), mem["parent"], mem["capacity"])
    surf = HitchhikerSurface(**row["surface"]) if row["surface_type"] == "HitchhikerSurface" else Surface(**row["surface"])
    return V6Case(
        row["case_id"],
        row["world_id"],
        row["family"],
        row["kind"],
        goal,
        PublicEnvironment(
            tuple(event(e) for e in row["initial"]),
            {k: tuple(event(e) for e in v) for k, v in row["channels"].items()},
        ),
        Truth(**{**row["truth"], "admitted_public": tuple(row["truth"]["admitted_public"])}),
        surf,
        memory,
        row["pair_id"],
        row["panel"],
    )


def serialize_suite_bundle(suite: dict[str, Any]) -> dict[str, Any]:
    """Serialize a counterfactual/structural/probe suite with case-level integrity hash."""
    if "cohorts" in suite:
        cohorts = {k: [hitchhiker_case_to_record(c) for c in v] for k, v in suite["cohorts"].items()}
        payload: dict[str, Any] = {
            "kind": "suite",
            "split": suite["split"],
            "seed": suite.get("seed"),
            "cohorts": cohorts,
            "permutations": suite.get("permutations", {}),
        }
    else:
        probes = {k: [hitchhiker_case_to_record(c) for c in v] for k, v in suite.items()}
        payload = {"kind": "probes", "cohorts": probes}
    payload["payload_sha256"] = sha256(canonical(payload["cohorts"])).hexdigest()
    return payload


def load_suite_bundle(path: Path) -> dict[str, Any]:
    """Load and SHA-verify a serialized suite or health-probe bundle."""
    doc = json.loads(Path(path).read_bytes())
    if sha256(canonical(doc["cohorts"])).hexdigest() != doc.get("payload_sha256"):
        raise ValueError(f"serialized suite failed integrity check: {path}")
    if doc["kind"] == "probes":
        return {k: tuple(hitchhiker_case_from_record(r) for r in v) for k, v in doc["cohorts"].items()}
    return {
        "split": doc["split"],
        "seed": doc.get("seed"),
        "cohorts": {k: tuple(hitchhiker_case_from_record(r) for r in v) for k, v in doc["cohorts"].items()},
        "permutations": doc.get("permutations", {}),
    }


def _assert_cuda_isolated(expected_gpu: int = 1) -> None:
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("worker requires isolated CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_available() or torch.cuda.device_count() != expected_gpu:
        raise RuntimeError(f"worker requires {expected_gpu} isolated CUDA device")


def _checkpoint_fingerprint(store: CheckpointStore, sha_id: str) -> dict[str, Any]:
    state, metadata, spec, hashes = store.load(sha_id, expected_commit=metadata_commit(store, sha_id), device="cpu")
    return {"global_update": state.global_update, "hashes": hashes, "spec": spec.sha256()}


def metadata_commit(store: CheckpointStore, sha_id: str) -> str:
    manifest = json.loads((store.object_dir(sha_id) / "manifest.json").read_text(encoding="utf-8"))
    return manifest["source_commit"]


def mode_recover(args: argparse.Namespace) -> dict[str, Any]:
    """Reload saved checkpoint in a fresh process, step forward, and compare exact state."""
    if not args.cpu_canary:
        _assert_cuda_isolated()
    else:
        configure_cpu()
    device = "cpu" if args.cpu_canary else "cuda:0"
    spec = replace(MODEL, vocabulary_size=UTF8.vocabulary_size, context_length=512) if args.cpu_canary else resolve_scale_profile(args.profile)
    base_data = read_dataset(args.prepared / "train_base.json")
    suite = load_suite_bundle(args.prepared / "validation_suite.json")
    job = json.loads(Path(args.job_file).read_text(encoding="utf-8"))
    trainer, _ = create_hitchhiker_trainer(
        job,
        source_commit=args.source,
        root=args.checkpoints,
        base_data=base_data,
        heldout_suites=(suite,),
        resume=True,
        device=device,
        cpu_canary=args.cpu_canary,
        spec_override=spec if args.cpu_canary else None,
        updates_override=args.to_update if args.cpu_canary else None,
        batch_override=4 if args.cpu_canary else None,
    )
    if trainer.state.global_update != args.from_update:
        raise RuntimeError(f"recover expected resumed update {args.from_update}, got {trainer.state.global_update}")
    step_logs = []
    for _ in range(args.from_update, args.to_update):
        step_logs.append(trainer.step())
    ref_sha = args.ref_sha
    resumed_payload = {
        "final_update": trainer.state.global_update,
        "step_losses": [round(float(x["loss"]), 6) for x in step_logs],
        "param_hashes": {
            k: sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for k, v in trainer.model.state_dict().items()
        },
    }
    if ref_sha:
        ref_state = json.loads(Path(args.ref_file).read_text(encoding="utf-8"))
        if "model.bin" in ref_state:
            compare_payloads(ref_state, resumed_payload, exact=True)
        elif resumed_payload != ref_state:
            raise ValueError(
                f"recovery mismatch vs ref checkpoint {ref_sha}: "
                f"update={resumed_payload['final_update']} vs {ref_state.get('final_update')}"
            )
    return {"status": "PASS", "recovery": resumed_payload}


def _build_canary_spec() -> tuple[Any, dict[str, Any]]:
    tiny = replace(MODEL, vocabulary_size=UTF8.vocabulary_size, context_length=512)
    return tiny, resolve_scale_profile("S9", {"spec": tiny, "parameters": tiny.parameter_receipt().total, "model_sha256": tiny.sha256()})


def _stratified_orig_sample(cases: tuple[Any, ...], per_kind: int = 2) -> tuple[tuple[Any, ...], bool]:
    """Select up to `per_kind` cases per original V6 kind for realistic original-kind profiling."""
    picked: list[Any] = []
    seen_ids: set[str] = set()
    by_kind: dict[str, list[Any]] = {k: [] for k in ORIGINAL_V6_KINDS}
    for case in cases:
        if case.kind in by_kind:
            by_kind[case.kind].append(case)
    all_kinds_present = all(bool(by_kind[k]) for k in ORIGINAL_V6_KINDS)
    for kind in ORIGINAL_V6_KINDS:
        for case in by_kind[kind][:per_kind]:
            if case.case_id not in seen_ids:
                picked.append(case)
                seen_ids.add(case.case_id)
    if not picked:
        picked = list(cases[: min(8, len(cases))])
    return tuple(picked), all_kinds_present


def _stratified_equivariance_sample(cases: tuple[Any, ...], max_pairs: int = 2) -> tuple[Any, ...]:
    """Select complete permutation pairs for score_entity_equivariance profiling."""
    grouped: dict[str, list[Any]] = {}
    for case in cases:
        grouped.setdefault(case.pair_id, []).append(case)
    sample: list[Any] = []
    for pair in grouped.values():
        if len(pair) == 2:
            sample.extend(pair)
            if len(sample) >= max_pairs * 2:
                break
    return tuple(sample)


def _measure_stream_probe(
    base_data: Any,
    *,
    heldout_suites: tuple[Any, ...],
    batch_size: int,
) -> dict[str, Any]:
    """Measure build_training_stream fixed heldout overhead vs per-update generation cost."""
    t2_0 = time.perf_counter()
    build_training_stream(
        "AB",
        base_data=base_data,
        total_updates=2,
        batch_size=batch_size,
        data_seed=606,
        order_seed=7,
        relabel_seed=1009,
        heldout_suites=heldout_suites,
    )
    t2 = time.perf_counter() - t2_0

    t4_0 = time.perf_counter()
    build_training_stream(
        "AB",
        base_data=base_data,
        total_updates=4,
        batch_size=batch_size,
        data_seed=606,
        order_seed=7,
        relabel_seed=1009,
        heldout_suites=heldout_suites,
    )
    t4 = time.perf_counter() - t4_0

    per_update_s = max(0.0, (t4 - t2) / 2.0)
    base_overhead_s = max(0.0, t2 - 2.0 * per_update_s)
    return {
        "probe_2_updates_s": t2,
        "probe_4_updates_s": t4,
        "base_stream_overhead_s": base_overhead_s,
        "ms_per_materialized_update": per_update_s * 1000.0,
        "batch_size": batch_size,
    }


def mode_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Measure isolated GPU latency, peak memory, checkpoint IO, stratified eval, and recover equivalence."""
    if not args.cpu_canary:
        _assert_cuda_isolated()
    else:
        configure_cpu()
    device = "cpu" if args.cpu_canary else "cuda:0"
    device_name = "cpu" if args.cpu_canary else torch.cuda.get_device_name(0)
    base_data = read_dataset(args.prepared / "train_base.json")
    suite = load_suite_bundle(args.prepared / "validation_suite.json")
    val_orig = read_dataset(args.prepared / "validation_orig.json") if (args.prepared / "validation_orig.json").is_file() else None
    val_probes = load_suite_bundle(args.prepared / "validation_probes.json") if (args.prepared / "validation_probes.json").is_file() else None
    struct_suite = load_suite_bundle(args.prepared / "structural_suite.json") if (args.prepared / "structural_suite.json").is_file() else None
    conf_suite = load_suite_bundle(args.prepared / "confirmation_suite.json") if (args.prepared / "confirmation_suite.json").is_file() else None
    heldout = tuple(x for x in (suite, conf_suite, struct_suite, val_probes, val_orig) if x is not None)
    stream_probe = _measure_stream_probe(
        base_data,
        heldout_suites=heldout,
        batch_size=4 if args.cpu_canary else 16,
    )
    s3_job = [j for j in stage_jobs("stage1_s3_screen") if j["arm"] == "AB" and j["seed"] == 7][0]

    profiles_measured: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for label, scale_name, updates in (("S3", "S3", 8), ("S8", "S8", 4), ("S20", "S20", 4)):
            spec_override = _build_canary_spec()[1] if args.cpu_canary else None
            job_spec = {**s3_job, "profile": scale_name} if scale_name == "S3" else {
                **[j for j in stage_jobs("stage3_scale_confirmation", winning_arm="AB") if j["profile"] == scale_name and j["arm"] == "AB" and j["seed"] == 7][0]
            }
            if args.cpu_canary:
                job_spec = s3_job
            ckpt_dir = tmp / f"ckpt_{label}"
            selection = {"screening_status": "PASS_SCREENING", "split": "validation", "winning_arm": "AB", "model_profile": scale_name} if job_spec["stage"] == "stage3_scale_confirmation" else None
            trainer, _ = create_hitchhiker_trainer(
                job_spec,
                source_commit=args.source,
                root=ckpt_dir,
                base_data=base_data,
                heldout_suites=heldout,
                device=device,
                cpu_canary=args.cpu_canary,
                spec_override=spec_override,
                updates_override=updates if args.cpu_canary else None,
                batch_override=4 if args.cpu_canary else None,
                winning_arm="AB",
                selection_receipt=selection,
            )
            if not args.cpu_canary:
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            step_logs = []
            ckpt_write_s = 0.0
            ckpt_saves = 0
            mid_sha = None
            for u in range(1, updates + 1):
                res = trainer.step()
                step_logs.append(res)
                if (label == "S3" and u in (4, 8)) or (label != "S3" and u == updates):
                    cw0 = time.perf_counter()
                    sha = trainer.save()
                    ckpt_write_s += time.perf_counter() - cw0
                    ckpt_saves += 1
                    if label == "S3" and u == 4:
                        mid_sha = sha
            wall = time.perf_counter() - t0
            peak_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if not args.cpu_canary else 0.0

            policy = RawHitchhikerPolicy(trainer.backend)
            eval_sample_s = 0.0
            projected_eval_s = 0.0

            # 1. Stratified counterfactual suite sample across ALL cohorts.
            cohort_sample_counts: dict[str, int] = {}
            cohort_full_counts: dict[str, int] = {}
            cohort_sample_ms: dict[str, float] = {}
            cohort_ms_per_case: dict[str, float] = {}
            cohort_projected_s: dict[str, float] = {}
            for cname, cases in suite["cohorts"].items():
                if not cases:
                    continue
                sample_n = min(4, len(cases))
                c_t0 = time.perf_counter()
                evaluate_counterfactual_suite(
                    {
                        "split": suite["split"],
                        "worlds_per_hop": sample_n,
                        "cohorts": {cname: cases[:sample_n]},
                    },
                    policy,
                )
                c_elapsed = time.perf_counter() - c_t0
                eval_sample_s += c_elapsed
                proj_c = (c_elapsed / sample_n) * len(cases)
                projected_eval_s += proj_c
                cohort_sample_counts[cname] = sample_n
                cohort_full_counts[cname] = len(cases)
                cohort_sample_ms[cname] = c_elapsed * 1000.0
                cohort_ms_per_case[cname] = (c_elapsed / sample_n) * 1000.0
                cohort_projected_s[cname] = proj_c

            # 2. validation_orig stratified sample.
            orig_component: dict[str, Any] = {
                "sample_cases": 0,
                "full_cases": len(val_orig.cases) if val_orig is not None else 0,
                "sample_ms": 0.0,
                "ms_per_case": 0.0,
                "projected_s": 0.0,
            }
            if val_orig is not None and val_orig.cases:
                orig_sample, all_kinds_present = _stratified_orig_sample(val_orig.cases)
                o_t0 = time.perf_counter()
                if all_kinds_present:
                    evaluate_original_kinds(orig_sample, policy, split="validation")
                else:
                    evaluate_hitchhiker(orig_sample, policy, split="validation")
                o_elapsed = time.perf_counter() - o_t0
                eval_sample_s += o_elapsed
                o_proj = (o_elapsed / len(orig_sample)) * len(val_orig.cases)
                projected_eval_s += o_proj
                orig_component = {
                    "sample_cases": len(orig_sample),
                    "full_cases": len(val_orig.cases),
                    "sample_ms": o_elapsed * 1000.0,
                    "ms_per_case": (o_elapsed / len(orig_sample)) * 1000.0,
                    "projected_s": o_proj,
                }

            # 3. validation_probes stratified sample across all probe categories.
            probes_component: dict[str, Any] = {
                "sample_cases": 0,
                "full_cases": sum(len(v) for v in val_probes.values()) if val_probes else 0,
                "sample_ms": 0.0,
                "ms_per_case": 0.0,
                "projected_s": 0.0,
            }
            if val_probes:
                sample_probes = {k: plist[: min(4, len(plist))] for k, plist in val_probes.items() if plist}
                sample_probe_n = sum(len(v) for v in sample_probes.values())
                full_probe_n = sum(len(v) for v in val_probes.values())
                if sample_probe_n > 0:
                    p_t0 = time.perf_counter()
                    evaluate_health_probes(sample_probes, policy, split="validation")
                    p_elapsed = time.perf_counter() - p_t0
                    eval_sample_s += p_elapsed
                    p_proj = (p_elapsed / sample_probe_n) * full_probe_n
                    if scale_name == "S3":
                        projected_eval_s += p_proj
                    probes_component = {
                        "sample_cases": sample_probe_n,
                        "full_cases": full_probe_n,
                        "sample_ms": p_elapsed * 1000.0,
                        "ms_per_case": (p_elapsed / sample_probe_n) * 1000.0,
                        "projected_s": p_proj,
                    }

            # 4. structural_suite stratified sample (measured for S8/S20).
            struct_component: dict[str, Any] = {
                "sample_cases": 0,
                "full_cases": sum(len(v) for v in struct_suite["cohorts"].values()) if struct_suite else 0,
                "sample_ms": 0.0,
                "ms_per_case": 0.0,
                "projected_s": 0.0,
            }
            if struct_suite and scale_name in ("S8", "S20"):
                sample_struct_cohorts = {
                    k: v[: min(4, len(v))] for k, v in struct_suite["cohorts"].items() if v
                }
                sample_struct_n = sum(len(v) for v in sample_struct_cohorts.values())
                full_struct_n = sum(len(v) for v in struct_suite["cohorts"].values())
                if sample_struct_n > 0:
                    st_t0 = time.perf_counter()
                    evaluate_counterfactual_suite(
                        {
                            "split": struct_suite["split"],
                            "worlds_per_hop": 4,
                            "cohorts": sample_struct_cohorts,
                        },
                        policy,
                    )
                    st_elapsed = time.perf_counter() - st_t0
                    eval_sample_s += st_elapsed
                    st_proj = (st_elapsed / sample_struct_n) * full_struct_n
                    projected_eval_s += st_proj
                    struct_component = {
                        "sample_cases": sample_struct_n,
                        "full_cases": full_struct_n,
                        "sample_ms": st_elapsed * 1000.0,
                        "ms_per_case": (st_elapsed / sample_struct_n) * 1000.0,
                        "projected_s": st_proj,
                    }

            # 5. score_entity_equivariance sample.
            perm_cases = suite["cohorts"].get("factor-entity-permutation", ())
            eq_component: dict[str, Any] = {
                "sample_cases": 0,
                "full_cases": len(perm_cases),
                "sample_ms": 0.0,
                "ms_per_case": 0.0,
                "projected_s": 0.0,
            }
            if perm_cases:
                eq_sample = _stratified_equivariance_sample(perm_cases, max_pairs=2)
                if eq_sample:
                    eq_t0 = time.perf_counter()
                    score_entity_equivariance(trainer.backend, eq_sample, suite["permutations"])
                    eq_elapsed = time.perf_counter() - eq_t0
                    eval_sample_s += eq_elapsed
                    eq_proj = (eq_elapsed / len(eq_sample)) * len(perm_cases)
                    projected_eval_s += eq_proj
                    eq_component = {
                        "sample_cases": len(eq_sample),
                        "full_cases": len(perm_cases),
                        "sample_ms": eq_elapsed * 1000.0,
                        "ms_per_case": (eq_elapsed / len(eq_sample)) * 1000.0,
                        "projected_s": eq_proj,
                    }

            recovery_verified = False
            if label == "S3":
                ref_payload = {
                    "final_update": trainer.state.global_update,
                    "step_losses": [round(float(x["loss"]), 6) for x in step_logs[4:8]],
                    "param_hashes": {
                        k: sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                        for k, v in trainer.model.state_dict().items()
                    },
                }
                ref_file = tmp / "ref_s3.json"
                job_file = tmp / "job_s3.json"
                ref_file.write_text(json.dumps(ref_payload), encoding="utf-8")
                job_file.write_text(json.dumps(job_spec), encoding="utf-8")
                # Force LATEST back to update-4 checkpoint for fresh-process recovery check.
                (ckpt_dir / job_spec["job_id"] / "LATEST").write_text(mid_sha + "\n", encoding="ascii")
                rec_out = tmp / "recover_receipt.json"
                cmd = [
                    sys.executable, "-m", "tools.elendril_hitchhiker_worker",
                    "--mode", "recover",
                    "--source", args.source,
                    "--prepared", str(args.prepared),
                    "--checkpoints", str(ckpt_dir),
                    "--job-file", str(job_file),
                    "--profile", "S3",
                    "--from-update", "4",
                    "--to-update", "8",
                    "--ref-sha", str(mid_sha),
                    "--ref-file", str(ref_file),
                    "--receipt", str(rec_out),
                ]
                if args.cpu_canary:
                    cmd.append("--cpu-canary")
                subprocess.run(cmd, cwd=REPO_ROOT, check=True, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
                recovery_verified = json.loads(rec_out.read_text(encoding="utf-8")).get("status") == "PASS"

            ckpt_bytes = sum(p.stat().st_size for p in ckpt_dir.rglob("*") if p.is_file())
            profiles_measured[label] = {
                "updates_measured": updates,
                "wall_seconds": wall,
                "ms_per_update": (wall / updates) * 1000.0,
                "peak_cuda_mb": peak_mb,
                "checkpoint_bytes": ckpt_bytes,
                "checkpoint_write_ms": ckpt_write_s * 1000.0,
                "checkpoint_write_per_save_ms": (ckpt_write_s / max(1, ckpt_saves)) * 1000.0,
                "eval_sample_ms": eval_sample_s * 1000.0,
                "projected_full_eval_seconds": projected_eval_s,
                "eval_components": {
                    "suite": {
                        "sample_cases": sum(cohort_sample_counts.values()),
                        "full_cases": sum(cohort_full_counts.values()),
                        "sample_ms": sum(cohort_sample_ms.values()),
                        "projected_s": sum(cohort_projected_s.values()),
                        "cohort_sample_counts": cohort_sample_counts,
                        "cohort_full_counts": cohort_full_counts,
                        "cohort_sample_ms": cohort_sample_ms,
                        "cohort_ms_per_case": cohort_ms_per_case,
                        "cohort_projected_s": cohort_projected_s,
                    },
                    "orig": orig_component,
                    "probes": probes_component,
                    "structural": struct_component,
                    "equivariance": eq_component,
                },
                "stream_probe": stream_probe,
                "recovery_verified": recovery_verified,
            }
            del trainer
            if not args.cpu_canary:
                torch.cuda.empty_cache()

    return {
        "status": "PASS",
        "lane_index": args.lane_index,
        "device_name": device_name,
        "stream_probe": stream_probe,
        "profiles": profiles_measured,
    }


def mode_stage0(args: argparse.Namespace) -> dict[str, Any]:
    """Audit frozen historical S20 low-lr checkpoint for one seed and export verified Stage 0 package."""
    if not args.cpu_canary and args.checkpoint_root is not None:
        _assert_cuda_isolated()
    else:
        configure_cpu()
    device = "cpu" if args.cpu_canary else ("cuda:0" if args.checkpoint_root else "cpu")
    seed = args.seed
    admission = admit_historical_s20_checkpoint(
        args.checkpoint_root,
        seed,
        expected_data_sha=args.expected_data_sha,
        device=device,
    )
    admission_receipt = {k: v for k, v in admission.items() if k != "backend"}
    out_dir = args.output / "analysis" / "stage0" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if admission["status"] != "ADMITTED":
        pending_hist_src = (
            _resolve_historical_source(admission_receipt)
            or HISTORICAL_S20_LOWLR_MANIFESTS[seed]["source_commit"]
        )
        summary = {
            "schema": "elendril-hitchhiker-stage0-seed-v1",
            "status": "CHECKPOINT_AUDIT_PENDING",
            "seed": seed,
            "executing_source_commit": args.source,
            "audited_v717_ancestor": AUDITED_V717_COMMIT,
            "historical_training_source": pending_hist_src,
            "admission_receipt": admission_receipt,
        }
        _write_derived(out_dir / "stage0_summary.json", canonical(summary) + b"\n")
        return summary

    historical_source = _resolve_historical_source(admission_receipt)
    suite = load_suite_bundle(args.prepared / "validation_suite.json")
    probes = load_suite_bundle(args.prepared / "validation_probes.json")
    orig_dev = read_dataset(args.prepared / "validation_orig.json")
    struct_suite = load_suite_bundle(args.prepared / "structural_suite.json")
    diag = _evaluate_backend_decoders(
        admission["backend"],
        suite,
        probes,
        original_dev=orig_dev,
        structural_suite=struct_suite,
        full_260_inspection=True,
    )
    pkg_dir = out_dir / "stage0_package"
    pkg_receipt = write_stage0_evidence_package(
        pkg_dir,
        executing_source_commit=args.source,
        historical_seed=seed,
        admission_receipt=admission_receipt,
        diag=diag,
        suite_case_count=sum(len(v) for v in suite["cohorts"].values()),
        probe_case_count=sum(len(v) for v in probes.values()),
    )
    pkg_verify = verify_and_rescore_stage0_package(
        pkg_dir,
        suite=suite,
        probes=probes,
        original_dev=orig_dev,
        structural_suite=struct_suite,
    )
    summary = {
        "schema": "elendril-hitchhiker-stage0-seed-v1",
        "status": "EXECUTED",
        "seed": seed,
        "executing_source_commit": args.source,
        "audited_v717_ancestor": AUDITED_V717_COMMIT,
        "historical_training_source": historical_source,
        "admission_receipt": admission_receipt,
        "diagnostic_classification": diag["diagnostic_classification"],
        "entity_equivariance": diag["entity_equivariance"],
        "decoders": {
            name: {k: v for k, v in dec.items() if not k.startswith("_")}
            for name, dec in diag["decoders"].items()
        },
        "stage0_package": {**pkg_receipt, "verification": pkg_verify},
    }
    _write_derived(out_dir / "stage0_summary.json", canonical(summary) + b"\n")
    return summary


def _parse_update_ledger(
    ledger_path: Path,
    resumed_from_update: int,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Split persisted update_logs.jsonl into committed (earliest <= resumed_from_update) and uncommitted rows."""
    committed_by_update: dict[int, dict[str, Any]] = {}
    uncommitted_rows: list[dict[str, Any]] = []
    if not ledger_path.is_file():
        return committed_by_update, uncommitted_rows
    for raw_line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        upd = int(row["update"])
        if 1 <= upd <= resumed_from_update:
            if upd not in committed_by_update:
                committed_by_update[upd] = row
        elif upd > resumed_from_update:
            uncommitted_rows.append(row)
    return committed_by_update, uncommitted_rows


def _append_update_ledger(ledger_path: Path, row: dict[str, Any]) -> None:
    line = (json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    with ledger_path.open("ab") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def mode_train(args: argparse.Namespace) -> dict[str, Any]:
    """Execute or resume a single registered Hitchhiker training job and emit complete eval receipts."""
    deadline: float | None = None
    if args.budget_remaining_seconds is not None:
        deadline = time.monotonic() + max(0.0, float(args.budget_remaining_seconds))
    if args.deadline_unix is not None:
        unix_dl = time.monotonic() + (float(args.deadline_unix) - time.time())
        deadline = unix_dl if deadline is None else min(deadline, unix_dl)

    if not args.cpu_canary:
        _assert_cuda_isolated()
    else:
        configure_cpu()
    device = "cpu" if args.cpu_canary else "cuda:0"
    job = json.loads(Path(args.job_file).read_text(encoding="utf-8"))
    job_id = job["job_id"]

    selection = None
    if args.selection_file and Path(args.selection_file).is_file():
        selection = json.loads(Path(args.selection_file).read_text(encoding="utf-8"))
    winning_arm = (
        args.winning_arm
        or (selection.get("winning_arm") if isinstance(selection, dict) else None)
        or ("AB" if job["stage"] != "stage3_scale_confirmation" else job["arm"])
    )
    validate_job(job, job["stage"], winning_arm=winning_arm)
    if not args.cpu_canary and job["stage"] in ("stage2_followups", "stage3_scale_confirmation"):
        validate_stage_selection(job["stage"], selection, winning_arm=winning_arm)

    job_dir = args.output / "analysis" / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_derived(job_dir / "job_spec.json", canonical(job) + b"\n")

    base_data = read_dataset(args.prepared / "train_base.json")
    val_suite = load_suite_bundle(args.prepared / "validation_suite.json")
    val_probes = load_suite_bundle(args.prepared / "validation_probes.json")
    val_orig = read_dataset(args.prepared / "validation_orig.json")
    conf_suite = load_suite_bundle(args.prepared / "confirmation_suite.json")
    struct_suite = load_suite_bundle(args.prepared / "structural_suite.json")

    spec_override = _build_canary_spec()[1] if args.cpu_canary else None
    batch_size = 4 if args.cpu_canary else job["batch_size"]
    total_updates = args.updates_override if args.cpu_canary else None

    # Pre-check deterministic skip if final eval + matching LATEST checkpoint already exist.
    ckpt_root = args.output / "checkpoints"
    latest_file = ckpt_root / job_id / "LATEST"
    eval_file = job_dir / "final_eval.json"
    receipt_file = job_dir / "run_receipt.json"
    ledger_path = job_dir / "update_logs.jsonl"
    resume = latest_file.is_file()

    # Build heldout suites ensuring train/eval disjoint custody on every arm.
    heldout = (val_suite, conf_suite, struct_suite, val_probes, val_orig)

    # If a pre-built serialized stream exists, load it cleanly; otherwise create deterministically.
    stream = None
    if not args.cpu_canary:
        expected_updates = math.ceil(len(base_data.samples) / batch_size) * job["epochs"]
        stream = build_training_stream(
            job["arm"],
            base_data=base_data,
            total_updates=expected_updates,
            batch_size=batch_size,
            data_seed=606,
            order_seed=job["order_seed"],
            relabel_seed=1009,
            heldout_suites=heldout,
        )

    def _on_signal(_signum: int, _frame: Any) -> None:
        raise WorkerInterrupted("received termination signal")

    old_term = signal.signal(signal.SIGTERM, _on_signal)
    trainer = None
    process_run_id = uuid4().hex[:16]
    try:
        trainer, active_stream = create_hitchhiker_trainer(
            job,
            source_commit=args.source,
            root=ckpt_root,
            base_data=base_data,
            stream=stream,
            heldout_suites=heldout,
            resume=resume,
            device=device,
            cpu_canary=args.cpu_canary,
            spec_override=spec_override,
            updates_override=total_updates,
            batch_override=batch_size if args.cpu_canary else None,
            winning_arm=winning_arm,
            selection_receipt=selection,
        )
        if (
            resume
            and eval_file.is_file()
            and receipt_file.is_file()
            and trainer.state.global_update == trainer.recipe.total_updates
        ):
            existing = json.loads(eval_file.read_text(encoding="utf-8"))
            if (
                existing.get("protocol_sha256") == PROTOCOL_SHA
                and existing.get("checkpoint_sha256") == trainer.store.latest_sha256()
            ):
                existing_receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
                return {
                    "status": "PASS",
                    "job_id": job_id,
                    "arm": job["arm"],
                    "stage": job["stage"],
                    "resumed_skipped": True,
                    "updates_executed_this_process": 0,
                    "checkpoint_sha256": existing["checkpoint_sha256"],
                    "run_receipt": existing_receipt,
                }

        resumed_from_update = trainer.state.global_update
        committed_by_update, uncommitted_rows = _parse_update_ledger(ledger_path, resumed_from_update)
        replayed_uncommitted_updates = len(uncommitted_rows)

        aux_probe = None
        if job["arm"] == "C+penalty":
            aux_probe = aux_penalty_gradient_probe(trainer)
            if aux_probe["status"] != "PASS":
                raise RuntimeError("aux_penalty_gradient_probe failed on C+penalty")

        save_interval = max(1, min(100, trainer.recipe.total_updates // 4)) if not args.cpu_canary else 1
        t0 = time.perf_counter()
        this_process_logs: list[dict[str, Any]] = []
        observed_step_s = 0.0
        observed_save_s = 0.0
        interrupted = False
        try:
            for _ in range(resumed_from_update, trainer.recipe.total_updates):
                estimated_step_plus_save_margin = max(
                    observed_step_s, 0.05 if args.cpu_canary else 1.0
                ) + max(observed_save_s, 0.05 if args.cpu_canary else 0.5)
                if deadline is not None and (time.monotonic() + estimated_step_plus_save_margin) >= deadline:
                    raise WorkerInterrupted("session deadline reached")
                step_t0 = time.perf_counter()
                step_log = trainer.step()
                step_wall_seconds = time.perf_counter() - step_t0
                observed_step_s = max(observed_step_s, step_wall_seconds)
                ledger_row = {
                    "process_run_id": process_run_id,
                    "update": int(step_log["update"]),
                    "loss": float(step_log["loss"]),
                    "supervised_tokens": int(step_log["supervised_tokens"]),
                    "input_positions": int(step_log["input_positions"]),
                    "examples": int(step_log["examples"]),
                    "step_wall_seconds": float(step_wall_seconds),
                }
                _append_update_ledger(ledger_path, ledger_row)
                this_process_logs.append(ledger_row)
                if step_log["update"] % save_interval == 0 or step_log["update"] == trainer.recipe.total_updates:
                    save_t0 = time.perf_counter()
                    trainer.save()
                    observed_save_s = max(observed_save_s, time.perf_counter() - save_t0)
                    if trainer.parent:
                        trainer.store.prune(keep={trainer.parent})
                if args.cpu_canary and args.cpu_interrupt_after is not None and step_log["update"] == args.cpu_interrupt_after:
                    raise WorkerInterrupted(f"canary interrupt at update {step_log['update']}")
        except (WorkerInterrupted, KeyboardInterrupt):
            interrupted = True
            if trainer is not None and trainer.state.global_update > trainer.published_update and not trainer.faulted:
                trainer.save()

        if interrupted:
            latest_sha = trainer.store.latest_sha256()
            _write_derived(
                job_dir / "status.json",
                canonical({"status": "INTERRUPTED", "job_id": job_id, "global_update": trainer.state.global_update, "checkpoint_sha256": latest_sha}) + b"\n",
            )
            return {"status": "INTERRUPTED", "job_id": job_id, "global_update": trainer.state.global_update, "checkpoint_sha256": latest_sha}

        if trainer.state.global_update != trainer.recipe.total_updates:
            raise RuntimeError(f"trainer stopped early at {trainer.state.global_update}/{trainer.recipe.total_updates}")
        latest_sha = trainer.store.latest_sha256() or trainer.save()
        if trainer.parent:
            trainer.store.prune(keep={trainer.parent})
        this_process_wall = time.perf_counter() - t0

        all_committed_by_update = dict(committed_by_update)
        for row in this_process_logs:
            upd = int(row["update"])
            if 1 <= upd <= trainer.state.global_update and upd not in all_committed_by_update:
                all_committed_by_update[upd] = row
        committed_logs = [all_committed_by_update[u] for u in sorted(all_committed_by_update)]

        updates_committed = int(trainer.state.global_update)
        updates_executed_this_process = len(this_process_logs)
        cumulative_supervised_tokens = sum(int(r["supervised_tokens"]) for r in committed_logs)
        cumulative_input_positions = sum(int(r["input_positions"]) for r in committed_logs)
        cumulative_examples_processed = sum(int(r["examples"]) for r in committed_logs)
        cumulative_wall_seconds = sum(float(r["step_wall_seconds"]) for r in committed_logs)
        this_process_supervised_tokens = sum(int(r["supervised_tokens"]) for r in this_process_logs)
        this_process_input_positions = sum(int(r["input_positions"]) for r in this_process_logs)
        this_process_examples_processed = sum(int(r["examples"]) for r in this_process_logs)

        policy = (
            ExternalSlotCanonicalizerPolicy(trainer.backend, assignment="first-mention")
            if job["arm"] == "D-external"
            else RawHitchhikerPolicy(trainer.backend)
        )
        orig_eval = evaluate_original_kinds(val_orig, policy, split="validation")
        if job["stage"] in ("stage1_s3_screen", "stage2_followups"):
            primary_eval = evaluate_counterfactual_suite(val_suite, policy)
            equivariance = score_entity_equivariance(
                trainer.backend,
                val_suite["cohorts"]["factor-entity-permutation"],
                val_suite["permutations"],
            )
            probes_eval = evaluate_health_probes(val_probes, policy, split="validation")
            structural_eval = None
        else:
            primary_eval = evaluate_counterfactual_suite(conf_suite, policy)
            equivariance = score_entity_equivariance(
                trainer.backend,
                conf_suite["cohorts"]["factor-entity-permutation"],
                conf_suite["permutations"],
            )
            probes_eval = None
            structural_eval = evaluate_counterfactual_suite(struct_suite, policy)

        run_receipt = {
            "schema": "elendril-hitchhiker-job-run-v1",
            "protocol_sha256": PROTOCOL_SHA,
            "source_commit": args.source,
            "job_id": job_id,
            "arm": job["arm"],
            "stage": job["stage"],
            "seed": job["seed"],
            "profile": job["profile"],
            "attribution_origin": job["attribution_origin"],
            "updates_committed": updates_committed,
            "updates_executed": updates_committed,
            "updates_executed_this_process": updates_executed_this_process,
            "replayed_uncommitted_updates": replayed_uncommitted_updates,
            "global_update": updates_committed,
            "target_updates": trainer.recipe.total_updates,
            "wall_seconds": cumulative_wall_seconds,
            "cumulative_wall_seconds": cumulative_wall_seconds,
            "this_process_wall_seconds": this_process_wall,
            "supervised_tokens": cumulative_supervised_tokens,
            "cumulative_supervised_tokens": cumulative_supervised_tokens,
            "this_process_supervised_tokens": this_process_supervised_tokens,
            "input_positions": cumulative_input_positions,
            "cumulative_input_positions": cumulative_input_positions,
            "this_process_input_positions": this_process_input_positions,
            "examples_processed": cumulative_examples_processed,
            "cumulative_examples_processed": cumulative_examples_processed,
            "this_process_examples_processed": this_process_examples_processed,
            "final_loss": committed_logs[-1]["loss"] if committed_logs else None,
            "checkpoint_sha256": latest_sha,
            "stream_metrics": active_stream["stream_metrics"],
            "aux_penalty_probe": aux_probe,
        }
        eval_bundle = {
            "schema": "elendril-hitchhiker-job-eval-v1",
            "protocol_sha256": PROTOCOL_SHA,
            "source_commit": args.source,
            "prepared_data_sha256": base_data.identity,
            "job_id": job_id,
            "arm": job["arm"],
            "stage": job["stage"],
            "seed": job["seed"],
            "profile": job["profile"],
            "checkpoint_sha256": latest_sha,
            "model_identity": primary_eval.get("model_identity"),
            "attribution_origin": primary_eval["attribution_origin"],
            "decoder_schema": primary_eval["decoder_schema"],
            "split": primary_eval["split"],
            "cohort_metrics": primary_eval["cohort_metrics"],
            "factor_summary": primary_eval["factor_summary"],
            "original_kind_accuracies": orig_eval["original_kind_accuracies"],
            "equivariance_report": equivariance,
            "probes_scores": probes_eval["scores"] if probes_eval else None,
            "structural_metrics": structural_eval["cohort_metrics"] if structural_eval else None,
            "suite_eval": {
                **primary_eval,
                "source_commit": args.source,
                "original_kind_accuracies": orig_eval["original_kind_accuracies"],
            },
            "original_eval": orig_eval,
            "probes_eval": probes_eval,
            "structural_eval": structural_eval,
        }
        _write_derived(receipt_file, canonical(run_receipt) + b"\n")
        _write_derived(eval_file, canonical(eval_bundle) + b"\n")
        _write_derived(job_dir / "status.json", canonical({"status": "PASS", "job_id": job_id, "checkpoint_sha256": latest_sha}) + b"\n")
        return {
            "status": "PASS",
            "job_id": job_id,
            "arm": job["arm"],
            "stage": job["stage"],
            "resumed_skipped": False,
            "updates_executed_this_process": updates_executed_this_process,
            "checkpoint_sha256": latest_sha,
            "run_receipt": run_receipt,
        }
    finally:
        signal.signal(signal.SIGTERM, old_term)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("profile", "recover", "stage0", "train"))
    p.add_argument("--source", required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--checkpoints", type=Path, default=None)
    p.add_argument("--checkpoint-root", type=Path, default=None)
    p.add_argument("--job-file", type=Path, default=None)
    p.add_argument("--selection-file", type=Path, default=None)
    p.add_argument("--winning-arm", default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--expected-data-sha", default=None)
    p.add_argument("--profile", default="S3")
    p.add_argument("--lane-index", type=int, default=0)
    p.add_argument("--from-update", type=int, default=4)
    p.add_argument("--to-update", type=int, default=8)
    p.add_argument("--ref-sha", default=None)
    p.add_argument("--ref-file", type=Path, default=None)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--cpu-canary", action="store_true")
    p.add_argument("--updates-override", type=int, default=None)
    p.add_argument("--cpu-interrupt-after", type=int, default=None)
    p.add_argument("--deadline-unix", type=float, default=None)
    p.add_argument("--budget-remaining-seconds", type=float, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.mode == "profile":
        result = mode_profile(args)
    elif args.mode == "recover":
        result = mode_recover(args)
    elif args.mode == "stage0":
        result = mode_stage0(args)
    else:
        result = mode_train(args)
    _write_derived(args.receipt, canonical(result) + b"\n")
    print(json.dumps({"status": result.get("status"), "mode": args.mode}, sort_keys=True), flush=True)
    return 0 if result.get("status") in ("PASS", "EXECUTED", "CHECKPOINT_AUDIT_PENDING") else 2 if result.get("status") == "INTERRUPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
