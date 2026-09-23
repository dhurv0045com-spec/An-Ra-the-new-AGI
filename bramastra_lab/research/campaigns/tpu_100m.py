"""Eight-core, zero-update Kaggle preflight for the 100M BRAMASTRA profile.

This command is deliberately a preflight, not a training campaign. It checks
the live PJRT mesh, initializes and synchronizes the exact model, then runs a
real B-arm token/world/action/value/pair forward-and-backward window built from
eight distinct exact-training-split examples. It never finalizes the optimizer.
Passing establishes that this backward preflight completed; it does not
qualify optimizer-state memory, checkpoint/resume, sustained throughput, or a
multi-hour training run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence


PROFILE = "tpu_100m"
EXPECTED_REPLICAS = 8
EXPECTED_PARAMETER_COUNT = 100_334_720
PREFLIGHT_SCHEMA = "bramastra-tpu-100m-preflight/v1"


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")


def _write_exclusive_json(path: str | os.PathLike[str],
                          value: Mapping[str, Any]) -> None:
    """Atomically publish one JSON object without replacing prior evidence."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f"{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with open(temporary, "xb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        # Same-directory hard-link gives exclusive publication. Unlike
        # os.replace, a repeated cell/worker cannot overwrite an earlier run.
        os.link(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _data_identity(data_dir: str | os.PathLike[str]) -> str:
    """Hash relative paths and file contents for the exact prepared bundle."""
    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"training bundle directory is missing: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError("training bundle contains no files")
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        file_hash = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_hash.update(chunk)
        digest.update(file_hash.digest())
        digest.update(b"\0")
    return digest.hexdigest()


def _compile_example(row: dict[str, Any], partner: dict[str, Any], *,
                     max_seq: int) -> dict[str, Any]:
    """Validate a real B-arm example entirely on CPU before XLA launch."""
    from bramastra_lab.research.campaigns.phases.e1 import ARM_ENABLED, ARM_WEIGHTS
    from bramastra_lab.research.campaigns.phases.compiler import (
        build_batch_for_trajectory,
        build_pair_rows,
        compile_channels_for_row,
    )

    if row.get("pool") != "training" or partner.get("pool") != "training":
        raise ValueError("TPU preflight examples must come from exact training pool")
    if str(row.get("answer")) == str(partner.get("answer")):
        raise ValueError("counterfactual pair requires different training answers")
    batch = build_batch_for_trajectory(row, max_seq=max_seq)
    compiled = compile_channels_for_row(
        row, batch, arm_weights=dict(ARM_WEIGHTS["B"]),
        arm_enabled=ARM_ENABLED["B"], max_seq=max_seq)
    pair_rows = build_pair_rows(row, partner, max_seq=max_seq)
    # A distinct-answer pair contains two ordered comparisons: each goal is
    # scored with its own answer and with the swapped answer. The TPU trainer
    # validates this denominator against the actual local comparison rows.
    compiled["pair"]["denominator"] = len(pair_rows[0])
    required = {term for term in ARM_ENABLED["B"]
                if float(ARM_WEIGHTS["B"].get(term, 0.0)) > 0}
    # Token supervision lives in the primary Batch object, not in
    # compile_channels_for_row's non-token sidecar dictionary.
    missing = required - {"token"} - set(compiled)
    if missing:
        raise ValueError(f"B-arm sample has no compiled objectives: {sorted(missing)}")
    return {"batch": batch, "compiled": compiled, "pair_rows": pair_rows,
            "target_count": int(batch.target_count),
            "objective_counts": {
                "token": int(batch.target_count),
                **{term: int(compiled[term].get("denominator", 1))
                   for term in required if term != "token"}}}


def _build_sample_plan(data_dir: str, *, seed: int,
                       expected_replicas: int = EXPECTED_REPLICAS
                       ) -> dict[str, Any]:
    """Choose valid, distinct learner rows before starting any TPU workers."""
    from bramastra_lab.research.campaigns.phases.compiler import (
        load_training_trajectories,
    )
    from bramastra_lab.research.config import BuildConfig

    config = BuildConfig.from_dict({"model": {"profile": PROFILE}})
    rows = load_training_trajectories(data_dir, seed=seed)
    if (not isinstance(expected_replicas, int)
            or isinstance(expected_replicas, bool) or expected_replicas < 1):
        raise ValueError("expected_replicas must be a positive integer")
    if len(rows) < expected_replicas:
        raise ValueError(
            f"need at least {expected_replicas} exact-training rows; found {len(rows)}")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    selected: list[dict[str, Any]] = []
    used_rows: set[int] = set()
    failures: list[str] = []
    for row_index in order:
        if row_index in used_rows:
            continue
        row = rows[row_index]
        partner_index = None
        for candidate_index in order:
            if (candidate_index != row_index
                    and str(rows[candidate_index].get("answer"))
                    != str(row.get("answer"))):
                partner_index = candidate_index
                break
        if partner_index is None:
            raise ValueError("training split has no distinct-answer pair")
        try:
            sample = _compile_example(row, rows[partner_index],
                                      max_seq=config.model.max_seq)
        except Exception as exc:
            failures.append(f"row {row_index}: {type(exc).__name__}: {exc}")
            continue
        selected.append({
            "rank": len(selected),
            "row_index": row_index,
            "pair_index": partner_index,
            "row_identity": hashlib.sha256(
                str(row["mechanism_id"]).encode()).hexdigest(),
            "pair_identity": hashlib.sha256(
                str(rows[partner_index]["mechanism_id"]).encode()).hexdigest(),
            "target_count": sample["target_count"],
            "objective_counts": sample["objective_counts"],
        })
        used_rows.add(row_index)
        if len(selected) == expected_replicas:
            break
    if len(selected) != expected_replicas:
        raise ValueError(
            f"could compile only {len(selected)}/{expected_replicas} valid B-arm "
            f"training examples; failures={failures[:5]}")
    return {"schema": "bramastra-tpu-100m-sample-plan/v1",
            "profile": PROFILE, "seed": seed,
            "expected_replicas": expected_replicas,
            "config_identity": config.identity(),
            "max_seq": config.model.max_seq,
            "training_rows": len(rows), "workers": selected}


def _memory_info(xm: Any, device: Any) -> dict[str, int] | None:
    probe = getattr(xm, "get_memory_info", None)
    if not callable(probe):
        return None
    try:
        raw = probe(device)
        parsed: dict[str, int] = {}
        for key, value in raw.items():
            if isinstance(value, bool):
                continue
            try:
                parsed[str(key)] = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
        return parsed
    except Exception:
        return None


def _xla_worker(local_ordinal: int, data_dir: str, report_dir: str,
                sample_plan_path: str) -> None:
    """One PJRT worker: synchronized initialization + real no-step backward."""
    import torch
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    from bramastra_lab.research.campaigns.phases.ops import ProductionOps
    from bramastra_lab.research.campaigns.phases.compiler import (
        load_training_trajectories,
    )
    from bramastra_lab.research.runtime.tpu import (
        broadcast_replica_parameters,
        verify_replica_initialization,
        write_replica_initialization_receipt,
    )

    rank = int(xr.global_ordinal())
    world_size = int(xr.world_size())
    if rank != int(local_ordinal) or world_size != EXPECTED_REPLICAS:
        raise RuntimeError(
            f"PJRT worker mismatch: local={local_ordinal}, rank={rank}, "
            f"world={world_size}; expected eight TPU replicas")
    device = xm.xla_device()
    sample_plan = _read_json(sample_plan_path)
    if sample_plan.get("schema") != "bramastra-tpu-100m-sample-plan/v1":
        raise RuntimeError("invalid preflight sample plan")

    ops = ProductionOps(precision="bf16_autocast")
    handle = ops.init_model(seed=int(sample_plan["seed"]), profile=PROFILE,
                            device=str(device))
    config, model, trainer = (handle["config"], handle["model"],
                              handle["trainer"])
    parameter_count = sum(int(parameter.numel())
                          for parameter in model.parameters())
    if (config.model.profile != PROFILE
            or parameter_count != EXPECTED_PARAMETER_COUNT
            or config.identity() != sample_plan.get("config_identity")
            or trainer.precision != "bf16_autocast"
            or trainer.replica_backend is None
            or trainer.replica_backend.world_size != EXPECTED_REPLICAS):
        raise RuntimeError("100M profile/config/precision/topology contract mismatch")

    broadcast_replica_parameters(model)
    write_replica_initialization_receipt(
        report_dir, rank=rank, world_size=world_size,
        config_identity=config.identity(), model=model)
    xm.rendezvous("bramastra-tpu-100m-init-receipts-v1")
    if rank == 0:
        try:
            verified = verify_replica_initialization(
                report_dir, expected_replicas=EXPECTED_REPLICAS)
            _write_exclusive_json(Path(report_dir) / "initialization.json",
                                  {"status": "REPLICAS_IDENTICAL", **verified})
        except Exception as exc:
            _write_exclusive_json(Path(report_dir) / "initialization.json",
                                  {"status": "FAILED",
                                   "error": f"{type(exc).__name__}: {exc}"})
    xm.rendezvous("bramastra-tpu-100m-init-verified-v1")
    initialization = _read_json(Path(report_dir) / "initialization.json")
    if initialization.get("status") != "REPLICAS_IDENTICAL":
        raise RuntimeError(f"replica initialization rejected: {initialization}")

    workers = sample_plan.get("workers")
    if not isinstance(workers, list) or len(workers) != EXPECTED_REPLICAS:
        raise RuntimeError("sample plan does not carry one row per TPU replica")
    sample = workers[rank]
    if sample.get("rank") != rank:
        raise RuntimeError(f"sample plan rank mismatch at {rank}")
    trajectories = load_training_trajectories(data_dir,
                                              seed=int(sample_plan["seed"]))
    row = trajectories[int(sample["row_index"])]
    partner = trajectories[int(sample["pair_index"])]
    row_hash = hashlib.sha256(str(row.get("mechanism_id")).encode()).hexdigest()
    pair_hash = hashlib.sha256(str(partner.get("mechanism_id")).encode()).hexdigest()
    if (row_hash != sample.get("row_identity")
            or pair_hash != sample.get("pair_identity")
            or row.get("pool") != "training"
            or partner.get("pool") != "training"):
        raise RuntimeError("worker training sample identity/split mismatch")

    prepared = _compile_example(row, partner, max_seq=config.model.max_seq)
    batch, compiled, pair_rows = (prepared["batch"], prepared["compiled"],
                                  prepared["pair_rows"])
    if int(batch.target_count) != int(sample.get("target_count", -1)):
        raise RuntimeError("worker compiled target count changed from preflight plan")
    started = time.perf_counter()
    with trainer._autocast_context():
        window, extra = ops.construct_objectives(
            handle=handle, batch=batch, compiled=compiled, arm="B")
    accumulated = trainer.accumulate_full_window(
        batch, window_builder=lambda _targets: window,
        extra_terms_fn=lambda: extra, pair_rows=pair_rows)
    xm.mark_step()
    # XLA execution is asynchronous. Wait before stopping the timer so the
    # report measures device completion instead of graph submission.
    import torch_xla
    torch_xla.sync(wait=True)
    elapsed = time.perf_counter() - started
    gradients = [parameter.grad for parameter in model.parameters()
                 if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("B-arm backward produced no gradients")
    finite = bool(torch.stack([torch.isfinite(gradient).all()
                               for gradient in gradients]).all().detach().cpu().item())
    if not finite:
        raise RuntimeError("B-arm preflight produced nonfinite gradients")
    memory = _memory_info(xm, device)
    updates = int(trainer.counters.optimizer_updates)
    attempts = int(trainer.attempted_updates)
    if updates != 0 or attempts != 0:
        raise RuntimeError(
            f"preflight crossed an optimizer boundary: updates={updates}, "
            f"attempts={attempts}")
    report = {
        "schema": "bramastra-tpu-100m-worker/v1",
        "rank": rank, "world_size": world_size,
        "status": "BACKWARD_PASS",
        "profile": PROFILE,
        "config_identity": config.identity(),
        "initial_state_sha256": initialization["state_sha256"],
        "parameter_count": parameter_count,
        "precision": trainer.precision,
        "data_row_identity": row_hash,
        "pair_row_identity": pair_hash,
        "target_count": int(batch.target_count),
        "input_tokens": int(batch.padding_mask.sum().item()),
        "objective_counts": {term: int(window.denominator(term))
                              for term in sorted(window.enabled_terms)},
        "gradient_tensors": len(gradients),
        "gradients_finite": finite,
        "backward_seconds": elapsed,
        "memory_info_bytes": memory,
        "optimizer_updates": updates,
        "optimizer_attempts": attempts,
        "training_started": False,
        "answer_loss_sum": float(accumulated["answer_loss_sum"]),
    }
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer._clear_pending()
    _write_exclusive_json(
        Path(report_dir) / f"worker-{rank:03d}.json", report)
    xm.rendezvous("bramastra-tpu-100m-workers-finished-v1")
    if rank == 0:
        try:
            worker_reports = [
                _read_json(Path(report_dir) / f"worker-{ordinal:03d}.json")
                for ordinal in range(EXPECTED_REPLICAS)]
            if any(item.get("status") != "BACKWARD_PASS"
                   or item.get("optimizer_updates") != 0
                   or item.get("optimizer_attempts") != 0
                   or item.get("gradients_finite") is not True
                   for item in worker_reports):
                raise RuntimeError("one or more TPU ranks failed backward checks")
            _write_exclusive_json(Path(report_dir) / "preflight.json", {
                "schema": PREFLIGHT_SCHEMA,
                "status": "BACKWARD_PREFLIGHT_PASS",
                "profile": PROFILE,
                "parameter_count": EXPECTED_PARAMETER_COUNT,
                "replicas": EXPECTED_REPLICAS,
                "config_identity": config.identity(),
                "initial_state_sha256": initialization["state_sha256"],
                "workers": worker_reports,
                "optimizer_updates": 0,
                "optimizer_attempts": 0,
                "training_started": False,
                "qualification": "backward-only; optimizer-state fit, committed update, "
                                 "resume, sustained throughput and full campaign remain unqualified",
            })
        except Exception as exc:
            _write_exclusive_json(Path(report_dir) / "preflight.json", {
                "schema": PREFLIGHT_SCHEMA, "status": "FAILED",
                "training_started": False,
                "error": f"{type(exc).__name__}: {exc}"})
    xm.rendezvous("bramastra-tpu-100m-preflight-published-v1")
    final = _read_json(Path(report_dir) / "preflight.json")
    if final.get("status") != "BACKWARD_PREFLIGHT_PASS":
        raise RuntimeError(f"100M TPU backward preflight failed: {final}")


def run_tpu_100m_preflight(*, data_dir: str, report_dir: str,
                           seed: int = 1701) -> dict[str, Any]:
    """Run the eight-replica no-update preflight on an already selected TPU."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    root = Path(report_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"preflight report directory is not empty; choose a fresh RUN_ID: {root}")
    root.mkdir(parents=True, exist_ok=True)
    # Runtime is checked before data work or model allocation. The TPU runtime
    # helper remains safe to import on CPU-only developer machines.
    from bramastra_lab.research.runtime.tpu import inspect_tpu_runtime

    runtime = inspect_tpu_runtime(expected_replicas=EXPECTED_REPLICAS)
    _write_exclusive_json(root / "runtime.json", runtime)
    if runtime.get("status") != "TPU_RUNTIME_READY_FOR_PREFLIGHT":
        raise RuntimeError(f"TPU runtime preflight refused: {runtime}")

    from bramastra_lab.research.data.k8_bundle import validate_bundle
    validation = validate_bundle(data_dir, min_confirmation=1)
    if not validation.get("valid"):
        raise ValueError(
            f"K8 training bundle is invalid: {validation.get('issues', [])[:5]}")
    sample_plan = _build_sample_plan(data_dir, seed=seed)
    from bramastra_lab.research.config import BuildConfig

    config = BuildConfig.from_dict({"model": {"profile": PROFILE}})
    if config.parameter_count() != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("registered tpu_100m parameter count changed")
    sample_plan["data_identity"] = _data_identity(data_dir)
    sample_path = root / "sample-plan.json"
    _write_exclusive_json(sample_path, sample_plan)
    _write_exclusive_json(root / "run.json", {
        "schema": PREFLIGHT_SCHEMA,
        "status": "LAUNCHING_ZERO_UPDATE_PREFLIGHT",
        "profile": PROFILE,
        "config_identity": config.identity(),
        "data_identity": sample_plan["data_identity"],
        "source_revision": _source_revision(),
        "seed": seed,
        "expected_replicas": EXPECTED_REPLICAS,
        "training_started": False,
    })
    from bramastra_lab.research.runtime.tpu import (
        launch_tpu_workers, verify_replica_initialization,
    )

    launch_tpu_workers(
        _xla_worker,
        args=(str(Path(data_dir).resolve()), str(root), str(sample_path)))
    initialization = verify_replica_initialization(
        str(root), expected_replicas=EXPECTED_REPLICAS)
    report = _read_json(root / "preflight.json")
    if (report.get("schema") != PREFLIGHT_SCHEMA
            or report.get("status") != "BACKWARD_PREFLIGHT_PASS"
            or report.get("parameter_count") != EXPECTED_PARAMETER_COUNT
            or report.get("replicas") != EXPECTED_REPLICAS
            or report.get("initial_state_sha256") != initialization["state_sha256"]
            or report.get("optimizer_updates") != 0
            or report.get("optimizer_attempts") != 0
            or report.get("training_started") is not False):
        raise RuntimeError(f"TPU preflight report failed validation: {report}")
    report["runtime"] = runtime
    report["data_identity"] = sample_plan["data_identity"]
    report["source_revision"] = _source_revision()
    report["initialization"] = initialization
    return report


def _source_revision() -> str:
    import subprocess

    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except Exception:
        return "unavailable"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Zero-update eight-core TPU preflight for BRAMASTRA 100M")
    parser.add_argument("preflight", choices=("preflight",))
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args(argv)
    try:
        report = run_tpu_100m_preflight(
            data_dir=args.data, report_dir=args.run_dir, seed=args.seed)
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        failure = {"status": "PREFLIGHT_FAILED",
                   "error": f"{type(exc).__name__}: {exc}",
                   "training_started": False}
        print(json.dumps(failure, sort_keys=True, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
