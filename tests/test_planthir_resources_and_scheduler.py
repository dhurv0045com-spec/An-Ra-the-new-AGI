"""Stage 3 verification: ResourceMonitor, ResourceQuota cleanup, and IndependentWorkerScheduler isolation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from anra_v5.elendril.v8.data import D1_TRAIN_SEED, build_d1_training_dataset
from anra_v5.planthir import CartridgeArmSpec, CartridgeRunner, ExperimentCartridge
from anra_v5.planthir.adapters.x1_rd2_adapter import (
    X1RD2EnvironmentKernel,
    X1RD2Recipe,
    X1RD2Spec,
    prepare_x1_rd2_training_data,
)
from anra_v5.planthir.causal import _probe_split_keys
from anra_v5.planthir.resources import (
    ResourceMonitor,
    ResourceQuota,
    ResourceQuotaExceededError,
)
from anra_v5.planthir.scheduler import IndependentWorkerScheduler, WorkerJobSpec, validate_worker_batch

COMMIT = "00b2128b34755d332414ffae8eb78dd3f430e04f"


def _tiny_cartridge() -> ExperimentCartridge:
    data, train_worlds = prepare_x1_rd2_training_data(num_groups=4)
    _, forbidden = _probe_split_keys(train_worlds)
    pool = build_d1_training_dataset(seed=D1_TRAIN_SEED + 37, quota_per_family=16).worlds
    disjoint = [w for w in pool if int(w["renderer_template"]) == 0 and not (_probe_split_keys([w])[1] & forbidden)]
    spec = X1RD2Spec(architecture="X1-P", hidden_dim=16, num_layers=1, num_heads=2, pointer_dim=8, max_sequence_length=512)
    recipe = X1RD2Recipe(seed=7, total_updates=2, batch_size=8, micro_batch_size=4, lr=1e-3, objective="pair_eq")
    arm = CartridgeArmSpec(
        arm_id="X1P",
        architecture_family="X1-P",
        is_control=True,
        v8_spec=None,
        recipe=recipe,
        dataset_identity=data.data_identity,
        milestone_updates=(1, 2),
        backend_id="x1_rd2",
        model_spec=spec,
    )
    return ExperimentCartridge(
        campaign_id="RESOURCE-TEST",
        program="X1-RD2",
        source_commit=COMMIT,
        kernel=X1RD2EnvironmentKernel(),
        control_arm=arm,
        treatment_arms=(),
        train_data=data,
        eval_worlds=tuple(disjoint[:4]),
        multihop_worlds=tuple(disjoint[4:6]),
        probe_train_worlds=tuple(disjoint[6:10]),
        train_worlds=train_worlds,
        backend_id="x1_rd2",
    )


def test_cartridge_runner_writes_measured_resource_receipt(tmp_path: Path) -> None:
    work = tmp_path / "measured_run"
    out = CartridgeRunner().run(_tiny_cartridge(), work_root=work, checkpoint_interval=1)
    rpath = work / "RESOURCE_RECEIPT.json"
    assert rpath.is_file()
    receipt = json.loads(rpath.read_text(encoding="utf-8"))
    assert receipt["schema"] == "anra-planthir-resource-receipt-v1"
    assert receipt["status"] == "COMPLETE"
    assert receipt["total_wall_seconds"] > 0.0
    assert receipt["peak_cpu_rss_bytes"] > 0
    assert receipt["cpu_rss_source"] in ("psutil", "win32_get_process_memory_info", "posix_rusage")
    assert receipt["cuda_measured"] is False and receipt["peak_cuda_allocated_bytes"] is None
    assert receipt["checkpoints_bytes"] > 0 and receipt["result_bundle_bytes"] > 0
    phase_names = [p["phase"] for p in receipt["phases"]]
    assert phase_names == [
        "gate1_and_gate2_pre_run",
        "training_and_gate3",
        "evaluation_and_observatory",
        "bundle_publication",
    ]
    assert out["resource_receipt"]["phases"] == receipt["phases"]


def test_resource_quota_enforcement_and_staging_cleanup(tmp_path: Path) -> None:
    work_t = tmp_path / "wall_limit"
    with pytest.raises(ResourceQuotaExceededError) as exc_t:
        CartridgeRunner().run(
            _tiny_cartridge(),
            work_root=work_t,
            resource_quota=ResourceQuota(max_wall_seconds=1e-6),
        )
    assert exc_t.value.quota_name == "max_wall_seconds"
    rec_t = json.loads((work_t / "RESOURCE_RECEIPT.json").read_text(encoding="utf-8"))
    assert rec_t["status"] == "QUOTA_EXCEEDED" and len(rec_t["violations"]) >= 1

    # Simulate out-of-space failure during bundle publication after staging directory creation
    work_d = tmp_path / "disk_limit"
    committed_file = work_d / "result_bundle" / "CURRENT.json"
    committed_file.parent.mkdir(parents=True, exist_ok=True)
    committed_file.write_text('{"status": "COMPLETE"}\n', encoding="utf-8")
    stale_stage = work_d / "result_bundle" / ".staging-gen-0009"
    stale_stage.mkdir()
    (stale_stage / "giant.bin").write_bytes(b"X" * 4096)

    monitor = ResourceMonitor(
        work_root=work_d,
        device="cpu",
        quota=ResourceQuota(min_free_disk_bytes=10_000),
        disk_usage_fn=lambda _p: 500,
    )
    with pytest.raises(ResourceQuotaExceededError) as exc_d:
        with monitor.phase("bundle_publication"):
            pass
    assert exc_d.value.quota_name == "min_free_disk_bytes"
    assert not stale_stage.exists()
    assert committed_file.is_file()


def test_independent_worker_scheduler_parallel_and_partial_failure_isolation(tmp_path: Path) -> None:
    # Validation catches shared work_root, nested work_root, duplicate worker_id, and duplicate CUDA devices
    with pytest.raises(ValueError, match="share work_root"):
        validate_worker_batch(
            [
                WorkerJobSpec("w1", tmp_path / "same", command=(sys.executable, "-c", "print(1)")),
                WorkerJobSpec("w2", tmp_path / "same", command=(sys.executable, "-c", "print(2)")),
            ]
        )
    with pytest.raises(ValueError, match="nested"):
        validate_worker_batch(
            [
                WorkerJobSpec("w1", tmp_path / "root", command=(sys.executable, "-c", "print(1)")),
                WorkerJobSpec("w2", tmp_path / "root" / "sub", command=(sys.executable, "-c", "print(2)")),
            ]
        )
    with pytest.raises(ValueError, match="CUDA device"):
        validate_worker_batch(
            [
                WorkerJobSpec("w1", tmp_path / "r1", device="cuda:0", command=(sys.executable, "-c", "print(1)")),
                WorkerJobSpec("w2", tmp_path / "r2", device="cuda:0", command=(sys.executable, "-c", "print(2)")),
            ]
        )

    # Parallel 2-worker success run
    ok_script = (
        "import json, os; from pathlib import Path; "
        "p = Path(os.environ['OUT_ROOT']); "
        "(p / 'RESULT.json').write_text(json.dumps({'worker': os.environ['PLANTHIR_WORKER_ID']}))"
    )
    w1_root = tmp_path / "par_w1"
    w2_root = tmp_path / "par_w2"
    coord = IndependentWorkerScheduler(coordinator_root=tmp_path / "coord_ok", max_parallel=2)
    receipt_ok = coord.run(
        [
            WorkerJobSpec("w1", w1_root, command=(sys.executable, "-c", ok_script), env={"OUT_ROOT": str(w1_root)}),
            WorkerJobSpec("w2", w2_root, command=(sys.executable, "-c", ok_script), env={"OUT_ROOT": str(w2_root)}),
        ]
    )
    assert receipt_ok["status"] == "COMPLETE"
    assert receipt_ok["succeeded_workers"] == ["w1", "w2"]
    assert not (w1_root / ".planthir_worker.lock").exists()
    assert json.loads((w1_root / "RESULT.json").read_text())["worker"] == "w1"
    assert json.loads((w2_root / "RESULT.json").read_text())["worker"] == "w2"
    assert (tmp_path / "coord_ok" / "COORDINATOR_RECEIPT.json").is_file()

    # Partial failure: one worker crashes cleanly while healthy sibling succeeds untouched
    ok2_root = tmp_path / "iso_ok"
    fail_root = tmp_path / "iso_fail"
    fail_script = "import sys; print('boom', file=sys.stderr); sys.exit(3)"
    coord_partial = IndependentWorkerScheduler(coordinator_root=tmp_path / "coord_partial", max_parallel=2)
    receipt_partial = coord_partial.run(
        [
            WorkerJobSpec("ok_worker", ok2_root, command=(sys.executable, "-c", ok_script), env={"OUT_ROOT": str(ok2_root)}),
            WorkerJobSpec("bad_worker", fail_root, command=(sys.executable, "-c", fail_script)),
        ]
    )
    assert receipt_partial["status"] == "PARTIAL_FAILURE"
    assert receipt_partial["succeeded_workers"] == ["ok_worker"]
    assert receipt_partial["failed_workers"] == ["bad_worker"]
    assert json.loads((ok2_root / "RESULT.json").read_text())["worker"] == "ok_worker"
    bad_entry = [w for w in receipt_partial["workers"] if w["worker_id"] == "bad_worker"][0]
    assert bad_entry["exit_code"] == 3 and "boom" in bad_entry["stderr_tail"]
