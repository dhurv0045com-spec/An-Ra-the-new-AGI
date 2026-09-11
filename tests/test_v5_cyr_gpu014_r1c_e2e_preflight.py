"""CYR-GPU-014 / R1C end-to-end engineering preflight (CPU, minimal updates).

Traverses the ACTUAL campaign executable path — the compatibility wrapper's
patched module, real ARK-002B manifest, real model build, real optimizer
construction through the patched boundary, one FULL arm update, one MASKED
arm update, one OFFSET arm update, diagnostics, checkpoint save/reload,
result schema, and partial packaging — with frozen constants monkeypatched
down to tiny engineering fixtures inside this process only. The frozen file
on disk is never modified. ENGINEERING evidence only; not scientific
evidence; no campaign-scale training runs locally.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from anra_v5 import cyr_gpu014_r1c_run as frozen  # noqa: E402
from anra_v5 import cyr_gpu014_r1c_run_v2 as wrapper  # noqa: E402  (applies patch)
from v5_experiments import cyr_gpu011 as base  # noqa: E402
from v5_experiments import cyr_gpu014_r1c as core  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TINY = {"UPDATES": 2, "EVAL_EVERY": 1, "CHECKPOINT_EVERY": 2,
        "DIAGNOSTIC_UPDATES": (0, 1, 2), "WALL_MINUTES": 5.0,
        "PACKAGING_RESERVE_MINUTES": 1.0}


@pytest.fixture(scope="module")
def cpu_torch():
    # Engineering shims: the frozen runner targets Colab T4 and calls CUDA
    # sync/cache unconditionally; on the CPU harness these are no-ops.
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.empty_cache = lambda *a, **k: None
    torch.cuda.set_rng_state_all = lambda *a, **k: None
    return torch


@pytest.fixture()
def tiny_core():
    saved = {name: getattr(core, name) for name in
             (*TINY, "ROW_PRESENTATIONS")}
    for name, value in TINY.items():
        setattr(core, name, value)
    setattr(core, "ROW_PRESENTATIONS", core.BATCH_ROWS * TINY["UPDATES"])
    yield TINY["UPDATES"]
    for name, value in saved.items():
        setattr(core, name, value)


@pytest.fixture(scope="module")
def campaign_data():
    manifest = REPO / "docs/cymek/experiments/CYR-GPU-011/ARK002B_TASK_MANIFEST.json"
    data = base.load_ark002b_manifest(manifest)
    assert data["source_split_sha256"] == \
        "0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236"
    return data


@pytest.fixture()
def out_dir(tmp_path: Path) -> Path:
    out = tmp_path / "r1c_e2e"
    out.mkdir(parents=True)
    (out / "PREEXECUTION_GATE.json").write_text(json.dumps({"status": "PASS"}))
    (out / "RESOLVED.json").write_text(
        json.dumps({"estimated_campaign_seconds": 60.0}))
    return out


def test_compatibility_wrapper_is_bound():
    assert frozen.build_optimizer.__module__ == wrapper.__name__


def test_run_arm_full_24576_end_to_end(cpu_torch, tiny_core, campaign_data,
                                       out_dir):
    battery = base.make_reasoning_battery(campaign_data)
    body = frozen.run_arm(
        out=out_dir, seed_index=0, arm="FULL_24576", data=campaign_data,
        battery=battery, torch=cpu_torch, device=torch.device("cpu"),
        deadline=time.monotonic() + 3600.0,
    )
    assert body["schema"] == "anra-cyr-gpu014-r1c-arm/v1"
    assert body["acquisition"]["status"] == "COMPLETE"
    assert body["acquisition"]["updates"] == tiny_core
    assert body["treatment"] == core.treatment_spec("FULL_24576")


def test_run_arm_masked_and_offset_end_to_end(cpu_torch, tiny_core,
                                              campaign_data, out_dir):
    battery = base.make_reasoning_battery(campaign_data)
    for arm in ("MASK_4096", "OFFSET_EQ4096"):
        body = frozen.run_arm(
            out=out_dir, seed_index=0, arm=arm, data=campaign_data,
            battery=battery, torch=cpu_torch, device=torch.device("cpu"),
            deadline=time.monotonic() + 3600.0,
        )
        assert body["acquisition"]["status"] == "COMPLETE", arm
        assert body["acquisition"]["trace"][0]["active_only_measurement_diagnostic"]
        checkpoint = out_dir / "arms" / core.arm_label(0, arm) / "resume.pt"
        assert checkpoint.exists()


def test_run_campaign_driver_end_to_end(cpu_torch, tiny_core, campaign_data,
                                        tmp_path):
    """Regression for the operator's launch failure: the campaign died on the
    MASK_8192 arm with `abort CLIP_BREACH: post-clip global norm 1.0000042915
    exceeds 1.0` — float32 reduction-order noise between the fused clip path
    and the certificate's recomputed norm. With the 1e-4 numerics tolerance the
    whole driver completes; the verdict stays honestly inconclusive because the
    engineering fixture runs one seed."""
    import anra_v5.cyr_gpu014_r1c_run_v2  # optimizer patch active
    saved = {name: getattr(core, name) for name in
             ("MODEL_SEEDS", "ORDER_SEEDS", "ARM_ORDERS")}
    core.MODEL_SEEDS = (3711,)
    core.ORDER_SEEDS = (6001,)
    core.ARM_ORDERS = [list(core.ARMS)]
    out = tmp_path / "driver"
    out.mkdir(parents=True)
    (out / "PREEXECUTION_GATE.json").write_text(json.dumps({"status": "PASS"}))
    (out / "RESOLVED.json").write_text(
        json.dumps({"estimated_campaign_seconds": 60.0}))
    try:
        result = frozen.run_campaign(
            repo=REPO, out=out, preregistration=None, torch=cpu_torch,
            device=torch.device("cpu"))
    finally:
        for name, value in saved.items():
            setattr(core, name, value)
    assert result["status"] == "COMPLETE"
    assert result["completed_arm_count"] == len(core.ARMS)
    assert result["decision"]["verdict"] == "INCONCLUSIVE_INCOMPLETE_FOUR_SEED_CAMPAIGN"
    assert (out / "CAMPAIGN_RECEIPT.json").exists()
    bundle = Path(result["bundle"]["path"])
    assert bundle.exists()
