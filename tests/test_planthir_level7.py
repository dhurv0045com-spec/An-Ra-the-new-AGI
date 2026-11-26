"""Comprehensive verification suite for Planthir / Palantiri Level-7 zero-late-crash upgrades."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from anra_v5.elendril.representation import SpecialToken, UTF8
from anra_v5.elendril.v8.contracts import build_v8_model_spec
from anra_v5.elendril.v8.data import build_d1_training_dataset
from anra_v5.elendril.v8.models import build_v8_model
from anra_v5.palantiri import (
    CampaignOrchestrator,
    CartridgeRunner,
    GenericExperimentTrainer,
    GenericModelSpec,
    GenericTrainRecipe,
    LegacyByteObservablePolicy,
    PLANTHIR_LEVEL,
    Palantiri,
    SessionJob,
    agent_info,
    build_cartridge_from_spec,
    build_paired_seed_observations,
    build_pinned_kaggle_notebook,
    build_x1_spec,
    decode_generated_bytes,
    dump_json_safe,
    export_session_archives,
    init_spec_template,
    level_info,
    planthir_doctor,
    run_fault_injection_suite,
    run_level7_runtime_shield_suite,
    sanitize_for_json,
    serialize_pinned_notebook,
    validate_byte_decoder_resilience,
    validate_end_to_end_lifecycle,
    validate_paired_arm_records_preflight,
    verify_pinned_notebook_bytes,
    verify_session_archive,
)
from anra_v5.elendril.v8x.evaluation import X1S20ObservablePolicy
from anra_v5.elendril.v8x.models import build_x1_s20_model
from anra_v5.planthir.adapters.v8_adapter import V8EnvironmentKernel, V8ObservablePolicy
from v5_model.core import initialize, packed_layout

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, "-m", "anra_v5.palantiri", *args],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, f"args={args}\nrc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
    return proc


def test_level7_lossy_byte_decoding_0xB5_and_b0_spec() -> None:
    bad_seq = [UTF8.bos_id, 4 + 0xB5, UTF8.eos_id]
    with pytest.raises(ValueError):
        UTF8.decode([4 + 0xB5])
    with pytest.raises(ValueError):
        UTF8.decode(bad_seq)

    lossy_text, valid_bad = UTF8.decode_lossy(bad_seq, strip_special=True)
    assert valid_bad is False
    assert "\ufffd" in lossy_text

    ok_seq = [UTF8.bos_id, *(4 + b for b in b"OK:42"), UTF8.eos_id]
    ok_text, valid_ok = UTF8.decode_lossy(ok_seq, strip_special=True)
    assert valid_ok is True
    assert ok_text == "OK:42"

    assert UTF8.is_ascii_stop_token(4 + ord("\n"), ("\n",)) is True
    assert UTF8.is_ascii_stop_token(4 + 0xB5, ("\n",)) is False

    dec_bad = decode_generated_bytes([UTF8.bos_id, 4 + ord("X"), 4 + 0xB5, UTF8.eos_id])
    assert dec_bad["valid_utf8"] is False
    assert dec_bad["terminated_by_eos"] is True

    dec_stop = decode_generated_bytes([UTF8.bos_id, 4 + ord("4"), 4 + ord("2"), 4 + ord("\n"), 4 + 0xB5, UTF8.eos_id])
    assert dec_stop["text"] == "42"
    assert dec_stop["valid_utf8"] is True
    assert dec_stop["terminated_by_eos"] is True

    base_spec = build_x1_spec("P", cpu_canary=True).base_spec
    core_model = initialize(base_spec, 7, torch_module=torch)
    legacy_pol = LegacyByteObservablePolicy(core_model, base_spec, device="cpu")
    op, arg = legacy_pol._decode_emitted_action([4 + 0xB5, 4 + ord("A")])
    assert (op, arg) == ("ABSTAIN", None)

    b0_spec = build_x1_spec("B0", cpu_canary=True)
    b0_model = build_x1_s20_model(b0_spec, 7, torch_module=torch)
    b0_pol = X1S20ObservablePolicy(b0_model, b0_spec, model_identity="b0-probe")
    assert b0_pol.v8_spec is not None and b0_pol.v8_spec.base_spec == b0_spec.base_spec


def test_level7_json_sanitizer_and_dump_safe() -> None:
    dirty = {
        "nan": float("nan"),
        "inf": float("inf"),
        "ninf": float("-inf"),
        "scalar_t": torch.tensor(1.25),
        "vec_t": torch.tensor([1.0, 2.0]),
        "big_t": torch.zeros(5000),
        "path": Path("a/b/c.bin"),
        "raw": b"\xb5hi",
        "tags": {"b", "a"},
    }
    with pytest.raises((TypeError, ValueError)):
        json.dumps(dirty, allow_nan=False)

    sanitized = sanitize_for_json(dirty)
    assert sanitized["nan"] is None
    assert sanitized["inf"] is None
    assert sanitized["ninf"] is None
    assert abs(float(sanitized["scalar_t"]) - 1.25) < 1e-6
    assert sanitized["vec_t"] == [1.0, 2.0]
    assert sanitized["big_t"]["__tensor__"] is True
    assert isinstance(sanitized["tags"], list)

    dumped = dump_json_safe(dirty)
    reparsed = json.loads(dumped)
    assert reparsed["big_t"]["numel"] == 5000


def test_level7_export_archives_order_before_summary_cap_and_compaction(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    core_f = src / "METRICS.json"
    core_f.write_text('{"acc": 1.0}\n', encoding="utf-8")
    evid_f = src / "JOB_RECEIPT.json"
    evid_f.write_text('{"job_id": "j1"}\n', encoding="utf-8")
    heavy_f = src / "HEAVY_TRACE.bin"
    heavy_f.write_bytes(random.Random(11).randbytes(48 * 1024))

    out_auto = tmp_path / "out_auto"
    receipt = export_session_archives(
        src,
        out_auto,
        prefix="AUTO",
        summary_files={"METRICS.json": core_f, "HEAVY_TRACE.bin": heavy_f},
        evidence_files={"METRICS.json": core_f, "JOB_RECEIPT.json": evid_f},
        analysis_files={"METRICS.json": core_f},
        full_files={"METRICS.json": core_f, "JOB_RECEIPT.json": evid_f, "HEAVY_TRACE.bin": heavy_f},
        status="COMPLETE",
        source_commit="3" * 40,
        summary_max_bytes=10 * 1024,
        strict_summary_cap=False,
    )
    assert receipt["summary_compacted"] is True
    assert "HEAVY_TRACE.bin" in receipt["summary_omitted_for_cap"]
    assert verify_session_archive(receipt["archives"]["full"], expected_source_commit="3" * 40)["verified"]
    assert verify_session_archive(receipt["archives"]["evidence"], expected_source_commit="3" * 40)["verified"]
    assert verify_session_archive(receipt["archives"]["analysis"], expected_source_commit="3" * 40)["verified"]
    assert verify_session_archive(
        receipt["archives"]["summary"],
        max_bytes=10 * 1024,
        expected_source_commit="3" * 40,
    )["verified"]

    out_strict = tmp_path / "out_strict"
    with pytest.raises(ValueError, match="exceeds cap"):
        export_session_archives(
            src,
            out_strict,
            prefix="STRICT",
            summary_files={"METRICS.json": core_f, "HEAVY_TRACE.bin": heavy_f},
            evidence_files={"METRICS.json": core_f, "JOB_RECEIPT.json": evid_f},
            analysis_files={"METRICS.json": core_f},
            full_files={"METRICS.json": core_f, "JOB_RECEIPT.json": evid_f, "HEAVY_TRACE.bin": heavy_f},
            status="COMPLETE",
            source_commit="3" * 40,
            summary_max_bytes=10 * 1024,
            strict_summary_cap=True,
        )
    # FULL, EVIDENCE, and ANALYSIS must already be committed on disk BEFORE SUMMARY fails.
    assert verify_session_archive(out_strict / "STRICT_FULL.zip", expected_source_commit="3" * 40)["verified"]
    assert verify_session_archive(out_strict / "STRICT_EVIDENCE.zip", expected_source_commit="3" * 40)["verified"]
    assert verify_session_archive(out_strict / "STRICT_ANALYSIS.zip", expected_source_commit="3" * 40)["verified"]


def test_level7_notebook_backend_modules_and_try_finally_mirror(tmp_path: Path) -> None:
    nb = build_pinned_kaggle_notebook(
        source_commit="4" * 40,
        title="Level 7 Crash Safe Notebook",
        launch_command="python -m custom_pkg.run --full",
        backend_modules=("custom_pkg.module",),
        mirror_outputs_to_working=True,
        preflight_canary_backend="custom_pkg",
        preflight_commands=("python -m anra_v5.palantiri shield",),
        preflight_cuda_canary=True,
    )
    nb_path = tmp_path / "L7_NB.ipynb"
    nb_path.write_bytes(serialize_pinned_notebook(nb))
    ver = verify_pinned_notebook_bytes(nb_path, nb)
    assert ver["match"] is True

    code_sources = ["".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code"]
    for src_text in code_sources:
        ast.parse(src_text)
    full_code = "\n".join(code_sources)
    assert "PLANTHIR_BACKEND_MODULES" in full_code
    assert "custom_pkg.module" in full_code

    launch_cell = code_sources[-1]
    assert "CRASH_RECEIPT.json" in launch_cell
    tree = ast.parse(launch_cell)
    try_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Try)]
    assert any(t.handlers and t.finalbody for t in try_nodes)


class _TinyAMPMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_level7_trainer_native_amp_and_shield_suite(tmp_path: Path) -> None:
    model = _TinyAMPMLP()
    spec = GenericModelSpec("tiny_amp", {"dim": 4}, sum(p.numel() for p in model.parameters()), "demo_amp")
    recipe = GenericTrainRecipe(seed=7, total_updates=2, batch_size=2, micro_batch_size=1, lr=1e-3, precision="amp")
    items = [(torch.ones(4), idx % 2) for idx in range(4)]

    def _loss(m: nn.Module, batch: list[tuple[torch.Tensor, int]], _rec: GenericTrainRecipe) -> torch.Tensor:
        xs = torch.stack([b[0] for b in batch])
        ys = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        return F.cross_entropy(m(xs), ys)

    ckpt_dir = tmp_path / "ckpt"
    trainer = GenericExperimentTrainer(
        backend_id="demo_amp",
        model=model,
        model_spec=spec,
        recipe=recipe,
        loss_fn=_loss,
        items=items,
        data_provenance={"dataset_name": "tiny"},
        order_seed=7,
        source_commit="5" * 40,
        root=ckpt_dir,
        lineage="main",
        device="cpu",
        cpu_canary=True,
    )
    assert trainer.amp_enabled is True
    assert trainer.amp_dtype == torch.float16
    trainer.step()
    trainer.save()

    resumed = GenericExperimentTrainer(
        backend_id="demo_amp",
        model=_TinyAMPMLP(),
        model_spec=spec,
        recipe=recipe,
        loss_fn=_loss,
        items=items,
        data_provenance={"dataset_name": "tiny"},
        order_seed=7,
        source_commit="5" * 40,
        root=ckpt_dir,
        lineage="main",
        device="cpu",
        cpu_canary=True,
        resume=True,
    )
    assert resumed.state.global_update == 1

    shield = run_level7_runtime_shield_suite()
    assert shield["passed"] is True
    assert shield["passed_shields"] == shield["total_shields"] == 5


def test_level7_orchestrator_preflight_blocks_and_crash_partial_custody(tmp_path: Path) -> None:
    jobs = (
        SessionJob("j1", "P", 7, 17, 2, extra={"dataset_counts": {"train": 8}}),
        SessionJob("j2", "P", 19, 29, 2, extra={"depends_on": ["j1"], "dataset_counts": {"train": 8}}),
    )

    orch_bad_dec = CampaignOrchestrator(campaign_id="L7-BDEC", source_commit="6" * 40, work_dir=tmp_path / "bdec")
    with pytest.raises(RuntimeError, match="U0000_BYTE_DECODER_SHIELD_FAILED"):
        orch_bad_dec.run_registered_plan(
            jobs,
            devices=("cuda:0",),
            measured_seconds_per_update=1.0,
            byte_decode_fn=UTF8.decode,
            job_runner=lambda _j, _w, _d: 0,
        )
    smoke_rec = json.loads((tmp_path / "bdec" / "U0000_SMOKE_RECEIPT.json").read_text(encoding="utf-8"))
    assert smoke_rec["passed"] is False
    assert smoke_rec["byte_decoder_shield"]["passed"] is False

    orch_bad_s5 = CampaignOrchestrator(campaign_id="L7-S5P", source_commit="6" * 40, work_dir=tmp_path / "s5p")
    with pytest.raises(RuntimeError, match="STAGE5_PREFLIGHT_FAILED"):
        orch_bad_s5.run_registered_plan(
            jobs,
            devices=("cuda:0",),
            measured_seconds_per_update=1.0,
            stage5_preflight_fn=lambda _d: {"passed": False, "reason": "bad_codec"},
            job_runner=lambda _j, _w, _d: 0,
        )
    s5_rec = json.loads((tmp_path / "s5p" / "STAGE5_PREFLIGHT_RECEIPT.json").read_text(encoding="utf-8"))
    assert s5_rec["passed"] is False and s5_rec["reason"] == "bad_codec"

    crash_dir = tmp_path / "crash_camp"
    orch_crash = CampaignOrchestrator(campaign_id="L7-CRASH", source_commit="6" * 40, work_dir=crash_dir)

    def _crash_on_j2(job: SessionJob, _wroot: Path, _dev: str) -> dict[str, object]:
        if job.job_id == "j2":
            raise RuntimeError("SIMULATED_WAVE2_CRASH")
        return {"exit_code": 0, "completed_updates": job.updates, "val_tensor": torch.tensor(0.95)}

    with pytest.raises(RuntimeError, match="SIMULATED_WAVE2_CRASH"):
        orch_crash.run_registered_plan(
            jobs,
            devices=("cuda:0",),
            measured_seconds_per_update=1.0,
            job_runner=_crash_on_j2,
        )

    comp = json.loads((crash_dir / "COMPLETED_JOBS.json").read_text(encoding="utf-8"))
    pend = json.loads((crash_dir / "PENDING_QUEUE.json").read_text(encoding="utf-8"))
    recs = json.loads((crash_dir / "JOB_RECEIPTS.json").read_text(encoding="utf-8"))
    st = json.loads((crash_dir / "REGISTERED_PLAN_STATUS.json").read_text(encoding="utf-8"))
    assert comp["completed_jobs"] == ["j1"]
    assert pend["pending_job_ids"] == ["j2"]
    assert len(recs["receipts"]) == 1 and abs(recs["receipts"][0]["val_tensor"] - 0.95) < 1e-5
    assert st["status"] == "FAILED" and "SIMULATED_WAVE2_CRASH" in st["error"]

    # Resume executes only j2 cleanly and completes campaign
    resumed_ids: list[str] = []

    def _ok_resume(job: SessionJob, _wroot: Path, _dev: str) -> int:
        resumed_ids.append(job.job_id)
        return 0

    orch_resume = CampaignOrchestrator(campaign_id="L7-CRASH", source_commit="6" * 40, work_dir=crash_dir)
    fin = orch_resume.run_registered_plan(
        jobs,
        devices=("cuda:0",),
        resume_input=(crash_dir,),
        measured_seconds_per_update=1.0,
        job_runner=_ok_resume,
    )
    assert fin["status"] == "COMPLETE"
    assert resumed_ids == ["j2"]


def test_level7_facade_cartridge_doctor_and_shield_cli(tmp_path: Path) -> None:
    assert Palantiri.level == PLANTHIR_LEVEL == 7
    assert "lossy_byte_decoding_shield" in level_info()["capabilities"]
    text_bad, ok_bad = Palantiri.decode_lossy([UTF8.bos_id, 4 + 0xB5, UTF8.eos_id])
    assert "\ufffd" in text_bad and ok_bad is False
    assert Palantiri.decode_generated_bytes([UTF8.bos_id, 4 + 0xB5, UTF8.eos_id])["valid_utf8"] is False
    assert Palantiri.validate_byte_decoder()["passed"] is True
    assert Palantiri.runtime_shield()["passed"] is True
    assert Palantiri.sanitize_json({"x": float("nan")})["x"] is None
    assert json.loads(Palantiri.dump_json_safe({"x": torch.tensor(3)})) == {"x": 3}

    d1 = build_d1_training_dataset(seed=7, quota_per_family=16)
    kernel = V8EnvironmentKernel()
    v8_spec = build_v8_model_spec(
        "F",
        base_spec=build_x1_spec("P", cpu_canary=True, context_length=512, max_sequence_length=512).base_spec,
    )
    v8_mod = build_v8_model(v8_spec, 7, torch_module=torch)
    pol = V8ObservablePolicy(v8_mod, v8_spec, model_identity="l7-probe")
    life = validate_end_to_end_lifecycle(pol, kernel, d1.worlds[:2])
    assert life["passed"] is True and life["rollouts_checked"] == 2
    assert Palantiri.validate_lifecycle(pol, kernel, d1.worlds[:2])["passed"] is True

    spec_doc = init_spec_template(backend_id="v8", canary=True)
    cartridge, run_kwargs = build_cartridge_from_spec(spec_doc, canary_override=True)
    c_out = CartridgeRunner().run(cartridge, work_root=tmp_path / "cart_l7", **run_kwargs)
    assert c_out["gates"]["gate1_instrument_integrity"]["checks"]["byte_decoder_shield"]["passed"] is True

    doc = planthir_doctor()
    assert doc["platform_level"] == 7
    assert doc["green_signal"] is True
    assert doc["checks"]["paired_comparison_preflight"]["passed"] is True
    assert doc["checks"]["paired_comparison_preflight"]["multi_stage_collision_detected"] is True
    assert doc["checks"]["paired_comparison_preflight"]["multi_stage_scoped_disambiguated"] is True
    assert doc["checks"]["level7_runtime_shield"]["passed"] is True
    assert doc["checks"]["level7_runtime_shield"]["passed_shields"] == 5
    assert "shield" in agent_info()["cli"]

    cal = run_fault_injection_suite()
    assert cal["total_faults"] == 6 and cal["detected_faults"] == 6
    assert cal["level7_shield_passed"] is True

    shield_cli = json.loads(_run_cli(["shield"]).stdout)
    assert shield_cli["passed"] is True and shield_cli["passed_shields"] == 5
    lvl_cli = json.loads(_run_cli(["level"]).stdout)
    assert lvl_cli["level"] == 7 and lvl_cli["schema"] == "anra-planthir-level7-v1"


def test_level7_multi_stage_paired_arm_preflight_scoping_dedup_and_orchestrator(tmp_path: Path) -> None:
    assert "multi_stage_paired_arm_preflight" in level_info()["capabilities"]

    seeds = (7, 19, 41)
    # Build a full 57-job multi-stage ledger where S2 and S3 share P-R-Iter_D3-Hybrid_S20
    full_ledger_jobs: list[SessionJob] = []
    # S0 (6), S1 (18), S2 (15), S3 (6 -> includes shared P-R-Iter_D3-Hybrid_S20), S4 (12) = 57 jobs
    stage_specs = (
        ("S0", ("REDGE_BASE", "REDGE_CTRL"), 6),
        ("S1", ("D3_A", "D3_B", "D3_C", "D3_D", "D3_E", "D3_F"), 18),
        ("S2", ("P-R-Iter_D3-Hybrid_S20", "ARCH_B", "ARCH_C", "ARCH_D", "ARCH_E"), 15),
        ("S3", ("P-R-Iter_D3-Hybrid_S20", "INTEG_TRT"), 6),
        ("S4", ("S50_A", "S50_B", "S50_C", "S50_D"), 12),
    )
    job_counter = 0
    for stg, arms, expected_count in stage_specs:
        created = 0
        for arm in arms:
            for s in seeds:
                if created >= expected_count:
                    break
                job_counter += 1
                full_ledger_jobs.append(
                    SessionJob(
                        job_id=f"X3-{stg}-{job_counter:02d}-s{s}",
                        architecture=arm,
                        init_seed=s,
                        order_seed=s,
                        updates=4,
                        session=stg,
                        extra={
                            "stage": stg,
                            "arm_id": arm,
                            "dataset_counts": {"train": 16},
                        },
                    )
                )
                created += 1
        assert created == expected_count
    assert len(full_ledger_jobs) == 57

    arm_records_57: list[dict[str, object]] = []
    for idx, j in enumerate(full_ledger_jobs):
        arm_id = str(j.extra["arm_id"])
        bonus = 0.04 if arm_id in ("ARCH_B", "INTEG_TRT") else 0.0
        if j.session == "S3" and arm_id == "P-R-Iter_D3-Hybrid_S20":
            bonus = 0.01
        arm_records_57.append(
            {
                "job_id": j.job_id,
                "stage": j.session,
                "arm_id": arm_id,
                "init_seed": j.init_seed,
                "order_seed": j.order_seed,
                "value": 0.60 + 0.015 * (idx % 3) + bonus,
            }
        )

    # 1. Preflight flags cross-stage shared arm when require_unique_without_scope=True
    pf_strict = validate_paired_arm_records_preflight(
        arm_records_57,
        arm_key="arm_id",
        require_unique_without_scope=True,
    )
    assert pf_strict["passed"] is False
    assert pf_strict["status"] == "COLLIDING_ARM_SEED_KEYS"
    assert pf_strict["collision_count"] == 3
    assert pf_strict["requires_scope_key"] is True
    assert all(c["arm"] == "P-R-Iter_D3-Hybrid_S20" and c["disambiguated_by_scope"] for c in pf_strict["collisions"])

    with pytest.raises(ValueError, match="COLLIDING_ARM_SEED_KEYS"):
        validate_paired_arm_records_preflight(
            arm_records_57,
            arm_key="arm_id",
            require_unique_without_scope=True,
            fail_on_collision=True,
        )

    # 2. Preflight passes when stage/study disambiguates cross-stage sharing, via facade too
    pf_scoped = Palantiri.validate_paired_arm_records(
        arm_records_57,
        arm_key="arm_id",
        require_unique_without_scope=False,
    )
    assert pf_scoped["passed"] is True
    assert pf_scoped["status"] == "SCOPED_ARM_SEED_KEYS_UNIQUE"
    assert pf_scoped["scoped_collision_count"] == 0

    # 3. Within-stage duplicate fails even with require_unique_without_scope=False
    bad_dup_rows = list(arm_records_57) + [
        {
            "job_id": "X3-S2-DUP-s7",
            "stage": "S2",
            "arm_id": "P-R-Iter_D3-Hybrid_S20",
            "init_seed": 7,
            "order_seed": 7,
            "value": 0.62,
        }
    ]
    pf_within_dup = validate_paired_arm_records_preflight(
        bad_dup_rows,
        arm_key="arm_id",
        require_unique_without_scope=False,
    )
    assert pf_within_dup["passed"] is False
    assert pf_within_dup["status"] == "COLLIDING_SCOPED_ARM_SEED_KEYS"
    assert pf_within_dup["scoped_collision_count"] == 1
    with pytest.raises(ValueError, match="COLLIDING_SCOPED_ARM_SEED_KEYS"):
        validate_paired_arm_records_preflight(
            bad_dup_rows,
            arm_key="arm_id",
            require_unique_without_scope=False,
            fail_on_collision=True,
        )

    # 4. build_paired_seed_observations: default error, scope_key filtering, dedup="first"/"last"
    with pytest.raises(ValueError, match="duplicate record for arm 'P-R-Iter_D3-Hybrid_S20'"):
        build_paired_seed_observations(
            arm_records_57,
            contrast_id="S2_ARCH",
            control_arm="P-R-Iter_D3-Hybrid_S20",
            treatment_arm="ARCH_B",
            arm_key="arm_id",
        )

    with pytest.raises(ValueError, match="scope_value"):
        build_paired_seed_observations(
            arm_records_57,
            contrast_id="S2_ARCH",
            control_arm="P-R-Iter_D3-Hybrid_S20",
            treatment_arm="ARCH_B",
            arm_key="arm_id",
            scope_key="stage",
        )
    with pytest.raises(ValueError, match="deduplicate"):
        build_paired_seed_observations(
            arm_records_57,
            contrast_id="S2_ARCH",
            control_arm="P-R-Iter_D3-Hybrid_S20",
            treatment_arm="ARCH_B",
            arm_key="arm_id",
            deduplicate="invalid",
        )

    s2_scoped = build_paired_seed_observations(
        arm_records_57,
        contrast_id="S2_ARCH",
        control_arm="P-R-Iter_D3-Hybrid_S20",
        treatment_arm="ARCH_B",
        arm_key="arm_id",
        scope_key="stage",
        scope_value="S2",
    )
    assert s2_scoped["status"] == "COMPLETE_PAIRS"
    assert s2_scoped["complete_pair_count"] == 3
    assert s2_scoped["deduplicated_records"] == []
    assert s2_scoped["scope_key"] == "stage" and s2_scoped["scope_value"] == "S2"

    dedup_first = build_paired_seed_observations(
        arm_records_57,
        contrast_id="S3_INTEG_FIRST",
        control_arm="P-R-Iter_D3-Hybrid_S20",
        treatment_arm="INTEG_TRT",
        arm_key="arm_id",
        deduplicate="first",
    )
    dedup_last = build_paired_seed_observations(
        arm_records_57,
        contrast_id="S3_INTEG_LAST",
        control_arm="P-R-Iter_D3-Hybrid_S20",
        treatment_arm="INTEG_TRT",
        arm_key="arm_id",
        deduplicate="last",
    )
    assert dedup_first["status"] == "COMPLETE_PAIRS" and dedup_first["complete_pair_count"] == 3
    assert len(dedup_first["deduplicated_records"]) == 3
    assert all(d["action"] == "skipped_duplicate" and d["record"]["stage"] == "S3" for d in dedup_first["deduplicated_records"])
    assert dedup_last["status"] == "COMPLETE_PAIRS" and dedup_last["complete_pair_count"] == 3
    assert len(dedup_last["deduplicated_records"]) == 3
    assert all(d["action"] == "replaced_earlier" and d["record"]["stage"] == "S2" for d in dedup_last["deduplicated_records"])
    # Last keeps the S3 control value (+0.01 higher than S2 control), so control_value increases by 0.01
    assert abs((dedup_last["observations"][0]["control_value"] - dedup_first["observations"][0]["control_value"]) - 0.01) < 1e-9

    # 5. CampaignOrchestrator.prepare & run_registered_plan integration on full 57-job ledger
    strict_dir = tmp_path / "orch_strict"
    orch_strict = CampaignOrchestrator(campaign_id="X3-STRICT", source_commit="8" * 40, work_dir=strict_dir)
    with pytest.raises(ValueError, match="COLLIDING_ARM_SEED_KEYS"):
        orch_strict.run_registered_plan(
            full_ledger_jobs,
            devices=("cuda:0", "cuda:1"),
            measured_seconds_per_update=1.0,
            inventory_only=True,
            require_unique_arms_without_scope=True,
            fail_on_paired_arm_collision=True,
        )
    strict_manifest = json.loads((strict_dir / "PREPARED_MANIFEST.json").read_text(encoding="utf-8"))
    assert strict_manifest["paired_arm_preflight"]["passed"] is False
    assert strict_manifest["paired_arm_preflight"]["collision_count"] == 3

    # Scoped multi-stage 57-job ledger passes pre-training inventory & prepare checks cleanly
    scoped_dir = tmp_path / "orch_scoped"
    orch_scoped = CampaignOrchestrator(campaign_id="X3-SCOPED", source_commit="8" * 40, work_dir=scoped_dir)
    inv_ok = orch_scoped.run_registered_plan(
        full_ledger_jobs,
        devices=("cuda:0", "cuda:1"),
        measured_seconds_per_update=1.0,
        inventory_only=True,
        require_unique_arms_without_scope=False,
        fail_on_paired_arm_collision=True,
    )
    assert inv_ok["status"] == "INVENTORY_ONLY"
    assert inv_ok["paired_arm_preflight"]["status"] == "SCOPED_ARM_SEED_KEYS_UNIQUE"
    assert inv_ok["paired_arm_preflight"]["passed"] is True
    assert len(inv_ok["prepared"]["jobs"]) == 57

    # Same-stage duplicate fails fail-closed even when scoped check is enabled
    dup_stage_jobs = list(full_ledger_jobs) + [
        SessionJob(
            job_id="X3-S2-DUP-s7",
            architecture="P-R-Iter_D3-Hybrid_S20",
            init_seed=7,
            order_seed=7,
            updates=4,
            session="S2",
            extra={"stage": "S2", "arm_id": "P-R-Iter_D3-Hybrid_S20"},
        )
    ]
    dup_dir = tmp_path / "orch_dup"
    with pytest.raises(ValueError, match="COLLIDING_SCOPED_ARM_SEED_KEYS"):
        CampaignOrchestrator(campaign_id="X3-DUP", source_commit="8" * 40, work_dir=dup_dir).prepare(
            dup_stage_jobs,
            require_unique_without_scope=False,
            fail_on_arm_collision=True,
        )


def test_level7_checkpoint_long_path_and_replay_mismatch_diagnostics(tmp_path: Path) -> None:
    from anra_v5.palantiri import ReplayVerificationError, explain_error_code, independent_replay_bundle
    from anra_v5.planthir.checkpoint_intelligence import CheckpointCatalog
    from anra_v5.planthir.evidence import _collect_checkpoint_lineages, _verify_local_checkpoint_streaming
    from anra_v5.planthir.archives import win_long_path

    # 1. Deep nested Windows path (>275 chars to objects/<sha>/manifest.json) succeeds cleanly
    deep_root = (
        tmp_path
        / ("d_" + "a" * 34)
        / ("d_" + "b" * 34)
        / ("d_" + "c" * 34)
        / ("d_" + "d" * 34)
        / ("d_" + "e" * 34)
        / ("d_" + "f" * 34)
    )
    win_long_path(deep_root).mkdir(parents=True, exist_ok=True)
    model = _TinyAMPMLP()
    spec = GenericModelSpec("tiny_lp", {"dim": 4}, sum(p.numel() for p in model.parameters()), "demo_lp")
    recipe = GenericTrainRecipe(seed=7, total_updates=4, batch_size=2, micro_batch_size=1, lr=1e-3, precision="fp32")
    items = [(torch.ones(4), idx % 2) for idx in range(4)]

    def _loss(m: nn.Module, batch: list[tuple[torch.Tensor, int]], _rec: GenericTrainRecipe) -> torch.Tensor:
        xs = torch.stack([b[0] for b in batch])
        ys = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        return F.cross_entropy(m(xs), ys)

    trainer = GenericExperimentTrainer(
        backend_id="demo_lp",
        model=model,
        model_spec=spec,
        recipe=recipe,
        loss_fn=_loss,
        items=items,
        data_provenance={"dataset_name": "tiny_lp"},
        order_seed=7,
        source_commit="9" * 40,
        root=deep_root,
        lineage="main",
        device="cpu",
        cpu_canary=True,
    )
    trainer.step()
    sha1 = trainer.save()
    trainer.step()
    sha2 = trainer.save()
    manifest_len = len(str(trainer.store.objects / sha2 / "manifest.json"))
    assert manifest_len > 275, f"expected deep manifest path > 275 chars, got {manifest_len}"

    latest_sha = trainer.store.latest_sha256()
    assert latest_sha == sha2
    restored_state, _ = trainer.store.restore(sha2)
    assert restored_state.global_update == 2
    assert _verify_local_checkpoint_streaming(trainer.store, sha2).global_update == 2
    trainer.store.prune(keep=set())
    assert sha1 in set() or trainer.store.latest_sha256() == sha2

    lineages = _collect_checkpoint_lineages(deep_root)
    assert "main" in lineages and lineages["main"]["latest_sha256"] == sha2
    catalog = CheckpointCatalog.discover_checkpoints([deep_root], include_all_generations=True)
    assert any(r.checkpoint_sha256 == sha2 for r in catalog.records)

    # 2. Structured ORIGINAL_PREDICTION_MISMATCH diagnostics in ReplayVerificationError & CLI stdout/stderr
    assert explain_error_code("ORIGINAL_PREDICTION_MISMATCH")["known"] is True
    cli_dir = tmp_path / "replay_diag"
    cli_spec = cli_dir / "spec.json"
    cli_run = cli_dir / "run"
    _run_cli(["init", "--backend", "v8", "--out", str(cli_spec)])
    _run_cli(["run", "--spec", str(cli_spec), "--work-root", str(cli_run), "--canary", "--device", "cpu"])
    bdir = cli_run / "result_bundle"

    cohort = json.loads((bdir / "REPLAY_COHORT.json").read_text(encoding="utf-8"))
    ev_worlds = cohort["worlds"]
    arch_refs = cohort["reference_predictions"]
    lin_0 = sorted(arch_refs)[0]

    tampered_refs = json.loads(json.dumps(arch_refs))
    tampered_refs[lin_0][0]["operation"] = "BOGUS_OP"
    with pytest.raises(ReplayVerificationError) as exc_info:
        independent_replay_bundle(bdir, worlds=ev_worlds, reference_predictions=tampered_refs)
    assert isinstance(exc_info.value, ValueError)
    assert exc_info.value.mismatch_reasons == ["PREDICTION_MISMATCH"]
    assert exc_info.value.mismatch_counts_by_reason["PREDICTION_MISMATCH"] >= 1
    assert exc_info.value.triage_hints

    alt_idx = next(i for i in range(1, len(ev_worlds)) if ev_worlds[i] != ev_worlds[0])
    bad_worlds = json.loads(json.dumps(ev_worlds))
    bad_worlds[0] = {**ev_worlds[alt_idx], "world_id": ev_worlds[0]["world_id"]}
    bad_worlds_path = cli_dir / "bad_worlds.json"
    bad_worlds_path.write_text(json.dumps(bad_worlds), encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONHASHSEED"] = "0"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "anra_v5.palantiri",
            "replay-bundle",
            "--bundle-dir",
            str(bdir),
            "--worlds-json",
            str(bad_worlds_path),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 1
    cli_err = json.loads(proc.stdout)
    assert cli_err["verified"] is False
    assert cli_err["error"] == "ReplayVerificationError"
    assert "ORIGINAL_PREDICTION_MISMATCH" in cli_err["detail"]
    assert cli_err["mismatch_reasons"] == ["RENDERED_INPUT_HASH_MISMATCH"]
    assert cli_err["mismatch_counts_by_reason"]["RENDERED_INPUT_HASH_MISMATCH"] >= 1
    assert len(cli_err["mismatches"]) >= 1
    assert any("PYTHONHASHSEED" in h for h in cli_err["triage_hints"])
    assert cli_err["pythonhashseed"] == "0"
    assert "ORIGINAL_PREDICTION_MISMATCH" in proc.stderr
    assert "RENDERED_INPUT_HASH_MISMATCH" in proc.stderr
    assert "PYTHONHASHSEED" in proc.stderr


def test_level7_production_quota_split_disjointness_and_orchestrator_gate(tmp_path: Path) -> None:
    from anra_v5.palantiri import (
        explain_error_code,
        verify_full_mode_dispatch,
        verify_production_split_disjointness,
        verify_split_disjointness,
    )

    assert "production_quota_split_disjointness" in level_info()["capabilities"]
    assert explain_error_code("SPLIT_DISJOINTNESS_VIOLATION")["known"] is True
    assert explain_error_code("UNDERQUOTA_SPLIT_DISJOINTNESS_CHECK")["known"] is True

    # Canary quota per_family=2 looks disjoint, but production quota per_family=8 collides across train & dev!
    def _leaky_at_prod_builder(*, per_family: int = 2) -> dict[str, list[dict[str, Any]]]:
        train = [
            {
                "world_id": f"tr_{fam}_{i}",
                "family": fam,
                "public_prompt": f"Solve {fam} instance {i}",
            }
            for fam in ("X01", "X02")
            for i in range(int(per_family))
        ]
        dev = [
            {
                "world_id": f"dv_{fam}_{i}",
                "family": fam,
                # At per_family >= 8, dev item 0 borrows index 6 from train!
                "public_prompt": f"Solve {fam} instance {6 if (per_family >= 8 and i == 0) else 100 + i}",
            }
            for fam in ("X01", "X02")
            for i in range(int(per_family))
        ]
        return {"train": train, "dev": dev}

    def _clean_prod_builder(*, per_family: int = 2) -> dict[str, list[dict[str, Any]]]:
        return {
            "train": [
                {"world_id": f"tr_{fam}_{i}", "family": fam, "public_prompt": f"Solve {fam} train {i}"}
                for fam in ("X01", "X02")
                for i in range(int(per_family))
            ],
            "dev": [
                {"world_id": f"dv_{fam}_{i}", "family": fam, "public_prompt": f"Solve {fam} dev {i}"}
                for fam in ("X01", "X02")
                for i in range(int(per_family))
            ],
        }

    # 1. Under-quota check at per_family=2 passes unguarded, but fails closed when min_per_family=8 is required
    canary_res = verify_split_disjointness(_leaky_at_prod_builder(per_family=2))
    assert canary_res["passed"] is True
    with pytest.raises(ValueError, match="UNDERQUOTA_SPLIT_DISJOINTNESS_CHECK"):
        verify_split_disjointness(_leaky_at_prod_builder(per_family=2), min_per_family=8)

    under_nonfatal = verify_split_disjointness(
        _leaky_at_prod_builder(per_family=2),
        min_per_family=8,
        fail_closed=False,
    )
    assert under_nonfatal["passed"] is False
    assert under_nonfatal["status"] == "UNDERQUOTA_SPLIT_DISJOINTNESS_CHECK"
    assert under_nonfatal["quota_check"]["passed"] is False
    assert len(under_nonfatal["quota_check"]["underquota_families"]) == 4

    # 2. verify_production_split_disjointness catches the per_family=8 leak locally before cloud submission
    with pytest.raises(ValueError, match="SPLIT_DISJOINTNESS_VIOLATION"):
        verify_production_split_disjointness(_leaky_at_prod_builder, production_quota=8)

    with pytest.raises(ValueError, match="SPLIT_DISJOINTNESS_VIOLATION"):
        Palantiri.verify_production_splits(_leaky_at_prod_builder, production_quota=8)

    ok_prod = Palantiri.verify_production_splits(_clean_prod_builder, production_quota=8)
    assert ok_prod["passed"] is True
    assert ok_prod["production_quota"] == 8
    assert ok_prod["quota_check"]["passed"] is True
    assert ok_prod["quota_check"]["per_family_counts"]["train"] == {"X01": 8, "X02": 8}

    # 3. CampaignOrchestrator.run_registered_plan blocks Wave 0 when split_disjointness_fn fails at production quota
    jobs = (
        SessionJob("sp_j1", "P", 7, 7, 2, extra={"dataset_counts": {"train": 16}}),
        SessionJob("sp_j2", "R", 19, 19, 2, extra={"dataset_counts": {"train": 16}}),
    )
    bad_sd_dir = tmp_path / "bad_sd"
    orch_bad = CampaignOrchestrator(campaign_id="L7-SD-BAD", source_commit="9" * 40, work_dir=bad_sd_dir)
    with pytest.raises(RuntimeError, match="SPLIT_DISJOINTNESS_PREFLIGHT_FAILED"):
        orch_bad.run_registered_plan(
            jobs,
            devices=("cuda:0", "cuda:1"),
            measured_seconds_per_update=0.5,
            split_disjointness_fn=lambda _d: verify_production_split_disjointness(
                _leaky_at_prod_builder,
                production_quota=8,
            ),
            job_runner=lambda _j, _w, _d: {"exit_code": 0},
        )
    sd_rec = json.loads((bad_sd_dir / "SPLIT_DISJOINTNESS_RECEIPT.json").read_text(encoding="utf-8"))
    assert sd_rec["passed"] is False and "SPLIT_DISJOINTNESS_VIOLATION" in sd_rec["error"]

    ok_sd_dir = tmp_path / "ok_sd"
    orch_ok = CampaignOrchestrator(campaign_id="L7-SD-OK", source_commit="9" * 40, work_dir=ok_sd_dir)
    res_ok = orch_ok.run_registered_plan(
        jobs,
        devices=("cuda:0", "cuda:1"),
        measured_seconds_per_update=0.5,
        split_disjointness_fn=lambda _d: verify_production_split_disjointness(
            _clean_prod_builder,
            production_quota=8,
        ),
        job_runner=lambda _j, _w, _d: {"exit_code": 0},
    )
    assert res_ok["status"] == "COMPLETE"
    assert res_ok["split_disjointness_receipt"]["passed"] is True

    # 4. verify_full_mode_dispatch enforces production split disjointness when split_disjointness_fn is supplied
    def _mock_dispatch(work_dir: Path, *, session_minutes: float, resume_input: Any = None) -> dict[str, Any]:
        return CampaignOrchestrator(
            campaign_id="L7-DISP",
            source_commit="9" * 40,
            work_dir=work_dir,
            max_minutes=session_minutes,
            reserve_minutes=5.0,
            min_disk_bytes=1024,
        ).run_registered_plan(
            (
                SessionJob("sp_j1", "P", 7, 7, 2, extra={"dataset_counts": {"train": 16}}),
                SessionJob("sp_j2", "R", 19, 19, 2, extra={"dataset_counts": {"train": 16}}),
                SessionJob("sp_j3", "P", 41, 41, 2, extra={"dataset_counts": {"train": 16}}),
            ),
            devices=("cuda:0", "cuda:1"),
            resume_input=resume_input,
            measured_seconds_per_update=2.5,
            per_job_disk_bytes=1024,
            available_disk_bytes=10**8,
            job_runner=lambda _j, _w, _d: {"exit_code": 0},
        )

    bad_disp = verify_full_mode_dispatch(
        _mock_dispatch,
        expected_job_ids=("sp_j1", "sp_j2", "sp_j3"),
        expected_seeds=(7, 19, 41),
        split_disjointness_fn=lambda: verify_production_split_disjointness(_leaky_at_prod_builder, production_quota=8),
        raise_on_failure=False,
    )
    assert bad_disp["passed"] is False
    assert any(d.startswith("PRODUCTION_SPLIT_DISJOINTNESS_FAILED:") for d in bad_disp["defects"])

    ok_disp = verify_full_mode_dispatch(
        _mock_dispatch,
        expected_job_ids=("sp_j1", "sp_j2", "sp_j3"),
        expected_seeds=(7, 19, 41),
        split_disjointness_fn=lambda: verify_production_split_disjointness(_clean_prod_builder, production_quota=8),
        tight_session_minutes=5.1,
        full_session_minutes=60.0,
    )
    assert ok_disp["passed"] is True
    assert ok_disp["split_disjointness"]["passed"] is True
