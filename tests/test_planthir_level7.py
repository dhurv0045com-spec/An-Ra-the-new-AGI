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
    assert shield["passed_shields"] == shield["total_shields"] == 4


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
    assert doc["checks"]["level7_runtime_shield"]["passed"] is True
    assert "shield" in agent_info()["cli"]

    cal = run_fault_injection_suite()
    assert cal["total_faults"] == 6 and cal["detected_faults"] == 6
    assert cal["level7_shield_passed"] is True

    shield_cli = json.loads(_run_cli(["shield"]).stdout)
    assert shield_cli["passed"] is True and shield_cli["passed_shields"] == 4
    lvl_cli = json.loads(_run_cli(["level"]).stdout)
    assert lvl_cli["level"] == 7 and lvl_cli["schema"] == "anra-planthir-level7-v1"
