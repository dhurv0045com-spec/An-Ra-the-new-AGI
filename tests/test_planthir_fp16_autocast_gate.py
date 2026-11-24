"""Regression tests for padded FP16 autocast validation in instrument, Gate 3, and run_registered_plan."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from anra_v5.planthir import (
    CampaignOrchestrator,
    CartridgeRunner,
    ExperimentModule,
    GenericDatasetBundle,
    Palantiri,
    SessionJob,
    build_cartridge_from_spec,
    init_spec_template,
    validate_fp16_autocast_forward_backward,
    validate_model_and_gradients,
    validate_spec_document,
)
from anra_v5.planthir.backend import _BACKENDS

REPO_ROOT = Path(__file__).resolve().parents[1]


class _MaskedAttentionNet(nn.Module):
    """Toy padded attention network reproducing `-1e9` Half overflow vs FP32 reduction fix."""

    def __init__(self, width: int = 8, *, overflow_bug: bool = False, seed: int = 7) -> None:
        super().__init__()
        torch.manual_seed(int(seed))
        self.overflow_bug = bool(overflow_bug)
        self.emb = nn.Embedding(264, int(width))
        self.proj = nn.Linear(int(width), int(width))
        self.head = nn.Linear(int(width), 2)
        self.register_buffer("forward_calls", torch.zeros(1, dtype=torch.int64))

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.forward_calls.add_(1)
        valid = tokens != 0
        h = self.emb(tokens)
        scores = torch.matmul(self.proj(h), h.transpose(-1, -2))
        if self.overflow_bug:
            masked = scores.masked_fill(~valid.unsqueeze(1), -1e9)
            attn = F.softmax(masked, dim=-1)
        else:
            masked = scores.float().masked_fill(~valid.unsqueeze(1), -1e9)
            attn = F.softmax(masked, dim=-1).to(h.dtype)
        ctx = torch.matmul(attn, h)[:, 0, :]
        return self.head(ctx), ctx


def _padded_attention_loss(model: nn.Module) -> torch.Tensor:
    toks = torch.tensor(
        [
            [1, 5, 6, 0, 0],
            [1, 7, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    targets = torch.tensor([0, 1], dtype=torch.long)
    logits, _ = model(toks)
    return F.cross_entropy(logits.float(), targets)


def test_fp16_autocast_gate_catches_c10_half_minus_1e9_overflow_and_passes_fp32_cast() -> None:
    bad_model = _MaskedAttentionNet(overflow_bug=True)
    bad_model.eval()
    bad_model.head.weight.grad = torch.ones_like(bad_model.head.weight) * 3.0
    prev_calls = int(bad_model.forward_calls.item())

    # FP32-only check misses the -1e9 Half overflow
    fp32_only = validate_model_and_gradients(bad_model, _padded_attention_loss, check_fp16_autocast=False)
    assert fp32_only["passed"] is True and fp32_only["fp16_autocast"] is None

    # FP16 autocast gate catches the c10::Half overflow and rolls back mode/buffers/grads
    fp16_bad = validate_fp16_autocast_forward_backward(bad_model, _padded_attention_loss)
    assert fp16_bad["passed"] is False
    assert fp16_bad["status"] == "INSTRUMENT_DEFECT_DETECTED"
    assert any("fp16_autocast_runtime_error" in d for d in fp16_bad["defects"])
    assert any("Half" in d or "overflow" in d for d in fp16_bad["defects"])
    assert bad_model.training is False
    assert int(bad_model.forward_calls.item()) == prev_calls
    assert bad_model.head.weight.grad is not None and float(bad_model.head.weight.grad.mean().item()) == pytest.approx(3.0)

    combined_bad = validate_model_and_gradients(bad_model, _padded_attention_loss, check_fp16_autocast=True)
    assert combined_bad["passed"] is False
    assert combined_bad["fp16_autocast"] is not None and combined_bad["fp16_autocast"]["passed"] is False

    # Fixed FP32 softmax reduction passes cleanly under FP16 autocast and restores state
    good_model = _MaskedAttentionNet(overflow_bug=False)
    good_model.eval()
    fp16_good = Palantiri.validate_fp16_autocast(good_model, _padded_attention_loss)
    assert fp16_good["passed"] is True and fp16_good["nonzero_grad_count"] > 0
    combined_good = validate_model_and_gradients(good_model, _padded_attention_loss, check_fp16_autocast=True)
    assert combined_good["passed"] is True
    assert combined_good["fp16_autocast"] is not None and combined_good["fp16_autocast"]["passed"] is True
    assert good_model.training is False and int(good_model.forward_calls.item()) == 0


def test_experiment_module_supports_fp16_triggers_gate3_autocast_in_cartridge_runner(tmp_path: Path) -> None:
    def _build_ds(dcfg: Mapping[str, Any], cpu_canary: bool) -> dict[str, Any]:
        n = int(dcfg.get("num_items", 8))
        tr = [
            {
                "world_id": f"tr_{i}",
                "sig": f"tr_{i}",
                "tokens": [1, 5 + (i % 4), 6, 0, 0],
                "label": i % 2,
                "hidden": {"expected": "A" if i % 2 == 0 else "B", "expected_operation": "ANSWER"},
            }
            for i in range(n)
        ]
        ev = [
            {
                "world_id": f"ev_{i}",
                "sig": f"ev_{i}",
                "tokens": [1, 7 + (i % 4), 8, 0, 0],
                "label": i % 2,
                "hidden": {"expected": "A" if i % 2 == 0 else "B", "expected_operation": "ANSWER"},
            }
            for i in range(6)
        ]
        return {
            "train_data": GenericDatasetBundle(
                dataset_name="amp_attn",
                items=tuple(tr),
                data_identity="amp-attn-v1",
                manifest={"train": len(tr), "eval": len(ev)},
            ),
            "train_worlds": tr,
            "eval_worlds": ev,
        }

    def _loss(model: nn.Module, chunk: Sequence[Mapping[str, Any]], _recipe: Any) -> torch.Tensor:
        toks = torch.tensor([row["tokens"] for row in chunk], dtype=torch.long)
        targets = torch.tensor([row["label"] for row in chunk], dtype=torch.long)
        logits, _ = model(toks)
        return F.cross_entropy(logits.float(), targets)

    bad_bid = "amp_overflow_bad_test"
    good_bid = "amp_overflow_good_test"
    mod_bad = ExperimentModule(
        backend_id=bad_bid,
        architectures=("bad_amp", "good_amp"),
        objectives=("ce",),
        model_factory=lambda arch, mcfg, seed: _MaskedAttentionNet(
            int(mcfg.get("width", 8)), overflow_bug=(arch == "bad_amp"), seed=seed
        ),
        dataset_factory=_build_ds,
        loss_fn=_loss,
        action_labels=("A", "B"),
        default_dataset_config={"dataset_name": "amp_attn", "num_items": 8},
        default_model_config={"width": 8},
        treatment_architecture="good_amp",
        supports_fp16=True,
    ).register()
    mod_good = ExperimentModule(
        backend_id=good_bid,
        architectures=("safe_base", "safe_trt"),
        objectives=("ce",),
        model_factory=lambda _arch, mcfg, seed: _MaskedAttentionNet(
            int(mcfg.get("width", 8)), overflow_bug=False, seed=seed
        ),
        dataset_factory=_build_ds,
        loss_fn=_loss,
        action_labels=("A", "B"),
        default_dataset_config={"dataset_name": "amp_attn", "num_items": 8},
        default_model_config={"width": 8},
        treatment_architecture="safe_trt",
        supports_fp16=True,
    ).register()
    try:
        assert mod_bad.capabilities().supports_fp16 is True
        tpl_bad = init_spec_template(backend_id=bad_bid, canary=True)
        assert validate_spec_document(tpl_bad, canary_override=True)["valid"] is True
        cart_bad, rkw_bad = build_cartridge_from_spec(tpl_bad, canary_override=True)
        with pytest.raises(RuntimeError, match="Gate 3 gradient check failed.*fp16_autocast_runtime_error"):
            CartridgeRunner().run(cart_bad, work_root=tmp_path / "bad_run", **rkw_bad)

        tpl_good = init_spec_template(backend_id=good_bid, canary=True)
        cart_good, rkw_good = build_cartridge_from_spec(tpl_good, canary_override=True)
        good_run = tmp_path / "good_run"
        res_good = CartridgeRunner().run(cart_good, work_root=good_run, **rkw_good)
        assert res_good["gates"]["all_5_gates_passed"] is True
        assert res_good["gates"]["gate3_gradient_and_mechanism_health"]["passed"] is True
        arm_state = json.loads((good_run / "arm_state" / f"{cart_good.control_arm.arm_id}.json").read_text(encoding="utf-8"))
        assert arm_state["grad_check"]["fp16_autocast"]["passed"] is True
    finally:
        _BACKENDS.pop(bad_bid, None)
        _BACKENDS.pop(good_bid, None)


def test_run_registered_plan_u0000_fp16_autocast_gate_enforced_for_amp_jobs(tmp_path: Path) -> None:
    amp_jobs = [
        SessionJob("job_amp_1", "bad_amp", 7, 7, 4, device="cuda:0", extra={"precision": "fp16"}),
    ]

    # 1. AMP job without an FP16 autocast smoke gate fails closed before launching waves
    orch_missing = CampaignOrchestrator(
        campaign_id="AMP-MISSING",
        source_commit="a" * 40,
        work_dir=tmp_path / "amp_missing",
        min_disk_bytes=1024,
    )
    with pytest.raises(RuntimeError, match="U0000_FP16_AUTOCAST_REQUIRED_FOR_AMP_JOB"):
        orch_missing.run_registered_plan(
            amp_jobs,
            job_runner=lambda _j, _w, _d: {"exit_code": 0, "status": "COMPLETE"},
            devices=("cuda:0",),
            measured_seconds_per_update=0.5,
            per_job_disk_bytes=1024,
            available_disk_bytes=10**8,
            u0000_smoke_fn=lambda _arch, _job, _dir: {"passed": True},
        )

    # 2. Failing u0000_fp16_autocast_fn aborts Wave 0 with U0000_FP16_AUTOCAST_FAILED
    def _bad_fp16_smoke(arch: str, _job: SessionJob, _dir: Path) -> dict[str, Any]:
        net = _MaskedAttentionNet(overflow_bug=True)
        return validate_fp16_autocast_forward_backward(net, _padded_attention_loss)

    orch_fail = CampaignOrchestrator(
        campaign_id="AMP-FAIL",
        source_commit="a" * 40,
        work_dir=tmp_path / "amp_fail",
        min_disk_bytes=1024,
    )
    with pytest.raises(RuntimeError, match="U0000_FP16_AUTOCAST_FAILED"):
        orch_fail.run_registered_plan(
            amp_jobs,
            job_runner=lambda _j, _w, _d: {"exit_code": 0, "status": "COMPLETE"},
            devices=("cuda:0",),
            measured_seconds_per_update=0.5,
            per_job_disk_bytes=1024,
            available_disk_bytes=10**8,
            u0000_smoke_fn=lambda _arch, _job, _dir: {"passed": True},
            u0000_fp16_autocast_fn=_bad_fp16_smoke,
        )

    # 3. Passing u0000_fp16_autocast_fn unlocks campaign and records fp16_autocast_checked in receipt
    def _good_fp16_smoke(_arch: str, _job: SessionJob, _dir: Path) -> dict[str, Any]:
        net = _MaskedAttentionNet(overflow_bug=False)
        return validate_fp16_autocast_forward_backward(net, _padded_attention_loss)

    ok_dir = tmp_path / "amp_ok"
    orch_ok = CampaignOrchestrator(
        campaign_id="AMP-OK",
        source_commit="a" * 40,
        work_dir=ok_dir,
        min_disk_bytes=1024,
    )
    res_ok = orch_ok.run_registered_plan(
        amp_jobs,
        job_runner=lambda _j, _w, _d: {"exit_code": 0, "status": "COMPLETE"},
        devices=("cuda:0",),
        measured_seconds_per_update=0.5,
        per_job_disk_bytes=1024,
        available_disk_bytes=10**8,
        u0000_smoke_fn=lambda _arch, _job, _dir: {"passed": True},
        u0000_fp16_autocast_fn=_good_fp16_smoke,
    )
    assert res_ok["status"] == "COMPLETE"
    receipt = res_ok["u0000_smoke_receipt"]
    assert receipt["passed"] is True
    assert receipt["fp16_autocast_checked"] is True
    assert receipt["architectures"]["bad_amp"]["fp16_autocast_passed"] is True
    disk_receipt = json.loads((ok_dir / "U0000_SMOKE_RECEIPT.json").read_text(encoding="utf-8"))
    assert disk_receipt["fp16_autocast_checked"] is True


def test_manuals_document_fp16_autocast_u0000_gate_and_c10_half_gotcha() -> None:
    for name in ("orchestrator_agent.md", "execution_agent.md", "agent.md"):
        root_b = (REPO_ROOT / name).read_bytes()
        assert root_b == (REPO_ROOT / "anra_v5" / "planthir" / name).read_bytes()
        assert root_b == (REPO_ROOT / "research" / "planthir" / name).read_bytes()
        text = root_b.decode("utf-8")
        for needle in (
            "validate_fp16_autocast_forward_backward",
            "u0000_fp16_autocast_fn",
            "supports_fp16",
            "c10::Half",
        ):
            assert needle in text, f"missing {needle!r} in {name}"
