from __future__ import annotations

import torch
import pytest

from evaluation.release_drills import evaluate_adversarial_gate, evaluate_canary
from inference.serving_gates import (
    LatencySample,
    evaluate_accelerator_gate,
    evaluate_latency_budget,
)
from inference.speculative import SpeculativeBenchmark
from training.dpo import direct_preference_loss
from training.gepa_cycles import run_gepa_cycles
from training.qat import qat_parity_report
from ui.usability import USABILITY_SCENARIOS, run_usability_script


def test_latency_and_accelerator_gate_require_every_evidence_type() -> None:
    latency = evaluate_latency_budget(
        [
            LatencySample(ttft_ms=100, decode_tokens=50, decode_ms=1000, verified=False),
            LatencySample(ttft_ms=140, decode_tokens=50, decode_ms=1000, verified=True),
        ]
    )
    report = evaluate_accelerator_gate(
        speculative=SpeculativeBenchmark(100, 40, 3.0, 1.5),
        parity={"token_parity": True, "distribution_parity": True},
        qat_max_relative_error=0.005,
        latency=latency,
    )
    assert latency["passed"] is True
    assert report["passed"] is True
    assert evaluate_latency_budget([])["passed"] is False


def test_verified_latency_gate_requires_comparator_and_counts_decode_overhead() -> None:
    missing_comparator = evaluate_latency_budget(
        [LatencySample(ttft_ms=100, decode_tokens=50, decode_ms=1000, verified=True)]
    )
    assert missing_comparator["passed"] is False
    assert missing_comparator["gates"]["verified_latency"] is False  # type: ignore[index]

    verifier_bound = evaluate_latency_budget(
        [
            LatencySample(ttft_ms=100, decode_tokens=1000, decode_ms=1000, verified=False),
            LatencySample(ttft_ms=100, decode_tokens=1000, decode_ms=3000, verified=True),
        ]
    )
    assert verifier_bound["gates"]["verified_latency"] is False  # type: ignore[index]
    assert verifier_bound["verified_p95_multiplier"] == pytest.approx(3100 / 1100)


def test_dpo_objective_is_differentiable_and_validates_alignment() -> None:
    chosen = torch.tensor([1.0, 2.0], requires_grad=True)
    loss = direct_preference_loss(
        chosen,
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.5, 0.5]),
        torch.tensor([0.0, 0.0]),
    )
    loss.backward()
    assert loss.item() > 0
    assert chosen.grad is not None


def test_qat_parity_gate_has_a_strict_one_percent_limit() -> None:
    reference = torch.tensor([1.0, 2.0])
    assert qat_parity_report(reference, reference.clone())["passed"] is True
    assert qat_parity_report(reference, torch.tensor([1.1, 2.0]))["passed"] is False


def test_ten_gepa_cycles_record_a_correct_rejection_without_auto_apply() -> None:
    def evidence(cycle: int) -> dict[str, object]:
        return {
            "eval_summary": {
                "results": [
                    {
                        "id": f"case-{cycle}",
                        "category": "symbolic",
                        "prompt": "2 + 2",
                        "response": "3",
                        "score": 0.0,
                    }
                ]
            }
        }

    report = run_gepa_cycles(
        cycles=10,
        evidence_for_cycle=evidence,
        review_candidate=lambda cycle, _candidate: (cycle != 1, "rejected_by_eval"),
    )
    assert report["ten_cycle_gate"] is True
    assert report["auto_apply_enabled"] is False
    assert report["rows"][0]["reviews"][0]["approved"] is False  # type: ignore[index]


def test_usability_script_contains_twenty_traceable_scenarios() -> None:
    assert len(USABILITY_SCENARIOS) == 20
    report = run_usability_script(
        lambda scenario: dict.fromkeys(scenario.required_evidence, True)
    )
    assert report["passed"] is True


def test_canary_and_adversarial_drills_fail_closed() -> None:
    canary = evaluate_canary([{"success": True, "regressed": False}] * 20)
    adversarial = evaluate_adversarial_gate(
        [{"blocked": True, "evidence": True}, {"blocked": True, "evidence": True}]
    )
    assert canary["passed"] is True
    assert adversarial["passed"] is True
    assert evaluate_canary([])["passed"] is False
    assert evaluate_adversarial_gate([])["passed"] is False
