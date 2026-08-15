from __future__ import annotations

from training.formal_proof_pilot import build_formal_proof_suite, run_formal_proof_pilot
from training.moonshot_pilots import evaluate_moonshot_pilot


def test_formal_proof_suite_is_balanced_and_adversarial() -> None:
    cases = build_formal_proof_suite()
    assert len(cases) == 100
    assert sum(case.expected_valid for case in cases) == 50
    assert len({case.case_id for case in cases}) == 100


def test_m6_formal_proof_pilot_passes_registered_gate() -> None:
    report = run_formal_proof_pilot()
    assert report["correct_cases"] == 100
    assert evaluate_moonshot_pilot("m6", report["metrics"])["passed"] is True  # type: ignore[arg-type]
