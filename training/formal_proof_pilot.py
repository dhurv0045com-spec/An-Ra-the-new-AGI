"""Deterministic M6 formal-proof verifier pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from runtime.experience_ledger import content_hash
from verification import DEFAULT_VERIFIER_REGISTRY, VerifierRegistry


@dataclass(frozen=True)
class FormalProofCase:
    case_id: str
    premises: tuple[str, ...]
    rules: tuple[str, ...]
    steps: tuple[str, ...]
    conclusion: str
    expected_valid: bool


def build_formal_proof_suite() -> tuple[FormalProofCase, ...]:
    """Build 50 valid and 50 adversarial certificates without randomness."""
    cases: list[FormalProofCase] = []
    for index in range(50):
        prefix = f"valid_{index}"
        cases.append(
            FormalProofCase(
                case_id=prefix,
                premises=(f"{prefix}_a",),
                rules=(
                    f"{prefix}_a -> {prefix}_b",
                    f"{prefix}_b -> {prefix}_c",
                ),
                steps=(f"{prefix}_b", f"{prefix}_c"),
                conclusion=f"{prefix}_c",
                expected_valid=True,
            )
        )
    for index in range(50):
        prefix = f"invalid_{index}"
        cases.append(
            FormalProofCase(
                case_id=prefix,
                premises=(f"{prefix}_a",),
                rules=(f"{prefix}_a -> {prefix}_b",),
                steps=(f"{prefix}_injected",),
                conclusion=f"{prefix}_injected",
                expected_valid=False,
            )
        )
    return tuple(cases)


def run_formal_proof_pilot(
    registry: VerifierRegistry | None = None,
) -> dict[str, object]:
    """Measure verifier classification accuracy on the fixed M6 suite."""
    verifier_registry = registry or DEFAULT_VERIFIER_REGISTRY
    cases = build_formal_proof_suite()
    rows: list[dict[str, object]] = []
    correct = 0
    for case in cases:
        verdict = verifier_registry.verify(
            "formal_proof",
            {
                "premises": case.premises,
                "rules": case.rules,
                "steps": case.steps,
                "conclusion": case.conclusion,
            },
        )
        observed_valid = float(verdict.score) >= 0.8
        matched = observed_valid is case.expected_valid
        correct += int(matched)
        rows.append(
            {
                "case_id": case.case_id,
                "expected_valid": case.expected_valid,
                "observed_valid": observed_valid,
                "matched": matched,
                "reason": str(verdict.reason),
                "certificate_hash": content_hash(asdict(case)),
            }
        )
    metrics = {
        "proof_cases": len(cases),
        "deterministic_pass_rate": correct / len(cases),
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "pilot_id": "m6",
        "suite_hash": content_hash([asdict(case) for case in cases]),
        "metrics": metrics,
        "correct_cases": correct,
        "rows": rows,
    }
    report["report_hash"] = content_hash(report)
    return report
