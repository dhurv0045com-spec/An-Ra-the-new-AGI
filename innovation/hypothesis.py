from __future__ import annotations

import time
import uuid

from innovation.schema import CapabilityGap, Hypothesis


_VERIFIER_MAP: dict[str, tuple[str, str]] = {
    "notimplementederror": (
        "run pytest on the module and confirm no NotImplementedError raised",
        "implement the function; pytest passes if no NotImplementedError",
    ),
    "todo": (
        "implement the TODO item; run the affected test file",
        "the TODO comment is removed and tests pass",
    ),
    "stub": (
        "implement the stub to return real data; assert output is non-None",
        "output is non-None and the type matches the return annotation",
    ),
    "weights_only=false": (
        "change to weights_only=True; run torch.load on a test checkpoint",
        "load succeeds without DeprecationWarning from weights_only",
    ),
    "use_hal=false": (
        "set use_hal=True in the config; run test_new_systems.py",
        "HAL tests pass and attention_temperature changes between turns",
    ),
    "frontier_dfc": (
        "run build_frontier_dataset.py; check all template counts >= minimum",
        "frontier_dfc.jsonl exists with >= 2000 examples, balanced templates",
    ),
    "kv_cache": (
        "enable kv cache in generate.py; measure tokens/sec before and after",
        "tokens/sec improves by >= 5x for 200-token generation",
    ),
    "hypothesis.py": (
        "create innovation/hypothesis.py; run run_innovation_cycle.py",
        "cycle runs without ImportError and produces a scored report",
    ),
}


def _match_verifier(description: str) -> tuple[str, str]:
    dl = description.lower()
    for kw, (verifier, falsifier) in _VERIFIER_MAP.items():
        if kw in dl:
            return verifier, falsifier
    return (
        f"implement the fix described in '{description[:60]}'; run pytest tests/ -x -q",
        "pytest passes with zero errors after the fix",
    )


def gap_to_hypothesis(gap: CapabilityGap) -> Hypothesis:
    verifier_path, falsifier = _match_verifier(gap.description)
    severity_delta = {
        "critical": {"test_pass_rate": +0.15, "capability_delta": +0.20},
        "moderate": {"test_pass_rate": +0.05, "capability_delta": +0.08},
        "minor": {"test_pass_rate": +0.02, "capability_delta": +0.02},
    }.get(gap.severity, {"test_pass_rate": +0.03, "capability_delta": +0.05})

    return Hypothesis(
        hyp_id=str(uuid.uuid4())[:8],
        gap_id=gap.gap_id,
        description=(
            f"Fixing '{gap.description}' in {gap.detected_in} "
            f"will improve system capability"
        ),
        falsifier=falsifier,
        predicted_delta=severity_delta,
        constraints=[
            "must not break existing passing tests",
            "must not change checkpoint format (no CausalTransformerV2 __init__ changes)",
            "must not add mandatory pip dependencies",
            "change must be <= 200 lines of diff",
            "history/ folder must not be touched",
        ],
        smallest_experiment=(
            f"Apply fix only to {gap.detected_in}. "
            f"Run: {verifier_path}. "
            f"Measure delta in test_pass_rate and capability_delta. "
            f"Roll back if either metric decreases."
        ),
        verifier_path=verifier_path,
        created_at=time.time(),
    )


def batch_hypotheses(gaps: list[CapabilityGap]) -> list[Hypothesis]:
    return [gap_to_hypothesis(g) for g in gaps]
