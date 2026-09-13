#!/usr/bin/env python3
"""Deterministic static validator for the V5.1 Canary-v2 contract."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "experiments" / "V5_1_CANARY_V2" / "PREREGISTRATION.json"
RUNNER = ROOT / "anra_v5" / "v51_canary_v2_run.py"
TESTS = ROOT / "tests" / "test_v51_canary_v2.py"


def main() -> int:
    problems: list[str] = []
    for path in (P, RUNNER, TESTS):
        if not path.is_file():
            problems.append(f"missing required artifact: {path.relative_to(ROOT)}")
    if problems:
        for p in problems:
            print("ERROR:", p)
        return 1

    prereg = json.loads(P.read_text(encoding="utf-8"))
    if prereg.get("schema") != "anra-v51-canary-v2-preregistration/v1":
        problems.append("wrong preregistration schema")
    if prereg.get("parent_canary", {}).get("verdict") != "CANARY_FAIL_FORMATION":
        problems.append("V2 must descend from the V1 formation failure")
    if prereg.get("post_r1c_context", {}).get("verdict") != "SOFTMAX_COMPETITION_NOT_SUFFICIENT":
        problems.append("R1C consequence not bound")
    model = prereg.get("model", {})
    if model.get("parameter_count") != 10_227_456:
        problems.append("Rung-A parameter count drift")
    if model.get("output_path") != "tied full softmax, canonical only":
        problems.append("canonical output path drift")
    if model.get("experimental_output_treatments") != "FORBIDDEN":
        problems.append("experimental output treatments must be forbidden")
    training = prereg.get("training", {})
    if training.get("target_updates") != 360:
        problems.append("fixed update endpoint drift")
    if training.get("tokens_per_update") != 4096:
        problems.append("tokens/update drift")
    if training.get("token_budget") != 1_474_560:
        problems.append("token budget drift")
    if training.get("target_updates", 0) * training.get("tokens_per_update", 0) != training.get("token_budget"):
        problems.append("update endpoint and token budget disagree")
    if training.get("checkpoint_every") != 24:
        problems.append("checkpoint cadence drift")
    if prereg.get("seed") == 20260913:
        problems.append("V2 must not reuse the consumed V1 split seed")
    thresholds = prereg.get("evaluation", {}).get("v1_thresholds_retained_without_change")
    if thresholds != {
        "identity_dev_min": 0.30,
        "binding_dev_min": 0.25,
        "dev_overall_min": 0.15,
        "formation_min_any_family": 0.30,
    }:
        problems.append("V1 formation thresholds moved")

    source = RUNNER.read_text(encoding="utf-8")
    required_fragments = [
        'LINEAGE_ID = "v51-canary-v2"',
        'V51_CANARY_V2_ROOT',
        'backend.step(state, batch)',
        'base.certify_update',
        'base.CheckpointStore',
        'SEALED_CONSUMPTION.json',
        'CONSUMED_AND_FINALIZED',
        'epoch=epoch',
        'state.global_update % checkpoint_every == 0',
        'CANARY_V2_FAIL_FORMATION',
        'CANARY_V2_FAIL_ENGINEERING',
        'CANARY_V2_PASS',
    ]
    for fragment in required_fragments:
        if fragment not in source:
            problems.append(f"runner missing required contract fragment: {fragment}")
    if 'MASK_4096' in source:
        problems.append("runner contains prohibited MASK_4096 treatment")
    if 'rung="B"' in source:
        problems.append("runner hardcodes unauthorized Rung B")

    if problems:
        print(f"V5.1 CANARY-V2 VALIDATION FAILED ({len(problems)} problems)")
        for p in problems:
            print(" -", p)
        return 1
    print("V5.1 CANARY-V2 STATIC VALIDATION PASSED")
    print("fixed endpoint: 360 updates / 1,474,560 real tokens")
    print("output path: canonical tied full softmax")
    print("sealed policy: fresh + one-shot fail-closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
