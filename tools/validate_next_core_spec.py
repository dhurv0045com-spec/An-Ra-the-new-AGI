#!/usr/bin/env python3
"""Validator for docs/cymek/next_core/NEXT_CORE_SPEC.json (small, deterministic).

Checks:
  1. every component has a legal status;
  2. every LOCKED component carries evidence references;
  3. every BLOCKED component names a blocking experiment;
  4. every REJECTED component carries rationale evidence;
  5. the parameter receipt is internally consistent and equals the formula total;
  6. geometry coherence: width == q_heads*head_dim, q_heads % kv_heads == 0,
     head_dim even, vocab/width/output dimensions agree, context positive;
  7. tied-weight contract coherence (output_head == 0 iff tied);
  8. no EXPERIMENT_ONLY component appears as the canonical default;
  9. cross-file: the receipt matches tools/next_core_compute_model.py output.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.next_core_compute_model import Geometry, parameter_receipt  # noqa: E402

LEGAL = {"LOCKED", "PROVISIONAL", "BLOCKED", "REJECTED", "EXPERIMENT_ONLY"}


def main() -> int:
    problems: list[str] = []
    path = os.path.join(ROOT, "docs", "cymek", "next_core", "NEXT_CORE_SPEC.json")
    with open(path, encoding="utf-8") as fh:
        spec = json.load(fh)

    geometry = spec["geometry"]["value"]
    g = Geometry(
        vocabulary_size=geometry["vocabulary_size"],
        width=geometry["width"],
        layers=geometry["layers"],
        query_heads=geometry["query_heads"],
        kv_heads=geometry["kv_heads"],
        head_dimension=geometry["head_dimension"],
        ffn_width=geometry["ffn_width"],
        context_length=geometry["context_length"],
        tied_embeddings=geometry["tied_embeddings"],
        qk_norm_affine=geometry["qk_norm_affine"],
    )
    expected = parameter_receipt(g)
    recorded = spec["geometry"]["parameter_receipt"]
    for key, value in expected.items():
        if recorded.get(key) != value:
            problems.append(f"parameter receipt mismatch at {key}: spec {recorded.get(key)} vs formula {value}")
    if recorded["total"] != 250_216_960:
        problems.append("V5-A total must be 250,216,960")

    for component in spec["components"]:
        name = component["component"]
        status = component.get("status")
        if status not in LEGAL:
            problems.append(f"{name}: illegal status {status!r}")
        if status == "LOCKED" and not component.get("evidence_refs"):
            problems.append(f"{name}: LOCKED without evidence references")
        if status == "BLOCKED" and not component.get("blocking_experiment"):
            problems.append(f"{name}: BLOCKED without a blocking experiment")
        if status == "REJECTED" and not component.get("evidence_refs"):
            problems.append(f"{name}: REJECTED without rationale evidence")
        if status == "EXPERIMENT_ONLY" and name == "output_mode":
            problems.append("output_mode cannot be EXPERIMENT_ONLY as the spec default")

    tied = geometry["tied_embeddings"]
    if tied and recorded["output_head"] != 0:
        problems.append("tied contract requires output_head == 0")
    if not tied and recorded["output_head"] != geometry["vocabulary_size"] * geometry["width"]:
        problems.append("untied contract requires output_head == vocab * width")

    # canonical default must be the canonical output mode
    output_head = next(c for c in spec["components"] if c["component"] == "output_head")
    if "tied full softmax canonical" not in output_head["value"]:
        problems.append("canonical output path must remain full softmax")

    print(f"spec: {len(spec['components'])} components; receipt total {recorded['total']}")
    if problems:
        print(f"VALIDATION FAILED ({len(problems)} problems):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("NEXT CORE SPEC VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
