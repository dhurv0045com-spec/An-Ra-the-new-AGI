"""P35-A dataset pair definition: matched control/treatment slices (M37-M40).

E3 Phase A fixes 200M-token arms at 5/15/30% cognition but does not define
the 0%-cognition control replacement (verified absent from the frozen
plan). This module freezes the policy: control preserves the 65:20
natural:code RATIO over 200M tokens (153M natural + 47M code/math),
treatment uses the E3 15% arm (130M + 40M + 30M verified cognition).
Absolute base tokens differ by design; ratios and total budget match, and
the choice is recorded, never silent (M38).

Treatment distinctness (M39): treatment semantic clusters must not intersect
evaluation clusters; the freeze checks overlap and fails closed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PAIR_SCHEMA = "anra-v5-p35a-dataset-pair/v1"
TOKEN_BUDGET = 200_000_000
TREATMENT_ALLOCATION = {"natural": 130_000_000, "code_math_formal": 40_000_000, "verified_cognition": 30_000_000}
CONTROL_ALLOCATION = {"natural": 153_000_000, "code_math_formal": 47_000_000}


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def control_replacement_policy() -> dict[str, object]:
    """Frozen M38 policy: ratio-preserving natural/code fill for cognition tokens."""

    return {
        "rule": "control consumes the full budget at the E3 natural:code ratio renormalized without cognition",
        "treatment_cognition_tokens": TREATMENT_ALLOCATION["verified_cognition"],
        "control_natural_tokens": CONTROL_ALLOCATION["natural"],
        "control_code_tokens": CONTROL_ALLOCATION["code_math_formal"],
        "ratio_preserved": "65:20",
    }


def check_cluster_disjointness(
    treatment_clusters: list[str], eval_clusters: list[str]
) -> dict[str, object]:
    """Fail closed when treatment training clusters intersect evaluation."""

    overlap = sorted(set(treatment_clusters) & set(eval_clusters))
    return {
        "treatment_clusters": len(set(treatment_clusters)),
        "eval_clusters": len(set(eval_clusters)),
        "overlap": overlap,
        "distinct": not overlap,
    }


def freeze_pair(
    *,
    base_manifests: Mapping[str, Any],
    cognition_generator: Mapping[str, Any],
    control_replacement: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Freeze the control/treatment pair definition plus DatasetPairSHA."""

    if not base_manifests:
        raise ValueError("pair freeze needs base manifests")
    replacement = dict(control_replacement or control_replacement_policy())
    pair = {
        "schema": PAIR_SCHEMA,
        "token_budget": TOKEN_BUDGET,
        "treatment": {
            "allocation": dict(TREATMENT_ALLOCATION),
            "base_manifests": dict(base_manifests),
            "cognition_generator": dict(cognition_generator),
        },
        "control": {
            "allocation": dict(CONTROL_ALLOCATION),
            "base_manifests": dict(base_manifests),
            "replacement_policy": replacement,
        },
        "control_replacement_policy": replacement,
    }
    pair["dataset_pair_sha256"] = _sha256_hex(_canonical_json(pair))
    return pair


def write_freeze(pair: Mapping[str, Any], path: Path) -> str:
    """Persist the frozen pair; return its DatasetPairSHA."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pair, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = _sha256_hex(_canonical_json(dict(pair)))
    return digest


__all__ = [
    "CONTROL_ALLOCATION",
    "PAIR_SCHEMA",
    "TOKEN_BUDGET",
    "TREATMENT_ALLOCATION",
    "check_cluster_disjointness",
    "control_replacement_policy",
    "freeze_pair",
    "write_freeze",
]
