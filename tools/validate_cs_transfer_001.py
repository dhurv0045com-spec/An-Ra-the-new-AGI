#!/usr/bin/env python3
"""Static preregistration validator for CS-TRANSFER-001."""
from __future__ import annotations

import json
from pathlib import Path

from tools.next_core_compute_model import Geometry, parameter_receipt

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "experiments" / "CS_TRANSFER_001" / "PREREGISTRATION.json"


def main() -> int:
    p = json.loads(PREREG.read_text(encoding="utf-8"))
    assert p["schema"] == "anra-cs-transfer-001-preregistration/v1"
    assert p["parent_evidence"]["r1c_verdict"] == "SOFTMAX_COMPETITION_NOT_SUFFICIENT"
    assert p["parent_evidence"]["canary_v2_verdict"] == "CANARY_V2_FAIL_FORMATION"
    assert p["training"]["target_updates"] == 480
    assert p["training"]["token_budget"] == 480 * 4096 == 1_966_080
    assert p["training"]["development_eval_updates"][-1] == 480
    assert len(p["matching"]["model_seeds"]) == len(p["matching"]["order_seeds"]) == 4
    assert p["causal_contrast"]["not_a_masking_proxy"] is True
    common = dict(p["model"]["geometry_except_vocab"])
    for arm, arm_cfg in p["model"]["arms"].items():
        g = Geometry(vocabulary_size=int(arm_cfg["vocabulary_size"]), **common)
        actual = parameter_receipt(g)["total"]
        assert actual == int(arm_cfg["expected_parameters"]), (arm, actual)
    assert p["model"]["arms"]["PHYS_4096"]["vocabulary_size"] == 4096
    assert p["model"]["arms"]["PHYS_24576"]["vocabulary_size"] == 24576
    assert p["data"]["rendering"]["max_common_token_id"] == 4095
    assert p["evaluation"]["primary_family"] == "identity"
    assert "500M" in p["claim_ceiling"]
    print(json.dumps({
        "status": "PASS",
        "experiment": p["experiment"],
        "arms": p["model"]["arms"],
        "matched_pairs": len(p["matching"]["model_seeds"]),
        "endpoint_updates": p["training"]["target_updates"],
        "token_budget": p["training"]["token_budget"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
