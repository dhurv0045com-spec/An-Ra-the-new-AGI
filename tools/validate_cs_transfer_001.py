#!/usr/bin/env python3
"""Static protocol validator for CS-TRANSFER-001."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from anra_v5 import cs_transfer_001_data as data
from tools.next_core_compute_model import Geometry, parameter_receipt

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "experiments" / "CS_TRANSFER_001" / "PREREGISTRATION.json"
AMENDMENT = ROOT / "experiments" / "CS_TRANSFER_001" / "AMENDMENT_1.json"
ENCODING = ROOT / "artifacts" / "e1" / "local_tournament" / "encoding-24576.json"


def main() -> int:
    p = json.loads(PREREG.read_text(encoding="utf-8"))
    a = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    e = json.loads(ENCODING.read_text(encoding="utf-8"))
    assert p["schema"] == "anra-cs-transfer-001-preregistration/v1"
    assert a["schema"] == "anra-cs-transfer-001-amendment/v1"
    assert a["status"] == "PROSPECTIVE_PREEXECUTION"
    assert a["outcomes_observed_before_amendment"] is False
    assert p["parent_evidence"]["r1c_verdict"] == "SOFTMAX_COMPETITION_NOT_SUFFICIENT"
    assert p["parent_evidence"]["canary_v2_verdict"] == "CANARY_V2_FAIL_FORMATION"
    assert p["training"]["target_updates"] == 480
    assert p["training"]["token_budget"] == 480 * 4096 == 1_966_080
    assert p["training"]["development_eval_updates"] == [0, 60, 120, 180, 240, 300, 360, 420, 480]
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

    # Amendment-1 evidence: original literal Answer delimiter is impossible
    # under the shared <4096 surface, while newline is observed low-ID.
    answer_probe = next(x for x in e["encodings"] if x["probe_id"] == "answer-01")
    assert 18224 in answer_probe["token_ids"] and 18224 >= 4096
    assert 202 in answer_probe["token_ids"] and 202 < 4096
    assert a["change"]["prompt_suffix_after"] == data.PROMPT_SUFFIX == "\n"
    assert a["change"]["answer_prefix_after"] == data.ANSWER_PREFIX == ""

    digest = hashlib.sha256()
    digest.update(PREREG.read_bytes())
    digest.update(b"\x00CS_TRANSFER_001_AMENDMENT_1\x00")
    digest.update(AMENDMENT.read_bytes())
    protocol_sha = digest.hexdigest()

    print(json.dumps({
        "status": "PASS",
        "experiment": p["experiment"],
        "effective_protocol_sha256": protocol_sha,
        "amendment": 1,
        "arms": p["model"]["arms"],
        "matched_pairs": len(p["matching"]["model_seeds"]),
        "endpoint_updates": p["training"]["target_updates"],
        "token_budget": p["training"]["token_budget"],
        "effective_prompt_suffix_repr": repr(data.PROMPT_SUFFIX),
        "effective_answer_prefix_repr": repr(data.ANSWER_PREFIX),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
