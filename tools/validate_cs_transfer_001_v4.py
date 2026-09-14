#!/usr/bin/env python3
"""Static fail-closed validation for CS-TRANSFER-001 Amendment 2 / runner v4."""
from __future__ import annotations

import json
from pathlib import Path

from anra_v5 import cs_transfer_001_data_v2 as data
from anra_v5 import cs_transfer_001_model as model

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "experiments/CS_TRANSFER_001/PREREGISTRATION.json"
A1 = ROOT / "experiments/CS_TRANSFER_001/AMENDMENT_1.json"
A2 = ROOT / "experiments/CS_TRANSFER_001/AMENDMENT_2.json"


class _Identity:
    vocabulary_size = 24576
    special_token_ids = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}


class _Tokenizer:
    identity = _Identity()


def main() -> int:
    p = json.loads(PREREG.read_text())
    a1 = json.loads(A1.read_text())
    a2 = json.loads(A2.read_text())
    assert p["experiment"] == "CS-TRANSFER-001"
    assert a1["status"] == a2["status"] == "PROSPECTIVE_PREEXECUTION"
    assert a1["outcomes_observed_before_amendment"] is False
    assert a2["outcomes_observed_before_amendment"] is False
    assert a2["evidence"]["gpu_training_started"] is False
    assert p["training"]["target_updates"] == 480
    assert p["training"]["token_budget"] == 1_966_080
    assert model.spec_for(4096).parameter_receipt().total == 8_920_320
    assert model.spec_for(24576).parameter_receipt().total == 14_163_200
    assert min(data.GRAMMAR_IDS) >= 4 and max(data.GRAMMAR_IDS) < 4096
    surface = data.build_shared_surface(
        tokenizer=_Tokenizer(),
        seed=int(p["data"]["fresh_seed"]),
        candidate_worlds_per_family=int(p["data"]["candidate_worlds_per_family"]),
        select_counts={k: int(v) for k, v in p["data"]["selected_rows_per_family"].items()},
    )
    manifest = surface["manifest"]
    assert manifest["surface_revision"] == "A2_DIRECT_TOKEN_COMMON_SPACE"
    assert manifest["contamination"]["clean"] is True
    assert manifest["predictive_shortcut_max"] < 0.35
    assert all(x["all_lt_4096"] for x in manifest["token_stats"].values())
    print(json.dumps({
        "status": "PASS",
        "surface_revision": manifest["surface_revision"],
        "manifest_sha256": surface["manifest_sha256"],
        "predictive_shortcut_max": manifest["predictive_shortcut_max"],
        "rows": {k: len(v) for k, v in surface["rows"].items()},
        "max_content_id": max(x["maximum_content_id"] for x in manifest["token_stats"].values()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
