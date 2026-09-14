"""Canonical FORMATION-MUX-001 Kaggle operator — Science S3, ops v6.

Engineering-only hardening over v5:
- preserves all worker/failure logs in the result bundle;
- applies a conservative 9.5h science-work ceiling inside Kaggle's 12h run,
  leaving >=2.5h for checkout, tokenizer/surface build, qualification,
  calibration, one-shot sealed evaluation, and packaging;
- records both the measured campaign projection and the operator safety gate.

No scientific treatment, seed, endpoint, threshold, or token/update budget is
changed here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import formation_mux_001_kaggle_operator_v2 as base
from tools import formation_mux_001_kaggle_operator_v5 as custody
from v5_experiments import formation_mux_protocol_v3 as proto

SCIENCE_COMMIT = custody.SCIENCE_COMMIT
OPS_SAFE_CAMPAIGN_MINUTES = 570.0  # 9.5h; leaves 2.5h of the 12h Kaggle ceiling
_orig_calibrate = base.calibrate


def calibrated_with_wall_reserve(*, repo: Path, surface: Path, out: Path):
    receipt = _orig_calibrate(repo, surface, out)
    projected = float(receipt["projected_full_campaign_minutes"])
    safe_partition = projected > OPS_SAFE_CAMPAIGN_MINUTES
    partition_minutes = projected / 2.0 if safe_partition else projected
    if safe_partition and partition_minutes > OPS_SAFE_CAMPAIGN_MINUTES:
        raise base.GlobalIntegrityError(
            f"measured full campaign projects {projected:.1f} min and a two-session "
            f"matched-bundle partition still projects {partition_minutes:.1f} min > "
            f"the ops-safe {OPS_SAFE_CAMPAIGN_MINUTES:.1f} min science ceiling; "
            "do not shorten or repartition the frozen matched science automatically"
        )
    receipt["original_protocol_partition_threshold_minutes"] = proto.PARTITION_THRESHOLD_MINUTES
    receipt["ops_safe_campaign_minutes"] = OPS_SAFE_CAMPAIGN_MINUTES
    receipt["reserved_noncampaign_minutes_within_12h"] = 720.0 - OPS_SAFE_CAMPAIGN_MINUTES
    receipt["partition_required"] = bool(receipt.get("partition_required") or safe_partition)
    receipt["projected_partition_minutes"] = partition_minutes
    receipt["wall_budget_policy"] = (
        "campaign projection must fit within 9.5h; remaining 2.5h is reserved for "
        "checkout/surface/qualification/calibration/sealed-finalization/packaging"
    )
    base._atomic_json(Path(out) / "CALIBRATION_RECEIPT.json", receipt)
    if receipt["partition_required"]:
        base._atomic_json(
            Path(out) / "PARTITION_PLAN.json",
            {
                "schema": "anra.formation-mux-partition-plan/v3",
                "science_commit": SCIENCE_COMMIT,
                "science_protocol_unchanged": True,
                "reason": "measured campaign does not fit the ops-safe single-session wall budget",
                "ops_safe_campaign_minutes": OPS_SAFE_CAMPAIGN_MINUTES,
                "projected_full_campaign_minutes": projected,
                "projected_partition_minutes": partition_minutes,
                "execution_policy": (
                    "each session selects at most one incomplete matched seed bundle per T4; "
                    "completed arms remain immutable; incomplete bundles remain pending"
                ),
            },
        )
    return receipt


# v4/v5 resolve calibration through this shared base module at runtime.
base.calibrate = calibrated_with_wall_reserve


def main(argv=None) -> int:
    return custody.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
