"""Canonical FORMATION-MUX operator v12.

Engineering hardening over v11: after the dual-T4 frontier qualifier creates
real Adam/checkpoint state, measure its largest resume.pt and fail closed unless
all remaining frontier checkpoint slots plus a 2 GiB reserve fit the current
saved Kaggle working volume. Science, arms, exposures and verdicts are unchanged.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import formation_mux_001_kaggle_operator_v10 as v10
from tools import formation_mux_001_kaggle_operator_v11 as v11
from v5_experiments import tie_role_protocol_v1 as frontier

OPERATOR_NAME = "tools/formation_mux_001_kaggle_operator_v12.py"
STORAGE_RESERVE_BYTES = 2 * 1024 ** 3
_ORIGINAL_QUALIFY_FRONTIER = v10.qualify_frontier


def frontier_storage_preflight(out: Path) -> dict[str, Any]:
    target = Path("/kaggle/working") if Path("/kaggle/working").exists() else out.parent
    usage = shutil.disk_usage(target)
    calibration_ckpts = list((out / "qualification_tie_role").rglob("resume.pt"))
    if not calibration_ckpts:
        raise v10.v9.v8.v7.base.GlobalIntegrityError(
            "frontier storage preflight cannot find real calibration checkpoints"
        )
    max_checkpoint_bytes = max(p.stat().st_size for p in calibration_ckpts)
    existing_official = 0
    for experiment, arms in (
        (frontier.EXPERIMENT_A, frontier.ARMS_A),
        (frontier.EXPERIMENT_B, frontier.ARMS_B),
    ):
        for arm in arms:
            for index, _bundle in enumerate(frontier.SEED_BUNDLES, start=1):
                if (out / experiment / arm / f"S{index}" / "resume.pt").exists():
                    existing_official += 1
    remaining_slots = max(frontier.total_official_arms() - existing_official, 0)
    projected_additional = remaining_slots * max_checkpoint_bytes
    required_free = projected_additional + STORAGE_RESERVE_BYTES
    receipt = {
        "schema": "anra.tie-role-storage-preflight/v1",
        "extension": frontier.EXTENSION,
        "path": str(target),
        "disk_total_bytes": usage.total,
        "disk_used_bytes": usage.used,
        "disk_free_bytes": usage.free,
        "largest_frontier_calibration_checkpoint_bytes": max_checkpoint_bytes,
        "frontier_checkpoint_slots_existing": existing_official,
        "frontier_checkpoint_slots_remaining": remaining_slots,
        "projected_additional_checkpoint_bytes": projected_additional,
        "reserve_bytes": STORAGE_RESERVE_BYTES,
        "required_free_bytes": required_free,
        "pass": usage.free >= required_free,
        "fail_closed_before_frontier_official_arms": True,
    }
    v10._atomic_json(out / "TIE_ROLE_STORAGE_PREFLIGHT.json", receipt)
    if not receipt["pass"]:
        raise v10.v9.v8.v7.base.GlobalIntegrityError(
            "insufficient saved Kaggle working space for remaining frontier exact-resume "
            f"checkpoints: free={usage.free} required={required_free}"
        )
    return receipt


def qualify_frontier_with_storage(repo: Path, public_path: Path, out: Path) -> dict[str, Any]:
    receipt = _ORIGINAL_QUALIFY_FRONTIER(repo, public_path, out)
    storage = frontier_storage_preflight(out)
    receipt = dict(receipt)
    receipt["frontier_storage_preflight"] = storage
    receipt["status"] = (
        "CPU_STATIC_PASS / GPU_E2E_PASS_ENGINEERING_ONLY / FRONTIER_STORAGE_PASS"
    )
    v10._atomic_json(out / v10.FRONTIER_QUALIFICATION, receipt)
    return receipt


def _bind() -> None:
    v11._bind()
    v10.OPERATOR_NAME = OPERATOR_NAME
    v10.qualify_frontier = qualify_frontier_with_storage


def main(argv=None) -> int:
    _bind()
    return v10.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
