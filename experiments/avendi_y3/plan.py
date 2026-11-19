"""Expand the Avendi Y3 preregistration. Running this file only prints the plan."""

from __future__ import annotations

import json


SEEDS = (7, 19, 41)
ARMS = {
    "P2": ("ce", "margin"),
    "P3": ("refresh", "curriculum", "address", "address_sham",
           "rename", "rename_sham", "low_lr"),
    "P4": ("categorical", "continuous", "dense"),
    "P5": ("direct_plus_neutral", "direct_plus_info", "replace_direct"),
    "P6": ("diagnostic_policy",),
    "P7": ("s20_ce", "s20_low_lr"),
}
STUDIES = {
    "P0": range(1, 7), "P1": range(7, 15), "P2": range(15, 21),
    "P3": range(21, 28), "P4": range(28, 36), "P5": range(36, 41),
    "P6": range(41, 47), "P7": range(47, 51),
}
BASE_REVISION = "ba1c0e978f890021a10e2365b61d5403bc486ce5"


def build_plan() -> dict:
    jobs = []
    for stage, arms in ARMS.items():
        for arm in arms:
            for seed in SEEDS:
                job_id = f"{stage}-{arm}-s{seed}"
                jobs.append({
                    "job_id": job_id, "stage": stage, "arm": arm, "init_seed": seed,
                    "status": "PLANNED", "receipt": f"jobs/{job_id}/receipt.json",
                })
    studies = []
    for stage, numbers in STUDIES.items():
        for number in numbers:
            study_id = f"U{number:02}"
            studies.append({
                "study_id": study_id, "home_stage": stage, "status": "PLANNED",
                "receipt_pattern": f"studies/{study_id}/<job>/<split>/receipt.json",
            })
    return {
        "campaign": "Avendi V8-Y3", "technology_baseline": BASE_REVISION,
        "phase": "DESIGN", "training_jobs": jobs, "studies": studies,
        "frozen_lineages": [f"Y1-t1_factorial-S3-{arm}-s{seed}"
                            for arm in ("F00", "F11") for seed in SEEDS],
        "reused_scale_controls": [f"{stage}-{arm}-s{seed}"
                                  for stage, arm in (("P2", "ce"), ("P3", "low_lr"))
                                  for seed in SEEDS],
    }


if __name__ == "__main__":
    print(json.dumps(build_plan(), indent=2))
