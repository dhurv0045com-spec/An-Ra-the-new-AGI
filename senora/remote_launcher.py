"""Headless launcher, remote execution packaging, and cluster dispatch templates for P35."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .experiment_design import build_p35_cms1_plan
from .trainer import LocalScientificComputeConstraintError


SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=anra-p35-{arm_name}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/p35_{arm_name}_%j.out
#SBATCH --error=logs/p35_{arm_name}_%j.err

set -euo pipefail

echo "Starting remote P35 scientific training on $(hostname) at $(date)"
echo "Arm: {arm_name}"
echo "Tokens: {tokens}"

# Environment verification
python -V
nvidia-smi

# Execute remote trainer with explicit remote compute authorization
python -m senora.remote_launcher \\
    --arm {arm_name} \\
    --remote-authorized \\
    --output-dir checkpoints/p35_{arm_name}
"""


def generate_cluster_artifacts(output_dir: Path) -> dict[str, str]:
    """Generate job dispatch scripts for remote authorized compute."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_p35_cms1_plan()

    created_files: dict[str, str] = {}
    for arm in plan.arms:
        arm_name = arm["name"]
        script_content = SLURM_TEMPLATE.format(
            arm_name=arm_name,
            tokens=arm["token_budget"],
        )
        script_path = output_dir / f"run_{arm_name}.sbatch"
        script_path.write_text(script_content, encoding="utf-8")
        created_files[arm_name] = str(script_path)

    return created_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Senora P35 Remote Execution Launcher")
    parser.add_argument("--arm", type=str, default="control-substrate-00", help="Arm name to launch")
    parser.add_argument("--remote-authorized", action="store_true", help="Explicitly declare target remote compute authorization")
    parser.add_argument("--generate-cluster-scripts", type=Path, help="Directory to emit SLURM cluster submission scripts")
    args = parser.parse_args()

    if args.generate_cluster_scripts:
        files = generate_cluster_artifacts(args.generate_cluster_scripts)
        print(f"Generated cluster scripts in {args.generate_cluster_scripts}:")
        for arm, path in files.items():
            print(f"  - {arm}: {path}")
        return 0

    if not args.remote_authorized:
        print(
            "ERROR: HARD COMPUTE CONSTRAINT ACTIVE.\n"
            "Scientific model training is strictly forbidden on the local machine.\n"
            "To generate remote submission scripts, run:\n"
            "  python -m senora.remote_launcher --generate-cluster-scripts ./cluster_jobs\n",
            file=sys.stderr,
        )
        return 1

    print(f"Target remote compute authorized. Initializing run for arm {args.arm}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())