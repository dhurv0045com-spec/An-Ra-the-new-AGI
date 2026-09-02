"""Headless launcher, remote execution packaging, and cluster dispatch generator for P35."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .experiment_design import build_p35_cms1_plan
from .run_experiment import EXECUTION_MANIFEST_SCHEMA, ExecutionManifest


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

# 1. Environment & GPU Verification
python -V
nvidia-smi

# 2. Execute Mandatory Remote Preflight Canary
echo "Running mandatory target canary..."
python -m senora.canary \\
    --device cuda \\
    --remote-authorized \\
    --output logs/canary_{arm_name}.json

# 3. Execute Real P35 Scientific Experiment
echo "Launching P35 experiment run..."
python -m senora.run_experiment \\
    --experiment artifacts/v5/p35_cms1_plan.json \\
    --arm {arm_name} \\
    --execution-manifest {manifest_relpath} \\
    --device cuda \\
    --remote-authorized \\
    --output-root output/p35_{arm_name}
"""


def generate_cluster_artifacts(output_dir: Path) -> dict[str, str]:
    """Generate verified job dispatch scripts and execution manifests for remote authorized compute."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_p35_cms1_plan()
    plan_sha = plan.sha256()

    created_files: dict[str, str] = {}
    for arm in plan.arms:
        arm_name = arm["name"]
        manifest_filename = f"manifest_{arm_name}.json"
        manifest_path = output_dir / manifest_filename

        # Generate cryptographic execution manifest
        manifest = ExecutionManifest(
            schema=EXECUTION_MANIFEST_SCHEMA,
            target_environment="remote-slurm-cuda",
            launch_nonce=f"launch-{uuid.uuid4().hex[:12]}",
            source_commit_sha="4a424fad1c21fa1ce8b5c47f636630ff92335e81",
            experiment_identity_sha256=plan_sha,
            authorized_by="cluster-orchestrator",
        )
        manifest.assert_valid()
        manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")

        script_content = SLURM_TEMPLATE.format(
            arm_name=arm_name,
            tokens=arm["token_budget"],
            manifest_relpath=str(Path(output_dir.name) / manifest_filename).replace("\\", "/"),
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
        print(f"Generated cluster scripts and manifests in {args.generate_cluster_scripts}:")
        for arm, path in files.items():
            print(f"  - {arm}: {path}")
        return 0

    if not args.remote_authorized:
        print(
            "ERROR: HARD COMPUTE CONSTRAINT ACTIVE.\n"
            "Scientific model training is strictly forbidden on the local machine.\n"
            "To generate remote submission scripts and execution manifests, run:\n"
            "  python -m senora.remote_launcher --generate-cluster-scripts ./cluster_jobs\n",
            file=sys.stderr,
        )
        return 1

    print(f"Target remote compute authorized. Initializing run for arm {args.arm}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())