#!/bin/bash
#SBATCH --job-name=anra-p35-control-substrate-00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/p35_control-substrate-00_%j.out
#SBATCH --error=logs/p35_control-substrate-00_%j.err

set -euo pipefail

echo "Starting remote P35 scientific training on $(hostname) at $(date)"
echo "Arm: control-substrate-00"
echo "Tokens: 50000000"

# 1. Environment & GPU Verification
python -V
nvidia-smi

# 2. Execute Mandatory Remote Preflight Canary
echo "Running mandatory target canary..."
python -m senora.canary \
    --device cuda \
    --remote-authorized \
    --output logs/canary_control-substrate-00.json

# 3. Execute Real P35 Scientific Experiment
echo "Launching P35 experiment run..."
python -m senora.run_experiment \
    --experiment artifacts/v5/p35_cms1_plan.json \
    --arm control-substrate-00 \
    --execution-manifest cluster_jobs/manifest_control-substrate-00.json \
    --device cuda \
    --remote-authorized \
    --output-root output/p35_control-substrate-00
