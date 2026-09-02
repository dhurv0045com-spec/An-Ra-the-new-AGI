#!/bin/bash
#SBATCH --job-name=anra-p35-cognition-mixture-15-ce
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/p35_cognition-mixture-15-ce_%j.out
#SBATCH --error=logs/p35_cognition-mixture-15-ce_%j.err

set -euo pipefail

echo "Starting remote P35 scientific training on $(hostname) at $(date)"
echo "Arm: cognition-mixture-15-ce"
echo "Tokens: 50000000"

# 1. Environment & GPU Verification
python -V
nvidia-smi

# 2. Execute Mandatory Remote Preflight Canary
echo "Running mandatory target canary..."
python -m senora.canary \
    --device cuda \
    --remote-authorized \
    --output logs/canary_cognition-mixture-15-ce.json

# 3. Execute Real P35 Scientific Experiment
echo "Launching P35 experiment run..."
python -m senora.run_experiment \
    --experiment artifacts/v5/p35_cms1_plan.json \
    --arm cognition-mixture-15-ce \
    --execution-manifest cluster_jobs/manifest_cognition-mixture-15-ce.json \
    --device cuda \
    --remote-authorized \
    --output-root output/p35_cognition-mixture-15-ce
