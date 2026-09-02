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

# Environment verification
python -V
nvidia-smi

# Execute remote trainer with explicit remote compute authorization
python -m senora.remote_launcher \
    --arm control-substrate-00 \
    --remote-authorized \
    --output-dir checkpoints/p35_control-substrate-00
