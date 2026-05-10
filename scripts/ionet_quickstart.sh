#!/bin/bash
# chmod +x scripts/ionet_quickstart.sh
set -e

echo "AN-RA QUICKSTART FOR IO.NET"
echo "==========================="

# 1. Clone repo
cd /workspace
if [ ! -d "An-Ra-the-new-AGI" ]; then
    git clone https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git
else
    cd An-Ra-the-new-AGI && git pull && cd ..
fi
cd An-Ra-the-new-AGI

# 2. Install deps
echo "Installing dependencies..."
pip install -r requirements.txt -q
pip install datasets boto3 psutil -q

# 3. GPU check
python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB')
"

# 4. Download data + train
# Edit SESSION_MINUTES here:
SESSION_MINUTES=${SESSION_MINUTES:-150}
MODEL_SIZE=${MODEL_SIZE:-1b}

echo "Starting training: $MODEL_SIZE, $SESSION_MINUTES minutes"

python scripts/train_oneshot.py \
    --model-size $MODEL_SIZE \
    --session-minutes $SESSION_MINUTES \
    2>&1 | tee /workspace/anra_training.log

echo "Done. Log: /workspace/anra_training.log"
