#!/usr/bin/env bash
# Setup exp-agi-models on a fresh machine (e.g. RunPod)
# Usage: bash setup.sh
set -euo pipefail

cd /workspace

# Clone if needed
[ -d exp-agi-models ] || git clone https://github.com/urmzd/exp-agi-models.git
cd exp-agi-models

# Install deps
pip install huggingface_hub sentencepiece 2>/dev/null || true

# Download data
python data/download_data.py --variant sp1024

echo "Setup complete. Run training with:"
echo "  cd /workspace/exp-agi-models"
echo "  MODEL_VERSION=v3 torchrun --standalone --nproc_per_node=\$(nvidia-smi -L | wc -l) train.py"
