#!/usr/bin/env bash
# Bootstrap exp-agi-models on RunPod from scratch
# Usage: curl -sSL https://raw.githubusercontent.com/urmzd/exp-agi-models/main/bootstrap.sh | bash
set -euo pipefail

cd /workspace

# Clone repo
[ -d exp-agi-models ] || git clone https://github.com/urmzd/exp-agi-models.git
cd exp-agi-models

# Install uv
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# Ensure uv is on PATH for this session and future shells
export PATH="$HOME/.local/bin:$PATH"
grep -q '.local/bin' ~/.bashrc 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Install Python deps (system pip for RunPod compatibility)
pip install -q huggingface_hub sentencepiece 2>/dev/null || \
    uv pip install --system huggingface_hub sentencepiece

# Download data
python data/download_data.py --variant sp1024

echo ""
echo "=== Ready ==="
echo "cd /workspace/exp-agi-models"
echo ""
echo "# Run v4 (101K params, associative memory):"
echo "MODEL_VERSION=v4 TRAIN_BATCH_TOKENS=491520 GRAD_ACCUM_STEPS=16 \\"
echo "TRAIN_LOG_EVERY=10 RUN_ID=v4_run \\"
echo "torchrun --standalone --nproc_per_node=\$(nvidia-smi -L | wc -l) train.py"
