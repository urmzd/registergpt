# AGI Models

## Workflow

- **Don't change defaults** — use environment variables to override hyperparameters at runtime, not by editing default values in code.
- All model and training hyperparameters are configurable via env vars (see `Hyperparameters` class in `train.py`).
- The repo is self-contained: `train.py` includes all model definitions, `data/download_data.py` fetches the dataset.

## RunPod deployment

SSH config alias: `runpod` (configured in `~/.ssh/config`, key: `~/.ssh/runpod`).

Setup on a fresh pod:
```bash
cd /workspace && \
git clone https://github.com/urmzd/agi-models.git && \
cd agi-models && bash setup.sh
```

Run training:
```bash
cd /workspace/agi-models && \
TRAIN_BATCH_TOKENS=491520 \
GRAD_ACCUM_STEPS=16 \
TRAIN_LOG_EVERY=10 \
MODEL_VERSION=v3 \
RUN_ID=<name> \
torchrun --standalone \
--nproc_per_node=$(nvidia-smi -L | wc -l) \
train.py
```

Batch size must be divisible by `num_gpus * GRAD_ACCUM_STEPS * TRAIN_SEQ_LEN(1024)`.

## Key env vars

| Variable | Default | Notes |
|----------|---------|-------|
| `MODEL_VERSION` | v3 | Model to train: `v3`, `v4`, `gauss` |
| `NUM_STEPS` | 8 | Recurrent steps |
| `N_FOURIER_BASIS` | 16 | Fourier basis / freq count |
| `N_CHANNELS` | 128 | Channel dim for memory/register ops |
| `LR` | 0.03 | Adam learning rate |
| `DECAY_INIT` | 3.0 | Memory decay logit (sigmoid→0.95) |
| `GRAD_ACCUM_STEPS` | 16 | Gradient accumulation |
| `TRAIN_BATCH_TOKENS` | 524288 | Global batch size in tokens |
| `TORCH_COMPILE` | 0 | Enable torch.compile (slow warmup) |

### v4-specific

| Variable | Default | Notes |
|----------|---------|-------|
| `N_HEADS` | 4 | Multi-head decay heads |
| `TRANSFORM_RANK` | 8 | Low-rank factorization rank |
| `UNIQUE_STEPS` | 5 | Unique step definitions |
| `INVOCATIONS_PER_STEP` | 2 | Times each step is reused |
