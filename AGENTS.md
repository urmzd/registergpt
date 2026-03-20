# RegisterGPT

## Workflow

- **Don't change defaults** — use environment variables to override hyperparameters at runtime, not by editing default values in code.
- All model and training hyperparameters are configurable via env vars (see `Hyperparameters` class in `train.py`).

## RunPod deployment

SSH config alias: `runpod` (configured in `~/.ssh/config`, key: `~/.ssh/runpod`).

Setup on a fresh pod:
```bash
cd /workspace && \
git clone https://github.com/urmzd/registergpt.git && \
git clone https://github.com/urmzd/parameter-golf.git && \
cd parameter-golf && pip install huggingface_hub sentencepiece && \
python data/cached_challenge_fineweb.py --variant sp1024
```

Run training:
```bash
cp /workspace/registergpt/train.py /workspace/parameter-golf/train_registergpt.py && \
cd /workspace/parameter-golf && \
TRAIN_BATCH_TOKENS=491520 \
GRAD_ACCUM_STEPS=16 \
TRAIN_LOG_EVERY=10 \
RUN_ID=<name> \
torchrun --standalone \
--nproc_per_node=$(nvidia-smi -L | wc -l) \
train_registergpt.py
```

Batch size must be divisible by `num_gpus * GRAD_ACCUM_STEPS * TRAIN_SEQ_LEN(1024)`.

## Key env vars

| Variable | Default | Notes |
|----------|---------|-------|
| `NUM_STEPS` | 8 | Recurrent steps |
| `N_FOURIER_BASIS` | 16 | Fourier basis functions (increase for higher-rank projections) |
| `N_CHANNELS` | 128 | Channel dim for memory/register ops |
| `LR` | 0.03 | Adam learning rate |
| `DECAY_INIT` | 3.0 | Memory decay logit (sigmoid→0.95) |
| `GRAD_ACCUM_STEPS` | 16 | Gradient accumulation |
| `TRAIN_BATCH_TOKENS` | 524288 | Global batch size in tokens |
| `TORCH_COMPILE` | 0 | Enable torch.compile (slow warmup) |
