# AGI Models — Agent Guidelines

## Principles

- **Don't change defaults** — use environment variables to override hyperparameters at runtime, not by editing default values in code.
- **Self-contained repo** — `train.py` is the single entry point for training all models. Shared infrastructure lives in `core/` (config, data loading, eval, quantization, model registry). Model definitions live in their own directories (`v9_meta_state/model.py`, etc.) and are registered in `core/registry.py`.
- **No embedding, no output projection** — every model operates in vocabulary space. Input is one-hot, output is the register state. Do not add embedding layers or output projections.
- **Environment variables for everything** — all hyperparameters live in the `Hyperparameters` class in `core/config.py` and are read from env vars. When adding a new model, add its specific env vars there with sensible defaults.

## Adding a new model version

1. Create a directory: `vN_descriptive_name/`
2. Add `__init__.py` and `model.py` with a single model class
3. The model class must implement `forward(input_ids: Tensor, target_ids: Tensor) -> Tensor` returning the loss
4. Add an entry to `REGISTRY` in `core/registry.py` with module path, class name, and kwargs function
5. Add any new env vars to the appropriate config class in `core/config.py`
6. Add any new control tensor name patterns to `CONTROL_TENSOR_NAME_PATTERNS` in `core/quantize.py` (these stay in fp32 during bfloat16 training)
7. Add the model to the `MODELS` list in `run_all.py`
8. Update `README.md` — add a row to the architecture table and a line to the evolution of ideas
9. Update `TODO.md` if relevant

## Control tensor patterns

Parameters matching patterns in `CONTROL_TENSOR_NAME_PATTERNS` are kept in float32 even when the model is cast to bfloat16. This includes: scales, biases, decay logits, gating parameters, and small learned scalars. When adding a new model, ensure any scalar/gate/scale parameters have names matching existing patterns or add new patterns.

## Training conventions

- All models train via `torchrun --standalone --nproc_per_node=N train.py`
- Multi-GPU via PyTorch DDP — batch size must be divisible by `num_gpus * GRAD_ACCUM_STEPS * TRAIN_SEQ_LEN(1024)`
- Mixed precision: bfloat16 for weights, float32 for control tensors, autocast during forward
- Models are initialized in float32, cast to bfloat16, then control tensors converted back to float32
- The `.float()` calls inside model forward methods are intentional — they upcast for numerical stability before projections

## RunPod deployment

SSH config alias: `runpod` (configured in `~/.ssh/config`, key: `~/.ssh/runpod`).

Setup on a fresh pod:
```bash
cd /workspace && \
git clone https://github.com/urmzd/exp-agi-models.git && \
cd exp-agi-models && bash setup.sh
```

Run training:
```bash
cd /workspace/exp-agi-models && \
TRAIN_BATCH_TOKENS=491520 \
GRAD_ACCUM_STEPS=16 \
TRAIN_LOG_EVERY=10 \
MODEL_VERSION=meta \
RUN_ID=<name> \
torchrun --standalone \
--nproc_per_node=$(nvidia-smi -L | wc -l) \
train.py
```

## Key env vars

| Variable | Default | Notes |
|----------|---------|-------|
| `MODEL_VERSION` | v3_assoc | Which model to train (see README for full list) |
| `NUM_STEPS` | 8 | Recurrent steps / depth |
| `STATE_DIM` | 64 | State space dimension (v9+) |
| `INNER_DIM` | 128 | Inner MLP dimension (v9+) |
| `N_FOURIER_BASIS` | 16 | Fourier basis count (v1-v6) |
| `N_CHANNELS` | 128 | Channel dim (v1-v6) |
| `N_OPS` | 8 | Op bank size (v7, v10) |
| `K_ACTIVE` | 256 | Active registers (v12) |
| `GUMBEL_TAU` | 1.0 | Gumbel temperature (v11b tpg) |
| `HALT_THRESHOLD` | 0.5 | Early-exit threshold (v11b tpg) |
| `PONDER_LAMBDA` | 0.01 | Ponder regularization (v11b tpg) |
| `LR` | 0.03 | Adam learning rate |
| `DECAY_INIT` | 3.0 | Memory decay logit |
| `GRAD_ACCUM_STEPS` | 16 | Gradient accumulation |
| `TRAIN_BATCH_TOKENS` | 524288 | Global batch size in tokens |
| `MAX_WALLCLOCK_SECONDS` | None | Wall-clock time limit (must be set manually, no default) |
| `ITERATIONS` | 500 | Max training iterations |
| `TORCH_COMPILE` | 0 | Enable torch.compile |
| `ROUNDTRIP_EVAL` | 0 | Run int8 quantization roundtrip eval after training |
| `NCCL_P2P_DISABLE` | 1 | Disable NCCL P2P; required on RunPod where GPUs span PCIe root complexes |

## File conventions

- `core/config.py` — all hyperparameter classes (`Hyperparameters`, `BaseSettings` subclasses)
- `core/data.py` — data loading (`TokenStream`, `DistributedTokenLoader`)
- `core/eval.py` — validation (`eval_val`, `build_sentencepiece_luts`)
- `core/quantize.py` — int8 quantization/dequantization, `CONTROL_TENSOR_NAME_PATTERNS`
- `core/registry.py` — model registry (`REGISTRY` dict, `build_model()`)
- `train.py` — training loop, DDP setup, checkpointing, serialization
- `benchmark.py` — synthetic data benchmarking (no GPU needed), tests all models
- `run_all.py` — sequential training of all models, results collection
- `results.py` — reads `logs/*_manifest.json` and prints a results table
- `tests/` — pytest tests for config, registry, quantization, and models
- Model directories are named `vN_descriptive_name/`
- Each model directory contains `__init__.py` and `model.py`
- Research notes go in `docs/`

## MODEL_VERSION values

| `MODEL_VERSION` | Directory | Description |
|-----------------|-----------|-------------|
| `v1_attention` | `v1_shared_attention/` | Shared attention + Fourier ops |
| `v2_conv` | `v2_causal_conv/` | Depthwise conv + Fourier ops |
| `v3_assoc` | `v3_assoc_memory/` | Associative memory + Fourier ops |
| `v4_golf` | `v4_param_optimized/` | 101K param golf |
| `v5_gauss` | `v5_gauss_fft/` | FFT-based ops |
| `v6_wave` | `v6_brain_wave/` | Oscillatory dynamics |
| `v7_lgp` | `v7_lgp/` | Differentiable register machine |
| `v8_graph` | `v8_word_graph/` | Word interaction graph (best at rank 8) |
| `v9_meta` | `v9_meta_state/` | Evolving Q-table |
| `v10_policy` | `v10_policy/` | State-dependent policy |
| `v11_brainwave` | `v11_brainwave/` | Oscillatory primitives |
| `v11_tpg` | `v11_tpg/` | Neural TPG |
| `v12_sparse` | `v12_sparse_register/` | Sparse register addressing |
| `v13_embed` | `v13_sparse_embed/` | Sparse + factored embedding |
| `v14_adaptive` | `v14_adaptive/` | Data-dependent decay (Mamba-inspired) |
| `v15_predictive` | `v15_predictive/` | Per-step aux losses (predictive coding) |
| `v16_columnar` | `v16_columnar/` | Multi-column voting (Thousand Brains) |
