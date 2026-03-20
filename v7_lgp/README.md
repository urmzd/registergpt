# v7: True LGP — Differentiable Register Machine

**Status**: Implemented, ready for training.

## Architecture
- Shared operation bank (8 ops: identity, relu, gelu, square, negate, abs, tanh, sigmoid)
- Each instruction: soft register READ → soft OP SELECT → APPLY → soft WRITE
- Operation selection via learned logits (differentiable program)
- Cross-position via causal decay memory
- The model learns a PROGRAM, not just weights

## Key Differences
| Standard NN | v7 (LGP) |
|---|---|
| Fixed layer type | Selected from op bank |
| Dense matrix transform | Soft register addressing |
| Uniform architecture | Each step is a unique instruction |
| Weights = knowledge | Program + weights = knowledge |

## Usage
```bash
MODEL_VERSION=lgp torchrun --standalone --nproc_per_node=1 train.py
```

## Env Vars
| Variable | Default | Notes |
|----------|---------|-------|
| `NUM_STEPS` | 16 | Number of instructions in program |
| `N_CHANNELS` | 64 | Channel dim for operations |
| `N_OPS` | 8 | Operations in the bank |
