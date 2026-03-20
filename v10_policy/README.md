# v10: Policy Register Machine — State-Dependent Action Selection

**Status**: Implemented, ready for training.

## Core Insight
Instead of fixed instructions (v7) or memorizing all relationships (v9),
learn a **policy** π(action | state). The same step can execute different
operations on different inputs — data-dependent branching.

## Architecture
```
for each step:
    state = compress(registers)        # V → d
    read_w, op_w, write_w = π(state)   # policy decides action
    selected = state * read_w          # READ (what matters now?)
    result = op_bank(selected, op_w)   # OP (what to do?)
    output = result * write_w          # WRITE (where to store?)
    registers += expand(output)        # d → V
```

Cross-position: causal decay memory in state space (like v9).

## Key Differences
| v7 (LGP) | v10 (Policy) |
|---|---|
| Fixed program | State-dependent actions |
| Same read/write every input | Adaptive read/write |
| Assembly language | Conditional execution |
| Open-loop control | Closed-loop control |

## Usage
```bash
MODEL_VERSION=policy TRAIN_BATCH_TOKENS=491520 GRAD_ACCUM_STEPS=16 \
TRAIN_LOG_EVERY=10 RUN_ID=policy_test \
torchrun --standalone --nproc_per_node=3 train.py
```

## Env Vars
| Variable | Default | Notes |
|----------|---------|-------|
| `STATE_DIM` | 64 | Compressed state dimension |
| `N_OPS` | 8 | Operations in the bank |
| `NUM_STEPS` | 8 | Policy execution steps |
