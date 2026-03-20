# v6: Brain Wave — Oscillatory Dynamics as Computation

**Status**: Implemented, ready for training.

## Core Thesis
Computation IS wave interference, not neurons firing. Multiple frequency bands with cross-frequency coupling encode and transform information through phase relationships and amplitude modulation.

## Architecture
- Fourier basis partitioned into 3 frequency bands (low/mid/high)
- Band-specific register ops (within-position transforms)
- Band-specific associative memories with different decay rates (cross-position)
- **Alpha coupling**: low band gates high band (suppression of irrelevant detail)
- **Theta-gamma coupling**: slow memory gates fast memory retrieval
- No attention. No embedding. No output projection.

## Key Differences from v3
| v3 (Assoc Memory) | v6 (Brain Wave) |
|---|---|
| All frequencies treated equally | Frequencies partitioned into bands |
| One memory per step | Two memories per cycle (slow + fast) |
| One register op per step | Three register ops per cycle (low/mid/high) |
| No cross-frequency interaction | Alpha gate + theta-gamma coupling |
| Sequential layers | Oscillatory cycles |

## Usage
```bash
MODEL_VERSION=wave python train.py
# or with torchrun for multi-GPU
```

### Hyperparameters
| Env Var | Default | Description |
|---------|---------|-------------|
| `NUM_STEPS` | 8 | Number of oscillatory cycles |
| `N_FOURIER_BASIS` | 16 | Total Fourier basis pairs |
| `N_CHANNELS` | 128 | Channel dimension |
| `BAND_SPLIT` | `4,4,8` | Low,mid,high basis allocation |
| `SLOW_DECAY_INIT` | 4.0 | Slow memory decay (theta, long-range) |
| `FAST_DECAY_INIT` | 2.0 | Fast memory decay (gamma, local) |

## Files
- `model.py` — BrainWaveGPT model definition
- `DESIGN.md` — Full design document with mathematical framework
