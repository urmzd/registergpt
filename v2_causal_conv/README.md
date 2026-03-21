# v2: Depthwise Causal Conv + Fourier Register Ops

**Status**: Second-best architecture. Strong baseline.

## Architecture
- Per-step depthwise causal conv1d for cross-position mixing
- Per-step Fourier register ops for within-position transforms
- No attention, no embedding, no output projection

## Results (3x A40, 10 min)

| Params | Steps | val_loss | val_bpb | tok/s |
|--------|-------|----------|---------|-------|
| 353K | 464 | 5.39 | 3.19 | 383K |

Still descending at step 464. Fastest throughput of any model (383K tok/s).

## Previous assessment was wrong

Earlier README said v2 was "abandoned" and "slower and worse than v1." That was from an initial test with LR=0.001 and 48 steps. With default hyperparameters (LR=0.03, 8 steps), v2 significantly beats v1 (5.39 vs 6.06 val_loss) with 10x fewer params and 2x throughput.

## Why it works

Depthwise causal convolution provides local positional context cheaply. Each vocab dimension is convolved independently along the sequence — "how active was word j in the last k positions?" The Fourier ops then handle cross-word mixing within each position.

## Now surpassed by v8_graph

v8_graph at rank 8 (164K params) reaches val_loss 5.24 in 100 steps — better loss with half the params in one-fifth the steps. v2_conv remains valuable as the highest-throughput model (383K vs 270K tok/s).

## Usage
```bash
MODEL_VERSION=v2_conv torchrun --standalone --nproc_per_node=3 train.py
```
