# v8: Word Interaction Graph

**Status**: Best architecture so far (at rank 8).

## Architecture
- Low-rank V x V word interaction matrix: `W = U @ V^T + diag`
- Multiple hops build compositional word associations
- Cross-position via causal decay-weighted word activation similarity
- No Fourier basis, no channel bottleneck

## Results (3x A40, 10 min)

| Rank | Params | Steps | val_loss | val_bpb | Behavior |
|------|--------|-------|----------|---------|----------|
| 8 | 164K | 100 | **5.24** | **3.10** | Generalizing (train ≈ val), still descending |
| 64 | 1.1M | 188 | ~6.5+ | — | Memorizing (train 0.04, val high) |

## Why rank matters

At rank 64, `U @ V^T` has enough capacity to memorize word-pair statistics from the training batch. At rank 8, the model is forced to learn compressed, generalizable structure instead. The constraint creates generalization.

## Honest assessment

`x @ U @ V^T` is mathematically just a rank-r linear layer. The "word graph" framing sounds novel but the computation is standard. What's genuinely useful is that direct bilinear word-to-word interaction is a good inductive bias for language — better than convolutions or Fourier ops at this scale.

The cross-position mechanism is O(T^2 x V) — full attention cost in vocab space. This won't scale to long sequences.

## Usage
```bash
INTERACTION_RANK=8 MODEL_VERSION=v8_graph torchrun --standalone --nproc_per_node=3 train.py
```
