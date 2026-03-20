# v8: Word Interaction Graph — Direct Word-to-Word Associations

**Status**: Implemented, ready for training.

## Architecture
- Learned V×V word interaction matrix (low-rank: W = U@V^T + diag)
- Multiple hops through the graph build compositional representations
- Cross-position via causal decay-weighted word activation similarity
- No Fourier basis, no channel bottleneck — direct word space operations

## Key Innovation
Fully interpretable: W["the"]["cat"] = 0.3 means "the" activates "cat" with weight 0.3.
The graph IS the model's knowledge of language structure.

## Usage
```bash
MODEL_VERSION=graph torchrun --standalone --nproc_per_node=1 train.py
```

## Env Vars
| Variable | Default | Notes |
|----------|---------|-------|
| `NUM_STEPS` | 8 | Number of hops through graph |
| `INTERACTION_RANK` | 64 | Rank of word interaction matrix |
