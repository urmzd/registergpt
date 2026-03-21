# exp-agi-models

Exploring whether vocabulary-space computation — where every hidden state is a readable distribution over words — can match opaque-embedding architectures at language modeling.

## What This Is

A collection of 17 experimental language model architectures that share one constraint: **hidden dimension = vocabulary size**. There is no learned embedding and no output projection. The register state IS the prediction. Every intermediate state is interpretable as "which words are active and how strongly."

This constraint is genuinely novel — no published architecture we're aware of operates this way. Whether it's a good idea is an open question we're trying to answer empirically.

## What We've Found So Far

### Benchmark results (10 min, 3x A40, batch=491,520 tokens)

| `MODEL_VERSION` | Architecture | Params | Steps | val_loss | val_bpb | tok/s | Status |
|---|---|---|---|---|---|---|---|
| **v8_graph** (rank 8) | Low-rank word interaction | **164K** | 100 | **5.24** | **3.10** | 270K | Still descending |
| v2_conv | Causal convolution + Fourier | 353K | 464 | 5.39 | 3.19 | 383K | Still descending |
| v6_wave | Oscillatory dynamics | 824K | 166 | 5.66 | 3.35 | 136K | Still descending |
| v1_attention | Shared GQA + Fourier | 3.4M | 239 | 6.06 | 3.59 | 196K | Plateaued |
| v7_lgp | Learned program (op bank) | 329K | 348 | 6.26 | 3.71 | 287K | Unstable (loss spikes) |
| v3_assoc | Associative memory | 329K | 397 | 6.81 | 4.03 | 326K | Stuck |
| v8_graph (rank 64) | Low-rank word interaction | 1.1M | 188 | — | — | 270K | Memorized (train 0.04, overfitting) |

### What these results mean

**The word graph (v8) is the best architecture so far.** At rank 8 with 164K params, it reaches val_loss 5.24 in 100 steps — better than v2_conv (353K params, 464 steps) with half the parameters in one-fifth the steps. The train/val gap is essentially zero, confirming it's learning, not memorizing.

**But at rank 64, the same architecture memorizes.** The 1.1M-param version drove train_loss to 0.04 while val_loss stayed high. The rank-64 U@V^T matrix has enough capacity to store a bigram lookup table. Rank 8 can't, so it's forced to learn compressed, generalizable word relationships instead.

**This is still far from useful.** val_loss 5.24 (3.10 bpb) is well above the ~1.7 loss needed for 1 bpb. GPT-2 at 124M params achieves ~0.93 bpb. We're at 164K params, so the comparison isn't fair, but the gap is large.

### What's actually unique here

1. **hidden_dim = vocab_size with no embedding or output projection.** No published architecture does this. The state IS the prediction at every step.

2. **Interpretability by construction.** You can read intermediate states as word distributions. This is not a post-hoc technique.

3. **The specific combination** of vocabulary-space state + various cross-position mechanisms (conv, decay memory, word graph) + recurrent depth has not been explored before.

### What's NOT unique

- Weight sharing across depth: Universal Transformer (2019), ALBERT (2020), DEQ (2019) all do this
- Fourier parameterization: FNet (2022), butterfly matrices, Fourier Neural Operators
- Causal decay memory: RWKV, Mamba, S4 all use equivalent mechanisms
- Low-rank word interaction: mathematically, `x @ U @ V^T` is just a rank-r linear layer. The "word graph" framing sounds novel but the computation is standard
- Recurrent register machines: Neural Turing Machine (2014), Neural GPU (2016)

### Honest assessment of the v8_graph results

The rank-8 word graph works well because **direct bilinear word-to-word interaction is a good inductive bias for language**. Language is fundamentally about which words predict which other words. A model that directly parameterizes `W[i,j] = "word i predicts word j"` captures this structure more efficiently than architectures that must discover it through generic operations (convolutions, MLPs, Fourier transforms).

But this is a well-known insight. Bigram and n-gram models encode the same structure. The question is whether multi-hop graph propagation (8 hops through the low-rank interaction matrix) can capture longer-range dependencies that simple n-grams cannot. The current results don't answer this — we'd need to test on tasks requiring longer-range reasoning.

## Architecture

All variants share the same skeleton:

```
Input:  one-hot("cat") -> R["cat"] = 1.0, everything else 0.0
Repeat N times:
  1. Cross-position mixing  (how do words at different positions interact?)
  2. Within-position transform  (how do word activations combine?)
Output: register state -> softcap -> cross-entropy loss
```

No embedding. No output projection.

## Model Versions

### Core architectures (v1-v13)

| `MODEL_VERSION` | Cross-position | Within-position | Notes |
|---|---|---|---|
| `v1_attention` | GQA + RoPE | Fourier ops | 3.4M params, plateaus early |
| `v2_conv` | Depthwise causal conv | Fourier ops | 353K params, strong baseline |
| `v3_assoc` | Fourier associative memory | Fourier ops | Stuck — Fourier bottleneck |
| `v4_golf` | Multi-head assoc (shared Q/K) | Factored ops | 102K params, DDP fixed |
| `v5_gauss` | FFT associative memory | FFT ops | Shape bug fixed |
| `v6_wave` | Oscillatory coupling | Band-specific ops | 824K, still descending |
| `v7_lgp` | Causal decay memory | Learned op bank | Unstable, loss spikes |
| `v8_graph` | Word activation similarity | Low-rank V×V interaction | **Best so far at rank 8** |
| `v9_meta` | Evolving Q-table (dense) | Dense MLP | 4.2M params |
| `v10_policy` | Causal decay + policy | State-dependent ops | Untested |
| `v11_brainwave` | EMA + causal decay | Oscillatory primitives | Untested |
| `v11_tpg` | Multi-scale Q-table | Hard Gumbel routing | Untested |
| `v12_sparse` | Causal decay (k-subspace) | MLP in k-subspace | Untested |
| `v13_embed` | Causal decay (k-subspace) | Factored embedding | Untested |

### Research-inspired architectures (v14-v16)

| `MODEL_VERSION` | Key Techniques | Inspiration |
|---|---|---|
| `v14_adaptive` | Data-dependent decay, input-modulated conv, DCT basis | Mamba, RWKV, Hyena |
| `v15_predictive` | Per-step aux losses, top-k sparsity, entropy-adaptive writes | Predictive coding, cortical sparse coding |
| `v16_columnar` | Multi-column voting, dendritic MLP branches, lateral inhibition | Thousand Brains, dendritic computation |

## Quick Start

```bash
# Setup on RunPod
curl -sSL https://raw.githubusercontent.com/urmzd/exp-agi-models/main/bootstrap.sh | bash

# Or manually
uv pip install --system -r pyproject.toml
python data/download_data.py --variant sp1024

# Train the best model (word graph, rank 8)
INTERACTION_RANK=8 MODEL_VERSION=v8_graph \
  torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) train.py

# Benchmark all models
benchmark

# Benchmark specific models
benchmark --versions v8_graph,v2_conv,v14_adaptive --minutes 10
```

All hyperparameters configurable via environment variables. See `core/config.py`.

## What We've Learned

**Inductive bias matters more than parameter count.** v8_graph (164K params, rank 8) beats v1_attention (3.4M params, 20x more) because direct word-to-word interaction is a better prior for language than generic attention in vocab space.

**Too much capacity in the right place enables memorization.** v8_graph at rank 64 memorizes the training batch (train loss 0.04). At rank 8 it generalizes (train ≈ val). The constraint forces learning.

**Fourier bottlenecks kill cross-position mixing.** v3_assoc and v5_gauss use rank-32 Fourier projections for cross-position operations. Both got stuck. The bottleneck can't capture word relationship complexity. Fourier ops work within positions but fail across positions.

**Attention in vocab space is expensive and unhelpful at this scale.** v1's shared attention is 3M params dominated by Q/K/V projections operating on 1024-dim vectors. The model plateaus at 6.05 — the overhead isn't justified.

**Training instability is a real problem.** v7_lgp had two catastrophic loss spikes (9.35 at step 161, 8.28 at step 181) before recovering. The soft op selection mechanism is fragile.

## Inspirations

- [Linear Genetic Programming](https://github.com/urmzd/linear-gp) — register machines, sequential cheap operations
- [Tangled Program Graphs](https://web.cs.dal.ca/~mheywood/) — hard bidding, multi-timescale memory
- Neural GPU (Kaiser 2016) — repeated convolution learns algorithms
- Deep Equilibrium Models (Bai 2019) — weight-shared iteration to convergence
- Mamba (Gu & Dao 2023) — data-dependent state transitions
- Predictive coding (Rao & Ballard 1999) — cortex passes prediction errors
- Sparse coding (Olshausen & Field 1996) — only 1-5% of cortical neurons fire
