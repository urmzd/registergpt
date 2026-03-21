# exp-agi-models

Can a language model learn *algorithms* instead of memorizing patterns?

Standard LLMs store knowledge in billions of parameters — statistical lookup tables trained on the internet. We're exploring a different question: what happens when you make a model so small it *can't* memorize, force it to operate in interpretable vocabulary space, and give it recurrent depth to learn programs?

## The Hypothesis

A 350K-parameter model with 1024 vocabulary can't memorize much. It has two choices: learn nothing, or learn *how to process language* — compressive algorithms that generalize. We're betting on the second path.

### Constraints as a forcing function

| Constraint | Standard LLM | This project |
|---|---|---|
| Hidden dimension | Arbitrary (4096+) | = vocab size (1024) |
| Embedding | Learned matrix (V x D) | One-hot (no parameters) |
| Output projection | Learned matrix (D x V) | None (state IS the logits) |
| Intermediate states | Opaque vectors | Readable word distributions |
| Parameters | 100M–1T | 100K–3M |

Every intermediate state is a distribution over words. You can literally read what the model is "thinking" at step 3 of 8. This isn't a post-hoc interpretability technique — it's the architecture.

### Why this might matter

If a tiny model in vocabulary space can reach competitive loss, it suggests:
- Current architectures waste most of their parameters on memorization, not computation
- The scaling laws we observe are properties of transformers, not of intelligence
- Interpretability and performance aren't necessarily at odds

We haven't proven any of this yet. Current best is val_loss 5.39 (3.19 bpb) with 353K params. That's far from frontier. But the loss curve is still descending.

## Architecture

All variants share the same skeleton:

```
Input:  one-hot("cat") -> R["cat"] = 1.0, everything else 0.0
Repeat N times:
  1. Cross-position mixing  (how do words at different positions interact?)
  2. Within-position transform  (Fourier register ops: read -> mix -> write)
Output: register state -> softcap -> cross-entropy loss
```

No embedding. No output projection. No attention (in the best variant).

The **Fourier register ops** use basis functions over vocabulary indices. Low frequencies group words broadly (nouns vs verbs). High frequencies distinguish specific words (cat vs dog). Each op costs ~585 parameters — 3,400x cheaper than a dense layer.

## Benchmark Results

10-minute wallclock, 3x NVIDIA A40, batch=491,520 tokens, 5 warmup steps:

| Version | Architecture | Params | Steps | val_loss | val_bpb | tok/s | Status |
|---------|-------------|--------|-------|----------|---------|-------|--------|
| **v2** | Causal convolution | **353K** | 464 | **5.39** | **3.19** | 383K | Still descending |
| v1 | Shared attention | 3.4M | 239 | 6.06 | 3.59 | 196K | Plateaued |
| v3 | Associative memory | 329K | 397 | 6.81 | 4.03 | 326K | Stuck early |
| v4 | Parameter-golf (101K) | 102K | — | — | — | — | DDP bug (fixed) |
| gauss | Gaussian FFT | 329K | — | — | — | — | Shape bug (fixed) |
| wave | Oscillatory dynamics | 824K | ~166 | ~5.7* | ~3.4* | 136K | Descending |

*wave estimate based on training loss trajectory, final val pending.

**Key finding:** v2 (causal convolution, no attention) beats v1 (shared attention) with 10x fewer parameters and 2x throughput. Attention in vocabulary space is expensive and doesn't pay for itself at this scale.

## What We've Learned

**Dense attention hurts at small scale.** v1's shared attention operates on 1024-dim vectors (= vocab size). That's 3M params dominated by Q/K/V projections. The model plateaus at 6.05 loss — the attention overhead isn't worth it.

**Convolutions > attention for cross-position mixing.** v2 replaces attention with depthwise causal convolution. Half the step time, 10x fewer params, lower loss. The positional structure in convolutions provides a better inductive bias when you can't afford to learn arbitrary attention patterns.

**Fourier bottlenecks kill some architectures.** v3 and v5 use Fourier-parameterized projections for cross-position mixing (rank-32 bottleneck). Both got stuck — the bottleneck can't capture word relationship complexity. Fourier ops work well *within* positions (register transforms) but fail for *cross-position* mixing.

**Phase transitions happen.** Don't kill runs during plateaus. Some models (v9 in prior runs) plateaued for 150 steps then loss dropped sharply.

**The loss curve shape matters more than the endpoint.** v2 was still descending at step 464. Given more time/data, the gap between architectures may widen.

## Model Versions

| ID | Name | Cross-position | Within-position |
|----|------|---------------|-----------------|
| `v1` | Shared Attention | GQA + RoPE (shared weights) | Fourier ops |
| `v2` | Causal Conv | Depthwise causal convolution | Fourier ops |
| `v3` | Assoc Memory | Fourier-projected associative memory | Fourier ops |
| `v4` | Param Golf | Multi-head assoc memory (shared Q/K) | Factored ops |
| `gauss` | Gauss FFT | FFT-based associative memory | FFT ops |
| `wave` | Brain Wave | Oscillatory cross-frequency coupling | Band-specific ops |
| `lgp` | LGP | Causal decay memory | Learned program (op bank) |
| `graph` | Word Graph | Word activation similarity | V x V interaction |
| `meta` | Meta-State | Evolving Q-table (dense) | Dense MLP |
| `policy` | Policy | Causal decay + policy | State-dependent ops |
| `brainwave` | BrainWave v2 | EMA + causal decay | Oscillatory primitives |
| `tpg` | Neural TPG | Multi-scale Q-table (3 decays) | Hard Gumbel routing |
| `sparse` | Sparse Register | Causal decay (k-subspace) | MLP in k-subspace |
| `embed` | Sparse Embed | Causal decay (k-subspace) | Factored embedding |

## Quick Start

```bash
# Setup on RunPod
curl -sSL https://raw.githubusercontent.com/urmzd/exp-agi-models/main/bootstrap.sh | bash

# Or manually
uv pip install --system -r pyproject.toml
python data/download_data.py --variant sp1024

# Train a model
MODEL_VERSION=v2 torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) train.py

# Benchmark all models (10 min each)
benchmark

# Benchmark specific models
benchmark --versions v2,sparse,embed --minutes 5
```

All hyperparameters configurable via environment variables. See `core/config.py` for the full list.

## Inspirations

- [Linear Genetic Programming](https://github.com/urmzd/linear-gp) — register machines, sequential cheap operations
- [Tangled Program Graphs](https://web.cs.dal.ca/~mheywood/) — hard bidding, multi-timescale memory
- Reinforcement learning — Q-tables as meta-learning primitives
- Hopfield networks — associative memory via outer products
- PonderNet — adaptive computation time
