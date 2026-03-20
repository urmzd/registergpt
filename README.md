# AGI Models

Experimental architectures exploring **interpretable, efficient computation** for language modeling — where the model's internal state is always human-readable.

## Core Principles

**Registers are words.** Standard transformers map tokens into opaque embedding spaces. These models keep computation in **vocabulary space** the entire time:

```
Input:  one-hot("cat") → R["cat"] = 1.0, all else 0.0
State:  always a distribution over words
Output: register state IS the prediction — R["dog"]=0.3, R["mat"]=0.25
```

No embedding matrix. No output projection. Interpretability by construction.

**Simple math only.** Dot products, outer products, dense projections, relu/gelu. If the math didn't exist before 1980, we don't use it. The power comes from composition and scale, not mathematical complexity.

**Meta-learning over memorization.** The model's trained weights define *how to learn*. The runtime state (Q-table, associative memory) stores *what was learned* from the current sequence. The model learns during inference.

## Research Questions

1. **Can we replace attention with an evolving Q-table?** Cross-position mixing as meta-learning, not static projections.
2. **What is the minimum computational substrate for language?** Sequential execution of cheap operations on a register bank.
3. **Do Fourier projections help or hurt?** Early results: dense projections (full-rank) work, Fourier (rank-constrained) don't learn.
4. **Can 101K params model language structure?** v4 reached val_bpb 3.65 — better than random but not competitive yet.

## Architecture Iterations

| Version | Cross-position | Within-position | Params | Size (int8) | val_bpb | Status |
|---------|---------------|-----------------|--------|-------------|---------|--------|
| [v0](v0_register_lm/) | Shared attention | Fourier ops | 485K | ~500KB | — | Prototype |
| [v1](v1_shared_attention/) | Shared attention | Fourier ops | 3.2M | 1.6MB | **2.83** | Best bpb |
| [v2](v2_causal_conv/) | Depthwise conv | Fourier ops | 1.3M | ~1.3MB | — | Abandoned |
| [v3](v3_assoc_memory/) | Assoc memory (Fourier) | Fourier ops | 328K–1.7M | ~1.7MB | ~3.9 | Fourier bottleneck |
| [v4](v4_param_optimized/) | Assoc memory (shared Q/K) | Factored ops | 101K | 419KB | 3.65 | Smallest |
| [v5](v5_gauss_fft/) | FFT-based assoc memory | FFT ops | 919K | ~900KB | ~4.1 | Flat loss |
| [v6](v6_brain_wave/) | Oscillatory coupling | Band-specific ops | 824K | ~800KB | ~3.7 | Flat loss |
| [v7](v7_lgp/) | Causal decay memory | Learned program (op bank) | — | — | — | Ready |
| [v8](v8_word_graph/) | Word activation similarity | V×V interaction graph | — | — | — | Ready |
| **[v9](v9_meta_state/)** | **Evolving Q-table (dense)** | **Dense MLP** | **4.2M** | **9.3MB** | **3.32** | **Still dropping** |
| [v10](v10_policy/) | Causal decay + policy | State-dependent ops | — | — | — | Ready |

### Key finding

Everything with Fourier projections (v3, v5, v6) fails to learn — the rank-32 bottleneck can't capture word relationships. Dense projections (v1, v4) work. **v9 combines dense projections with the Q-table meta-learning insight.**

## Quick Start

```bash
# One-command setup on RunPod
curl -sSL https://raw.githubusercontent.com/urmzd/agi-models/main/bootstrap.sh | bash

# Or manually
pip install huggingface_hub sentencepiece
python data/download_data.py --variant sp1024

# Train v9 (meta-state Q-table)
MODEL_VERSION=meta torchrun --standalone --nproc_per_node=1 train.py

# All model versions
MODEL_VERSION=v3    # associative memory
MODEL_VERSION=v4    # param-optimized (101K params)
MODEL_VERSION=gauss # FFT-based
MODEL_VERSION=wave  # oscillatory dynamics
MODEL_VERSION=lgp   # true LGP (differentiable register machine)
MODEL_VERSION=graph # word interaction graph
MODEL_VERSION=meta  # Q-table meta-state (recommended)
```

All hyperparameters configurable via env vars. See [AGENTS.md](AGENTS.md).

## Project Structure

```
train.py                       # Unified training script (all models)
model.py                       # v3 reference model
model_v4.py                    # v4 param-optimized model
data/download_data.py          # Data download (FineWeb sp1024)
bootstrap.sh                   # One-command RunPod setup
v0_register_lm/               # Original prototype (learned embeddings)
v1_shared_attention/           # Shared attention (best results)
v2_causal_conv/                # Depthwise conv (abandoned)
v3_assoc_memory/               # Associative memory (Fourier projections)
v4_param_optimized/            # Param-optimized (101K params, shared Q/K)
v5_gauss_fft/                  # FFT-based (Gauss)
v6_brain_wave/                 # Oscillatory dynamics
v7_lgp/                        # True LGP (op bank + soft addressing)
v8_word_graph/                 # Direct word-to-word graph
v9_meta_state/                 # Q-table meta-state (dense, no Fourier)
docs/                          # Research notes and design docs
```

## Inspirations

- [Linear Genetic Programming](https://github.com/urmzd/linear-gp) — register machines, Q-tables, evolution of programs
- [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) — constraints that force innovation
- Hopfield networks (1982) — associative memory via outer products
- Reinforcement learning — Q-tables as meta-learning substrate
