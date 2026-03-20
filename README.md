# AGI Models

Exploring what computation *actually is* — not copying the brain's structure, but finding the simplest mathematical substrate that produces intelligence.

## Philosophy

### 1. The medium is not the message
Everyone in ML copies the brain's **structure**: neurons → hidden units, synapses → weights, cortical layers → transformer layers. This is like trying to fly by building mechanical feathers. We don't copy structure — we look for the **underlying dynamics** that produce intelligent behavior, regardless of substrate.

### 2. Registers are words
Standard transformers map tokens into opaque embedding spaces where intermediate states are uninterpretable. Our models keep computation in **vocabulary space** — every intermediate state is readable as "which words are active and how strongly." Interpretability by construction, not by post-hoc analysis.

```
Input:  one-hot("cat") → R["cat"] = 1.0, all else 0.0
State:  always a distribution over words
Output: register state IS the prediction — no output projection needed
```

### 3. Simple math, composed deeply
Dot products, outer products, dense projections, relu. If the math didn't exist before 1980, we probably don't need it. The power comes from **composition and scale**, not mathematical complexity. This is why attention works — it's just a weighted average. And it's why Fourier projections failed us — too clever, not enough capacity.

### 4. Meta-learning over memorization
The trained weights define **how to learn**. The runtime state (Q-table, associative memory, policy) stores **what was learned** from the current sequence. The model learns during inference — like a Q-table in reinforcement learning that starts empty and fills up through experience.

### 5. Policy over lookup
Instead of memorizing every possible relationship (like attention's Q·K^T over all positions), learn a compact **policy** that decides what to do given the current state. The same mechanism can execute different operations on different inputs — data-dependent branching, not fixed computation.

## What we've learned

**Fourier projections don't work for cross-position mixing.** v3, v5, v6 all used Fourier-parameterized projections (rank-32 bottleneck) for Q/K/V. All produced flat loss. The bottleneck can't capture the complexity of word relationships.

**Dense projections work.** v1 (attention with dense matrices) and v9 (Q-table with dense projections) both learn. The key ingredient is full-rank learned projection matrices.

**Phase transitions are real.** v9 plateaued for 150 steps then loss dropped sharply from 6.30 → 5.60 in 80 steps. Don't kill runs during plateaus — the model may be coordinating internal representations before a breakthrough.

**101K params can learn language structure.** v4 reached val_bpb 3.65 with only 101K parameters (419KB compressed). Not competitive yet, but proves the architectural ideas have merit at extreme compression.

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

### Evolution of ideas

```
v0-v1: Can registers = words?              → Yes, with attention (v1 best bpb)
v2:    Can convolutions replace attention?  → No (too slow, no cross-word mixing)
v3-v6: Can Fourier projections replace      → No (rank bottleneck kills learning)
       dense matrices?
v4:    How small can we go?                 → 101K params, 419KB, val_bpb 3.65
v9:    Q-table with dense projections?      → Yes! Phase transition, val_bpb 3.32
v10:   Policy instead of lookup?            → Testing
```

## Quick Start

```bash
# One-command setup on RunPod
curl -sSL https://raw.githubusercontent.com/urmzd/agi-models/main/bootstrap.sh | bash

# Or manually
pip install huggingface_hub sentencepiece
python data/download_data.py --variant sp1024

# Train (pick a model)
MODEL_VERSION=meta   torchrun --standalone --nproc_per_node=1 train.py  # v9 Q-table (recommended)
MODEL_VERSION=policy torchrun --standalone --nproc_per_node=1 train.py  # v10 policy
MODEL_VERSION=v4     torchrun --standalone --nproc_per_node=1 train.py  # 101K params
```

All hyperparameters configurable via env vars. See [AGENTS.md](AGENTS.md).

## Project Structure

```
train.py                       # Unified training script (all models inline + imports)
data/download_data.py          # Data download (FineWeb sp1024)
bootstrap.sh                   # One-command RunPod setup
v0_register_lm/               # Prototype (learned embeddings)
v1_shared_attention/           # Shared attention (best bpb)
v2_causal_conv/                # Depthwise conv (abandoned)
v3_assoc_memory/               # Associative memory (Fourier — bottlenecked)
v4_param_optimized/            # Param-optimized (101K params)
v5_gauss_fft/                  # FFT-based (flat loss)
v6_brain_wave/                 # Oscillatory dynamics (flat loss)
v7_lgp/                        # True LGP (op bank + soft addressing)
v8_word_graph/                 # Direct word-to-word graph
v9_meta_state/                 # Q-table meta-state (best non-attention)
v10_policy/                    # State-dependent policy execution
docs/                          # Research notes and design docs
```

## Inspirations

- [Linear Genetic Programming](https://github.com/urmzd/linear-gp) — register machines, Q-tables, sequential cheap operations
- Reinforcement learning — Q-tables as meta-learning, policy over lookup
- Hopfield networks (1982) — associative memory via outer products
- [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) — constraints that force architectural innovation
