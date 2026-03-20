# AGI Models

Experimental architectures exploring **interpretable, efficient computation** for language modeling — where the model's internal state is always human-readable.

## Core Principle

Standard transformers map tokens into opaque embedding spaces. These models keep computation in **vocabulary space** the entire time:

```
Input:  one-hot("cat") → R["cat"] = 1.0, all else 0.0
State:  always a distribution over words
Output: register state IS the prediction — R["dog"]=0.3, R["mat"]=0.25
```

No embedding matrix. No output projection. Every intermediate state is readable as "which words are active and how strongly." Interpretability by construction, not by post-hoc analysis.

## Research Questions

1. **Can we build competitive LMs without attention?** Using associative memory, convolutions, or FFT-based operations instead.
2. **What is the minimum mathematical machinery needed?** All architectures here use math from the 1970s or earlier — dot products, outer products, Fourier transforms.
3. **Can LGP-style register machines scale to language?** Sequential execution of cheap operations on a narrow register bank, inspired by [linear-gp](https://github.com/urmzd/linear-gp).
4. **What happens when hidden_dim = vocab_size?** The model thinks in word-space, not in a learned latent space.

## Architecture Iterations

| Version | Cross-position mixing | Within-position | Params | Best val_bpb | Status |
|---------|----------------------|-----------------|--------|-------------|--------|
| [v0](v0_register_lm/) | Shared attention | Fourier ops | 485K | — | Prototype (uses learned embeddings) |
| [v1](v1_shared_attention/) | Shared attention | Fourier ops | 3.2M | **2.83** | Best so far |
| [v2](v2_causal_conv/) | Depthwise causal conv | Fourier ops | 1.3M | — | Abandoned |
| [v3](v3_assoc_memory/) | Associative memory | Fourier ops | 328K–1.7M | — | In progress |
| [v4](v4_param_optimized/) | Assoc memory (shared Q/K) | Factored ops | ~101K | — | In progress |
| [v5](v5_gauss_fft/) | FFT-based assoc memory | FFT register ops | 919K | — | Tested (flat loss) |
| [v6](v6_brain_wave/) | Oscillatory coupling | Band-specific memory + alpha/theta-gamma gates | — | — | Ready |
| [v7](v7_lgp/) | Causal decay memory | Learned program (op bank + soft addressing) | — | — | Ready |
| [v8](v8_word_graph/) | Causal word propagation | Direct V×V word interaction graph | — | — | Ready |

## Quick Start

```bash
# Download data
pip install huggingface_hub sentencepiece
python data/download_data.py --variant sp1024

# Train (single GPU)
torchrun --standalone --nproc_per_node=1 train.py

# Train a specific model version
MODEL_VERSION=gauss torchrun --standalone --nproc_per_node=1 train.py
MODEL_VERSION=v4 torchrun --standalone --nproc_per_node=1 train.py
```

All hyperparameters are configurable via environment variables. See [AGENTS.md](AGENTS.md).

## Project Structure

```
train.py                       # Unified training script (all models)
model.py                       # Current model (v3 reference)
model_v4.py                    # v4 param-optimized model
data/download_data.py          # Data download (FineWeb sp1024)
v0_register_lm/               # Original prototype
v1_shared_attention/           # Shared attention (best results)
v2_causal_conv/                # Depthwise conv (abandoned)
v3_assoc_memory/               # Associative memory
v4_param_optimized/            # Param-optimized design
v5_gauss_fft/                  # FFT-based design
v6_brain_wave/                 # Oscillatory dynamics
v7_lgp/                        # True LGP (differentiable register machine)
v8_word_graph/                 # Direct word-to-word interaction graph
docs/                          # Research notes and design docs
```

## Inspirations

- [Linear Genetic Programming](https://github.com/urmzd/linear-gp) — complex behavior from sequential execution of cheap operations on a narrow register bank
- [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) — training constraints that force architectural innovation
- Hopfield networks (1982) — associative memory via outer products
- Gauss's FFT (1805) — frequency-domain computation predating modern ML by 220 years
